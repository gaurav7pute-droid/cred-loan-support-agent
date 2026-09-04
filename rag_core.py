"""
rag_core.py -- Part 1 Tasks 3-5: embedding + dual-collection ChromaDB
indexing + grounded generation (MOCK_LLM) + doc-level precision/recall
evaluation across both chunking strategies.

Two independent ChromaDB collections are built:
  - "kb_fixed"    -- chunks from chunking.fixed_size_overlap_chunks
  - "kb_sentence" -- chunks from chunking.sentence_based_chunks
Both are embedded with the same local, free SentenceTransformers model
(all-MiniLM-L6-v2) so retrieval quality differences come only from the
chunking strategy, not the embedding model.
"""
import os
import shutil

import chromadb
from sentence_transformers import SentenceTransformer

from kb import KNOWLEDGE_BASE, DOC_BY_ID
from chunking import chunk_all

CHROMA_DIR = os.path.join(os.path.dirname(__file__), ".chroma_store")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

_embedder = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=CHROMA_DIR)


def reset_chroma_store():
    """Wipe any previous index so re-running this module is fully deterministic."""
    if os.path.isdir(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)


COLLECTION_NAMES = {"fixed": "kb_fixed", "sentence": "kb_sentence"}


def build_collection(client: chromadb.ClientAPI, strategy: str) -> "chromadb.Collection":
    """Chunk, embed, and collection.upsert() every chunk for one strategy."""
    name = COLLECTION_NAMES[strategy]
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

    chunks = chunk_all(KNOWLEDGE_BASE, strategy)
    embedder = get_embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True).tolist()

    collection.upsert(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"doc_id": c["doc_id"], "topic": c["topic"]} for c in chunks],
    )
    return collection


def build_both_collections():
    reset_chroma_store()
    client = get_client()
    fixed = build_collection(client, "fixed")
    sentence = build_collection(client, "sentence")
    return client, {"fixed": fixed, "sentence": sentence}


# Process-wide singleton so crew.py, app.py, and eval scripts that run in the
# same process never rebuild (and thereby wipe/reset) each other's index.
_shared_client = None
_shared_collections = None


def get_shared_collections() -> dict:
    global _shared_client, _shared_collections
    if _shared_collections is None:
        _shared_client, _shared_collections = build_both_collections()
    return _shared_collections


def retrieve(collection: "chromadb.Collection", query: str, k: int = 3) -> list:
    embedder = get_embedder()
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=k)
    hits = []
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    for cid, text, meta, dist in zip(ids, docs, metas, dists):
        # collection was created with hnsw:space=cosine -> distance is (1 - cosine_sim)
        similarity = 1.0 - dist
        hits.append(
            {
                "chunk_id": cid,
                "doc_id": meta["doc_id"],
                "topic": meta["topic"],
                "text": text,
                "similarity": similarity,
            }
        )
    return hits


# ---------------------------------------------------------------------------
# Grounded generation (MOCK_LLM: retrieval similarity is the only groundedness
# signal available, per the brief). IDK_THRESHOLD is set in calibrate_threshold()
# below and then hardcoded here from the measured values -- see README.md.
# ---------------------------------------------------------------------------
IDK_THRESHOLD = 0.42  # calibrated empirically; see calibrate_threshold() / README.md

IDK_RESPONSE = (
    "I don't have grounded information about that in the Cred policy knowledge base, "
    "so I can't answer confidently. Please rephrase your question or contact a human "
    "support agent for anything outside loan and account policy."
)


def grounded_answer(collection: "chromadb.Collection", query: str, k: int = 3,
                     threshold: float = IDK_THRESHOLD) -> dict:
    hits = retrieve(collection, query, k=k)
    top_similarity = hits[0]["similarity"] if hits else 0.0

    if not hits or top_similarity < threshold:
        return {
            "query": query,
            "answer": IDK_RESPONSE,
            "grounded": False,
            "top_similarity": top_similarity,
            "used_chunks": [],
        }

    # MOCK_LLM "generation": since there is no real model to phrase a novel
    # sentence, the grounded answer is composed ONLY from the retrieved
    # chunk text (extractive), deduplicated by parent document, capped at 2
    # supporting chunks so the answer stays concise.
    seen_docs = set()
    used = []
    for h in hits:
        if h["doc_id"] in seen_docs:
            continue
        seen_docs.add(h["doc_id"])
        used.append(h)
        if len(used) == 2:
            break

    answer = "Based on Cred policy: " + " ".join(h["text"] for h in used)
    return {
        "query": query,
        "answer": answer,
        "grounded": True,
        "top_similarity": top_similarity,
        "used_chunks": used,
    }


# ---------------------------------------------------------------------------
# Threshold calibration (Task 4): measure top-1 similarity for known
# in-scope and out-of-scope queries, then choose a threshold between the
# two observed clusters.
# ---------------------------------------------------------------------------
CALIBRATION_IN_SCOPE_QUERIES = [
    "What documents do I need for KYC?",
    "How is my EMI calculated?",
    "What is the annual fee on my credit card?",
    "Can I close my savings account for free?",
    "What happens if I report a fraudulent transaction late?",
]
CALIBRATION_OUT_OF_SCOPE_QUERIES = [
    "What is the best recipe for butter chicken?",
    "Who won the cricket world cup last year?",
]


def calibrate_threshold(collection: "chromadb.Collection") -> dict:
    in_scope_sims = []
    for q in CALIBRATION_IN_SCOPE_QUERIES:
        hits = retrieve(collection, q, k=1)
        in_scope_sims.append(hits[0]["similarity"])
    out_scope_sims = []
    for q in CALIBRATION_OUT_OF_SCOPE_QUERIES:
        hits = retrieve(collection, q, k=1)
        out_scope_sims.append(hits[0]["similarity"])

    chosen = (min(in_scope_sims) + max(out_scope_sims)) / 2
    return {
        "in_scope_sims": in_scope_sims,
        "out_scope_sims": out_scope_sims,
        "min_in_scope": min(in_scope_sims),
        "max_out_scope": max(out_scope_sims),
        "chosen_threshold": chosen,
    }


if __name__ == "__main__":
    client, collections = build_both_collections()
    print("Built collections:", {k: v.count() for k, v in collections.items()})

    print("\n--- Threshold calibration on kb_sentence collection ---")
    calib = calibrate_threshold(collections["sentence"])
    print("In-scope top-1 similarities:", [f"{s:.3f}" for s in calib["in_scope_sims"]])
    print("Out-of-scope top-1 similarities:", [f"{s:.3f}" for s in calib["out_scope_sims"]])
    print(f"min(in-scope)={calib['min_in_scope']:.3f}  max(out-of-scope)={calib['max_out_scope']:.3f}")
    print(f"Chosen threshold (midpoint) = {calib['chosen_threshold']:.3f}  "
          f"(hardcoded as IDK_THRESHOLD={IDK_THRESHOLD} in this file after this measurement)")

    print("\n--- Grounded generation demo (kb_sentence) ---")
    for q in CALIBRATION_IN_SCOPE_QUERIES:
        r = grounded_answer(collections["sentence"], q)
        print(f"\nQ: {q}\n  top_sim={r['top_similarity']:.3f} grounded={r['grounded']}\n  A: {r['answer'][:200]}")

    for q in CALIBRATION_OUT_OF_SCOPE_QUERIES[:1]:
        r = grounded_answer(collections["sentence"], q)
        print(f"\nQ: {q}\n  top_sim={r['top_similarity']:.3f} grounded={r['grounded']}\n  A: {r['answer'][:200]}")
