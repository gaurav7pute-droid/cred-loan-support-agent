"""
eval_retrieval.py -- Part 1 Task 5: doc-level precision/recall for BOTH
chunking-strategy collections on the same 5 in-scope queries, with visible
per-query arithmetic, plus a numbers-cited recommendation.
"""
from rag_core import get_shared_collections, retrieve, CALIBRATION_IN_SCOPE_QUERIES

# Ground-truth relevant parent document for each evaluation query. Each query
# was written to target exactly one KB topic/document, so the relevant set is
# a singleton -- this keeps the precision/recall arithmetic exact and legible.
EVAL_QUERIES = [
    (CALIBRATION_IN_SCOPE_QUERIES[0], {"kb-04"}),  # KYC documents
    (CALIBRATION_IN_SCOPE_QUERIES[1], {"kb-02"}),  # EMI calculation
    (CALIBRATION_IN_SCOPE_QUERIES[2], {"kb-03"}),  # credit-card fees
    (CALIBRATION_IN_SCOPE_QUERIES[3], {"kb-06"}),  # account closure
    (CALIBRATION_IN_SCOPE_QUERIES[4], {"kb-05"}),  # fraud dispute
]

TOP_K = 3


def precision_recall_at_k(collection, query: str, relevant_docs: set, k: int = TOP_K):
    hits = retrieve(collection, query, k=k)
    retrieved_docs = []
    for h in hits:
        if h["doc_id"] not in retrieved_docs:
            retrieved_docs.append(h["doc_id"])  # dedup, preserve rank order

    hit_set = set(retrieved_docs) & relevant_docs
    precision = len(hit_set) / len(retrieved_docs) if retrieved_docs else 0.0
    recall = len(hit_set) / len(relevant_docs) if relevant_docs else 0.0
    return {
        "query": query,
        "retrieved_docs": retrieved_docs,
        "relevant_docs": relevant_docs,
        "hit_set": hit_set,
        "precision": precision,
        "recall": recall,
    }


def evaluate_collection(collection, label: str) -> list:
    print(f"\n=== {label} (top-{TOP_K}) ===")
    results = []
    for query, relevant in EVAL_QUERIES:
        r = precision_recall_at_k(collection, query, relevant)
        results.append(r)
        print(f"Q: {query}")
        print(f"  retrieved parent docs (deduped, ranked): {r['retrieved_docs']}")
        print(f"  relevant set: {sorted(r['relevant_docs'])}")
        print(f"  |retrieved ∩ relevant| = {len(r['hit_set'])}  "
              f"precision = {len(r['hit_set'])}/{len(r['retrieved_docs'])} = {r['precision']:.3f}  "
              f"recall = {len(r['hit_set'])}/{len(r['relevant_docs'])} = {r['recall']:.3f}")
    avg_p = sum(r["precision"] for r in results) / len(results)
    avg_r = sum(r["recall"] for r in results) / len(results)
    print(f"  -- {label} averages: precision={avg_p:.3f} recall={avg_r:.3f} --")
    return results, avg_p, avg_r


if __name__ == "__main__":
    collections = get_shared_collections()

    fixed_results, fixed_avg_p, fixed_avg_r = evaluate_collection(collections["fixed"], "kb_fixed (fixed-size-with-overlap)")
    sent_results, sent_avg_p, sent_avg_r = evaluate_collection(collections["sentence"], "kb_sentence (sentence-based)")

    print("\n=== Summary ===")
    print(f"kb_fixed:    avg precision={fixed_avg_p:.3f}  avg recall={fixed_avg_r:.3f}")
    print(f"kb_sentence: avg precision={sent_avg_p:.3f}  avg recall={sent_avg_r:.3f}")

    winner = "sentence-based" if (sent_avg_p + sent_avg_r) >= (fixed_avg_p + fixed_avg_r) else "fixed-size-with-overlap"
    print(f"\nRecommendation: deploy the {winner} chunking strategy, citing the numbers above "
          f"(kb_fixed: P={fixed_avg_p:.3f}/R={fixed_avg_r:.3f} vs kb_sentence: P={sent_avg_p:.3f}/R={sent_avg_r:.3f}).")
