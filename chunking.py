"""
chunking.py 

Both operate on a single KB document's `text` field and return a list of
chunk dicts: {"chunk_id", "doc_id", "topic", "text"}. Kept dependency-free
(no embeddings/Chroma here) so the chunking logic itself can be unit-tested
without needing sentence-transformers or ChromaDB installed.
"""
import re

# ---------------------------------------------------------------------------
# Strategy A: fixed-size-with-overlap (character-based sliding window)
# ---------------------------------------------------------------------------
FIXED_CHUNK_SIZE = 220   # characters
FIXED_CHUNK_OVERLAP = 60  # characters


def fixed_size_overlap_chunks(doc: dict, chunk_size: int = FIXED_CHUNK_SIZE,
                               overlap: int = FIXED_CHUNK_OVERLAP) -> list:
    text = doc["text"].strip()
    chunks = []
    start = 0
    idx = 0
    n = len(text)
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    while start < n:
        end = min(start + chunk_size, n)
        # avoid cutting a word in half: extend to the next whitespace if we're
        # not already at the end of the document
        if end < n:
            while end < n and not text[end].isspace():
                end += 1
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                {
                    "chunk_id": f"{doc['doc_id']}::fixed::{idx}",
                    "doc_id": doc["doc_id"],
                    "topic": doc["topic"],
                    "text": chunk_text,
                }
            )
            idx += 1
        if end >= n:
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# Strategy B: sentence-based chunking (one sentence == one chunk)
# ---------------------------------------------------------------------------
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def split_sentences(text: str) -> list:
    text = text.strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def sentence_based_chunks(doc: dict) -> list:
    sentences = split_sentences(doc["text"])
    chunks = []
    for idx, sent in enumerate(sentences):
        chunks.append(
            {
                "chunk_id": f"{doc['doc_id']}::sent::{idx}",
                "doc_id": doc["doc_id"],
                "topic": doc["topic"],
                "text": sent,
            }
        )
    return chunks


def chunk_all(documents: list, strategy: str) -> list:
    """strategy in {'fixed', 'sentence'}"""
    fn = fixed_size_overlap_chunks if strategy == "fixed" else sentence_based_chunks
    all_chunks = []
    for doc in documents:
        all_chunks.extend(fn(doc))
    return all_chunks


if __name__ == "__main__":
    from kb import KNOWLEDGE_BASE

    fixed = chunk_all(KNOWLEDGE_BASE, "fixed")
    sentence = chunk_all(KNOWLEDGE_BASE, "sentence")
    print(f"fixed-size-with-overlap: {len(fixed)} chunks from {len(KNOWLEDGE_BASE)} docs "
          f"({len(fixed)/len(KNOWLEDGE_BASE):.1f} chunks/doc avg)")
    print(f"sentence-based:          {len(sentence)} chunks from {len(KNOWLEDGE_BASE)} docs "
          f"({len(sentence)/len(KNOWLEDGE_BASE):.1f} chunks/doc avg)")
    print("\nSample doc kb-05 chunked both ways:")
    from kb import DOC_BY_ID
    for c in fixed_size_overlap_chunks(DOC_BY_ID["kb-05"]):
        print("  [fixed]  ", c["chunk_id"], "->", c["text"][:80], "...")
    for c in sentence_based_chunks(DOC_BY_ID["kb-05"]):
        print("  [sentence]", c["chunk_id"], "->", c["text"][:80], "...")
