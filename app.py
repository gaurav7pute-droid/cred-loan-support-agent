"""


Endpoints:
  POST /ask                  -- ask a policy question / check a loan status
  POST /add-document          -- add a new KB document, chunk+embed+index it live
  WS   /ws/chat                -- real-time multi-turn chat (session memory)

Every request produces one structured JSON-Lines log entry (Task 12) with a
trace ID and timing, and the logged query text is masked with the SAME
guardrail used before the model ever sees it (Task 10) -- a fixed-format PII
field never reaches disk in the clear.
"""
import json
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from langchain_core.messages import AIMessage, HumanMessage

from cache import ResponseCache
from crew import run_crew_query
from governance import check_budget
from guardrails import apply_input_guardrails
from memory_demo import _RECORD_ID_RE, _last_record_id_from_history, get_session_history
from rag_core import get_embedder, get_shared_collections
from chunking import fixed_size_overlap_chunks, sentence_based_chunks
from schemas import AddDocumentRequest, AddDocumentResponse, AskRequest, AskResponse
from kb import KNOWLEDGE_BASE

app = FastAPI(title="Cred Domain Support Agent")

_cache = ResponseCache()
_log_lock = threading.Lock()
LOG_PATH = Path(__file__).parent / "logs" / "requests.jsonl"
LOG_PATH.parent.mkdir(exist_ok=True)

RECOMMENDED_STRATEGY = "sentence"


def log_request(trace_id: str, endpoint: str, session_id: str, masked_query: str,
                 latency_ms: float, extra: dict = None) -> None:
    entry = {
        "trace_id": trace_id,
        "endpoint": endpoint,
        "session_id": session_id,
        "timestamp": time.time(),
        "query_masked": masked_query,  # NEVER the raw text -- already guardrail-masked before this call
        "latency_ms": round(latency_ms, 2),
    }
    if extra:
        entry.update(extra)
    line = json.dumps(entry)
    with _log_lock:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def handle_query(raw_query: str, session_id: str, endpoint: str) -> AskResponse:
    trace_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    guard = apply_input_guardrails(raw_query)
    safe_text = guard["safe_text"]  # PII-masked, used for logging AND as the effective query text

    budget = check_budget(safe_text)
    if not budget.allowed:
        latency_ms = (time.perf_counter() - t0) * 1000
        log_request(trace_id, endpoint, session_id, safe_text, latency_ms,
                    {"status": "rejected_budget", "reason": budget.reason})
        raise HTTPException(status_code=400, detail=budget.reason)

    history = get_session_history(session_id)
    m = _RECORD_ID_RE.search(safe_text)
    record_id = m.group(0) if m else _last_record_id_from_history(history.messages)
    # The cache key MUST include the resolved record_id, not just the query
    # text: two different sessions can send the identical follow-up phrase
    # (e.g. "Is that one flagged for fraud review?") that resolves to a
    # DIFFERENT record_id per session's own history. Keying on text alone
    # would let one member's loan-application answer leak into another
    # session's cache hit. safe_text is stripped BEFORE concatenation --
    # cache.normalize_query() only strips the true edges of the combined
    # key, so any leading/trailing whitespace on safe_text itself would
    # otherwise land in the middle of the key (after "record_id::") and
    # survive normalization as a stray space, defeating cache hits for
    # queries that differ only in incidental leading/trailing whitespace.
    cache_key = f"{record_id or ''}::{safe_text.strip()}"

    cached = _cache.get(cache_key)
    if cached is not None:
        latency_ms = (time.perf_counter() - t0) * 1000
        log_request(trace_id, endpoint, session_id, safe_text, latency_ms,
                    {"status": "ok", "cache_hit": True})
        return AskResponse(
            trace_id=trace_id, session_id=session_id, response=cached, cache_hit=True,
            latency_ms=round(latency_ms, 2),
        )

    response, _task_outputs, _agents = run_crew_query(safe_text, record_id=record_id)
    _cache.set(cache_key, response)

    history.add_messages([HumanMessage(content=safe_text), AIMessage(content=response.answer)])

    latency_ms = (time.perf_counter() - t0) * 1000
    log_request(trace_id, endpoint, session_id, safe_text, latency_ms,
                {"status": "ok", "cache_hit": False, "injection_blocked": guard["blocked"],
                 "pii_masked": guard["pii_detected"]})
    return AskResponse(
        trace_id=trace_id, session_id=session_id, response=response, cache_hit=False,
        latency_ms=round(latency_ms, 2),
    )


@app.on_event("startup")
def _on_startup():
    get_shared_collections()  # build/warm the index once at process startup


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return handle_query(req.query, req.session_id, endpoint="POST /ask")


@app.post("/add-document", response_model=AddDocumentResponse)
def add_document(req: AddDocumentRequest) -> AddDocumentResponse:
    trace_id = str(uuid.uuid4())
    doc = {"doc_id": req.doc_id, "topic": req.topic, "title": req.title, "text": req.text}

    fixed_chunks = fixed_size_overlap_chunks(doc)
    sentence_chunks = sentence_based_chunks(doc)

    collections = get_shared_collections()
    embedder = get_embedder()
    for chunks, coll_key in ((fixed_chunks, "fixed"), (sentence_chunks, "sentence")):
        if not chunks:
            continue
        texts = [c["text"] for c in chunks]
        embeddings = embedder.encode(texts, normalize_embeddings=True).tolist()
        collections[coll_key].upsert(
            ids=[c["chunk_id"] for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"doc_id": c["doc_id"], "topic": c["topic"]} for c in chunks],
        )

    KNOWLEDGE_BASE.append(doc)  

    log_request(trace_id, "POST /add-document", "n/a", f"[doc_id={req.doc_id}]", 0.0, {"status": "ok"})
    return AddDocumentResponse(
        trace_id=trace_id, doc_id=req.doc_id,
        fixed_chunks_added=len(fixed_chunks), sentence_chunks_added=len(sentence_chunks),
    )


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    await websocket.accept()
    session_id = f"ws-{uuid.uuid4().hex[:8]}"
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
                query = payload.get("query", "")
                session_id = payload.get("session_id", session_id)
            except json.JSONDecodeError:
                query = raw
            try:
                response = handle_query(query, session_id, endpoint="WS /ws/chat")
                await websocket.send_text(response.model_dump_json())
            except HTTPException as e:
                await websocket.send_text(json.dumps({"error": e.detail}))
    except WebSocketDisconnect:
        
        log_request(str(uuid.uuid4()), "WS /ws/chat", session_id, "[client disconnected]", 0.0,
                    {"status": "disconnected"})
        return


if __name__ == "__main__":
    import uvicorn

    get_shared_collections()
    uvicorn.run(app, host="127.0.0.1", port=8000)
