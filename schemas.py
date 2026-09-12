"""
schemas.py  the Pydantic structured-output schema every
crew response must conform to, plus the Autogen review-stage verdict model
 and the FastAPI request/response models .
"""
from typing import List, Optional

from pydantic import BaseModel, Field


class CredResponse(BaseModel):
    """response_format for the CrewAI Composer agent's final answer
    (Part 2 Task 9). Every crew.kickoff() result is validated against this
    model in code before it is returned to a caller."""

    answer: str = Field(..., description="The final answer text shown to the member.")
    used_rag: bool = Field(..., description="Whether the Retrieval Agent's RAG tool was consulted.")
    used_lookup: bool = Field(..., description="Whether the Lookup Agent's status tool was consulted.")
    record_id: Optional[str] = Field(None, description="Loan application record_id, if one was looked up.")
    grounded: bool = Field(..., description="Whether the RAG portion of the answer met the groundedness threshold.")
    escalation_recommended: Optional[bool] = Field(
        None, description="Escalation recommendation from check_loan_application_status, if a lookup occurred."
    )
    sources: List[str] = Field(default_factory=list, description="KB doc_ids cited in the answer, if any.")


class ReviewVerdict(BaseModel):
     the Autogen Final-Editor's structured verdict."""

    approved: bool
    final_answer: str
    reason: str


# ---------------------------------------------------------------------------
# FastAPI request/response models (Part 3 Task 11)
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    # max_length is intentionally generous (not a hard content cap) -- the
    # actual per-request cost gate is governance.check_budget(), demonstrated
    # rejecting an oversized request in Part 4 Task 15 and wired live here.
    query: str = Field(..., min_length=1, max_length=20000)
    session_id: str = Field(default="default", description="Conversation/session id for memory.")


class AskResponse(BaseModel):
    trace_id: str
    session_id: str
    response: CredResponse
    cache_hit: bool
    latency_ms: float


class AddDocumentRequest(BaseModel):
    doc_id: str = Field(..., min_length=1, max_length=64)
    topic: str = Field(..., min_length=1, max_length=200)
    title: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=1, max_length=4000)


class AddDocumentResponse(BaseModel):
    trace_id: str
    doc_id: str
    fixed_chunks_added: int
    sentence_chunks_added: int
