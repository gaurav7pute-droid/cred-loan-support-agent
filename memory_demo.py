"""
memory_demo.py --  LangChain session-based memory wrapping the
, using RunnableWithMessageHistory + an in-memory chat
history store keyed by session_id.

NOTE (expected, per the brief): RunnableWithMessageHistory raises a
LangChainDeprecationWarning pointing at LangGraph's persistence layer. This
is expected and is not silenced -- the class still functions correctly here,
and this project's memory only needs to survive one process run (Task 8).
"""
import re
from typing import List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

try:
    from langchain_core.chat_history import InMemoryChatMessageHistory as _LCHistory

    def _new_history() -> BaseChatMessageHistory:
        return _LCHistory()
except ImportError:  # pragma: no cover -- fallback for older langchain-core
    from langchain_core.messages import BaseMessage
    from pydantic import BaseModel, Field

    class _FallbackInMemoryHistory(BaseChatMessageHistory, BaseModel):
        messages: List[BaseMessage] = Field(default_factory=list)

        def add_messages(self, messages: List[BaseMessage]) -> None:
            self.messages.extend(messages)

        def clear(self) -> None:
            self.messages = []

    def _new_history() -> BaseChatMessageHistory:
        return _FallbackInMemoryHistory()

from crew import run_crew_query

_RECORD_ID_RE = re.compile(r"\bCRED-\d{4}\b")
_SESSION_STORE: dict = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = _new_history()
    return _SESSION_STORE[session_id]


def _last_record_id_from_history(history_messages) -> Optional[str]:
    for msg in reversed(history_messages):
        content = getattr(msg, "content", "") or ""
        m = _RECORD_ID_RE.search(content)
        if m:
            return m.group(0)
    return None


def _answer(inputs: dict) -> str:
    """The underlying runnable: resolves a record_id either from the current
    question or, if absent, from this session's own prior turns -- this is
    the concrete, verifiable proof that history is actually being used
    (not just cosmetically carried)."""
    question = inputs["question"]
    history = inputs.get("history", [])

    m = _RECORD_ID_RE.search(question)
    record_id = m.group(0) if m else _last_record_id_from_history(history)

    response, _, _ = run_crew_query(question, record_id=record_id)
    return f"[resolved_record_id={record_id}] {response.answer}"


answer_runnable = RunnableLambda(_answer)

chain_with_memory = RunnableWithMessageHistory(
    answer_runnable,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)


if __name__ == "__main__":
    print("=== Transcript 1: multi-turn memory CARRIED across turns (session 'member-42') ===")
    config = {"configurable": {"session_id": "member-42"}}

    turn1 = chain_with_memory.invoke(
        {"question": "What is the status of my loan application CRED-0010?"}, config=config
    )
    print("Turn 1:", turn1)

    turn2 = chain_with_memory.invoke({"question": "Is that one flagged for fraud review?"}, config=config)
    print("Turn 2:", turn2)
    assert "resolved_record_id=CRED-0010" in turn2, (
        "expected turn 2 to reuse CRED-0010 from session history even though turn 2's "
        "question never restates the record_id"
    )
    print("VERIFIED: turn 2 correctly reused CRED-0010 carried over from turn 1's session history.\n")

    print("=== Transcript 2 (SEPARATE fresh conversation, session 'member-99') ===")
    config2 = {"configurable": {"session_id": "member-99"}}
    turn1_fresh = chain_with_memory.invoke(
        {"question": "Is that one flagged for fraud review?"}, config=config2
    )
    print("Turn 1 (fresh session, no prior turns):", turn1_fresh)
    assert "resolved_record_id=None" in turn1_fresh, (
        "expected a brand-new session to have NO prior record_id to resolve -- state must be absent here"
    )
    print("VERIFIED: fresh session 'member-99' correctly has no memory of 'member-42' session's record_id.")
