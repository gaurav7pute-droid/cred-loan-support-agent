"""
crew.py  (Retrieval, Lookup,
Composer), each using its own role-configured MockCrewLLM, wired with tools
built from the Part 1 RAG core

least-autonomy wiring lives:
`check_loan_application_status` is attached to exactly one Agent's
tools=[...] list (the Lookup Agent) -- see build_agents() below and
governance.assert_least_autonomy().
"""
import json
from typing import Optional, Type

import env_setup  # noqa: F401 -- must run before crewai is imported
from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from mock_llm import MockCrewLLM
from rag_core import get_shared_collections, grounded_answer
from schemas import CredResponse
from tools import check_loan_application_status

# Chosen after Part 1 Task 5's precision/recall comparison -- see
# eval_retrieval.py output and README.md for the numbers behind this choice.
RECOMMENDED_STRATEGY = "sentence"


def _ensure_index():
    return get_shared_collections()[RECOMMENDED_STRATEGY]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
class RagLookupInput(BaseModel):
    query: str = Field(..., description="The member's Cred policy question, in natural language.")


class RagLookupTool(BaseTool):
    name: str = "rag_lookup"
    description: str = (
        "Answer a Cred loan/account policy question using the knowledge-base RAG index. "
        "Input: a natural-language policy question."
    )
    args_schema: Type[BaseModel] = RagLookupInput
    _log: dict = PrivateAttr(default_factory=dict)

    def _run(self, query: str) -> str:
        collection = _ensure_index()
        result = grounded_answer(collection, query)
        self._log["called"] = True
        self._log["result"] = result
        if not result["grounded"]:
            return result["answer"]
        sources = ",".join(sorted({h["doc_id"] for h in result["used_chunks"]}))
        return f"{result['answer']} [sources: {sources}]"


class StatusLookupInput(BaseModel):
    record_id: str = Field(..., description="The loan application record_id, e.g. CRED-0001.")


class StatusLookupTool(BaseTool):
    name: str = "check_loan_application_status"
    description: str = "Look up a loan application's status, amount, and escalation score by record_id."
    args_schema: Type[BaseModel] = StatusLookupInput
    _log: dict = PrivateAttr(default_factory=dict)

    def _run(self, record_id: str) -> str:
        result = check_loan_application_status(record_id)
        self._log["called"] = True
        self._log["result"] = result
        return json.dumps(result)


# ---------------------------------------------------------------------------
# Agents (Application-layer least-autonomy: ONLY lookup_agent gets status_tool)
# ---------------------------------------------------------------------------
def build_agents():
    rag_tool = RagLookupTool()
    status_tool = StatusLookupTool()

    retrieval_agent = Agent(
        role="Retrieval Agent",
        goal="Answer Cred policy questions using ONLY the rag_lookup knowledge-base tool.",
        backstory="A policy specialist who never answers a policy question without checking the knowledge base first.",
        tools=[rag_tool],
        llm=MockCrewLLM(role="retrieval"),
        verbose=True,
        allow_delegation=False,
    )
    lookup_agent = Agent(
        role="Lookup Agent",
        goal="Check a specific loan application's status and escalation score using check_loan_application_status.",
        backstory=(
            "An operations specialist and the ONLY member of this crew permitted to look up a "
            "member's loan application record (least-autonomy, Part 4 Task 15)."
        ),
        tools=[status_tool],
        llm=MockCrewLLM(role="lookup"),
        verbose=True,
        allow_delegation=False,
    )
    composer_agent = Agent(
        role="Response Composer",
        goal="Combine the Retrieval Agent's and Lookup Agent's results into one clear final answer for the member.",
        backstory="A support-quality specialist who writes the final member-facing answer. Has no tools of its own.",
        tools=[],
        llm=MockCrewLLM(role="composer"),
        verbose=True,
        allow_delegation=False,
    )
    return {
        "retrieval": retrieval_agent,
        "lookup": lookup_agent,
        "composer": composer_agent,
    }, rag_tool, status_tool


# ---------------------------------------------------------------------------
# Run one query through the crew and validate the result against CredResponse
# ---------------------------------------------------------------------------
def run_crew_query(query: str, record_id: Optional[str] = None):
    agents, rag_tool, status_tool = build_agents()

    retrieval_task = Task(
        # The QUERY_START/...>>>/<<</QUERY_END markers are a deliberately
        # distinctive, hard-to-lose anchor: mock_llm.py's
        # _extract_user_query() regexes them out of the full (possibly
        # framework-wrapped) task text, rather than guessing at line
        # position -- robust regardless of what boilerplate CrewAI's own
        # prompt template adds before/after this description.
        description=(
            f"Use the rag_lookup tool to answer this Cred policy question: "
            f"QUERY_START>>>{query}<<<QUERY_END"
        ),
        expected_output="A grounded answer to the policy question, or the IDK fallback text.",
        agent=agents["retrieval"],
    )
    lookup_description = (
        f"Check the status of loan application {record_id} using the check_loan_application_status tool."
        if record_id
        else (
            "No loan application record_id was given in this request. Do not invent one -- respond with "
            "exactly NO_RECORD_ID_PROVIDED."
        )
    )
    lookup_task = Task(
        description=lookup_description,
        expected_output="The application's status/escalation info as JSON, or NO_RECORD_ID_PROVIDED.",
        agent=agents["lookup"],
    )
    compose_task = Task(
        description=(
            "Combine the Retrieval Agent's and Lookup Agent's results above into one final, "
            "clear answer for the Cred member."
        ),
        expected_output="One final member-facing answer combining both results.",
        agent=agents["composer"],
        context=[retrieval_task, lookup_task],
    )

    crew = Crew(
        agents=[agents["retrieval"], agents["lookup"], agents["composer"]],
        tasks=[retrieval_task, lookup_task, compose_task],
        process=Process.sequential,
        verbose=True,
    )
    crew_output = crew.kickoff()

    rag_result = rag_tool._log.get("result")
    lookup_result = status_tool._log.get("result")

    response = CredResponse(
        answer=str(crew_output),
        used_rag=bool(rag_tool._log.get("called")),
        used_lookup=bool(status_tool._log.get("called")),
        record_id=record_id,
        grounded=bool(rag_result and rag_result.get("grounded")),
        escalation_recommended=(lookup_result.get("recommend_escalation") if lookup_result else None),
        sources=(
            sorted({h["doc_id"] for h in rag_result["used_chunks"]})
            if rag_result and rag_result.get("grounded")
            else []
        ),
    )
    task_outputs = {
        "retrieval_task_output": str(retrieval_task.output),
        "lookup_task_output": str(lookup_task.output),
        "compose_task_output": str(compose_task.output),
    }
    return response, task_outputs, agents


if __name__ == "__main__":
    from governance import assert_least_autonomy

    print("=== Query A (policy-only -- expects rag_lookup invoked, status tool NOT invoked) ===")
    response_a, outputs_a, agents_a = run_crew_query("What documents do I need for KYC?")
    print("\nValidated CredResponse:", response_a.model_dump_json(indent=2))
    assert response_a.used_rag is True, "expected rag_lookup to be invoked for a pure policy question"
    assert response_a.used_lookup is False, "expected check_loan_application_status NOT to be invoked"

    print("\n=== Query B (status-only -- expects check_loan_application_status invoked) ===")
    response_b, outputs_b, agents_b = run_crew_query(
        "What is the status of my loan application?", record_id="CRED-0010"
    )
    print("\nValidated CredResponse:", response_b.model_dump_json(indent=2))
    assert response_b.used_lookup is True, "expected check_loan_application_status to be invoked"

    print("\n=== Least-autonomy check (Part 4 Task 15) ===")
    print(assert_least_autonomy(agents_b))
    print("VERIFIED: only the Lookup Agent is wired with check_loan_application_status.")

    print("\nBoth CredResponse objects validated successfully against the Pydantic schema (Part 2 Task 9).")
