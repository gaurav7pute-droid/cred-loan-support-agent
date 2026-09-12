"""
governance.py -- the four-layer AI-governance model applied
at the Application and Runtime layers.

Application layer -- principle of least autonomy
--------------------------------------------------
Only the Lookup Agent may call check_loan_application_status. This is
enforced structurally, not by convention: the tool function lives in
tools.py and crew.py wires it into exactly ONE CrewAI Agent's `tools=[...]`
list (the Lookup Agent). The Retrieval Agent and Composer agent are never
given a reference to it, so there is nothing in their toolset for the
underlying ReAct loop to invoke -- an agent can only call a tool CrewAI told
it about via its own `tools` list, so the guard is "don't wire it" rather
than a runtime permission check. GUARD_DEMO below demonstrates this by
inspecting each agent's tool list at runtime and asserting the invariant.

Risk classification
---------------------
This system is classified HIGH RISK. It handles individual members' loan
records (financial data) and a fraud-escalation recommendation that could
affect a real support decision -- which places it squarely in the brief's
High tier ("medical data, hiring decisions, financial data"), not Medium
(customer-support tickets in the abstract) or Low (summarization). Because
the domain is financial and a wrong or fabricated answer about eligibility,
fees, or fraud-dispute timelines carries real member and regulatory
consequences, this system carries mandatory groundedness refusal (Task 10),
mandatory PII masking on ingestion and on logs (Tasks 10, 12), a second
independent review stage before any answer reaches a member (Task 14), and
a runtime cost cap (below) -- controls appropriate to High, not Medium, risk.

Runtime layer -- per-request token/cost budget cap
-----------------------------------------------------
Each request is charged an estimated token cost (a simple 4-chars-per-token
heuristic over the prompt + expected completion) against a fixed per-request
budget. A request that would exceed the budget is rejected before any
LLM/tool call is made, rather than silently allowed to run over cost.
"""
from dataclasses import dataclass

from tools import check_loan_application_status

# ---------------------------------------------------------------------------
# Application layer: least-autonomy enforcement + demonstration
# ---------------------------------------------------------------------------
RISK_LEVEL = "High"
RISK_JUSTIFICATION = (
    "Financial data (individual loan records, amounts, statuses) plus a fraud-"
    "escalation recommendation that can influence a real support decision -> "
    "High risk per the Low/Medium/High scheme, requiring groundedness refusal, "
    "PII masking end-to-end, independent second-agent review, and a runtime "
    "cost cap."
)


def assert_least_autonomy(crew_agents: dict) -> dict:
    """crew_agents: {'retrieval': Agent, 'lookup': Agent, 'composer': Agent}
    Returns a report and raises AssertionError if the invariant is violated.
    """
    report = {}
    for name, agent in crew_agents.items():
        tool_names = [getattr(t, "name", getattr(t, "__name__", str(t))) for t in getattr(agent, "tools", []) or []]
        has_lookup_tool = any("check_loan_application_status" in n or "status" in n.lower() for n in tool_names)
        report[name] = {"tool_names": tool_names, "has_lookup_tool": has_lookup_tool}
        if name != "lookup":
            assert not has_lookup_tool, (
                f"LEAST-AUTONOMY VIOLATION: agent '{name}' was wired with the loan-status "
                f"lookup tool. Only the Lookup Agent may call check_loan_application_status."
            )
    assert report["lookup"]["has_lookup_tool"], "Lookup Agent must be wired with the lookup tool."
    return report


# ---------------------------------------------------------------------------
# Runtime layer: per-request token/cost budget cap
# ---------------------------------------------------------------------------
CHARS_PER_TOKEN = 4.0
MAX_TOKENS_PER_REQUEST = 2000  # runtime budget cap
COST_PER_1K_TOKENS_USD = 0.002  # illustrative rate for the cost figure shown to the user


@dataclass
class BudgetCheckResult:
    estimated_tokens: int
    estimated_cost_usd: float
    budget_tokens: int
    allowed: bool
    reason: str = ""


def estimate_tokens(text: str, expected_completion_tokens: int = 300) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + expected_completion_tokens


def check_budget(prompt_text: str, expected_completion_tokens: int = 300,
                  budget_tokens: int = MAX_TOKENS_PER_REQUEST) -> BudgetCheckResult:
    estimated = estimate_tokens(prompt_text, expected_completion_tokens)
    cost = round((estimated / 1000) * COST_PER_1K_TOKENS_USD, 5)
    if estimated > budget_tokens:
        return BudgetCheckResult(
            estimated_tokens=estimated,
            estimated_cost_usd=cost,
            budget_tokens=budget_tokens,
            allowed=False,
            reason=(
                f"Estimated {estimated} tokens exceeds the per-request budget of "
                f"{budget_tokens} tokens -- request rejected before any LLM/tool call."
            ),
        )
    return BudgetCheckResult(
        estimated_tokens=estimated, estimated_cost_usd=cost, budget_tokens=budget_tokens, allowed=True
    )


if __name__ == "__main__":
    print(f"Risk classification: {RISK_LEVEL}")
    print(f"Justification: {RISK_JUSTIFICATION}")

    print("\n--- Runtime budget cap demo ---")
    normal_request = "What documents do I need for KYC?"
    oversized_request = "Please explain in extreme detail: " + ("loan policy edge case. " * 400)

    r_ok = check_budget(normal_request)
    r_bad = check_budget(oversized_request)
    print(f"normal request  ({len(normal_request)} chars): {r_ok}")
    print(f"oversized request ({len(oversized_request)} chars): {r_bad}")
    assert r_ok.allowed and not r_bad.allowed
    print("VERIFIED: oversized request correctly rejected before any LLM/tool call; normal request allowed.")
