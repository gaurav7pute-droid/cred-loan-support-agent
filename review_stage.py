"""
review_stage.py -- Part 4 Task 14: a 2-agent Autogen RoundRobinGroupChat
review stage (Policy-Compliance-Reviewer + Final-Editor) that runs after the
CrewAI Composer's draft answer, using a keyless, MOCK_LLM-only
ReplayChatCompletionClient (autogen_ext.models.replay) for each agent so
this stage needs zero API keys and zero network access, per the brief.

The Final-Editor produces a structured ReviewVerdict via
output_content_type=ReviewVerdict; per the brief, the Team itself must also
be constructed with custom_message_types=[StructuredMessage[ReviewVerdict]]
or the run crashes with "Message type ... is not registered".
"""
import asyncio
import json

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.replay import ReplayChatCompletionClient

from schemas import ReviewVerdict


def _build_team(reviewer: AssistantAgent, editor: AssistantAgent) -> RoundRobinGroupChat:
    """max_turns=2 is the real RoundRobinGroupChat constructor parameter (not
    max_iterations); MaxMessageTermination(3) is the equivalent fallback,
    since MaxMessageTermination counts the initiating task message as
    message 1, so both agents only get to speak once each in a chat of 3
    messages total (task, reviewer, editor)."""
    try:
        return RoundRobinGroupChat(
            participants=[reviewer, editor],
            max_turns=2,
            custom_message_types=[StructuredMessage[ReviewVerdict]],
        )
    except TypeError:
        return RoundRobinGroupChat(
            participants=[reviewer, editor],
            termination_condition=MaxMessageTermination(3),
            custom_message_types=[StructuredMessage[ReviewVerdict]],
        )


async def run_review(
    draft_answer: str,
    retrieved_context: str,
    reviewer_scripted_reply: str,
    editor_scripted_verdict: dict,
) -> ReviewVerdict:
    """Run the 2-agent review team once. The two `*_scripted_*` arguments
    are the deterministic MOCK_LLM outputs for this run's Reviewer and
    Editor turns (see the two demo scenarios in __main__ below) -- a fresh
    ReplayChatCompletionClient is built per call since it is index-based and
    stateful."""
    reviewer_client = ReplayChatCompletionClient([reviewer_scripted_reply])
    editor_client = ReplayChatCompletionClient([json.dumps(editor_scripted_verdict)])

    reviewer = AssistantAgent(
        name="policy_compliance_reviewer",
        model_client=reviewer_client,
        system_message=(
            "You are the Policy-Compliance-Reviewer. Check the draft answer against the "
            "retrieved policy context and flag any claim that is not supported by it."
        ),
    )
    editor = AssistantAgent(
        name="final_editor",
        model_client=editor_client,
        system_message=(
            "You are the Final-Editor. Read the reviewer's feedback and produce the final "
            "verdict: approve the draft unchanged, or revise it to remove any ungrounded claim."
        ),
        output_content_type=ReviewVerdict,
    )
    team = _build_team(reviewer, editor)

    task = f"Draft answer:\n{draft_answer}\n\nRetrieved context:\n{retrieved_context}"
    result = await team.run(task=task)

    for msg in reversed(result.messages):
        content = getattr(msg, "content", None)
        if isinstance(content, ReviewVerdict):
            return content
    raise RuntimeError("Final-Editor did not produce a structured ReviewVerdict")


# ---------------------------------------------------------------------------
# Two demo scenarios (Part 4 Task 14 acceptance criteria):
#   1. the review stage approves a fully-grounded draft unchanged
#   2. the review stage revises a draft with a deliberately injected,
#      ungrounded claim
# ---------------------------------------------------------------------------
async def _demo():
    print("=== Scenario 1: fully-grounded draft -> APPROVED unchanged ===")
    draft_ok = (
        "Cash withdrawals via your Cred credit card attract a 2.5% fee with a minimum of "
        "INR 500, plus interest from the day of withdrawal with no interest-free period."
    )
    context_ok = (
        "kb-03 Credit-Card Fee Structure: Cash withdrawals via credit card attract a 2.5% fee "
        "with a minimum of INR 500, plus interest from the day of withdrawal with no "
        "interest-free period."
    )
    verdict1 = await run_review(
        draft_answer=draft_ok,
        retrieved_context=context_ok,
        reviewer_scripted_reply=(
            "The draft answer's cash-withdrawal fee figures match the retrieved context exactly. "
            "No unsupported claims found. Approve unchanged."
        ),
        editor_scripted_verdict={
            "approved": True,
            "final_answer": draft_ok,
            "reason": "Fully supported by the retrieved KB context (kb-03); no changes needed.",
        },
    )
    print(verdict1.model_dump_json(indent=2))
    assert verdict1.approved is True and verdict1.final_answer == draft_ok

    print("\n=== Scenario 2: draft with an INJECTED ungrounded claim -> REVISED ===")
    draft_bad = (
        "Cash withdrawals via your Cred credit card are completely free with no fees or "
        "limits, and you can withdraw as much as you like at any time."
    )
    context_bad = (
        "kb-03 Credit-Card Fee Structure: Cash withdrawals via credit card attract a 2.5% fee "
        "with a minimum of INR 500, plus interest from the day of withdrawal with no "
        "interest-free period."
    )
    corrected = (
        "Cash withdrawals via your Cred credit card are NOT free: they attract a 2.5% fee "
        "(minimum INR 500) plus interest from the day of withdrawal, with no interest-free period."
    )
    verdict2 = await run_review(
        draft_answer=draft_bad,
        retrieved_context=context_bad,
        reviewer_scripted_reply=(
            "The claim that cash withdrawals are 'completely free with no fees or limits' "
            "directly contradicts the retrieved context, which states a 2.5% fee (min INR 500) "
            "applies plus immediate interest. This is an ungrounded, non-compliant claim and "
            "must be corrected before it reaches a member."
        ),
        editor_scripted_verdict={
            "approved": False,
            "final_answer": corrected,
            "reason": (
                "Removed an ungrounded claim ('completely free with no fees or limits') that "
                "contradicted kb-03's documented 2.5% cash-withdrawal fee; replaced with the "
                "grounded figure."
            ),
        },
    )
    print(verdict2.model_dump_json(indent=2))
    assert verdict2.approved is False and "2.5%" in verdict2.final_answer
    print("\nVERIFIED: scenario 1 approved unchanged, scenario 2 was revised to remove the injected ungrounded claim.")


if __name__ == "__main__":
    asyncio.run(_demo())
