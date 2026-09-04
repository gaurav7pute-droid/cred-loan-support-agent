"""
tools.py -- Part 2 Task 6: check_loan_application_status with a DESIGNED
escalation_score (not a bare boolean OR of flagged_for_fraud_review).

Escalation-score formula
-------------------------
    fraud_signal   = 1.0 if flagged_for_fraud_review else 0.0
    recency_signal = min(days_since_created, 30) / 30        (normalized to [0, 1])
    escalation_score = 0.5 * fraud_signal + 0.5 * recency_signal

Both terms carry equal weight (0.5 each) so the score is a genuine
combination rather than fraud dominating outright: a fresh (days=0)
fraud-flagged application (score=0.50) lands at roughly the same urgency as
a 30-day-old unflagged application still stuck in the pipeline (score=0.50)
-- both represent a real operational risk (active fraud signal vs. an SLA
breach risk) worth a human's attention, and a flagged *and* stale
application (score up to 1.0) is unambiguously the top priority.

Escalation threshold
---------------------
ESCALATION_THRESHOLD is set to the 80th percentile of escalation_score
across the actual generated LOAN_APPLICATIONS dataset (computed below, not
hardcoded), i.e. the top 20% highest-urgency applications -- whether that
urgency comes from the fraud flag, from having sat unresolved near the
30-day ceiling, or from both -- are recommended for escalation.
"""
from dataset import LOAN_APPLICATIONS, RECORDS_BY_ID

FRAUD_WEIGHT = 0.5
RECENCY_WEIGHT = 0.5
MAX_DAYS = 30


def _percentile(sorted_values: list, pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f, c = int(k), min(int(k) + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def compute_escalation_score(flagged_for_fraud_review: bool, days_since_created: int) -> float:
    fraud_signal = 1.0 if flagged_for_fraud_review else 0.0
    recency_signal = min(max(days_since_created, 0), MAX_DAYS) / MAX_DAYS
    return round(FRAUD_WEIGHT * fraud_signal + RECENCY_WEIGHT * recency_signal, 4)


_ALL_SCORES = sorted(
    compute_escalation_score(r["flagged_for_fraud_review"], r["days_since_created"])
    for r in LOAN_APPLICATIONS
)
ESCALATION_THRESHOLD = round(_percentile(_ALL_SCORES, 80), 4)


def check_loan_application_status(record_id: str) -> dict:
    """Tool used ONLY by the Lookup Agent (see governance.py Task 15 for the
    least-autonomy enforcement that keeps every other agent from calling it).
    """
    record = RECORDS_BY_ID.get(record_id)
    if record is None:
        return {
            "record_id": record_id,
            "found": False,
            "error": f"No loan application found with record_id={record_id!r}",
        }
    score = compute_escalation_score(record["flagged_for_fraud_review"], record["days_since_created"])
    return {
        "record_id": record_id,
        "found": True,
        "status": record["status"],
        "loan_amount_inr": record["loan_amount_inr"],
        "flagged_for_fraud_review": record["flagged_for_fraud_review"],
        "days_since_created": record["days_since_created"],
        "escalation_score": score,
        "escalation_threshold": ESCALATION_THRESHOLD,
        "recommend_escalation": score > ESCALATION_THRESHOLD,
    }


if __name__ == "__main__":
    print(f"escalation_score formula: {FRAUD_WEIGHT}*fraud_signal + {RECENCY_WEIGHT}*recency_signal")
    print(f"ESCALATION_THRESHOLD (80th percentile of escalation_score across "
          f"{len(LOAN_APPLICATIONS)} generated records) = {ESCALATION_THRESHOLD}")

    # a flagged, fresh record
    flagged_example = next(r for r in LOAN_APPLICATIONS if r["flagged_for_fraud_review"])
    # an unflagged, stale record
    stale_example = max(
        (r for r in LOAN_APPLICATIONS if not r["flagged_for_fraud_review"]),
        key=lambda r: r["days_since_created"],
    )
    for label, rec in [("flagged example", flagged_example), ("stale unflagged example", stale_example)]:
        print(f"\n{label}: {rec['record_id']}")
        print(" ", check_loan_application_status(rec["record_id"]))

    print("\nUnknown record_id:")
    print(" ", check_loan_application_status("CRED-9999"))
