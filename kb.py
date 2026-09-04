"""
kb.py -- The Cred lending-operations knowledge base.

Every document below is original text written for this capstone, covering
one required topic from the brief (>=12 documents, 2-5 sentences each, every
required topic covered at least once). KNOWLEDGE_BASE is the list rag_core.py
chunks and indexes; each entry's `doc_id` is the unit precision/recall in
Part 1 Task 5 is scored against after mapping chunks back to their parent.
"""

KNOWLEDGE_BASE = [
    {
        "doc_id": "kb-01",
        "topic": "loan eligibility criteria by loan type",
        "title": "Loan Eligibility Criteria by Loan Type",
        "text": (
            "Personal Loans require the applicant to be a salaried or self-employed "
            "individual aged 21-58 with a minimum monthly income of INR 25,000 and at "
            "least 12 months in the current job or business. Home Loans require a "
            "minimum income of INR 40,000 per month, a maximum age of 65 at loan "
            "maturity, and the property being financed must be free of legal disputes. "
            "Auto Loans require a valid driving license or a co-applicant who holds one, "
            "plus a minimum income of INR 20,000 per month. Education Loans are "
            "available to Indian students with a confirmed admission offer at a "
            "recognized institution, with a parent or guardian as co-applicant for loans "
            "above INR 750,000. Business Loans require the business to have been "
            "operational for at least 2 years with audited financials for the most "
            "recent fiscal year."
        ),
    },
    {
        "doc_id": "kb-02",
        "topic": "EMI calculation rules",
        "title": "EMI Calculation Rules",
        "text": (
            "The Equated Monthly Installment (EMI) is calculated using the standard "
            "reducing-balance formula EMI = P x r x (1+r)^n / ((1+r)^n - 1), where P is "
            "the principal, r is the monthly interest rate (annual rate divided by 12 "
            "and by 100), and n is the tenure in months. EMIs are due on the 5th of "
            "every month, and the first EMI is charged one full calendar month after "
            "disbursement. A grace period of 3 days is allowed before a payment is "
            "marked late. Foreclosure or part-payment reduces either the tenure or the "
            "EMI amount at the borrower's choice, and the EMI schedule is recalculated "
            "from the next billing cycle."
        ),
    },
    {
        "doc_id": "kb-03",
        "topic": "credit-card fee structure",
        "title": "Credit-Card Fee Structure",
        "text": (
            "Cred-linked credit cards charge an annual fee between INR 500 and INR "
            "5,000 depending on the card tier, which is waived if annual spend exceeds "
            "INR 200,000. A late payment fee of INR 100 to INR 1,300 applies on a "
            "sliding scale based on the outstanding amount if the minimum due is not "
            "paid by the due date. Cash withdrawals via credit card attract a 2.5% fee "
            "with a minimum of INR 500, plus interest from the day of withdrawal with no "
            "interest-free period. Foreign currency transactions carry a 3.5% markup "
            "fee. Overlimit transactions, when permitted, incur a flat 2.5% overlimit "
            "fee on the amount exceeding the credit limit."
        ),
    },
    {
        "doc_id": "kb-04",
        "topic": "KYC document requirements",
        "title": "KYC Document Requirements",
        "text": (
            "All applicants must submit one government-issued photo ID (Aadhaar, "
            "Passport, Voter ID, or Driving License) and one address proof, which may "
            "be the same document if it shows a current address. PAN card is mandatory "
            "for any loan or credit product above INR 50,000 or any credit card "
            "application, per RBI and Income Tax rules. Self-employed applicants must "
            "additionally submit the last 2 years of income tax returns and 6 months of "
            "business bank statements. NRI applicants must submit a valid passport, "
            "visa or work permit, and an overseas address proof in addition to the "
            "standard KYC set. All KYC documents must be re-verified every 10 years for "
            "individual accounts and every 8 years for higher-risk categories, per "
            "periodic KYC update rules."
        ),
    },
    {
        "doc_id": "kb-05",
        "topic": "fraud-dispute resolution process",
        "title": "Fraud-Dispute Resolution Process",
        "text": (
            "A member who spots an unauthorized transaction must report it within 3 "
            "days of the transaction date to receive zero-liability protection under "
            "RBI's limited-liability guidelines. Upon report, the card or account is "
            "immediately blocked and a provisional credit is issued within 10 working "
            "days for card-not-present disputes while the investigation proceeds. The "
            "fraud investigation team has up to 90 days to complete the review for "
            "domestic transactions and 120 days for international ones. If the dispute "
            "is upheld, the provisional credit is made permanent; if rejected, the "
            "member is notified in writing with the specific reason and supporting "
            "evidence before any reversal of the provisional credit."
        ),
    },
    {
        "doc_id": "kb-06",
        "topic": "account-closure process",
        "title": "Account-Closure Process",
        "text": (
            "Members may request closure of a savings account, credit card, or loan "
            "account through the app, provided all dues are cleared and there is no "
            "pending dispute. Savings accounts closed within 14 days of opening are "
            "closed free of charge; closure between 14 days and 12 months incurs a "
            "closure fee of INR 500. Credit card closure requires the outstanding "
            "balance to be paid in full and any linked reward points are forfeited "
            "after 30 days unless redeemed beforehand. Loan accounts can only be closed "
            "via full prepayment or at natural tenure completion, and a no-dues "
            "certificate is issued within 7 working days of closure."
        ),
    },
    {
        "doc_id": "kb-07",
        "topic": "interest-rate slabs",
        "title": "Interest-Rate Slabs",
        "text": (
            "Personal Loan interest rates range from 11.5% to 24% per annum based on "
            "the applicant's credit score band, with scores above 750 qualifying for "
            "the lowest slab. Home Loan rates range from 8.5% to 11% per annum and are "
            "linked to the repo-linked lending rate, resetting quarterly. Auto Loan "
            "rates range from 9% to 14% per annum for new vehicles and 13% to 18% for "
            "used vehicles. Education Loan rates range from 9.5% to 13% per annum, with "
            "a 0.5% concession for female applicants. Business Loan rates range from "
            "14% to 22% per annum depending on collateral and business vintage."
        ),
    },
    {
        "doc_id": "kb-08",
        "topic": "prepayment-penalty rules",
        "title": "Prepayment-Penalty Rules",
        "text": (
            "Floating-rate Personal, Home, and Business Loans taken by individual "
            "borrowers carry zero prepayment penalty, in line with RBI guidelines for "
            "floating-rate retail loans. Fixed-rate loans of any type carry a "
            "prepayment penalty of 2% to 4% of the outstanding principal if foreclosed "
            "within the first 12 months, reducing to 1% to 2% thereafter. Part-payments "
            "are capped at 4 per financial year and each part-payment must be at least "
            "10% of the outstanding principal. Business Loans issued to non-individual "
            "entities (companies, partnerships) may carry prepayment penalties even on "
            "floating rates, as the RBI exemption applies only to individual borrowers."
        ),
    },
    {
        "doc_id": "kb-09",
        "topic": "minimum-balance requirements",
        "title": "Minimum-Balance Requirements",
        "text": (
            "Regular savings accounts require an average monthly balance of INR 5,000 "
            "in metro areas and INR 2,500 in non-metro areas; falling short attracts a "
            "penalty of up to INR 400 per quarter, capped at the shortfall amount. Zero-"
            "balance Basic Savings Bank Deposit accounts have no minimum balance "
            "requirement, per RBI financial-inclusion rules, but carry transaction "
            "limits. Salary accounts are exempt from minimum-balance rules for as long "
            "as a salary credit is received each month, converting to a regular savings "
            "account with standard minimum-balance rules after 3 consecutive months "
            "without a salary credit. NRI (NRO/NRE) accounts require a minimum average "
            "balance of INR 10,000."
        ),
    },
    {
        "doc_id": "kb-10",
        "topic": "credit-score impact factors",
        "title": "Credit-Score Impact Factors",
        "text": (
            "Payment history is the single largest factor in a member's credit score, "
            "accounting for roughly 35% of the score, and even one payment more than 30 "
            "days late can lower a score by 50-100 points. Credit utilization -- the "
            "ratio of outstanding balance to total credit limit -- accounts for about "
            "30%, with utilization above 30% starting to hurt the score. The length of "
            "credit history, the mix of credit types (secured versus unsecured), and "
            "the number of recent hard inquiries make up most of the remainder. "
            "Closing an old credit card can lower the score by shortening credit "
            "history and raising utilization, so Cred recommends keeping unused cards "
            "open unless they carry a high annual fee."
        ),
    },
    {
        "doc_id": "kb-11",
        "topic": "joint-account rules",
        "title": "Joint-Account Rules",
        "text": (
            "Joint accounts can be opened as 'Either or Survivor,' 'Former or "
            "Survivor,' or 'Joint' (all signatures required), and the mode must be "
            "declared at account opening. All joint holders must individually complete "
            "KYC; the account cannot be opened if any holder's KYC is incomplete. For "
            "joint loans, all co-borrowers are jointly and severally liable for the "
            "full outstanding amount, and a missed payment affects the credit score of "
            "every co-borrower, not just the primary applicant. Removing or adding a "
            "joint holder on an existing account requires the written consent of all "
            "current holders and re-verification of KYC for any newly added holder."
        ),
    },
    {
        "doc_id": "kb-12",
        "topic": "NRI-account eligibility",
        "title": "NRI-Account Eligibility",
        "text": (
            "Non-Resident Indians (NRIs) may open an NRE (Non-Resident External) "
            "account to hold foreign earnings repatriated to India, or an NRO (Non-"
            "Resident Ordinary) account to manage income earned within India such as "
            "rent or dividends. NRE account balances and interest are fully "
            "repatriable and tax-free in India, while NRO account interest is subject "
            "to TDS and repatriation is capped at USD 1 million per financial year "
            "after tax compliance certification. Eligibility requires valid proof of "
            "NRI status (visa, work permit, or overseas residence permit), a valid "
            "passport, and an overseas address proof. NRIs applying for a Home Loan "
            "under this program must additionally nominate a Power-of-Attorney holder "
            "resident in India to complete local formalities."
        ),
    },
]

TOPIC_TO_DOC_ID = {d["topic"]: d["doc_id"] for d in KNOWLEDGE_BASE}
DOC_BY_ID = {d["doc_id"]: d for d in KNOWLEDGE_BASE}

REQUIRED_TOPICS = [
    "loan eligibility criteria by loan type",
    "EMI calculation rules",
    "credit-card fee structure",
    "KYC document requirements",
    "fraud-dispute resolution process",
    "account-closure process",
    "interest-rate slabs",
    "prepayment-penalty rules",
    "minimum-balance requirements",
    "credit-score impact factors",
    "joint-account rules",
    "NRI-account eligibility",
]

if __name__ == "__main__":
    missing = [t for t in REQUIRED_TOPICS if t not in TOPIC_TO_DOC_ID]
    print(f"Documents: {len(KNOWLEDGE_BASE)}")
    print(f"Required topics covered: {len(REQUIRED_TOPICS) - len(missing)}/{len(REQUIRED_TOPICS)}")
    if missing:
        print("MISSING:", missing)
    for d in KNOWLEDGE_BASE:
        n_sentences = d["text"].count(". ") + 1
        print(f"  {d['doc_id']:8s} ~{n_sentences} sentences  -- {d['title']}")
