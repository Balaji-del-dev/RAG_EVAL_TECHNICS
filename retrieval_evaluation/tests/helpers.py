"""The worked example from the design, shared by the tests."""

WORKED_RETRIEVED = ["C104", "C101", "C105", "C102", "C106"]
WORKED_GRADES = {"C101": 3, "C102": 2, "C103": 1}
WORKED_RELEVANT = {"C101", "C102", "C103"}

WORKED_DATASET = {
    "name": "Worked",
    "version": "1.0",
    "chunks": [
        {"chunk_id": cid, "title": title, "text": text}
        for cid, title, text in [
            ("C101", "Maternity eligibility", "maternity leave eligibility after 80 days of service"),
            ("C102", "Maternity duration", "maternity leave lasts 26 weeks"),
            ("C103", "Application process", "apply for maternity leave in the HR portal"),
            ("C104", "Paternity leave", "paternity leave policy is 5 days, separate from maternity leave"),
            ("C105", "Work from home", "work from home two days a week"),
            ("C106", "Employee insurance", "group health insurance covers maternity hospitalisation"),
        ]
    ],
    "queries": [
        {
            "query_id": "Q1",
            "text": "What is the maternity leave policy?",
            "ground_truth": [{"chunk_id": c, "grade": g} for c, g in WORKED_GRADES.items()],
        }
    ],
}
