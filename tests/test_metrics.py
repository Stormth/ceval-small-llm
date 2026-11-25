from ceval_eval.metrics import compute_summary


def row(subject, correct, valid=True, latency=1.0, calls=1):
    return {
        "subject": subject,
        "category": "STEM",
        "correct": correct,
        "format_valid": valid,
        "truncated": False,
        "latency_sec": latency,
        "num_calls": calls,
        "output_tokens": 4,
        "error": None,
    }


def test_compute_summary_overall_and_groups():
    rows = [
        row("college_economics", True),
        row("college_economics", False),
        row("college_programming", True, latency=3.0, calls=3),
    ]
    summary = compute_summary(rows, "test", {"model": {"name_or_path": "mock"}})
    assert summary["overall"]["n"] == 3
    assert summary["overall"]["accuracy"] == 2 / 3
    assert summary["overall"]["format_valid_rate"] == 1.0
    assert summary["overall"]["total_latency_sec"] == 5.0
    assert summary["overall"]["num_calls"] == 5
    assert summary["per_subject"]["college_economics"]["n"] == 2
    assert summary["per_subject"]["college_economics"]["accuracy"] == 0.5
    assert summary["per_subject"]["college_programming"]["accuracy"] == 1.0
