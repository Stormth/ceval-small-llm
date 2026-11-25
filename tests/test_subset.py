import pytest

from ceval_eval.data import validate_subset


def make_records():
    records = []
    for subject_index in range(8):
        for item_index in range(25):
            records.append(
                {
                    "sample_id": f"subject{subject_index}:val:{item_index}",
                    "subject": f"subject{subject_index}",
                    "answer": "ABCD"[item_index % 4],
                }
            )
    return records


def test_validate_subset_accepts_minimum_shape():
    stats = validate_subset(make_records())
    assert stats["total"] == 200
    assert stats["num_subjects"] == 8


def test_validate_subset_rejects_too_few_questions():
    with pytest.raises(ValueError, match="need >="):
        validate_subset(make_records()[:199])


def test_validate_subset_rejects_too_few_per_subject():
    records = make_records()
    filtered = [row for row in records if row["subject"] != "subject0"]
    filtered += [
        {
            "sample_id": f"subject0:val:{index}",
            "subject": "subject0",
            "answer": "A",
        }
        for index in range(19)
    ]
    filtered += [
        {
            "sample_id": f"subject1:val:extra{index}",
            "subject": "subject1",
            "answer": "A",
        }
        for index in range(6)
    ]
    with pytest.raises(ValueError, match="below"):
        validate_subset(filtered)
