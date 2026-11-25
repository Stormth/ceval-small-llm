from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .subjects import subject_category, subject_name_cn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET_NAME = "ceval/ceval-exam"


def resolve_path(path: str | Path) -> Path:
    """Resolve a project-relative path against the repository root."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _request_json(url: str, retries: int = 3, timeout: int = 60) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "ceval-small-llm/0.1 (research evaluation)"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def fetch_dataset_revision(dataset: str = DATASET_NAME) -> dict[str, Any]:
    """Return the Hugging Face dataset repo revision for provenance."""
    payload = _request_json(f"https://huggingface.co/api/datasets/{dataset}")
    return {
        "revision": payload.get("sha"),
        "last_modified": payload.get("lastModified"),
    }


def fetch_split_rows(
    subject: str,
    split: str,
    dataset: str = DATASET_NAME,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    """Fetch every row of one C-Eval subject/split via the HF rows API."""
    rows: list[dict[str, Any]] = []
    offset = 0
    total: int | None = None
    while total is None or offset < total:
        query = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": subject,
                "split": split,
                "offset": offset,
                "length": page_size,
            }
        )
        payload = _request_json(f"{HF_ROWS_URL}?{query}")
        page = payload.get("rows", [])
        if not page:
            break
        rows.extend(item["row"] for item in page)
        total = int(payload.get("num_rows_total", len(rows)))
        offset += len(page)
    return rows


def normalize_row(subject: str, split: str, row: dict[str, Any]) -> dict[str, Any]:
    answer = str(row.get("answer", "")).strip().upper()
    if answer not in {"A", "B", "C", "D"}:
        raise ValueError(f"Invalid answer {answer!r} for {subject} id={row.get('id')}")
    options = {
        letter: str(row.get(letter, "")).strip()
        for letter in ("A", "B", "C", "D")
    }
    if any(not value for value in options.values()):
        raise ValueError(f"Missing option for {subject} id={row.get('id')}")
    source_id = int(row["id"])
    return {
        "sample_id": f"{subject}:{split}:{source_id}",
        "subject": subject,
        "subject_cn": subject_name_cn(subject),
        "category": subject_category(subject),
        "split": split,
        "source_id": source_id,
        "question": str(row["question"]).strip(),
        "options": options,
        "answer": answer,
        "explanation": str(row.get("explanation", "")).strip(),
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = resolve_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with resolve_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_subset(
    records: list[dict[str, Any]],
    min_total: int = 200,
    min_subjects: int = 5,
    min_per_subject: int = 20,
) -> dict[str, Any]:
    if not records:
        raise ValueError("Subset is empty")
    subject_counts = Counter(record["subject"] for record in records)
    unique_ids = {record["sample_id"] for record in records}
    if len(unique_ids) != len(records):
        raise ValueError("Subset contains duplicate sample_id values")
    if len(records) < min_total:
        raise ValueError(f"Subset has {len(records)} questions, need >= {min_total}")
    if len(subject_counts) < min_subjects:
        raise ValueError(
            f"Subset covers {len(subject_counts)} subjects, need >= {min_subjects}"
        )
    too_small = {
        subject: count
        for subject, count in subject_counts.items()
        if count < min_per_subject
    }
    if too_small:
        raise ValueError(f"Subjects below {min_per_subject} questions: {too_small}")
    for record in records:
        if record["answer"] not in {"A", "B", "C", "D"}:
            raise ValueError(f"Invalid answer in {record['sample_id']}")
    return {
        "total": len(records),
        "num_subjects": len(subject_counts),
        "subject_counts": dict(sorted(subject_counts.items())),
    }
