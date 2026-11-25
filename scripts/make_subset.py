from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ceval_eval.config import load_config  # noqa: E402
from ceval_eval.data import (  # noqa: E402
    fetch_dataset_revision,
    fetch_split_rows,
    file_sha256,
    normalize_row,
    resolve_path,
    validate_subset,
    write_jsonl,
)
from ceval_eval.subjects import SUBJECT_META  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the fixed C-Eval subset.")
    parser.add_argument("--config", default="configs/subset.yaml")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_subset = resolve_path(config["output"]["subset"])
    output_manifest = resolve_path(config["output"]["manifest"])
    output_fewshot = resolve_path(config["output"]["fewshot"])

    if output_subset.exists() and not args.force:
        raise SystemExit(
            f"{output_subset} already exists; pass --force to rebuild the fixed subset."
        )

    seed = int(config["seed"])
    per_subject = int(config["per_subject"])
    split = str(config.get("split", "val"))
    dataset = str(config.get("dataset", "ceval/ceval-exam"))
    subject_rows: dict[str, int] = {}
    subset: list[dict] = []

    for subject in config["subjects"]:
        if subject not in SUBJECT_META:
            raise SystemExit(f"Unknown subject metadata: {subject}")
        print(f"Fetching {dataset} / {subject} / {split} ...")
        rows = fetch_split_rows(subject, split, dataset=dataset)
        subject_rows[subject] = len(rows)
        if len(rows) < per_subject:
            raise SystemExit(
                f"{subject}: only {len(rows)} rows in {split}, need {per_subject}"
            )
        rng = random.Random(f"{seed}:{subject}:{split}")
        chosen_indices = sorted(rng.sample(range(len(rows)), per_subject))
        for index in chosen_indices:
            subset.append(normalize_row(subject, split, rows[index]))
        time.sleep(0.2)

    subset.sort(key=lambda row: (row["subject"], row["source_id"]))
    stats = validate_subset(
        subset,
        min_total=int(config.get("min_total", 200)),
        min_subjects=int(config.get("min_subjects", 5)),
        min_per_subject=int(config.get("min_per_subject", 20)),
    )

    fewshot: list[dict] = []
    fewshot_cfg = config.get("fewshot", {})
    fewshot_split = str(fewshot_cfg.get("split", "dev"))
    fewshot_per_subject = int(fewshot_cfg.get("per_subject", 3))
    for subject in config["subjects"]:
        rows = fetch_split_rows(subject, fewshot_split, dataset=dataset)
        selected = sorted(rows, key=lambda row: int(row["id"]))[:fewshot_per_subject]
        if len(selected) < fewshot_per_subject:
            print(
                f"Warning: {subject} has only {len(selected)} {fewshot_split} examples",
                file=sys.stderr,
            )
        for row in selected:
            record = normalize_row(subject, fewshot_split, row)
            fewshot.append(record)
        time.sleep(0.2)

    subset_ids = {row["sample_id"] for row in subset}
    overlap = subset_ids & {row["sample_id"] for row in fewshot}
    if overlap:
        raise SystemExit(f"Few-shot examples overlap evaluation subset: {sorted(overlap)}")
    if not fewshot:
        raise SystemExit("No few-shot examples were collected")

    write_jsonl(output_subset, subset)
    write_jsonl(output_fewshot, fewshot)

    subject_manifest = {}
    for subject in config["subjects"]:
        count = sum(1 for row in subset if row["subject"] == subject)
        subject_manifest[subject] = {
            "name_cn": SUBJECT_META[subject]["name_cn"],
            "category": SUBJECT_META[subject]["category"],
            "available_in_val": subject_rows[subject],
            "selected": count,
        }

    dataset_revision = fetch_dataset_revision(dataset)
    manifest = {
        "dataset": dataset,
        "dataset_revision": dataset_revision["revision"],
        "dataset_last_modified": dataset_revision["last_modified"],
        "split": split,
        "source": "Hugging Face datasets-server rows API",
        "source_url": "https://huggingface.co/datasets/ceval/ceval-exam",
        "selection_rule": (
            "Per-subject uniform random sample without replacement; "
            "each subject has an independent RNG seeded by "
            "'{seed}:{subject}:{split}'. The subset is frozen before any model run."
        ),
        "seed": seed,
        "per_subject": per_subject,
        "total": len(subset),
        "num_subjects": stats["num_subjects"],
        "subjects": subject_manifest,
        "fewshot": {
            "split": fewshot_split,
            "per_subject": fewshot_per_subject,
            "total": len(fewshot),
            "path": str(output_fewshot.relative_to(ROOT)),
        },
        "sha256": file_sha256(output_subset),
        "fewshot_sha256": file_sha256(output_fewshot),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote {len(subset)} questions to {output_subset}")
    print(f"Wrote {len(fewshot)} few-shot examples to {output_fewshot}")
    print(f"Wrote manifest to {output_manifest}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
