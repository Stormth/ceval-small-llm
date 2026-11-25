from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ceval_eval.data import read_jsonl, resolve_path  # noqa: E402


def load_results(run_dir: str) -> dict[str, dict]:
    path = resolve_path(run_dir) / "per_question.jsonl"
    if not path.exists():
        raise SystemExit(f"Missing {path}; run merge_shards.py first")
    return {row["sample_id"]: row for row in read_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and candidate runs.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    baseline = load_results(args.baseline)
    candidate = load_results(args.candidate)
    shared = sorted(set(baseline) & set(candidate))
    if not shared:
        raise SystemExit("No overlapping sample_id values")

    fixed = [sid for sid in shared if not baseline[sid]["correct"] and candidate[sid]["correct"]]
    broken = [sid for sid in shared if baseline[sid]["correct"] and not candidate[sid]["correct"]]
    both_wrong = [
        sid for sid in shared if not baseline[sid]["correct"] and not candidate[sid]["correct"]
    ]

    output_path = resolve_path(
        args.output or f"{args.candidate}/comparison_vs_baseline.md"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Baseline vs candidate",
        "",
        f"- baseline: `{args.baseline}`",
        f"- candidate: `{args.candidate}`",
        f"- shared questions: {len(shared)}",
        f"- fixed: {len(fixed)}",
        f"- broken: {len(broken)}",
        f"- both wrong: {len(both_wrong)}",
        "",
        "## Fixed cases",
        "",
        "| sample_id | subject | reference | baseline | candidate |",
        "|---|---|---|---|---|",
    ]
    for sid in fixed:
        lines.append(
            f"| {sid} | {baseline[sid]['subject']} | {baseline[sid]['reference']} | "
            f"{baseline[sid]['extracted']} | {candidate[sid]['extracted']} |"
        )
    lines.extend(
        [
            "",
            "## Broken cases",
            "",
            "| sample_id | subject | reference | baseline | candidate |",
            "|---|---|---|---|---|",
        ]
    )
    for sid in broken:
        lines.append(
            f"| {sid} | {baseline[sid]['subject']} | {baseline[sid]['reference']} | "
            f"{baseline[sid]['extracted']} | {candidate[sid]['extracted']} |"
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "shared": len(shared),
                "fixed": len(fixed),
                "broken": len(broken),
                "both_wrong": len(both_wrong),
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
