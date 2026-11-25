from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ceval_eval.config import load_config  # noqa: E402
from ceval_eval.data import read_jsonl, resolve_path  # noqa: E402
from ceval_eval.metrics import write_summary  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge data-parallel shards.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    run_dir = resolve_path(args.run_dir)
    shard_files = sorted(run_dir.glob("per_question.shard-*.jsonl"))
    if not shard_files:
        raise SystemExit(f"No shard files found in {run_dir}")

    merged: dict[str, dict] = {}
    for shard_file in shard_files:
        for row in read_jsonl(shard_file):
            merged[row["sample_id"]] = row
    results = sorted(
        merged.values(),
        key=lambda row: (row["subject"], int(row["source_id"])),
    )

    meta_path = run_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    experiment_id = (
        args.experiment_id
        or meta.get("experiment_id")
        or "experiment"
    )
    config = load_config(args.config) if args.config else meta.get("config")

    output_jsonl = run_dir / "per_question.jsonl"
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = write_summary(run_dir, results, experiment_id, config)
    print(
        f"Merged {len(shard_files)} shards -> {len(results)} questions; "
        f"accuracy={summary['overall']['accuracy']:.2%}; "
        f"format_valid={summary['overall']['format_valid_rate']:.2%}"
    )
    print(f"Wrote {output_jsonl}")
    print(f"Wrote {run_dir / 'summary.json'}")
    print(f"Wrote {run_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
