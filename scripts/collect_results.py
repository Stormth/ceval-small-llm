from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ceval_eval.data import resolve_path  # noqa: E402


def main() -> None:
    outputs_dir = resolve_path("outputs")
    summaries = sorted(outputs_dir.glob("*/summary.json"))
    if not summaries:
        raise SystemExit("No summary.json files found under outputs/")

    rows = []
    for path in summaries:
        data = json.loads(path.read_text(encoding="utf-8"))
        overall = data["overall"]
        rows.append(
            {
                "experiment": data["experiment_id"],
                "model": data.get("model") or "",
                "n": overall["n"],
                "accuracy": f"{overall['accuracy']:.4f}",
                "format_valid_rate": f"{overall['format_valid_rate']:.4f}",
                "truncation_rate": f"{overall['truncation_rate']:.4f}",
                "avg_latency_sec": f"{overall['avg_latency_sec']:.3f}",
                "avg_calls": f"{overall['avg_calls_per_question']:.2f}",
                "avg_output_tokens": f"{overall['avg_output_tokens']:.1f}",
                "summary_path": str(path.relative_to(ROOT)),
            }
        )

    columns = list(rows[0].keys())
    csv_path = outputs_dir / "comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Experiment comparison",
        "",
        "| " + " | ".join(columns[:-1]) + " |",
        "|" + "|".join(["---"] * (len(columns) - 1)) + "|",
    ]
    for row in rows:
        md_lines.append(
            "| " + " | ".join(str(row[column]) for column in columns[:-1]) + " |"
        )
    md_lines.append("")
    md_path = outputs_dir / "comparison.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
