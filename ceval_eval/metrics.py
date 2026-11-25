from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .data import resolve_path


def _rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(bool(row.get("correct")) for row in rows)
    valid = sum(bool(row.get("format_valid")) for row in rows)
    truncated = sum(bool(row.get("truncated")) for row in rows)
    errors = sum(bool(row.get("error")) for row in rows)
    latency = sum(float(row.get("latency_sec", 0.0)) for row in rows)
    calls = sum(int(row.get("num_calls", 1)) for row in rows)
    output_tokens = sum(int(row.get("output_tokens", 0)) for row in rows)
    return {
        "n": total,
        "correct": correct,
        "accuracy": _rate(correct, total),
        "format_valid": valid,
        "format_valid_rate": _rate(valid, total),
        "truncated": truncated,
        "truncation_rate": _rate(truncated, total),
        "errors": errors,
        "total_latency_sec": latency,
        "avg_latency_sec": _rate(latency, total),
        "num_calls": calls,
        "avg_calls_per_question": _rate(calls, total),
        "output_tokens": output_tokens,
        "avg_output_tokens": _rate(output_tokens, total),
    }


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(key, "unknown"))].append(row)
    return {name: _stats(group) for name, group in sorted(buckets.items())}


def compute_summary(
    results: list[dict[str, Any]],
    experiment_id: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not results:
        raise ValueError("Cannot summarize an empty result set")
    return {
        "experiment_id": experiment_id,
        "model": (config or {}).get("model", {}).get("name_or_path"),
        "overall": _stats(results),
        "per_subject": _group(results, "subject"),
        "per_category": _group(results, "category"),
        "error_types": dict(
            Counter(
                str(row.get("error"))
                for row in results
                if row.get("error")
            )
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    lines = [
        f"# {summary['experiment_id']}",
        "",
        f"- Model: `{summary.get('model')}`",
        f"- Questions: {overall['n']}",
        f"- Accuracy: **{overall['accuracy']:.2%}**",
        f"- Format valid rate: **{overall['format_valid_rate']:.2%}**",
        f"- Truncation rate: {overall['truncation_rate']:.2%}",
        f"- Total latency: {overall['total_latency_sec']:.2f} s",
        f"- Average latency: {overall['avg_latency_sec']:.3f} s/question",
        f"- Model calls: {overall['num_calls']}",
        f"- Average output tokens: {overall['avg_output_tokens']:.1f}",
        "",
        "## Per subject",
        "",
        "| Subject | N | Accuracy | Format valid | Truncated | Avg latency (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for subject, stats in summary["per_subject"].items():
        lines.append(
            f"| {subject} | {stats['n']} | {stats['accuracy']:.2%} | "
            f"{stats['format_valid_rate']:.2%} | {stats['truncation_rate']:.2%} | "
            f"{stats['avg_latency_sec']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Per category",
            "",
            "| Category | N | Accuracy | Format valid |",
            "|---|---:|---:|---:|",
        ]
    )
    for category, stats in summary["per_category"].items():
        lines.append(
            f"| {category} | {stats['n']} | {stats['accuracy']:.2%} | "
            f"{stats['format_valid_rate']:.2%} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_summary(
    run_dir: str | Path,
    results: Iterable[dict[str, Any]],
    experiment_id: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = list(results)
    summary = compute_summary(rows, experiment_id, config)
    target_dir = resolve_path(run_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    with (target_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    (target_dir / "summary.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )
    return summary
