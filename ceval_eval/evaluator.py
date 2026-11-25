from __future__ import annotations

import json
import random
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .config import get
from .data import (
    PROJECT_ROOT,
    append_jsonl,
    file_sha256,
    read_jsonl,
    resolve_path,
)
from .metrics import compute_summary, render_markdown
from .model_runner import build_runner
from .parsing import majority_vote, parse_answer
from .prompts import build_prompt, build_selfcheck_prompt


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _experiment_id(config: dict[str, Any]) -> str:
    return str(
        get(config, "experiment.id")
        or get(config, "output.experiment_name")
        or "experiment"
    )


def _load_fewshot(path: str | Path | None) -> dict[str, list[dict[str, Any]]]:
    if not path:
        return {}
    resolved = resolve_path(path)
    if not resolved.exists():
        return {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(resolved):
        grouped[row["subject"]].append(row)
    return dict(grouped)


def _make_prompt(
    record: dict[str, Any],
    config: dict[str, Any],
    examples: dict[str, list[dict[str, Any]]],
) -> str:
    template = str(get(config, "prompt.template", "baseline"))
    num_examples = int(get(config, "prompt.num_examples", 3))
    trimmed = {
        subject: rows[:num_examples] for subject, rows in examples.items()
    }
    return build_prompt(record, template, trimmed)


def _evaluate_one(
    record: dict[str, Any],
    config: dict[str, Any],
    runner: Any,
    examples: dict[str, list[dict[str, Any]]],
    *,
    shard_index: int,
    num_shards: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    prompt = _make_prompt(record, config, examples)
    system = str(get(config, "prompt.system", "") or "")
    gen_cfg = dict(config.get("generation", {}))
    method = str(get(config, "method.name", "single"))
    base = {
        "experiment_id": _experiment_id(config),
        "method": method,
        "prompt_template": str(get(config, "prompt.template", "baseline")),
        "sample_id": record["sample_id"],
        "subject": record["subject"],
        "subject_cn": record["subject_cn"],
        "category": record["category"],
        "source_id": record["source_id"],
        "split": record["split"],
        "question": record["question"],
        "options": record["options"],
        "reference": record["answer"],
        "prompt": prompt,
        "system": system,
        "shard_index": shard_index,
        "num_shards": num_shards,
    }

    extra: dict[str, Any] = {}
    if method == "single":
        outputs = runner.generate([prompt], system, gen_cfg, 1)[0]
        raw_outputs = [item["text"] for item in outputs]
        extracted = parse_answer(raw_outputs[0])
        extra["raw_outputs"] = raw_outputs
        extra["votes"] = {extracted: 1} if extracted else {}
    elif method == "self_consistency":
        num_samples = int(get(config, "method.num_samples", 3))
        sample_cfg = dict(gen_cfg)
        sample_cfg["do_sample"] = True
        sample_cfg["temperature"] = float(
            get(config, "method.temperature", gen_cfg.get("temperature", 0.7))
        )
        outputs = runner.generate([prompt], system, sample_cfg, num_samples)[0]
        raw_outputs = [item["text"] for item in outputs]
        answers = [parse_answer(text) for text in raw_outputs]
        extracted = majority_vote(answers)
        extra["raw_outputs"] = raw_outputs
        extra["sample_answers"] = answers
        extra["votes"] = dict(Counter(answer for answer in answers if answer))
    elif method == "self_check":
        first_outputs = runner.generate([prompt], system, gen_cfg, 1)[0]
        first_raw = first_outputs[0]["text"]
        first_extracted = parse_answer(first_raw)
        check_prompt = build_selfcheck_prompt(record, first_raw)
        check_outputs = runner.generate([check_prompt], system, gen_cfg, 1)[0]
        check_raw = check_outputs[0]["text"]
        extracted = parse_answer(check_raw)
        raw_outputs = [first_raw, check_raw]
        outputs = first_outputs + check_outputs
        extra.update(
            {
                "raw_outputs": raw_outputs,
                "first_output": first_raw,
                "first_extracted": first_extracted,
                "check_prompt": check_prompt,
                "check_output": check_raw,
                "votes": {extracted: 1} if extracted else {},
            }
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    output_tokens = sum(int(item["tokens"]) for item in outputs)
    truncated = any(bool(item["truncated"]) for item in outputs)
    latency = time.perf_counter() - started
    return {
        **base,
        **extra,
        "raw_output": raw_outputs[-1] if raw_outputs else "",
        "extracted": extracted,
        "correct": extracted == record["answer"],
        "format_valid": extracted is not None,
        "truncated": truncated,
        "latency_sec": latency,
        "num_calls": len(raw_outputs),
        "output_tokens": output_tokens,
        "error": None,
    }


def _write_meta(
    run_dir: Path,
    config: dict[str, Any],
    *,
    subset_sha256: str,
    num_shards: int,
    runner_meta: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    meta_path = run_dir / "run_meta.json"
    meta = {
        "experiment_id": _experiment_id(config),
        "config_path": get(config, "_config_path"),
        "config": config,
        "subset_path": str(get(config, "data.subset_path")),
        "subset_sha256": subset_sha256,
        "num_shards": num_shards,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "model": runner_meta,
        "packages": {
            name: _package_version(name)
            for name in ("torch", "transformers", "safetensors", "numpy")
        },
    }
    if meta_path.exists() and not overwrite:
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        if existing.get("subset_sha256") != subset_sha256:
            raise RuntimeError(
                "Existing run uses a different subset hash; use a new run dir "
                "or pass --overwrite."
            )
        if int(existing.get("num_shards", 0)) != num_shards:
            raise RuntimeError(
                "Existing run uses a different num_shards; use a new run dir "
                "or pass --overwrite."
            )
        return existing
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def run_evaluation(
    config: dict[str, Any],
    *,
    run_dir: str | Path | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
    limit: int | None = None,
    mock: bool | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    experiment_id = _experiment_id(config)
    if mock is not None:
        config.setdefault("runtime", {})["mock"] = bool(mock)
    shard_index = int(
        shard_index if shard_index is not None else get(config, "runtime.shard_index", 0)
    )
    num_shards = int(
        num_shards if num_shards is not None else get(config, "runtime.num_shards", 1)
    )
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")

    seed = int(config.get("seed", 42)) + shard_index
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    subset_path = resolve_path(get(config, "data.subset_path", "data/subset.jsonl"))
    records = read_jsonl(subset_path)
    if limit is not None:
        records = records[: int(limit)]
    records = records[shard_index::num_shards]
    if not records:
        raise RuntimeError("No records selected for this shard")

    output_dir = resolve_path(
        run_dir
        or get(config, "output.run_dir")
        or f"outputs/{experiment_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"per_question.shard-{shard_index:03d}.jsonl"
    if overwrite and result_path.exists():
        result_path.unlink()

    subset_sha = file_sha256(subset_path)
    runner = build_runner(config)
    _write_meta(
        output_dir,
        config,
        subset_sha256=subset_sha,
        num_shards=num_shards,
        runner_meta=runner.meta(),
        overwrite=overwrite,
    )

    completed: set[str] = set()
    if result_path.exists() and not overwrite:
        completed = {row["sample_id"] for row in read_jsonl(result_path)}
    examples = _load_fewshot(get(config, "data.fewshot_path"))
    pending = [row for row in records if row["sample_id"] not in completed]

    print(
        f"[{experiment_id}] shard {shard_index}/{num_shards}: "
        f"{len(records)} selected, {len(completed)} done, {len(pending)} pending"
    )
    log_every = int(get(config, "runtime.log_every", 10))
    for index, record in enumerate(
        tqdm(pending, desc=f"shard {shard_index}", unit="q"), start=1
    ):
        try:
            result = _evaluate_one(
                record,
                config,
                runner,
                examples,
                shard_index=shard_index,
                num_shards=num_shards,
            )
        except Exception as exc:  # keep a complete denominator on failures
            result = {
                "experiment_id": experiment_id,
                "method": str(get(config, "method.name", "single")),
                "prompt_template": str(get(config, "prompt.template", "baseline")),
                "sample_id": record["sample_id"],
                "subject": record["subject"],
                "subject_cn": record["subject_cn"],
                "category": record["category"],
                "source_id": record["source_id"],
                "split": record["split"],
                "question": record["question"],
                "options": record["options"],
                "reference": record["answer"],
                "prompt": "",
                "system": str(get(config, "prompt.system", "") or ""),
                "raw_output": "",
                "raw_outputs": [],
                "extracted": None,
                "correct": False,
                "format_valid": False,
                "truncated": False,
                "latency_sec": 0.0,
                "num_calls": 0,
                "output_tokens": 0,
                "error": repr(exc),
                "shard_index": shard_index,
                "num_shards": num_shards,
            }
        append_jsonl(result_path, result)
        if index % log_every == 0:
            print(f"[{experiment_id}] shard {shard_index}: {index}/{len(pending)}")

    all_results = read_jsonl(result_path)
    summary = compute_summary(all_results, experiment_id, config)
    (output_dir / f"summary.shard-{shard_index:03d}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / f"summary.shard-{shard_index:03d}.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )
    print(
        f"[{experiment_id}] shard {shard_index} complete: "
        f"acc={summary['overall']['accuracy']:.2%}, "
        f"valid={summary['overall']['format_valid_rate']:.2%}, "
        f"n={summary['overall']['n']}"
    )
    return summary
