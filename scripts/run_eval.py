from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ceval_eval.config import load_config  # noqa: E402
from ceval_eval.evaluator import run_evaluation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one C-Eval experiment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Global question limit for smoke tests")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock backend")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite this shard's output")
    parser.add_argument("--model-path", default=None, help="Local model directory")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache directory")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--offline", action="store_true", help="Do not access Hugging Face")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.model_path:
        config.setdefault("model", {})["local_path"] = args.model_path
    if args.cache_dir:
        config.setdefault("model", {})["cache_dir"] = args.cache_dir
    if args.device:
        config.setdefault("model", {})["device"] = args.device
    if args.offline:
        config.setdefault("model", {})["local_files_only"] = True
    summary = run_evaluation(
        config,
        run_dir=args.run_dir,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        limit=args.limit,
        mock=args.mock or None,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
