from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ceval_eval.data import file_sha256, read_jsonl, resolve_path, validate_subset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the frozen C-Eval subset.")
    parser.add_argument("--subset", default="data/subset.jsonl")
    parser.add_argument("--manifest", default="data/subset_manifest.json")
    args = parser.parse_args()

    records = read_jsonl(args.subset)
    stats = validate_subset(records)
    manifest_path = resolve_path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_sha = file_sha256(args.subset)
    if actual_sha != manifest["sha256"]:
        raise SystemExit("Subset SHA256 does not match manifest")
    if len(records) != manifest["total"]:
        raise SystemExit("Subset size does not match manifest")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"OK: {len(records)} questions, sha256={actual_sha[:16]}...")


if __name__ == "__main__":
    main()
