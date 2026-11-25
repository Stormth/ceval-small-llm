from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record model files and checksums.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", default="data/model_manifest.json")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.exists():
        raise SystemExit(f"Model directory not found: {model_dir}")

    files = []
    for path in sorted(model_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {
            ".json",
            ".safetensors",
            ".bin",
            ".model",
            ".txt",
            ".py",
        }:
            files.append(
                {
                    "path": str(path.relative_to(model_dir)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest = {
        "model_dir": str(model_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(files)} file records to {output}")


if __name__ == "__main__":
    main()
