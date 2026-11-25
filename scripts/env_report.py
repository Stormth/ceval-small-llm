from __future__ import annotations

import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

import torch


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main() -> None:
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpus": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
