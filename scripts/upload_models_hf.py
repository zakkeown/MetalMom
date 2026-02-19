#!/usr/bin/env python3
"""Upload CoreML models to Hugging Face Hub.

Usage:
    python scripts/upload_models_hf.py [--repo-id zkeown/metalmom-coreml-models] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)


MODELS_DIR = Path(__file__).parent.parent / "models" / "converted"

MODEL_CARD = """\
---
license: cc-by-nc-sa-4.0
tags:
  - audio
  - music-information-retrieval
  - coreml
  - metalmom
---

# MetalMom CoreML Models

67 CoreML models converted from [madmom](https://github.com/CPJKU/madmom) for use with [MetalMom](https://github.com/zakkeown/MetalMom).

## Model Families

| Family | Count | Description |
|--------|-------|-------------|
| beats | 16 | RNN beat tracking (LSTM + BLSTM) |
| chords | 1 | Deep chroma chord recognition |
| chroma | 1 | DNN chroma extraction |
| downbeats | 16 | RNN downbeat tracking (BLSTM + BGRU) |
| key | 1 | CNN key recognition |
| notes | 14 | RNN note/onset detection |
| onsets | 16 | RNN onset detection |

## Usage

```python
from metalmom.models import download_models, list_models

# List available models
list_models()

# Download all models
download_models()

# Download a specific family
download_models("beats")
```

## Provenance

Converted from madmom's pickled NumPy weights using `coremltools` NeuralNetworkBuilder.
Peephole LSTM connections are preserved (not available via PyTorch conversion path).

## License

CC-BY-NC-SA 4.0 (same as original madmom model weights).
"""


def build_config(models_dir: Path) -> dict:
    """Build a config.json mapping model paths to metadata."""
    config = {"models": {}}
    for family_dir in sorted(models_dir.iterdir()):
        if not family_dir.is_dir() or family_dir.name.startswith("."):
            continue
        for model_file in sorted(family_dir.glob("*.mlmodel")):
            key = f"{family_dir.name}/{model_file.stem}"
            config["models"][key] = {
                "file": f"{family_dir.name}/{model_file.name}",
                "family": family_dir.name,
                "size_bytes": model_file.stat().st_size,
            }
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload CoreML models to HF Hub")
    parser.add_argument(
        "--repo-id", default="zkeown/metalmom-coreml-models",
        help="Hugging Face repo ID",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not MODELS_DIR.exists():
        print(f"ERROR: Models directory not found: {MODELS_DIR}")
        return 1

    model_files = list(MODELS_DIR.rglob("*.mlmodel"))
    print(f"Found {len(model_files)} models in {MODELS_DIR}")

    config = build_config(MODELS_DIR)
    print(f"Config has {len(config['models'])} entries")

    if args.dry_run:
        print(f"\nDRY RUN -- would upload to {args.repo_id}:")
        for key in config["models"]:
            print(f"  {key}")
        return 0

    api = HfApi()

    # Create repo if it doesn't exist
    create_repo(args.repo_id, repo_type="model", exist_ok=True)

    # Upload README
    api.upload_file(
        path_or_fileobj=MODEL_CARD.encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
    )

    # Upload config.json
    config_bytes = json.dumps(config, indent=2).encode()
    api.upload_file(
        path_or_fileobj=config_bytes,
        path_in_repo="config.json",
        repo_id=args.repo_id,
    )

    # Upload model files
    api.upload_folder(
        folder_path=str(MODELS_DIR),
        repo_id=args.repo_id,
        path_in_repo=".",
        ignore_patterns=["*.mlmodelc/*", ".gitignore"],
    )

    print(f"\nDone! Models uploaded to https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
