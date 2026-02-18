"""Model download and management for MetalMom CoreML models.

Models are hosted on Hugging Face Hub at zkeown/metalmom-coreml-models.
Install the optional dependency: pip install metalmom[models]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

REPO_ID = "zkeown/metalmom-coreml-models"

# Canonical model list (matches models/converted/ structure)
_MODEL_FAMILIES = {
    "beats": [
        "beats_lstm_1", "beats_lstm_2", "beats_lstm_3", "beats_lstm_4",
        "beats_lstm_5", "beats_lstm_6", "beats_lstm_7", "beats_lstm_8",
        "beats_blstm_1", "beats_blstm_2", "beats_blstm_3", "beats_blstm_4",
        "beats_blstm_5", "beats_blstm_6", "beats_blstm_7", "beats_blstm_8",
    ],
    "chords": ["chords_dnn"],
    "chroma": ["chroma_dnn"],
    "downbeats": [
        "downbeats_blstm_1", "downbeats_blstm_2", "downbeats_blstm_3", "downbeats_blstm_4",
        "downbeats_blstm_5", "downbeats_blstm_6", "downbeats_blstm_7", "downbeats_blstm_8",
        "downbeats_bgru_harmonic_0", "downbeats_bgru_harmonic_1", "downbeats_bgru_harmonic_2",
        "downbeats_bgru_harmonic_3", "downbeats_bgru_harmonic_4", "downbeats_bgru_harmonic_5",
        "downbeats_bgru_rhythmic_0", "downbeats_bgru_rhythmic_1", "downbeats_bgru_rhythmic_2",
        "downbeats_bgru_rhythmic_3", "downbeats_bgru_rhythmic_4", "downbeats_bgru_rhythmic_5",
    ],
    "key": ["key_cnn"],
    "notes": ["notes_brnn"],
    "onsets": [
        "onsets_rnn_1", "onsets_rnn_2", "onsets_rnn_3", "onsets_rnn_4",
        "onsets_rnn_5", "onsets_rnn_6", "onsets_rnn_7", "onsets_rnn_8",
        "onsets_brnn_1", "onsets_brnn_2", "onsets_brnn_3", "onsets_brnn_4",
        "onsets_brnn_5", "onsets_brnn_6", "onsets_brnn_7", "onsets_brnn_8",
        "onsets_brnn_pp_1", "onsets_brnn_pp_2", "onsets_brnn_pp_3", "onsets_brnn_pp_4",
        "onsets_brnn_pp_5", "onsets_brnn_pp_6", "onsets_brnn_pp_7", "onsets_brnn_pp_8",
        "onsets_cnn",
    ],
}


def _get_cache_dir() -> Path:
    """Return the local model cache directory."""
    return Path.home() / ".cache" / "metalmom" / "models"


def list_models(family: Optional[str] = None) -> list[str]:
    """List available model identifiers.

    Parameters
    ----------
    family : str, optional
        Filter to a specific family (e.g. "beats", "key").
        If None, returns all models.

    Returns
    -------
    list of str
        Model identifiers in "family/name" format.
    """
    cache_dir = _get_cache_dir()
    config_path = cache_dir / "config.json"

    if config_path.exists():
        config = json.loads(config_path.read_text())
        models = sorted(config.get("models", {}).keys())
    else:
        models = []
        if cache_dir.exists():
            for family_dir in sorted(cache_dir.iterdir()):
                if family_dir.is_dir() and not family_dir.name.startswith("."):
                    for f in sorted(family_dir.glob("*.mlmodel")):
                        models.append(f"{family_dir.name}/{f.stem}")

        if not models:
            for fam, names in sorted(_MODEL_FAMILIES.items()):
                for name in names:
                    models.append(f"{fam}/{name}")

    if family:
        models = [m for m in models if m.startswith(f"{family}/")]

    return models


def model_path(model_id: str) -> Optional[Path]:
    """Get the local path for a cached model.

    Parameters
    ----------
    model_id : str
        Model identifier in "family/name" format (e.g. "beats/beats_lstm_1").

    Returns
    -------
    Path or None
        Path to the .mlmodel file if cached, None otherwise.
    """
    cache_dir = _get_cache_dir()
    path = cache_dir / f"{model_id}.mlmodel"
    return path if path.exists() else None


def download_models(family: Optional[str] = None, cache_dir: Optional[Path] = None) -> Path:
    """Download models from Hugging Face Hub.

    Parameters
    ----------
    family : str, optional
        Download only a specific family (e.g. "beats").
        If None, downloads all models.
    cache_dir : Path, optional
        Override the default cache directory (~/.cache/metalmom/models/).

    Returns
    -------
    Path
        Path to the local cache directory containing downloaded models.

    Raises
    ------
    ImportError
        If huggingface_hub is not installed.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for model downloads. "
            "Install with: pip install metalmom[models]"
        )

    dest = cache_dir or _get_cache_dir()

    if family:
        allow_patterns = [f"{family}/*.mlmodel", "config.json"]
    else:
        allow_patterns = ["*.mlmodel", "config.json"]

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(dest),
        allow_patterns=allow_patterns,
        ignore_patterns=["*.mlmodelc/*"],
    )

    return dest
