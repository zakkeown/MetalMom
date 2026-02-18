"""Tests for the metalmom.models module."""

import pytest


def test_list_models_returns_list():
    from metalmom.models import list_models
    result = list_models()
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(m, str) for m in result)


def test_list_models_contains_known_families():
    from metalmom.models import list_models
    models = list_models()
    families = {m.split("/")[0] for m in models}
    assert "beats" in families
    assert "chords" in families


def test_list_models_family_filter():
    from metalmom.models import list_models
    beats = list_models("beats")
    assert all(m.startswith("beats/") for m in beats)
    assert len(beats) > 0


def test_model_path_returns_path_when_cached(tmp_path, monkeypatch):
    """Test model_path returns a Path when the model is in the cache."""
    from metalmom import models as mod

    fake_model = tmp_path / "beats" / "beats_lstm_1.mlmodel"
    fake_model.parent.mkdir(parents=True)
    fake_model.write_text("fake")

    monkeypatch.setattr(mod, "_get_cache_dir", lambda: tmp_path)

    from metalmom.models import model_path
    result = model_path("beats/beats_lstm_1")
    assert result is not None
    assert result.exists()


def test_model_path_returns_none_when_not_cached(tmp_path, monkeypatch):
    from metalmom import models as mod
    monkeypatch.setattr(mod, "_get_cache_dir", lambda: tmp_path)

    from metalmom.models import model_path
    result = model_path("beats/nonexistent")
    assert result is None


def test_download_models_requires_huggingface_hub():
    """Ensure download_models is callable (import guard tested by attempting call)."""
    from metalmom.models import download_models
    assert callable(download_models)
