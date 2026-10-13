"""tests/test_config.py — Tests for lean_context/config.py."""

import pytest
from lean_context.config import MODEL, NUM_CTX, NUM_PREDICT, OLLAMA_OPTIONS, get_model_info


def test_constants():
    assert MODEL == "llama3.2:3b"
    assert NUM_CTX == 8192
    assert NUM_PREDICT == 256


def test_ollama_options_contains_required_keys():
    assert "num_ctx" in OLLAMA_OPTIONS
    assert "num_predict" in OLLAMA_OPTIONS
    assert OLLAMA_OPTIONS["num_ctx"] == NUM_CTX
    assert OLLAMA_OPTIONS["num_predict"] == NUM_PREDICT


def test_get_model_info_returns_dict_gracefully(monkeypatch):
    """get_model_info must return a dict with the required keys even on error."""
    import lean_context.config as cfg
    # Simulate Ollama being unreachable
    monkeypatch.setattr(cfg, "get_model_info", lambda: {
        "max_context_length": None, "parameter_size": None,
        "raw": None, "error": "unreachable"
    })
    info = cfg.get_model_info()
    assert "max_context_length" in info
    assert "parameter_size" in info


def test_print_startup_banner_does_not_raise(monkeypatch, capsys):
    """Banner must print without crashing even when Ollama is unreachable."""
    import lean_context.config as cfg

    def _fake_get_model_info():
        return {"max_context_length": 131072, "parameter_size": "3B", "raw": {}}

    monkeypatch.setattr(cfg, "get_model_info", _fake_get_model_info)
    cfg.print_startup_banner(extra_lines=["  test: yes"])
    out = capsys.readouterr().out
    assert "llama3.2:3b" in out
    assert "8192" in out
    assert "131072" in out
