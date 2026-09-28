from pathlib import Path

from inference import LTX25Service, RuntimeSettings, output_dir_from_env


def test_frames_align_to_model_grid():
    assert LTX25Service._frames_for_seconds(3) == 73
    assert LTX25Service._frames_for_seconds(5) == 121


def test_runtime_settings_can_use_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("LTX_PERSISTENT_ROOT", str(tmp_path / "missing"))
    monkeypatch.setenv("LTX_FALLBACK_ROOT", str(tmp_path / "fallback"))
    monkeypatch.delenv("LTX_MODEL_DIR", raising=False)
    monkeypatch.delenv("LTX_OUTPUT_DIR", raising=False)
    settings = RuntimeSettings.from_env()
    assert settings.model_dir == Path(tmp_path / "fallback/models/ltx-2.5")
    assert settings.output_dir == Path(tmp_path / "fallback/outputs/ltx25")


def test_auto_offload_is_the_default(tmp_path, monkeypatch):
    monkeypatch.setenv("LTX_PERSISTENT_ROOT", str(tmp_path))
    monkeypatch.delenv("LTX_OFFLOAD", raising=False)
    assert RuntimeSettings.from_env().offload_mode == "auto"


def test_auto_offload_keeps_bf16_transformer_on_a100():
    assert LTX25Service._resolve_offload_mode("auto", 79.0) == "none"


def test_auto_offload_streams_on_48gb_gpu():
    assert LTX25Service._resolve_offload_mode("auto", 44.6) == "cpu"


def test_explicit_output_dir_wins(tmp_path, monkeypatch):
    output_dir = tmp_path / "custom-outputs"
    monkeypatch.setenv("LTX_OUTPUT_DIR", str(output_dir))
    assert output_dir_from_env() == output_dir
