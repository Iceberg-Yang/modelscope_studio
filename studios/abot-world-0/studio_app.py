"""ModelScope Docker Studio entry point for the A10 deployment profile."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ID = "amap_cvlab/ABot-World-0-5B-LF"
MODEL_DIR_NAME = "ABot-World-0-5B-LF"


def _checkpoint_is_complete(path: Path) -> bool:
    required = (
        "config.json",
        "diffusion_pytorch_model.safetensors",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "taew2_2.pth",
        "Wan2.2_VAE.pth",
        "google/umt5-xxl/tokenizer_config.json",
    )
    return all((path / relative).is_file() for relative in required)


def _prepare_checkpoints() -> Path:
    persistent_root = Path(
        os.environ.get("ABOT_CHECKPOINT_ROOT", "/mnt/workspace/checkpoints")
    )
    target = persistent_root / MODEL_DIR_NAME
    target.parent.mkdir(parents=True, exist_ok=True)

    if not _checkpoint_is_complete(target):
        if os.environ.get("ABOT_AUTO_DOWNLOAD", "1").strip().lower() not in {
            "1",
            "true",
            "yes",
        }:
            raise FileNotFoundError(
                f"Model checkpoint is incomplete at {target}; enable ABOT_AUTO_DOWNLOAD."
            )
        print(f"[STUDIO] Downloading {MODEL_ID} to {target} ...", flush=True)
        from modelscope import snapshot_download

        snapshot_download(MODEL_ID, local_dir=str(target))

    if not _checkpoint_is_complete(target):
        raise RuntimeError(f"Checkpoint download finished but required files are missing: {target}")

    project_checkpoints = PROJECT_ROOT / "checkpoints"
    project_checkpoints.mkdir(parents=True, exist_ok=True)
    link = project_checkpoints / MODEL_DIR_NAME
    if link.is_symlink() and link.resolve() == target.resolve():
        return target
    if link.exists() or link.is_symlink():
        raise RuntimeError(
            f"Refusing to replace existing checkpoint path {link}; expected a symlink to {target}."
        )
    link.symlink_to(target, target_is_directory=True)
    return target


def main() -> None:
    os.chdir(PROJECT_ROOT)
    os.environ.setdefault("PORT", "7860")
    os.environ.setdefault("GRADIO_SHARE", "false")
    os.environ.setdefault("ABOT_STREAM_HEIGHT", "480")
    os.environ.setdefault("ABOT_STREAM_WIDTH", "832")
    os.environ.setdefault("ABOT_UI_FRAME_STRIDE", "3")
    os.environ.setdefault("ABOT_OUTPUT_DIR", "/mnt/workspace/outputs")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

    Path(os.environ["ABOT_OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    checkpoint_path = _prepare_checkpoints()
    print(f"[STUDIO] Checkpoints ready: {checkpoint_path}", flush=True)
    runpy.run_path(str(PROJECT_ROOT / "web_client" / "app.py"), run_name="__main__")


if __name__ == "__main__":
    main()
