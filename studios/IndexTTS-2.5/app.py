"""ModelScope Studio entrypoint for the official IndexTTS-2.5 WebUI."""

import os
import sys
from pathlib import Path


def _default_model_dir() -> str:
    """Prefer ModelScope's persistent volume and stay runnable elsewhere."""
    persistent_root = Path("/mnt/workspace")
    if persistent_root.is_dir() and os.access(persistent_root, os.W_OK):
        return str(persistent_root / "IndexTTS-2.5")
    return str(Path(__file__).resolve().parent / "checkpoints")


model_dir = os.environ.get("INDEXTTS_MODEL_DIR", _default_model_dir())

# webui.py owns the argument parser and constructs the Gradio application.
# BF16 is exposed as --fp16 for backward-compatible CLI naming.
sys.argv = [
    sys.argv[0],
    "--version",
    "2.5",
    "--model_dir",
    model_dir,
    "--host",
    "0.0.0.0",
    "--port",
    "7860",
    "--fp16",
]

from webui import demo  # noqa: E402


if __name__ == "__main__":
    # A single model instance is shared by all requests. Serial inference keeps
    # GPU memory usage predictable while still allowing users to queue jobs.
    demo.queue(default_concurrency_limit=1, max_size=20)
    demo.launch(server_name="0.0.0.0", server_port=7860)
