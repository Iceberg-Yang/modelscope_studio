"""Detect the assigned L20N, build its SageAttention2 target, then start ABot."""

import json
import os
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from scripts.l20n_bootstrap import prepare_runtime


STATUS_PATH = Path(
    os.environ.get("ABOT_RUNTIME_STATUS", "/tmp/abot-l20n-runtime.json")
)


def _seed_reference_cache():
    """Copy bundled demo caches into persistent storage without overwriting it."""
    source = Path(__file__).resolve().parent / "outputs" / "ref_image_cache"
    output_dir = Path(
        os.environ.get(
            "ABOT_OUTPUT_DIR",
            "/mnt/workspace/abot-world/outputs",
        )
    )
    destination = output_dir / "ref_image_cache"
    if not source.is_dir():
        print(
            f"[BOOT][REF] bundled reference cache not found: {source}",
            flush=True,
        )
        return

    copied = 0
    existing = 0
    for source_file in source.rglob("*"):
        if not source_file.is_file():
            continue
        relative_path = source_file.relative_to(source)
        destination_file = destination / relative_path
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        if destination_file.exists():
            existing += 1
            continue
        shutil.copy2(source_file, destination_file)
        copied += 1
    print(
        f"[BOOT][REF] seeded={copied}, preserved={existing}, "
        f"destination={destination}",
        flush=True,
    )


class _PreflightHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?", 1)[0] != "/healthz":
            self.send_response(503)
            self.end_headers()
            return
        payload = {
            "ok": True,
            "model_ready": False,
            "service_phase": "l20n_preflight",
            "runtime_bootstrap": {"phase": "starting"},
        }
        if STATUS_PATH.exists():
            try:
                payload["runtime_bootstrap"] = json.loads(
                    STATUS_PATH.read_text(encoding="utf-8")
                )
            except Exception as exc:
                payload["runtime_bootstrap"] = {
                    "phase": "invalid_status",
                    "error": str(exc),
                }
        runtime = payload["runtime_bootstrap"]
        payload["gpu"] = runtime.get("hardware", {})
        payload["ok"] = runtime.get("phase") not in {
            "failed",
            "unsupported_cc",
            "sageattention_failed",
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):
        return


if __name__ == "__main__":
    _seed_reference_cache()
    server = ThreadingHTTPServer(
        ("0.0.0.0", int(os.environ.get("PORT", "7860"))),
        _PreflightHandler,
    )
    server_thread = threading.Thread(
        target=server.serve_forever,
        name="abot-l20n-preflight",
        daemon=True,
    )
    server_thread.start()
    print("[BOOT][L20N] preflight /healthz is listening.", flush=True)
    try:
        ready = prepare_runtime()
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
    runtime_status = {}
    if STATUS_PATH.exists():
        try:
            runtime_status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    phase = runtime_status.get("phase")
    unsupported_runtime = phase in {"failed", "unsupported_cc"}
    sage_required = os.environ.get("ABOT_REQUIRE_SAGEATTENTION", "1") == "1"
    if not ready and (unsupported_runtime or sage_required):
        raise SystemExit(
            f"L20N runtime preparation failed (phase={phase!r}, "
            f"ABOT_REQUIRE_SAGEATTENTION={int(sage_required)})"
        )
    os.execv(
        sys.executable,
        [sys.executable, "-u", os.path.join(os.path.dirname(__file__), "app.py")],
    )
