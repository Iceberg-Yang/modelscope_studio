"""
ABot-World ModelScope Studio 入口
混合架构：gr.Blocks UI + iframe/WebSocket 二进制帧流
  - mount_gradio_app 将 gr.Blocks 挂载到 FastAPI（HTTP 路由可用）
  - GET /stream_page 返回完整 HTML 页面（iframe 加载，<script> 正常执行）
  - WS /stream_ws 推送二进制 JPEG 帧并接收键盘控制
  - GET /healthz 和 /metrics 提供运行诊断
"""
import sys
import os
import time
import threading
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# 确保日志立即输出
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

print(
    "[BOOT] Root app.py starting "
    f"(python={sys.executable}, Gradio + binary WebSocket)...",
    flush=True,
)

try:
    from web_client.app import (
        demo, _KEY_HANDLER_JS, _COMBINED_CSS,
        SERVER_NAME, SERVER_PORT, SHUTDOWN_GRACE_SECONDS,
    )
    from web_client.state import state
    print("[BOOT] Import successful.", flush=True)
except Exception as e:
    print(f"[FATAL] Import failed: {e}", flush=True)
    traceback.print_exc()
    # 创建一个最小 demo，让 SDK 不至于完全崩溃，同时显示错误信息
    import gradio as gr
    demo = gr.Blocks(title="Import Error")
    with demo:
        gr.Markdown(f"# ❌ Import Error\n\n```\n{traceback.format_exc()}\n```")
    _KEY_HANDLER_JS = ""
    _COMBINED_CSS = ""
    SERVER_NAME = os.environ.get("SERVER_NAME", "0.0.0.0")
    SERVER_PORT = int(os.environ.get("PORT", os.environ.get("SERVER_PORT", "7860")))
    SHUTDOWN_GRACE_SECONDS = 5
    state = type("State", (), {"is_running": False})()

# ── 创建 FastAPI app + WebSocket 路由 + mount Gradio ───────────────────────
from fastapi import FastAPI
import gradio as gr
import uvicorn

fastapi_app = FastAPI()

# 在 mount_gradio_app 之前注册，确保实时路由优先匹配。
try:
    from web_client import ws_stream
    ws_stream.register_routes(fastapi_app)
    print("[BOOT] WebSocket + diagnostic routes registered.", flush=True)
except Exception as e:
    print(f"[WARN] Route registration failed: {e}", flush=True)
    traceback.print_exc()

# 挂载 Gradio Blocks demo 到 FastAPI（path="/" 作为根路由）
app = gr.mount_gradio_app(
    fastapi_app,
    demo,
    path="/",
    css=_COMBINED_CSS,
    js=_KEY_HANDLER_JS,
    footer_links=[],
)

if __name__ == "__main__":
    try:
        uvicorn.run(app, host=SERVER_NAME, port=SERVER_PORT)
    except KeyboardInterrupt:
        pass
    finally:
        state.is_running = False
        def _force_exit():
            time.sleep(SHUTDOWN_GRACE_SECONDS)
            os._exit(0)
        threading.Thread(target=_force_exit, daemon=True).start()
