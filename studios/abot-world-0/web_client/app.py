"""
ABot-World - 实时可交互世界模型 Gradio UI

入口文件：定义 Gradio UI 布局、事件绑定、服务启动。
业务逻辑由 config / state / keyboard / inference / pipeline_loader 模块提供。
"""

# ── A. 环境引导（必须在所有项目导入之前执行）────────────────────────────────
import sys
import os
import time
import json
import threading
import traceback
from pathlib import Path

# 确保日志立即输出（ModelScope SDK 可能缓冲 stdout）
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
os.chdir(_project_root)
os.environ["PROJECT_ROOT"] = str(_project_root)

_gradio_tmp = _project_root / ".gradio_cache"
_gradio_tmp.mkdir(parents=True, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = str(_gradio_tmp)

# ── B. 导入 ─────────────────────────────────────────────────────────────────
from omegaconf import OmegaConf
import gradio as gr
import torch

from web_client.config import (
    PROJECT_ROOT, WEB_DIR, DEBUG_FRONTEND_ONLY,
    STREAM_HEIGHT, STREAM_WIDTH, DEFAULT_REF_IMAGE, PRESETS_FALLBACK_PROMPT,
    SCENE_PRESETS_PATH,
    SERVER_NAME, SERVER_PORT, SHUTDOWN_GRACE_SECONDS,
    VAE_TYPE,
    USE_FP8_GEMM,
)
from web_client.state import state
from web_client.inference import on_click_update_prompt, on_stop, check_model_ready_ui, show_completion_toast, on_click_start_ws, on_stop_ws
from web_client.ui_helpers import format_current_prompt_md
from web_client.pipeline_loader import get_pipeline, _init_lock

# ── C. 前端资源加载 ─────────────────────────────────────────────────────────
_KEY_HUD_WASD_HTML = (WEB_DIR / "key_hud_wasd.html").read_text(encoding="utf-8")
_KEY_HUD_IJKL_HTML = (WEB_DIR / "key_hud_ijkl.html").read_text(encoding="utf-8")
_KEY_HANDLER_JS = (WEB_DIR / "key_handler.js").read_text(encoding="utf-8")

# ── D. 主题 CSS（从外部文件加载）─────────────────────────────────────────
_theme_path = WEB_DIR / "theme.css"
_THEME_CSS = _theme_path.read_text(encoding="utf-8") if _theme_path.is_file() else ""
_HUD_CSS = (WEB_DIR / "key_hud.css").read_text(encoding="utf-8")
_PROMPT_BTNS_CSS = (WEB_DIR / "prompt_buttons.css").read_text(encoding="utf-8")
_COMBINED_CSS = _THEME_CSS + "\n" + _HUD_CSS + "\n" + _PROMPT_BTNS_CSS

# ── D2. 旧版 SSE canvas JS（Docker入口不注入，仅保留兼容）
# 使用 MutationObserver 监听 #ws-canvas-slot 内容变化，不依赖 onload
_SSE_CANVAS_JS = f"""
function abot_start_stream(session_id) {{
  // 关闭已有连接
  if (window._abot_sse) {{ window._abot_sse.close(); window._abot_sse = null; }}
  if (window._abot_ctrl_timer) {{ clearInterval(window._abot_ctrl_timer); window._abot_ctrl_timer = null; }}
  const imgEl = document.querySelector('.fe-video-fixed img') || document.querySelector('.fe-video-fixed');
  if (!imgEl) {{ console.error('abot: .fe-video-fixed not found'); return; }}
  const wrap = imgEl.closest('.fe-video-wrap') || imgEl.parentElement;
  wrap.style.position = 'relative';
  // 移除旧 canvas
  const old = document.getElementById('abot-canvas');
  if (old) old.remove();
  const canvas = document.createElement('canvas');
  canvas.id = 'abot-canvas';
  canvas.width = {STREAM_WIDTH};
  canvas.height = {STREAM_HEIGHT};
  canvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;z-index:10;object-fit:cover;';
  wrap.appendChild(canvas);
  const ctx = canvas.getContext('2d', {{ alpha: false, desynchronized: true }});
  // SSE 连接
  const es = new EventSource('/stream?session_id=' + session_id);
  window._abot_sse = es;
  // Jitter buffer
  const JITTER_MAX = 16, JITTER_MIN = 3;
  let jitterBuf = [];
  let renderPrimed = false;
  let renderInterval = 1000 / 8;
  let lastPaint = 0, lastArrival = 0, arrivalEma = 0;
  es.addEventListener('frame', async (e) => {{
    const data = JSON.parse(e.data);
    const now = performance.now();
    if (lastArrival) {{
      const gap = now - lastArrival;
      arrivalEma = arrivalEma ? (0.2 * gap + 0.8 * arrivalEma) : gap;
      const tgt = Math.min(200, Math.max(20, arrivalEma));
      renderInterval = 0.2 * tgt + 0.8 * renderInterval;
    }}
    lastArrival = now;
    try {{
      const raw = atob(data.jpeg);
      const arr = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) arr[i] = raw.charCodeAt(i);
      const bitmap = await createImageBitmap(new Blob([arr], {{type:'image/jpeg'}}));
      jitterBuf.push({{ bitmap, blk: data.block, fps: data.fps }});
      while (jitterBuf.length > JITTER_MAX) {{
        const stale = jitterBuf.shift();
        if (stale.bitmap && stale.bitmap.close) stale.bitmap.close();
      }}
      if (!renderPrimed && jitterBuf.length >= JITTER_MIN) {{
        renderPrimed = true;
        lastPaint = performance.now() - renderInterval;
      }}
    }} catch(err) {{ console.error('frame decode', err); }}
  }});
  es.addEventListener('status', (e) => {{
    const msg = JSON.parse(e.data);
    const el = document.querySelector('.fe-status');
    if (el) el.innerHTML = '**' + msg + '**';
  }});
  es.addEventListener('ended', () => {{ es.close(); }});
  es.addEventListener('error', (e) => {{ console.error('SSE error:', e); }});
  function renderClock() {{
    requestAnimationFrame(renderClock);
    if (!renderPrimed || jitterBuf.length === 0) return;
    const now = performance.now();
    // buffer 深度越高，播放越快；越低越慢，但不低于 ~3 FPS
    const minInterval = 83;
    const maxInterval = 333;
    const ratio = Math.min(1, jitterBuf.length / JITTER_MAX);
    let targetInterval = maxInterval - (maxInterval - minInterval) * ratio;
    if (now - lastPaint < targetInterval) return;
    const entry = jitterBuf.shift();
    lastPaint = now;
    ctx.drawImage(entry.bitmap, 0, 0, canvas.width, canvas.height);
    if (entry.bitmap && entry.bitmap.close) entry.bitmap.close();
  }}
  requestAnimationFrame(renderClock);
  // Keyboard: send held keys via HTTP POST at 10Hz
  if (!window._abot_keys_bound) {{
    window._abot_keys_bound = true;
    window._abot_pressed = new Set();
    const KEY_MAP = {{ KeyW:'W', KeyA:'A', KeyS:'S', KeyD:'D', KeyI:'I', KeyJ:'J', KeyK:'K', KeyL:'L' }};
    document.addEventListener('keydown', (e) => {{
      const k = KEY_MAP[e.code];
      if (k) {{ e.preventDefault(); window._abot_pressed.add(k); updateKeycaps(); }}
      if (e.code === 'Escape') {{ window._abot_pressed.clear(); updateKeycaps(); }}
    }});
    document.addEventListener('keyup', (e) => {{
      const k = KEY_MAP[e.code];
      if (k) {{ window._abot_pressed.delete(k); updateKeycaps(); }}
    }});
    function updateKeycaps() {{
      document.querySelectorAll('.keycap[data-key]').forEach(el => {{
        el.classList.toggle('active', window._abot_pressed.has(el.dataset.key));
      }});
    }}
  }}
  if (window._abot_ctrl_timer) clearInterval(window._abot_ctrl_timer);
  window._abot_ctrl_timer = setInterval(() => {{
    fetch('/control?session_id=' + session_id, {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{ buttons: Array.from(window._abot_pressed || []) }})
    }}).catch(() => {{}});
  }}, 100);
}}
function abot_stop_stream() {{
  if (window._abot_sse) {{ window._abot_sse.close(); window._abot_sse = null; }}
  if (window._abot_ctrl_timer) {{ clearInterval(window._abot_ctrl_timer); window._abot_ctrl_timer = null; }}
  const c = document.getElementById('abot-canvas');
  if (c) c.remove();
}}
// 轮询监听 #ws-canvas-slot 内容变化（比 MutationObserver 更可靠，不受 Svelte DOM 替换影响）
setInterval(function() {{
  const slot = document.getElementById('ws-canvas-slot');
  if (!slot) return;
  const startEl = slot.querySelector('[data-session-id]');
  if (startEl) {{
    const sid = startEl.getAttribute('data-session-id');
    if (sid && window._abot_last_sid !== sid) {{
      console.log('[ABOT] start stream:', sid);
      window._abot_last_sid = sid;
      abot_start_stream(sid);
    }}
  }}
  const stopEl = slot.querySelector('[data-action="stop"]');
  if (stopEl && window._abot_last_sid) {{
    console.log('[ABOT] stop stream');
    window._abot_last_sid = null;
    abot_stop_stream();
  }}
}}, 200);
"""
_COMBINED_JS = _KEY_HANDLER_JS + "\n" + _SSE_CANVAS_JS

# ── E. scene_presets.yaml（唯一数据源：图 + Prompt + default_prompt）──────────
REF_IMAGE_ENTRIES: list[tuple[str, str]] = []  # [(path, caption), ...]
REF_IMAGE_PATHS: list[str] = []


def _read_caption_prompt(caption_path: Path) -> str:
    """从 caption JSON 文件中读取 scene_static 字段作为 prompt 文本。"""
    try:
        with open(caption_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        text = data.get("scene_static", "")
        if isinstance(text, str) and text.strip():
            return text.strip()
    except Exception:
        pass
    return ""


def _resolve_prompt(raw: str, base_dir: Path) -> str:
    """解析 prompt 值：若以 .json 结尾则视为 caption 文件路径并读取 scene_static，否则直接作为文本。"""
    if not raw:
        return ""
    if raw.endswith(".json"):
        p = Path(raw)
        caption_path = p if p.is_absolute() else base_dir / raw
        return _read_caption_prompt(caption_path)
    return raw


_CAPTION_PREFIX = ""


def _apply_caption_prefix(prompt: str) -> str:
    """直接返回 prompt，不加前缀。"""
    return prompt


def _load_scene_presets() -> tuple[list[tuple[str, str]], list[str], list[str], int, str]:
    """加载 web_client/scene_presets.yaml。

    image 为相对于 web_client/ 的路径。
    prompt 支持两种形式：.json 文件路径（读取 scene_static 字段）或直接字符串。
    返回 (entries, paths, prompts, default_index, default_prompt 字符串)。
    """
    entries: list[tuple[str, str]] = []
    prompts: list[str] = []
    seen: set[str] = set()

    def add_file(p: Path, caption: str, prompt_text: str) -> None:
        if not p.is_file():
            return
        key = str(p.resolve())
        if key in seen:
            return
        seen.add(key)
        entries.append((key, caption))
        prompts.append(prompt_text)

    if not SCENE_PRESETS_PATH.is_file():
        return [], [], [], 0, _apply_caption_prefix(PRESETS_FALLBACK_PROMPT)

    try:
        cfg = OmegaConf.load(str(SCENE_PRESETS_PATH))
    except Exception:
        return [], [], [], 0, _apply_caption_prefix(PRESETS_FALLBACK_PROMPT)

    def _resolve(rel: str) -> Path:
        """将相对路径解析为绝对路径（相对于 WEB_DIR）。"""
        p = Path(rel)
        return p if p.is_absolute() else WEB_DIR / rel

    # default_prompt：支持 JSON 文件路径或直接字符串
    dp_raw = str(cfg.get("default_prompt") or "").strip()
    if dp_raw:
        default_prompt = _resolve_prompt(dp_raw, WEB_DIR) or PRESETS_FALLBACK_PROMPT
    else:
        default_prompt = PRESETS_FALLBACK_PROMPT
    default_index = int(cfg.get("default_index") or 0)

    for g in cfg.get("groups") or []:
        gname = (g.get("name") or "").strip() or "预设"
        for item in (g.get("items") or []):
            # image: 相对路径 -> {WEB_DIR}/{image}
            img_rel = (item.get("image") or "").strip()
            if not img_rel:
                continue
            p = _resolve(img_rel)

            label = (item.get("label") or "").strip()
            caption = label if label else f"{gname} · {p.stem}"

            # prompt: 支持 JSON 文件路径（读取 scene_static）或直接字符串
            # 显式 prompt: "" → 空字符串；省略 prompt 键 → 继承 default_prompt
            if "prompt" in item:
                pr_raw = str(item.get("prompt") or "").strip()
                if pr_raw:
                    row_prompt = _resolve_prompt(pr_raw, WEB_DIR) or default_prompt
                else:
                    row_prompt = ""
            else:
                row_prompt = default_prompt
            add_file(p, caption, _apply_caption_prefix(row_prompt))

    if not entries:
        return [], [], [], 0, _apply_caption_prefix(default_prompt)

    if default_index < 0 or default_index >= len(entries):
        default_index = 0
    paths = [x for x, _ in entries]
    return entries, paths, prompts, default_index, _apply_caption_prefix(default_prompt)


PRESET_PROMPTS: list[str] = []
REF_IMAGE_ENTRIES, REF_IMAGE_PATHS, PRESET_PROMPTS, _preset_default_idx, _file_default_prompt = _load_scene_presets()
if REF_IMAGE_PATHS:
    INITIAL_PROMPT = PRESET_PROMPTS[_preset_default_idx]
    state.ref_image_path = REF_IMAGE_PATHS[_preset_default_idx]
else:
    INITIAL_PROMPT = _file_default_prompt
    state.ref_image_path = None


def on_ref_image_update(filepath: str | None):
    """参考图选中回调：将路径写入 StreamState，下一个 block 生效。"""
    state.ref_image_path = filepath if filepath else None


def on_ref_gallery_select(evt: gr.SelectData):
    """画廊选中：
    1) 更新首帧参考图路径（影响后续生成/流结束后的最终替换）。
    2) 若正在流式生成：请求停止，让已入队帧先播放完，start_stream 最后会替换为选中图。
    3) 若未在生成：立刻用选中图替换 gr.Image。
    """
    if not REF_IMAGE_PATHS or evt.index < 0 or evt.index >= len(REF_IMAGE_PATHS):
        return gr.update(), gr.skip(), gr.skip()

    path = REF_IMAGE_PATHS[evt.index]

    prompt_val = PRESET_PROMPTS[evt.index] if evt.index < len(PRESET_PROMPTS) else gr.update()

    return prompt_val, gr.update(value=path), path


def on_video_gallery_select(evt: gr.SelectData):
    """Videos Gallery 选中回调：显示视频播放区并播放选中的视频。
    
    Gallery 条目格式为 (缩略图路径, 视频路径)，evt.value 结构为:
    {'image': {...}, 'caption': '视频路径'}
    """
    try:
        # evt.value 是字典，视频路径在 caption 字段
        video_path = None
        if isinstance(evt.value, dict):
            video_path = evt.value.get('caption')
        elif isinstance(evt.value, str):
            video_path = evt.value

        if video_path and Path(video_path).exists():
            return gr.update(value=video_path), gr.update(visible=True)
        return gr.skip(), gr.skip()
    except Exception:
        return gr.skip(), gr.skip()


def on_close_video_player():
    """关闭视频播放区。"""
    return gr.update(value=None), gr.update(visible=False)


# ── F. Gradio Blocks UI 定义 ────────────────────────────────────────────────
with gr.Blocks(title="ABot-World - 实时可交互世界模型") as demo:
    ws_session_state = gr.State("")
    ref_path_state = gr.State(
        state.ref_image_path or DEFAULT_REF_IMAGE
    )
    gr.Markdown(
        "# ABot-World - 实时可交互世界模型\n\n",
        elem_classes=["fe-header", "fe-header--intro-gap"],
    )
    # ── 主区：实时画面 ────────────────────────────────────────────────────────
    with gr.Column(elem_classes=["fe-video-wrap"]):
        overlay_label = gr.HTML(
            '<div class="fe-video-overlay-label">正在链接你的世界</div>',
            visible=not DEBUG_FRONTEND_ONLY,
            elem_classes=["fe-overlay-wrapper"],
        )
        # 初始无图：去掉所有填充图标
        image_output = gr.Image(
            label=None,
            value=state.ref_image_path or DEFAULT_REF_IMAGE,
            visible=True,
            height=STREAM_HEIGHT,
            show_label=False,
            elem_classes=["fe-video-fixed"],
            format="jpeg"
        )
        # WebSocket canvas overlay 容器（由 JS 动态创建 canvas）
        ws_html = gr.HTML("", visible=True, elem_id="ws-canvas-slot")

        # ── 进度条：frame 生成与播放进度 ──────────────────────────────────────────
        progress_bar = gr.HTML(
            value="",
            elem_classes=["fe-progress-container"],
            visible=True,
        )

        # ── 控制区：一行 左|中|右（模型状态与当前 Prompt 见页面底部 Debug 区，Ctrl+D 切换显示）──
        with gr.Column(elem_classes=["fe-controls-stack"]):
            with gr.Row(elem_classes=["fe-bottom-row"]):
                with gr.Column(scale=9, min_width=132, elem_classes=["fe-hud-col", "fe-hud-cyber"]):
                    key_display_wasd = gr.HTML(_KEY_HUD_WASD_HTML)

                with gr.Column(scale=30, elem_classes=["fe-prompt-col"]):
                    prompt_input = gr.Textbox(
                        label="描绘你的世界",
                        lines=5,
                        placeholder="e.g., A cat walking on a sunny beach...",
                        value=INITIAL_PROMPT,
                    )
                    with gr.Row(elem_classes=["fe-prompt-btns"]):
                        update_btn = gr.Button("唤醒你的世界", variant="primary", interactive=DEBUG_FRONTEND_ONLY)
                        stop_btn = gr.Button("封存你的世界", variant="stop")

                with gr.Column(scale=9, min_width=132, elem_classes=["fe-hud-col", "fe-hud-cyber"]):
                    key_display_ijkl = gr.HTML(_KEY_HUD_IJKL_HTML)

    # ── 7: 场景预设（探索平行宇宙 参考图 + Prompt），数据来自 scene_presets.yaml ───────────────
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("**探索平行宇宙**", elem_classes=["fe-ref-title"])
            _ref_n = len(REF_IMAGE_ENTRIES)
            ref_gallery = gr.Gallery(
                value=REF_IMAGE_ENTRIES if REF_IMAGE_ENTRIES else None,
                label=None,
                columns=None,  # 自适应单行横向排列
                rows=1,
                height=172,
                object_fit="cover",
                show_label=False,
                allow_preview=False,
                interactive=True,
                elem_classes=["fe-ref-gallery-hscroll"],
            )

    # ── Debug 区：默认隐藏，前端 Ctrl+D 切换（见 key_handler.js / theme .fe-debug-panel）──
    with gr.Column(elem_classes=["fe-debug-panel"]):
        with gr.Row(elem_classes=["fe-meta-below-row", "fe-debug-panel__row"]):
            with gr.Column(scale=1, min_width=0, elem_classes=["fe-meta-slot-3-wrap"]):
                status_output = gr.Markdown(
                    value="**[仅前端调试]** 未加载模型，界面可正常操作。" if DEBUG_FRONTEND_ONLY else "**模型加载中…** 加载完成后「唤醒你的世界」将可用。",
                    elem_classes=["fe-status", "fe-status-inline", "fe-meta-slot-3"],
                )
            with gr.Column(scale=1, min_width=0, elem_classes=["fe-meta-slot-4-wrap", "fe-vae-hud-wrap"]):
                current_prompt_display = gr.Markdown(
                    value=format_current_prompt_md(INITIAL_PROMPT),
                    elem_classes=["fe-current-prompt", "fe-status-inline"],
                )

    model_ready_timer = gr.Timer(2)
    model_ready_timer.tick(
        fn=check_model_ready_ui,
        inputs=[ws_session_state],
        outputs=[
            status_output,
            update_btn,
            overlay_label,
            model_ready_timer,
        ],
        queue=False,
    )

    # ── G. 事件绑定 ───────────────────────────────────────────────────────────
    # iframe+WebSocket 版：启动会话并返回画布 iframe。
    update_btn.click(
        fn=on_click_start_ws,
        inputs=[prompt_input, ref_path_state],
        outputs=[
            ws_html,
            status_output,
            current_prompt_display,
            update_btn,
            overlay_label,
            progress_bar,
            image_output,
            ws_session_state,
        ],
        queue=False,
    )

    ref_gallery.select(
        fn=on_ref_gallery_select,
        inputs=None,
        outputs=[prompt_input, image_output, ref_path_state],
        queue=False,
    )

    # iframe+WebSocket 版：Stop 按钮（同时恢复参考图）
    stop_btn.click(
        fn=on_stop_ws,
        inputs=[ws_session_state, ref_path_state, prompt_input],
        outputs=[
            ws_html,
            status_output,
            current_prompt_display,
            update_btn,
            overlay_label,
            progress_bar,
            image_output,
            ws_session_state,
        ],
        queue=False,
    )

    # 键盘 HUD 仅负责视觉；实时控制由 iframe 内的双向 WebSocket 发送，
    # 不再触发高频 Gradio queue 回调。

demo.queue(default_concurrency_limit=1)

# ── H. 模型下载 + 后台加载 pipeline（模块级，import 时即启动）──────────────────
_MODEL_ID = "amap_cvlab/ABot-World-0-5B-LF"
_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "ABot-World-0-5B-LF"
_MODEL_CACHE_ROOT = Path(
    os.environ.get("MODELSCOPE_CACHE", "/mnt/workspace/.cache/modelscope")
)


def _missing_checkpoint_assets() -> list[str]:
    """Return missing/truncated assets required by the configured pipeline."""
    required_files = {
        "config.json": 32,
        "diffusion_pytorch_model.safetensors": 1024 * 1024 * 1024,
        "models_t5_umt5-xxl-enc-bf16.pth": 1024 * 1024 * 1024,
        "Wan2.2_VAE.pth": 100 * 1024 * 1024,
        "taew2_2.pth": 1024 * 1024,
    }
    missing = []
    for relative_path, minimum_bytes in required_files.items():
        path = _CHECKPOINT_DIR / relative_path
        try:
            if not path.is_file() or path.stat().st_size < minimum_bytes:
                missing.append(relative_path)
        except OSError:
            missing.append(relative_path)

    tokenizer_dir = _CHECKPOINT_DIR / "google" / "umt5-xxl"
    try:
        has_tokenizer_files = tokenizer_dir.is_dir() and any(
            item.is_file() and item.stat().st_size > 0
            for item in tokenizer_dir.rglob("*")
        )
    except OSError:
        has_tokenizer_files = False
    if not has_tokenizer_files:
        missing.append("google/umt5-xxl/")
    return missing


def ensure_model_downloaded():
    """从 ModelScope 下载模型检查点（如果尚未下载）。"""
    import shutil
    missing = _missing_checkpoint_assets()
    if not missing:
        print(f"[SETUP] Model already exists at {_CHECKPOINT_DIR}", flush=True)
        return
    print(
        f"[SETUP] Downloading/repairing {_MODEL_ID}; "
        f"missing_or_truncated={missing}",
        flush=True,
    )
    from modelscope import snapshot_download
    _MODEL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    model_dir = snapshot_download(_MODEL_ID, cache_dir=str(_MODEL_CACHE_ROOT))
    print(f"[SETUP] Model downloaded to {model_dir}", flush=True)
    _CHECKPOINT_DIR.parent.mkdir(parents=True, exist_ok=True)
    if _CHECKPOINT_DIR.is_symlink():
        _CHECKPOINT_DIR.unlink()
    elif _CHECKPOINT_DIR.exists():
        shutil.rmtree(_CHECKPOINT_DIR)
    os.symlink(model_dir, str(_CHECKPOINT_DIR))
    print(f"[SETUP] Symlinked {model_dir} -> {_CHECKPOINT_DIR}", flush=True)
    missing = _missing_checkpoint_assets()
    if missing:
        raise RuntimeError(
            f"ModelScope snapshot is incomplete after download: {missing}"
        )


if DEBUG_FRONTEND_ONLY:
    state.model_ready = True
else:
    def _download_and_load_pipeline():
        """后台线程：先下载模型，再加载 pipeline。"""
        try:
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                print(
                    "[BOOT][GPU] "
                    f"name={props.name}, cc={props.major}.{props.minor}, "
                    f"vram={props.total_memory / (1024 ** 3):.1f}GiB, "
                    f"torch={torch.__version__}, cuda={torch.version.cuda}",
                    flush=True,
                )
            ensure_model_downloaded()
            print("[INIT] Loading pipeline in background...", flush=True)
            with _init_lock:
                p, cfg, dev = get_pipeline(
                    vae_type=VAE_TYPE,
                    use_fp8_gemm=USE_FP8_GEMM,
                )
            # 注入 pipeline 引用到 WebSocket worker。
            from web_client import ws_stream
            ws_stream.set_pipeline(p, cfg, dev)
            print("[INIT] Pipeline injected into ws_stream.", flush=True)
        except Exception as e:
            state.model_error = f"{type(e).__name__}: {e}"
            print(f"[INIT] Pipeline initialization failed: {state.model_error}", flush=True)
            traceback.print_exc()
            return
        state.model_error = None
        state.model_ready = True
        print("[INIT] Pipeline ready. WebSocket streaming + 「唤醒你的世界」已可用。", flush=True)

    threading.Thread(target=_download_and_load_pipeline, daemon=True).start()
    print("[INIT] Web UI starting (model downloads & loads in background).", flush=True)


# ── I. Main 入口 ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        demo.launch(
            server_name=SERVER_NAME,
            server_port=SERVER_PORT,
            js=_KEY_HANDLER_JS,
            css=_COMBINED_CSS,
            footer_links=[],  # 隐藏 Gradio 默认页脚（API / Built with Gradio / 设置）
        )
    except KeyboardInterrupt:
        pass
    finally:
        state.is_running = False
        def _force_exit():
            time.sleep(SHUTDOWN_GRACE_SECONDS)
            os._exit(0)
        threading.Thread(target=_force_exit, daemon=True).start()
