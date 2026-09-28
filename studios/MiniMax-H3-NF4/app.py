"""MiniMax-H3 NF4 Studio for ModelScope xGPU.

The public UI and Gradio API intentionally stay compatible with the original
Hugging Face Space.  The inference backend is local DiffSynth-Studio: no
ZeroGPU decorator, remote conditioner, or Hugging Face token forwarding.
"""

from __future__ import annotations

import gc
import os
import tempfile
import threading
import time
import traceback

import gradio as gr

MODEL_REPO = os.environ.get("H3_MODEL_REPO", "DiffSynth-Studio/MiniMax-H3-NF4")
PROCESSOR_REPO = os.environ.get("H3_PROCESSOR_REPO", "MiniMax/MiniMax-H3")
GPU_RESERVE_GB = float(os.environ.get("H3_GPU_RESERVE_GB", "4"))
try:
    MAX_OUTPUTS = max(1, int(os.environ.get("H3_MAX_OUTPUTS", "5")))
except ValueError:
    MAX_OUTPUTS = 5
    print("[storage] invalid H3_MAX_OUTPUTS; using 5", flush=True)

PERSIST_ROOT = os.environ.get("H3_WORKSPACE", "/mnt/workspace")
try:
    os.makedirs(PERSIST_ROOT, exist_ok=True)
    if not os.access(PERSIST_ROOT, os.W_OK):
        raise PermissionError(PERSIST_ROOT)
except (OSError, PermissionError):
    PERSIST_ROOT = tempfile.gettempdir()

MODEL_CACHE = os.path.join(PERSIST_ROOT, "modelscope-cache")
OUTPUT_DIR = os.path.join(PERSIST_ROOT, "h3-outputs")
os.makedirs(MODEL_CACHE, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ.setdefault("MODELSCOPE_CACHE", MODEL_CACHE)

CANVASES = {
    "960x544 · 16:9 fast": (544, 960),
    "1024x576 · 16:9 fast": (576, 1024),
    "1152x640 · 16:9": (640, 1152),
    "1280x704 · 16:9": (704, 1280),
    "1344x768 · 16:9 full": (768, 1344),
    "544x960 · 9:16 fast": (960, 544),
    "640x1152 · 9:16": (1152, 640),
    "768x1344 · 9:16 full": (1344, 768),
    "544x544 · 1:1 fast": (544, 544),
    "768x768 · 1:1 full": (768, 768),
    "768x576 · 4:3 fast": (576, 768),
    "1024x768 · 4:3 full": (768, 1024),
    "576x768 · 3:4 fast": (768, 576),
    "768x1024 · 3:4 full": (1024, 768),
    "1152x512 · 21:9 fast": (512, 1152),
    "1536x672 · 21:9 full": (672, 1536),
}
DEFAULT_CANVAS = "960x544 · 16:9 fast"
FPS, FRAMES_PER_CHUNK, LATENTS_PER_CHUNK = 24, 17, 5
MIN_UI_DURATION, MAX_UI_DURATION = 2, 14

PIPE = None
LOAD_ERROR: str | None = None
LOADED_IN: float | None = None
ACTIVE_LORA = "off"
MODEL_LOCK = threading.RLock()
OUTPUT_LOCK = threading.Lock()


def _cleanup_output_videos(keep: int, reason: str) -> None:
    """Keep only the newest completed MP4 files without touching model caches."""
    keep = max(0, int(keep))
    videos: list[tuple[float, str]] = []
    try:
        names = os.listdir(OUTPUT_DIR)
    except OSError as error:
        print(f"[storage] {reason}; unable to scan outputs: {type(error).__name__}: {error}", flush=True)
        return

    for name in names:
        if not name.startswith("h3-") or not name.endswith(".mp4"):
            continue
        path = os.path.join(OUTPUT_DIR, name)
        try:
            if os.path.isfile(path):
                videos.append((os.path.getmtime(path), path))
        except OSError as error:
            print(f"[storage] {reason}; unable to inspect {name}: {type(error).__name__}: {error}", flush=True)

    videos.sort(key=lambda item: (item[0], item[1]))
    deleted = 0
    released = 0
    for _, path in videos[: max(0, len(videos) - keep)]:
        try:
            size = os.path.getsize(path)
            os.remove(path)
            released += size
            deleted += 1
        except FileNotFoundError:
            continue
        except OSError as error:
            print(
                f"[storage] {reason}; unable to delete {os.path.basename(path)}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )

    remaining = max(0, len(videos) - deleted)
    print(
        f"[storage] {reason}; kept={remaining}/{keep}; deleted={deleted}; "
        f"released={released / 1024**3:.2f} GiB",
        flush=True,
    )


def _remove_partial_output(path: str) -> None:
    try:
        os.remove(path)
        print(f"[storage] removed incomplete output {os.path.basename(path)}", flush=True)
    except FileNotFoundError:
        pass
    except OSError as error:
        print(
            f"[storage] unable to remove incomplete output {os.path.basename(path)}: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )


_cleanup_output_videos(MAX_OUTPUTS, "startup retention")


def _cuda_memory(label: str) -> None:
    """Log allocator and device memory without making cleanup itself fatal."""
    try:
        import torch

        if not torch.cuda.is_available():
            return
        free, total = torch.cuda.mem_get_info("cuda")
        gib = 1024**3
        print(
            f"[vram] {label}; free={free / gib:.2f}/{total / gib:.2f} GiB; "
            f"allocated={torch.cuda.memory_allocated() / gib:.2f} GiB; "
            f"reserved={torch.cuda.memory_reserved() / gib:.2f} GiB",
            flush=True,
        )
    except Exception as error:
        print(f"[vram] {label}; stats unavailable: {type(error).__name__}: {error}", flush=True)


def _release_inference_vram(label: str) -> None:
    """Offload all managed models, then release unreferenced CUDA allocations."""
    try:
        import torch

        if PIPE is not None and getattr(PIPE, "vram_management_enabled", False):
            PIPE.load_models_to_device([])
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        _cuda_memory(label)
    except Exception as error:
        print(f"[vram] {label}; cleanup failed: {type(error).__name__}: {error}", flush=True)


def snap_frames(seconds: float) -> int:
    frames = max(1, round(float(seconds) * FPS))
    while frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        frames += 1
    return frames


def status() -> str:
    if LOAD_ERROR:
        return LOAD_ERROR
    if PIPE is None:
        return f"正在加载 `{MODEL_REPO}`（NF4，本地 DiffSynth 后端）。首次启动约需下载 52 GB。"
    return (
        f"已就绪 · `{MODEL_REPO}` · NF4 · DiffSynth-Studio · xGPU · "
        f"LoRA `{ACTIVE_LORA}` · 加载耗时 {LOADED_IN:.0f} 秒"
    )


def load_models() -> str | None:
    global PIPE, LOAD_ERROR, LOADED_IN
    if PIPE is not None or LOAD_ERROR is not None:
        return LOAD_ERROR

    started = time.time()
    try:
        import torch
        from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. This Studio must run on ModelScope xGPU.")

        total_gb = torch.cuda.mem_get_info("cuda")[1] / (1024**3)
        vram_limit = max(1.0, total_gb - GPU_RESERVE_GB)
        vram_config = {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": torch.bfloat16,
            "onload_device": "cpu",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }
        print(
            f"[load] {MODEL_REPO}; GPU {total_gb:.1f} GiB, vram_limit={vram_limit:.1f} GiB, cache={MODEL_CACHE}",
            flush=True,
        )
        pipe = MiniMaxH3Pipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(model_id=MODEL_REPO, origin_file_pattern="minimax-h3-fl2va-nf4.safetensors", **vram_config),
                ModelConfig(model_id=MODEL_REPO, origin_file_pattern="minimax-h3-text-encoder-nf4.safetensors", **vram_config),
                ModelConfig(model_id=MODEL_REPO, origin_file_pattern="video_vae_nf4.safetensors", **vram_config),
                ModelConfig(model_id=MODEL_REPO, origin_file_pattern="audio_vae_nf4.safetensors", **vram_config),
            ],
            processor_config=ModelConfig(model_id=PROCESSOR_REPO, origin_file_pattern="FL2VA/processor/"),
            vram_limit=vram_limit,
        )
        PIPE = pipe
        LOADED_IN = time.time() - started
        print(f"[load] ready in {LOADED_IN:.0f}s", flush=True)
    except Exception as error:
        traceback.print_exc()
        LOAD_ERROR = (
            f"**Loading `{MODEL_REPO}` failed** after {time.time() - started:.0f}s: "
            f"`{type(error).__name__}: {error}`"
        )
    return LOAD_ERROR


def _fit_keyframe(image_path: str, current_canvas: str):
    """Cover-crop a keyframe and choose the closest supported aspect ratio."""
    from PIL import Image

    img = Image.open(image_path)
    aspect = img.width / img.height
    fastest = {}
    for label, (height, width) in CANVASES.items():
        ratio = width / height
        if ratio not in fastest or width * height < fastest[ratio][1][0] * fastest[ratio][1][1]:
            fastest[ratio] = (label, (height, width))
    ratio = min(fastest, key=lambda item: abs(item - aspect))
    label, (height, width) = fastest[ratio]

    if current_canvas not in CANVASES:
        current_canvas = DEFAULT_CANVAS
    cur_h, cur_w = CANVASES[current_canvas]
    if abs(cur_w / cur_h - aspect) <= abs(ratio - aspect):
        label, height, width = current_canvas, cur_h, cur_w

    target = width / height
    if abs(img.width / img.height - target) > 1e-3:
        if img.width / img.height > target:
            new_width = int(img.height * target)
            left = (img.width - new_width) // 2
            img = img.crop((left, 0, left + new_width, img.height))
        else:
            new_height = int(img.width / target)
            top = (img.height - new_height) // 2
            img = img.crop((0, top, img.width, top + new_height))
        img.save(image_path)
    return image_path, label


def _resolve_lora(lora, use_lora) -> str:
    if isinstance(lora, str) and lora in ("larry", "lightx", "off"):
        return lora
    return "larry" if use_lora else "off"


def recommended_steps(lora: str) -> int:
    """UI defaults only; API callers may still deliberately choose another value."""
    return {"larry": 6, "lightx": 4, "off": 50}.get(lora, 6)


def _open_keyframe(path):
    if not path:
        return None
    from PIL import Image, ImageOps

    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def generate(
    prompt,
    image_path=None,
    last_image_path=None,
    canvas=DEFAULT_CANVAS,
    duration=5,
    steps=6,
    seed=42,
    upsample=False,
    use_lora=True,
    lora="",
    ip_token=None,
):
    """Generate one synchronized video/audio sample; signature matches the HF Space."""
    global ACTIVE_LORA
    del ip_token  # ModelScope xGPU is attached to this Studio; caller token forwarding is unnecessary.
    if LOAD_ERROR:
        raise Exception(LOAD_ERROR)
    if PIPE is None:
        raise Exception("The NF4 pipeline is still loading. Check /status or the Studio logs.")
    if not prompt or not prompt.strip():
        raise Exception("MiniMax-H3 always takes a prompt, keyframes or not.")
    if canvas not in CANVASES:
        canvas = DEFAULT_CANVAS

    first = image_path["path"] if isinstance(image_path, dict) else image_path
    last = last_image_path["path"] if isinstance(last_image_path, dict) else last_image_path
    if first:
        first, canvas = _fit_keyframe(first, canvas)
    if last:
        last, canvas = _fit_keyframe(last, canvas)

    height, width = CANVASES[canvas]
    num_frames = snap_frames(duration)
    requested_lora = _resolve_lora(lora, use_lora)
    keyframes = [image for image in (_open_keyframe(first), _open_keyframe(last)) if image is not None]
    keyframe_indices = None
    if first and last:
        keyframe_indices = [0, -1]
    elif first:
        keyframe_indices = [0]
    elif last:
        keyframe_indices = [-1]

    started = time.time()
    with MODEL_LOCK:
        import h3_lora

        _release_inference_vram("before inference")
        try:
            active_lora = h3_lora.set_active(PIPE, requested_lora, cache_dir=MODEL_CACHE)
            ACTIVE_LORA = active_lora
            video, audio = PIPE(
                prompt=prompt.strip(),
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=int(steps),
                seed=int(seed),
                keyframes=keyframes or None,
                keyframe_indices=keyframe_indices,
                tiled=True,
            )
        finally:
            _release_inference_vram("after inference")
    generate_seconds = time.time() - started

    from diffsynth.utils.data.audio_video import write_video_audio

    path = os.path.join(OUTPUT_DIR, f"h3-{int(time.time() * 1000)}.mp4")
    with OUTPUT_LOCK:
        # Make room before encoding so a full persistent volume can recover on
        # the next request. The successful result then becomes the newest item.
        _cleanup_output_videos(MAX_OUTPUTS - 1, "before video write")
        try:
            write_video_audio(video=video, audio=audio, output_path=path, fps=FPS, audio_sample_rate=32000)
        except BaseException:
            _remove_partial_output(path)
            raise
        _cleanup_output_videos(MAX_OUTPUTS, "after video write")

    refined = ""
    if upsample:
        refined = "NF4 本地后端未启用远程 prompt upsampler；本次已使用原始 prompt。"
    report = (
        f"{width}x{height} · {num_frames} frames ({num_frames / FPS:.3f} s) · {int(steps)} steps · "
        f"local conditioner · generation {generate_seconds:.0f}s ({generate_seconds / max(1, int(steps)):.1f} s/step) · "
        f"turbo LoRA {active_lora} · seed {int(seed)}"
    )
    print(f"[gen] {report}", flush=True)
    return path, report, refined


APP_CSS = """
.gradio-container { max-width: none !important; width: 100% !important; padding-inline: 16px !important; }
#studio-title h1 { margin-bottom: 0.2rem; }
#studio-title p { color: var(--body-text-color-subdued); }
#aigc-notice {
    margin: 8px 0 14px;
    padding: 12px 16px;
    border: 1px solid rgba(109, 93, 252, 0.32);
    border-left: 4px solid #6d5dfc;
    border-radius: 10px;
    background: rgba(109, 93, 252, 0.08);
    color: var(--body-text-color);
}
#aigc-notice p { margin: 0 !important; }
#aigc-notice a {
    color: #6d5dfc !important;
    font-weight: 700;
    text-decoration: underline;
    text-underline-offset: 2px;
}
#status-card { border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 10px 14px; }
#generate-button { background: #f59e0b; color: #111; border: 0; font-weight: 700; }
#keyframe-row { gap: 8px; }
#keyframe-row .image-container { min-height: 116px !important; height: 116px !important; }
#keyframe-row .image-container > div { min-height: 116px !important; height: 116px !important; }
#keyframe-row .wrap { min-height: 82px !important; height: 82px !important; }
#output-column { min-height: calc(100vh - 170px); }
#output-video { min-height: calc(100vh - 250px); height: calc(100vh - 250px); }
#output-video video { width: 100% !important; height: 100% !important; object-fit: contain; }
"""


def status_markdown() -> str:
    ready = PIPE is not None and LOAD_ERROR is None
    icon = "🟢" if ready else ("🔴" if LOAD_ERROR else "🟠")
    return f"{icon} **后端状态** — {status()}"


with gr.Blocks(title="MiniMax-H3 NF4 Studio", css=APP_CSS, fill_width=True) as app:
    gr.Markdown(
        "# MiniMax-H3 NF4 Studio\n"
        "Video with synchronized soundtrack · DiffSynth-Studio NF4 backend · ModelScope xGPU",
        elem_id="studio-title",
    )
    gr.Markdown(
        "魔搭 AIGC 专区现已支持 MiniMax H3 的训练与推理。如需体验完整的视频生成能力与更精细的参数控制，可前往 "
        "[H3 视频生成](https://modelscope.cn/aigc/video-generation) 或 "
        "[模型训练](https://modelscope.cn/aigc/model-training)。",
        elem_id="aigc-notice",
    )
    status_box = gr.Markdown(status_markdown(), elem_id="status-card")

    with gr.Row(equal_height=True):
        with gr.Column(scale=3, min_width=330):
            prompt = gr.Textbox(
                label="Prompt",
                value="A red fox trotting through a snowy pine forest at dawn, snow crunching underfoot",
                lines=5,
            )
            lora = gr.Dropdown(
                choices=[
                    ("Larry Turbo v4 · recommended 6 steps", "larry"),
                    ("LightX2V Turbo v0.1 · recommended 4 steps", "lightx"),
                    ("NF4 base model · 50 steps", "off"),
                ],
                value="larry",
                label="Turbo LoRA",
            )
            upsample = gr.Checkbox(value=False, label="Upsample prompt")
            use_lora = gr.Checkbox(value=True, visible=False)

            with gr.Row(elem_id="keyframe-row"):
                image_path = gr.Image(type="filepath", label="First frame (optional)", height=116)
                last_image_path = gr.Image(type="filepath", label="Last frame (optional)", height=116)

            canvas = gr.Dropdown(choices=list(CANVASES), value=DEFAULT_CANVAS, label="Canvas")
            duration = gr.Slider(MIN_UI_DURATION, MAX_UI_DURATION, value=5, step=1, label="Duration (seconds)")
            steps = gr.Slider(2, 50, value=6, step=1, label="Inference steps")
            seed = gr.Number(value=42, precision=0, label="Seed")
            gr.Examples(
                examples=[
                    ["A red fox trotting through a snowy pine forest at dawn, snow crunching underfoot"],
                    ["A busy night market, neon signs reflecting in puddles, sizzling street food"],
                    ["A cellist playing a slow melody in an empty concert hall"],
                    ["Waves crashing against basalt cliffs at golden hour, gulls crying overhead"],
                ],
                inputs=[prompt],
                label="示例 Prompt",
            )
            run = gr.Button("▶ Generate", variant="primary", elem_id="generate-button")

        with gr.Column(scale=9, min_width=640, elem_id="output-column"):
            video = gr.Video(label="Generated video + audio", elem_id="output-video")
            report = gr.Textbox(label="Generation report", interactive=False)
            refined = gr.Textbox(label="Prompt note", interactive=False)

    lora.change(
        fn=recommended_steps,
        inputs=lora,
        outputs=steps,
        api_visibility="private",
    )

    run.click(
        fn=generate,
        inputs=[prompt, image_path, last_image_path, canvas, duration, steps, seed, upsample, use_lora, lora],
        outputs=[video, report, refined],
        api_name="generate",
    )
    app.load(fn=status_markdown, outputs=status_box, api_visibility="private")
    status_timer = gr.Timer(5)
    status_timer.tick(fn=status_markdown, outputs=status_box, api_visibility="private")


threading.Thread(target=load_models, name="h3-model-loader", daemon=True).start()

if __name__ == "__main__":
    app.queue(max_size=4, default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        show_error=True,
        allowed_paths=[OUTPUT_DIR],
    )
