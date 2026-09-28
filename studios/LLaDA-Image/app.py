"""LLaDA-Image Studio for ModelScope xGPU.

Unified text-to-image / image-editing demo with two sampling presets:
- Turbo (4 steps, distilled, fast)
- Base  (50 steps, high quality)

Inference backend is the official `src.LLaDAImagePipeline` vendored from
https://github.com/inclusionAI/LLaDA-Image @ abd68bd (unmodified).
Weights are downloaded from ModelScope (never Hugging Face) and cached under
/mnt/workspace so restarts do not re-download.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import tempfile
import threading
import time
import traceback
from pathlib import Path

import gradio as gr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# "fp8" loads the official FP8 checkpoints (fits L20 48GB on-GPU);
# "bf16" falls back to BF16 repos with the text encoder kept on CPU.
PRECISION = os.environ.get("LLADA_PRECISION", "fp8").lower()
# "auto" | "triton" | "eager" — MoE kernel backend inside the remote code.
os.environ.setdefault("LLADA_MOE_BACKEND", os.environ.get("LLADA_MOE_BACKEND", "auto"))

VARIANTS = {
    "turbo": {
        "label": "Turbo · 4 步快速",
        "repos": {
            "fp8": "inclusionAI/LLaDA-Image-Turbo-FP8",
            "bf16": "inclusionAI/LLaDA-Image-Turbo",
        },
        "steps": 4,
        "guidance": 1.0,
        "hint": "4 步蒸馏模型，数秒至半分钟出图",
    },
    "base": {
        "label": "Base · 50 步高质量",
        "repos": {
            "fp8": "inclusionAI/LLaDA-Image-FP8",
            "bf16": "inclusionAI/LLaDA-Image",
        },
        "steps": 50,
        "guidance": 5.0,
        "hint": "50 步高质量模型 + CFG，单次生成预计分钟级",
    },
}
DEFAULT_VARIANT = "turbo"

PERSIST_ROOT = os.environ.get("LLADA_WORKSPACE", "/mnt/workspace")
try:
    os.makedirs(PERSIST_ROOT, exist_ok=True)
    if not os.access(PERSIST_ROOT, os.W_OK):
        raise PermissionError(PERSIST_ROOT)
except (OSError, PermissionError):
    PERSIST_ROOT = tempfile.gettempdir()

MODEL_CACHE = os.path.join(PERSIST_ROOT, "modelscope-cache")
CONVERT_CACHE = os.path.join(PERSIST_ROOT, "llada-converted")
os.makedirs(MODEL_CACHE, exist_ok=True)
os.makedirs(CONVERT_CACHE, exist_ok=True)
os.environ.setdefault("MODELSCOPE_CACHE", MODEL_CACHE)

# 官方仅约束整除性（text 模式 16 的倍数、editing 模式 32 的倍数，见 pipeline check_inputs），
# 无固定尺寸表；这里按总像素≈1MP（训练分辨率量级）取 32 倍数，覆盖常规宽高比。
SIZES_TEXT = [
    ("1024×1024 · 1:1", "1024x1024"),
    ("1312×736 · 16:9", "736x1312"),
    ("736×1312 · 9:16", "1312x736"),
    ("1184×896 · 4:3", "896x1184"),
    ("896×1184 · 3:4", "1184x896"),
    ("1024×800 · 8:5 横", "800x1024"),
    ("800×1024 · 5:8 竖", "1024x800"),
]
# Editing requires both dimensions divisible by 32; text/VQ by 16.
SIZES_EDIT = [
    ("1024×1024 · 1:1", "1024x1024"),
    ("1312×736 · 16:9", "736x1312"),
    ("736×1312 · 9:16", "1312x736"),
    ("1184×896 · 4:3", "896x1184"),
    ("896×1184 · 3:4", "1184x896"),
]


def _parse_size(text_hw: str) -> tuple[int, int]:
    h, w = text_hw.lower().split("x")
    return int(h), int(w)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_LOCK = threading.RLock()
_PIPELINES: dict[str, object] = {}
_ON_GPU: dict[str, bool] = {}
_MODEL_DIR: dict[str, Path] = {}
_STATUS: dict[str, str] = {"startup": "初始化中…"}
_READY = threading.Event()
TE_ON_CPU = False  # set True after an OOM fallback (bf16 strategy)
_LABEL2KEY = {v["label"]: k for k, v in VARIANTS.items()}


def _log(msg: str) -> None:
    print(f"[llada-studio] {msg}", flush=True)


def _downloaded_keys() -> set:
    """Variants whose weight snapshot is already complete in the persistent cache."""
    out = set()
    for key, v in VARIANTS.items():
        repo = v["repos"][PRECISION]
        try:
            path = Path(MODEL_CACHE) / repo.lower().replace("/", "__")
            if not (path / "model_index.json").exists():  # modelscope cache layout
                path = Path(MODEL_CACHE) / repo
            if (path / "model_index.json").exists():
                out.add(key)
        except OSError:  # noqa: BLE001 — never let status probing break the UI
            pass
    return out


def _split_fused_weight(key: str, w: "torch.Tensor") -> dict:  # noqa: F821
    """Unfuse FP8-checkpoint merged linears back to the official parameter names."""
    if key.endswith("attention.to_qkv.weight") and w.shape[0] == 3 * w.shape[1]:
        base = key[: -len("to_qkv.weight")]
        q, k_, v = w.split(w.shape[0] // 3, dim=0)
        return {
            base + "to_q.weight": q.contiguous(),
            base + "to_k.weight": k_.contiguous(),
            base + "to_v.weight": v.contiguous(),
        }
    if key.endswith("feed_forward.w13.weight") and w.shape[0] == 2 * (w.shape[0] // 2):
        base = key[: -len("w13.weight")]
        w1, w3 = w.split(w.shape[0] // 2, dim=0)
        return {
            base + "w1.weight": w1.contiguous(),
            base + "w3.weight": w3.contiguous(),
        }
    return {key: w}


def _snapshot(variant_key: str) -> Path:
    repo = VARIANTS[variant_key]["repos"][PRECISION]
    if repo in _MODEL_DIR:
        return _MODEL_DIR[repo]
    from modelscope import snapshot_download

    t0 = time.time()
    _log(f"downloading {repo} (cached under {MODEL_CACHE})")
    path = Path(snapshot_download(repo, cache_dir=MODEL_CACHE))
    _log(f"snapshot ready {repo} in {time.time() - t0:.1f}s -> {path}")
    _MODEL_DIR[repo] = path
    return path


def _dequant_block_fp8(w: "torch.Tensor", s: "torch.Tensor", block, out, in_):  # noqa: F821
    import torch

    bn, bk = block
    s_full = (
        s.float().repeat_interleave(bn, dim=0).repeat_interleave(bk, dim=1)[:out, :in_]
    )
    return w.float() * s_full


def _dequant_transformer(tf_dir: Path, dst_dir: Path) -> Path:
    """Convert the FP8 block-quantized DiT checkpoint to plain bf16.

    The FP8 checkpoint fuses linears at quantization time (`to_qkv` = q|k|v,
    `w13` = w1|w3) and stores `.weight` in float8_e4m3fn with per-128x128-block
    `.weight_scale_inv`. The official model code expects the *unfused* bf16
    names (matching the BF16 checkpoint index), so we dequantize and split
    shard by shard, writing a converted model directory once, cached on
    persistent disk.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if (dst_dir / "conversion.done").exists():
        return dst_dir
    tmp_dir = dst_dir.with_name(dst_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    cfg = json.loads((tf_dir / "config.json").read_text())
    block = (cfg.get("quantization_config") or {}).get("weight_block_size", [128, 128])
    cfg.pop("quantization_config", None)
    (tmp_dir / "config.json").write_text(json.dumps(cfg))

    index = json.loads((tf_dir / "diffusion_pytorch_model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = {}
    for shard in sorted(set(index["weight_map"].values())):
        out_sd: dict[str, torch.Tensor] = {}
        with safe_open(tf_dir / shard, framework="pt") as f:
            keys = list(f.keys())
            for k in keys:
                t = f.get_tensor(k)
                if k.endswith("_scale_inv"):
                    continue
                if t.dtype == torch.float8_e4m3fn:
                    s_key = k + "_scale_inv"
                    if s_key in index["weight_map"]:
                        with safe_open(tf_dir / index["weight_map"][s_key], framework="pt") as fs:
                            scale = fs.get_tensor(s_key)
                        w = _dequant_block_fp8(
                            t, scale, block, t.shape[0], t.shape[1]
                        ).to(torch.bfloat16)
                        _log(f"dequantized {k} {tuple(t.shape)}")
                    else:
                        w = t.to(torch.bfloat16)
                    parts = _split_fused_weight(k, w)
                    for pk, pv in parts.items():
                        out_sd[pk] = pv
                else:
                    out_sd[k] = t
        out_shard = shard  # keep the original shard file naming
        save_file(out_sd, str(tmp_dir / out_shard))
        for k in out_sd:
            weight_map[k] = out_shard
        out_sd.clear()
        gc.collect()

    total_size = sum(
        (tmp_dir / f).stat().st_size for f in set(weight_map.values())
    )
    (tmp_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map})
    )
    (tmp_dir / "conversion.done").write_text("ok")
    tmp_dir.rename(dst_dir)
    _log(f"transformer dequantized -> {dst_dir}")
    return dst_dir


def _load_pipeline(variant_key: str):
    import torch

    from src import LLaDAImagePipeline

    model_dir = _snapshot(variant_key)
    t0 = time.time()
    _STATUS[variant_key] = f"正在加载 {VARIANTS[variant_key]['label']} 模型…"
    _log(f"loading pipeline variant={variant_key} precision={PRECISION}")

    tf_cache = Path(CONVERT_CACHE) / f"{model_dir.name}-dit-bf16-v2"
    tf_load_dir = model_dir / "transformer"
    if PRECISION == "fp8":
        tf_cfg = json.loads((tf_load_dir / "config.json").read_text())
        if tf_cfg.get("quantization_config"):
            tf_load_dir = _dequant_transformer(tf_load_dir, tf_cache)

    # Load components explicitly (same order as the official from_pretrained),
    # keeping the huge text encoder on CPU when requested and everything else
    # on GPU.
    from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
    from transformers import AutoModel, AutoTokenizer

    from src.models import (
        LLaDAImageQueryFormerModel,
        LLaDAImageSigVQModel,
        LLaDAImageTextProjectionModel,
        LLaDAImageTransformer2DModel,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    te_on_cpu = TE_ON_CPU or PRECISION == "bf16"

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_dir / "scheduler")
    vae = AutoencoderKLFlux2.from_pretrained(model_dir / "vae", torch_dtype=dtype).to(device)
    text_encoder = AutoModel.from_pretrained(
        model_dir / "text_encoder", dtype=dtype, trust_remote_code=True
    )
    if not te_on_cpu and device == "cuda":
        text_encoder = text_encoder.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    queryformer = LLaDAImageQueryFormerModel.from_pretrained(
        model_dir / "queryformer", torch_dtype=dtype
    ).to(device)
    text_projection = LLaDAImageTextProjectionModel.from_pretrained(
        model_dir / "text_projection", torch_dtype=dtype
    ).to(device)
    sigvq = LLaDAImageSigVQModel.from_pretrained(
        model_dir / "sigvq", torch_dtype=dtype
    ).to(device)
    transformer = LLaDAImageTransformer2DModel.from_pretrained(
        tf_load_dir, torch_dtype=dtype
    ).to(device)

    pipe = LLaDAImagePipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        queryformer=queryformer,
        text_projection=text_projection,
        sigvq=sigvq,
        transformer=transformer,
    )
    _log(f"pipeline ready variant={variant_key} in {time.time() - t0:.1f}s")
    _STATUS[variant_key] = "已就绪"
    return pipe


def _get_pipeline(variant_key: str):
    """Return the GPU-resident pipeline for the variant, parking others on CPU.

    Each FP8 variant needs ~32 GiB of VRAM, so only one variant lives on the
    GPU at a time. Default unloads inactive variants entirely (xGPU free tier
    host has 64 GiB RAM; parking both variants on CPU would be borderline);
    set LLADA_KEEP_BOTH=1 on larger hosts to keep one parked variant for
    second-level switching.
    """
    keep_both = os.environ.get("LLADA_KEEP_BOTH", "0") != "0"
    with _LOCK:
        import torch

        # 1) free the GPU from other variants first
        for key in list(_PIPELINES):
            if key == variant_key or not _ON_GPU.get(key):
                continue
            _STATUS[key] = "已暂存于内存（切回时秒级恢复）" if keep_both else "已卸载"
            if keep_both:
                _PIPELINES[key].to("cpu")
            else:
                _PIPELINES[key].to("cpu")
                del _PIPELINES[key]
            _ON_GPU[key] = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2) load or restore the requested variant onto the GPU
        if variant_key not in _PIPELINES:
            pipe = _load_pipeline(variant_key)
            _PIPELINES[variant_key] = pipe
            _ON_GPU[variant_key] = True
            return pipe
        pipe = _PIPELINES[variant_key]
        if not _ON_GPU.get(variant_key) and torch.cuda.is_available() and not _cpu_text_encoder(pipe):
            t0 = time.time()
            _STATUS[variant_key] = "正在从内存恢复模型到 GPU…"
            pipe.to("cuda")
            _ON_GPU[variant_key] = True
            _log(f"restored {variant_key} to GPU in {time.time() - t0:.1f}s")
        return pipe


def _cpu_text_encoder(pipe) -> bool:
    """True when the text encoder deliberately stays on CPU (bf16/OOM mode)."""
    return str(next(pipe.text_encoder.parameters()).device) == "cpu"


def _warmup() -> None:
    try:
        if not shutil.which("nvidia-smi"):
            _STATUS["startup"] = "警告：未检测到 GPU，请在 xGPU 规格上运行本创空间。"
        _get_pipeline(DEFAULT_VARIANT)
        _STATUS["startup"] = "默认模型已就绪，可以开始生成。"
        _get_extra_variants()
    except Exception:  # noqa: BLE001
        _STATUS["startup"] = "模型加载失败，请查看运行日志。"
        _log(f"warmup failed:\n{traceback.format_exc()}")
    finally:
        _READY.set()  # surface the error through generation responses


def _get_extra_variants() -> None:
    # Other variants load lazily on first use; optionally prefetch behind a flag.
    if os.environ.get("LLADA_PREFETCH_ALL", "0") == "1":
        for key in VARIANTS:
            if key != DEFAULT_VARIANT:
                try:
                    _get_pipeline(key)
                except Exception:  # noqa: BLE001
                    _log(f"prefetch {key} failed:\n{traceback.format_exc()}")


threading.Thread(target=_warmup, daemon=True).start()

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _generate(variant_key: str, prompt: str, negative: str, size_hw, seed: int,
              steps: int, guidance: float, image=None):
    import torch

    if not prompt.strip() and image is None:
        return None, "请输入提示词。"
    if not _READY.is_set():
        return None, "模型仍在加载/下载中（首次需下载约 28 GB），请稍候再试。"
    height, width = _parse_size(size_hw)
    try:
        with _LOCK:
            pipe = _get_pipeline(variant_key)
            _log(f"generate variant={variant_key} mode={'editing' if image else 'text'} "
                 f"{height}x{width} steps={steps} cfg={guidance}")
            t0 = time.time()
            kwargs = {
                "prompt": prompt.strip() or None,
                "generation_mode": "editing" if image is not None else "text",
                "height": height,
                "width": width,
                "num_inference_steps": int(steps),
                "guidance_scale": float(guidance),
                "generator": torch.Generator(device=pipe.transformer.device).manual_seed(int(seed)),
            }
            if negative.strip():
                kwargs["negative_prompt"] = negative.strip()
            if image is not None:
                kwargs["image"] = image
            out = pipe(**kwargs)
            result = out.images[0]
        dt = time.time() - t0
        info = (f"{VARIANTS[variant_key]['label']} · {steps} 步 · {height}×{width} · "
                f"耗时 {dt:.1f}s · seed={seed}")
        _log(f"generated in {dt:.1f}s")
        return result, info
    except torch.cuda.OutOfMemoryError:
        global TE_ON_CPU
        if not TE_ON_CPU:
            TE_ON_CPU = True
            for pipe in _PIPELINES.values():
                try:
                    pipe.to("cpu")
                except Exception:  # noqa: BLE001
                    pass
            _PIPELINES.clear()
            _ON_GPU.clear()
            gc.collect()
            torch.cuda.empty_cache()
            _log("CUDA OOM: retrying once with text encoder on CPU")
            return _generate(variant_key, prompt, negative, size_hw, seed, steps, guidance, image)
        return None, "显存不足，且 CPU 文本编码器兜底仍失败。请换更小分辨率或申请更高规格 GPU。"
    except Exception:  # noqa: BLE001
        err = traceback.format_exc(limit=8)
        _log(f"generate failed:\n{err}")
        return None, f"生成失败：{type(err).__name__}（详见运行日志）"


def on_variant_change(label: str):
    key = _LABEL2KEY[label]
    v = VARIANTS[key]
    return (
        gr.update(value=label),
        gr.update(value=label),
        gr.update(value=v["steps"]),
        gr.update(value=v["guidance"]),
        gr.update(value=v["steps"]),
        gr.update(value=v["guidance"]),
    )


def status_tick():
    done = _downloaded_keys()
    rows = []
    for k, v in VARIANTS.items():
        if k in done:
            st = _STATUS.get(k)
            note = "已下载" if st is None or "就绪" in st else st.rstrip("…")
            light = "#22c55e"
        else:
            light, note = "#eab308", "下载中"
        rows.append(
            '<div style="display:flex;align-items:center;gap:8px;margin:2px 0">'
            f'<span style="width:11px;height:11px;border-radius:50%;background:{light};'
            'display:inline-block"></span>'
            f"<span>{v['label']}：{note}</span></div>"
        )
    return "".join(rows)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

HEADER = """
# LLaDA-Image · 统一图像生成与编辑
基于 **inclusionAI/LLaDA-Image** —— 6B 参数统一扩散模型，单权重同时支持文生图与指令编辑，
中英双语文本渲染能力突出。
"""

with gr.Blocks(title="LLaDA-Image", theme=gr.themes.Soft()) as demo:
    gr.Markdown(HEADER)
    with gr.Row():
        status_lights = gr.HTML("", label="模型状态")
    timer = gr.Timer(5.0)
    timer.tick(status_tick, outputs=status_lights)

    VARIANT_CHOICES = [v["label"] for v in VARIANTS.values()]
    DEFAULT_LABEL = VARIANTS[DEFAULT_VARIANT]["label"]

    with gr.Tabs():
        with gr.Tab("文本生成图像"):
            with gr.Row():
                with gr.Column(scale=3):
                    t_variant = gr.Radio(
                        VARIANT_CHOICES, value=DEFAULT_LABEL, label="生成档位",
                        container=True, scale=1,
                    )
                    t_prompt = gr.Textbox(
                        label="提示词", lines=3,
                        placeholder="例：A cinematic photograph of a red fox standing in fresh snow, soft winter light, detailed fur, shallow depth of field",
                    )
                    t_negative = gr.Textbox(label="负向提示词（可选）", lines=1)
                    with gr.Row():
                        t_size = gr.Dropdown(
                            choices=SIZES_TEXT, value="1024x1024", label="输出尺寸",
                        )
                        t_steps = gr.Slider(1, 50, value=VARIANTS[DEFAULT_VARIANT]["steps"], step=1, label="采样步数")
                        t_cfg = gr.Slider(1.0, 10.0, value=VARIANTS[DEFAULT_VARIANT]["guidance"], step=0.1, label="提示词引导 (CFG)")
                    t_seed = gr.Number(value=42, precision=0, label="随机种子 seed")
                    t_btn = gr.Button("生成图像", variant="primary")
                with gr.Column(scale=2):
                    t_out = gr.Image(type="pil", label="生成结果")
            t_info = gr.Markdown()

        with gr.Tab("图像编辑"):
            with gr.Row():
                with gr.Column(scale=3):
                    e_variant = gr.Radio(
                        VARIANT_CHOICES, value=DEFAULT_LABEL, label="生成档位",
                        container=True, scale=1,
                    )
                    e_image = gr.Image(type="pil", label="参考图")
                    e_prompt = gr.Textbox(
                        label="编辑指令", lines=2,
                        placeholder="例：Turn it into a watercolor painting",
                    )
                    with gr.Row():
                        e_size = gr.Dropdown(
                            choices=SIZES_EDIT, value="1024x1024", label="输出尺寸",
                        )
                        e_steps = gr.Slider(1, 50, value=VARIANTS[DEFAULT_VARIANT]["steps"], step=1, label="采样步数")
                        e_cfg = gr.Slider(1.0, 10.0, value=VARIANTS[DEFAULT_VARIANT]["guidance"], step=0.1, label="提示词引导 (CFG)")
                    e_seed = gr.Number(value=43, precision=0, label="随机种子 seed")
                    e_btn = gr.Button("执行编辑", variant="primary")
                with gr.Column(scale=2):
                    e_out = gr.Image(type="pil", label="编辑结果")
            e_info = gr.Markdown()

    t_btn.click(
        lambda label, p, n, s, sd, st, cg: _generate(
            _LABEL2KEY[label], p, n, s, sd, st, cg
        ),
        inputs=[t_variant, t_prompt, t_negative, t_size, t_seed, t_steps, t_cfg],
        outputs=[t_out, t_info],
        api_name="text_to_image",
    )
    e_btn.click(
        lambda label, im, p, s, sd, st, cg: _generate(
            _LABEL2KEY[label], p, "", s, sd, st, cg, image=im
        ),
        inputs=[e_variant, e_image, e_prompt, e_size, e_seed, e_steps, e_cfg],
        outputs=[e_out, e_info],
        api_name="image_editing",
    )
    for src in (t_variant, e_variant):
        src.change(
            on_variant_change,
            inputs=src,
            outputs=[t_variant, e_variant, t_steps, t_cfg, e_steps, e_cfg],
        )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
    )
