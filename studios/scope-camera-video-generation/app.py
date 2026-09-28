"""SCoPE — camera-trajectory controlled image-to-video (Wan2.2-I2V-A14B + SCoPE).

Faithful port of TencentARC/SCoPE's reference inference path (scope/inference.py,
scope/weights.py, the vendored DiffSynth `wan_video_panshot` pipeline) onto ZeroGPU.

Deviations from the reference are forced by the 48 GB / ~2 min ZeroGPU budget and are
listed in the README: fp8 weight quantization, the Wan2.2-Lightning 4-step distillation
LoRA with cfg_scale = 1.0, and shard-streamed weight loading.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_persist_root = os.environ.get("MODELSCOPE_PERSIST_ROOT", "/mnt/workspace")
if not os.path.isdir(_persist_root):
    _persist_root = "/tmp"
os.environ.setdefault("MODELSCOPE_CACHE", os.path.join(_persist_root, "modelscope-cache"))

import gc  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from io import BytesIO  # noqa: E402
from pathlib import Path  # noqa: E402

import gradio as gr  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from modelscope import snapshot_download  # noqa: E402
from modelscope.hub.file_download import model_file_download  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors import safe_open  # noqa: E402
from torchao.quantization import (  # noqa: E402
    Float8DynamicActivationFloat8WeightConfig,
    Int8WeightOnlyConfig,
    quantize_,
)
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration  # noqa: E402

from diffsynth.data.video import save_video  # noqa: E402
from diffsynth.models import ModelManager  # noqa: E402
from diffsynth.models.utils import init_weights_on_device  # noqa: E402
from diffsynth.models.wan_video_dit import WanModel  # noqa: E402
from scope.config import InferenceConfig  # noqa: E402
from scope.pipeline import SCoPEPipeline  # noqa: E402
from scope.weights import _DIT_CONFIG, _install_scope_architecture  # noqa: E402

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO_ID = "TencentARC/SCoPE"
LORA_REPO = "lightx2v/Wan2.2-Lightning"
LORA_SUBDIR = "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1"
# A small vision-language model used to auto-write a caption when the user leaves the
# scene description blank — SCoPE always needs a caption for the content it should draw.
CAPTIONER_REPO = "Qwen/Qwen2-VL-2B-Instruct"
CAPTION_INSTRUCTION = (
    "Write one vivid paragraph describing this image for a video generation model. "
    "Cover the main subjects, the setting, materials and colours, and the lighting. "
    "Do not mention the camera, camera motion, shot type, or the word 'image'."
)
PERSIST_ROOT = Path(_persist_root)
WORK_DIR = Path(os.environ.get("SCOPE_WEIGHT_DIR", str(PERSIST_ROOT / "scope-weights")))
MODELSCOPE_CACHE = Path(
    os.environ.get("MODELSCOPE_CACHE", str(PERSIST_ROOT / "modelscope-cache"))
)
MODEL_REVISION = os.environ.get("MODELSCOPE_MODEL_REVISION", "master")
DEVICE = "cuda"

CFG = InferenceConfig()  # 480x832, 81 frames, fps 16, sigma_shift 5.0, boundary 0.9
HEIGHT, WIDTH, NUM_FRAMES, FPS = CFG.height, CFG.width, CFG.num_frames, CFG.fps
# The ModelScope 48 GB xGPU exposes 44.64 GiB. Spatial VAE tiling keeps the
# 81-frame conditioning encode/decode below the remaining activation budget.
VAE_TILE_SIZE = (30, 52)
VAE_TILE_STRIDE = (15, 26)
INFERENCE_LOCK = threading.Lock()
# Fused+quantised expert checkpoints persisted from a previous cold start. Re-loading
# them on restart skips the shard downloads, LoRA fusion and fp8 quantisation (~10 min).
FP8_CACHE_DIR = Path(
    os.environ.get("SCOPE_FP8_CACHE_DIR", str(PERSIST_ROOT / "scope-fp8-cache"))
)
FP8_CACHE_VERSION = 1  # bump whenever the fusion/quantisation recipe changes

# The reference `x_fov` for every AI-generated showcase case in examples/manifest.json.
DEFAULT_FOV_DEG = round(math.degrees(1.4078388214111328), 1)  # 80.7 deg
MAX_SEED = np.iinfo(np.int32).max

NEGATIVE_PROMPT = (HERE / "configs" / "negative_prompt.txt").read_text(encoding="utf-8").strip()

# Camera presets. Every .npy is [81, 3, 4] OpenCV camera-to-world, already expressed
# relative to frame 0 (frame 0 is the identity pose), matching what SCoPE was trained on.
PRESETS: list[tuple[str, str]] = [
    ("Dolly in — push straight into the scene", "dolly_in"),
    ("Dolly out — pull straight back", "dolly_out"),
    ("Truck left — slide sideways to the left", "truck_left"),
    ("Truck right — slide sideways to the right", "truck_right"),
    ("Orbit left — arc around the subject", "orbit_left"),
    ("Crane up + forward — rise while pushing in", "crane_up_fwd"),
    ("Snake forward — weaving push-in", "snake_fwd"),
    ("Grand tour — long sweeping traversal (bold)", "grand_tour"),
    ("Push + sweep — drive in, then sweep across (bold)", "push_sweep"),
    ("Wide orbit — large arc around the scene (bold)", "wide_orbit"),
    ("Spiral climb — rising corkscrew (bold)", "spiral_climb"),
    ("Spiral rise — steep rising turn (bold)", "greek_spiral_rise"),
    ("Crane arc — lift and curve (bold)", "crane_arc"),
    ("Flyover left — fly past on the left (bold)", "flyover_left"),
    ("S-curve reveal — weave and reveal (bold)", "s_curve_reveal"),
    ("Pull back + rise — retreat and lift (bold)", "pullback_rise"),
]
PRESET_LABELS = {value: label for label, value in PRESETS}


def load_trajectory(name: str, motion_scale: float = 1.0) -> np.ndarray:
    """Load a [81, 3, 4] camera-to-world preset and optionally rescale its translation."""
    path = HERE / "trajectories" / f"{name}.npy"
    if not path.is_file():
        raise gr.Error(f"Unknown camera trajectory: {name}")
    poses = np.load(path).astype(np.float32)
    if poses.shape != (NUM_FRAMES, 3, 4):
        raise gr.Error(f"Malformed trajectory {name}: {poses.shape}")
    poses = poses.copy()
    poses[:, :3, 3] *= float(motion_scale)
    return poses


# --------------------------------------------------------------------------------------
# Weight loading — streamed shard by shard so peak disk stays ~1 shard (the full
# TencentARC/SCoPE package is 71 GB, well over a Space's ephemeral disk).
# --------------------------------------------------------------------------------------

FP8_CONFIG = Float8DynamicActivationFloat8WeightConfig()


def _quant_filter(module: torch.nn.Module, fqn: str) -> bool:
    """fp8 the big projections only; SCoPE's tiny Plucker/gate MLPs stay bf16."""
    return (
        isinstance(module, torch.nn.Linear)
        and "plucker_pe" not in fqn
        and module.in_features >= 512
        and module.out_features >= 512
    )


def _download(filename: str, repo_id: str = REPO_ID) -> Path:
    return Path(
        model_file_download(
            repo_id,
            filename,
            revision=MODEL_REVISION,
            local_dir=str(WORK_DIR),
        )
    )


def _load_lightning_lora(expert: str) -> dict[str, tuple[torch.Tensor, torch.Tensor, float]]:
    """Read the Wan2.2-Lightning 4-step LoRA for one expert as {param_name: (down, up, scale)}."""
    path = _download(f"{LORA_SUBDIR}/{expert}.safetensors", repo_id=LORA_REPO)
    table: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if not key.endswith(".lora_down.weight"):
                continue
            stem = key[: -len(".lora_down.weight")]
            down = handle.get_tensor(key).clone()
            up = handle.get_tensor(f"{stem}.lora_up.weight").clone()
            alpha = float(handle.get_tensor(f"{stem}.alpha"))
            target = stem.replace("diffusion_model.", "", 1) + ".weight"
            table[target] = (down, up, alpha / down.shape[0])
    path.unlink(missing_ok=True)
    print(f"[SCoPE] Lightning LoRA ({expert}): {len(table)} fused projections", flush=True)
    return table


def _fuse_lora(module: torch.nn.Module, table: dict, prefix: str) -> int:
    fused = 0
    for name, param in module.named_parameters(recurse=True):
        entry = table.pop(f"{prefix}{name}", None)
        if entry is None:
            continue
        down, up, scale = entry
        delta = torch.mm(up.float(), down.float()).mul_(scale)
        param.data = (param.data.float() + delta).to(torch.bfloat16)
        del delta, down, up
        fused += 1
    return fused


def _block_materialized(block: torch.nn.Module) -> bool:
    tensors = list(block.parameters(recurse=True)) + list(block.buffers(recurse=True))
    return all(not tensor.is_meta for tensor in tensors)


def _finalize_block(block: torch.nn.Module, index: int, table: dict, is_low_expert: bool) -> None:
    _fuse_lora(block, table, f"blocks.{index}.")
    encoding = block.self_attn.plucker_pe
    q_out = encoding.eq[2] if encoding.use_mlp else encoding.eq
    nonzero = int(torch.count_nonzero(q_out.weight))
    if is_low_expert and nonzero != 0:
        raise RuntimeError(f"low-noise expert block {index} is not a zero-delta SCoPE model")
    if not is_low_expert and nonzero == 0:
        raise RuntimeError(f"high-noise expert block {index} has no SCoPE weights")
    block.requires_grad_(False)
    block.to(DEVICE)
    quantize_(block, FP8_CONFIG, filter_fn=_quant_filter)


def _stream_expert(model: WanModel, subfolder: str, is_low_expert: bool) -> None:
    """Materialise one 29.7 GB expert: download -> assign -> delete -> fuse -> fp8."""
    lora_table = _load_lightning_lora("low_noise_model" if is_low_expert else "high_noise_model")
    index_path = _download(f"{subfolder}/diffusion_pytorch_model.safetensors.index.json")
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    shards = list(dict.fromkeys(weight_map.values()))

    expected = set(model.state_dict())
    loaded: set[str] = set()
    pending = set(range(len(model.blocks)))

    for position, shard in enumerate(shards, start=1):
        started = time.time()
        shard_path = _download(f"{subfolder}/{shard}")
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                # clone(): safetensors hands back mmap views, and the file is deleted below.
                tensors[key] = handle.get_tensor(key).clone()
        unexpected = set(tensors) - expected
        if unexpected:
            raise RuntimeError(f"unexpected keys in {shard}: {sorted(unexpected)[:5]}")
        model.load_state_dict(tensors, strict=False, assign=True)
        loaded.update(tensors)
        del tensors
        shard_path.unlink(missing_ok=True)
        gc.collect()

        for index in sorted(pending):
            if _block_materialized(model.blocks[index]):
                _finalize_block(model.blocks[index], index, lora_table, is_low_expert)
                pending.discard(index)
        gc.collect()
        print(
            f"[SCoPE] {subfolder}: shard {position}/{len(shards)} in "
            f"{time.time() - started:.0f}s, {len(model.blocks) - len(pending)}"
            f"/{len(model.blocks)} blocks quantised",
            flush=True,
        )

    missing = expected - loaded
    if missing:
        raise RuntimeError(f"incomplete {subfolder}: {sorted(missing)[:5]}")
    if pending:
        raise RuntimeError(f"{subfolder}: blocks never materialised: {sorted(pending)[:5]}")
    if lora_table:
        raise RuntimeError(f"unused Lightning LoRA keys: {sorted(lora_table)[:5]}")

    # Everything outside `blocks` (patch/text/time embeddings, head) is small.
    for name, child in model.named_children():
        if name == "blocks":
            continue
        child.requires_grad_(False)
        child.to(DEVICE)
        quantize_(child, FP8_CONFIG, filter_fn=_quant_filter)
    for _, param in model.named_parameters(recurse=False):
        param.data = param.data.to(DEVICE)
    leftover = [name for name, p in model.named_parameters() if p.is_meta]
    if leftover:
        raise RuntimeError(f"unmaterialised parameters: {leftover[:5]}")
    gc.collect()


def _expert_cache_path(name: str) -> Path:
    return FP8_CACHE_DIR / f"{name}.v{FP8_CACHE_VERSION}.pt"


def _try_load_expert_cache(model: WanModel, name: str) -> bool:
    """Restore a fully fused+quantised expert from /mnt/workspace, if cached."""
    path = _expert_cache_path(name)
    if not path.is_file():
        return False
    started = time.time()
    state = torch.load(path, map_location="cpu", weights_only=False)
    if set(state) != set(model.state_dict()):
        print(f"[SCoPE] {name}: fp8 cache key mismatch, falling back to streaming", flush=True)
        return False
    model.load_state_dict(state, strict=True, assign=True)
    del state
    gc.collect()
    leftover = [n for n, p in model.named_parameters() if p.is_meta]
    if leftover:
        raise RuntimeError(f"{name}: cache left meta parameters: {leftover[:5]}")
    model.requires_grad_(False)
    model.to(DEVICE)
    print(f"[SCoPE] {name}: fp8 cache loaded in {time.time() - started:.0f}s", flush=True)
    return True


def _save_expert_cache(model: WanModel, name: str) -> None:
    """Best-effort persist of the quantised expert; never blocks startup on failure."""
    try:
        FP8_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        target = _expert_cache_path(name)
        tmp = target.with_suffix(".tmp")
        torch.save(model.state_dict(), tmp)
        tmp.replace(target)
        print(f"[SCoPE] {name}: fp8 cache saved to {target}", flush=True)
    except Exception as exc:
        print(f"[SCoPE] {name}: fp8 cache save failed: {exc}", flush=True)


def build_pipeline() -> SCoPEPipeline:
    total = time.time()
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    pipe = SCoPEPipeline(device="cpu", torch_dtype=torch.bfloat16)
    with init_weights_on_device():
        pipe.dit = WanModel(**_DIT_CONFIG)
        pipe.dit2 = WanModel(**_DIT_CONFIG)
        _install_scope_architecture(pipe, CFG)

    # T5 + VAE first: the .pth loader is not mmap-based, so get its 11 GB peak out of
    # the way before the experts occupy RAM.
    for filename in (
        "google/umt5-xxl/spiece.model",
        "google/umt5-xxl/special_tokens_map.json",
        "google/umt5-xxl/tokenizer.json",
        "google/umt5-xxl/tokenizer_config.json",
    ):
        _download(filename)
    manager = ModelManager(torch_dtype=torch.bfloat16, device=DEVICE)
    for filename in ("models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth"):
        path = _download(filename)
        manager.load_model(str(path))
        path.unlink(missing_ok=True)
        gc.collect()
    pipe.text_encoder = manager.fetch_model("wan_video_text_encoder")
    pipe.vae = manager.fetch_model("wan_video_vae")
    if pipe.text_encoder is None or pipe.vae is None:
        raise RuntimeError("the SCoPE package must ship both the T5 encoder and the VAE")
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    quantize_(pipe.text_encoder, Int8WeightOnlyConfig())
    gc.collect()

    pipe.prompter.fetch_models(pipe.text_encoder)
    pipe.prompter.fetch_tokenizer(str(WORK_DIR / "google" / "umt5-xxl"))

    for model, name, is_low in (
        (pipe.dit, "high_noise_model", False),
        (pipe.dit2, "low_noise_model", True),
    ):
        if not _try_load_expert_cache(model, name):
            _stream_expert(model, name, is_low_expert=is_low)
            _save_expert_cache(model, name)

    pipe.height_division_factor = pipe.vae.upsampling_factor * 2
    pipe.width_division_factor = pipe.vae.upsampling_factor * 2
    pipe.switch_DiT_boundary = CFG.switch_dit_boundary
    pipe.device = DEVICE
    pipe.eval()
    gc.collect()
    print(f"[SCoPE] pipeline ready in {time.time() - total:.0f}s", flush=True)
    return pipe


PIPE = build_pipeline()


# --------------------------------------------------------------------------------------
# Optional prompt writer — Qwen2-VL captions the first frame when no prompt is given.
# Downloaded and held on CPU at build time; only lifted onto the GPU for the few seconds
# it is actually needed, then evicted so it never competes with the two 14B experts.
# --------------------------------------------------------------------------------------


def build_captioner() -> tuple[Qwen2VLForConditionalGeneration, AutoProcessor]:
    started = time.time()
    captioner_dir = snapshot_download(
        CAPTIONER_REPO,
        revision=MODEL_REVISION,
        cache_dir=str(MODELSCOPE_CACHE),
    )
    processor = AutoProcessor.from_pretrained(
        captioner_dir, min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        captioner_dir, torch_dtype=torch.bfloat16
    )
    model.requires_grad_(False)
    model.eval()
    print(f"[SCoPE] caption model ready in {time.time() - started:.0f}s", flush=True)
    return model, processor


CAPTIONER, CAPTION_PROCESSOR = build_captioner()


def autocaption(image: Image.Image) -> str:
    """Describe `image` with Qwen2-VL so a promptless request still has scene content."""
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": CAPTION_INSTRUCTION}],
        }
    ]
    text = CAPTION_PROCESSOR.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = CAPTION_PROCESSOR(text=[text], images=[image], return_tensors="pt")
    CAPTIONER.to(DEVICE)
    try:
        inputs = inputs.to(DEVICE)
        with torch.inference_mode():
            generated = CAPTIONER.generate(**inputs, max_new_tokens=220, do_sample=False)
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        caption = CAPTION_PROCESSOR.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()
    finally:
        CAPTIONER.to("cpu")
        torch.cuda.empty_cache()
    return caption


# --------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------


def prepare_image(path: str | None) -> Image.Image:
    if not path:
        raise gr.Error("Please provide an input image — it becomes the first video frame.")
    image = Image.open(path).convert("RGB")
    target = WIDTH / HEIGHT
    width, height = image.size
    if abs(width / height - target) > 1e-3:
        # Centre-crop to 16:9 first so non-16:9 uploads are not squashed.
        if width / height > target:
            crop = int(round(height * target))
            left = (width - crop) // 2
            image = image.crop((left, 0, left + crop, height))
        else:
            crop = int(round(width / target))
            top = (height - crop) // 2
            image = image.crop((0, top, width, top + crop))
    return image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


def _generate_unlocked(
    image=None,
    prompt="",
    trajectory="dolly_in",
    steps=4,
    motion_scale=1.0,
    fov_degrees=DEFAULT_FOV_DEG,
    seed=42,
    randomize_seed=True,
    progress=gr.Progress(track_tqdm=True),
):
    first_frame = prepare_image(image)
    prompt = (prompt or "").strip()
    autocaptioned = False
    if not prompt:
        # SCoPE needs a caption for the content, so write one from the frame instead of
        # failing on an empty prompt.
        prompt = autocaption(first_frame)
        autocaptioned = True
        if not prompt:
            raise gr.Error("Could not caption the image automatically — please add a description.")

    used_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    poses = load_trajectory(trajectory, motion_scale)
    camera = {
        "pose": torch.from_numpy(poses)[None].to(device=DEVICE, dtype=PIPE.torch_dtype),
        "x_fov": torch.tensor(
            [math.radians(float(fov_degrees))], device=DEVICE, dtype=PIPE.torch_dtype
        ),
        "xi": torch.tensor([0.0], device=DEVICE, dtype=PIPE.torch_dtype),
    }

    started = time.time()
    try:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        ):
            frames = PIPE(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                input_image=first_frame,
                camera_control_panshot=camera,
                seed=used_seed,
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
                num_inference_steps=int(steps),
                sigma_shift=CFG.sigma_shift,
                cfg_scale=1.0,  # distilled 4-step LoRA is guidance-free
                camera_cfg_scale=1.0,
                switch_DiT_boundary=CFG.switch_dit_boundary,
                lock_first_frame=False,
                tiled=True,
                tile_size=VAE_TILE_SIZE,
                tile_stride=VAE_TILE_STRIDE,
            )
    finally:
        # Release failed-request activations as well as allocator cache before
        # the next xGPU invocation.
        torch.cuda.empty_cache()
    elapsed = time.time() - started
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    print(f"[SCoPE] generated in {elapsed:.0f}s, peak VRAM {peak_gib:.1f} GiB", flush=True)

    output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    output.close()
    save_video(frames, output.name, fps=FPS, quality=9)
    status = (
        f"{int(steps)} steps · seed {used_seed} · {PRESET_LABELS.get(trajectory, trajectory)} "
        f"· motion x{motion_scale:g} · {elapsed:.0f}s"
    )
    if autocaptioned:
        status += f"\n\n*Auto-generated prompt (Qwen2-VL):* {prompt}"
    return output.name, used_seed, status


def generate(
    image=None,
    prompt="",
    trajectory="dolly_in",
    steps=4,
    motion_scale=1.0,
    fov_degrees=DEFAULT_FOV_DEG,
    seed=42,
    randomize_seed=True,
    progress=gr.Progress(track_tqdm=True),
):
    """Serialize access to the stateful Wan VAE and the single xGPU."""
    with INFERENCE_LOCK:
        return _generate_unlocked(
            image=image,
            prompt=prompt,
            trajectory=trajectory,
            steps=steps,
            motion_scale=motion_scale,
            fov_degrees=fov_degrees,
            seed=seed,
            randomize_seed=randomize_seed,
            progress=progress,
        )


# --------------------------------------------------------------------------------------
# Camera path preview (CPU only)
# --------------------------------------------------------------------------------------


def preview_path(trajectory: str, motion_scale: float) -> Image.Image:
    poses = load_trajectory(trajectory, motion_scale)
    # OpenCV camera axes are (right, down, forward); plot as (right, forward, up).
    xs, ys, zs = poses[:, 0, 3], poses[:, 2, 3], -poses[:, 1, 3]
    us, vs, ws = poses[:, 0, 2], poses[:, 2, 2], -poses[:, 1, 2]

    # Equal aspect on every axis (so the shape of the move is honest) but centred on the
    # path itself rather than the origin, so short moves still fill the frame.
    stacked = np.stack([xs, ys, zs])
    half = max(float((stacked.max(axis=1) - stacked.min(axis=1)).max()) * 0.65, 0.3)
    centre = (stacked.max(axis=1) + stacked.min(axis=1)) / 2.0

    figure = plt.figure(figsize=(4.4, 3.6), dpi=150)
    axes = figure.add_subplot(111, projection="3d")
    axes.plot(xs, ys, zs, color="#2563eb", linewidth=2)
    axes.scatter([xs[0]], [ys[0]], [zs[0]], color="#16a34a", s=30, label="start")
    axes.scatter([xs[-1]], [ys[-1]], [zs[-1]], color="#dc2626", s=30, label="end")
    step = 8
    axes.quiver(
        xs[::step], ys[::step], zs[::step],
        us[::step], vs[::step], ws[::step],
        length=half * 0.45, normalize=True, color="#94a3b8", linewidth=0.9,
        arrow_length_ratio=0.35, label="sightline",
    )

    axes.set_xlim(centre[0] - half, centre[0] + half)
    axes.set_ylim(centre[1] - half, centre[1] + half)
    axes.set_zlim(centre[2] - half, centre[2] + half)
    axes.set_box_aspect((1.0, 1.0, 1.0))
    axes.set_xlabel("right", fontsize=7, labelpad=-8)
    axes.set_ylabel("forward", fontsize=7, labelpad=-8)
    axes.set_zlabel("up", fontsize=7, labelpad=-8)
    axes.set_xticklabels([])
    axes.set_yticklabels([])
    axes.set_zticklabels([])
    axes.tick_params(length=0, pad=-2)
    axes.set_title(PRESET_LABELS.get(trajectory, trajectory).split(" — ")[0], fontsize=9)
    axes.legend(fontsize=7, loc="upper left", frameon=False)
    figure.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    buffer = BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(figure)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------

EXAMPLES = [
    [
        str(HERE / "examples" / "istock-motorbike-rice-field.jpg"),
        "A serene rural landscape in soft early-morning light with long shadows. A dirt road "
        "winds through a lush green area; on the left, dense clusters of tall trees including palms "
        "and a few small houses with red roofs, and on the right a vast expanse of green rice "
        "paddies neatly divided into sections. A motorcycle travels along the dirt road, flanked by "
        "utility poles with wires running along them. The overall atmosphere is peaceful and "
        "idyllic, with greenery and natural elements dominating the scene.",
        "orbit_left",
    ],
    [
        str(HERE / "examples" / "omni-greek-square.jpg"),
        "A lively ancient Greek market square with stone-paved pathways, wooden crates, and "
        "fabric-draped stalls full of clay pottery. The setting features classical architecture, "
        "including a prominent colonnaded building on the left and a vast cityscape stretching into "
        "the distance. The clear weather and bright sunlight enhance the vividness of the scene, "
        "creating a dynamic and immersive atmosphere.",
        "crane_up_fwd",
    ],
    [
        str(HERE / "examples" / "istock-country-road.jpg"),
        "A white car driving on a winding road surrounded by lush green grass, seen from an aerial "
        "perspective. The road is narrow and curves gently through the landscape. The grass on "
        "either side is vibrant and well-maintained, with some patches of darker green. The overall "
        "scene is serene and picturesque, with the car travelling through the greenery.",
        "dolly_in",
    ],
    [
        str(HERE / "examples" / "istock-skier.jpg"),
        "A snowy mountain landscape under a clear blue sky. A person is skiing down a well-groomed "
        "slope, leaving tracks in the snow. The skier is dressed in dark clothing and is using ski "
        "poles for balance. In the background, there are snow-covered mountains with rocky "
        "outcrops. A ski lift with red support towers is visible to the left, and a small building "
        "is seen at the bottom of the slope.",
        "orbit_left",
    ],
    [
        str(HERE / "examples" / "ai-airmountains.jpg"),
        "A vast sky filled with multiple floating islands of different sizes, suspended above a "
        "dense cloud layer. Each island has distinct terrain such as cliffs, forests, stone ruins, "
        "and grassy plateaus. Large waterfalls fall from the edges of islands into the clouds "
        "below, creating vertical movement through space. Bright daylight above the cloud sea with "
        "soft volumetric haze. Cinematic fantasy realism, natural lighting, subtle atmospheric "
        "scattering.",
        "grand_tour",
    ],
    [
        str(HERE / "examples" / "ai-valley.jpg"),
        "A wide alpine valley surrounded by tall snow-covered mountains. In the center, a calm "
        "lake reflects the sky and surrounding peaks. The valley floor contains open grasslands, "
        "scattered pine forests, rocky slopes, and small villages connected by winding dirt roads. "
        "A river flows from the mountains through the valley into the lake. Soft morning sunlight "
        "with atmospheric haze in the far mountains. Realistic natural environment, subtle "
        "cinematic tone, physically based rendering.",
        "crane_arc",
    ],
    [
        str(HERE / "examples" / "ai-middleages.jpg"),
        "A vast medieval valley with rolling green hills and a winding river flowing through the "
        "landscape. A stone bridge connects two small villages built along the riverbanks, with "
        "wooden houses, farms, and scattered windmills. In the distance, a large stone castle sits "
        "on top of a hill surrounded by forests, with mountain ranges extending far into the "
        "horizon. Soft daylight with mild shadows and natural atmospheric perspective. Unreal "
        "Engine 5 style, realistic rendering, subtle cinematic lighting.",
        "push_sweep",
    ],
]

# Scenes that contain a person / character. Kept in their own group so the showcase above
# stays character-free, per the release curation.
CHARACTER_EXAMPLES = [
    [
        str(HERE / "examples" / "omni-misty-forest.jpg"),
        "A character in red armor and a straw hat progresses along a forest path, their steps "
        "deliberate as they navigate over stones and through patches of grass. The environment is "
        "a misty forest with ancient stone structures on the left and moss-covered cliffs on the "
        "right. The dense fog and surrounding greenery contribute to a mysterious ambiance.",
        "orbit_left",
    ],
    [
        str(HERE / "examples" / "omni-horse-trail.jpg"),
        "A character dressed in dark attire rides a white horse steadily along a rugged dirt path "
        "that meanders through rocky terrain interspersed with patches of grass and shrubs. To the "
        "left, a wooden fence lines the trail. In the distance, the landscape opens up to reveal "
        "rolling hills covered in vegetation and distant mountains under a bright sky.",
        "crane_up_fwd",
    ],
]

# A compact image-only strip for the left rail — every showcase frame, in the same order.
# Clicking a thumbnail loads its image, prompt and camera move into the input boxes.
QUICK_PICKS = CHARACTER_EXAMPLES + EXAMPLES
QUICK_PICK_GALLERY = [
    (row[0], Path(row[0]).stem.replace("istock-", "").replace("ai-", "").replace("omni-", "").replace("-", " "))
    for row in QUICK_PICKS
]


def load_quick_pick(event: gr.SelectData) -> tuple[str, str, str]:
    image, prompt, trajectory = QUICK_PICKS[event.index]
    return image, prompt, trajectory


CSS = """
#col-container { margin: 0 auto; max-width: 1180px; }
.dark .gradio-container { color: var(--body-text-color); }
"""

# ModelScope brand purple (#624aff, rgb(98, 74, 255)) sampled from the modelscope.cn
# site stylesheet; shades built around it so every Gradio accent matches the hub.
MS_PURPLE = gr.themes.Color(
    c50="#f6f4ff", c100="#edeaff", c200="#dbd4ff", c300="#c0b1ff",
    c400="#a18aff", c500="#8166ff", c600="#624aff", c700="#5238d8",
    c800="#432eb0", c900="#38298d", c950="#221856",
)
MS_THEME = gr.themes.Citrus(primary_hue=MS_PURPLE, secondary_hue=MS_PURPLE)

with gr.Blocks() as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# SCoPE — steer the camera through a still image")
        with gr.Row():
            with gr.Column(scale=1, min_width=110):
                quick_picks = gr.Gallery(
                    value=QUICK_PICK_GALLERY,
                    label="Quick picks — click to load",
                    columns=1,
                    height=600,
                    object_fit="cover",
                    allow_preview=False,
                )
            with gr.Column(scale=5):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="First frame", type="filepath", height=300,
                            sources=["upload", "clipboard"],
                        )
                        prompt_input = gr.Textbox(
                            label="Scene description",
                            placeholder="Describe what is in the image… (leave empty to auto-caption it)",
                            lines=4,
                            info="Optional — if left blank, Qwen2-VL writes a caption from your image.",
                        )
                        trajectory_input = gr.Dropdown(
                            label="Camera move", choices=PRESETS, value="dolly_in",
                        )
                        run_button = gr.Button("Generate video", variant="primary")
                    with gr.Column(scale=1):
                        video_output = gr.Video(
                            label="Generated video", autoplay=True, loop=True, height=300
                        )
                        path_preview = gr.Image(
                            label="Camera path (start green, end red)",
                            height=280,
                            interactive=False,
                        )
                        status_output = gr.Markdown()

        with gr.Accordion("Advanced settings", open=False):
            with gr.Row():
                steps_input = gr.Slider(
                    label="Sampling steps",
                    minimum=4,
                    maximum=16,
                    step=1,
                    value=4,
                    info=(
                        "The distillation LoRA is trained for 4 steps (2 high-noise + "
                        "2 low-noise) — about 68s. Each extra step adds ~14s of GPU time; "
                        "gains past ~8 steps are marginal for the distilled LoRA."
                    ),
                )
                motion_input = gr.Slider(
                    label="Camera motion scale",
                    minimum=0.25,
                    maximum=2.0,
                    step=0.05,
                    value=1.0,
                    info="Multiplies the preset's translation. 1.0 is the authored path.",
                )
            with gr.Row():
                fov_input = gr.Slider(
                    label="Horizontal field of view (degrees)",
                    minimum=40.0,
                    maximum=110.0,
                    step=0.1,
                    value=DEFAULT_FOV_DEG,
                    info="Camera intrinsics used to build the Plücker rays.",
                )
                seed_input = gr.Slider(
                    label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=CFG.seed
                )
                randomize_input = gr.Checkbox(label="Randomize seed", value=True)

        gr.Examples(
            examples=CHARACTER_EXAMPLES,
            inputs=[image_input, prompt_input, trajectory_input],
            label="Camera control with people in the scene — click a row to load it",
            examples_per_page=8,
        )
        gr.Examples(
            examples=EXAMPLES,
            inputs=[image_input, prompt_input, trajectory_input],
            label="More examples — click a row to load its image, prompt and camera move",
            examples_per_page=8,
        )

        gr.Markdown(
            """
            **Notes** · Camera paths are OpenCV camera-to-world matrices `[81, 3, 4]` relative to
            the first frame, exactly the format SCoPE was trained on — the presets are taken from
            the release's own `examples/` trajectory set. Non-16:9 uploads are centre-cropped.

            To fit the shared 48 GB xGPU this Studio serves both 14B experts in **fp8** and
            samples with the **Wan2.2-Lightning 4-step** distillation LoRA at `cfg_scale = 1.0`
            instead of the paper's 40 steps at `cfg_scale = 3.5`; expect slightly softer detail
            than the official samples.
            """
        )

    quick_picks.select(
        load_quick_pick, None, [image_input, prompt_input, trajectory_input], show_progress="hidden"
    )
    preview_inputs = [trajectory_input, motion_input]
    trajectory_input.change(preview_path, preview_inputs, path_preview, show_progress="hidden")
    motion_input.change(preview_path, preview_inputs, path_preview, show_progress="hidden")
    demo.load(preview_path, preview_inputs, path_preview, show_progress="hidden")

    gr.on(
        triggers=[run_button.click, prompt_input.submit],
        fn=generate,
        inputs=[
            image_input,
            prompt_input,
            trajectory_input,
            steps_input,
            motion_input,
            fov_input,
            seed_input,
            randomize_input,
        ],
        outputs=[video_output, seed_input, status_output],
        concurrency_limit=1,
        concurrency_id="scope-gpu",
    )

if __name__ == "__main__":
    demo.launch(theme=MS_THEME, css=CSS, mcp_server=True)
