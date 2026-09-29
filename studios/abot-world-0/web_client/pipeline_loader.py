"""
web_client/pipeline_loader.py — Pipeline 单例加载与帧解码辅助。
"""
import sys
import os
import importlib
import threading
from functools import lru_cache
from pathlib import Path

# 确保 web_client 模块在路径中
_web_client_dir = Path(__file__).parent
_project_root = _web_client_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
from omegaconf import OmegaConf

from pipeline import CausalInferencePipeline
from utils.misc import set_seed
from utils.wan_wrapper import create_vae_from_config
from utils.memory import gpu
from wan.modules.helios_kernels import replace_all_norms_with_flash_norms, replace_rope_with_flash_rope

_init_lock = threading.Lock()


class FrameCapture:
    """Implements the append_data interface expected by decode_block_and_write,
    capturing numpy frames in memory instead of writing to a file."""
    def __init__(self):
        self.frames = []

    def append_data(self, frame):
        self.frames.append(frame)


_CONFIG_YAML = "configs/long_forcing_dmd.yaml"


def get_pipeline(
    vae_type: str | None = None,
    use_fp8_gemm: bool = False,
    quant_type: str | None = None,
):
    """加载并缓存推理 pipeline（进程内单例）。

    Args:
        vae_type: VAE 模型类型 (wan2.2 | taew2_2 | mg_lightvae | mg_lightvae_v2)。
            若为 ``None``，则从 ``configs/default_config.yaml`` 的 ``vae_type`` 字段读取。
    """
    # Resolve vae_type from default config if not explicitly provided
    if vae_type is None:
        try:
            _dc = OmegaConf.load("configs/default_config.yaml")
            vae_type = str(getattr(_dc, "vae_type", "")).strip().lower() or "taew2_2"
        except Exception:
            vae_type = "taew2_2"
    return _get_pipeline_cached(
        vae_type, use_fp8_gemm, quant_type,
    )


@lru_cache(maxsize=32)
def _get_pipeline_cached(
    vae_type: str,
    use_fp8_gemm: bool,
    quant_type: str | None,
):
    yaml_path = _CONFIG_YAML

    device = torch.device("cuda")
    set_seed(42)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    config = OmegaConf.load(yaml_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    quant_mode = os.environ.get(
        "ABOT_QUANT_MODE", "fp8" if use_fp8_gemm else "bf16"
    ).strip().lower()
    if quant_mode not in {"fp8", "bf16", "hybrid"}:
        raise ValueError(
            "ABOT_QUANT_MODE must be one of: fp8, bf16, hybrid"
        )
    effective_fp8 = quant_mode != "bf16"
    config.use_fp8_gemm = effective_fp8
    config.quant_mode = quant_mode

    # Ensure config has the resolved vae_type so create_vae_from_config works
    config.vae_type = vae_type

    vae = create_vae_from_config(config)

    pipeline = CausalInferencePipeline(config, device=device, vae=vae)

    replace_all_norms_with_flash_norms(pipeline.generator.model)
    replace_rope_with_flash_rope()

    # L20N/RTX PRO 5000 profile: keep all hot components resident.
    # every hot component resident. Avoid converting
    # the TAEW2.2 decoder to BF16; it was trained for FP16 and its autocast
    # context also uses FP16.
    pipeline.generator.to(device=gpu, dtype=torch.bfloat16)
    pipeline.text_encoder.to(device=gpu, dtype=torch.bfloat16)
    if pipeline.encoder is not None:
        pipeline.encoder.to(device=gpu, dtype=torch.bfloat16)
    vae_dtype = torch.float16 if vae_type == "taew2_2" else torch.bfloat16
    pipeline.vae.to(device=gpu, dtype=vae_dtype)
    if hasattr(pipeline.vae, "dtype"):
        pipeline.vae.dtype = vae_dtype
    pipeline.torch_dtype = torch.bfloat16

    if effective_fp8:
        from quantizer import apply_fp8_quantization
        final_quant_type = str(quant_type) if quant_type is not None else str(
            getattr(config, "quant_type", "fp8-per-token")
        )
        apply_fp8_quantization(
            model=pipeline.generator.model,
            quant_type=final_quant_type,
            min_weight_numel=(
                int(os.environ.get("ABOT_HYBRID_FP8_MIN_WEIGHT_NUMEL", "16000000"))
                if quant_mode == "hybrid"
                else 0
            ),
        )

    pipeline.attention_benchmark = {}
    if os.environ.get("ABOT_ATTN_BENCHMARK", "1") == "1":
        attention_module = importlib.import_module("wan.modules.attention")

        model = pipeline.generator.model
        patch_t, patch_h, patch_w = [int(v) for v in model.patch_size]
        frame_tokens = int(pipeline.frame_seq_length)
        num_fpb = int(pipeline.num_frame_per_block)
        ref_slots = int(getattr(config, "ref_num_slots", 5))
        ref_resolution = int(getattr(config, "ref_resolution", 512))
        # ABot reference latents are encoded at 16x spatial downsampling.
        ref_latent_side = ref_resolution // 16
        ref_tokens = (
            ref_slots
            * (1 // patch_t)
            * (ref_latent_side // patch_h)
            * (ref_latent_side // patch_w)
        )
        first_query_tokens = ref_tokens + num_fpb * frame_tokens
        steady_query_tokens = num_fpb * frame_tokens
        local_frames = int(getattr(model, "local_attn_size", -1))
        steady_kv_tokens = (
            ref_tokens
            + (
                local_frames * frame_tokens
                if local_frames != -1
                else int(config.image_or_video_shape[1]) * frame_tokens
            )
        )
        pipeline.attention_benchmark = (
            attention_module.benchmark_attention_backends(
                num_heads=int(model.num_heads),
                head_dim=int(model.dim // model.num_heads),
                first_query_tokens=first_query_tokens,
                steady_query_tokens=steady_query_tokens,
                steady_kv_tokens=steady_kv_tokens,
                warmup=int(os.environ.get("ABOT_ATTN_BENCH_WARMUP", "1")),
                repeats=int(os.environ.get("ABOT_ATTN_BENCH_REPEATS", "3")),
                include_sdpa=os.environ.get(
                    "ABOT_ATTN_BENCHMARK_SDPA", "0"
                ) == "1",
                min_speedup_ratio=float(
                    os.environ.get("ABOT_ATTN_AUTO_MIN_SPEEDUP", "0.03")
                ),
            )
        )

    quantized_linear_layers = sum(
        1
        for module in pipeline.generator.model.modules()
        if module.__class__.__name__.startswith(("FP8", "FP4"))
        and module.__class__.__name__.endswith("Linear")
    )

    props = torch.cuda.get_device_properties(device)
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    print(
        "[INIT][GPU] "
        f"name={props.name}, cc={props.major}.{props.minor}, "
        f"vram={props.total_memory / (1024 ** 3):.1f}GiB, "
        f"allocated={allocated:.1f}GiB, reserved={reserved:.1f}GiB, "
        f"generator={next(pipeline.generator.parameters()).dtype}, "
        f"vae={vae_dtype}, quant_mode={quant_mode}, "
        f"fp8_gemm={effective_fp8}, "
        f"quantized linear layers: {quantized_linear_layers}",
        flush=True,
    )
    if effective_fp8 and quantized_linear_layers <= 0:
        raise RuntimeError(
            "FP8 GEMM was requested but no quantized linear layers were installed"
        )

    return pipeline, config, device


def decode_block_to_frames(pipeline, lat_block) -> list:
    capture = FrameCapture()
    pipeline.decode_block_and_write(lat_block, capture)
    return capture.frames
