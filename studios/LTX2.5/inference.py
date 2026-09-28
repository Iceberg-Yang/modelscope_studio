from __future__ import annotations

import gc
import logging
import os
import shutil
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger("ltx25.inference")

MODEL_REPO = "Lightricks/LTX-2.5"
MODEL_FILES = (
    "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
)

RESOLUTIONS = {
    "横屏 3:2": (768, 512),
    "竖屏 2:3": (512, 768),
    "方形 1:1": (640, 640),
}


def persistent_root_from_env() -> Path:
    persistent_root = Path(os.getenv("LTX_PERSISTENT_ROOT", "/mnt/workspace"))
    if not persistent_root.exists() or not os.access(persistent_root, os.W_OK):
        return Path(os.getenv("LTX_FALLBACK_ROOT", "/tmp/ltx25-workspace"))
    return persistent_root


def output_dir_from_env() -> Path:
    return Path(os.getenv("LTX_OUTPUT_DIR", str(persistent_root_from_env() / "outputs" / "ltx25")))


@dataclass(frozen=True)
class RuntimeSettings:
    model_dir: Path
    output_dir: Path
    offload_mode: str

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        persistent_root = persistent_root_from_env()
        return cls(
            model_dir=Path(os.getenv("LTX_MODEL_DIR", str(persistent_root / "models" / "ltx-2.5"))),
            output_dir=output_dir_from_env(),
            offload_mode=os.getenv("LTX_OFFLOAD", "auto").lower(),
        )


@dataclass(frozen=True)
class GenerationMetrics:
    width: int
    height: int
    num_frames: int
    peak_vram_gib: float


class LTX25Service:
    def __init__(self, settings: RuntimeSettings):
        self.settings = settings
        disk_preflight(settings)
        self._pipeline = None
        self._lock = threading.Lock()
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)

    def _check_runtime(self):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("没有检测到 CUDA GPU。")
        name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
        logger.info(
            "GPU=%s capability=%s VRAM=%.1fGiB torch=%s cuda=%s",
            name,
            capability,
            total_gib,
            torch.__version__,
            torch.version.cuda,
        )
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"GPU {name} 不支持原生 BF16，无法运行当前权重。")
        if total_gib < 44:
            raise RuntimeError(f"当前GPU仅有 {total_gib:.1f} GiB 可见显存，需要48GB级别资源。")
        self._resolve_offload_mode(self.settings.offload_mode, total_gib)
        if capability not in {(8, 0), (8, 9)}:
            logger.warning("Unvalidated GPU compute capability: %s", capability)
        return torch

    @staticmethod
    def _resolve_offload_mode(requested_mode: str, total_gib: float) -> str:
        if requested_mode not in {"auto", "none", "cpu", "disk"}:
            raise RuntimeError("LTX_OFFLOAD 必须是 auto、none、cpu 或 disk。")
        if requested_mode == "auto":
            return "none" if total_gib >= 75 else "cpu"
        if requested_mode == "none" and total_gib < 75:
            raise RuntimeError("低于75GiB可见显存时必须启用 LTX_OFFLOAD=cpu 或 disk。")
        return requested_mode

    def _ensure_models(self) -> None:
        missing = [name for name in MODEL_FILES if not (self.settings.model_dir / name).is_file()]
        if not missing:
            return

        from modelscope import snapshot_download

        self.settings.model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Downloading %d required model components from ModelScope %s to %s",
            len(missing),
            MODEL_REPO,
            self.settings.model_dir,
        )
        snapshot_download(
            model_id=MODEL_REPO,
            revision="master",
            allow_file_pattern=list(MODEL_FILES),
            local_dir=self.settings.model_dir,
            max_workers=4,
        )

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        torch = self._check_runtime()
        self._ensure_models()

        import transformers
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        logger.info("transformers=%s", transformers.__version__)
        if "gemma4_unified" not in CONFIG_MAPPING:
            raise RuntimeError(
                f"Transformers {transformers.__version__} 未注册 gemma4_unified；"
                "需要官方验证的5.14.1版本。"
            )

        from ltx_pipelines.distilled import DistilledPipeline
        from ltx_pipelines.utils.model_paths import ModelPaths
        from ltx_pipelines.utils.types import OffloadMode

        offload_modes = {
            "none": OffloadMode.NONE,
            "cpu": OffloadMode.CPU,
            "disk": OffloadMode.DISK,
        }
        total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
        resolved_offload_mode = self._resolve_offload_mode(self.settings.offload_mode, total_gib)

        root = self.settings.model_dir
        paths = ModelPaths.from_split(
            transformer_path=str(
                root / "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
            ),
            text_encoder_path=str(
                root / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
            ),
            video_vae_path=str(root / "vae/ltx-2.5-video-vae-conv-bf16.safetensors"),
            audio_vae_path=str(root / "vae/ltx-2.5-audio-vae-bf16.safetensors"),
        )

        logger.info(
            "Loading official LTX-2.5 DistilledPipeline in BF16: requested_offload=%s, "
            "resolved_offload=%s, visible_vram=%.1fGiB",
            self.settings.offload_mode,
            resolved_offload_mode,
            total_gib,
        )
        self._pipeline = DistilledPipeline(
            model_paths=paths,
            spatial_upsampler_path=str(
                root / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
            ),
            loras=[],
            device=torch.device("cuda:0"),
            quantization=None,
            offload_mode=offload_modes[resolved_offload_mode],
        )
        return self._pipeline

    @staticmethod
    def _frames_for_seconds(seconds: int, fps: int = 24) -> int:
        return max(25, (seconds * fps // 8) * 8 + 1)

    def _prepare_image(self, image_path: str | None, width: int, height: int) -> str | None:
        if not image_path:
            return None
        from PIL import Image

        source = Image.open(image_path).convert("RGB")
        scale = max(width / source.width, height / source.height)
        resized = source.resize(
            (round(source.width * scale), round(source.height * scale)),
            Image.Resampling.LANCZOS,
        )
        left = (resized.width - width) // 2
        top = (resized.height - height) // 2
        prepared_path = self.settings.output_dir / f"input-{uuid.uuid4().hex}.jpg"
        resized.crop((left, top, left + width, top + height)).save(prepared_path, quality=95)
        return str(prepared_path)

    def _cleanup_outputs(self, keep: int = 20) -> None:
        files = sorted(self.settings.output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old_file in files[keep:]:
            old_file.unlink(missing_ok=True)

    @staticmethod
    def _validate_output(output_path: Path) -> float:
        import av

        if not output_path.is_file() or output_path.stat().st_size < 1024:
            raise RuntimeError(f"视频编码未产生有效文件：{output_path}")
        with av.open(str(output_path)) as container:
            video_streams = [stream for stream in container.streams if stream.type == "video"]
            if not video_streams:
                raise RuntimeError(f"输出文件不包含视频流：{output_path}")
            duration = float(container.duration or 0) / av.time_base
        if duration <= 0:
            raise RuntimeError(f"输出视频时长无效：{output_path}")
        logger.info(
            "Validated output %s (%.2fs, %.1fMiB)",
            output_path,
            duration,
            output_path.stat().st_size / 2**20,
        )
        return duration

    def generate(
        self,
        *,
        prompt: str,
        image_path: str | None,
        aspect_ratio: str,
        duration_seconds: int,
        seed: int,
    ) -> tuple[str, GenerationMetrics]:
        if aspect_ratio not in RESOLUTIONS:
            raise RuntimeError(f"不支持的画面比例：{aspect_ratio}")
        if duration_seconds not in (3, 5):
            raise RuntimeError("时长只能选择3秒或5秒。")

        with self._lock:
            torch = self._check_runtime()
            # Match the official distilled CLI entry point, whose main() is
            # decorated with @torch.inference_mode(). PromptEncoder returns
            # inference tensors, so pipeline construction, denoising and the
            # lazy video iterator must all be consumed in the same context.
            with torch.inference_mode():
                pipeline = self._load_pipeline()

                from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
                from ltx_pipelines.utils.args import ImageConditioningInput
                from ltx_pipelines.utils.media_io import encode_video

                width, height = RESOLUTIONS[aspect_ratio]
                num_frames = self._frames_for_seconds(duration_seconds)
                prepared_image = self._prepare_image(image_path, width, height)
                images = (
                    [ImageConditioningInput(path=prepared_image, frame_idx=0, strength=1.0)]
                    if prepared_image
                    else []
                )
                output_path = self.settings.output_dir / f"ltx25-{uuid.uuid4().hex}.mp4"

                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                try:
                    video, audio, resolved_frames, tiling_config = pipeline(
                        prompt=prompt,
                        seed=seed,
                        height=height,
                        width=width,
                        frame_rate=24.0,
                        images=images,
                        num_frames=num_frames,
                        tiling_config=AUTO_TILING,
                        enhance_prompt=False,
                    )
                    encode_video(
                        video=video,
                        fps=24,
                        audio=audio,
                        output_path=str(output_path),
                        video_chunks_number=get_video_chunks_number(resolved_frames, tiling_config),
                    )
                    self._validate_output(output_path)
                    peak_vram = torch.cuda.max_memory_allocated() / 2**30
                finally:
                    if prepared_image:
                        Path(prepared_image).unlink(missing_ok=True)
                    gc.collect()
                    torch.cuda.empty_cache()

            self._cleanup_outputs()
            return str(output_path), GenerationMetrics(
                width=width,
                height=height,
                num_frames=resolved_frames,
                peak_vram_gib=peak_vram,
            )


def disk_preflight(settings: RuntimeSettings) -> None:
    """Useful for deployment logs without importing torch or loading the model."""
    target = settings.model_dir if settings.model_dir.exists() else settings.model_dir.parent
    target.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(target).free / 2**30
    if free_gib < 80:
        logger.warning("Only %.1f GiB free at %s; at least 80 GiB is recommended", free_gib, target)
