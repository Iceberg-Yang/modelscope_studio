from __future__ import annotations

import io
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image


LOGGER = logging.getLogger("anima-studio")

MODEL_ID = "Gazingstars123/Anima-2.9B"
BASE_MODEL_ID = "circlestone-labs/Anima"
MODEL_FILE = "Anima-2.9B-preview-v1.safetensors"
TEXT_ENCODER_FILE = "qwen_3_06b_base.safetensors"
VAE_FILE = "qwen_image_vae.safetensors"

COMFYUI_REPOSITORY = "https://github.com/Comfy-Org/ComfyUI.git"
COMFYUI_COMMIT = "5ab2f7a2d676c1fb7b410c22e82e2ed8f217b56c"
COMFYUI_PORT = int(os.getenv("COMFYUI_PORT", "8188"))
COMFYUI_URL = f"http://127.0.0.1:{COMFYUI_PORT}"

RESOLUTIONS = {
    "竖图 · 812 × 1216": (812, 1216),
    "横图 · 1216 × 812": (1216, 812),
    "方图 · 1024 × 1024": (1024, 1024),
    "高清竖图 · 1152 × 1536": (1152, 1536),
    "高清横图 · 1536 × 1152": (1536, 1152),
    "实验方图 · 1536 × 1536": (1536, 1536),
}

SAMPLERS = {
    "Euler（均衡）": "euler",
    "ER-SDE（线条清晰）": "er_sde",
    "Res Multistep（构图优先）": "res_multistep",
}

SCHEDULERS = {
    "SGM Uniform（推荐）": "sgm_uniform",
    "Beta": "beta",
    "Linear Quadratic（构图优先）": "linear_quadratic",
    "Simple": "simple",
}

DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, low quality, score_1, score_2, score_3, blurry, "
    "jpeg artifacts, chromatic aberration, malformed hands, extra fingers"
)


def _runtime_root() -> Path:
    configured = os.getenv("ANIMA_RUNTIME_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    persistent = Path("/mnt/workspace")
    if persistent.exists() and os.access(persistent, os.W_OK):
        return persistent / "anima-2.9b-studio"
    return Path(__file__).resolve().parent / "runtime"


@dataclass
class RuntimeState:
    phase: str = "starting"
    detail: str = "准备初始化"
    error: str | None = None
    ready: bool = False
    started_at: float = 0.0


class AnimaRuntime:
    def __init__(self) -> None:
        self.root = _runtime_root()
        self.comfy_dir = self.root / "ComfyUI"
        self.model_cache = self.root / "model-cache"
        self.outputs = self.root / "outputs"
        self.logs = self.root / "logs"
        self.state = RuntimeState()
        self._state_lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None

    def _set_state(
        self,
        phase: str,
        detail: str,
        *,
        ready: bool = False,
        error: str | None = None,
    ) -> None:
        with self._state_lock:
            self.state.phase = phase
            self.state.detail = detail
            self.state.ready = ready
            self.state.error = error
        LOGGER.info("runtime phase=%s detail=%s", phase, detail)

    def status_markdown(self) -> str:
        with self._state_lock:
            state = RuntimeState(**vars(self.state))
        if state.error:
            return f"🔴 **初始化失败**：{state.error}"
        if state.ready:
            return "🟢 **模型已就绪**"
        return f"🟡 **正在准备模型** · {state.detail}"

    def wait_until_ready(
        self,
        timeout: int = 1200,
        progress: Callable[[float, str], None] | None = None,
    ) -> None:
        """Wait for the one-time model bootstrap instead of failing early clicks."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._state_lock:
                ready = self.state.ready
                error = self.state.error
                detail = self.state.detail
            if error:
                raise RuntimeError(error)
            if ready:
                return
            if progress:
                progress(0.02, f"模型首次初始化：{detail}")
            time.sleep(2)
        raise TimeoutError("模型首次初始化超过 20 分钟，请刷新状态后重试。")

    def start_background(self) -> None:
        thread = threading.Thread(target=self._initialize, name="anima-init", daemon=True)
        thread.start()

    def _initialize(self) -> None:
        try:
            self.state.started_at = time.time()
            for directory in (self.root, self.model_cache, self.outputs, self.logs):
                directory.mkdir(parents=True, exist_ok=True)
            self._ensure_comfyui()
            self._ensure_models()
            self._start_comfyui()
            # xGPU may suspend background work between HTTP invocations. The
            # first ComfyUI import can therefore take several minutes even
            # though the GPU and model cache are healthy.
            self._wait_for_comfyui(timeout=1200)
            self._set_state("ready", "模型与推理服务已加载", ready=True)
        except Exception as exc:  # pragma: no cover - exercised on the GPU host
            LOGGER.exception("Anima runtime initialization failed")
            self._set_state("failed", "初始化失败", error=str(exc))

    def _ensure_comfyui(self) -> None:
        marker = self.comfy_dir / ".anima-runtime-commit"
        if marker.exists() and marker.read_text(encoding="utf-8").strip() == COMFYUI_COMMIT:
            self._set_state("runtime", "复用持久化的 ComfyUI 运行时")
            return

        self._set_state("runtime", "首次安装 ComfyUI 运行时")
        if self.comfy_dir.exists():
            shutil.rmtree(self.comfy_dir)
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                COMFYUI_REPOSITORY,
                str(self.comfy_dir),
            ],
            check=True,
            timeout=300,
        )
        subprocess.run(
            ["git", "-C", str(self.comfy_dir), "checkout", COMFYUI_COMMIT],
            check=True,
            timeout=300,
        )
        marker.write_text(COMFYUI_COMMIT, encoding="utf-8")

    def _ensure_models(self) -> None:
        from modelscope.hub.snapshot_download import snapshot_download

        self._set_state("models", "下载 Anima-2.9B BF16 权重（约 5.84 GB）")
        anima_dir = Path(
            snapshot_download(
                model_id=MODEL_ID,
                revision="master",
                cache_dir=str(self.model_cache),
                allow_patterns=[MODEL_FILE],
            )
        )

        self._set_state("models", "下载 Qwen3 文本编码器与 Qwen Image VAE")
        base_dir = Path(
            snapshot_download(
                model_id=BASE_MODEL_ID,
                revision="master",
                cache_dir=str(self.model_cache),
                allow_patterns=[
                    f"split_files/text_encoders/{TEXT_ENCODER_FILE}",
                    f"split_files/vae/{VAE_FILE}",
                ],
            )
        )

        targets = {
            self.comfy_dir / "models" / "diffusion_models" / MODEL_FILE: anima_dir / MODEL_FILE,
            self.comfy_dir / "models" / "text_encoders" / TEXT_ENCODER_FILE: (
                base_dir / "split_files" / "text_encoders" / TEXT_ENCODER_FILE
            ),
            self.comfy_dir / "models" / "vae" / VAE_FILE: (
                base_dir / "split_files" / "vae" / VAE_FILE
            ),
        }
        for target, source in targets.items():
            if not source.exists():
                raise FileNotFoundError(f"ModelScope 下载完成后未找到文件：{source}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink() or target.exists():
                target.unlink()
            target.symlink_to(source)

    def _start_comfyui(self) -> None:
        self._set_state("server", "启动 ComfyUI 推理服务")
        log_path = self.logs / "comfyui.log"
        log_handle = log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        command = [
            sys.executable,
            "main.py",
            "--listen",
            "127.0.0.1",
            "--port",
            str(COMFYUI_PORT),
            "--disable-auto-launch",
            "--disable-all-custom-nodes",
            "--disable-api-nodes",
            "--disable-metadata",
            "--output-directory",
            str(self.outputs),
        ]
        self._process = subprocess.Popen(
            command,
            cwd=self.comfy_dir,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def _wait_for_comfyui(self, timeout: int) -> None:
        deadline = time.time() + timeout
        last_error: Exception | None = None
        while time.time() < deadline:
            if self._process and self._process.poll() is not None:
                tail = self._log_tail()
                raise RuntimeError(f"ComfyUI 启动失败（exit={self._process.returncode}）：{tail}")
            try:
                with urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=3) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
            time.sleep(2)
        raise TimeoutError(f"等待 ComfyUI 就绪超时：{last_error}; {self._log_tail()}")

    def _log_tail(self, lines: int = 30) -> str:
        path = self.logs / "comfyui.log"
        if not path.exists():
            return "无运行日志"
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        resolution: str,
        steps: int,
        cfg: float,
        sampler_label: str,
        scheduler_label: str,
        seed: int,
        randomize_seed: bool,
        progress: Callable[[float, str], None] | None = None,
    ) -> tuple[Image.Image, int, str]:
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("请输入提示词。")
        self.wait_until_ready(timeout=1200, progress=progress)
        if resolution not in RESOLUTIONS:
            raise ValueError("不支持的分辨率。")
        if sampler_label not in SAMPLERS or scheduler_label not in SCHEDULERS:
            raise ValueError("不支持的采样参数。")

        width, height = RESOLUTIONS[resolution]
        actual_seed = random.randint(0, 2**63 - 1) if randomize_seed else int(seed)
        steps = max(28, min(int(steps), 50))
        cfg = max(3.5, min(float(cfg), 5.0))
        workflow = build_workflow(
            prompt=prompt,
            negative_prompt=(negative_prompt or DEFAULT_NEGATIVE_PROMPT).strip(),
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            sampler=SAMPLERS[sampler_label],
            scheduler=SCHEDULERS[scheduler_label],
            seed=actual_seed,
        )

        with self._generation_lock:
            if progress:
                progress(0.05, "提交生成任务")
            prompt_id = self._queue_prompt(workflow)
            image = self._wait_for_image(prompt_id, timeout=1200, progress=progress)

        info = (
            f"完成 · {width}×{height} · {steps} steps · CFG {cfg:g} · "
            f"{SAMPLERS[sampler_label]} / {SCHEDULERS[scheduler_label]}"
        )
        return image, actual_seed, info

    def _queue_prompt(self, workflow: dict) -> str:
        request = urllib.request.Request(
            f"{COMFYUI_URL}/prompt",
            data=json.dumps({"prompt": workflow}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
        if "prompt_id" not in result:
            raise RuntimeError(f"ComfyUI 未返回任务 ID：{result}")
        return str(result["prompt_id"])

    def _wait_for_image(
        self,
        prompt_id: str,
        timeout: int,
        progress: Callable[[float, str], None] | None,
    ) -> Image.Image:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10) as response:
                history = json.load(response)
            task = history.get(prompt_id)
            if task:
                status = task.get("status", {})
                if status.get("status_str") == "error":
                    messages = status.get("messages", [])
                    raise RuntimeError(f"生成失败：{messages[-1] if messages else status}")
                for node_output in task.get("outputs", {}).values():
                    images = node_output.get("images", [])
                    if images:
                        if progress:
                            progress(0.98, "读取生成结果")
                        return self._fetch_image(images[0])
            if progress:
                elapsed = timeout - max(0, deadline - time.time())
                progress(min(0.95, 0.08 + 0.87 * elapsed / timeout), "正在生成图像")
            time.sleep(2)
        raise TimeoutError("生成超过 20 分钟，任务已超时。")

    @staticmethod
    def _fetch_image(image_info: dict) -> Image.Image:
        query = urllib.parse.urlencode(
            {
                "filename": image_info["filename"],
                "subfolder": image_info.get("subfolder", ""),
                "type": image_info.get("type", "output"),
            }
        )
        with urllib.request.urlopen(f"{COMFYUI_URL}/view?{query}", timeout=60) as response:
            data = response.read()
        image = Image.open(io.BytesIO(data))
        image.load()
        return image


def build_workflow(
    *,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    seed: int,
) -> dict:
    """Return a ComfyUI API workflow based on the official Anima blueprint."""
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": MODEL_FILE, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": TEXT_ENCODER_FILE,
                "type": "stable_diffusion",
                "device": "default",
            },
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_FILE}},
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": prompt},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": negative_prompt},
        },
        "6": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": 1.0,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["7", 0], "vae": ["3", 0]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["8", 0], "filename_prefix": "Anima-2.9B"},
        },
    }
