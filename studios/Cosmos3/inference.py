# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cosmos3-Edge 推理引擎 — Physical AI Reasoning.

仅使用 Reasoner tower（Cosmos3EdgeForConditionalGeneration），
支持图片/视频输入 + 文本指令 → 流式文本输出 + 思维链。

遵循 NVIDIA 官方 HuggingFace Space demo 的 API 调用方式。
参考: https://huggingface.co/spaces/nvidia/nvidia-cosmos3-edge
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# 持久化缓存（创空间关键：/mnt/workspace 是唯一持久卷）
# ---------------------------------------------------------------------------
_PERSISTENT = "/mnt/workspace" if os.path.isdir("/mnt/workspace") else os.path.expanduser("~/.cache")
MODEL_CACHE_DIR = os.path.join(_PERSISTENT, "modelscope_cache")
os.environ.setdefault("MODELSCOPE_CACHE", MODEL_CACHE_DIR)

# 模型 ID（魔搭优先，HuggingFace 回退）
MS_MODEL_ID = "nv-community/Cosmos3-Edge"
HF_MODEL_ID = "nvidia/Cosmos3-Edge"

# Reasoner 采样参数（来自官方文档）
MAX_NEW_TOKENS_DEFAULT = 1024
MAX_VIDEO_FRAMES = 16  # Reasoner 训练于 ~4fps 视频


# ---------------------------------------------------------------------------
# 状态数据类
# ---------------------------------------------------------------------------
@dataclass
class EngineState:
    """跟踪推理引擎加载状态，供 UI 状态栏展示。"""
    reasoner_loaded: bool = False
    status: str = "idle"          # idle / loading / ready / error
    message: str = "模型未加载"
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# ModelManager 单例
# ---------------------------------------------------------------------------
class ModelManager:
    """管理 Reasoner 引擎的懒加载与推理调度。线程安全。"""

    _instance: Optional["ModelManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._model = None
            self._processor = None
            self.state = EngineState()
            self._engine_lock = threading.Lock()
            self._initialized = True

    def _resolve_model_path(self) -> str:
        """解析模型路径：魔搭优先，HuggingFace 回退。"""
        try:
            from modelscope import snapshot_download
            path = snapshot_download(MS_MODEL_ID, cache_dir=MODEL_CACHE_DIR)
            if path:
                self._model_path = path
                return path
        except Exception:
            pass
        self._model_path = HF_MODEL_ID
        return self._model_path

    def load_reasoner(self) -> None:
        """加载 Transformers Reasoner（Cosmos3EdgeForConditionalGeneration）。

        Edge 模型使用 cosmos3_edge 模块的 Cosmos3EdgeForConditionalGeneration
        Nano/Super 使用 cosmos3_omni 模块的 Cosmos3OmniForConditionalGeneration
        两者是独立的 transformers 子模块，不可混用。
        """
        if self._model is not None:
            self.state.reasoner_loaded = True
            return
        with self._engine_lock:
            self.state.status = "loading"
            self.state.message = "正在加载推理引擎（Reasoner Tower）..."
            try:
                from transformers import AutoProcessor, Cosmos3EdgeForConditionalGeneration

                model_path = self._resolve_model_path()
                self._processor = AutoProcessor.from_pretrained(model_path)
                self._model = Cosmos3EdgeForConditionalGeneration.from_pretrained(
                    model_path,
                    dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                ).to("cuda")
                self._model.eval()
                self.state.reasoner_loaded = True
                self.state.status = "ready"
                self.state.message = "推理引擎已就绪"
            except Exception as exc:
                self.state.status = "error"
                self.state.message = f"推理引擎加载失败: {exc}"
                self.state.error = str(exc)
                raise

    def get_status(self) -> tuple[str, str]:
        return self.state.status, self.state.message


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------
_manager = ModelManager()


def get_status() -> tuple[str, str]:
    """获取引擎状态，供 UI 状态栏调用。"""
    return _manager.get_status()


# ---------------------------------------------------------------------------
# 视频帧采样
# ---------------------------------------------------------------------------
def _sample_video_frames(video_path: str, max_frames: int = MAX_VIDEO_FRAMES):
    """将视频解码为均匀采样的 PIL 帧列表。

    Reasoner 训练于 ~4fps 视频，采样有界帧数即可。
    使用 imageio + pyav 解码，避免依赖 torchvision 的 read_video。
    """
    import imageio.v3 as iio
    import numpy as np
    from PIL import Image

    frames = iio.imread(video_path, plugin="pyav")  # (T, H, W, C)
    frames = np.asarray(frames)
    total = len(frames)
    if total == 0:
        raise ValueError("无法从视频中读取任何帧")
    n = min(max_frames, total)
    idx = np.linspace(0, total - 1, n).round().astype(int)
    return [Image.fromarray(frames[i]).convert("RGB") for i in idx]


# ---------------------------------------------------------------------------
# 消息构建
# ---------------------------------------------------------------------------
def _build_messages(prompt: str, image, video):
    """组装用户消息（图片/视频/文本）。

    遵循 Qwen3-VL 兼容的消息格式。
    """
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    elif video is not None:
        content.append({"type": "video", "video": _sample_video_frames(video)})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# 流式推理函数
# ---------------------------------------------------------------------------
def reason_stream(
    prompt: str,
    image=None,
    video: Optional[str] = None,
    enable_thinking: bool = True,
    max_new_tokens: int = MAX_NEW_TOKENS_DEFAULT,
    temperature: float = 0.6,
    top_p: float = 0.95,
):
    """流式推理：文本 + 图片/视频 → 流式文本输出。

    支持思维链（enable_thinking）和视频输入。
    使用 TextIteratorStreamer 实时输出推理过程。

    Args:
        prompt: 推理指令或问题。
        image: 可选的 PIL 图片条件输入。
        video: 可选的视频文件路径。
        enable_thinking: 启用后先输出思维链推理过程，再给出最终答案。
        max_new_tokens: 最大生成 token 数。
        temperature: 采样温度（0 = 贪心解码）。
        top_p: 核采样概率。

    Yields:
        (cleared_prompt, accumulated_text) — 清空输入框，逐步更新输出
    """
    if not prompt or not prompt.strip():
        yield "", "请输入指令或问题"
        return

    # 加载引擎
    if not _manager.state.reasoner_loaded:
        yield "", "正在加载推理引擎（首次加载约 30 秒）..."
        _manager.load_reasoner()

    yield "", "正在推理..."

    try:
        from transformers import TextIteratorStreamer

        messages = _build_messages(prompt, image, video)

        inputs = _manager._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=bool(enable_thinking),
        ).to(_manager._model.device)

        streamer = TextIteratorStreamer(
            _manager._processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=int(max_new_tokens),
            do_sample=temperature > 0,
            temperature=float(temperature) if temperature > 0 else 1.0,
            top_p=float(top_p),
        )

        thread = threading.Thread(target=_manager._model.generate, kwargs=gen_kwargs)
        thread.start()

        accumulated = ""
        for token in streamer:
            accumulated += token
            yield "", accumulated

        thread.join()

        if not accumulated.strip():
            yield "", "(未生成任何内容)"

    except Exception as exc:
        yield "", f"推理失败: {exc}"
