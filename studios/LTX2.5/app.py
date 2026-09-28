from __future__ import annotations

import logging
import os
import random
import threading
import time

import gradio as gr

from inference import LTX25Service, RuntimeSettings, output_dir_from_env


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ltx25.app")

MAX_SEED = 2**31 - 1
_service: LTX25Service | None = None
_service_lock = threading.Lock()


def get_service() -> LTX25Service:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = LTX25Service(RuntimeSettings.from_env())
    return _service


def generate(
    prompt: str,
    image_path: str | None,
    aspect_ratio: str,
    duration_seconds: int,
    seed: int,
    randomize_seed: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[str, int, str]:
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("请先输入视频描述。")
    if len(prompt.split()) > 220:
        raise gr.Error("提示词过长，请控制在约 200 个英文单词以内。")

    actual_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    progress(0.02, desc="准备模型")
    started = time.perf_counter()
    try:
        output_path, metrics = get_service().generate(
            prompt=prompt,
            image_path=image_path,
            aspect_ratio=aspect_ratio,
            duration_seconds=int(duration_seconds),
            seed=actual_seed,
        )
    except RuntimeError as exc:
        logger.exception("Generation failed")
        message = str(exc)
        if "out of memory" in message.lower():
            raise gr.Error("GPU 显存不足，请缩短视频时长后重试。") from exc
        raise gr.Error(f"生成失败：{message}") from exc

    elapsed = time.perf_counter() - started
    status = (
        f"完成，用时 {elapsed:.1f} 秒；{metrics.width}×{metrics.height}，"
        f"{metrics.num_frames} 帧；GPU 峰值 {metrics.peak_vram_gib:.1f} GiB。"
    )
    return output_path, actual_seed, status


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="LTX-2.5 音视频生成", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # LTX-2.5 音视频生成
            输入一段镜头描述，可选上传首帧图片，生成带同步音频的视频。
            当前使用官方 LTX-2.5 Distilled BF16 两阶段 Pipeline。
            A100 80GB默认让完整DiT权重在扩散阶段内驻留GPU；低于75GiB可见显存时自动切换CPU逐层流送。
            单任务串行运行。
            """
        )
        with gr.Row():
            with gr.Column(scale=5):
                prompt = gr.Textbox(
                    label="视频描述",
                    lines=8,
                    placeholder=(
                        "请按时间顺序描述主体动作、环境、镜头运动、光线和声音。"
                        "例如：A medium shot of ..."
                    ),
                )
                image = gr.Image(label="首帧图片（可选）", type="filepath")
                with gr.Row():
                    aspect = gr.Radio(
                        choices=["横屏 3:2", "竖屏 2:3", "方形 1:1"],
                        value="横屏 3:2",
                        label="画面比例",
                    )
                    duration = gr.Radio(
                        choices=[3, 5],
                        value=3,
                        label="时长（秒）",
                    )
                with gr.Row():
                    seed = gr.Number(value=42, precision=0, label="随机种子")
                    randomize = gr.Checkbox(value=True, label="每次随机种子")
                submit = gr.Button("生成视频", variant="primary")
            with gr.Column(scale=5):
                video = gr.Video(label="生成结果", autoplay=False)
                actual_seed = gr.Number(label="实际随机种子", precision=0)
                status = gr.Textbox(label="运行信息", interactive=False)

        submit.click(
            fn=generate,
            inputs=[prompt, image, aspect, duration, seed, randomize],
            outputs=[video, actual_seed, status],
            concurrency_limit=1,
        )

    return demo


demo = build_demo()

if __name__ == "__main__":
    output_dir = output_dir_from_env()
    output_dir.mkdir(parents=True, exist_ok=True)
    demo.queue(default_concurrency_limit=1, max_size=5).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        allowed_paths=[str(output_dir.resolve())],
    )
