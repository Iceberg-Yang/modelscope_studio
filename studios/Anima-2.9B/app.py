from __future__ import annotations

import logging
import os

import gradio as gr

from anima_backend import (
    DEFAULT_NEGATIVE_PROMPT,
    RESOLUTIONS,
    SAMPLERS,
    SCHEDULERS,
    AnimaRuntime,
)


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

runtime = AnimaRuntime()
runtime.start_background()

EXAMPLES = [
    [
        "masterpiece, best quality, newest, safe, 1girl, solo, silver hair, "
        "blue eyes, flowing white coat, standing above a luminous futuristic city, "
        "dramatic clouds, cinematic composition, detailed background, anime illustration"
    ],
    [
        "year 2025, best quality, safe, 1boy, solo, black hair, red scarf, "
        "walking through a quiet snow-covered shrine at dusk, warm lantern light, "
        "delicate snowflakes, atmospheric perspective, detailed anime artwork"
    ],
    [
        "masterpiece, best quality, safe. A young witch studies an ancient star map "
        "inside a vast observatory. Brass instruments surround her, constellations "
        "glow through the glass dome, rich blue and gold palette, intricate illustration."
    ],
]

CSS = """
:root { --accent: #7c5cff; }
.gradio-container {
  box-sizing: border-box;
  width: 100% !important;
  max-width: none !important;
  margin-inline: auto !important;
  padding-inline: clamp(16px, 3vw, 48px) !important;
}
.hero {
  padding: 26px 30px;
  border-radius: 24px;
  background: radial-gradient(circle at 85% 15%, rgba(124,92,255,.18), transparent 36%),
              linear-gradient(135deg, #f8f5ff, #eeeaff 58%, #e8efff);
  border: 1px solid rgba(124, 92, 255, .16);
  box-shadow: 0 14px 34px rgba(72, 52, 128, .08);
  color: #2c2450;
  margin-bottom: 18px;
}
.hero h1 { margin: 0 0 8px; color: #352767; font-size: 2.2rem; }
.hero p { margin: 0; color: #595273; max-width: 760px; }
.status-card { border-radius: 14px; }
.generate-button { min-height: 48px; font-weight: 700; }
.fineprint { color: #74788a; font-size: .88rem; line-height: 1.55; }
"""


def generate_image(
    prompt: str,
    negative_prompt: str,
    resolution: str,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    seed: int,
    randomize_seed: bool,
    progress=gr.Progress(track_tqdm=False),
):
    try:
        return runtime.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            resolution=resolution,
            steps=steps,
            cfg=cfg,
            sampler_label=sampler,
            scheduler_label=scheduler,
            seed=seed,
            randomize_seed=randomize_seed,
            progress=lambda value, desc: progress(value, desc=desc),
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


with gr.Blocks(css=CSS, title="Anima 2.9B · 动漫插画生成器") as demo:
    gr.HTML(
        """
        <section class="hero">
          <h1>Anima 2.9B</h1>
          <p>面向动漫、插画与非写实艺术的高质量文生图模型。写得越具体，画面越丰富。</p>
        </section>
        """
    )

    status = gr.Markdown(runtime.status_markdown(), elem_classes=["status-card"])
    refresh_status = gr.Button("刷新模型状态", size="sm")
    status_timer = gr.Timer(5)

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            prompt = gr.Textbox(
                label="画面描述",
                placeholder=(
                    "建议包含：质量标签、人物数量、角色外观、服装、动作、场景、光线、构图和画风……"
                ),
                lines=8,
            )
            negative_prompt = gr.Textbox(
                label="反向提示词",
                value=DEFAULT_NEGATIVE_PROMPT,
                lines=3,
            )

            with gr.Row():
                resolution = gr.Dropdown(
                    choices=list(RESOLUTIONS),
                    value="竖图 · 812 × 1216",
                    label="画面尺寸",
                )
                steps = gr.Slider(28, 50, value=30, step=1, label="采样步数")
                cfg = gr.Slider(3.5, 5.0, value=4.0, step=0.1, label="CFG")

            with gr.Accordion("高级参数", open=False):
                with gr.Row():
                    sampler = gr.Dropdown(
                        choices=list(SAMPLERS), value="Euler（均衡）", label="采样器"
                    )
                    scheduler = gr.Dropdown(
                        choices=list(SCHEDULERS),
                        value="SGM Uniform（推荐）",
                        label="调度器",
                    )
                with gr.Row():
                    seed = gr.Number(value=0, precision=0, label="Seed")
                    randomize_seed = gr.Checkbox(value=True, label="每次随机 Seed")

            generate = gr.Button(
                "生成插画",
                variant="primary",
                elem_classes=["generate-button"],
            )

        with gr.Column(scale=6):
            output = gr.Image(label="生成结果", type="pil", format="png")
            with gr.Row():
                used_seed = gr.Number(label="实际 Seed", precision=0, interactive=False)
                generation_info = gr.Textbox(label="生成参数", interactive=False)

    gr.Markdown("### 灵感示例")
    gr.Examples(examples=EXAMPLES, inputs=[prompt], cache_examples=False)

    gr.Markdown(
        """
        <div class="fineprint">
        <strong>使用说明：</strong>推荐使用详细英文标签或自然语言描述。模型主要用于动漫、插画和非写实内容，
        不擅长照片级写实与长文本渲染。1536×1536 为实验尺寸，构图稳定性可能下降。<br>
        <strong>许可与归属：</strong>Anima-2.9B 是 circlestone-labs/Anima 的衍生模型，
        模型权重适用 CircleStone Labs Non-Commercial License，仅供非商业用途。
        Copyright CircleStone Labs LLC. 本创空间不是 CircleStone Labs 的官方产品。
        </div>
        """
    )

    refresh_status.click(fn=runtime.status_markdown, outputs=status, queue=False)
    status_timer.tick(fn=runtime.status_markdown, outputs=status, queue=False)
    generate.click(
        fn=generate_image,
        inputs=[
            prompt,
            negative_prompt,
            resolution,
            steps,
            cfg,
            sampler,
            scheduler,
            seed,
            randomize_seed,
        ],
        outputs=[output, used_seed, generation_info],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
    )
