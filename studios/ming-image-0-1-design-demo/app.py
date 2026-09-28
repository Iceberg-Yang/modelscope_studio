"""Gradio demo for inclusionAI/Ming-Image-0.1-Design on ModelScope.

Ported from the Hugging Face Space `hugging-apps/ming-image-0-1-design-demo`.
The upstream Space runs on a ZeroGPU `xlarge` slice (~96 GB); ModelScope's
largest xGPU instance has 48 GB, so the ~53 GB checkpoint is split between VRAM
and CPU RAM and the thinker is paged in only for prompt encoding.
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time

import gradio as gr
import torch

from ming.pipeline import (
    MingImagePipeline,
    decode_to_pil,
    load_connector,
    load_mlp_module,
    load_scheduler,
    load_thinker,
    load_tokenizer,
    load_transformer,
    load_vae,
)

_t0 = time.perf_counter()
DEVICE = "cuda"


def _step(label):
    print(f"[ming] {label} ({time.perf_counter() - _t0:.1f}s)", flush=True)


_step("resolving inclusionAI/Ming-Image-0.1-Design from ModelScope")

tokenizer = load_tokenizer()
_step("tokenizer ready")

mlp_cfg, mlp_sd = load_mlp_module()
_step("prompt-projection weights ready")

# Load small-to-large and park each module on the GPU as soon as it exists, so
# the CPU side never holds the full ~53 GB at once (the instance has 64 GB RAM).
connector = load_connector().to(DEVICE)
torch.cuda.empty_cache()
_step("connector ready on cuda")

transformer = load_transformer().to(DEVICE)
torch.cuda.empty_cache()
_step("transformer ready on cuda")

vae = load_vae().to(DEVICE)
scheduler = load_scheduler()
_step("vae / scheduler ready on cuda")

# The thinker stays on meta until prompt encoding, when its bf16 safetensor shards
# are adopted directly on CUDA and then discarded without a CPU offload.
thinker, mllm_cfg = load_thinker()
_step("thinker indexed on meta")

pipe = MingImagePipeline(
    thinker=thinker,
    mllm_cfg=mllm_cfg,
    connector=connector,
    transformer=transformer,
    vae=vae,
    scheduler=scheduler,
    tokenizer=tokenizer,
    mlp_cfg=mlp_cfg,
    mlp_sd=mlp_sd,
    device=DEVICE,
)
_step("pipeline assembled")

RESOLUTIONS = {
    "1024 x 1024 (推荐)": (1024, 1024),
    "2048 x 2048 (高分辨率)": (2048, 2048),
}

DEFAULT_PROMPT = (
    "A clean, modern mobile app onboarding screen for a plant-care app, soft sage green and cream "
    'palette, a large rounded illustration of a monstera leaf, bold sans-serif headline reading '
    '"Grow Together", a rounded primary button labelled "Get Started", generous white space, '
    "flat vector style, high-fidelity UI design"
)

EXAMPLES = [
    [
        DEFAULT_PROMPT,
        "1024 x 1024 (推荐)",
        12,
        1.0,
        42,
    ],
    [
        "A cozy wooden cabin in the woods shown across the four seasons, a four-panel "
        "editorial layout: spring blossom, summer green, autumn red, winter snow, the same "
        "cabin in each panel, clean dividing lines, muted natural palette, the title "
        '"FOUR SEASONS CABIN" in a refined serif at the top, poster design',
        "1024 x 1024 (推荐)",
        12,
        1.0,
        1234,
    ],
    [
        "An infographic poster explaining the water cycle, three horizontal panels with flat icons of "
        "evaporation, condensation and precipitation, arrows connecting the panels, muted blue and sand "
        'colour palette, the title "THE WATER CYCLE" in condensed uppercase type at the top, clean grid '
        "layout, editorial design",
        "1024 x 1024 (推荐)",
        12,
        1.0,
        7,
    ],
    [
        "A minimal dark-mode dashboard UI for a music analytics app, deep charcoal background, a line chart "
        'of weekly listeners in neon mint, three rounded stat cards reading "128K Plays", "+12.4% Growth" '
        'and "48 Countries", left icon rail, crisp modern sans-serif labels, pixel-precise layout',
        "1024 x 1024 (推荐)",
        12,
        1.0,
        2024,
    ],
]


def generate(
    prompt,
    resolution="1024 x 1024 (推荐)",
    steps=12,
    guidance_scale=1.0,
    seed=42,
    progress=gr.Progress(),
):
    """Generate one RGBA design image from a text prompt."""
    if not prompt or not prompt.strip():
        raise gr.Error("请输入提示词。")

    height, width = RESOLUTIONS.get(resolution, (1024, 1024))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def cb(done, total):
        progress(done / total, desc=f"去噪 {done}/{total}")

    t0 = time.perf_counter()
    try:
        image = pipe(
            prompt.strip(),
            height=height,
            width=width,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
            callback=cb,
        )
    except Exception as exc:
        # The log service truncates every block at ~2 KB, which eats the tail
        # of Gradio's traceback and hides the exception. Emit a one-line
        # summary so the failure is still readable from the studio logs.
        free, total = torch.cuda.mem_get_info()
        print(
            f"[ming] FAILED {type(exc).__name__}: {exc} | "
            f"vram {free / 1e9:.1f}G free / {total / 1e9:.1f}G",
            flush=True,
        )
        raise
    elapsed = time.perf_counter() - t0
    print(
        f"[ming] generated {width}x{height} in {elapsed:.1f}s "
        f"(steps={steps}, cfg={guidance_scale}, seed={seed})",
        flush=True,
    )

    pil = decode_to_pil(image)
    return pil, f"{pil.size[0]} x {pil.size[1]} RGBA，耗时 {elapsed:.1f} 秒 · {steps} 步 · CFG {guidance_scale}"


with gr.Blocks(title="Ming-Image-0.1-Design") as demo:
    gr.Markdown(
        """
        # Ming-Image 0.1 Design
        面向 **UI 界面、信息图、海报等文字密集型视觉设计** 的 60 亿参数文生图模型。
        它直接生成包含标题文字的完整画面，并解码为 **RGBA**，透明背景可以完整通过 VAE 往返。

        模型权重：**inclusionAI/Ming-Image-0.1-Design**
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(
                label="提示词",
                value=DEFAULT_PROMPT,
                lines=5,
                placeholder="描述你想要的设计……",
            )
            resolution = gr.Dropdown(
                label="分辨率",
                choices=list(RESOLUTIONS.keys()),
                value="1024 x 1024 (推荐)",
            )
            run = gr.Button("生成", variant="primary")

            with gr.Accordion("高级选项", open=False):
                steps = gr.Slider(4, 24, value=12, step=1, label="采样步数")
                guidance_scale = gr.Slider(
                    0.0, 4.0, value=1.0, step=0.1, label="CFG 引导强度（1.0 为推荐值，0 表示关闭）"
                )
                seed = gr.Number(value=42, label="随机种子", precision=0)

            with gr.Accordion("官方示例（来自模型卡）", open=False):
                gr.Markdown(
                    "作者发布的文生图与透明背景输出示例。第二张图中的棋盘格仅用于预览 alpha 通道。"
                )
                gr.Gallery(
                    value=[
                        ("assets/showcase.webp", "文生图"),
                        ("assets/transparency_showcase.webp", "透明背景"),
                    ],
                    label="Ming-Image-0.1-Design 示例",
                    columns=2,
                    height=260,
                    show_label=False,
                )

        with gr.Column(scale=4):
            out_image = gr.Image(label="生成结果（RGBA）", type="pil", format="png", height=520)
            out_info = gr.Markdown()

    run.click(
        generate,
        inputs=[prompt, resolution, steps, guidance_scale, seed],
        outputs=[out_image, out_info],
    )
    prompt.submit(
        generate,
        inputs=[prompt, resolution, steps, guidance_scale, seed],
        outputs=[out_image, out_info],
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[prompt, resolution, steps, guidance_scale, seed],
        outputs=[out_image, out_info],
        fn=generate,
        label="示例设计需求（英文提示词为模型卡原始示例）",
    )


if __name__ == "__main__":
    demo.queue(max_size=4, default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
    )
