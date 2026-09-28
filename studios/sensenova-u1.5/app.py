from __future__ import annotations

import os

import math
import random

import gradio as gr
import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoTokenizer

import sensenova_u1
from sensenova_u1 import check_checkpoint_compatibility
from sensenova_u1.models.neo_unify.utils import smart_resize

MODEL_ID = "SenseNova/SenseNova-U1.5-8B-MoT"


def resolve_model_source() -> str:
    """Resolve model from ModelScope first, then fall back to HuggingFace."""
    try:
        from modelscope import snapshot_download
        cache_dir = os.getenv("MODELSCOPE_CACHE", "")
        if not cache_dir and os.path.isdir("/mnt/workspace"):
            cache_dir = "/mnt/workspace/modelscope-cache"
        download_kwargs = {"model_id": MODEL_ID}
        if cache_dir:
            download_kwargs["cache_dir"] = cache_dir
        local_path = str(snapshot_download(**download_kwargs))
        print(f"[startup] model resolved from ModelScope: {local_path}")
        return local_path
    except Exception as exc:
        print(f"[startup] ModelScope download failed ({exc}), falling back to HuggingFace")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        return MODEL_ID


MODEL_SOURCE = resolve_model_source()

NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

# Officially recommended infographic settings (model card / examples).
CFG_SCALE = 4.0            # text CFG
IMG_CFG_SCALE = 1.0        # image CFG (editing) — 1.0 = disabled, per the repo default
TIMESTEP_SHIFT = 3.0
NUM_STEPS = 50
CFG_NORM = "global"

# Output H/W must be divisible by patch_size * merge_size = 16 * 2 = 32.
IMAGE_GRID_FACTOR = 32

# Auto-derived editing output / input budgets (from examples/editing/inference.py).
EDIT_TARGET_PIXELS = 2048 * 2048       # output pixels (aspect preserved)
EDIT_INPUT_MAX_PIXELS = 2048 * 2048    # input image budget before vision encoding

# Trained T2I aspect-ratio buckets: aspect_label -> (W, H)
T2I_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1:1": (2048, 2048),
    "16:9": (2720, 1536),
    "9:16": (1536, 2720),
    "3:2": (2496, 1664),
    "2:3": (1664, 2496),
    "4:3": (2368, 1760),
    "3:4": (1760, 2368),
}

MAX_SEED = 2**31 - 1


def _denorm(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(NORM_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def _to_pil(batch: torch.Tensor) -> list[Image.Image]:
    arr = _denorm(batch.float()).permute(0, 2, 3, 1).cpu().numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return [Image.fromarray(a) for a in arr]


def _load_input_image(image: Image.Image) -> Image.Image:
    """Flatten RGBA onto white, convert to RGB, resize to the input pixel budget."""
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg
    image = image.convert("RGB")
    h, w = smart_resize(
        height=image.height,
        width=image.width,
        factor=IMAGE_GRID_FACTOR,
        min_pixels=EDIT_INPUT_MAX_PIXELS,
        max_pixels=EDIT_INPUT_MAX_PIXELS,
    )
    if (w, h) != image.size:
        image = image.resize((w, h), Image.LANCZOS)
    return image


def _output_size_for_edit(image: Image.Image) -> tuple[int, int]:
    """Match the input aspect ratio, normalize total pixels to the target."""
    h, w = smart_resize(
        height=image.height,
        width=image.width,
        factor=IMAGE_GRID_FACTOR,
        min_pixels=EDIT_TARGET_PIXELS,
        max_pixels=EDIT_TARGET_PIXELS,
    )
    return w, h


print("[startup] loading SenseNova-U1.5-8B-MoT (this may take a few minutes)...")
sensenova_u1.set_attn_backend("auto")
print(f"[startup] attn backend: {sensenova_u1.effective_attn_backend()!r}")

config = AutoConfig.from_pretrained(MODEL_SOURCE)
check_checkpoint_compatibility(config)
tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)
model = AutoModel.from_pretrained(MODEL_SOURCE, config=config, torch_dtype=torch.bfloat16).to("cuda").eval()
print("[startup] model ready.")


def generate(
    prompt: str,
    aspect_ratio: str = "1:1",
    seed: int = 42,
    randomize_seed: bool = True,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate an infographic image from a text prompt.

    Args:
        prompt: Description of the infographic (title, sections, data, icons, layout).
        aspect_ratio: One of the trained aspect-ratio buckets (e.g. "1:1", "16:9").
        seed: RNG seed for reproducibility.
        randomize_seed: If True, pick a random seed and ignore `seed`.
    """
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt describing the infographic.")
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    width, height = T2I_RESOLUTIONS[aspect_ratio]
    print(f"[t2i] {width}x{height}, steps={NUM_STEPS}, cfg={CFG_SCALE}, shift={TIMESTEP_SHIFT}, seed={seed}")

    with torch.inference_mode():
        tensor = model.t2i_generate(
            tokenizer,
            prompt,
            image_size=(width, height),
            cfg_scale=CFG_SCALE,
            cfg_norm=CFG_NORM,
            timestep_shift=TIMESTEP_SHIFT,
            cfg_interval=(0.0, 1.0),
            num_steps=NUM_STEPS,
            batch_size=1,
            seed=int(seed),
            think_mode=False,
        )

    return _to_pil(tensor)[0], seed


def edit(
    image: Image.Image,
    prompt: str,
    seed: int = 42,
    randomize_seed: bool = True,
    progress=gr.Progress(track_tqdm=True),
):
    """Edit an existing infographic / image following a natural-language instruction.

    Args:
        image: The input image to edit.
        prompt: The edit instruction (text edits, style transfer, layout tweaks, etc.).
        seed: RNG seed for reproducibility.
        randomize_seed: If True, pick a random seed and ignore `seed`.
    """
    if image is None:
        raise gr.Error("Please upload an image to edit.")
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter an edit instruction.")
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    src = _load_input_image(image)
    width, height = _output_size_for_edit(src)
    print(f"[it2i] out {width}x{height}, steps={NUM_STEPS}, cfg={CFG_SCALE}, img_cfg={IMG_CFG_SCALE}, seed={seed}")

    with torch.inference_mode():
        tensor = model.it2i_generate(
            tokenizer,
            prompt,
            [src],
            image_size=(width, height),
            cfg_scale=CFG_SCALE,
            img_cfg_scale=IMG_CFG_SCALE,
            cfg_norm=CFG_NORM,
            timestep_shift=TIMESTEP_SHIFT,
            cfg_interval=(0.0, 1.0),
            num_steps=NUM_STEPS,
            batch_size=1,
            seed=int(seed),
            think_mode=False,
        )

    return _to_pil(tensor)[0], seed


T2I_EXAMPLES = [
    ["A clean modern data infographic titled \"Global Renewable Energy Mix 2026\" with the subtitle \"Share of total renewable generation\". The main visual is a vertical bar chart with four bars, each labelled with its exact percentage directly above the bar and a category name below it: Solar 38%, Wind 29%, Hydro 21%, Geothermal 12%. Each category has a small flat icon (sun, wind turbine, water drop, volcano) under its label. Light grey horizontal gridlines behind the bars, a teal-to-orange gradient fill, white background, bold sans-serif typography, and a small legend in the top-right corner.", "16:9"],
    ["A circular flow infographic titled \"The Water Cycle\" with four stages arranged in a clockwise loop connected by curved arrows, each stage in its own rounded card containing a flat icon, a bold heading and one short sentence: \"Evaporation — the sun heats water, turning it into vapour\", \"Condensation — vapour cools and forms clouds\", \"Precipitation — water falls as rain or snow\", \"Collection — water gathers in rivers, lakes and oceans\". Soft blue and green palette, white background, clean sans-serif text.", "1:1"],
    ["A vertical step-by-step infographic titled \"How to Brew Pour-Over Coffee\" with five numbered cards stacked from top to bottom, each with a circular number badge, an icon, a bold heading and one line of detail: \"1. Grind — 30 g of medium-coarse coffee\", \"2. Heat — bring water to 93°C\", \"3. Bloom — pour 60 ml and wait 30 seconds\", \"4. Pour — add water slowly in circles up to 500 ml\", \"5. Serve — total brew time about 3 minutes\". Warm brown and cream colours, hand-drawn coffee illustrations, white background, bold sans-serif headings.", "9:16"],
    ["A comparison-table infographic titled \"Electric vs. Petrol Cars\" with two columns headed \"Electric\" (green, with a plug icon) and \"Petrol\" (orange, with a fuel-pump icon), compared across four labelled rows listed on the left: \"Cost per 100 km\", \"CO₂ emissions\", \"Maintenance\", \"Refuel / charge time\". The cells contain explicit values — Electric: \"$3\", \"0 g/km\", \"Low\", \"30 min\"; Petrol: \"$9\", \"180 g/km\", \"High\", \"5 min\". Clean minimalist editorial style, light background, thin dividing lines, bold sans-serif headings.", "16:9"],
    ["A horizontal timeline infographic titled \"Milestones in Computing\" laid out as a single straight line with five evenly spaced markers, each showing a year, a small icon and a one-line caption placed alternately above and below the line: \"1945 — ENIAC, the first general-purpose computer\", \"1971 — Intel 4004, the first microprocessor\", \"1991 — the World Wide Web goes public\", \"2007 — the first iPhone launches\", \"2022 — generative AI reaches the mainstream\". Retro-futuristic colour blocks, bold condensed headings, white background.", "16:9"],
]

EDIT_EXAMPLES = [
    ["assets/pulp.webp", "Replace the text \"Pulp Fiction\" with \"Juicy Reality\" in the exact same font and style."],
    ["assets/edit_necklace.webp", "Add a bouquet of flowers."],
    ["assets/edit_portrait.webp", "Turn the image into an American comic style."],
]

CSS = """
.fillable { max-width: 1150px !important; }
.dark .gradio-container { color: var(--body-text-color); }
"""

with gr.Blocks(title="SenseNova-U1.5-8B-MoT") as demo:
    gr.Markdown(
        """
        # SenseNova-U1.5-8B-MoT 📊 
        
        Unified image generation and editing based on **[SenseNova-U1.5-8B-MoT](https://modelscope.ai/models/SenseNova/SenseNova-U1.5-8B-MoT)**
        A monolithic Any-to-Any (NEO-Unify) model supporting bilingual (Chinese and English) rendering of text-dense infographics, including charts, posters, and knowledge diagrams, as well as natural language instruction-guided editing of existing images (e.g., local text modification, content addition/removal, style, and layout adjustments).

        """
    )

    with gr.Tabs():
        with gr.Tab("Generate (Text → Infographic)"):
            with gr.Row():
                with gr.Column(scale=1):
                    t2i_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe the infographic: title, sections, data values, icons, colors, layout…",
                        lines=6,
                    )
                    t2i_aspect = gr.Dropdown(
                        label="Aspect ratio",
                        choices=list(T2I_RESOLUTIONS.keys()),
                        value="1:1",
                    )
                    with gr.Accordion("Advanced settings", open=False):
                        with gr.Row():
                            t2i_seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=42)
                            t2i_randomize = gr.Checkbox(label="Randomize seed", value=True)
                    t2i_button = gr.Button("Generate infographic", variant="primary")
                with gr.Column(scale=1):
                    t2i_output = gr.Image(label="Output", type="pil", format="png", interactive=False)

            t2i_inputs = [t2i_prompt, t2i_aspect, t2i_seed, t2i_randomize]
            t2i_outputs = [t2i_output, t2i_seed]

            gr.Examples(
                examples=T2I_EXAMPLES,
                inputs=[t2i_prompt, t2i_aspect],
                fn=generate,
                outputs=t2i_outputs,
                cache_examples=True,
                cache_mode="lazy",
            )

            t2i_button.click(fn=generate, inputs=t2i_inputs, outputs=t2i_outputs, api_name="generate")
            t2i_prompt.submit(fn=generate, inputs=t2i_inputs, outputs=t2i_outputs)

        with gr.Tab("Edit (Image + Instruction → Image)"):
            with gr.Row():
                with gr.Column(scale=1):
                    edit_image = gr.Image(label="Input image", type="pil", sources=["upload", "clipboard"])
                    edit_prompt = gr.Textbox(
                        label="Edit instruction",
                        placeholder="e.g. Replace the title text with '2026 Report'; change the style to cyberpunk…",
                        lines=4,
                    )
                    with gr.Accordion("Advanced settings", open=False):
                        with gr.Row():
                            edit_seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=42)
                            edit_randomize = gr.Checkbox(label="Randomize seed", value=True)
                    edit_button = gr.Button("Apply edit", variant="primary")
                with gr.Column(scale=1):
                    edit_output = gr.Image(label="Edited output", type="pil", format="png", interactive=False)

            edit_inputs = [edit_image, edit_prompt, edit_seed, edit_randomize]
            edit_outputs = [edit_output, edit_seed]

            gr.Examples(
                examples=EDIT_EXAMPLES,
                inputs=[edit_image, edit_prompt],
                fn=edit,
                outputs=edit_outputs,
                cache_examples=True,
                cache_mode="lazy",
            )

            edit_button.click(fn=edit, inputs=edit_inputs, outputs=edit_outputs, api_name="edit")


if __name__ == "__main__":
    # Gradio 6 moved app-level params (theme, css) from Blocks() to launch().
    demo.launch(theme=gr.themes.Soft(), css=CSS, mcp_server=True)
