import base64
import os
from pathlib import Path
from modelscope import snapshot_download

# The base image ships flash_attn compiled against an older PyTorch; drop it before anything imports it.
os.system("pip uninstall flash_attn flash-attn -y 2>/dev/null || true")

os.system("pip freeze")

# ---------------------------------------------------------------------------
# Weights come from ModelScope, not the Hub, and are cached on the workspace volume.
# ---------------------------------------------------------------------------
BASE_MODEL_PATH = "/mnt/workspace/Qwen/Qwen-Image-Edit-2509"
MARIGOLD_MODEL_PATH = "/mnt/workspace/huawei-bayerlab/marigold-v2-0"
# Only these are read: the VAE and the transformer of the base model, everything of ours.
# The text encoder is not needed, the prompt embeddings ship with our weights.
BASE_MODEL_PARTS = ("transformer/*", "vae/*", "model_index.json")


def ensure_model(model_id: str, local_dir: str, sentinel: str, parts: tuple[str, ...] = ()) -> None:
    if os.path.exists(os.path.join(local_dir, sentinel)):
        print(f"Model {model_id} found in {local_dir}", flush=True)
        return
    print(f"Downloading {model_id} from ModelScope into {local_dir} ...", flush=True)
    try:
        snapshot_download(model_id, local_dir=local_dir, allow_patterns=list(parts) or None)
    except Exception as exc:
        raise RuntimeError(f"Failed to download {model_id} from ModelScope.") from exc
    if not os.path.exists(os.path.join(local_dir, sentinel)):
        raise RuntimeError(f"Downloaded {model_id}, but {sentinel} is missing in {local_dir}.")
    print(f"Model {model_id} ready", flush=True)


ensure_model("Qwen/Qwen-Image-Edit-2509", BASE_MODEL_PATH, "transformer", BASE_MODEL_PARTS)
ensure_model("huawei-bayerlab/marigold-v2-0", MARIGOLD_MODEL_PATH, "depth/Log-stage2/trainables.safetensors")

import gradio as gr
import torch
from gradio_dualvision import DualVisionApp
from PIL import Image

from marigoldv2_inference import MarigoldV2

DEFAULT_RIGHT_MODALITY = "See-through Depth"

uri_base = os.environ.get("QWEN_IMAGE_EDIT_URI", BASE_MODEL_PATH)
uri_model = os.environ.get("MARIGOLD_V2_URI", MARIGOLD_MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MarigoldV2(uri_base, uri_model, device)


SHIELDS_DIR = Path(__file__).parent / "assets" / "shields"
LINKS = [
    ("website", "Website", "https://www.modelscope.ai/studios/Iceberg-Yang/marigold-v2-web"),
    ("paper", "Paper", "https://www.modelscope.cn/papers/2609.08084"),
    ("code", "Code", "https://github.com/huawei-bayerlab/marigold-v2"),
    ("weights", "Weights", "https://www.modelscope.ai/models/huawei-bayerlab/marigold-v2-0"),
    ("follow", "Follow", "https://twitter.com/antonobukhov1"),
]
# Badges are the committed SVGs, shared with the README, model card, and website. Inlined
# as data URIs so the HTML header needs no static file route.
BADGES = {
    name: "data:image/svg+xml;base64,"
    + base64.b64encode((SHIELDS_DIR / f"{name}.svg").read_bytes()).decode()
    for name, _, _ in LINKS
}
FONT_FILE = Path(__file__).parent / "assets" / "fonts" / "InstrumentSerif-Regular.ttf"
# Google Fonts is not reachable from here, so the title face ships with the repo and is
# inlined into the stylesheet (SIL OFL 1.1, see assets/fonts/OFL.txt).
FONT_CSS = (
    "@font-face { font-family: 'Instrument Serif'; font-style: normal; font-weight: 400; font-display: swap;"
    " src: url(data:font/ttf;base64,"
    + base64.b64encode(FONT_FILE.read_bytes()).decode()
    + ") format('truetype'); }\n"
)
HEADER_CSS = FONT_CSS + """
    .mv2-header { max-width: 70vw; margin: 0 auto; display: grid; grid-template-columns: minmax(0, 6fr) minmax(0, 7fr); gap: 28px; align-items: center; color: var(--body-text-color); }
    .mv2-identity { display: flex; flex-direction: column; gap: 6px; border-right: 1px solid var(--border-color-primary); padding-right: 28px; }
    .mv2-titles { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 14px; }
    .mv2-title { font-family: 'Instrument Serif', Georgia, serif; font-weight: 400; font-size: 36px; line-height: 40px; color: var(--body-text-color); }
    .mv2-subtitle { font-size: 12px; line-height: 16px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--body-text-color-subdued); }
    .mv2-links { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 8px; margin-top: 8px; }
    .mv2-link-group { display: inline-flex; gap: inherit; white-space: nowrap; }
    .mv2-links a { display: inline-flex; line-height: 0; }
    .mv2-links img { height: 22px; width: auto; transition: transform 0.15s ease; }
    .mv2-links a:hover img { transform: translateY(-1px); }
    .mv2-intro { margin: 0; font-size: 14px; line-height: 20px; }
    @media (max-width: 900px) {
        .mv2-header { max-width: none; grid-template-columns: minmax(0, 1fr); gap: 10px; }
        .mv2-identity { display: grid; grid-template-columns: max-content minmax(0, 1fr); column-gap: 16px; row-gap: 4px; align-items: center; border-right: 0; border-bottom: 1px solid var(--border-color-primary); padding: 0 0 10px 0; }
        .mv2-titles { display: contents; }
        .mv2-title { grid-column: 1; grid-row: 1; font-size: 30px; line-height: 34px; }
        .mv2-links { grid-column: 2; grid-row: 1; justify-content: flex-end; gap: 6px 8px; margin: 0; }
        .mv2-subtitle { grid-column: 1 / -1; grid-row: 2; }
        .mv2-intro { font-size: 13px; line-height: 18px; }
    }
    @media (max-width: 529px) {
        .mv2-identity { grid-template-columns: minmax(0, 1fr) max-content; }
        .mv2-subtitle { grid-column: 1; font-size: 10px; letter-spacing: 0.12em; }
        .mv2-links { grid-row: 1 / span 2; align-self: center; flex-direction: column; align-items: flex-end; gap: 6px; }
        .mv2-link-group { gap: 6px; }
        .mv2-links img { height: 20px; }
    }
    @media (max-width: 370px) {
        .mv2-title { font-size: 26px; line-height: 30px; }
    }
    /* Each modality group centered inside its half of the selector row, with a
       hairline on the midline separating the left pane's choices from the right's. */
    #selector_right > .wrap,
    div:has(> #selector_right) > fieldset:first-child > .wrap { justify-content: center; }
    div:has(> #selector_right) { position: relative; }
    div:has(> #selector_right)::before {
        content: "";
        position: absolute;
        top: 3px;
        bottom: 3px;
        left: 50%;
        width: 1px;
        background: var(--border-color-primary);
    }
    /* Footer links wrap as whole items; outside .contain, so plain rules only. */
    footer.svelte-1byz9vf { flex-wrap: wrap; justify-content: center; gap: 2px 24px; }
    footer.svelte-1byz9vf a { white-space: nowrap; }
    footer.svelte-1byz9vf .divider { display: none; }
    /* Examples block as wide as the drop zone, tiles in a full rectangle (column counts that divide 12). */
    .block:has(> .gallery.svelte-p5q82i) { max-width: 70vw; margin-left: auto; margin-right: auto; }
    .gallery.svelte-p5q82i.svelte-p5q82i.svelte-p5q82i { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: var(--spacing-lg); }
    .gallery-item .gallery.svelte-a9zvka { width: 100%; min-width: 0; max-width: none; height: auto; min-height: 0; aspect-ratio: 1; }
    .gallery.svelte-p5q82i .gallery-item img { width: 100%; min-width: 0; max-width: none; height: 100%; min-height: 0; object-fit: cover; }
    @media (max-width: 900px) {
        .block:has(> .gallery.svelte-p5q82i) { max-width: none; }
        .sliderrow .slider { max-width: none; width: 100%; }
        .slider .wrap.half-wrap { width: 100%; }
    }
    @media (max-width: 829px) { .gallery.svelte-p5q82i.svelte-p5q82i.svelte-p5q82i { grid-template-columns: repeat(4, minmax(0, 1fr)); } }
    @media (max-width: 529px) { .gallery.svelte-p5q82i.svelte-p5q82i.svelte-p5q82i { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
"""


class MarigoldV2App(DualVisionApp):
    def make_header(self):
        links = "".join(
            '<span class="mv2-link-group">'
            + "".join(
                f'<a href="{href}" target="_blank" rel="noopener noreferrer">'
                f'<img src="{BADGES[name]}" alt="{label}"></a>'
                for name, label, href in group
            )
            + "</span>"
            for group in (LINKS[:3], LINKS[3:])
        )
        gr.HTML(
            f"""
            <div class="mv2-header">
                <div class="mv2-identity">
                    <div class="mv2-titles">
                        <div class="mv2-title">Marigold V2</div>
                        <div class="mv2-subtitle">Generative Computer Vision</div>
                    </div>
                    <div class="mv2-links remove-elements">{links}</div>
                </div>
                <p class="mv2-intro remove-elements">
                    The authors' demo of <i>Marigold V2: Revisiting Diffusion Transformers for Monocular Depth Estimation</i>.
                    Upload a photo or pick an example below: four models run on it live, predicting depth, see-through depth, normals, and albedo.
                    Then choose any two views with the selectors under the slider, drag to compare them side by side, and zoom in for detail.
                </p>
            </div>
            """,
            padding=False,
        )

    def process_components(
        self, image_in, modality_selector_left, modality_selector_right, **kwargs
    ):
        if modality_selector_right is None:
            modality_selector_right = DEFAULT_RIGHT_MODALITY
        return super().process_components(
            image_in, modality_selector_left, modality_selector_right, **kwargs
        )

    def build_user_components(self):
        return {}

    def process(self, image_in: Image.Image, **kwargs):
        return model(image_in), {}


with MarigoldV2App(
    title="Marigold V2",
    key_original_image="Input",
    examples_path="example_images",
    examples_per_page=12,
    left_selector_visible=True,
    advanced_settings_visible=False,
    squeeze_canvas=True,
    css=HEADER_CSS,
) as demo:
    demo.queue(
        api_open=False,
    ).launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
    )
