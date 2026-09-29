import base64
import os
import sys
from pathlib import Path
from modelscope import snapshot_download

# Remove flash_attn - compiled against old PyTorch, incompatible with 2.5+
os.system("pip uninstall flash_attn flash-attn -y 2>/dev/null || true")

os.system("pip freeze")

# ============================================================
# Check and download models from ModelScope if not exists
# ============================================================
print("=" * 60)
print("Checking models...")
print("=" * 60)

def download_model(model_id: str, local_dir: str, patterns: tuple[str, ...] = ()) -> None:
    try:
        snapshot_download(model_id, local_dir=local_dir, allow_patterns=list(patterns) or None)
    except Exception as exc:
        raise RuntimeError(f"Failed to download {model_id} from ModelScope.") from exc


# Model paths
base_model_path = "/mnt/workspace/Qwen/Qwen-Image-Edit-2509"
BASE_MODEL_PARTS = ("transformer/*", "vae/*", "model_index.json")
lora_model_path = "/mnt/workspace/huawei-bayerlab/windowseat-reflection-removal-v1-0"

# Check base model (Qwen-Image-Edit-2509)
if not os.path.exists(base_model_path) or not os.path.exists(os.path.join(base_model_path, "transformer")):
    print("\n📥 Downloading Qwen-Image-Edit-2509 model from ModelScope...")
    download_model("Qwen/Qwen-Image-Edit-2509", base_model_path, BASE_MODEL_PARTS)
    print("✓ Qwen-Image-Edit-2509 model downloaded successfully!")
else:
    print("✓ Qwen-Image-Edit-2509 model found")

# Check WindowSeat LoRA model
lora_check_file = os.path.join(lora_model_path, "model_index.json")
if not os.path.exists(lora_check_file):
    print("\n📥 Downloading WindowSeat LoRA model from ModelScope...")
    download_model("huawei-bayerlab/windowseat-reflection-removal-v1-0", lora_model_path)
    print("✓ WindowSeat LoRA model downloaded successfully!")
else:
    print("✓ WindowSeat LoRA model found")

print("=" * 60)
print("✓ All models ready!")
print("=" * 60)

import tempfile
import shutil

import gradio as gr
import torch as torch
from gradio_dualvision import DualVisionApp
from PIL import Image
from windowseat_inference import load_network, run_inference

# Use local paths instead of HuggingFace URIs
uri_base = base_model_path
uri_lora = lora_model_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

vae, transformer, embeds_dict, processing_resolution = load_network(uri_base, uri_lora, device)

# # As of transformers==4.57.1 , xformers is not supported in QwenImageTransformer2DModel
# try:
#     transformer.enable_xformers_memory_efficient_attention()
#     print("xformers enabled")
# except:
#     print("xformers not enabled")


SHIELDS_DIR = Path(__file__).parent / "assets" / "shields"
LINKS = [
    ("website", "Website", "https://www.modelscope.ai/studios/Iceberg-Yang/windowseat-reflection-removal-web"),
    ("paper", "Paper", "https://www.modelscope.cn/papers/2512.05000"),
    ("code", "Code", "https://github.com/huawei-bayerlab/windowseat-reflection-removal"),
    ("weights", "Weights", "https://www.modelscope.ai/models/huawei-bayerlab/windowseat-reflection-removal-v1-0"),
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

    .ws-header { max-width: 70vw; margin: 0 auto; display: grid; grid-template-columns: minmax(0, 6fr) minmax(0, 7fr); gap: 28px; align-items: center; color: var(--body-text-color); }
    .ws-identity { display: flex; flex-direction: column; gap: 6px; border-right: 1px solid var(--border-color-primary); padding-right: 28px; }
    .ws-titles { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 14px; }
    .ws-title { font-family: 'Instrument Serif', Georgia, serif; font-weight: 400; font-size: 36px; line-height: 40px; color: var(--body-text-color); }
    .ws-subtitle { font-size: 12px; line-height: 16px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--body-text-color-subdued); }
    .ws-links { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 8px; margin-top: 8px; }
    .ws-link-group { display: inline-flex; gap: inherit; white-space: nowrap; }
    .ws-links a { display: inline-flex; line-height: 0; }
    .ws-links img { height: 22px; width: auto; transition: transform 0.15s ease; }
    .ws-links a:hover img { transform: translateY(-1px); }
    .ws-intro { margin: 0; font-size: 14px; line-height: 20px; }
    @media (max-width: 900px) {
        .ws-header { max-width: none; grid-template-columns: minmax(0, 1fr); gap: 10px; }
        .ws-identity { display: grid; grid-template-columns: max-content minmax(0, 1fr); column-gap: 16px; row-gap: 4px; align-items: center; border-right: 0; border-bottom: 1px solid var(--border-color-primary); padding: 0 0 10px 0; }
        .ws-titles { display: contents; }
        .ws-title { grid-column: 1; grid-row: 1; font-size: 30px; line-height: 34px; }
        .ws-links { grid-column: 2; grid-row: 1; justify-content: flex-end; gap: 6px 8px; margin: 0; }
        .ws-subtitle { grid-column: 1 / -1; grid-row: 2; }
        .ws-intro { font-size: 13px; line-height: 18px; }
    }
    @media (max-width: 529px) {
        .ws-identity { grid-template-columns: minmax(0, 1fr) max-content; }
        .ws-subtitle { grid-column: 1; font-size: 10px; letter-spacing: 0.12em; }
        .ws-links { grid-row: 1 / span 2; align-self: center; flex-direction: column; align-items: flex-end; gap: 6px; }
        .ws-link-group { gap: 6px; }
        .ws-links img { height: 20px; }
    }
    @media (max-width: 370px) {
        .ws-title { font-size: 26px; line-height: 30px; }
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


class WindowSeatApp(DualVisionApp):
    DEFAULT_SEED = 2025

    def make_header(self):
        links = "".join(
            '<span class="ws-link-group">'
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
            <div class="ws-header">
                <div class="ws-identity">
                    <div class="ws-titles">
                        <div class="ws-title">WindowSeat</div>
                        <div class="ws-subtitle">Image Reflection Removal</div>
                    </div>
                    <div class="ws-links remove-elements">{links}</div>
                </div>
                <p class="ws-intro remove-elements">
                    The authors' demo of <i>Reflection Removal through Efficient Adaptation of Diffusion Transformers</i>.
                    Upload a photo through glass or pick an example below, wait for the result,
                    then drag the slider to compare it with the input and zoom in for detail.
                </p>
            </div>
            """,
            padding=False,
        )

    def build_user_components(self):
        return {}

    def process(self, image_in: Image.Image, **kwargs):
        input_temp_dir = tempfile.mkdtemp()
        output_temp_dir = tempfile.mkdtemp()

        try:
            input_image_path = os.path.join(input_temp_dir, "image.png")
            image_in.save(input_image_path)
            run_inference(
                vae,
                transformer,
                embeds_dict,
                processing_resolution,
                input_temp_dir,
                output_temp_dir,
                use_short_edge_tile=True,
                save_comparison=False,
                save_alternating=False,
            )
            output_image_path = os.path.join(output_temp_dir, "image_windowseat_output.png")
            result_image = Image.open(output_image_path)
            result_image.load()

            out_modalities = {
                "Result": result_image,
            }

            out_settings = {}

            return out_modalities, out_settings

        finally:
            if os.path.exists(input_temp_dir):
                shutil.rmtree(input_temp_dir)
            if os.path.exists(output_temp_dir):
                shutil.rmtree(output_temp_dir)


with WindowSeatApp(
    title="WindowSeat Reflection Removal",
    examples_path="example_images",
    examples_per_page=12,
    right_selector_visible=False,
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
