# app.py - AnimeGen T2V (ModelScope Studio Edition)
import os
import uuid
import gc
from pathlib import Path

# ── HuggingFace Mirror (set before any HF imports) ──
# On ModelScope, direct access to huggingface.co may be blocked.
# Use hf-mirror.com by default; override via HF_ENDPOINT env var if needed.
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# Speed up downloads (use persistent cache to survive restarts)
PERSISTENT_CACHE = "/mnt/workspace/hf_cache"
os.makedirs(PERSISTENT_CACHE, exist_ok=True)
os.environ["HF_HOME"] = PERSISTENT_CACHE
os.environ["TRANSFORMERS_CACHE"] = PERSISTENT_CACHE

import torch
import gradio as gr

from diffusers import (
    WanPipeline,
    WanTransformer3DModel,
    FlowMatchEulerDiscreteScheduler,
    AutoencoderKLWan,
)
from diffusers.utils import export_to_video

# ── Optional copyright classifier ──
try:
    from copyright_classifier import contains_copyrighted_ip
    COPYRIGHT_CHECK_ENABLED = True
except Exception:
    COPYRIGHT_CHECK_ENABLED = False

# ── Configuration ──
BASE_MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
ANIMEGEN_T2V_REPO = "aidealab/AnimeGen-T2V"

HIGH_NOISE_FILENAME = os.getenv("HIGH_NOISE_FILENAME", "high_noise.safetensors")
LOW_NOISE_FILENAME = os.getenv("LOW_NOISE_FILENAME", "low_noise.safetensors")

# Negative words for prompt filtering (optional, safe defaults)
NG_WORDS = []
for env_var in ("NG_WORD", "NG_WORD_JA"):
    try:
        val = os.getenv(env_var, "[]")
        NG_WORDS.extend(eval(val))
    except Exception:
        pass

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Utility ──
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def download_weights_with_fallback():
    """Download AnimeGen transformer weights, trying ModelScope first then HuggingFace."""
    # Try ModelScope
    try:
        from modelscope import snapshot_download
        snapshot_dir = snapshot_download(ANIMEGEN_T2V_REPO, cache_dir=PERSISTENT_CACHE)
        high = os.path.join(snapshot_dir, HIGH_NOISE_FILENAME)
        low = os.path.join(snapshot_dir, LOW_NOISE_FILENAME)
        if os.path.exists(high) and os.path.exists(low):
            print(f"[ModelScope] Downloaded weights from {ANIMEGEN_T2V_REPO}")
            return high, low
    except Exception as e:
        print(f"[ModelScope] Download failed: {e}")

    # Fallback to HuggingFace (uses persistent cache via HF_HOME)
    try:
        from huggingface_hub import hf_hub_download
        print(f"[HuggingFace] Downloading {HIGH_NOISE_FILENAME} (~28.6GB, may take ~30 min)...")
        high = hf_hub_download(
            repo_id=ANIMEGEN_T2V_REPO,
            filename=HIGH_NOISE_FILENAME,
            cache_dir=PERSISTENT_CACHE,
        )
        print(f"[HuggingFace] Downloading {LOW_NOISE_FILENAME} (~28.6GB, may take ~30 min)...")
        low = hf_hub_download(
            repo_id=ANIMEGEN_T2V_REPO,
            filename=LOW_NOISE_FILENAME,
            cache_dir=PERSISTENT_CACHE,
        )
        print(f"[HuggingFace] Downloaded weights from {ANIMEGEN_T2V_REPO}")
        return high, low
    except Exception as e:
        raise RuntimeError(
            f"Failed to download model weights from both ModelScope and HuggingFace. "
            f"Note: Files are cached to {PERSISTENT_CACHE} — if timeout occurred, "
            f"just restart the studio and it will resume from cache. Error: {e}"
        )


def load_pipeline():
    """Load the full AnimeGen T2V pipeline."""
    high_noise_path, low_noise_path = download_weights_with_fallback()

    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)

    transformer_high = WanTransformer3DModel.from_single_file(
        high_noise_path, torch_dtype=torch.bfloat16,
    )
    transformer_low = WanTransformer3DModel.from_single_file(
        low_noise_path, torch_dtype=torch.bfloat16,
    )

    vae = AutoencoderKLWan.from_pretrained(
        BASE_MODEL_ID, subfolder="vae", torch_dtype=torch.float32,
    )

    pipe = WanPipeline.from_pretrained(
        BASE_MODEL_ID,
        transformer=transformer_high,
        transformer_2=transformer_low,
        scheduler=scheduler,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )

    # Load Lightning LoRA for 4-step acceleration
    pipe.load_lora_weights(
        "lightx2v/Wan2.2-Lightning",
        weight_name="Wan2.2-T2V-A14B-4steps-lora-250928/high_noise_model.safetensors",
        adapter_name="high",
    )
    pipe.load_lora_weights(
        "lightx2v/Wan2.2-Lightning",
        weight_name="Wan2.2-T2V-A14B-4steps-lora-250928/low_noise_model.safetensors",
        adapter_name="low",
        load_into_transformer_2=True,
    )
    pipe.set_adapters(["high", "low"], adapter_weights=[2.0, 1.0])

    # FP8 quantization for memory efficiency
    transformer_high.enable_layerwise_casting(
        storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16,
    )
    transformer_low.enable_layerwise_casting(
        storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16,
    )

    pipe.enable_model_cpu_offload()
    return pipe


def resolve_size(size_name):
    if size_name == "portrait":
        return 480, 832
    return 832, 480


# ── Load pipeline on startup ──
print("Loading AnimeGen T2V pipeline... (this may take several minutes)")
pipe = load_pipeline()
print("Pipeline loaded successfully!")


def generate_video(prompt, negative_prompt, size_name, progress=gr.Progress()):
    clear_memory()

    width, height = resolve_size(size_name)
    num_frames = int(16 * 3 + 1)

    full_prompt = "Japanese anime style, " + prompt.strip()

    # Keyword-based prompt filtering
    prompt_for_check = full_prompt.lower()
    for word in NG_WORDS:
        if word in prompt_for_check:
            raise gr.Error(f"Your prompt contains a restricted word. Please revise your input.")

    # LLM-based copyright filtering (optional)
    if COPYRIGHT_CHECK_ENABLED:
        try:
            if contains_copyrighted_ip(full_prompt):
                raise gr.Error("Your prompt appears to reference copyrighted characters or IPs. Please use original descriptions.")
        except gr.Error:
            raise
        except Exception as e:
            print(f"Copyright check skipped due to error: {e}")

    progress(0.1, desc="Generating video frames...")
    frames = pipe(
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        guidance_scale=1.0,
        num_inference_steps=8,
    ).frames[0]

    progress(0.9, desc="Encoding video...")
    output_path = OUTPUT_DIR / f"{uuid.uuid4().hex}.mp4"
    export_to_video(frames, str(output_path), fps=16)

    clear_memory()
    progress(1.0, desc="Done!")
    return str(output_path)


# ── Default values ──
DEFAULT_PROMPT = (
    "A young girl with blue hair is walking in a cherry blossom park, "
    "petals gently falling around her. The girl wears a school uniform. "
    "The girl is smiling."
)
DEFAULT_NEGATIVE_PROMPT = "3d, cg, photo, stop, wait"

# ── Custom CSS (AIdeaLab-inspired design) ──
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root variables (AIdeaLab light theme) ── */
:root {
    --primary-blue: #3b82f6;
    --primary-cyan: #06b6d4;
    --primary-purple: #8b5cf6;
    --accent-pink: #c084fc;
    --bg-page: #f8fafc;
    --bg-card: #ffffff;
    --bg-input: #f1f5f9;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --text-muted: #94a3b8;
    --border-light: #e2e8f0;
    --gradient-brand: linear-gradient(135deg, #3b82f6, #06b6d4, #8b5cf6);
    --gradient-header: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 40%, #faf5ff 100%);
    --shadow-card: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-hover: 0 4px 12px rgba(59,130,246,0.12);
}

/* ── Global ── */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Noto Sans JP', sans-serif !important;
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 0 !important;
    background: var(--bg-page) !important;
    border-radius: 0 !important;
    border: none !important;
}

.gradio-container > .main {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 0 24px !important;
}

/* ── Header area ── */
.header-section {
    background: var(--gradient-header);
    border-bottom: 1px solid var(--border-light);
    padding: 36px 24px 28px !important;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.header-logo {
    position: absolute;
    top: 20px;
    left: 24px;
    z-index: 10;
}

.header-logo svg {
    height: 32px;
    width: auto;
}

.header-section::before {
    content: '';
    position: absolute;
    top: -60%;
    right: -15%;
    width: 350px;
    height: 350px;
    background: radial-gradient(circle, rgba(6,182,212,0.08) 0%, rgba(139,92,246,0.05) 40%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}

.brand-label {
    font-size: 30px !important;
    font-weight: 600 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    color: var(--primary-blue) !important;
    margin-bottom: 6px !important;
}

.title-main {
    font-size: 38px !important;
    font-weight: 200 !important;
    letter-spacing: -0.5px !important;
    background: var(--gradient-brand);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin-bottom: 6px !important;
    line-height: 1.2 !important;
}

.subtitle-main {
    font-size: 14px !important;
    color: var(--text-secondary) !important;
    font-weight: 400 !important;
    letter-spacing: 0.5px !important;
}

/* ── Card panels ── */
.input-card, .output-card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 12px !important;
    padding: 24px !important;
    box-shadow: var(--shadow-card) !important;
}

.input-card > label, .output-card > label {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}

/* ── Form elements ── */
.gr-textbox textarea, .gr-textbox input {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

.gr-textbox textarea:focus, .gr-textbox input:focus {
    border-color: var(--primary-blue) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
    outline: none !important;
}

.gr-radio label, .gr-checkbox label {
    color: var(--text-secondary) !important;
}

.gr-form {
    background: transparent !important;
    border: none !important;
}

/* ── Generate button ── */
#generate-btn {
    background: linear-gradient(135deg, #3b82f6, #06b6d4) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    padding: 14px 32px !important;
    height: 50px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 8px rgba(59,130,246,0.25) !important;
}

#generate-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(6,182,212,0.35) !important;
    filter: brightness(1.05) !important;
}

/* ── Section labels ── */
.section-label {
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    color: var(--primary-blue) !important;
    margin-bottom: 14px !important;
    padding-bottom: 10px !important;
    border-bottom: 1px solid var(--border-light) !important;
}

/* ── Video output ── */
.gr-video video {
    border-radius: 12px !important;
    border: 1px solid var(--border-light) !important;
    background: var(--bg-card) !important;
}

.gr-video {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
}

/* ── Footer ── */
.footer-section {
    text-align: center;
    padding: 20px !important;
    border-top: 1px solid var(--border-light) !important;
    margin-top: 16px !important;
}

.footer-section a {
    color: var(--primary-blue) !important;
    text-decoration: none !important;
    font-weight: 500 !important;
}

.footer-section a:hover {
    color: var(--primary-cyan) !important;
}

/* ── Markdown overrides ── */
.prose {
    color: var(--text-primary) !important;
}

.prose h1, .prose h2, .prose h3 {
    color: var(--text-primary) !important;
}

/* ── Examples ── */
.gr-examples {
    border: 1px solid var(--border-light) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    background: var(--bg-card) !important;
}

/* ── Labels ── */
.gr-block-label, .gr-form label {
    color: var(--text-secondary) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 3px;
}
"""


# ── Build Gradio UI ──
with gr.Blocks(
    title="AnimeGen T2V | AIdeaLab",
    css=CUSTOM_CSS,
    theme=gr.themes.Default(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ),
) as demo:

    # ── Header ──
    with gr.Column(elem_classes="header-section"):
        gr.HTML(
            '<div class="header-logo">'
            '<svg width="153" height="40" viewBox="0 0 153 40" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<g clip-path="url(#clip0)"><g clip-path="url(#clip1)">'
            '<path d="M21.6796 25.075L15.2823 31.4404C13.5163 33.198 13.5151 36.0493 15.2799 37.8081C17.0446 39.567 19.9075 39.5682 21.6735 37.8106L31.2683 28.262L28.072 25.0763C26.3073 23.3174 23.4444 23.3162 21.6784 25.0738L21.6796 25.075Z" fill="#2F54F7"/>'
            '<path d="M35.748 23.8069C37.514 22.0493 37.5152 19.198 35.7504 17.4391L32.5542 14.2534L29.3555 17.4367C27.5895 19.1943 27.5883 22.0456 29.3531 23.8045C31.1179 25.5633 33.9807 25.5645 35.7467 23.8069H35.748Z" fill="#2F54F7"/>'
            '<path d="M1.32625 17.3221C-0.439758 19.0797 -0.440984 21.931 1.3238 23.6899L10.9112 33.2458L14.1099 30.0625C15.8759 28.3049 15.8771 25.4536 14.1123 23.6948L7.72112 17.3233C5.95634 15.5645 3.09348 15.5632 1.32747 17.3209L1.32625 17.3221Z" fill="#2F54F7"/>'
            '<path d="M21.7924 3.32088C20.0276 1.56203 17.1647 1.5608 15.3987 3.31844L5.80273 12.8658L15.3901 22.4218C17.1549 24.1806 20.0178 24.1818 21.7838 22.4242L28.1811 16.0589C29.9471 14.3012 29.9484 11.45 28.1836 9.6911L21.7924 3.31966V3.32088Z" fill="#2F54F7"/>'
            '<path d="M51.479 11.0264H49.8613C48.1774 14.7272 46.5524 18.2559 44.9886 21.6137C43.4248 24.9715 42.535 26.8659 42.3193 27.2943L43.9861 28.1926C44.0363 28.0681 44.2986 27.4774 44.7741 26.4203C45.2484 25.3621 45.9237 23.8705 46.7987 21.9457H54.1679C54.9841 23.7387 55.641 25.2059 56.1385 26.3459C56.6361 27.4859 56.9021 28.1023 56.9351 28.1926L58.8727 27.1954C58.8065 27.055 57.972 25.2498 56.3714 21.7809C54.7696 18.312 53.1396 14.7259 51.4803 11.0264H51.479ZM47.5071 20.3724L47.5132 20.3602C47.898 19.518 48.3845 18.4341 48.974 17.1097C49.5635 15.7842 50.0672 14.6686 50.4839 13.7592L50.5341 13.7483C51.0709 14.9481 51.5844 16.0967 52.0722 17.1964C52.5612 18.2961 53.0244 19.3507 53.4644 20.3602L53.4693 20.3724H47.5071Z" fill="#2F54F7"/>'
            '<path d="M78.773 9.6875H76.806L76.8427 15.4023C75.9628 15.4023 75.0669 15.4865 74.1551 15.6561C73.2421 15.8258 72.4332 16.0748 71.7286 16.4056C70.5251 16.9512 69.6047 17.7079 68.9662 18.6783C68.3277 19.6487 68.0078 20.7606 68.0078 22.0154C68.0078 23.0883 68.2946 24.0623 68.8669 24.9374C69.4392 25.8126 70.2775 26.5144 71.3805 27.0429C72.2274 27.447 73.0938 27.7179 73.9811 27.8546C74.8684 27.9913 76.0964 28.0585 77.6651 28.0585C77.8232 28.0585 78.034 28.056 78.2999 28.0524C78.5659 28.0487 78.7227 28.0463 78.773 28.0463C78.7644 27.0307 78.7546 25.8187 78.7423 24.4102C78.7301 23.0016 78.7239 21.4271 78.7239 19.6841C78.7239 17.4797 78.7325 15.3034 78.7485 13.1564C78.7644 11.0094 78.773 9.8535 78.773 9.6875ZM76.844 18.2511V19.3399C76.844 21.709 76.8403 23.4727 76.8317 24.6323C76.8231 25.7919 76.8195 26.4082 76.8195 26.4827H76.4959C75.7079 26.4827 74.9542 26.4058 74.2372 26.2532C73.5191 26.1007 72.8744 25.8712 72.3021 25.566C71.6464 25.2194 71.0986 24.7409 70.6587 24.1306C70.2187 23.5203 69.9993 22.7892 69.9993 21.9397C69.9993 20.9901 70.2273 20.1638 70.6844 19.4583C71.1403 18.7528 71.7837 18.2096 72.6134 17.83C73.1367 17.5908 73.7642 17.4004 74.4995 17.26C75.2336 17.1196 76.0155 17.05 76.8452 17.05V18.2511H76.844Z" fill="#2F54F7"/>'
            '<path d="M92.6374 18.3463C92.3801 17.5737 92.0063 16.9231 91.5173 16.3946C91.0773 15.9235 90.5589 15.572 89.9621 15.3413C89.3652 15.1106 88.6801 14.9946 87.9081 14.9946C87.136 14.9946 86.3749 15.1472 85.6494 15.4536C84.9239 15.7599 84.316 16.1932 83.8258 16.7547C83.262 17.4077 82.838 18.1144 82.5561 18.8749C82.2742 19.6353 82.1333 20.486 82.1333 21.4283C82.1333 22.3706 82.322 23.2592 82.6995 24.044C83.077 24.8289 83.5353 25.4733 84.0746 25.9774C84.7045 26.5804 85.5121 27.054 86.495 27.397C87.4779 27.7399 88.6336 27.9108 89.9608 27.9108C90.467 27.9108 90.907 27.8998 91.2807 27.8803C91.6545 27.8596 91.8739 27.8449 91.9401 27.8364L92.1644 26.1874C92.0724 26.1959 91.8298 26.2118 91.4364 26.2374C91.0418 26.2618 90.6091 26.274 90.1361 26.274C89.2733 26.274 88.5343 26.1971 87.9203 26.0446C87.3063 25.892 86.7585 25.6625 86.2781 25.3561C85.5231 24.8765 84.9631 24.271 84.5979 23.5399C84.2327 22.8088 84.0464 22.0044 84.0378 21.1293H93.0357C93.0271 20.0466 92.8948 19.119 92.6374 18.3463ZM84.1861 19.7402C84.4263 18.732 84.8577 17.9326 85.4803 17.3418C86.1028 16.751 86.8786 16.4557 87.8076 16.4557C88.7365 16.4557 89.4853 16.7474 90.054 17.3296C90.6226 17.9118 90.9731 18.7162 91.1055 19.7402H84.1861Z" fill="#2F54F7"/>'
            '<path d="M103.54 16.4554C103.034 16.0014 102.396 15.673 101.629 15.4704C100.862 15.2678 99.9882 15.1665 99.009 15.1665C98.3705 15.1665 97.8655 15.1775 97.4967 15.197C97.1278 15.2178 96.8802 15.2324 96.7564 15.241L96.5567 16.89C97.0052 16.8656 97.3594 16.8448 97.6204 16.8277C97.8815 16.8118 98.1989 16.8033 98.5727 16.8033C99.1695 16.8033 99.6928 16.8338 100.141 16.8961C100.59 16.9583 100.996 17.0633 101.361 17.2122C101.891 17.427 102.295 17.7407 102.568 18.1545C102.841 18.5683 103.003 19.0968 103.053 19.7412C101.9 19.8328 100.873 19.967 99.9735 20.144C99.0727 20.3222 98.3533 20.5224 97.8141 20.7458C96.9513 21.101 96.2773 21.566 95.7919 22.1397C95.3066 22.7146 95.064 23.3932 95.064 24.1793C95.064 24.9226 95.2699 25.5598 95.6804 26.0883C96.091 26.6168 96.6278 27.0342 97.292 27.3406C97.7896 27.5725 98.3827 27.7495 99.0715 27.874C99.7602 27.9985 100.444 28.0595 101.125 28.0595C102.213 28.0595 103.1 28.0327 103.789 27.9789C104.477 27.9252 104.863 27.8947 104.946 27.8862C104.946 27.7873 104.949 27.0928 104.958 25.8039C104.966 24.5149 104.97 22.8452 104.97 20.7958C104.97 19.5972 104.85 18.6818 104.61 18.0507C104.37 17.4185 104.013 16.8875 103.54 16.4579V16.4554ZM103.067 23.7814C103.067 24.6077 103.064 25.2644 103.061 25.7526C103.057 26.2408 103.054 26.513 103.054 26.5704C102.789 26.5789 102.536 26.5875 102.296 26.5948C102.056 26.6034 101.852 26.607 101.686 26.607C101.138 26.607 100.617 26.5765 100.123 26.5142C99.6291 26.452 99.1793 26.3507 98.7725 26.2103C98.2418 26.0285 97.8227 25.766 97.515 25.423C97.2087 25.0801 97.0542 24.6272 97.0542 24.0658C97.0542 23.5531 97.2344 23.1235 97.5959 22.7768C97.9562 22.4302 98.3693 22.169 98.8337 21.9957C99.4563 21.7564 100.223 21.5721 101.137 21.444C102.05 21.3158 102.688 21.2438 103.053 21.2267C103.062 21.7882 103.066 22.2849 103.066 22.7146V23.7801L103.067 23.7814Z" fill="#2F54F7"/>'
            '<path d="M114.604 26.136C114.214 26.136 113.778 26.1335 113.298 26.1299C112.816 26.1262 112.405 26.1238 112.066 26.1238C112.066 26.0408 112.062 25.273 112.054 23.8205C112.045 22.3668 112.041 20.5555 112.041 18.3828C112.041 17.4906 112.045 16.3481 112.054 14.9518C112.062 13.5567 112.074 12.3263 112.09 11.2607H110.036C110.045 11.7563 110.057 12.6205 110.073 13.8508C110.089 15.0824 110.098 16.8718 110.098 19.2177C110.098 20.6226 110.092 22.0543 110.079 23.5129C110.067 24.9715 110.052 26.4118 110.035 27.8326C110.641 27.8326 111.381 27.8289 112.257 27.8204C113.132 27.8118 114.055 27.8082 115.027 27.8082C116.504 27.8082 117.717 27.8118 118.667 27.8204C119.616 27.8289 120.141 27.8326 120.241 27.8326V26.0969C120.167 26.0969 119.58 26.103 118.48 26.1152C117.381 26.1274 116.088 26.1335 114.604 26.1335V26.136Z" fill="#2F54F7"/>'
            '<path d="M130.373 16.4554C129.867 16.0014 129.23 15.673 128.463 15.4704C127.695 15.2678 126.822 15.1665 125.842 15.1665C125.204 15.1665 124.699 15.1775 124.33 15.197C123.961 15.2178 123.714 15.2324 123.59 15.241L123.39 16.89C123.839 16.8656 124.193 16.8448 124.454 16.8277C124.715 16.8118 125.032 16.8033 125.406 16.8033C126.003 16.8033 126.526 16.8338 126.975 16.8961C127.423 16.9583 127.829 17.0633 128.194 17.2122C128.725 17.427 129.128 17.7407 129.401 18.1545C129.675 18.5683 129.837 19.0968 129.887 19.7412C128.734 19.8328 127.707 19.967 126.807 20.144C125.906 20.3222 125.187 20.5224 124.648 20.7458C123.785 21.101 123.111 21.566 122.625 22.1397C122.14 22.7146 121.897 23.3932 121.897 24.1793C121.897 24.9226 122.103 25.5598 122.514 26.0883C122.924 26.6168 123.461 27.0342 124.125 27.3406C124.623 27.5725 125.216 27.7495 125.905 27.874C126.594 27.9985 127.278 28.0595 127.959 28.0595C129.046 28.0595 129.933 28.0327 130.622 27.9789C131.311 27.9252 131.697 27.8947 131.779 27.8862C131.779 27.7873 131.783 27.0928 131.791 25.8039C131.8 24.5149 131.804 22.8452 131.804 20.7958C131.804 19.5972 131.683 18.6818 131.443 18.0507C131.203 17.4185 130.846 16.8875 130.373 16.4579V16.4554ZM129.9 23.7814C129.9 24.6077 129.898 25.2644 129.894 25.7526C129.89 26.2408 129.888 26.513 129.888 26.5704C129.622 26.5789 129.37 26.5875 129.129 26.5948C128.889 26.6034 128.686 26.607 128.519 26.607C127.971 26.607 127.45 26.5765 126.956 26.5142C126.463 26.452 126.013 26.3507 125.606 26.2103C125.075 26.0285 124.656 25.766 124.349 25.423C124.042 25.0801 123.888 24.6272 123.888 24.0658C123.888 23.5531 124.068 23.1235 124.429 22.7768C124.79 22.4302 125.203 22.169 125.667 21.9957C126.29 21.7564 127.057 21.5721 127.97 21.444C128.883 21.3158 129.522 21.2438 129.887 21.2267C129.895 21.7882 129.899 22.2849 129.899 22.7146V23.7801L129.9 23.7814Z" fill="#2F54F7"/>'
            '<path d="M146.44 18.9004C146.066 18.1449 145.527 17.5151 144.822 17.0122C144.142 16.5252 143.289 16.1395 142.265 15.8539C141.24 15.5695 139.877 15.4181 138.176 15.4023L138.213 9.6875H136.246C136.263 11.1412 136.276 12.5656 136.284 13.9608C136.293 15.3559 136.296 16.7974 136.296 18.2841C136.296 20.7118 136.29 22.7892 136.278 24.5151C136.266 26.241 136.255 27.4225 136.247 28.0585H137.355C138.682 28.0585 139.82 27.9865 140.766 27.8412C141.712 27.6972 142.554 27.4677 143.293 27.154C144.429 26.6756 145.332 25.9493 146 24.9741C146.668 24 147.001 22.8356 147.001 21.482C147.001 20.5165 146.815 19.656 146.441 18.9004H146.44ZM144.424 23.9768C144.034 24.6665 143.441 25.2206 142.645 25.6417C142.097 25.931 141.446 26.1434 140.691 26.2789C139.936 26.4156 139.102 26.4827 138.19 26.4827C138.19 25.8968 138.186 25.3195 138.178 24.7495C138.169 24.1807 138.165 23.2067 138.165 21.8286C138.165 21.0938 138.168 20.1601 138.171 19.0249C138.175 17.8898 138.178 17.2319 138.178 17.05H138.788C139.36 17.05 139.917 17.094 140.456 17.1794C140.995 17.2661 141.498 17.3881 141.962 17.5444C142.858 17.8495 143.591 18.3402 144.158 19.0176C144.727 19.6938 145.011 20.5324 145.011 21.5308C145.011 22.4719 144.816 23.286 144.427 23.9756L144.424 23.9768Z" fill="#2F54F7"/>'
            '<path d="M61.0469 13.3238H61.9734C61.9832 14.2416 61.9918 15.1424 61.9954 16.0225C62.004 17.5519 62.0077 18.7871 62.0077 19.7282C62.0077 22.1083 61.9991 23.9343 61.9832 25.2074C61.9807 25.4148 61.9783 25.6114 61.9758 25.7981H61.0469V27.8646H64.9V25.7981H63.9722C63.9698 25.5662 63.9661 25.3172 63.9637 25.0523C63.9465 23.6767 63.9391 21.7324 63.9391 19.2204C63.9391 18.2293 63.9453 16.9501 63.9575 15.3841C63.9637 14.6554 63.9698 13.9695 63.9759 13.3238H64.9V11.2573H61.0469V13.3238Z" fill="#2F54F7"/>'
            '</g></g>'
            '<defs>'
            '<clipPath id="clip0"><rect width="153" height="38.6443" fill="white" transform="translate(0 0.798828)"/></clipPath>'
            '<clipPath id="clip1"><rect width="147" height="37.1289" fill="white" transform="translate(0 2)"/></clipPath>'
            '</defs></svg>'
            '</div>',
        )
        gr.Markdown(
            '<p class="brand-label">AnimeGen-T2V</p>',
        )
        gr.Markdown(
            '<h1 class="title-main">AnimeGen T2V</h1>',
        )
        gr.Markdown(
            '<p class="subtitle-main">Generate stunning anime-style videos from text descriptions</p>',
        )

    gr.Markdown("<br>")

    # ── Main content ──
    with gr.Row(equal_height=False):
        # Left: Input panel
        with gr.Column(scale=1, elem_classes="input-card"):
            gr.Markdown("### Input", elem_classes="section-label")

            prompt = gr.Textbox(
                label="Prompt",
                value=DEFAULT_PROMPT,
                lines=4,
                placeholder="Describe the anime scene you want to generate...",
            )

            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value=DEFAULT_NEGATIVE_PROMPT,
                lines=2,
                placeholder="What to avoid in the output...",
            )

            size_name = gr.Radio(
                label="Aspect Ratio",
                choices=[("Landscape (832x480)", "landscape"), ("Portrait (480x832)", "portrait")],
                value="landscape",
            )

            generate_button = gr.Button(
                "Generate Video",
                variant="primary",
                elem_id="generate-btn",
            )

        # Right: Output panel
        with gr.Column(scale=1, elem_classes="output-card"):
            gr.Markdown("### Output", elem_classes="section-label")
            video = gr.Video(label="Generated Video", height=420)

    # ── Examples ──
    gr.Markdown("<br>")
    gr.Examples(
        examples=[
            [
                "A young girl with blue hair is walking in a cherry blossom park, petals gently falling around her. The girl wears a school uniform. The girl is smiling.",
                DEFAULT_NEGATIVE_PROMPT,
                "landscape",
            ],
            [
                "A brave samurai standing on a cliff overlooking a vast ocean at sunset, wind blowing his cloak dramatically. Cinematic lighting.",
                DEFAULT_NEGATIVE_PROMPT,
                "landscape",
            ],
            [
                "A magical cat wizard casting a spell in an enchanted forest, glowing particles swirling around. Fantasy anime style.",
                DEFAULT_NEGATIVE_PROMPT,
                "portrait",
            ],
            [
                "Two friends running through a neon-lit cyberpunk city street at night, rain reflecting colorful lights. Dynamic camera angle.",
                DEFAULT_NEGATIVE_PROMPT,
                "landscape",
            ],
        ],
        inputs=[prompt, negative_prompt, size_name],
        label="Try these examples",
    )

    # ── Footer ──
    gr.Markdown(
        '<div class="footer-section">'
        '<span style="color:#a0a0b8;font-size:13px;">Powered by </span>'
        '<a href="https://aidealab.com/" target="_blank" style="font-size:13px;font-weight:500;">AIdeaLab</a>'
        '<span style="color:#a0a0b8;font-size:13px;"> & Wan2.2-T2V-A14B</span>'
        '</div>',
    )

    # ── Event binding ──
    generate_button.click(
        fn=generate_video,
        inputs=[prompt, negative_prompt, size_name],
        outputs=[video],
    )


if __name__ == "__main__":
    demo.queue(max_size=10).launch(server_name="0.0.0.0", server_port=7860)
