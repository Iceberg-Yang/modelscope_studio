"""FireRedTTS3 — unified speech generation & editing demo (ModelScope Studio).

Three capabilities of https://huggingface.co/FireRedTeam/FireRedTTS3 :
  * Zero-shot voice cloning  (FireRedTTS3-Base,     24 languages + 21 ZH dialects)
  * Voice design             (FireRedTTS3-Instruct, natural-language timbre prompt)
  * Speech editing           (FireRedTTS3-Instruct, semantic + acoustic)

Migrated from https://huggingface.co/spaces/hugging-apps/firered-tts3 :
ZeroGPU scheduling removed, weights served from ModelScope, Apache-2.0.
"""

import functools
import os
import re

import numpy as np
import soundfile as sf
import torch
import gradio as gr
from modelscope import snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# Text front-end assets
# --------------------------------------------------------------------------- #
# fastText lid.176 powers automatic language routing (see upstream README).
# The model file ships with this repo; no runtime download.
_LID_PATH = os.path.join(HERE, "fireredtts3", "utils", "llm_tn", "models", "lid.176.ftz")
if not os.path.exists(_LID_PATH):
    print(f"[WARN] fastText lid.176 missing at {_LID_PATH}; "
          "language auto-detect will fall back to heuristics.", flush=True)

# The upstream llm_tn TextNormalizer refuses to construct without API creds, and
# the fastText language detector lives on that object. We only use it for
# language *identification* (use_llm_tn=False -> local wetext TN), so give it
# placeholder creds and disable its LLM fallback further below.
os.environ.setdefault("LLM_TN_API_URL", "http://127.0.0.1:1/unused")
os.environ.setdefault("LLM_TN_API_KEY", "unused")

# --------------------------------------------------------------------------- #
# Weights (ModelScope mirror of FireRedTeam/FireRedTTS3)
# --------------------------------------------------------------------------- #
MODEL_REPO = "FireRedTeam/FireRedTTS3"
# Cache under the persistent volume so restarts skip the ~21 GB download.
_PERSIST_ROOT = "/mnt/workspace"
_CACHE_DIR = (
    os.path.join(_PERSIST_ROOT, "fireredtts3_cache")
    if os.path.isdir(_PERSIST_ROOT) and os.access(_PERSIST_ROOT, os.W_OK)
    else None
)
MODEL_DIR = snapshot_download(MODEL_REPO, cache_dir=_CACHE_DIR)
print(f"[INFO] weights at {MODEL_DIR}", flush=True)

from fireredtts3.core import FireRedTTS3, FireRedTTS3Instruct  # noqa: E402
from fireredtts3.redae.redae import RedAE  # noqa: E402
from fireredtts3.utils.llm_tn.text_normalizer import TextNormalizer  # noqa: E402
from fireredtts3.utils.text_tokenizer import (  # noqa: E402
    MULTI_DIALECT_TAGS,
    MULTI_LANG_TAGS,
)

# Base and Instruct each instantiate their own RedAE from the very same
# checkpoint; share one instance instead (~3.8 GB saved, identical numerics).
_redae_real_from_pretrained = RedAE.from_pretrained
_redae_singleton = None


def _shared_redae(*args, **kwargs):
    global _redae_singleton
    if _redae_singleton is None:
        _redae_singleton = _redae_real_from_pretrained(*args, **kwargs)
    return _redae_singleton


RedAE.from_pretrained = _shared_redae

tts = FireRedTTS3(MODEL_DIR, use_fasttext=True, use_llm_tn=False, use_wetext=True)
instruct = FireRedTTS3Instruct(MODEL_DIR, use_fasttext=True, use_llm_tn=False, use_wetext=True)

for _pipe in (tts, instruct):
    _norm = getattr(_pipe, "_llm_tn", None)
    if _norm is not None:
        # No API creds here -> never let language ID fall back to an LLM call.
        _norm.detect_locale = functools.partial(
            TextNormalizer.detect_locale, _norm, use_llm_fallback=False
        )
print("[INFO] FireRedTTS3 Base + Instruct ready", flush=True)

SAMPLE_RATE = tts.redae.sample_rate

# --------------------------------------------------------------------------- #
# Language choices
# --------------------------------------------------------------------------- #
AUTO = "Auto-detect"
LANGUAGES = [t.strip("<|>") for t in MULTI_LANG_TAGS]
DIALECTS = [t.strip("<|>") for t in MULTI_DIALECT_TAGS]
LANG_CHOICES = (
    [AUTO]
    + LANGUAGES
    + [f"{d} (Chinese dialect)" for d in DIALECTS]
)


def _resolve_language(choice: str):
    if not choice or choice == AUTO:
        return None
    return choice.split(" (")[0]


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #
MAX_PROMPT_SECONDS = 20.0
MAX_EDIT_SECONDS = 20.0
MAX_TEXT_CHARS = 400


def _load_audio(path: str, max_seconds: float):
    if not path:
        raise gr.Error("Please provide an audio file first.")
    wav, sr = sf.read(path, always_2d=True, dtype="float32")
    wav = wav[:, 0]
    if wav.shape[0] > int(max_seconds * sr):
        wav = wav[: int(max_seconds * sr)]
        gr.Info(f"Audio truncated to the first {max_seconds:.0f}s.")
    peak = float(np.abs(wav).max()) if wav.size else 0.0
    if peak > 0:
        wav = wav / peak * 0.95
    return torch.from_numpy(np.ascontiguousarray(wav)[None, :]), sr


def _to_gradio_audio(audio: torch.Tensor, sr: int):
    x = audio.detach().float().cpu().numpy()
    if x.ndim > 1:
        x = x[0]
    x = np.clip(x, -1.0, 1.0)
    return sr, (x * 32767.0).astype(np.int16)


_EDIT_MASK_RE = re.compile(r"<\|edit\|>(?:<\|frame_patch\|>)*<\|end_edit\|>")


def _pretty_edit_text(text: str) -> str:
    """The model marks the re-synthesized span with edit/frame-patch tokens."""
    text = _EDIT_MASK_RE.sub(" ⟨edited span⟩ ", text or "")
    text = re.sub(r"<\|[^|]*\|>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _check_text(text: str, what: str = "Text"):
    text = (text or "").strip()
    if not text:
        raise gr.Error(f"{what} must not be empty.")
    if len(text) > MAX_TEXT_CHARS:
        gr.Info(f"{what} truncated to {MAX_TEXT_CHARS} characters.")
        text = text[:MAX_TEXT_CHARS]
    return text


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def voice_clone(
    prompt_audio,
    prompt_text,
    text,
    language=AUTO,
    inference_cfg=2.0,
    n_timesteps=10,
    seed=1234,
    do_tn=True,
):
    """FireRedTTS3-Base zero-shot voice cloning."""
    text = _check_text(text, "Text to synthesize")
    prompt_text = (prompt_text or "").strip()
    if not prompt_text:
        raise gr.Error("Please provide the transcript of the reference audio.")
    wav, sr = _load_audio(prompt_audio, MAX_PROMPT_SECONDS)

    gen_audio, gen_sr = tts.generate(
        text=text,
        language=_resolve_language(language),
        prompt_text=prompt_text,
        prompt_audio=wav,
        prompt_audio_sr=sr,
        n_timesteps=int(n_timesteps),
        inference_cfg=float(inference_cfg),
        seed=int(seed),
        do_tn=bool(do_tn),
    )
    return _to_gradio_audio(gen_audio, gen_sr)


def voice_design(
    instruction,
    text,
    inference_cfg=1.2,
    n_timesteps=10,
    seed=2,
    do_tn=True,
):
    """FireRedTTS3-Instruct voice design (no reference audio)."""
    instruction = _check_text(instruction, "Voice description")
    text = _check_text(text, "Text to synthesize")

    gen_audio, gen_sr, gen_text = instruct.generate_voice_design(
        instruction=instruction,
        text=text,
        n_timesteps=int(n_timesteps),
        inference_cfg=float(inference_cfg),
        seed=int(seed),
        do_tn=bool(do_tn),
    )
    return _to_gradio_audio(gen_audio, gen_sr), (gen_text or "").strip()


def semantic_edit(
    audio_in,
    instruction,
    inference_cfg=1.2,
    n_timesteps=10,
    seed=1234,
):
    """FireRedTTS3-Instruct content editing: insert / delete / substitute."""
    instruction = _check_text(instruction, "Edit instruction")
    wav, sr = _load_audio(audio_in, MAX_EDIT_SECONDS)

    gen_audio, gen_sr, gen_text = instruct.generate_semantic_edit(
        instruction=instruction,
        audio_in=wav,
        audio_in_sr=sr,
        n_timesteps=int(n_timesteps),
        inference_cfg=float(inference_cfg),
        seed=int(seed),
    )
    return _to_gradio_audio(gen_audio, gen_sr), _pretty_edit_text(gen_text)


def compose_acoustic_instruction(attribute: str, value: float) -> str:
    """Acoustic edits only accept the templates the model was trained on."""
    if attribute == "Speed":
        return f"adjust the speed to {value:.1f}x"
    if attribute == "Volume":
        return f"adjust the volume to {value:.1f}"
    steps = int(round(value))
    return f"shift the pitch by {steps} step{'' if abs(steps) == 1 else 's'}"


def acoustic_edit(
    audio_in,
    attribute="Speed",
    value=0.8,
    inference_cfg=1.2,
    n_timesteps=10,
    seed=1234,
):
    """FireRedTTS3-Instruct acoustic editing: speed / pitch / volume."""
    wav, sr = _load_audio(audio_in, MAX_EDIT_SECONDS)
    instruction = compose_acoustic_instruction(attribute, float(value))

    gen_audio, gen_sr = instruct.generate_acoustic_edit(
        instruction=instruction,
        audio_in=wav,
        audio_in_sr=sr,
        n_timesteps=int(n_timesteps),
        inference_cfg=float(inference_cfg),
        seed=int(seed),
    )
    return _to_gradio_audio(gen_audio, gen_sr), instruction


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
EN_PROMPT = os.path.join(HERE, "examples", "en_prompt.wav")
ZH_PROMPT = os.path.join(HERE, "examples", "zh_prompt.wav")
EN_PROMPT_TEXT = (
    "Just by listening a few minutes a day, you'll be able to eliminate negative "
    "thoughts by conditioning your mind to be more positive."
)
ZH_PROMPT_TEXT = "比如具体一点的，他觉得最大的一个跟他预想的不一样的是在什么地方。"

# Example rows shown via gr.Dataframe + .select(). gr.Examples/gr.Dataset is
# avoided on purpose: the gradio 6.x frontend only renders the FIRST Dataset
# component in an app, leaving every later Examples table blank.
CLONE_EXAMPLES = [
    [
        EN_PROMPT,
        EN_PROMPT_TEXT,
        "FireRedTTS3 turns a handful of seconds of speech into a voice "
        "that can read anything you write.",
        "English",
    ],
    [
        ZH_PROMPT,
        ZH_PROMPT_TEXT,
        "法院与不动产登记部门加强沟通，并督促银行提前办理抵押预约登记。",
        "Chinese",
    ],
    [
        EN_PROMPT,
        EN_PROMPT_TEXT,
        "Le modèle peut aussi parler français avec la même voix de référence.",
        "French",
    ],
]
DESIGN_EXAMPLES = [
    [
        "一个年轻女性的温柔嗓音，语速稍慢，带一点俏皮。",
        "今天天气很好，我们一起去公园散步吧。",
    ],
    [
        "An old sailor with a deep, gravelly voice, speaking slowly and "
        "warmly, as if telling a story by the fire.",
        "The sea was calm that morning, and every rope on deck was "
        "still wet with salt.",
    ],
    [
        "A bright, energetic young man hosting a sports broadcast, fast "
        "paced and excited.",
        "And with ten seconds left on the clock, he takes the shot — "
        "and it is in!",
    ],
]
SEM_EXAMPLES = [
    [EN_PROMPT, "Replace 'positive' with 'optimistic'."],
    [EN_PROMPT, "Delete the word 'negative'."],
    [ZH_PROMPT, "把“最大的”替换成“最有意思的”。"],
]
ACO_EXAMPLES = [
    [EN_PROMPT, "Speed", 0.7],
    [ZH_PROMPT, "Pitch", 2],
    [EN_PROMPT, "Volume", 1.6],
]

CSS = """
.gradio-container {max-width: 1200px !important; margin: auto !important;}
.dark .gradio-container {color: var(--body-text-color);}
"""

with gr.Blocks(title="FireRedTTS3") as demo:
    gr.Markdown(
        """
        # 🔥 FireRedTTS3 — Unified Speech Generation & Editing
        Zero-shot voice cloning in **24 languages + 21 Chinese dialects**, natural-language
        **voice design**, and instruction-driven **speech editing** — all from
        [FireRedTeam/FireRedTTS3](https://modelscope.cn/models/FireRedTeam/FireRedTTS3).
        """
    )

    with gr.Tabs():
        # ------------------------------------------------------------------ #
        with gr.Tab("🎙️ Voice Cloning"):
            gr.Markdown(
                "Clone any voice from a short reference clip. For best quality the "
                "reference should be spoken in the **same language / dialect** as the "
                "text you synthesize."
            )
            with gr.Row():
                with gr.Column():
                    clone_prompt_audio = gr.Audio(
                        label="Reference audio (5–20 s)",
                        sources=["upload", "microphone"],
                        type="filepath",
                    )
                    clone_prompt_text = gr.Textbox(
                        label="Reference transcript",
                        placeholder="Exactly what is said in the reference audio…",
                        lines=2,
                    )
                    clone_text = gr.Textbox(
                        label="Text to synthesize",
                        placeholder="Type the text you want spoken in that voice…",
                        lines=4,
                    )
                    clone_language = gr.Dropdown(
                        LANG_CHOICES, value=AUTO, label="Language / dialect"
                    )
                    clone_btn = gr.Button("Generate speech", variant="primary")
                with gr.Column():
                    clone_out = gr.Audio(label="Generated speech", type="numpy")
                    with gr.Accordion("Advanced options", open=False):
                        clone_cfg = gr.Slider(
                            0.0, 4.0, value=2.0, step=0.1,
                            label="CFG strength",
                            info="Higher sticks closer to the reference timbre.",
                        )
                        clone_steps = gr.Slider(
                            4, 30, value=10, step=1, label="Flow-matching timesteps"
                        )
                        clone_seed = gr.Number(value=1234, precision=0, label="Seed")
                        clone_tn = gr.Checkbox(
                            value=True,
                            label="Text normalization (numbers, dates, units → words)",
                        )
            clone_ex = gr.Dataframe(
                value=[[row[3], row[2]] for row in CLONE_EXAMPLES],
                headers=["Language", "Text"],
                label="Examples (click a row)",
                interactive=False,
                wrap=True,
            )

            def _clone_pick(evt: gr.SelectData):
                return CLONE_EXAMPLES[evt.index[0]]

            clone_ex.select(
                _clone_pick,
                None,
                [clone_prompt_audio, clone_prompt_text, clone_text, clone_language],
            ).then(
                voice_clone,
                [clone_prompt_audio, clone_prompt_text, clone_text, clone_language,
                 clone_cfg, clone_steps, clone_seed, clone_tn],
                [clone_out],
            )

        # ------------------------------------------------------------------ #
        with gr.Tab("🎨 Voice Design"):
            gr.Markdown(
                "Describe a voice in plain language — no reference audio needed. The "
                "model first writes a voice-attribute plan, then renders the audio."
            )
            with gr.Row():
                with gr.Column():
                    design_instruction = gr.Textbox(
                        label="Voice description",
                        placeholder="e.g. A young woman with a gentle voice, speaking slowly…",
                        lines=3,
                    )
                    design_text = gr.Textbox(
                        label="Text to synthesize", lines=4,
                        placeholder="Type the text you want spoken…",
                    )
                    design_btn = gr.Button("Design voice", variant="primary")
                with gr.Column():
                    design_out = gr.Audio(label="Generated speech", type="numpy")
                    design_plan = gr.Textbox(
                        label="Voice-attribute plan (model chain-of-thought)", lines=4
                    )
                    with gr.Accordion("Advanced options", open=False):
                        design_cfg = gr.Slider(
                            0.0, 4.0, value=1.2, step=0.1, label="CFG strength"
                        )
                        design_steps = gr.Slider(
                            4, 30, value=10, step=1, label="Flow-matching timesteps"
                        )
                        design_seed = gr.Number(value=2, precision=0, label="Seed")
                        design_tn = gr.Checkbox(value=True, label="Text normalization")
            design_ex = gr.Dataframe(
                value=DESIGN_EXAMPLES,
                headers=["Voice description", "Text"],
                label="Examples (click a row)",
                interactive=False,
                wrap=True,
            )

            def _design_pick(evt: gr.SelectData):
                return DESIGN_EXAMPLES[evt.index[0]]

            design_ex.select(
                _design_pick, None, [design_instruction, design_text]
            ).then(
                voice_design,
                [design_instruction, design_text, design_cfg, design_steps,
                 design_seed, design_tn],
                [design_out, design_plan],
            )

        # ------------------------------------------------------------------ #
        with gr.Tab("✂️ Speech Editing"):
            with gr.Tabs():
                with gr.Tab("Semantic (content)"):
                    gr.Markdown(
                        "Insert, delete or substitute words in an existing recording "
                        "while keeping the original voice. The model transcribes the "
                        "audio itself — just say what to change."
                    )
                    with gr.Row():
                        with gr.Column():
                            sem_audio = gr.Audio(
                                label="Input speech (≤ 20 s)",
                                sources=["upload", "microphone"],
                                type="filepath",
                            )
                            sem_instruction = gr.Textbox(
                                label="Edit instruction",
                                placeholder="e.g. Replace 'positive' with 'optimistic'.",
                                lines=2,
                            )
                            sem_btn = gr.Button("Apply edit", variant="primary")
                        with gr.Column():
                            sem_out = gr.Audio(label="Edited speech", type="numpy")
                            sem_text = gr.Textbox(label="Edited transcript", lines=3)
                            with gr.Accordion("Advanced options", open=False):
                                sem_cfg = gr.Slider(
                                    0.0, 4.0, value=1.2, step=0.1, label="CFG strength"
                                )
                                sem_steps = gr.Slider(
                                    4, 30, value=10, step=1,
                                    label="Flow-matching timesteps",
                                )
                                sem_seed = gr.Number(
                                    value=1234, precision=0, label="Seed"
                                )
                    sem_ex = gr.Dataframe(
                        value=[[row[1]] for row in SEM_EXAMPLES],
                        headers=["Edit instruction"],
                        label="Examples (click a row)",
                        interactive=False,
                        wrap=True,
                    )

                    def _sem_pick(evt: gr.SelectData):
                        return SEM_EXAMPLES[evt.index[0]]

                    sem_ex.select(
                        _sem_pick, None, [sem_audio, sem_instruction]
                    ).then(
                        semantic_edit,
                        [sem_audio, sem_instruction, sem_cfg, sem_steps, sem_seed],
                        [sem_out, sem_text],
                    )

                with gr.Tab("Acoustic (speed / pitch / volume)"):
                    gr.Markdown(
                        "Re-render the same utterance with a different speaking rate, "
                        "pitch or loudness. These edits follow fixed instruction "
                        "templates the model was trained on."
                    )
                    with gr.Row():
                        with gr.Column():
                            aco_audio = gr.Audio(
                                label="Input speech (≤ 20 s)",
                                sources=["upload", "microphone"],
                                type="filepath",
                            )
                            aco_attr = gr.Radio(
                                ["Speed", "Pitch", "Volume"],
                                value="Speed",
                                label="Attribute",
                            )
                            aco_value = gr.Slider(
                                0.5, 2.0, value=0.8, step=0.1,
                                label="Speed (×)",
                            )
                            aco_btn = gr.Button("Apply edit", variant="primary")
                        with gr.Column():
                            aco_out = gr.Audio(label="Edited speech", type="numpy")
                            aco_instruction = gr.Textbox(
                                label="Instruction sent to the model", lines=1
                            )
                            with gr.Accordion("Advanced options", open=False):
                                aco_cfg = gr.Slider(
                                    0.0, 4.0, value=1.2, step=0.1, label="CFG strength"
                                )
                                aco_steps = gr.Slider(
                                    4, 30, value=10, step=1,
                                    label="Flow-matching timesteps",
                                )
                                aco_seed = gr.Number(
                                    value=1234, precision=0, label="Seed"
                                )
                    aco_ex = gr.Dataframe(
                        value=[[row[1], row[2]] for row in ACO_EXAMPLES],
                        headers=["Attribute", "Value"],
                        label="Examples (click a row)",
                        interactive=False,
                        wrap=True,
                    )

                    def _aco_pick(evt: gr.SelectData):
                        return ACO_EXAMPLES[evt.index[0]]

                    aco_ex.select(
                        _aco_pick, None, [aco_audio, aco_attr, aco_value]
                    ).then(
                        acoustic_edit,
                        [aco_audio, aco_attr, aco_value, aco_cfg, aco_steps, aco_seed],
                        [aco_out, aco_instruction],
                    )

    gr.Markdown(
        """
        ---
        **Model:** [FireRedTeam/FireRedTTS3](https://modelscope.cn/models/FireRedTeam/FireRedTTS3)
        (Apache-2.0) · Base = cloning, Instruct = design + editing. Text normalization
        runs locally through *wetext* (Chinese / English); other languages get basic
        cleaning only. Voice cloning is provided **for academic research purposes only** —
        do not use it for impersonation or any illegal activity.
        """
    )

    def _attr_changed(attribute, current):
        lo, hi, step, label = {
            "Speed": (0.5, 2.0, 0.1, "Speed (×)"),
            "Volume": (0.3, 2.0, 0.1, "Volume (×)"),
        }.get(attribute, (-6, 6, 1, "Pitch shift (semitone steps)"))
        try:
            value = min(max(float(current), lo), hi)
        except (TypeError, ValueError):
            value = lo
        if attribute == "Pitch":
            value = int(round(value)) or 1
        return gr.update(minimum=lo, maximum=hi, step=step, value=value, label=label)

    aco_attr.change(_attr_changed, inputs=[aco_attr, aco_value], outputs=[aco_value])

    clone_btn.click(
        voice_clone,
        inputs=[clone_prompt_audio, clone_prompt_text, clone_text, clone_language,
                clone_cfg, clone_steps, clone_seed, clone_tn],
        outputs=[clone_out],
        api_name="voice_clone",
    )
    design_btn.click(
        voice_design,
        inputs=[design_instruction, design_text, design_cfg, design_steps,
                design_seed, design_tn],
        outputs=[design_out, design_plan],
        api_name="voice_design",
    )
    sem_btn.click(
        semantic_edit,
        inputs=[sem_audio, sem_instruction, sem_cfg, sem_steps, sem_seed],
        outputs=[sem_out, sem_text],
        api_name="semantic_edit",
    )
    aco_btn.click(
        acoustic_edit,
        inputs=[aco_audio, aco_attr, aco_value, aco_cfg, aco_steps, aco_seed],
        outputs=[aco_out, aco_instruction],
        api_name="acoustic_edit",
    )

if __name__ == "__main__":
    # Single concurrency: both Base and Instruct stay resident on one GPU.
    demo.queue(default_concurrency_limit=1).launch(theme=gr.themes.Soft(), css=CSS)
