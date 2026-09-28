"""Breeze TTS 2 — Gradio demo for ModelScope Studio.

Three modes: Voice Design, Voice Clone, Voice Direction.
Model: BreezeBlue/Breeze-TTS-2
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import tempfile
import threading
import time
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from modelscope import snapshot_download

from breeze_infer.runtime import (
    load_runtime,
    set_all_seeds,
    update_generation_config_for_breeze,
)
from breeze_infer.templates import get_template, prepare_inputs
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig

MODEL_ID = "BreezeBlue/Breeze-TTS-2"
MODEL_REVISION = os.environ.get("MODEL_REVISION", "master")
ASR_MODEL_ID = "openai-mirror/whisper-tiny"
ASR_MODEL_REVISION = os.environ.get("ASR_MODEL_REVISION", "master")
CACHE_ROOT = Path(
    os.environ.get(
        "BREEZE_CACHE_DIR",
        "/mnt/workspace/modelscope"
        if Path("/mnt/workspace").exists()
        else str(Path.home() / ".cache" / "modelscope"),
    )
)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

MAX_NEW_TOKENS = 1500
MAX_SEQ_LEN = 2048
REPETITION_PENALTY = 1.1

# Download from the ModelScope model repository into persistent Studio storage.
_ckpt_dir = snapshot_download(
    MODEL_ID,
    revision=MODEL_REVISION,
    cache_dir=str(CACHE_ROOT),
)
_ckpt_path = Path(_ckpt_dir)

print(f"Loading Breeze TTS 2 from {_ckpt_path} ...")
_tokenizer, _model, _audio_tokenizer = load_runtime(
    _ckpt_path,
    device="cuda",
    attn_implementation="eager",
)
update_generation_config_for_breeze(_model)

_streaming_config = FastStreamingConfig(
    max_new_tokens=MAX_NEW_TOKENS,
    max_seq_len=MAX_SEQ_LEN,
    fast_all=None,
    fast_text_encoder=False,
    fast_backbone_prefill=False,
    fast_backbone_decode=False,
    fast_depth_decoder=False,
    fast_codec=False,
    repetition_penalty=REPETITION_PENALTY,
)
_runtime = FastBreezeStreamingRuntime(
    _model, _audio_tokenizer, _streaming_config, tokenizer=_tokenizer
)
print("Breeze TTS 2 loaded successfully.")
print(f"Sample rate: {_runtime.sample_rate}")


_asr_pipe = None
_asr_lock = threading.Lock()


def _get_asr():
    """Lazily build the CPU Whisper pipeline for reference transcription."""
    global _asr_pipe
    with _asr_lock:
        if _asr_pipe is None:
            from transformers import pipeline

            asr_dir = snapshot_download(
                ASR_MODEL_ID,
                revision=ASR_MODEL_REVISION,
                cache_dir=str(CACHE_ROOT),
                allow_file_pattern=["*.json", "*.txt", "model.safetensors"],
            )
            print(f"Loading Whisper Tiny from {asr_dir} on CPU ...")
            _asr_pipe = pipeline(
                "automatic-speech-recognition",
                model=asr_dir,
                device=-1,
                chunk_length_s=30,
            )
    return _asr_pipe


def transcribe_reference(
    ref_audio_path: str | None,
    progress=gr.Progress(track_tqdm=True),
):
    """Transcribe reference audio and return editable text for voice cloning."""
    if not ref_audio_path:
        return gr.update()

    try:
        asr = _get_asr()
    except Exception as exc:  # pragma: no cover - download/load failure
        raise gr.Error(f"Could not load the transcription model: {exc}") from exc

    try:
        import librosa

        audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        asr_input = {"raw": audio, "sampling_rate": 16000}
    except Exception:
        asr_input = str(ref_audio_path)

    try:
        result = asr(asr_input, generate_kwargs={"task": "transcribe"})
    except Exception as exc:
        raise gr.Error(f"Automatic transcription failed: {exc}") from exc

    text = (result.get("text") or "").strip()
    if not text:
        raise gr.Error(
            "Automatic transcription produced no text — please type the transcript manually."
        )
    return text


def generate_voice_design(
    text: str,
    instruction: str,
    cfg_scale: float,
    seed: int,
    progress=gr.Progress(track_tqdm=True),
) -> str:
    """Generate speech with a designed voice from a natural-language description.

    Args:
        text: The text to speak. Supports inline vocal events like (laugh), (sigh), etc.
        instruction: Natural-language description of the desired voice character.
        cfg_scale: Classifier-free guidance scale. Higher = stronger instruction following.
        seed: Random seed for reproducibility.
    """
    return _run_inference(text, instruction, None, None, cfg_scale, seed)


def generate_voice_clone(
    text: str,
    ref_audio_path: str,
    ref_text: str,
    seed: int,
    progress=gr.Progress(track_tqdm=True),
) -> str:
    """Clone a voice from reference audio and its exact transcript.

    Args:
        text: The text to speak. Supports inline vocal events.
        ref_audio_path: Path to reference audio file (clean speech, minimal noise).
        ref_text: Exact transcript of the reference audio.
        seed: Random seed for reproducibility.
    """
    return _run_inference(text, "Speak clearly and naturally.", ref_audio_path, ref_text, 1.0, seed)


def generate_voice_direction(
    text: str,
    instruction: str,
    ref_audio_path: str,
    ref_text: str,
    cfg_scale: float,
    seed: int,
    progress=gr.Progress(track_tqdm=True),
) -> str:
    """Clone a reference voice while steering tone, emotion, and pace via instructions.

    Args:
        text: The text to speak. Supports inline vocal events.
        instruction: Direction for tone, emotion, pace, and delivery.
        ref_audio_path: Path to reference audio file.
        ref_text: Exact transcript of the reference audio.
        cfg_scale: Classifier-free guidance scale. Use 4 for stronger instruction following.
        seed: Random seed for reproducibility.
    """
    return _run_inference(text, instruction, ref_audio_path, ref_text, cfg_scale, seed)


def _run_inference(
    text: str,
    instruction: str,
    ref_audio_path: str | None,
    ref_text: str | None,
    cfg_scale: float,
    seed: int,
) -> str:
    """Run Breeze TTS 2 inference and return a path to the output WAV."""
    text = text.strip()
    if not text:
        raise gr.Error("Text to speak is required.")

    has_ref = ref_audio_path is not None and ref_text and ref_text.strip()
    if ref_audio_path is not None and not has_ref:
        raise gr.Error("Reference audio and reference text must be provided together.")

    set_all_seeds(int(seed))

    request = {
        "id": f"demo-{int(time.time())}",
        "text": text,
        "instruction": instruction,
        "speaker": "S0",
    }
    template_name = "tts_instruction"
    if has_ref:
        request["ref_audio_path"] = str(ref_audio_path)
        request["ref_text"] = ref_text.strip()
        template_name = "ref_edit_tata"

    inputs = prepare_inputs(
        _tokenizer,
        _audio_tokenizer,
        _model,
        [request],
        get_template(template_name),
        guidance_scale=cfg_scale,
        guidance_scale_ref=None,
        guidance_scale_ins=None,
    )

    audio_chunks = []
    for chunk in _runtime.iter_audio_chunks(inputs, request_id=request["id"]):
        audio_chunks.append(chunk.audio)

    if not audio_chunks:
        raise gr.Error("Model produced no audio output.")

    audio = np.concatenate(audio_chunks, axis=0)
    sr = _runtime.sample_rate

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, sr, subtype="PCM_16")
        return tmp.name


CSS = """
#col-container { max-width: 1100px; margin: 0 auto; }
.dark .gradio-container { color: var(--body-text-color); }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=CSS) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            "# 🎙️ Breeze TTS 2\n"
            "Open-weight bilingual (English/Chinese) text-to-speech with voice design, "
            "voice cloning, and voice direction. "
            "Supports inline vocal events: `(laugh)`, `(sigh)`, `(cough)`, `(clears throat)`."
        )

        with gr.Tabs():
            # === Voice Design Tab ===
            with gr.Tab("🎨 Voice Design"):
                gr.Markdown(
                    "Create a distinctive voice from a natural-language description — "
                    "no reference audio needed. Use `cfg_scale = 4` for stronger instruction following."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        design_text = gr.Textbox(
                            label="Text to speak",
                            placeholder="Welcome aboard. Your journey begins now.",
                            lines=3,
                        )
                        design_instruction = gr.Textbox(
                            label="Voice description",
                            placeholder="A warm, thoughtful young woman with a clear voice and a calm, reflective delivery.",
                            lines=2,
                        )
                        with gr.Accordion("Advanced settings", open=False):
                            design_cfg = gr.Slider(
                                label="CFG scale",
                                minimum=1.0,
                                maximum=10.0,
                                value=4.0,
                                step=0.5,
                            )
                            design_seed = gr.Number(label="Seed", value=42, precision=0)
                        design_btn = gr.Button("Generate", variant="primary")
                    with gr.Column(scale=2):
                        design_output = gr.Audio(
                            label="Generated audio",
                            type="filepath",
                        )

                gr.Examples(
                    examples=[
                        [
                            "(sigh) Welcome aboard. Your journey begins now.",
                            "A warm, thoughtful young woman with a clear voice and a calm, reflective delivery.",
                            4.0,
                            42,
                        ],
                        [
                            "The stars above are calling, and tonight we answer.",
                            "A deep, resonant male voice with theatrical gravitas and a slow, deliberate pace.",
                            4.0,
                            42,
                        ],
                        [
                            "[笑] 欢迎来到今晚的故事时间，让我们一起开始吧。",
                            "一位温柔自信的年轻女性，声音清晰，语气亲切，表达轻快而富有感染力。",
                            4.0,
                            42,
                        ],
                    ],
                    inputs=[design_text, design_instruction, design_cfg, design_seed],
                    outputs=design_output,
                    fn=generate_voice_design,
                    cache_examples=False,
                )

                design_btn.click(
                    fn=generate_voice_design,
                    inputs=[design_text, design_instruction, design_cfg, design_seed],
                    outputs=design_output,
                    api_name="voice_design",
                )

            # === Voice Clone Tab ===
            with gr.Tab("🎙️ Voice Clone"):
                gr.Markdown(
                    "Clone a speaker from clean reference audio and its exact transcript. "
                    "Reference audio should contain clean speech with minimal background noise."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        clone_text = gr.Textbox(
                            label="Text to speak",
                            placeholder="It is good to hear your voice again after all this time.",
                            lines=3,
                        )
                        clone_ref_audio = gr.Audio(
                            label="Reference audio",
                            type="filepath",
                        )
                        clone_ref_text = gr.Textbox(
                            label="Reference transcript (exact)",
                            info="Auto-transcribed with Whisper Tiny when audio is added — please review and edit if needed.",
                            placeholder="This is the exact transcript of the reference audio.",
                            lines=2,
                        )
                        clone_ref_audio.change(
                            fn=transcribe_reference,
                            inputs=[clone_ref_audio],
                            outputs=[clone_ref_text],
                            api_name="transcribe_reference",
                        )
                        with gr.Accordion("Advanced settings", open=False):
                            clone_seed = gr.Number(label="Seed", value=42, precision=0)
                        clone_btn = gr.Button("Generate", variant="primary")
                    with gr.Column(scale=2):
                        clone_output = gr.Audio(
                            label="Generated audio",
                            type="filepath",
                        )

                clone_btn.click(
                    fn=generate_voice_clone,
                    inputs=[clone_text, clone_ref_audio, clone_ref_text, clone_seed],
                    outputs=clone_output,
                    api_name="voice_clone",
                )

            # === Voice Direction Tab ===
            with gr.Tab("🎛️ Voice Direction"):
                gr.Markdown(
                    "Clone a reference voice while directing tone, emotion, pace, and delivery. "
                    "Use `cfg_scale = 4` for stronger instruction following."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        dir_text = gr.Textbox(
                            label="Text to speak",
                            placeholder="We need to discuss what happened last night.",
                            lines=3,
                        )
                        dir_instruction = gr.Textbox(
                            label="Direction",
                            placeholder="Speak slowly with a restrained, serious tone.",
                            lines=2,
                        )
                        dir_ref_audio = gr.Audio(
                            label="Reference audio",
                            type="filepath",
                        )
                        dir_ref_text = gr.Textbox(
                            label="Reference transcript (exact)",
                            info="Auto-transcribed with Whisper Tiny when audio is added — please review and edit if needed.",
                            placeholder="This is the exact transcript of the reference audio.",
                            lines=2,
                        )
                        dir_ref_audio.change(
                            fn=transcribe_reference,
                            inputs=[dir_ref_audio],
                            outputs=[dir_ref_text],
                            api_name="transcribe_reference_direction",
                        )
                        with gr.Accordion("Advanced settings", open=False):
                            dir_cfg = gr.Slider(
                                label="CFG scale",
                                minimum=1.0,
                                maximum=10.0,
                                value=4.0,
                                step=0.5,
                            )
                            dir_seed = gr.Number(label="Seed", value=42, precision=0)
                        dir_btn = gr.Button("Generate", variant="primary")
                    with gr.Column(scale=2):
                        dir_output = gr.Audio(
                            label="Generated audio",
                            type="filepath",
                        )

                dir_btn.click(
                    fn=generate_voice_direction,
                    inputs=[
                        dir_text,
                        dir_instruction,
                        dir_ref_audio,
                        dir_ref_text,
                        dir_cfg,
                        dir_seed,
                    ],
                    outputs=dir_output,
                    api_name="voice_direction",
                )

        gr.Markdown(
            "---\n"
            "Model: [BreezeBlue/Breeze-TTS-2](https://modelscope.cn/models/BreezeBlue/Breeze-TTS-2) · "
            "Code: [breezeblue-ai/breeze-tts](https://github.com/breezeblue-ai/breeze-tts) · "
            "Weights/outputs are for research and non-commercial use only."
        )

try:
    _get_asr()
    print(f"Whisper ASR ({ASR_MODEL_ID}) loaded on CPU for reference transcription.")
except Exception as _exc:  # pragma: no cover
    print(f"Warning: could not preload Whisper ASR ({_exc}); will retry on first use.")

demo.queue(default_concurrency_limit=1, max_size=8).launch(mcp_server=True)
