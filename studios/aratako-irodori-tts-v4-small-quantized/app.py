"""Gradio demo for phasefield-audio/Irodori-TTS-v4.1-Anime.

Anime-style Japanese TTS: a Rectified-Flow DiT over Semantic-DACVAE latents,
fine-tuned from Aratako/Irodori-TTS-v4.1-Small.
"""

import io
import os
import time
import importlib.util

import gradio as gr
import numpy as np
import torch

from irodori_tts.gradio_emoji_palette import EMOJI_PALETTE_CSS, build_emoji_palette
from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    download_hf_checkpoint,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_REPO = os.environ.get("MODEL_REPO", "phasefield-audio/Irodori-TTS-v4.1-Anime")
BASE_MODEL_REPO = "Aratako/Irodori-TTS-v4.1-Small"
CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
GITHUB_REPO = "https://github.com/Aratako/Irodori-TTS"

CSS = (
    EMOJI_PALETTE_CSS
    + """
#col-container { max-width: 1100px; margin: 0 auto; }
.dark .gradio-container { color: var(--body-text-color); }
"""
)

# ---------------------------------------------------------------------------
# Model loading (module scope, eager .to("cuda") on the persistent studio GPU)
# ---------------------------------------------------------------------------

print(f"[Info] Downloading checkpoint + tokenizer from {MODEL_REPO} ...", flush=True)
_checkpoint_path = download_hf_checkpoint(MODEL_REPO)
print(f"[Info] Checkpoint: {_checkpoint_path}", flush=True)

_key = RuntimeKey(
    checkpoint=_checkpoint_path,
    model_device="cuda",
    codec_repo=CODEC_REPO,
    model_precision="bf16",
    codec_device="cuda",
    codec_precision="bf16",
)

print("[Info] Building inference runtime ...", flush=True)
runtime = InferenceRuntime.from_key(_key)
print("[Info] Runtime ready.", flush=True)

USE_SPEAKER_CONDITION = bool(runtime.model_cfg.use_speaker_condition_resolved)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _optional_float(raw, label: str):
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise gr.Error(f"{label} must be a number or left blank.") from exc


def _optional_int(raw, label: str):
    value = _optional_float(raw, label)
    return None if value is None else int(value)


def _resolve_ref_wavs(uploaded) -> list[str]:
    if uploaded is None:
        return []
    values = uploaded if isinstance(uploaded, (list, tuple)) else [uploaded]
    paths: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            candidate = value.strip()
        elif isinstance(value, dict):
            candidate = str(value.get("path") or value.get("name") or "").strip()
        else:
            candidate = str(getattr(value, "name", "") or "").strip()
        if candidate:
            paths.append(candidate)
    return paths


def _on_schedule_change(mode: str):
    return gr.update(interactive=str(mode).strip().lower() == "sway")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


# Runs directly on the studio's persistent GPU (ModelScope has no ZeroGPU
# scheduling). Upstream measured ~2.7 s of compute even at 120 steps with a
# reference clip and a 19 s output; the L20 48GB xGPU has ample headroom.
def generate(
    text: str,
    caption: str = "",
    reference_audio: object = None,
    num_steps: int = 40,
    seed_raw: str = "",
    duration_scale: float = 1.0,
    seconds_raw: str = "",
    cfg_scale_text: float = 3.0,
    cfg_scale_caption: float = 4.0,
    cfg_scale_speaker: float = 5.0,
    t_schedule_mode: str = "linear",
    sway_coeff: float = -1.0,
    cfg_min_t: float = 0.5,
    cfg_max_t: float = 1.0,
) -> tuple[tuple[int, np.ndarray], str]:
    """Synthesize anime-style Japanese speech with Irodori-TTS-v4.1-Anime.

    Args:
        text: Japanese text to speak. Control emoji (e.g. 🤭, 👂, ⏸️) may be inlined.
        caption: Optional Japanese style prompt describing voice, emotion and delivery.
        reference_audio: Optional list of reference audio file paths for zero-shot
            voice cloning. Clips are concatenated in order, up to 120 seconds.
        num_steps: Number of rectified-flow sampling steps.
        seed_raw: Random seed as a string; blank means a random seed.
        duration_scale: Multiplier applied to the automatically predicted duration.
        seconds_raw: Exact output length in seconds; blank means automatic.
        cfg_scale_text: Classifier-free guidance strength for the text branch.
        cfg_scale_caption: Classifier-free guidance strength for the caption branch.
        cfg_scale_speaker: Classifier-free guidance strength for the speaker branch.
        t_schedule_mode: Timestep schedule, either "linear" or "sway".
        sway_coeff: Sway-sampling coefficient, used when t_schedule_mode is "sway".
        cfg_min_t: Lower timestep bound over which guidance is applied.
        cfg_max_t: Upper timestep bound over which guidance is applied.

    Returns:
        A (sample_rate, waveform) audio tuple and a run-log string.
    """
    text_value = "" if text is None else str(text).strip()
    if not text_value:
        raise gr.Error("Please enter some Japanese text to synthesize.")
    caption_value = "" if caption is None else str(caption).strip()

    log_buffer = io.StringIO()

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_buffer.write(msg + "\n")

    seed = _optional_int(seed_raw, "Seed")
    manual_seconds = _optional_float(seconds_raw, "Seconds")

    ref_wavs = _resolve_ref_wavs(reference_audio) if USE_SPEAKER_CONDITION else []
    no_ref = not ref_wavs

    log(
        f"[Info] steps={int(num_steps)} seed={'random' if seed is None else seed} "
        f"caption={'on' if caption_value else 'off'} refs={len(ref_wavs)} "
        f"seconds={'auto' if manual_seconds is None else manual_seconds} "
        f"schedule={t_schedule_mode}"
    )

    started = time.perf_counter()
    result = runtime.synthesize(
        SamplingRequest(
            text=text_value,
            caption=caption_value or None,
            ref_wav=None,
            ref_wavs=ref_wavs or None,
            ref_latent=None,
            no_ref=bool(no_ref),
            ref_normalize_db=-16.0,
            ref_ensure_max=True,
            num_candidates=1,
            decode_mode="sequential",
            seconds=manual_seconds,
            duration_scale=float(duration_scale),
            max_ref_seconds=None,
            max_text_len=None,
            max_caption_len=None,
            num_steps=int(num_steps),
            seed=seed,
            cfg_guidance_mode="independent",
            cfg_scale_text=float(cfg_scale_text),
            cfg_scale_caption=float(cfg_scale_caption),
            cfg_scale_speaker=0.0 if no_ref else float(cfg_scale_speaker),
            cfg_scale=None,
            cfg_min_t=float(cfg_min_t),
            cfg_max_t=float(cfg_max_t),
            truncation_factor=None,
            rescale_k=None,
            rescale_sigma=None,
            context_kv_cache=True,
            speaker_kv_scale=None,
            speaker_kv_min_t=None,
            speaker_kv_max_layers=None,
            t_schedule_mode=str(t_schedule_mode),
            sway_coeff=float(sway_coeff),
            trim_tail=True,
        ),
        log_fn=log,
    )
    elapsed = time.perf_counter() - started

    waveform = result.audios[0].squeeze(0).float().cpu().numpy()
    log(f"[Info] seed_used={result.used_seed}")
    log(f"[Info] audio={waveform.shape[-1] / result.sample_rate:.2f}s generated in {elapsed:.1f}s")
    return (result.sample_rate, waveform), log_buffer.getvalue()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

EXAMPLES = [
    [
        "こんにちは！今日はとってもいい天気だね。一緒にお散歩に行こうよ！",
        "元気で明るい少女の声で、弾むように楽しそうに話す。",
    ],
    [
        "どうしてもっと早く教えてくれなかったの？私、ずっと待ってたのに。",
        "深く傷つき、今にも泣き出しそうな様子。声が震えており、悲痛なトーンで弱々しく話す。",
    ],
    [
        "ふふっ、まさか本当に来てくれるなんてね🤭",
        "からかうような、いたずらっぽい口調で話す少女の声。",
    ],
    [
        "静かに……誰かが来る。ここは私に任せて、先に行って。",
        "落ち着いた低めの女性の声で、緊張感を持って囁くように話す。",
    ],
    [
        "おはようございます。本日もどうぞよろしくお願いいたします。",
        "落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。",
    ],
]

DESCRIPTION = f"""\
[模型](https://modelscope.cn/models/{MODEL_REPO}) · [基座模型](https://modelscope.cn/models/{BASE_MODEL_REPO}) · [代码]({GITHUB_REPO}) · [Codec](https://modelscope.cn/models/{CODEC_REPO})

动漫风格日语音合成 —— 约 766M 参数的 Rectified-Flow DiT，在 DACVAE 潜空间上由
`Irodori-TTS-v4.1-Small` 于动漫语音数据微调而来。输出采样率 48 kHz。

- **风格描述** —— 用日语描述音色与情绪。本微调模型的数据独立于基座模型标注，
  因此描述词的行为与基座模型不同。
- **参考音频** *（可选）* —— 上传同一说话人的一段或多段音频用于零样本声音克隆；
  留空则由模型自由创造音色。
- **控制表情符号** —— 在文本中内联插入控制 emoji，实现耳语、叹息、嬉笑等效果。
"""

with gr.Blocks(title="Irodori-TTS v4.1 Anime") as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# 🎙️ Irodori-TTS-v4.1-Anime")
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label="文本（日语）",
                    lines=3,
                    placeholder="请输入要合成的日语文本，可内联插入控制 emoji。",
                    elem_id="irodori-text-input",
                )
                build_emoji_palette(text, open=False)
                caption = gr.Textbox(
                    label="风格描述（可选）",
                    lines=2,
                    placeholder="例：元気で明るい少女の声で、弾むように楽しそうに話す。",
                )
                reference_audio = gr.File(
                    label="声音克隆参考音频（可选，多段自动拼接）",
                    type="filepath",
                    file_count="multiple",
                    file_types=["audio"],
                    allow_reordering=True,
                    visible=USE_SPEAKER_CONDITION,
                )
                run_btn = gr.Button("生成语音", variant="primary")

            with gr.Column(scale=2):
                audio_out = gr.Audio(
                    label="生成结果",
                    type="numpy",
                    autoplay=False,
                    # Gradio's default waveform colour does not follow the Soft
                    # theme and renders orange; pin it to the theme's blue so the
                    # player matches the rest of the UI.
                    waveform_options=gr.WaveformOptions(
                        waveform_color="#bfdbfe",
                        waveform_progress_color="#3b82f6",
                    ),
                )

        with gr.Accordion("高级设置", open=False):
            with gr.Row():
                num_steps = gr.Slider(
                    label="采样步数", minimum=8, maximum=120, value=40, step=1
                )
                duration_scale = gr.Slider(
                    label="时长缩放", minimum=0.5, maximum=1.5, value=1.0, step=0.01
                )
            with gr.Row():
                seed_raw = gr.Textbox(label="随机种子（留空 = 随机）", value="")
                seconds_raw = gr.Textbox(label="输出秒数（留空 = 自动）", value="")
            with gr.Row():
                cfg_scale_text = gr.Slider(
                    label="CFG · 文本", minimum=0.0, maximum=10.0, value=3.0, step=0.1
                )
                cfg_scale_caption = gr.Slider(
                    label="CFG · 风格描述", minimum=0.0, maximum=10.0, value=4.0, step=0.1
                )
                cfg_scale_speaker = gr.Slider(
                    label="CFG · 说话人", minimum=0.0, maximum=10.0, value=5.0, step=0.1
                )
            with gr.Row():
                t_schedule_mode = gr.Dropdown(
                    label="时间步调度", choices=["linear", "sway"], value="linear"
                )
                sway_coeff = gr.Slider(
                    label="Sway 系数",
                    minimum=-1.0,
                    maximum=1.5,
                    value=-1.0,
                    step=0.1,
                    interactive=False,
                )
            with gr.Row():
                cfg_min_t = gr.Number(label="CFG 最小 t", value=0.5)
                cfg_max_t = gr.Number(label="CFG 最大 t", value=1.0)
            run_log = gr.Textbox(label="运行日志", lines=6)

        gr.Examples(
            examples=EXAMPLES,
            inputs=[text, caption],
            outputs=[audio_out, run_log],
            fn=generate,
            cache_examples=False,
            label="示例（文本 + 风格描述）",
        )

    t_schedule_mode.change(_on_schedule_change, inputs=[t_schedule_mode], outputs=[sway_coeff])

    run_btn.click(
        fn=generate,
        inputs=[
            text,
            caption,
            reference_audio,
            num_steps,
            seed_raw,
            duration_scale,
            seconds_raw,
            cfg_scale_text,
            cfg_scale_caption,
            cfg_scale_speaker,
            t_schedule_mode,
            sway_coeff,
            cfg_min_t,
            cfg_max_t,
        ],
        outputs=[audio_out, run_log],
        api_name="generate",
    )

if __name__ == "__main__":
    # gradio's mcp_server integration needs the optional `mcp` package. Enable it
    # only when present so a base image without `mcp` still launches (instead of
    # crashing after the ~3.3 GB model download + load).
    _mcp_available = importlib.util.find_spec("mcp") is not None
    demo.queue(default_concurrency_limit=1).launch(
        # Bind 0.0.0.0:7860 so the studio is reachable from outside the container.
        # Honour platform-injected GRADIO_SERVER_NAME/PORT when present.
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        theme=gr.themes.Soft(),
        css=CSS,
        mcp_server=_mcp_available,
    )
