"""YuE2-3B Music Generator — ModelScope Studio (Gradio).

Unofficial ModelScope migration of the m-a-p/YuE2-3B music generation demo.
The heavy lifting is done by the official ``yue2`` inference package (bundled as
``yue2_infer-0.1.5-py3-none-any.whl``). Model weights and the VAE decoder are
fetched from ModelScope into the persistent ``/mnt/workspace`` directory and are
loaded fully offline via local directory paths, so no external model hub is
contacted at runtime.

UI: a Suno-inspired composer + player layout. The visual design language (Gradio
Soft theme, centered container, card / chip / segmented design tokens) follows
the sister MiniMax-Music3 ModelScope studio. Every custom rule rides Gradio theme
CSS variables, so it tracks the theme and dark mode natively.

Model weights: CC BY-NC 4.0 (non-commercial). See LICENSE / THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import gradio as gr

# --- Persistent model cache -------------------------------------------------
# ModelScope studios keep /mnt/workspace across restarts; fall back to a local
# directory when it is not writable (e.g. local smoke tests).
def _workspace_root() -> Path:
    candidate = Path(os.environ.get("YUE2_WORKSPACE", "/mnt/workspace"))
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".yue2_write_probe"
        probe.write_text("ok")
        probe.unlink()
        return candidate
    except OSError:
        return Path(__file__).resolve().parent / "_models"


WORKSPACE = _workspace_root()
MODEL_ROOT = WORKSPACE / "yue2-models"
MODEL_DIR = MODEL_ROOT / "YuE2-3B"
VAE_DIR = MODEL_ROOT / "YuE2-Vae"

MODEL_ID = "m-a-p/YuE2-3B"
VAE_ID = "m-a-p/YuE2-Vae"

# Site switch: the international (.ai) studio sets YUE2_SITE=ai so this one codebase
# can adapt per deployment (model-page link, and the examples-widget workaround below).
_IS_AI = os.environ.get("YUE2_SITE", "").strip().lower() == "ai"
_MODEL_URL = (
    "https://www.modelscope.ai/models/m-a-p/YuE2-3B"
    if _IS_AI
    else "https://modelscope.cn/models/m-a-p/YuE2-3B"
)

_pipe = None
_pipe_lock = threading.Lock()
_status = {"state": "idle", "detail": ""}


def _download_snapshot(model_id: str, local_dir: Path) -> None:
    """Fetch a model snapshot from ModelScope into ``local_dir`` (idempotent)."""
    if (local_dir / "model.safetensors").is_file():
        return
    # NOTE: ModelScope's platform-native model downloader (endpoint
    # modelscope.cn), not the HF hub client. This is the intended way to fetch
    # weights on a ModelScope studio and needs no external HF endpoint.
    from modelscope import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(model_id, local_dir=str(local_dir))


def ensure_models() -> tuple[str, str]:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    _download_snapshot(MODEL_ID, MODEL_DIR)
    _download_snapshot(VAE_ID, VAE_DIR)
    return str(MODEL_DIR), str(VAE_DIR)


def _wheel_version_key(path: Path) -> tuple:
    """Numeric sort key for ``yue2_infer-<version>-py3-none-any.whl`` names."""
    try:
        return tuple(int(part) for part in path.name.split("-", 2)[1].split("."))
    except (IndexError, ValueError):
        return (0,)


def _ensure_yue2_package(model_dir: str) -> None:
    """Make the official ``yue2`` inference package importable at runtime.

    The package ships as a pure-Python wheel *inside* the ModelScope model repo
    (``m-a-p/YuE2-3B``), so it arrives together with the snapshot we already
    download. A wheel is just a zip: we unpack the newest one into a persistent
    libs directory and prepend it to ``sys.path``. This runs the exact code
    published alongside the model — with no source vendored into this studio repo
    and no runtime ``pip install``. The wheel committed at the repo root is kept
    only as an offline fallback should a snapshot ever lack it.
    """
    try:  # already importable (extracted on a previous start) — nothing to do
        import yue2  # noqa: F401
        return
    except ImportError:
        pass

    libs = MODEL_ROOT / "_pylibs"
    marker = libs / ".yue2_wheel_extracted"
    if not marker.is_file():
        import zipfile

        candidates = sorted(Path(model_dir).glob("yue2_infer-*.whl"), key=_wheel_version_key)
        if not candidates:  # fall back to the copy bundled at this repo's root
            candidates = sorted(
                Path(__file__).resolve().parent.glob("yue2_infer-*.whl"), key=_wheel_version_key
            )
        if not candidates:
            raise RuntimeError(
                "yue2_infer wheel not found in the ModelScope snapshot or repo root; "
                "expected m-a-p/YuE2-3B to include yue2_infer-*.whl"
            )
        wheel = candidates[-1]
        libs.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(libs)
        marker.write_text(wheel.name, encoding="utf-8")
    if str(libs) not in sys.path:
        sys.path.insert(0, str(libs))


def get_pipe():
    """Return a resident pipeline, downloading + loading it on first use."""
    global _pipe
    if _pipe is not None:
        return _pipe
    with _pipe_lock:
        if _pipe is not None:
            return _pipe
        import torch

        _status.update(state="downloading", detail="Fetching weights from ModelScope…")
        model_dir, vae_dir = ensure_models()
        _ensure_yue2_package(model_dir)
        from yue2 import YuE2Pipeline
        print("[yue2] inference package imported from the ModelScope snapshot", flush=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _status.update(state="loading", detail=f"Loading pipeline on {device}…")
        _pipe = YuE2Pipeline.from_pretrained(
            model_dir,
            vae=vae_dir,
            device=device,
            local_files_only=True,  # offline: weights + modeling come from the local snapshot
            progress=False,
        )
        _status.update(state="ready", detail=f"Ready on {device}.")
        print(f"[yue2] pipeline ready on {device}", flush=True)
        return _pipe


def _warmup() -> None:
    try:
        get_pipe()
        print(f"[yue2] warmup complete — {_status.get('detail', '')}", flush=True)
    except Exception as exc:  # surface errors in the run log, keep UI alive
        import traceback
        traceback.print_exc()
        _status.update(state="error", detail=f"{type(exc).__name__}: {exc}")
        print(f"[yue2] warmup FAILED: {type(exc).__name__}: {exc}", flush=True)


# --- Generation helpers -----------------------------------------------------
def _run(style: str, lyrics: str, cot: str, seed, cfg_scale, abc: str | None):
    if not style or not style.strip():
        raise gr.Error("Please provide a style description.")
    if not lyrics or not lyrics.strip():
        raise gr.Error("Please provide lyrics.")
    seed = int(seed)
    kwargs = {"style": style.strip(), "lyrics": lyrics, "cot": cot, "seed": seed}
    if cfg_scale is not None and float(cfg_scale) > 0:
        kwargs["cfg_scale"] = float(cfg_scale)
    if abc and abc.strip():
        if cot == "off":
            raise gr.Error("An ABC score requires cot='melody' or 'full'.")
        kwargs["abc"] = abc

    pipe = get_pipe()
    started = time.perf_counter()
    result = pipe(**kwargs)
    elapsed = time.perf_counter() - started

    audio = result.audio  # numpy float32, shape [samples, channels]
    seconds = len(audio) / float(result.sample_rate)
    trunc = result.truncated
    info = (
        f"**{seconds:.1f}s** &nbsp;·&nbsp; {result.sample_rate // 1000}&nbsp;kHz stereo "
        f"&nbsp;·&nbsp; generated in {elapsed:.1f}s  \n"
        f"cot=`{cot}` &nbsp;·&nbsp; seed=`{seed}`"
        + (f" &nbsp;·&nbsp; cfg=`{kwargs.get('cfg_scale', 'default')}`" if "cfg_scale" in kwargs else "")
        + (f"  \n**Truncated:** {trunc}" if any(trunc.values()) else "")
    )
    score = result.abc or ""
    return (result.sample_rate, audio), info, score


def create_song(style, lyrics, cot, seed, cfg_scale, progress=gr.Progress(track_tqdm=False)):
    progress(0.1, desc="Planning + generating song (this can take a few minutes)…")
    audio, info, score = _run(style, lyrics, cot, seed, cfg_scale, None)
    progress(1.0, desc="Done")
    return audio, info, score


def cover_song(style, lyrics, abc, seed, cfg_scale, progress=gr.Progress(track_tqdm=False)):
    progress(0.1, desc="Generating cover (melody-conditioned)…")
    audio, info, score = _run(style, lyrics, "melody", seed, cfg_scale, abc)
    progress(1.0, desc="Done")
    return audio, info, score


# --- Examples & quick-start ideas -------------------------------------------
_EX_PATH = Path(__file__).resolve().parent / "examples" / "tonight-awake.json"
_EXAMPLE: dict = {}
if _EX_PATH.is_file():
    try:
        _EXAMPLE = json.loads(_EX_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _EXAMPLE = {}

EXAMPLES = []
if _EXAMPLE:
    EXAMPLES = [[
        _EXAMPLE.get("style", ""),
        _EXAMPLE.get("lyrics", ""),
        _EXAMPLE.get("cot", "full"),
        _EXAMPLE.get("seed", 12300),
        0,
    ]]


# Quick-start style ideas: clicking a chip fills the Style box.
_IDEA_CHIPS = [
    ("City Pop", "City Pop, upbeat, danceable, groovy bass, electric guitar, synth, energetic, joyful, neon city night"),
    ("Lo-fi Chill", "lo-fi hip hop, mellow, warm Rhodes piano, soft drums, relaxed late-night vibe, gentle male vocals"),
    ("Arena Rock", "arena rock, driving electric guitars, punchy drums, anthemic, powerful male vocals, uplifting chorus"),
    ("Dream Pop", "dreamy shoegaze, layered reverb guitars, ethereal female vocals, slow build, hazy and warm"),
]


# --- Design system ----------------------------------------------------------
# Ported from the sister MiniMax-Music3 studio. Component chrome keys off Gradio
# theme CSS variables so it inherits the active Soft theme palette and dark mode
# natively. The hero brand art (logo tile + wordmark gradient) is the one place
# with fixed brand colors, so it keeps its identity across themes.
_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "yue-logo.png"
_LOGO_URI = ""
if _LOGO_PATH.is_file():
    import base64

    _LOGO_URI = "data:image/png;base64," + base64.b64encode(_LOGO_PATH.read_bytes()).decode("ascii")

if _LOGO_URI:
    _MARK_INNER = f'<img src="{_LOGO_URI}" alt="YuE logo">'
else:  # offline fallback glyph if the asset is ever missing
    _MARK_INNER = (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>'
    )

_CSS = """
#col-container { max-width: 1300px; margin: 0 auto; }
.dark .gradio-container { color: var(--body-text-color); }

/* Hero ------------------------------------------------------------------ */
.yue-hero { display: flex; flex-direction: column; align-items: center; text-align: center; gap: 10px; padding: 24px 0 12px; }
.yue-hero-mark { position: relative; display: inline-flex; align-items: center; justify-content: center; width: 88px; height: 88px;
  border-radius: 26px; overflow: hidden; padding: 15px; box-sizing: border-box;
  background: linear-gradient(150deg, #ffffff 0%, #e9edf4 100%); border: 1px solid rgba(15, 23, 42, .10);
  box-shadow: 0 1px 2px rgba(15,23,42,.08), 0 14px 30px rgba(15,23,42,.16), inset 0 1px 0 rgba(255,255,255,.85); }
.yue-hero-mark img, .yue-hero-mark svg { width: 100%; height: 100%; display: block; object-fit: contain; }
.yue-hero-mark::after { content: ""; position: absolute; inset: 0; border-radius: inherit; pointer-events: none;
  background: linear-gradient(160deg, rgba(255,255,255,.50) 0%, rgba(255,255,255,0) 46%); }
.dark .yue-hero-mark { background: linear-gradient(150deg, #232b38 0%, #10141b 100%); border-color: rgba(255,255,255,.12);
  box-shadow: 0 1px 2px rgba(0,0,0,.55), 0 16px 34px rgba(0,0,0,.5), inset 0 1px 0 rgba(255,255,255,.10); }
.dark .yue-hero-mark img { filter: invert(1); }
.dark .yue-hero-mark svg { color: #fff; }
.dark .yue-hero-mark::after { background: linear-gradient(160deg, rgba(255,255,255,.14) 0%, rgba(255,255,255,0) 46%); }
.yue-hero-title { font-size: 44px; line-height: 1.05; font-weight: 800; letter-spacing: -.5px; margin: 2px 0 0;
  background: linear-gradient(92deg, #4f46e5 0%, #7c3aed 52%, #2563eb 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent; }
.dark .yue-hero-title { background: linear-gradient(92deg, #818cf8 0%, #c084fc 52%, #60a5fa 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent; }
.yue-hero-sub { color: var(--body-text-color-subdued); font-size: 14.5px; max-width: 680px; margin: 0; }
.yue-hero-links { display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; margin-top: 4px; }
.yue-pill { display: inline-flex; align-items: center; gap: 6px; text-decoration: none; font-size: 12.5px; font-weight: 600;
  color: var(--body-text-color-subdued); border: 1px solid var(--border-color-primary); background: var(--block-background-fill);
  border-radius: 999px; padding: 5px 14px; transition: color .15s, border-color .15s, background .15s; }
.yue-pill:hover { color: var(--color-accent); border-color: var(--color-accent); }
.yue-pill-solid { color: var(--button-primary-text-color); background: var(--button-primary-background-fill); border-color: transparent; }
.yue-pill-solid:hover { color: var(--button-primary-text-color); filter: brightness(1.05); border-color: transparent; }

/* Layout / cards -------------------------------------------------------- */
.yue-main { gap: 16px; align-items: start; }
.yue-card { background: var(--block-background-fill); border: var(--block-border-width, 1px) solid var(--block-border-color, var(--border-color-primary));
  border-radius: 16px; box-shadow: 0 1px 2px rgba(0,0,0,.04), 0 10px 30px rgba(0,0,0,.05); padding: 16px; gap: 12px; }
.yue-card-title { display: inline-flex; align-items: center; gap: 8px; font-size: 13.5px; font-weight: 700;
  color: var(--block-title-text-color, var(--body-text-color)); }

/* Segmented tabs -------------------------------------------------------- */
#yue-tabs .tab-nav { gap: 6px; border: none; background: transparent; padding: 0; margin-bottom: 6px; }
#yue-tabs .tab-nav button { border-radius: 999px; border: 1px solid var(--border-color-primary); background: var(--background-fill-secondary);
  padding: 7px 20px; font-weight: 700; font-size: 14px; color: var(--body-text-color-subdued); transition: background .15s, color .15s, border-color .15s; }
#yue-tabs .tab-nav button.selected, #yue-tabs .tab-nav button[aria-selected="true"] {
  background: var(--button-primary-background-fill); color: var(--button-primary-text-color); border-color: transparent; }

/* Inputs ---------------------------------------------------------------- */
.yue-desc textarea { font-size: 15px; }
.yue-lyrics textarea { font-family: var(--font-mono, ui-monospace, SFMono-Regular, Menlo, monospace); font-size: 13px; line-height: 1.55; }
.yue-abc textarea { font-family: var(--font-mono, ui-monospace, SFMono-Regular, Menlo, monospace); font-size: 12.5px; }

/* Chips ----------------------------------------------------------------- */
.yue-chips { display: flex; flex-wrap: wrap; gap: 6px; margin: -4px 0 2px; }
.yue-chip button { border-radius: 999px !important; border: 1px solid var(--border-color-primary) !important;
  background: var(--button-secondary-background-fill) !important; color: var(--button-secondary-text-color) !important;
  padding: 5px 14px !important; font-size: 12.5px !important; font-weight: 600 !important; box-shadow: none !important; min-width: 0 !important; }
.yue-chip button:hover { border-color: var(--color-accent) !important; color: var(--color-accent) !important; }

/* Primary button -------------------------------------------------------- */
.yue-primary button, button.yue-primary { border-radius: 999px !important; font-weight: 700 !important; font-size: 16px !important;
  padding: 12px 26px !important; transition: transform .05s, box-shadow .15s, filter .15s !important; }
.yue-primary button:hover, button.yue-primary:hover { filter: brightness(1.05); }
.yue-primary button:active, button.yue-primary:active { transform: translateY(1px); }

/* Player ---------------------------------------------------------------- */
.yue-player-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.yue-live { display: inline-flex; align-items: center; gap: 6px; color: var(--color-accent); font-size: 11.5px; font-weight: 700; }
.yue-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--color-accent); animation: yue-pulse 1.6s ease-in-out infinite; }
@keyframes yue-pulse { 0%,100% { opacity:.3; transform: scale(.85);} 50% { opacity:1; transform: scale(1.15);} }
.yue-stats { color: var(--body-text-color-subdued); font-size: 13px; }

"""

_HERO = f"""
<div class="yue-hero">
  <span class="yue-hero-mark">{_MARK_INNER}</span>
  <h1 class="yue-hero-title">YuE2-3B</h1>
  <p class="yue-hero-sub">Turn a <b>style description</b> and <b>structured lyrics</b> into a complete song &mdash;
    vocals, accompaniment and an ABC score &mdash; rendered at 48&nbsp;kHz stereo.</p>
  <div class="yue-hero-links">
    <a class="yue-pill yue-pill-solid" href="{_MODEL_URL}" target="_blank" rel="noopener">&#9834;&nbsp; Model &middot; m-a-p/YuE2-3B</a>
    <a class="yue-pill" href="https://creativecommons.org/licenses/by-nc/4.0/" target="_blank" rel="noopener">License &middot; CC BY-NC 4.0</a>
    <span class="yue-pill">Non-commercial use only</span>
  </div>
</div>
"""

_PLAYER_HEAD = """
<div class="yue-player-head">
  <span class="yue-card-title">&#9835;&nbsp; Your song</span>
  <span class="yue-live"><span class="yue-dot"></span>48&nbsp;kHz stereo</span>
</div>
"""



# --- UI ---------------------------------------------------------------------
with gr.Blocks(
    theme=gr.themes.Soft(),
    css=_CSS,
    title="YuE2-3B Music Generator",
) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(_HERO, container=False, padding=False)

        with gr.Row(elem_classes="yue-main", equal_height=False):
            # ---------- LEFT: composer ----------
            with gr.Column(scale=3, min_width=360, elem_classes="yue-card"):
                with gr.Tabs(elem_id="yue-tabs"):
                    with gr.Tab("🎶 Create"):
                        c_style = gr.Textbox(
                            label="Style",
                            elem_classes="yue-desc",
                            lines=3,
                            placeholder="City Pop, upbeat, groovy bass, electric guitar, synth, energetic…",
                        )
                        with gr.Row(elem_classes="yue-chips"):
                            for _name, _desc in _IDEA_CHIPS:
                                _chip = gr.Button(_name, size="sm", elem_classes="yue-chip")
                                _chip.click(lambda d=_desc: d, None, c_style)
                        c_lyrics = gr.Textbox(
                            label="Lyrics",
                            elem_classes="yue-lyrics",
                            lines=11,
                            placeholder="[Verse]\n…\n[Chorus]\n…",
                        )
                        c_cot = gr.Radio(
                            ["full", "melody", "off"],
                            value="full",
                            label="Planning (cot)",
                            info="full = lyrics + melody · melody = melody only · off = direct",
                        )
                        with gr.Accordion("Advanced", open=False):
                            c_seed = gr.Number(value=12300, label="Seed", precision=0)
                            c_cfg = gr.Slider(0, 3, value=0, step=0.05, label="CFG scale (0 = default)")
                        c_btn = gr.Button("♪  Generate song", variant="primary", elem_classes="yue-primary")
                        if EXAMPLES:
                            if _IS_AI:
                                # modelscope.ai renders gr.Examples (Dataset) as empty skeleton
                                # rows, while plain components render and click fine there (same
                                # fix as irodori-tts-anime-demo). Offer the example as a button
                                # that populates the inputs server-side.
                                gr.Markdown("### Example · 今晚不眠 (City Pop)")
                                with gr.Row():
                                    _ex_btn = gr.Button(
                                        f"1. {_EXAMPLE.get('style', '')[:12]}…",
                                        size="sm",
                                        variant="secondary",
                                    )
                                _ex_btn.click(
                                    fn=lambda: tuple(EXAMPLES[0]),
                                    inputs=[],
                                    outputs=[c_style, c_lyrics, c_cot, c_seed, c_cfg],
                                )
                            else:
                                gr.Examples(
                                    examples=EXAMPLES,
                                    inputs=[c_style, c_lyrics, c_cot, c_seed, c_cfg],
                                    label="Example · 今晚不眠 (City Pop)",
                                    cache_examples=False,  # never pre-run a generation at startup
                                )

                    with gr.Tab("🎙️ Cover"):
                        gr.Markdown(
                            "Re-imagine an existing song. Provide a **melody ABC score** (without chord "
                            "symbols), the **lyrics**, and a **target style**. Uses melody-only planning."
                        )
                        v_style = gr.Textbox(
                            label="Target style",
                            elem_classes="yue-desc",
                            lines=2,
                            placeholder="Jazz-funk, warm lead vocal, Rhodes piano, tight drums",
                        )
                        v_lyrics = gr.Textbox(label="Lyrics", elem_classes="yue-lyrics", lines=8)
                        v_abc = gr.Textbox(
                            label="Melody ABC score",
                            elem_classes="yue-abc",
                            lines=7,
                            placeholder="X:1\nT:Melody\nK:C\n…",
                        )
                        with gr.Accordion("Advanced", open=False):
                            v_seed = gr.Number(value=831001, label="Seed", precision=0)
                            v_cfg = gr.Slider(0, 3, value=0, step=0.05, label="CFG scale (0 = default)")
                        v_btn = gr.Button("♪  Generate cover", variant="primary", elem_classes="yue-primary")

            # ---------- RIGHT: player ----------
            with gr.Column(scale=2, min_width=320, elem_classes="yue-card"):
                gr.HTML(_PLAYER_HEAD, container=False, padding=False)
                out_audio = gr.Audio(label=None, type="numpy", elem_classes="yue-audio")
                out_info = gr.Markdown(elem_classes="yue-stats")
                with gr.Accordion("ABC score", open=False):
                    out_score = gr.Textbox(label=None, lines=10, interactive=False, elem_classes="yue-abc")

    # Both modes share the single player on the right.
    c_btn.click(create_song, inputs=[c_style, c_lyrics, c_cot, c_seed, c_cfg],
                outputs=[out_audio, out_info, out_score])
    v_btn.click(cover_song, inputs=[v_style, v_lyrics, v_abc, v_seed, v_cfg],
                outputs=[out_audio, out_info, out_score])

# Single song at a time: the pipeline holds a mutable model/VAE cache.
demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    # Warm the model in the background so the UI is reachable immediately and the
    # first request does not pay the full download + load cost. Kept under the
    # entrypoint guard so merely importing this module (e.g. a local UI preview)
    # does not kick off a multi-GB download.
    threading.Thread(target=_warmup, daemon=True).start()
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", os.environ.get("PORT", "7860"))),
        root_path=os.environ.get("GRADIO_ROOT_PATH", ""),
        show_error=True,
    )
