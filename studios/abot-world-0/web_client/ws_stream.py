"""
Low-latency binary WebSocket transport for the L20N Docker deployment.

Routes:
  GET       /stream_page?session_id=...  browser canvas page
  WEBSOCKET /stream_ws?session_id=...    binary JPEG frames + controls
  POST      /control?session_id=...       compatibility control endpoint
  GET       /healthz                     service/model/GPU readiness
  GET       /metrics                     latest per-session performance data
"""
import asyncio
import contextlib
import io
import json
import logging
import math
import os
import queue
import statistics
import struct
import subprocess
import threading
import time
import uuid
from collections import deque
from pathlib import Path as PathLib
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

from web_client.config import (
    DEFAULT_REF_IMAGE,
    FRAME_QUEUE_SIZE,
    KEY_ORDER,
    MAX_BLOCKS,
    OUTPUT_DIR,
    STREAM_HEIGHT,
    STREAM_WIDTH,
    VIDEO_FPS,
)
from web_client.pipeline_loader import decode_block_to_frames
from web_client.state import state

logger = logging.getLogger("ws_stream")

JPEG_QUALITY = max(35, min(90, int(os.environ.get("ABOT_JPEG_QUALITY", "65"))))
SESSION_TIMEOUT = int(os.environ.get("ABOT_SESSION_TIMEOUT", "90"))
SESSION_MAX_BLOCKS = max(
    1,
    min(MAX_BLOCKS, int(os.environ.get("ABOT_SESSION_MAX_BLOCKS", "120"))),
)
PERF_LOG_EVERY = max(1, int(os.environ.get("ABOT_PERF_LOG_EVERY", "1")))
WS_SEND_FPS = max(1, int(os.environ.get("ABOT_WS_SEND_FPS", str(VIDEO_FPS))))
WS_CATCHUP_FPS = max(
    WS_SEND_FPS,
    int(os.environ.get("ABOT_WS_CATCHUP_FPS", str(WS_SEND_FPS))),
)
WS_CATCHUP_QUEUE_THRESHOLD = max(
    1,
    int(os.environ.get("ABOT_WS_CATCHUP_QUEUE_THRESHOLD", "4")),
)
WS_CATCHUP_MIN_GENERATED_FPS = max(
    0.0,
    float(os.environ.get("ABOT_WS_CATCHUP_MIN_GENERATED_FPS", "11.5")),
)
WS_MAX_INFLIGHT = max(1, min(8, int(os.environ.get("ABOT_WS_MAX_INFLIGHT", "2"))))
WS_ACK_TIMEOUT = max(0.5, float(os.environ.get("ABOT_WS_ACK_TIMEOUT", "2.0")))
DISPLAY_FPS_MIN = max(
    1,
    min(VIDEO_FPS, int(os.environ.get("ABOT_DISPLAY_FPS_MIN", "10"))),
)
DISPLAY_FPS_TARGET = max(
    DISPLAY_FPS_MIN,
    min(VIDEO_FPS, int(os.environ.get("ABOT_DISPLAY_FPS_TARGET", "11"))),
)
CLIENT_JITTER_PRIME = max(
    1,
    int(os.environ.get("ABOT_CLIENT_JITTER_PRIME", "2")),
)
CLIENT_JITTER_TARGET = max(
    CLIENT_JITTER_PRIME,
    int(os.environ.get("ABOT_CLIENT_JITTER_TARGET", "3")),
)
CLIENT_JITTER_MAX = max(
    CLIENT_JITTER_TARGET + 1,
    int(os.environ.get("ABOT_CLIENT_JITTER_MAX", "6")),
)
# frame_id, block_id, applied_control_seq, generated_fps, generated_at_ms
PACKET_HEADER = struct.Struct("!IIIfd")

_pipeline = None
_config = None
_device = None
_quantized_linear_layers = 0
_fp8_gemm_enabled = False
_quant_mode = "bf16"
_recent_block_metrics: deque[dict] = deque(
    maxlen=max(20, int(os.environ.get("ABOT_METRICS_HISTORY_BLOCKS", "120")))
)
_recent_metrics_lock = threading.Lock()

_gpu_condition = threading.Condition()
_gpu_waiting: deque[str] = deque()
_active_session_id: Optional[str] = None


def set_pipeline(p, cfg, dev):
    global _pipeline, _config, _device
    global _quantized_linear_layers, _fp8_gemm_enabled, _quant_mode
    _pipeline, _config, _device = p, cfg, dev
    _fp8_gemm_enabled = bool(getattr(cfg, "use_fp8_gemm", False))
    _quant_mode = str(getattr(cfg, "quant_mode", "fp8" if _fp8_gemm_enabled else "bf16"))
    _quantized_linear_layers = sum(
        1
        for module in p.generator.model.modules()
        if module.__class__.__name__.startswith(("FP8", "FP4"))
        and module.__class__.__name__.endswith("Linear")
    )


def is_model_ready():
    return _pipeline is not None


def _gpu_runtime_metrics():
    if not torch.cuda.is_available():
        return {}
    result = {
        "allocated_gib": round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2),
        "reserved_gib": round(torch.cuda.memory_reserved(0) / (1024 ** 3), 2),
    }
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,clocks.sm,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        values = [item.strip() for item in completed.stdout.splitlines()[0].split(",")]
        def numeric(index):
            try:
                return float(values[index])
            except (IndexError, TypeError, ValueError):
                return None
        result.update(
            {
                "gpu_util_percent": numeric(0),
                "memory_util_percent": numeric(1),
                "sm_clock_mhz": numeric(2),
                "power_w": numeric(3),
                "power_limit_w": numeric(4),
            }
        )
    except Exception as exc:
        result["telemetry_error"] = str(exc)
    return result


def _recent_metrics_snapshot():
    with _recent_metrics_lock:
        history = list(_recent_block_metrics)
    summary = {"sample_blocks": len(history)}
    for source_key, output_key in (
        ("diffusion_ms", "diffusion_ms"),
        ("vae_ms", "vae_ms"),
        ("postprocess_ms", "postprocess_ms"),
        ("overhead_ms", "overhead_ms"),
        ("block_ms", "block_ms"),
        ("generated_fps", "generated_fps"),
    ):
        values = [
            float(item[source_key])
            for item in history
            if source_key in item
        ]
        if not values:
            continue
        ordered = sorted(values)
        p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
        summary[f"{output_key}_p50"] = round(statistics.median(ordered), 3)
        summary[f"{output_key}_p95"] = round(ordered[p95_index], 3)
    return history, summary


class GameSession:
    def __init__(self, session_id: str, seed_path: str, prompt: str, seed: int):
        self.session_id = session_id
        self.seed_path = seed_path
        self.seed = seed
        self.frame_queue: queue.Queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.status_queue: queue.Queue = queue.Queue(maxsize=8)
        self.stop_event = threading.Event()
        self.last_touch = time.time()
        self.worker: Optional[threading.Thread] = None
        self._control_lock = threading.Lock()
        self._buttons: set[str] = set()
        self._activated_buttons: set[str] = set()
        self._control_seq = 0
        self._prompt = prompt
        self._metrics_lock = threading.Lock()
        self._metrics = {
            "block": 0,
            "generated_frames": 0,
            "generated_fps": 0.0,
            "diffusion_ms": 0.0,
            "vae_ms": 0.0,
            "postprocess_ms": 0.0,
            "overhead_ms": 0.0,
            "block_ms": 0.0,
            "jpeg_ms": 0.0,
            "jpeg_kib": 0.0,
            "server_queue": 0,
            "server_send_target_fps": float(WS_SEND_FPS),
            "server_catchup_active": False,
            "server_sent_frames": 0,
            "server_catchup_frames": 0,
            "dropped_frames": 0,
            "control_dropped_frames": 0,
            "overflow_dropped_frames": 0,
            "control_seq": 0,
            "inflight_frames": 0,
            "last_acked_frame": 0,
            "ack_timeouts": 0,
            "client_actual_display_fps": 0.0,
            "client_display_fps_p50": 0.0,
            "client_display_fps_p95": 0.0,
            "client_playback_target_fps": 0.0,
            "client_buffer_frames": 0,
            "client_playback_primed": False,
            "client_displayed_frames": 0,
            "client_buffer_underruns": 0,
            "client_receive_age_ms": 0.0,
            "client_receive_age_ms_p50": 0.0,
            "client_receive_age_ms_p95": 0.0,
            "client_display_age_ms": 0.0,
            "client_display_age_ms_p50": 0.0,
            "client_display_age_ms_p95": 0.0,
            "client_control_response_ms": 0.0,
            "client_control_response_ms_p50": 0.0,
            "client_control_response_ms_p95": 0.0,
            "client_clock_synced": False,
            "client_clock_rtt_ms": 0.0,
            "client_clock_offset_ms": 0.0,
            "client_overflow_drops": 0,
            "client_control_drops": 0,
            "client_stale_seq_drops": 0,
            "client_active_control_seq": 0,
            "client_requested_control_seq": 0,
            "queue_position": 0,
            "lease_blocks": SESSION_MAX_BLOCKS,
            "lease_blocks_remaining": SESSION_MAX_BLOCKS,
            "session_state": "created",
        }

    def touch(self):
        self.last_touch = time.time()

    @staticmethod
    def _valid_buttons(buttons) -> set[str]:
        return {
            str(key).upper()
            for key in (buttons or [])
            if str(key).upper() in KEY_ORDER
        }

    def update_control(
        self,
        buttons=None,
        activated=None,
        prompt=None,
        control_seq=None,
    ):
        control_changed = False
        with self._control_lock:
            if buttons is not None:
                new_buttons = self._valid_buttons(buttons)
                if new_buttons != self._buttons:
                    control_changed = True
                self._buttons = new_buttons
            if activated:
                # A short tap may start and end between two block boundaries.
                # Latch it until the generator consumes the next action sample.
                self._activated_buttons.update(self._valid_buttons(activated))
                control_changed = True
            if control_seq is not None:
                try:
                    new_seq = max(0, int(control_seq))
                except (TypeError, ValueError):
                    new_seq = self._control_seq
                if new_seq > self._control_seq:
                    self._control_seq = new_seq
                    control_changed = True
            if prompt is not None:
                self._prompt = str(prompt)
        if control_changed:
            # Preserve queued frames while the next control block is generated.
            # The browser keeps playing them and switches atomically when fresh
            # frames carrying the requested control sequence arrive.
            self.update_metrics(control_seq=self._control_seq)
        self.touch()

    def discard_frames_before_control_seq(self, control_seq: int):
        """Drop only the old tail once a fresh control block is ready."""
        retained = []
        discarded = 0
        while True:
            try:
                item = self.frame_queue.get_nowait()
            except queue.Empty:
                break
            if item[3] >= control_seq:
                retained.append(item)
            else:
                discarded += 1
        for item in retained:
            try:
                self.frame_queue.put_nowait(item)
            except queue.Full:
                break
        if discarded:
            with self._metrics_lock:
                self._metrics["dropped_frames"] += discarded
                self._metrics["control_dropped_frames"] += discarded
                self._metrics["server_queue"] = self.frame_queue.qsize()
        return discarded

    def snapshot_control(self):
        with self._control_lock:
            return set(self._buttons), self._prompt

    def consume_control(self):
        with self._control_lock:
            buttons = set(self._buttons)
            buttons.update(self._activated_buttons)
            self._activated_buttons.clear()
            return buttons, self._prompt, self._control_seq

    def clear_control(self):
        with self._control_lock:
            self._buttons.clear()
            self._activated_buttons.clear()
        self.touch()

    def push_status(self, msg: str):
        while self.status_queue.full():
            try:
                self.status_queue.get_nowait()
            except queue.Empty:
                break
        try:
            self.status_queue.put_nowait(msg)
        except queue.Full:
            pass

    def update_metrics(self, **values):
        with self._metrics_lock:
            self._metrics.update(values)

    def metrics(self):
        with self._metrics_lock:
            return dict(self._metrics)

    def enqueue_frame(self, item):
        try:
            self.frame_queue.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            self.frame_queue.get_nowait()
            with self._metrics_lock:
                self._metrics["dropped_frames"] += 1
                self._metrics["overflow_dropped_frames"] += 1
            self.frame_queue.put_nowait(item)
        except (queue.Empty, queue.Full):
            pass

_sessions: dict[str, GameSession] = {}
_sessions_lock = threading.Lock()


def _get_session(session_id: str) -> Optional[GameSession]:
    with _sessions_lock:
        return _sessions.get(session_id)


def _refresh_queue_positions_locked():
    for index, sid in enumerate(_gpu_waiting, start=1):
        session = _get_session(sid)
        if session is not None:
            session.update_metrics(
                queue_position=index,
                session_state="queued",
            )


def _enqueue_for_gpu(session: GameSession):
    with _gpu_condition:
        _gpu_waiting.append(session.session_id)
        _refresh_queue_positions_locked()
        position = list(_gpu_waiting).index(session.session_id) + 1
        if _active_session_id is not None:
            session.push_status(f"GPU忙，排队中 · 前方 {position - 1} 位")
        else:
            session.push_status("等待 GPU 调度…")
        _gpu_condition.notify_all()


def _cancel_gpu_wait(session_id: str):
    with _gpu_condition:
        with contextlib.suppress(ValueError):
            _gpu_waiting.remove(session_id)
        _refresh_queue_positions_locked()
        _gpu_condition.notify_all()


def _acquire_gpu_turn(session: GameSession) -> bool:
    global _active_session_id
    last_position = None
    with _gpu_condition:
        while True:
            if session.stop_event.is_set():
                with contextlib.suppress(ValueError):
                    _gpu_waiting.remove(session.session_id)
                _refresh_queue_positions_locked()
                _gpu_condition.notify_all()
                return False

            is_head = bool(_gpu_waiting) and _gpu_waiting[0] == session.session_id
            if _active_session_id is None and is_head:
                _gpu_waiting.popleft()
                _active_session_id = session.session_id
                _refresh_queue_positions_locked()
                session.update_metrics(
                    queue_position=0,
                    session_state="running",
                )
                session.push_status(
                    f"GPU 已分配 · 本次最多 {SESSION_MAX_BLOCKS} 块"
                )
                return True

            try:
                position = list(_gpu_waiting).index(session.session_id) + 1
            except ValueError:
                return False
            if position != last_position:
                session.update_metrics(
                    queue_position=position,
                    session_state="queued",
                )
                session.push_status(f"GPU忙，排队中 · 前方 {position - 1} 位")
                last_position = position
            _gpu_condition.wait(timeout=0.5)


def _release_gpu_turn(session: GameSession):
    global _active_session_id
    with _gpu_condition:
        if _active_session_id == session.session_id:
            _active_session_id = None
        with contextlib.suppress(ValueError):
            _gpu_waiting.remove(session.session_id)
        session.update_metrics(queue_position=0, session_state="finished")
        _refresh_queue_positions_locked()
        _gpu_condition.notify_all()


def _cleanup_sessions():
    now = time.time()
    with _sessions_lock:
        expired = [
            sid for sid, session in _sessions.items()
            if now - session.last_touch > SESSION_TIMEOUT
        ]
    for sid in expired:
        session = _get_session(sid)
        if session is not None:
            session.stop_event.set()
            _cancel_gpu_wait(sid)
            logger.info("[WS] Session %s expired", sid)
        with _sessions_lock:
            _sessions.pop(sid, None)


def _gpu_worker(session: GameSession):
    """Generate one latent block, decode it, then publish display frames."""
    pipeline = _pipeline
    config = _config
    device = _device

    if not _acquire_gpu_turn(session):
        return
    logger.info("[WS] GPU lease acquired for session %s", session.session_id)

    try:
        state.shared_prompt = session.snapshot_control()[1]
        state.is_running = True
        state.block_count = 0
        state.frame_count = 0
        state.reset_stats()

        num_fpb = int(getattr(
            config,
            "num_frame_per_block",
            getattr(config, "model_kwargs", {}).get("num_frame_per_block", 1),
        ))
        shape_vae = pipeline.encoder if pipeline.encoder is not None else pipeline.vae
        upsampling_factor = getattr(shape_vae, "upsampling_factor", 8)
        latent_channels = shape_vae.z_dim
        latent_height = STREAM_HEIGHT // upsampling_factor
        latent_width = STREAM_WIDTH // upsampling_factor
        generator = torch.Generator(device=device).manual_seed(session.seed)
        noise = torch.randn(
            [1, num_fpb, latent_channels, latent_height, latent_width],
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )

        buttons, current_prompt = session.snapshot_control()
        pipeline.torch_dtype = torch.bfloat16
        pipeline.set_prompts([current_prompt], device=device)

        ref_image = session.seed_path or DEFAULT_REF_IMAGE
        ref_hash = __import__("hashlib").md5(PathLib(ref_image).read_bytes()).hexdigest()[:16]
        ref_cache_dir = OUTPUT_DIR / "ref_image_cache" / ref_hash
        logger.info(
            "[WS] Session %s seed=%s cache=%s exists=%s",
            session.session_id,
            ref_image,
            ref_cache_dir,
            ref_cache_dir.exists(),
        )
        pipeline.set_ref_latent_mask_from_exists_paths(
            ref_dir=str(ref_cache_dir),
            device=device,
        )
        pipeline.reset_stream(
            batch_size=1,
            dtype=torch.bfloat16,
            device=device,
            initial_latent=None,
        )
        pipeline.set_first_frame_latent(
            ref_image,
            height=STREAM_HEIGHT,
            width=STREAM_WIDTH,
            device=device,
        )
        session.push_status("准备中…")

        frame_id = 0
        published_control_seq = 0
        with torch.inference_mode():
            while (
                not session.stop_event.is_set()
                and state.block_count < SESSION_MAX_BLOCKS
            ):
                buttons, prompt, applied_control_seq = session.consume_control()
                if prompt != current_prompt:
                    current_prompt = prompt
                    state.shared_prompt = prompt
                    pipeline.set_prompts([prompt], device=device)

                key_snapshot = {key: key in buttons for key in KEY_ORDER}
                block_t0 = time.monotonic()
                pipeline.set_act(
                    key_snapshot,
                    height=STREAM_HEIGHT,
                    width=STREAM_WIDTH,
                    num_frames=num_fpb,
                    device=device,
                )
                lat_block = pipeline.generate_next_block(noise)
                if session.stop_event.is_set():
                    break
                noise = torch.randn_like(noise)
                frames = decode_block_to_frames(pipeline, lat_block)
                block_elapsed = time.monotonic() - block_t0

                state.block_count += 1
                state.frame_count += len(frames)
                generated_fps = len(frames) / max(block_elapsed, 1e-6)
                diffusion_s = pipeline._stream_block_diffusion_times[-1]
                vae_s = pipeline._stream_block_decode_times[-1]
                post_s = pipeline._stream_block_postprocess_times[-1]
                overhead_s = max(
                    0.0,
                    block_elapsed - diffusion_s - vae_s - post_s,
                )
                generated_at_ms = time.time() * 1000.0

                if applied_control_seq > published_control_seq:
                    # Old frames kept playback continuous during inference.
                    # Now that the fresh block exists, remove only the unsent
                    # old tail so the new action is not delayed behind it.
                    session.discard_frames_before_control_seq(
                        applied_control_seq
                    )
                    published_control_seq = applied_control_seq

                for frame in frames:
                    frame_id += 1
                    session.enqueue_frame((
                        np.asarray(frame),
                        frame_id,
                        state.block_count,
                        applied_control_seq,
                        generated_fps,
                        generated_at_ms,
                    ))

                session.update_metrics(
                    block=state.block_count,
                    generated_frames=state.frame_count,
                    generated_fps=round(generated_fps, 3),
                    diffusion_ms=round(diffusion_s * 1000, 2),
                    vae_ms=round(vae_s * 1000, 2),
                    postprocess_ms=round(post_s * 1000, 2),
                    overhead_ms=round(overhead_s * 1000, 2),
                    block_ms=round(block_elapsed * 1000, 2),
                    server_queue=session.frame_queue.qsize(),
                    control_seq=applied_control_seq,
                    lease_blocks_remaining=max(
                        0,
                        SESSION_MAX_BLOCKS - state.block_count,
                    ),
                )
                with _recent_metrics_lock:
                    _recent_block_metrics.append(
                        {
                            "timestamp": round(time.time(), 3),
                            "session_id": session.session_id,
                            "block": state.block_count,
                            "diffusion_ms": round(diffusion_s * 1000, 2),
                            "vae_ms": round(vae_s * 1000, 2),
                            "postprocess_ms": round(post_s * 1000, 2),
                            "overhead_ms": round(overhead_s * 1000, 2),
                            "block_ms": round(block_elapsed * 1000, 2),
                            "generated_fps": round(generated_fps, 3),
                        }
                    )
                session.push_status(
                    f"已生成 {state.block_count}/{SESSION_MAX_BLOCKS} 块 · "
                    f"{state.frame_count} 帧 · {generated_fps:.1f} fps"
                )
                if state.block_count % PERF_LOG_EVERY == 0:
                    metrics = session.metrics()
                    logger.info(
                        "[PERF] session=%s block=%d diffusion=%.1fms "
                        "vae=%.1fms post=%.1fms overhead=%.1fms "
                        "total=%.1fms fps=%.2f "
                        "queue=%d dropped=%d",
                        session.session_id,
                        state.block_count,
                        metrics["diffusion_ms"],
                        metrics["vae_ms"],
                        metrics["postprocess_ms"],
                        metrics["overhead_ms"],
                        metrics["block_ms"],
                        metrics["generated_fps"],
                        metrics["server_queue"],
                        metrics["dropped_frames"],
                    )

        session.push_status(
            "本次体验已结束，GPU 将交给下一位"
            if state.block_count >= SESSION_MAX_BLOCKS
            else "已停止"
        )
    except Exception as exc:
        logger.exception("[WS] Worker error: %s", exc)
        session.push_status(f"error:{exc}")
    finally:
        state.is_running = False
        session.stop_event.set()
        _release_gpu_turn(session)
        logger.info("[WS] GPU lease released for session %s", session.session_id)


def start_game(seed_path: str, prompt: str, seed: int = 42) -> str:
    _cleanup_sessions()
    session_id = uuid.uuid4().hex[:12]
    session = GameSession(session_id, seed_path, prompt, seed)
    with _sessions_lock:
        _sessions[session_id] = session
    _enqueue_for_gpu(session)
    session.worker = threading.Thread(
        target=_gpu_worker,
        args=(session,),
        daemon=True,
    )
    session.worker.start()
    logger.info("[WS] Started session %s", session_id)
    return session_id


def stop_game(session_id: str):
    session = _get_session(session_id)
    if session is not None:
        session.clear_control()
        session.stop_event.set()
        _cancel_gpu_wait(session_id)
        logger.info("[WS] Stopped session %s", session_id)


def _session_reaper():
    while True:
        time.sleep(5)
        _cleanup_sessions()


threading.Thread(
    target=_session_reaper,
    name="abot-session-reaper",
    daemon=True,
).start()


def _encode_jpeg(frame: np.ndarray):
    started = time.perf_counter()
    buffer = io.BytesIO()
    Image.fromarray(frame).save(
        buffer,
        format="JPEG",
        quality=JPEG_QUALITY,
        optimize=False,
    )
    jpeg = buffer.getvalue()
    return jpeg, (time.perf_counter() - started) * 1000.0


def _stream_page_html() -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#000; overflow:hidden; width:100vw; height:100vh; }}
#canvas {{ width:100%; height:100%; object-fit:cover; }}
#status {{ position:absolute; top:8px; left:8px; color:#0ff; font:14px monospace;
  background:rgba(0,0,0,.65); padding:4px 8px; border-radius:4px; }}
</style>
</head>
<body>
<canvas id="canvas" tabindex="0" width="{STREAM_WIDTH}" height="{STREAM_HEIGHT}"></canvas>
<div id="status">连接中…</div>
<script>
const SESSION_ID = new URLSearchParams(location.search).get('session_id') || '';
const DISPLAY_FPS_MIN = {DISPLAY_FPS_MIN};
const DISPLAY_FPS_TARGET = {DISPLAY_FPS_TARGET};
const DISPLAY_FPS_MAX = {VIDEO_FPS};
const HEADER_BYTES = {PACKET_HEADER.size};
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', {{alpha:false, desynchronized:true}});
const statusEl = document.getElementById('status');
const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
const ws = new WebSocket(proto + '//' + location.host +
  '/stream_ws?session_id=' + encodeURIComponent(SESSION_ID));
ws.binaryType = 'arraybuffer';

const JITTER_PRIME = {CLIENT_JITTER_PRIME};
const CONTROL_PRIME = 1;
const JITTER_TARGET = {CLIENT_JITTER_TARGET};
const JITTER_MAX = {CLIENT_JITTER_MAX};
let jitterBuf = [];
let playbackPrimed = false;
let primeThreshold = JITTER_PRIME;
let nextPaintAt = 0;
let serverStatus = '连接中…';
let latestFps = 0;
let supplyFpsEma = 0;
let lastSupplyBlockId = 0;
let overflowDrops = 0;
let controlDrops = 0;
let staleSeqDrops = 0;
let actualDisplayFps = 0;
let displayedInWindow = 0;
let displayWindowStarted = performance.now();
let controlSeq = 0;
let requestedControlSeq = 0;
let activeControlSeq = 0;
let lastDisplayedControlSeq = 0;
let displayedFramesTotal = 0;
let bufferUnderruns = 0;
let playbackStarted = false;
let bufferWasEmpty = true;
let receiveAgeMs = 0;
let displayAgeMs = 0;
let controlResponseMs = 0;
let clockSynced = false;
let clockRttMs = 0;
let clockOffsetMs = 0;
const METRIC_WINDOW = 120;
const displayFpsSamples = [];
const receiveAgeSamples = [];
const displayAgeSamples = [];
const controlResponseSamples = [];
const controlSentAt = new Map();
const pendingActivated = new Set();

function addSample(samples, value) {{
  if (!Number.isFinite(value) || value < 0) return;
  samples.push(value);
  if (samples.length > METRIC_WINDOW) samples.shift();
}}

function percentile(samples, fraction) {{
  if (!samples.length) return 0;
  const ordered = samples.slice().sort((a, b) => a - b);
  const index = Math.min(
    ordered.length - 1,
    Math.max(0, Math.ceil(ordered.length * fraction) - 1)
  );
  return ordered[index];
}}

function serverTimestampOnClientClock(serverTimestamp) {{
  return serverTimestamp - clockOffsetMs;
}}

function sendClockSync() {{
  if (ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({{
    type:'clock_sync',
    client_sent_ms:Date.now()
  }}));
}}

function closeEntry(entry) {{
  if (entry && entry.bitmap && entry.bitmap.close) entry.bitmap.close();
}}

function activateControlSequence(sequence) {{
  const retained = [];
  let dropped = 0;
  for (const entry of jitterBuf) {{
    if (entry.appliedControlSeq >= sequence) {{
      retained.push(entry);
    }} else {{
      closeEntry(entry);
      dropped++;
    }}
  }}
  jitterBuf = retained;
  controlDrops += dropped;
  activeControlSeq = sequence;
  playbackPrimed = false;
  primeThreshold = CONTROL_PRIME;
  nextPaintAt = 0;
}}

function sendAck(frameId) {{
  if (ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({{type:'ack', frame_id:frameId}}));
}}

ws.onmessage = async (event) => {{
  if (typeof event.data === 'string') {{
    try {{
      const msg = JSON.parse(event.data);
      if (msg.type === 'status') {{
        serverStatus = msg.message;
        statusEl.textContent = serverStatus;
      }}
      if (msg.type === 'clock_sync') {{
        const receivedAt = Date.now();
        const sentAt = Number(msg.client_sent_ms);
        const serverReceivedAt = Number(msg.server_received_ms);
        const rtt = receivedAt - sentAt;
        if (
          Number.isFinite(rtt) &&
          rtt >= 0 &&
          Number.isFinite(serverReceivedAt) &&
          (!clockSynced || rtt < clockRttMs)
        ) {{
          clockRttMs = rtt;
          clockOffsetMs = serverReceivedAt - (sentAt + receivedAt) / 2;
          clockSynced = true;
        }}
      }}
      if (msg.type === 'ended') serverStatus = '已结束';
      if (msg.type === 'error') {{
        serverStatus = '错误';
        statusEl.textContent = '连接错误';
      }} else if (msg.type === 'ended') {{
        statusEl.textContent = '已结束 · ' + actualDisplayFps.toFixed(1) + ' FPS';
      }}
    }} catch (_) {{}}
    return;
  }}
  try {{
    const buffer = event.data;
    if (buffer.byteLength <= HEADER_BYTES) return;
    const view = new DataView(buffer);
    const frameId = view.getUint32(0);
    const blockId = view.getUint32(4);
    const appliedControlSeq = view.getUint32(8);
    const fps = view.getFloat32(12);
    const generatedAt = view.getFloat64(16);
    receiveAgeMs = clockSynced
      ? Math.max(0, Date.now() - serverTimestampOnClientClock(generatedAt))
      : 0;
    if (clockSynced) addSample(receiveAgeSamples, receiveAgeMs);
    serverStatus = '播放中';
    // ACK means the packet reached the browser. The bounded server window then
    // limits data queued in ASGI/TCP/ModelScope before the browser.
    sendAck(frameId);
    if (appliedControlSeq < activeControlSeq) {{
      staleSeqDrops++;
      return;
    }}
    const jpeg = buffer.slice(HEADER_BYTES);
    const bitmap = await createImageBitmap(new Blob([jpeg], {{type:'image/jpeg'}}));
    // A newer control sequence may become active while JPEG decoding is in
    // flight. Reject only frames older than the sequence already on screen.
    if (appliedControlSeq < activeControlSeq) {{
      closeEntry({{bitmap}});
      staleSeqDrops++;
      return;
    }}
    // Continue playing old frames while inference works. Once a frame from the
    // latest requested sequence is ready, discard only the unplayed old tail.
    // The complete fresh block is already queued server-side, so one decoded
    // frame is enough to switch without introducing another visible pause.
    if (
      appliedControlSeq >= requestedControlSeq &&
      appliedControlSeq > activeControlSeq
    ) {{
      activateControlSequence(appliedControlSeq);
    }}
    jitterBuf.push({{
      bitmap, frameId, blockId, appliedControlSeq, fps, generatedAt
    }});
    bufferWasEmpty = false;
    jitterBuf.sort((a, b) => a.frameId - b.frameId);
    if (!playbackPrimed && jitterBuf.length >= primeThreshold) {{
      playbackPrimed = true;
      playbackStarted = true;
      primeThreshold = JITTER_PRIME;
    }}
    latestFps = fps;
    if (blockId !== lastSupplyBlockId) {{
      supplyFpsEma = supplyFpsEma
        ? 0.25 * fps + 0.75 * supplyFpsEma
        : fps;
      lastSupplyBlockId = blockId;
    }}
    if (jitterBuf.length > JITTER_MAX) {{
      while (jitterBuf.length > JITTER_TARGET) {{
        closeEntry(jitterBuf.shift());
        overflowDrops++;
      }}
    }}
  }} catch (err) {{
    console.error('frame decode', err);
  }}
}};

ws.onopen = () => {{
  statusEl.textContent = '已连接，等待首帧…';
  sendClockSync();
  setTimeout(sendClockSync, 500);
  setTimeout(sendClockSync, 1000);
  sendControl();
}};
ws.onerror = () => {{ statusEl.textContent = '连接错误'; }};
ws.onclose = () => {{
  if (serverStatus !== '已结束') statusEl.textContent = '连接已关闭';
}};

function currentPlaybackFps() {{
  // When generation temporarily falls below the 11 FPS playback target,
  // preserve continuity by slowing down before the buffer reaches one frame.
  const lowSupply = supplyFpsEma > 0 && supplyFpsEma < DISPLAY_FPS_TARGET;
  const slowThreshold = lowSupply ? JITTER_TARGET : 1;
  if (jitterBuf.length <= slowThreshold) return DISPLAY_FPS_MIN;
  if (jitterBuf.length >= JITTER_TARGET + 1) return DISPLAY_FPS_MAX;
  return DISPLAY_FPS_TARGET;
}}

function renderClock(now) {{
  requestAnimationFrame(renderClock);
  const fpsElapsed = now - displayWindowStarted;
  if (fpsElapsed >= 1000) {{
    actualDisplayFps = displayedInWindow * 1000 / fpsElapsed;
    if (actualDisplayFps > 0) addSample(displayFpsSamples, actualDisplayFps);
    displayedInWindow = 0;
    displayWindowStarted = now;
  }}
  if (jitterBuf.length === 0) {{
    if (playbackStarted && !bufferWasEmpty) {{
      bufferUnderruns++;
      bufferWasEmpty = true;
    }}
    playbackPrimed = false;
    nextPaintAt = 0;
    return;
  }}
  if (!playbackPrimed) return;
  const playbackFps = currentPlaybackFps();
  const frameInterval = 1000 / playbackFps;
  if (!nextPaintAt) nextPaintAt = now;
  if (now < nextPaintAt) return;
  const entry = jitterBuf.shift();
  nextPaintAt += frameInterval;
  // Keep the deadline phase across 60 Hz requestAnimationFrame ticks. This
  // avoids a nominal 12 FPS target collapsing to 10 FPS after every late tick.
  if (nextPaintAt < now - frameInterval) {{
    nextPaintAt = now + frameInterval;
  }}
  ctx.drawImage(entry.bitmap, 0, 0, canvas.width, canvas.height);
  if (clockSynced) {{
    displayAgeMs = Math.max(
      0,
      Date.now() - serverTimestampOnClientClock(entry.generatedAt)
    );
  addSample(displayAgeSamples, displayAgeMs);
  }}
  if (entry.appliedControlSeq > lastDisplayedControlSeq) {{
    let earliestSentAt = Infinity;
    for (const [sequence, sentAt] of controlSentAt.entries()) {{
      if (
        sequence > lastDisplayedControlSeq &&
        sequence <= entry.appliedControlSeq &&
        Number.isFinite(sentAt)
      ) {{
        earliestSentAt = Math.min(earliestSentAt, sentAt);
      }}
    }}
    if (Number.isFinite(earliestSentAt)) {{
      controlResponseMs = Math.max(0, Date.now() - earliestSentAt);
      addSample(controlResponseSamples, controlResponseMs);
    }}
    lastDisplayedControlSeq = entry.appliedControlSeq;
    for (const sequence of controlSentAt.keys()) {{
      if (sequence <= lastDisplayedControlSeq) controlSentAt.delete(sequence);
    }}
  }}
  closeEntry(entry);
  displayedInWindow++;
  displayedFramesTotal++;
  statusEl.textContent = actualDisplayFps.toFixed(1) + ' FPS';
}}
requestAnimationFrame(renderClock);

function sendClientMetrics() {{
  if (ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({{
    type:'client_metrics',
    actual_display_fps:actualDisplayFps,
    display_fps_p50:percentile(displayFpsSamples, 0.50),
    display_fps_p95:percentile(displayFpsSamples, 0.95),
    playback_target_fps:currentPlaybackFps(),
    buffer_frames:jitterBuf.length,
    playback_primed:playbackPrimed,
    displayed_frames:displayedFramesTotal,
    buffer_underruns:bufferUnderruns,
    receive_age_ms:receiveAgeMs,
    receive_age_ms_p50:percentile(receiveAgeSamples, 0.50),
    receive_age_ms_p95:percentile(receiveAgeSamples, 0.95),
    display_age_ms:displayAgeMs,
    display_age_ms_p50:percentile(displayAgeSamples, 0.50),
    display_age_ms_p95:percentile(displayAgeSamples, 0.95),
    control_response_ms:controlResponseMs,
    control_response_ms_p50:percentile(controlResponseSamples, 0.50),
    control_response_ms_p95:percentile(controlResponseSamples, 0.95),
    clock_synced:clockSynced,
    clock_rtt_ms:clockRttMs,
    clock_offset_ms:clockOffsetMs,
    overflow_drops:overflowDrops,
    control_drops:controlDrops,
    stale_seq_drops:staleSeqDrops,
    active_control_seq:activeControlSeq,
    requested_control_seq:requestedControlSeq
  }}));
}}

setInterval(() => {{
  if (serverStatus !== '已结束') {{
    if (
      serverStatus.includes('排队') ||
      serverStatus.includes('等待') ||
      serverStatus.includes('分配') ||
      serverStatus.includes('准备')
    ) {{
      statusEl.textContent = serverStatus;
    }} else {{
      statusEl.textContent = actualDisplayFps.toFixed(1) + ' FPS';
    }}
  }}
  sendClientMetrics();
}}, 1000);

const pressed = new Set();
const VALID_KEYS = new Set(['W','A','S','D','I','J','K','L']);
const KEY_MAP = {{KeyW:'W', KeyA:'A', KeyS:'S', KeyD:'D',
  KeyI:'I', KeyJ:'J', KeyK:'K', KeyL:'L'}};

function sendControl(activated = [], stateChanged = false, eventSentAt = 0) {{
  if (stateChanged) {{
    controlSeq++;
    // Do not interrupt playback. Old frames remain visible until frames from
    // this requested sequence have actually arrived and decoded.
    requestedControlSeq = controlSeq;
    const sentAt = Number(eventSentAt);
    controlSentAt.set(
      controlSeq,
      Number.isFinite(sentAt) && sentAt > 0 ? sentAt : Date.now()
    );
    while (controlSentAt.size > 64) {{
      controlSentAt.delete(controlSentAt.keys().next().value);
    }}
  }}
  if (activated.length) {{
    pendingActivated.clear();
    for (const key of activated) {{
      if (VALID_KEYS.has(key)) pendingActivated.add(key);
    }}
  }}
  if (ws.readyState !== WebSocket.OPEN) return;
  try {{
    ws.send(JSON.stringify({{
      type:'control',
      buttons:Array.from(pressed),
      activated:Array.from(pendingActivated),
      control_seq:controlSeq
    }}));
    pendingActivated.clear();
  }} catch (err) {{
    console.warn('control send failed', err);
  }}
}}

// The parent Gradio page owns the primary keyboard listener. Forwarded
// controls work even when the iframe/canvas does not have browser focus.
window.addEventListener('message', event => {{
  if (event.source !== window.parent || event.origin !== window.location.origin) return;
  const data = event.data || {{}};
  if (data.type !== 'abot-control') return;
  pressed.clear();
  for (const key of (data.pressed || [])) {{
    const normalized = String(key).toUpperCase();
    if (VALID_KEYS.has(normalized)) pressed.add(normalized);
  }}
  const activated = (data.activated || [])
    .map(key => String(key).toUpperCase())
    .filter(key => VALID_KEYS.has(key));
  sendControl(activated, true, data.sentAt);
}});

document.addEventListener('keydown', event => {{
  const key = KEY_MAP[event.code];
  if (!key) return;
  event.preventDefault();
  if (!pressed.has(key)) {{
    pressed.add(key);
    sendControl([key], true);
  }}
}});
document.addEventListener('keyup', event => {{
  const key = KEY_MAP[event.code];
  if (!key) return;
  pressed.delete(key);
  sendControl([], true);
}});
canvas.addEventListener('pointerdown', () => canvas.focus());
canvas.addEventListener('mouseenter', () => window.focus());
window.addEventListener('blur', () => {{
  if (pressed.size) {{
    pressed.clear();
    sendControl([], true);
  }}
}});
document.addEventListener('visibilitychange', () => {{
  if (document.hidden && pressed.size) {{
    pressed.clear();
    sendControl([], true);
  }}
}});
setInterval(sendControl, 200);
</script>
</body>
</html>"""


def register_routes(fastapi_app: FastAPI):
    @fastapi_app.get("/stream_page")
    def stream_page(session_id: str = ""):
        return HTMLResponse(content=_stream_page_html())

    @fastapi_app.websocket("/stream_ws")
    async def stream_ws(websocket: WebSocket, session_id: str = ""):
        session = _get_session(session_id)
        if session is None:
            await websocket.close(code=4404, reason="session not found")
            return
        await websocket.accept()
        session.touch()

        inflight = deque()
        ack_event = asyncio.Event()
        client_disconnected = asyncio.Event()
        reply_queue: asyncio.Queue = asyncio.Queue(maxsize=8)
        last_acked_frame = 0

        def bounded_client_number(payload, key, upper):
            try:
                value = float(payload.get(key, 0))
            except (TypeError, ValueError):
                return 0.0
            if not math.isfinite(value):
                return 0.0
            return min(upper, max(0.0, value))

        def bounded_signed_client_number(payload, key, absolute_upper):
            try:
                value = float(payload.get(key, 0))
            except (TypeError, ValueError):
                return 0.0
            if not math.isfinite(value):
                return 0.0
            return min(absolute_upper, max(-absolute_upper, value))

        async def receive_controls():
            nonlocal last_acked_frame
            try:
                while True:
                    raw_message = await websocket.receive_text()
                    try:
                        payload = json.loads(raw_message)
                    except json.JSONDecodeError:
                        logger.warning("[WS] Ignoring malformed client message")
                        continue
                    if payload.get("type") == "control":
                        session.update_control(
                            buttons=payload.get("buttons", []),
                            activated=payload.get("activated", []),
                            prompt=payload.get("prompt"),
                            control_seq=payload.get("control_seq"),
                        )
                    elif payload.get("type") == "clock_sync":
                        try:
                            client_sent_ms = float(
                                payload.get("client_sent_ms", 0)
                            )
                        except (TypeError, ValueError):
                            continue
                        if not math.isfinite(client_sent_ms):
                            continue
                        reply = {
                            "type": "clock_sync",
                            "client_sent_ms": client_sent_ms,
                            "server_received_ms": time.time() * 1000.0,
                        }
                        with contextlib.suppress(asyncio.QueueFull):
                            reply_queue.put_nowait(reply)
                    elif payload.get("type") == "ack":
                        try:
                            acked_frame = max(0, int(payload.get("frame_id", 0)))
                        except (TypeError, ValueError):
                            continue
                        if acked_frame > last_acked_frame:
                            last_acked_frame = acked_frame
                            while inflight and inflight[0] <= acked_frame:
                                inflight.popleft()
                            session.update_metrics(
                                inflight_frames=len(inflight),
                                last_acked_frame=last_acked_frame,
                            )
                            ack_event.set()
                    elif payload.get("type") == "client_metrics":
                        session.update_metrics(
                            client_actual_display_fps=round(
                                bounded_client_number(
                                    payload,
                                    "actual_display_fps",
                                    120.0,
                                ),
                                2,
                            ),
                            client_display_fps_p50=round(
                                bounded_client_number(
                                    payload,
                                    "display_fps_p50",
                                    120.0,
                                ),
                                2,
                            ),
                            client_display_fps_p95=round(
                                bounded_client_number(
                                    payload,
                                    "display_fps_p95",
                                    120.0,
                                ),
                                2,
                            ),
                            client_playback_target_fps=round(
                                bounded_client_number(
                                    payload,
                                    "playback_target_fps",
                                    float(VIDEO_FPS),
                                ),
                                2,
                            ),
                            client_buffer_frames=int(
                                bounded_client_number(
                                    payload,
                                    "buffer_frames",
                                    100.0,
                                )
                            ),
                            client_playback_primed=bool(
                                payload.get("playback_primed", False)
                            ),
                            client_displayed_frames=int(
                                bounded_client_number(
                                    payload,
                                    "displayed_frames",
                                    1_000_000_000.0,
                                )
                            ),
                            client_buffer_underruns=int(
                                bounded_client_number(
                                    payload,
                                    "buffer_underruns",
                                    1_000_000_000.0,
                                )
                            ),
                            client_receive_age_ms=round(
                                bounded_client_number(
                                    payload,
                                    "receive_age_ms",
                                    600_000.0,
                                ),
                                2,
                            ),
                            client_receive_age_ms_p50=round(
                                bounded_client_number(
                                    payload,
                                    "receive_age_ms_p50",
                                    600_000.0,
                                ),
                                2,
                            ),
                            client_receive_age_ms_p95=round(
                                bounded_client_number(
                                    payload,
                                    "receive_age_ms_p95",
                                    600_000.0,
                                ),
                                2,
                            ),
                            client_display_age_ms=round(
                                bounded_client_number(
                                    payload,
                                    "display_age_ms",
                                    600_000.0,
                                ),
                                2,
                            ),
                            client_display_age_ms_p50=round(
                                bounded_client_number(
                                    payload,
                                    "display_age_ms_p50",
                                    600_000.0,
                                ),
                                2,
                            ),
                            client_display_age_ms_p95=round(
                                bounded_client_number(
                                    payload,
                                    "display_age_ms_p95",
                                    600_000.0,
                                ),
                                2,
                            ),
                            client_control_response_ms=round(
                                bounded_client_number(
                                    payload,
                                    "control_response_ms",
                                    600_000.0,
                                ),
                                2,
                            ),
                            client_control_response_ms_p50=round(
                                bounded_client_number(
                                    payload,
                                    "control_response_ms_p50",
                                    600_000.0,
                                ),
                                2,
                            ),
                            client_control_response_ms_p95=round(
                                bounded_client_number(
                                    payload,
                                    "control_response_ms_p95",
                                    600_000.0,
                                ),
                                2,
                            ),
                            client_clock_synced=bool(
                                payload.get("clock_synced", False)
                            ),
                            client_clock_rtt_ms=round(
                                bounded_client_number(
                                    payload,
                                    "clock_rtt_ms",
                                    60_000.0,
                                ),
                                2,
                            ),
                            client_clock_offset_ms=round(
                                bounded_signed_client_number(
                                    payload,
                                    "clock_offset_ms",
                                    86_400_000.0,
                                ),
                                2,
                            ),
                            client_overflow_drops=int(
                                bounded_client_number(
                                    payload,
                                    "overflow_drops",
                                    1_000_000_000.0,
                                )
                            ),
                            client_control_drops=int(
                                bounded_client_number(
                                    payload,
                                    "control_drops",
                                    1_000_000_000.0,
                                )
                            ),
                            client_stale_seq_drops=int(
                                bounded_client_number(
                                    payload,
                                    "stale_seq_drops",
                                    1_000_000_000.0,
                                )
                            ),
                            client_active_control_seq=int(
                                bounded_client_number(
                                    payload,
                                    "active_control_seq",
                                    1_000_000_000.0,
                                )
                            ),
                            client_requested_control_seq=int(
                                bounded_client_number(
                                    payload,
                                    "requested_control_seq",
                                    1_000_000_000.0,
                                )
                            ),
                        )
            except (WebSocketDisconnect, RuntimeError):
                client_disconnected.set()
                session.stop_event.set()

        receiver = asyncio.create_task(receive_controls())
        jpeg_ema_ms = 0.0
        jpeg_ema_kib = 0.0
        generated_fps_ema = 0.0
        last_sent_block_id = -1
        sent_frames = 0
        catchup_frames = 0
        next_send_at = time.monotonic()

        async def wait_for_send_slot():
            while len(inflight) >= WS_MAX_INFLIGHT:
                if client_disconnected.is_set():
                    return False
                ack_event.clear()
                try:
                    await asyncio.wait_for(
                        ack_event.wait(),
                        timeout=WS_ACK_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    # Do not deadlock forever on a stale pre-ACK browser. Drop
                    # one accounting slot, but retain pacing to avoid bursts.
                    if inflight:
                        inflight.popleft()
                    metrics = session.metrics()
                    session.update_metrics(
                        inflight_frames=len(inflight),
                        ack_timeouts=metrics["ack_timeouts"] + 1,
                    )
            return True

        try:
            while True:
                while True:
                    try:
                        reply = reply_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    await websocket.send_json(reply)

                while True:
                    try:
                        status = session.status_queue.get_nowait()
                    except queue.Empty:
                        break
                    if status.startswith("error:"):
                        await websocket.send_json({
                            "type": "error",
                            "message": status[6:],
                        })
                    else:
                        await websocket.send_json({
                            "type": "status",
                            "message": status,
                        })

                if session.stop_event.is_set() and session.frame_queue.empty():
                    await websocket.send_json({"type": "ended"})
                    break

                if not await wait_for_send_slot():
                    break

                try:
                    item = await asyncio.to_thread(
                        session.frame_queue.get,
                        True,
                        0.1,
                    )
                except queue.Empty:
                    continue

                (
                    frame,
                    frame_id,
                    block_id,
                    applied_control_seq,
                    fps,
                    generated_at_ms,
                ) = item
                if block_id != last_sent_block_id:
                    generated_fps_ema = (
                        0.25 * fps + 0.75 * generated_fps_ema
                        if generated_fps_ema
                        else fps
                    )
                    last_sent_block_id = block_id
                jpeg, encode_ms = await asyncio.to_thread(_encode_jpeg, frame)
                send_delay = next_send_at - time.monotonic()
                if send_delay > 0:
                    await asyncio.sleep(send_delay)
                jpeg_kib = len(jpeg) / 1024.0
                jpeg_ema_ms = encode_ms if not jpeg_ema_ms else 0.2 * encode_ms + 0.8 * jpeg_ema_ms
                jpeg_ema_kib = jpeg_kib if not jpeg_ema_kib else 0.2 * jpeg_kib + 0.8 * jpeg_ema_kib
                session.update_metrics(
                    jpeg_ms=round(jpeg_ema_ms, 2),
                    jpeg_kib=round(jpeg_ema_kib, 2),
                    server_queue=session.frame_queue.qsize(),
                )
                packet = PACKET_HEADER.pack(
                    frame_id,
                    block_id,
                    applied_control_seq,
                    float(fps),
                    generated_at_ms,
                ) + jpeg
                await websocket.send_bytes(packet)
                inflight.append(frame_id)
                queue_depth = session.frame_queue.qsize()
                catchup_active = (
                    queue_depth >= WS_CATCHUP_QUEUE_THRESHOLD
                    and generated_fps_ema >= WS_CATCHUP_MIN_GENERATED_FPS
                )
                send_target_fps = (
                    WS_CATCHUP_FPS if catchup_active else WS_SEND_FPS
                )
                sent_frames += 1
                if catchup_active:
                    catchup_frames += 1
                next_send_at = time.monotonic() + 1.0 / send_target_fps
                session.update_metrics(
                    inflight_frames=len(inflight),
                    server_queue=queue_depth,
                    server_send_target_fps=float(send_target_fps),
                    server_catchup_active=catchup_active,
                    server_sent_frames=sent_frames,
                    server_catchup_frames=catchup_frames,
                )
                session.touch()
        except WebSocketDisconnect:
            logger.info("[WS] Client disconnected from session %s", session_id)
        finally:
            receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receiver
            session.clear_control()
            session.stop_event.set()
            _cancel_gpu_wait(session_id)

    @fastapi_app.post("/control")
    async def control(request: Request, session_id: str = ""):
        session = _get_session(session_id)
        if session is None:
            return JSONResponse({"error": "session not found"}, status_code=404)
        body = await request.json()
        session.update_control(
            buttons=body.get("buttons", []),
            activated=body.get("activated", []),
            prompt=body.get("prompt"),
            control_seq=body.get("control_seq"),
        )
        return {"ok": True}

    @fastapi_app.get("/healthz")
    def healthz():
        gpu = {}
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu = {
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_gib": round(props.total_memory / (1024 ** 3), 2),
                "allocated_gib": round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2),
                "reserved_gib": round(torch.cuda.memory_reserved(0) / (1024 ** 3), 2),
            }
        runtime_status = {}
        runtime_status_path = PathLib(
            os.environ.get(
                "ABOT_RUNTIME_STATUS",
                "/tmp/abot-l20n-runtime.json",
            )
        )
        if runtime_status_path.exists():
            try:
                runtime_status = json.loads(
                    runtime_status_path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                runtime_status = {"phase": "invalid_status", "error": str(exc)}
        try:
            import importlib
            attention_module = importlib.import_module("wan.modules.attention")
            attention = {
                "sageattention2": bool(
                    attention_module.SAGE_ATTN_AVAILABLE
                ),
                "sageattention3": bool(
                    attention_module.SAGE_ATTN_3_BLACKWELL_AVAILABLE
                ),
                "flash_attention2": bool(
                    attention_module.FLASH_ATTN_2_AVAILABLE
                ),
                **attention_module.attention_backend_status(),
                "benchmark": getattr(_pipeline, "attention_benchmark", {}),
            }
        except Exception as exc:
            attention = {"error": str(exc)}
        with _gpu_condition:
            scheduler = {
                "active_session": _active_session_id,
                "queued_sessions": list(_gpu_waiting),
                "session_max_blocks": SESSION_MAX_BLOCKS,
            }
        return {
            "ok": state.model_error is None,
            "model_ready": is_model_ready(),
            "model_error": state.model_error,
            "running": state.is_running,
            "gpu": gpu,
            "runtime_bootstrap": runtime_status,
            "attention": attention,
            "inference": {
                "fp8_gemm": _fp8_gemm_enabled,
                "quant_mode": _quant_mode,
                "quantized_linear_layers": _quantized_linear_layers,
                "display_fps_min": DISPLAY_FPS_MIN,
                "display_fps_target": DISPLAY_FPS_TARGET,
                "display_fps_max": VIDEO_FPS,
                "ws_send_fps": WS_SEND_FPS,
                "ws_catchup_fps": WS_CATCHUP_FPS,
                "ws_catchup_queue_threshold": WS_CATCHUP_QUEUE_THRESHOLD,
                "ws_catchup_min_generated_fps": WS_CATCHUP_MIN_GENERATED_FPS,
                "ws_max_inflight": WS_MAX_INFLIGHT,
                "client_jitter_prime": CLIENT_JITTER_PRIME,
                "client_jitter_target": CLIENT_JITTER_TARGET,
                "client_jitter_max": CLIENT_JITTER_MAX,
            },
            "gpu_runtime": _gpu_runtime_metrics(),
            "scheduler": scheduler,
        }

    @fastapi_app.get("/metrics")
    def metrics():
        recent_history, recent_summary = _recent_metrics_snapshot()
        with _gpu_condition:
            scheduler = {
                "active_session": _active_session_id,
                "queued_sessions": list(_gpu_waiting),
                "session_max_blocks": SESSION_MAX_BLOCKS,
            }
        with _sessions_lock:
            sessions = {
                sid: session.metrics()
                for sid, session in _sessions.items()
            }
        try:
            import importlib
            attention_module = importlib.import_module("wan.modules.attention")
            attention_status = attention_module.attention_backend_status()
        except Exception as exc:
            attention_status = {"error": str(exc)}
        return {
            "scheduler": scheduler,
            "sessions": sessions,
            "recent_summary": recent_summary,
            "recent_blocks": recent_history,
            "gpu_runtime": _gpu_runtime_metrics(),
            "inference": {
                "quant_mode": _quant_mode,
                "fp8_gemm": _fp8_gemm_enabled,
                "quantized_linear_layers": _quantized_linear_layers,
                "attention": attention_status,
                "display_fps_min": DISPLAY_FPS_MIN,
                "display_fps_target": DISPLAY_FPS_TARGET,
                "display_fps_max": VIDEO_FPS,
                "ws_send_fps": WS_SEND_FPS,
                "ws_catchup_fps": WS_CATCHUP_FPS,
                "ws_catchup_queue_threshold": WS_CATCHUP_QUEUE_THRESHOLD,
                "ws_catchup_min_generated_fps": WS_CATCHUP_MIN_GENERATED_FPS,
                "ws_max_inflight": WS_MAX_INFLIGHT,
                "client_jitter_prime": CLIENT_JITTER_PRIME,
                "client_jitter_target": CLIENT_JITTER_TARGET,
                "client_jitter_max": CLIENT_JITTER_MAX,
            },
        }

    @fastapi_app.get("/example_seeds")
    def example_seeds():
        from web_client.app import REF_IMAGE_PATHS
        return {"examples": REF_IMAGE_PATHS[:5] if REF_IMAGE_PATHS else []}

    logger.info(
        "[WS] Routes registered: /stream_page, /stream_ws, /control, "
        "/healthz, /metrics, /example_seeds"
    )
