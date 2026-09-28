"""LingBot 验证入口：环境探测及需显式开启的原生推理实验。"""
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import gradio as gr

from interactive_session import SessionManager
from space_adapter import CAMERA_CHOICES, CAMERA_LABELS, PUBLIC_MAX_FRAMES, load_examples, normalize_image, validate_parameters

REPORT = {"status": "not_started", "scope": "点击生成后检测环境"}
LOCK = threading.Lock()
RUN_LOCK = threading.Lock()
MODEL_STATE = {"status": "idle", "scope": "单段 MP4 推理实验；实时交互尚未验证"}
MODEL_ROOT = Path("/mnt/workspace/lingbot-validation/native-v1")
OUTPUT_ROOT = MODEL_ROOT / "runs"
LOG_ROOT = Path("/tmp/lingbot-validation/logs")
MEDIA_ROOT = Path(os.environ.get("LINGBOT_PUBLIC_MEDIA_ROOT", "/tmp/lingbot-public-media"))
FILE_ACCESS = {
    "allowed_paths": [str(MEDIA_ROOT)],
    "blocked_paths": [str(MODEL_ROOT.parent), str(LOG_ROOT.parent)],
}
JOB_TIMEOUT = 7200
WORKER_ENV_KEYS = (
    "PATH", "LANG", "LC_ALL", "TZ", "TMPDIR",
    "LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH",
    "CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
)
PHASE_LABELS = {
    "runtime_check": "检测运行环境",
    "prepare_venv": "创建独立推理环境",
    "resolve_dependencies": "解析依赖",
    "install_dependencies": "安装依赖",
    "environment_ready": "检查 cuDNN 兼容性",
    "cudnn_preflight_start": "运行 cuDNN 预检",
    "cudnn_preflight_result": "记录 cuDNN 预检结果",
    "cudnn_preflight_finished": "cuDNN 预检结束",
    "source_ready": "官方示例准备完成",
    "check_cache": "校验权重缓存",
    "cache_hit": "复用已校验缓存",
    "download": "下载固定版本权重",
    "download_progress": "下载固定版本权重",
    "download_error": "下载重试",
    "model_probe_start": "开始模型验证",
    "load_pipeline": "加载模型",
    "move_model_to_cuda": "将模型载入 GPU",
    "load_t5": "加载文本编码器",
    "load_vae": "加载视频编解码器",
    "encode_and_verify_mp4": "编码并回读校验 MP4",
    "model_probe_finished": "推理验证结束",
}
SAMPLE_PROMPT = "The video presents a soaring journey through a fantasy jungle. The wind whips past the rider's blue hands gripping the reins, causing the leather straps to vibrate. The ancient gothic castle approaches steadily, its stone details becoming clearer against the backdrop of floating islands and distant waterfalls."


def worker_environment():
    """仅按名称读取运行所需变量，保留平台 GPU 注入及动态库设置。"""
    env = {}
    for key in WORKER_ENV_KEYS:
        value = os.environ.get(key)
        if value is not None:
            env[key] = value
    home = LOG_ROOT.parent / "worker-home"
    home.mkdir(parents=True, exist_ok=True)
    env.update(HOME=str(home), XDG_CACHE_HOME=str(home / ".cache"),
               PIP_CONFIG_FILE=os.devnull, PYTHONUNBUFFERED="1")
    return env


def command_result(argv, timeout=30):
    try:
        completed = subprocess.run(argv, env=worker_environment(), capture_output=True, text=True, timeout=timeout)
        return {"returncode": completed.returncode, "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-2000:]}
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


def collect_runtime():
    result = {
        "status": "complete",
        "scope": "仅环境检测，模型推理与实时交互尚未测试",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {},
    }
    for name in ("gradio", "torch", "torchvision", "diffusers", "transformers", "flash-attn", "sglang", "modelscope"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    result["nvidia_smi"] = command_result(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    result["nvcc"] = command_result(["nvcc", "--version"])
    workspace = Path("/mnt/workspace")
    result["persistent_workspace_exists"] = workspace.is_dir()
    if workspace.is_dir():
        result["disk_free_gib"] = round(shutil.disk_usage(workspace).free / 1024**3, 2)
    try:
        import torch
        result["torch_cuda"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()
        if result["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            result["gpu"] = {
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "visible_vram_gib": round(props.total_memory / 1024**3, 3),
            }
            torch.cuda.reset_peak_memory_stats()
            checks = []
            for iteration in range(2):
                start = time.perf_counter()
                with torch.inference_mode():
                    matrix = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
                    product = matrix @ matrix.T
                    query = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.bfloat16)
                    output = torch.nn.functional.scaled_dot_product_attention(query, query, query)
                    finite = bool(torch.isfinite(product).all() and torch.isfinite(output).all())
                    torch.cuda.synchronize()
                    checks.append({"iteration": iteration + 1, "finite": finite, "seconds": time.perf_counter() - start})
                    del matrix, product, query, output
            result["bf16_matmul_sdpa_checks"] = checks
            result["probe_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
            torch.cuda.empty_cache()
            child = "import json,torch; x=torch.ones(1,device='cuda'); print(json.dumps({'cuda':torch.cuda.is_available(),'device':torch.cuda.get_device_name(0),'value':x.item()}))"
            result["cuda_subprocess"] = command_result([sys.executable, "-c", child], timeout=120)
        if importlib.util.find_spec("flash_attn"):
            result["flash_attn_import"] = command_result([sys.executable, "-c", "from flash_attn import flash_attn_func; print('import_ok')"], timeout=60)
    except Exception as error:
        result["runtime_error"] = f"{type(error).__name__}: {error}"
        result["status"] = "error"
    if workspace.is_dir():
        try:
            target = workspace / "lingbot-validation" / "runtime-probe.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            result["persistent_report_written"] = True
        except OSError as error:
            result["persistent_report_error"] = str(error)
    with LOCK:
        REPORT.clear()
        REPORT.update(result)
    print("LINGBOT_RUNTIME_PROBE=" + safe_log(json.dumps(result, ensure_ascii=False)), flush=True)


def get_report():
    with LOCK:
        return dict(REPORT)


def job_directory(run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id):
        return None
    directory = OUTPUT_ROOT / run_id
    if directory.is_symlink() or directory.resolve().parent != OUTPUT_ROOT.resolve():
        return None
    return directory


def get_model_report(state=None):
    with LOCK:
        result = dict(MODEL_STATE if state is None else state)
    directory = job_directory(result.get("run_id"))
    if directory is None:
        return result
    for name, filename in (("probe", "model-probe.json"), ("cudnn_preflight", "cudnn-probe.json")):
        path = directory / filename
        if path.is_file() and not path.is_symlink():
            try:
                probe = json.loads(path.read_text())
                if isinstance(probe, dict) and probe.get("run_id") == result.get("run_id"):
                    result[name] = probe
            except (OSError, ValueError):
                pass
    return result


def video_result(report):
    directory = job_directory(report.get("run_id"))
    probe = report.get("probe", {})
    if directory is None or not isinstance(probe, dict) or probe.get("run_id") != report.get("run_id"):
        return None
    if report.get("status") != "finished" or probe.get("status") != "experimental_smoke_passed":
        return None
    sequences = probe.get("sequences", [])
    if not isinstance(sequences, list) or not sequences or not isinstance(sequences[0], dict):
        return None
    if sequences[0].get("mp4_verified") is not True:
        return None
    path = directory / "smoke-1.mp4"
    if path.is_file() and not path.is_symlink() and path.stat().st_size > 0 and path.resolve().parent == directory.resolve():
        if sequences[0].get("video") == str(path):
            return str(path)
    return None


def safe_log(text):
    """仅处理日志文本；不读取环境中的凭据值。"""
    text = re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1<redacted>@", text)
    text = re.sub(r"(?i)Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <redacted>", text)
    return re.sub(r"(?i)((?:token|secret|password|api_key|access_key|credential)[\"']?\s*[:=]\s*[\"']?)[^\s\"'&,}]+", r"\1<redacted>", text)


def public_phase(stage):
    if not isinstance(stage, str):
        return "准备环境"
    if stage in PHASE_LABELS:
        return PHASE_LABELS[stage]
    if re.fullmatch(r"read_weight_shard:model-\d{5}-of-\d{5}\.safetensors", stage):
        return "严格加载权重分片"
    match = re.fullmatch(r"dit_forward:([1-9][0-9]?|100)/([1-9][0-9]?|100)", stage)
    if match and int(match.group(1)) <= int(match.group(2)):
        return f"模型推理 {match.group(1)}/{match.group(2)}"
    if stage == "generate_sequence:1":
        return "生成短视频"
    if stage == "generate_sequence:2":
        return "同实例第二轮性能对照"
    return "准备环境"


def public_numbers(data, keys):
    if not isinstance(data, dict):
        return {}
    return {key: data[key] for key in keys if type(data.get(key)) in (int, float)
            and math.isfinite(data[key]) and data[key] >= 0}


PERFORMANCE_STAGES = (
    "t5_to_device", "t5_encode", "camera_condition_ops", "vae_encode", "self_kv_init", "cross_kv_init",
    "dit_denoise", "dit_cache_update", "flow_conversion", "scheduler_add_noise", "vae_decode",
    "dit_finite_check", "progress_report", "cudnn_source_check", "video_validation", "video_to_cpu",
    "video_statistics", "frames_to_uint8", "mp4_encode", "mp4_readback", "mp4_publish", "result_report",
)


def public_stage_timings(data, names=PERFORMANCE_STAGES):
    if not isinstance(data, dict):
        return {}
    return {name: public_numbers(data[name], ("calls", "wall_seconds", "cuda_stream_seconds"))
            for name in names if isinstance(data.get(name), dict)}


def public_performance(data):
    """重新构造性能摘要；不透传报告中的自由文本、路径或诊断对象。"""
    if not isinstance(data, dict):
        return {}
    result = public_numbers(data, ("generate_seconds", "generation_unattributed_seconds", "sequence_total_seconds", "profiled_frames_per_second"))
    if data.get("run_kind") in ("first_in_process", "repeat_same_instance"):
        result["run_kind"] = data["run_kind"]
    for key in ("t5_cache_hit_before", "kv_cache_reused_between_sequences"):
        if type(data.get(key)) is bool:
            result[key] = data[key]
    for group in ("generation_stages", "postprocess_stages"):
        result[group] = public_stage_timings(data.get(group))
    result["chunks"] = []
    chunks = data.get("chunks")
    for chunk in chunks[:20] if isinstance(chunks, list) else []:
        if not isinstance(chunk, dict) or type(chunk.get("chunk")) is not int or not 0 <= chunk["chunk"] < 20:
            continue
        safe = public_numbers(chunk, ("chunk", "forward_window_seconds"))
        safe["stages"] = public_stage_timings(chunk.get("stages"))
        result["chunks"].append(safe)
    return result


def public_report(report):
    """页面及生成接口只返回固定状态和有限数值。"""
    status = report.get("status")
    result = {
        "status": status if status in ("idle", "preparing", "finished", "failed", "cancelled") else "unknown",
        "stage": public_phase(report.get("stage")),
    }
    if job_directory(report.get("run_id")) is not None:
        result["run_id"] = report["run_id"]
    elapsed = report.get("elapsed_seconds")
    if type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0:
        result["elapsed_seconds"] = round(elapsed, 1)
    if video_result(report):
        result["mp4_verified"] = True
        sequence = report["probe"]["sequences"][0]
        for key in ("seconds", "peak_allocated_gib", "peak_reserved_gib", "full_task_peak_allocated_gib", "full_task_peak_reserved_gib", "host_peak_rss_gib", "decoded_frame_count", "file_bytes"):
            value = sequence.get(key)
            if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                result[key] = value
        probe = report["probe"]
        result.update(public_numbers(probe, ("load_seconds",)))
        result["load_stages"] = public_stage_timings(probe.get("load_stages"), ("load_main", "load_t5", "load_vae"))
        result["performance"] = [public_performance(item.get("performance")) for item in probe["sequences"][:2]
                                 if isinstance(item, dict) and item.get("mp4_verified") is True]
    return result


def publish_media(report, filename):
    """诊断目录不可下载；仅复制本轮合格媒体至独立公开目录。"""
    if filename not in ("input.jpg", "smoke-1.mp4"):
        return None
    directory = job_directory(report.get("run_id"))
    if directory is None or report.get("status") not in ("preparing", "finished"):
        return None
    if filename == "smoke-1.mp4" and video_result(report) is None:
        return None
    source = directory / filename
    if source.is_symlink() or not source.is_file() or source.resolve().parent != directory.resolve():
        return None
    target_dir = MEDIA_ROOT / report["run_id"]
    if target_dir.is_symlink() or target_dir.resolve().parent != MEDIA_ROOT.resolve():
        return None
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename
    if target.is_symlink():
        return None
    if not target.exists():
        temporary = target.with_name(filename + ".partial")
        if temporary.is_symlink():
            return None
        if filename == "input.jpg":
            from PIL import Image
            try:
                with Image.open(source) as image:
                    if image.format != "JPEG" or image.width * image.height > 4096 * 4096:
                        return None
                    image.convert("RGB").save(temporary, format="JPEG")
            except (OSError, ValueError, Image.DecompressionBombError):
                return None
        else:
            shutil.copyfile(source, temporary)
        temporary.replace(target)
    return str(target)


def stop_process_group(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # 主进程已退出也可能留下 pip 或推理孙进程，按本次独立进程组清理。
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def read_log_tail(path):
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 16000))
        return safe_log(stream.read().decode("utf-8", errors="replace").replace("\r", "\n"))


def relay_logs(path, offset):
    if not path.is_file():
        return offset
    with path.open("rb") as stream:
        stream.seek(offset)
        data = stream.read(65536)
        offset = stream.tell()
    if data:
        print(safe_log(data.decode("utf-8", errors="replace")), end="", flush=True)
    return offset


def generation_outputs(log_path, elapsed, state=None):
    report = get_model_report(state)
    log = read_log_tail(log_path)
    phase = report.get("stage", "runtime_check")
    for line in log.splitlines():
        for prefix in ("LINGBOT_VALIDATION_STAGE=", "LINGBOT_MODEL_PROBE="):
            if line.startswith(prefix):
                try:
                    event = json.loads(line[len(prefix):])
                    candidate = event.get("stage") if isinstance(event, dict) else None
                    if public_phase(candidate) != "准备环境":
                        phase = candidate
                except ValueError:
                    pass
    report["stage"] = phase
    report["elapsed_seconds"] = round(elapsed, 1)
    with LOCK:
        (MODEL_STATE if state is None else state)["stage"] = phase
    video = publish_media(report, "smoke-1.mp4")
    image = publish_media(report, "input.jpg")
    status = report.get("status")
    if status == "finished" and video is None:
        report["status"] = "failed"
        status = "failed"
        with LOCK:
            (MODEL_STATE if state is None else state).update(status="failed", error="本轮媒体未能发布")
    if status == "finished":
        summary = "已生成并校验 MP4，可播放或下载。请人工检查画质与镜头方向；当前仍为实验配置。"
    elif status in ("failed", "cancelled"):
        summary = "任务已取消。" if status == "cancelled" else "生成失败，请由空间维护者查看运行日志。"
        if report.get("error_code") == "timeout":
            summary = "本次生成超时，请由空间维护者查看运行日志。"
    else:
        summary = f"正在运行：{public_phase(phase)} · 已用 {elapsed:.0f} 秒。首次运行需要准备依赖和校验权重缓存。"
    return summary, video, video, public_report(report), image


def run_validation(accept_experiment, compare_cached=False):
    if type(compare_cached) is not bool:
        raise gr.Error("性能对照选项必须为布尔值。")
    if not accept_experiment or os.environ.get("LINGBOT_ACCEPT_EXPERIMENTAL_CONFIG") != "1":
        raise gr.Error("请先确认实验配置；空间也需要开启 LINGBOT_ACCEPT_EXPERIMENTAL_CONFIG。")
    if not RUN_LOCK.acquire(blocking=False):
        raise gr.Error("已有推理任务运行，请等待完成。")
    process = None
    offset = 0
    final_outputs = None
    started = time.monotonic()
    run_id = uuid.uuid4().hex
    log_path = LOG_ROOT / (run_id + ".log")
    try:
        directory = job_directory(run_id)
        directory.mkdir(parents=True, exist_ok=False)
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        with LOCK:
            MODEL_STATE.clear()
            MODEL_STATE.update(status="preparing", run_id=run_id, stage="runtime_check")
        yield generation_outputs(log_path, 0)
        collect_runtime()
        if get_report().get("status") == "error" or not get_report().get("cuda_available"):
            raise RuntimeError("GPU 环境检测失败，未开始模型加载")
        # 推理进程不需要创空间管理凭据，不继承令牌、密钥或密码。
        child_env = worker_environment()
        child_env.update(PYTHONUNBUFFERED="1", LINGBOT_VALIDATION_RUN_ID=run_id)
        argv = [sys.executable, str(Path(__file__).with_name("prepare_validation.py")), "--accept-experimental-config", "--output-dir", str(directory), "--repetitions", "2" if compare_cached else "1"]
        with log_path.open("w") as log:
            process = subprocess.Popen(argv, env=child_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed > JOB_TIMEOUT:
                    raise TimeoutError("本次任务超时，已终止本次进程组；请检查最后阶段日志")
                offset = relay_logs(log_path, offset)
                yield generation_outputs(log_path, elapsed)
                time.sleep(2)
            while offset < log_path.stat().st_size:
                offset = relay_logs(log_path, offset)
            with LOCK:
                MODEL_STATE.update(status="finished" if process.returncode == 0 else "failed", returncode=process.returncode)
            if process.returncode:
                report = get_model_report()
                detail = report.get("probe", {}).get("error") or report.get("cudnn_preflight", {}).get("error")
                raise RuntimeError(detail or f"验证进程退出码 {process.returncode}；请查看最后阶段日志")
            if video_result(get_model_report()) is None:
                raise RuntimeError("进程已退出，但没有本轮已校验的 MP4，不能标记推理成功")
    except Exception as error:
        with LOCK:
            MODEL_STATE.update(status="failed", error=safe_log(f"{type(error).__name__}: {error}"),
                               error_code="timeout" if isinstance(error, TimeoutError) else "generation_failed")
    finally:
        try:
            stop_process_group(process)
            with LOCK:
                if MODEL_STATE.get("status") == "preparing":
                    MODEL_STATE.update(status="cancelled", error="生成事件已结束，子进程已清理")
            final_outputs = generation_outputs(log_path, time.monotonic() - started)
            print("LINGBOT_VALIDATION_PROCESS=" + safe_log(json.dumps(get_model_report(), ensure_ascii=False)), flush=True)
        finally:
            RUN_LOCK.release()
    yield final_outputs


SESSIONS = SessionManager(RUN_LOCK, MEDIA_ROOT, worker_environment, stop_process_group, relay_logs, timeout=JOB_TIMEOUT)


def start_interactive(accepted, request: gr.Request):
    if type(accepted) is not bool or not accepted or os.environ.get("LINGBOT_ACCEPT_EXPERIMENTAL_CONFIG") != "1":
        raise gr.Error("请先确认实验配置；空间也需要开启 LINGBOT_ACCEPT_EXPERIMENTAL_CONFIG。")
    owner = request.session_hash
    stream = SESSIONS.run(owner)
    try:
        for update in stream:
            yield update["run_id"], json.dumps(update, ensure_ascii=False, allow_nan=False)
    except (ValueError, RuntimeError):
        raise gr.Error("无法开始会话：已有GPU任务运行或浏览器会话无效，请稍后重试。") from None
    finally:
        stream.close()


def interactive_control(run_id, request: gr.Request, event: gr.EventData):
    try:
        SESSIONS.event(request.session_hash, run_id, event._data)
    except (ValueError, TypeError, OSError):
        # 非归属、畸形及过期事件不回显客户端内容或私有诊断。
        return


def example_thumbnail():
    """只读取经哈希确认的内置示例图片，不展开或执行源码。"""
    import base64
    import hashlib
    import tarfile
    here = Path(__file__).parent
    bundle = json.loads((here / "validation_contract.json").read_text())["source"]["bundle"]
    archive = here / "source_bundle.tar"
    if archive.is_symlink() or archive.stat().st_size != bundle["size"] or hashlib.sha256(archive.read_bytes()).hexdigest() != bundle["sha256"]:
        raise RuntimeError("内置示例包校验失败")
    with tarfile.open(archive, "r:") as packaged:
        item = packaged.getmember("examples/00/image.jpg")
        if not item.isfile() or not 0 < item.size <= 2 * 1024 * 1024:
            raise RuntimeError("内置示例图片无效")
        return "data:image/jpeg;base64," + base64.b64encode(packaged.extractfile(item).read()).decode("ascii")


def cleanup_expired_jobs():
    """持锁执行；只清理本应用随机ID目录，保留最近8份且不超过一天。"""
    for root in (OUTPUT_ROOT, MEDIA_ROOT):
        if not root.is_dir():
            continue
        entries = [p for p in root.iterdir() if re.fullmatch(r"[0-9a-f]{32}", p.name) and p.is_dir() and not p.is_symlink()]
        entries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for index, path in enumerate(entries):
            if index >= 8 or time.time() - path.stat().st_mtime > 86400:
                shutil.rmtree(path)
    if LOG_ROOT.is_dir():
        for path in LOG_ROOT.iterdir():
            if re.fullmatch(r"[0-9a-f]{32}\.log", path.name) and not path.is_symlink() and time.time() - path.stat().st_mtime > 86400:
                path.unlink()


def run_space_job(job, directory, log_path, done):
    """监督线程独立拥有进程和GPU锁；页面断开不会使超时与清理失效。"""
    process, offset = None, 0
    started = time.monotonic()
    terminal = "failed"
    try:
        env = worker_environment()
        env.update(LINGBOT_VALIDATION_RUN_ID=job["run_id"], TOKENIZERS_PARALLELISM="false")
        argv = [sys.executable, str(Path(__file__).with_name("prepare_validation.py")),
                "--accept-experimental-config", "--space-request", "--output-dir", str(directory), "--repetitions", "1"]
        with log_path.open("w") as log:
            process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                if time.monotonic() - started > JOB_TIMEOUT:
                    raise TimeoutError("任务超时")
                offset = relay_logs(log_path, offset)
                time.sleep(2)
            while offset < log_path.stat().st_size:
                offset = relay_logs(log_path, offset)
        candidate = get_model_report({**job, "status": "finished"})
        if process.returncode or video_result(candidate) is None:
            raise RuntimeError("worker退出或未产生合格MP4")
        terminal = "finished"
    except Exception as error:
        with LOCK:
            job["error_code"] = "timeout" if isinstance(error, TimeoutError) else "generation_failed"
        print("LINGBOT_SPACE_ERROR=" + safe_log(str(error)), flush=True)
    finally:
        cleaned = False
        try:
            try:
                stop_process_group(process)
                cleaned = True
            except Exception as error:
                terminal = "failed"
                with LOCK:
                    job["error_code"] = "cleanup_failed"
                print("LINGBOT_SPACE_CLEANUP_ERROR=" + safe_log(str(error)), flush=True)
            if terminal == "finished":
                candidate = get_model_report({**job, "status": "finished"})
                if publish_media(candidate, "smoke-1.mp4") is None:
                    terminal = "failed"
            with LOCK:
                job.update(status=terminal, elapsed_seconds=time.monotonic() - started)
            report = public_report(get_model_report(job))
            print("LINGBOT_SPACE_RESULT=" + json.dumps(report, ensure_ascii=False), flush=True)
        except Exception as error:
            with LOCK:
                job.update(status="failed", error_code="generation_failed")
            print("LINGBOT_SPACE_ERROR=" + safe_log(str(error)), flush=True)
        finally:
            # 无法确认进程组退出时保留互斥锁，防止残留worker与新请求争用GPU。
            if cleaned:
                RUN_LOCK.release()
            done.set()


def run_space(image, prompt, camera_motion, num_frames, turn_rate, seed, randomize_seed):
    if os.environ.get("LINGBOT_ACCEPT_EXPERIMENTAL_CONFIG") != "1":
        raise gr.Error("空间尚未开启生成服务，请联系空间维护者。")
    try:
        parameters = validate_parameters(prompt, camera_motion, num_frames, turn_rate, seed, randomize_seed)
        image = normalize_image(image)
    except ValueError as error:
        raise gr.Error(str(error)) from None
    if not RUN_LOCK.acquire(blocking=False):
        raise gr.Error("已有生成任务运行，请稍后重试。")
    done = threading.Event()
    run_id = uuid.uuid4().hex
    job = {"status": "preparing", "run_id": run_id, "stage": "runtime_check"}
    started = time.monotonic()
    log_path = LOG_ROOT / (run_id + ".log")
    try:
        cleanup_expired_jobs()
        directory = job_directory(run_id)
        directory.mkdir(parents=True, exist_ok=False)
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        image.save(directory / "input.png", format="PNG")
        (directory / "request.json").write_text(json.dumps(parameters, ensure_ascii=False))
        thread = threading.Thread(target=run_space_job, args=(job, directory, log_path, done), daemon=True)
        thread.start()
    except Exception:
        RUN_LOCK.release()
        raise gr.Error("无法创建生成任务，请联系空间维护者。") from None
    while True:
        completed = done.is_set()
        outputs = generation_outputs(log_path, time.monotonic() - started, state=job)
        yield outputs[0], outputs[1], outputs[2], parameters["seed"]
        if completed:
            break
        time.sleep(2)


CSS = """
#col-container { max-width: 1180px; margin: 0 auto; }
#hero { padding: 18px 0 8px; }
#generate-video { min-height: 48px; }
.dark .gradio-container { color: var(--body-text-color); }
"""
HEADER = """
# LingBot-World V2 · 1.3B

[魔搭模型](https://modelscope.cn/models/Robbyant/lingbot-world-v2-1.3b-causal-fast) ·
[github](https://github.com/Robbyant/lingbot-world-v2)
"""

with gr.Blocks(title="LingBot-World V2 · 图生视频", analytics_enabled=False, delete_cache=(3600, 86400)) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(HEADER, elem_id="hero")
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="首帧图片", type="pil", height=280, sources=["upload", "clipboard"])
                prompt = gr.Textbox(label="Prompt · 场景描述", lines=3, max_lines=6, placeholder="描述场景与希望发生的变化，建议参考示例中的英文提示词…")
                camera_motion = gr.Dropdown(label="镜头轨迹", choices=list(zip(CAMERA_LABELS, CAMERA_CHOICES)), value="Dolly forward", info="15种合成运动 + 5种作者录制轨迹。")
                generate = gr.Button("生成视频", variant="primary", elem_id="generate-video")
            with gr.Column(scale=1):
                video = gr.Video(label="生成结果", height=380, autoplay=True, loop=True, interactive=False, format="mp4")
                status = gr.Textbox(label="当前状态", value="选择示例或上传图片后点击生成。首次运行需要准备环境与权重。", interactive=False, lines=2)
                download = gr.File(label="下载本轮 MP4", interactive=False)
        with gr.Accordion("高级设置", open=False):
            num_frames = gr.Slider(label="视频帧数 · 16 FPS播放", minimum=45, maximum=PUBLIC_MAX_FRAMES, step=12, value=81, info="默认81帧约5.06秒；237帧上限待独立验收。输出面积约480p，尺寸随输入比例对齐。")
            turn_rate = gr.Slider(label="合成轨迹转向速度（°/s）", minimum=1, maximum=30, step=0.5, value=8, info="仅影响带旋转的合成轨迹；不改变录制轨迹。")
            with gr.Row():
                seed = gr.Number(label="Seed", value=0, precision=0, minimum=0, maximum=2**31 - 1)
                randomize_seed = gr.Checkbox(label="随机 Seed", value=False)
                used_seed = gr.Number(label="本轮实际 Seed", interactive=False, precision=0)
        gr.Examples(examples=load_examples(MEDIA_ROOT), inputs=[input_image, prompt, camera_motion], cache_examples=False,
                    label="示例")
        generate.click(run_space, inputs=[input_image, prompt, camera_motion, num_frames, turn_rate, seed, randomize_seed],
                       outputs=[status, video, download, used_seed], api_name="generate", concurrency_limit=1,
                       concurrency_id="lingbot_gpu", trigger_mode="once")

if __name__ == "__main__":
    demo.queue(max_size=2, default_concurrency_limit=1)
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=False, theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple", font=["Arial", "sans-serif"], font_mono=["monospace"]),
                css=CSS, mcp_server=False, max_file_size="12mb", **FILE_ACCESS)
