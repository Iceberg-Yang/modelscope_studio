"""不读取模型权重的 cuDNN 复现；仅在新子进程中调整动态库查找顺序。"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path


PACKAGES = ("torch", "diffusers", "peft", "torchao", "transformer-engine", "cudnn", "nvidia-cudnn-cu12")
LOADER_ROOT = Path("/tmp/lingbot-validation/cudnn-loaders")
CUDNN_LIBRARY = re.compile(r"(libcudnn(?:_adv|_cnn|_ops|_graph|_heuristic|_engines_precompiled|_engines_runtime_compiled)?\.so)(?:\.(\d+)(?:\.\d+)*)?")


def emit(stage, **details):
    print("LINGBOT_VALIDATION_STAGE=" + json.dumps({"stage": stage, **details}, ensure_ascii=False), flush=True)


def loaded_cudnn():
    path = Path("/proc/self/maps")
    if not path.is_file():
        return []
    return sorted({line.split()[-1] for line in path.read_text().splitlines() if "/" in line and "libcudnn" in line})


def wheel_directory():
    distribution = importlib.metadata.distribution("nvidia-cudnn-cu12")
    directory = Path(distribution.locate_file("nvidia/cudnn/lib")).resolve()
    if not (directory / "libcudnn.so.9").is_file() or not (directory / "libcudnn_graph.so.9").is_file():
        raise RuntimeError("Torch 配套 cuDNN 目录不完整，停止试验")
    return directory


def preload_entries(value):
    return [entry for entry in re.split(r"[:\s]+", value) if entry]


def wheel_library(directory, filename):
    directory = Path(directory).resolve()
    library = directory / filename
    if not library.is_file() or not library.resolve().is_relative_to(directory):
        raise RuntimeError("cuDNN 库不存在或指向配套目录之外: " + filename)
    if re.search(r"[:\s]", str(library)):
        raise ValueError("cuDNN 库路径不能包含预加载分隔符")
    return str(library)


def unified_preload(previous, directory):
    # 只替换明确的 cuDNN 9 条目，xGPU 等其他预加载库保持原有相对顺序。
    result = [wheel_library(directory, "libcudnn.so.9")]
    for entry in preload_entries(previous):
        match = CUDNN_LIBRARY.fullmatch(Path(entry).name)
        if match:
            if match.group(2) not in (None, "9"):
                raise RuntimeError("预加载包含非 cuDNN 9 库，停止自动替换")
            replacement = wheel_library(directory, match.group(1) + ".9")
            if replacement not in result:
                result.append(replacement)
        else:
            result.append(entry)
    return " ".join(result)


def loader_aliases(directory):
    # 仅补齐 cuDNN 9 的无版本号名称；不把其他主版本伪装为兼容库。
    directory = Path(directory).resolve()
    names = {path.name for path in directory.glob("libcudnn*.so.9")
             if CUDNN_LIBRARY.fullmatch(path.name)}
    if not {"libcudnn.so.9", "libcudnn_graph.so.9"}.issubset(names):
        raise RuntimeError("cuDNN 配套目录缺少核心库")
    aliases = {}
    for name in sorted(names):
        target = Path(wheel_library(directory, name))
        aliases[name] = target
        aliases[name[:-2]] = target
    return aliases


def validate_loader_directory(loader, directory):
    loader = Path(loader)
    if not loader.is_absolute() or re.search(r"[:\s]", str(loader)):
        raise ValueError("cuDNN 私有链接目录必须是无分隔符的绝对路径")
    if loader.is_symlink() or not loader.is_dir() or loader.resolve() != loader:
        raise RuntimeError("cuDNN 私有链接目录无效")
    expected = loader_aliases(directory)
    if {path.name for path in loader.iterdir()} != set(expected):
        raise RuntimeError("cuDNN 私有链接目录包含缺失或额外文件")
    for name, target in expected.items():
        path = loader / name
        if not path.is_symlink() or not path.is_file() or path.resolve() != target.resolve():
            raise RuntimeError("cuDNN 私有链接目标不匹配: " + name)
    return loader


def loader_directory_for(output):
    # 按任务输出目录隔离，链接留在容器本地，避免依赖持久化文件系统的链接语义。
    key = hashlib.sha256(str(Path(output).resolve()).encode()).hexdigest()[:32]
    return LOADER_ROOT.resolve() / key


def prepare_loader_directory(loader, directory):
    loader = Path(loader)
    if not loader.is_absolute() or loader.is_symlink() or loader.resolve() != loader or re.search(r"[:\s]", str(loader)):
        raise ValueError("拒绝通过相对路径、分隔符或符号链接创建 cuDNN 链接目录")
    expected = loader_aliases(directory)
    loader.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    loader.mkdir(mode=0o700, exist_ok=True)
    for name, target in expected.items():
        path = loader / name
        if not path.exists() and not path.is_symlink():
            path.symlink_to(target)
    return validate_loader_directory(loader, directory)


def native_cudnn_loads(stderr):
    # LD_DEBUG=files 能覆盖 C/C++ 的 dlopen；只提取 cuDNN 请求及加载方。
    events = []
    pattern = r"file=(\S+)\s+\[\d+\];\s+(dynamically loaded by|needed by)\s+(\S+)"
    for line in stderr.splitlines():
        match = re.search(pattern, line)
        if not match or not CUDNN_LIBRARY.fullmatch(Path(match[1]).name):
            continue
        event = {"library": match[1][:200], "relation": match[2], "caller": match[3][:200]}
        if event not in events:
            events.append(event)
        if len(events) >= 24:
            break
    return events


def loader_diagnostics(base=None, system_path=Path("/etc/ld.so.preload")):
    env = os.environ if base is None else base
    entries = preload_entries(env.get("LD_PRELOAD", ""))
    system_entries = []
    system_error = None
    try:
        if system_path.is_file():
            lines = system_path.read_text().splitlines()
            system_entries = preload_entries(" ".join(line.split("#", 1)[0] for line in lines))
    except OSError as error:
        system_error = type(error).__name__
    # 不输出完整环境或其他平台库路径，只记录 cuDNN 相关预加载证据。
    return {
        "env_cudnn": [entry for entry in entries if CUDNN_LIBRARY.fullmatch(Path(entry).name)],
        "other_preload_count": sum(not CUDNN_LIBRARY.fullmatch(Path(entry).name) for entry in entries),
        "system_cudnn": [entry for entry in system_entries if CUDNN_LIBRARY.fullmatch(Path(entry).name)],
        "system_read_error": system_error,
    }


def trace_cudnn_dlopen(events):
    # 只记录 Python 显式加载 cuDNN 的目标和调用位置，不拦截或改写加载行为。
    def audit(event, args):
        if event != "ctypes.dlopen" or not args or len(events) >= 16:
            return
        library = args[0]
        if not isinstance(library, str) or not CUDNN_LIBRARY.fullmatch(Path(library).name):
            return
        frames = traceback.extract_stack(limit=6)[:-1]
        events.append({"library": library, "callers": [
            "/".join(Path(frame.filename).parts[-2:]) + ":" + frame.name
            for frame in frames[-3:]
        ]})
    sys.addaudithook(audit)


def source_check(paths, directory):
    directory = Path(directory).resolve()
    unexpected = [path for path in paths if not Path(path).resolve().is_relative_to(directory)]
    return {"passed": bool(paths) and not unexpected, "unexpected": unexpected, "library_count": len(paths)}


def require_single_source(phase):
    check = source_check(loaded_cudnn(), wheel_directory())
    emit("cudnn_model_source", phase=phase, **check)
    if not check["passed"]:
        raise RuntimeError("模型进程 cuDNN 来源不一致或未加载，停止推理: " + phase)
    return check


def child_environment(mode, directory=None, base=None, loader_directory=None):
    env = dict(os.environ if base is None else base)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if mode == "wheel":
        if directory is None or not Path(directory).is_absolute():
            raise ValueError("cuDNN 目录必须是绝对路径")
        previous = env.get("LD_LIBRARY_PATH", "")
        directory = Path(directory).resolve()
        search = [str(directory)]
        if loader_directory is not None:
            loader = validate_loader_directory(loader_directory, directory)
            search.insert(0, str(loader))
        env["LD_LIBRARY_PATH"] = ":".join(search) + (":" + previous if previous else "")
        env["LD_PRELOAD"] = unified_preload(env.get("LD_PRELOAD", ""), directory)
    elif mode != "baseline":
        raise ValueError("未知动态库测试模式")
    return env


def attempt_passed(attempt):
    checks = attempt.get("checks", {})
    return attempt.get("status") == "passed" and all(
        checks.get(name, {}).get("passed") is True
        for name in ("diffusers_to", "cudnn_conv3d", "single_cudnn_source")
    )


def select_mode(attempts):
    # 优先采用库来源一致且通过算子验证的环境；失败时不能继续加载大模型。
    for mode in ("wheel", "baseline"):
        if any(item.get("mode") == mode and attempt_passed(item) for item in attempts):
            return mode
    raise RuntimeError("cuDNN 最小复现均未通过；保留报告并停止模型加载")


def probe(mode):
    result = {"mode": mode, "status": "failed", "checks": {}, "packages": {}, "loader": loader_diagnostics(), "library_snapshots": {}, "ctypes_cudnn_loads": []}
    trace_cudnn_dlopen(result["ctypes_cudnn_loads"])

    def snapshot(phase):
        paths = loaded_cudnn()
        result["library_snapshots"][phase] = paths
        return paths

    started = time.perf_counter()
    for name in PACKAGES:
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    result["loaded_before_torch"] = snapshot("before_torch")
    try:
        directory = wheel_directory()
        result["wheel_library_dir"] = str(directory)
        result["library_candidates"] = {
            str(folder): sorted(str(path) for path in folder.glob("libcudnn*.so*"))
            for folder in (directory, Path("/usr/lib/x86_64-linux-gnu"))
        }
        import torch
        result["loaded_after_torch"] = snapshot("after_torch")
        from diffusers import ModelMixin
        snapshot("after_diffusers")

        result["torch_cuda"] = torch.version.cuda
        if torch.__version__.split("+")[0] != "2.10.0" or torch.version.cuda != "12.8":
            raise RuntimeError("平台 Torch/CUDA 已改变，停止试验")

        class TinyModel(ModelMixin):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv3d(2, 2, 3, padding=1)

        # 触发上一轮失败的 ModelMixin.to -> hooks 导入链，不加载 LingBot。
        try:
            tiny = TinyModel().eval().requires_grad_(False)
            snapshot("after_model_cpu_init")
            tiny.to(device="cpu", dtype=torch.float32)
            snapshot("after_diffusers_cpu_to")
            torch.cuda.init()
            snapshot("after_cuda_init")
            tiny.to(device="cuda", dtype=torch.float32)
            snapshot("after_cuda_transfer")
            torch.cuda.synchronize()
            result["checks"]["diffusers_to"] = {"passed": True}
            del tiny
        except Exception as error:
            result["checks"]["diffusers_to"] = {"passed": False, "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}

        snapshot("after_diffusers_to")

        # 即使 hooks 导入失败，也单独验证 cuDNN 卷积，区分导入与计算故障。
        try:
            if not torch.backends.cudnn.is_available() or not torch.backends.cudnn.enabled:
                raise RuntimeError("cuDNN 未启用")
            result["cudnn_version"] = torch.backends.cudnn.version()
            with torch.inference_mode():
                value = torch.randn(1, 2, 3, 8, 8, device="cuda")
                weight = torch.randn(2, 2, 3, 3, 3, device="cuda")
                # 显式调用 cuDNN 算子，防止普通 Conv3d 静默回退后误判通过。
                output = torch.ops.aten.cudnn_convolution.default(value, weight, [1, 1, 1], [1, 1, 1], [1, 1, 1], 1, False, True, False)
                reference = torch.nn.functional.conv3d(value.cpu(), weight.cpu(), padding=1).cuda()
                torch.cuda.synchronize()
                if not torch.isfinite(output).all() or not torch.allclose(output, reference, atol=0.001, rtol=0.001):
                    raise RuntimeError("cuDNN 卷积有限值或 CPU 对照失败")
                result["checks"]["cudnn_conv3d"] = {"passed": True, "max_abs_diff": (output - reference).abs().max().item()}
        except Exception as error:
            result["checks"]["cudnn_conv3d"] = {"passed": False, "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
        result["loaded_after_checks"] = snapshot("after_checks")
        result["checks"]["single_cudnn_source"] = source_check(result["loaded_after_checks"], directory)
        if all(check["passed"] for check in result["checks"].values()):
            result["status"] = "passed"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    snapshot("probe_exit")
    result["seconds"] = time.perf_counter() - started
    return result


def save_report(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def emit_attempt_summary(mode, attempt, inherited, directory):
    first_foreign = None
    for phase, paths in attempt.get("library_snapshots", {}).items():
        if source_check(paths, directory)["unexpected"]:
            first_foreign = phase
            break
    foreign_loads = [item for item in attempt.get("ctypes_cudnn_loads", [])
                     if Path(item["library"]).is_absolute() and source_check([item["library"]], directory)["unexpected"]]
    first_load = foreign_loads[0] if foreign_loads else None
    # 将结论放在长堆栈之前，降低平台合并日志截断时丢失关键证据的概率。
    native = attempt.get("native_cudnn_loads", [])
    direct = [item for item in native if item["relation"] == "dynamically loaded by"]
    suspect = [item for item in direct if item["library"].endswith(".so") or (
        Path(item["library"]).is_absolute() and source_check([item["library"]], directory)["unexpected"])]
    emit("cudnn_preflight_result", mode=mode, strategy="wheel_alias_v2" if mode == "wheel" else "inherited",
         status=attempt["status"], cudnn_version=attempt.get("cudnn_version"),
         native_dlopen=(suspect or direct or [None])[0],
         checks={name: details.get("passed") for name, details in attempt.get("checks", {}).items()},
         first_foreign_phase=first_foreign,
         first_foreign_dlopen=None if first_load is None else {
             "library": first_load["library"][:128], "callers": [item[:96] for item in first_load["callers"][-2:]]},
         inherited_cudnn_preload=[entry[:128] for entry in inherited["env_cudnn"][:2]],
         system_cudnn_preload=[entry[:128] for entry in inherited["system_cudnn"][:2]],
         other_preload_count=inherited["other_preload_count"], system_read_error=inherited["system_read_error"],
         error=str(attempt.get("error") or "")[:160])


def run_preflight(output):
    result = {"run_id": os.environ.get("LINGBOT_VALIDATION_RUN_ID"), "status": "running", "attempts": []}
    save_report(output, result)
    try:
        directory = wheel_directory()
        result["wheel_library_dir"] = str(directory)
        result["inherited_loader"] = loader_diagnostics()
        result["strategy"] = "wheel_alias_v2"
        loader = prepare_loader_directory(loader_directory_for(output.parent), directory)
        result["loader_dir"] = str(loader)
        for mode in ("baseline", "wheel"):
            emit("cudnn_preflight_start", mode=mode)
            report_path = output.with_name("cudnn-" + mode + ".json")
            try:
                env = child_environment(mode, directory, loader_directory=loader if mode == "wheel" else None)
                # 原生链接跟踪只用于小型预检，不传给实际模型进程。
                env["LD_DEBUG"] = "files"
                completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--mode", mode, "--output", str(report_path)], env=env, capture_output=True, text=True, timeout=180)
                attempt = json.loads(report_path.read_text()) if report_path.is_file() else {"status": "failed", "error": "测试子进程未生成报告"}
                if attempt.get("run_id") != result["run_id"]:
                    attempt = {"status": "failed", "error": "拒绝使用上一轮报告"}
                if completed.returncode:
                    attempt["status"] = "failed"
                if attempt.get("status") == "passed" and not attempt_passed(attempt):
                    attempt.update(status="failed", error="预检缺少通过的算子或单一来源检查")
                attempt.update(mode=mode, returncode=completed.returncode, stderr=completed.stderr[-12000:],
                               native_cudnn_loads=native_cudnn_loads(completed.stderr))
            except subprocess.TimeoutExpired:
                attempt = {"mode": mode, "status": "failed", "error": "最小复现超过180秒"}
            except (OSError, ValueError, RuntimeError) as error:
                attempt = {"mode": mode, "status": "failed", "error": f"{type(error).__name__}: {error}"}
            result["attempts"].append(attempt)
            save_report(output, result)
            emit_attempt_summary(mode, attempt, result["inherited_loader"], directory)
            for event in attempt.get("native_cudnn_loads", []):
                emit("cudnn_native_load", mode=mode, **event)
            # 每行分开打印，避免平台将长异常栈截断为一个日志事件。
            for check, details in attempt.get("checks", {}).items():
                emit("cudnn_check", mode=mode, check=check, passed=details["passed"], error=details.get("error"))
                for line in details.get("traceback", "").splitlines():
                    emit("cudnn_trace", mode=mode, line=line)
            for path in attempt.get("loaded_after_checks", []):
                emit("cudnn_loaded_library", mode=mode, path=path)
        result["selected_mode"] = select_mode(result["attempts"])
        result["status"] = "passed"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    save_report(output, result)
    emit("cudnn_preflight_finished", status=result["status"], selected_mode=result.get("selected_mode"), error=result.get("error"))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "wheel"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode:
        report = probe(args.mode)
        report["run_id"] = os.environ.get("LINGBOT_VALIDATION_RUN_ID")
        save_report(args.output, report)
    else:
        report = run_preflight(args.output)
    raise SystemExit(0 if report["status"] == "passed" else 1)
