"""在独立进程准备固定版本实验；不安装 SGLang，不修改平台 Torch/CUDA。"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
import venv
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
ROOT = Path("/mnt/workspace/lingbot-validation/native-v1")
RUNTIME_ROOT = Path("/tmp/lingbot-validation/native-v1")
PIP_INDEX_URL = "https://mirrors.aliyun.com/pypi/simple/"


def safe_relative(path):
    value = PurePosixPath(path)
    if not path or value.is_absolute() or ".." in value.parts or "\\" in path:
        raise ValueError("不安全的组件路径")
    return Path(*value.parts)

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def emit(stage, **details):
    print("LINGBOT_VALIDATION_STAGE=" + json.dumps({"stage": stage, **details}, ensure_ascii=False), flush=True)


def download(component, item, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    emit("check_cache", file=item["path"])
    if destination.is_file() and destination.stat().st_size == item["size"] and sha256(destination) == item["sha256"]:
        emit("cache_hit", file=item["path"])
        return
    query = urllib.parse.urlencode({"Revision": component["revision"], "FilePath": item["path"]})
    url = f"https://modelscope.cn/api/v1/models/{component['repo']}/repo?{query}"
    partial = destination.with_name(destination.name + ".partial")
    for attempt in range(3):
        try:
            emit("download", file=item["path"], bytes=item["size"], attempt=attempt + 1)
            digest = hashlib.sha256()
            total = 0
            started = last_progress = time.monotonic()
            with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as stream:
                while block := response.read(8 * 1024 * 1024):
                    total += len(block)
                    if total > item["size"]:
                        raise ValueError("下载超过声明大小")
                    digest.update(block)
                    stream.write(block)
                    now = time.monotonic()
                    if now - last_progress >= 5 or total == item["size"]:
                        emit("download_progress", file=item["path"], bytes=total, total_bytes=item["size"], mib_per_second=round(total / 1024**2 / max(now - started, 0.001), 2))
                        last_progress = now
            if total != item["size"] or digest.hexdigest() != item["sha256"]:
                raise ValueError("下载长度或 SHA256 不匹配")
            partial.replace(destination)
            return
        except Exception as error:
            emit("download_error", file=item["path"], attempt=attempt + 1, error=f"{type(error).__name__}: {error}")
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def process_snapshot(pid):
    result = {"pid": pid}
    for name in ("status", "wchan", "io"):
        try:
            text = (Path("/proc") / str(pid) / name).read_text()
            if name == "status":
                text = "\n".join(line for line in text.splitlines() if line.startswith(("State:", "VmRSS:", "Threads:")))
            result[name] = text.strip()[:1000]
        except OSError:
            pass
    return result


def run(argv, **kwargs):
    timeout = kwargs.pop("timeout")
    command = [str(value) for value in argv]
    started = time.monotonic()
    # 保留命令自己的输出，同时记录等待时间和进程状态；网络无响应不再只剩一行下载日志。
    with subprocess.Popen(command, **kwargs) as process:
        while True:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                emit("command_timeout", command=command, seconds=timeout, process=process_snapshot(process.pid))
                process.kill()
                process.wait()
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                returncode = process.wait(timeout=min(10, remaining))
                if returncode:
                    raise subprocess.CalledProcessError(returncode, command)
                return
            except subprocess.TimeoutExpired:
                emit("command_wait", command=command, elapsed_seconds=round(time.monotonic() - started, 1), process=process_snapshot(process.pid))


def prepare_environment():
    # venv 名称包含依赖清单摘要，避免复用旧环境；只继承平台二进制包。
    digest = hashlib.sha256((HERE / "requirements-worker.txt").read_bytes() + (HERE / "constraints-worker.txt").read_bytes()).hexdigest()[:12]
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    environment = RUNTIME_ROOT / ("venv-" + digest)
    python = environment / "bin/python"
    marker = environment / "ready.json"
    if not marker.exists():
        emit("prepare_venv", directory=str(environment), persistent_cache=str(ROOT / "pip-cache"))
        venv.EnvBuilder(system_site_packages=True, with_pip=True).create(environment)
        report = environment / "install-report.json"
        # 先审查解析结果，禁止任何 Torch / CUDA 依赖被隐式拉入。
        args = [python, "-m", "pip", "install", "--index-url", PIP_INDEX_URL, "--disable-pip-version-check", "--progress-bar", "on", "--timeout", "30", "--retries", "2", "--only-binary=:all:", "--no-compile", "--cache-dir", ROOT / "pip-cache", "-r", HERE / "requirements-worker.txt", "-c", HERE / "constraints-worker.txt"]
        emit("resolve_dependencies", timeout_seconds=1800, index_url=PIP_INDEX_URL)
        run(args + ["--dry-run", "--report", report], timeout=1800)
        plan = json.loads(report.read_text())
        for package in plan["install"]:
            name = package["metadata"]["name"].lower().replace("_", "-")
            if name.startswith(("torch", "nvidia-", "cuda-", "flash-attn", "triton")):
                raise RuntimeError("拒绝替换平台 GPU 包: " + name)
        emit("install_dependencies", timeout_seconds=1800, index_url=PIP_INDEX_URL)
        run(args, timeout=1800)
        marker.write_text(json.dumps({"requirements_digest": digest}))
    emit("environment_ready")
    return python


def prepare_source(source, private_root=None):
    """只使用部署附带的固定源码包；失败即停止，不回退到网络下载。"""
    revision = source["revision"]
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("源码提交格式无效")
    bundle = source["bundle"]
    archive = HERE / safe_relative(bundle["path"])
    if archive.is_symlink() or archive.resolve().parent != HERE.resolve() or not archive.is_file():
        raise RuntimeError("缺少部署附带的源码包或路径不安全")
    if archive.stat().st_size != bundle["size"] or sha256(archive) != bundle["sha256"]:
        raise RuntimeError("源码包长度或 SHA256 不匹配")
    with tarfile.open(archive, "r:") as packaged:
        if packaged.pax_headers.get("comment") != revision:
            raise RuntimeError("源码包 Git 提交不匹配")
        members = packaged.getmembers()
        names, files, total = set(), set(), 0
        for member in members:
            relative = safe_relative(member.name)
            name = relative.as_posix()
            if name in names or member.name.rstrip("/") != name:
                raise ValueError("源码包包含重复或非规范路径")
            names.add(name)
            if not (member.isdir() or member.isfile()) or member.size < 0:
                raise ValueError("源码包禁止链接和特殊文件")
            space_bundle = source.get("revision") == "792aa925d91127ba4d78e0752adfc038605369c9"
            allowed = (name in ("LICENSE.txt", "README.md", "wan", "examples", "examples/00")
                       or name.startswith(("wan/", "examples/00/")))
            if space_bundle:
                allowed = allowed or name == "camera.py" or bool(re.fullmatch(r"examples/0[0-5](?:/(?:image\.jpg|prompt\.txt|poses\.npy|intrinsics\.npy))?", name))
            if not allowed or any(part.startswith(".") for part in relative.parts):
                raise ValueError("源码包包含范围外文件")
            total += member.size
            if member.isfile():
                files.add(name)
        if len(members) > 512 or total > 16 * 1024 * 1024:
            raise ValueError("源码包超过安全大小限制")
        required = {"LICENSE.txt", "README.md", "wan/__init__.py", "wan/image2video.py",
                    "wan/modules/model_fast.py", "examples/00/image.jpg", "examples/00/prompt.txt",
                    "examples/00/poses.npy", "examples/00/intrinsics.npy"}
        if not required.issubset(files):
            raise RuntimeError("源码包缺少推理、示例或许可文件")
        if private_root is None:
            RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        # 每次从已校验的包重新展开到私有目录，不执行上次任务留下的可修改源码。
        target = Path(tempfile.mkdtemp(prefix="source-" + revision + "-", dir=private_root or RUNTIME_ROOT))
        for member in members:
            destination = target / safe_relative(member.name)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with packaged.extractfile(member) as src, destination.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
    emit("source_bundle_ready", revision=revision, sha256=bundle["sha256"], files=len(files), network=False)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-experimental-config", action="store_true", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repetitions", type=int, choices=(1, 2), default=2)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--interactive", action="store_true")
    mode_group.add_argument("--space-request", action="store_true")
    parser.add_argument("--interactive-chunks", type=int, choices=(2, 8, 32), default=32, help="仅维护者CLI使用；页面固定32片段上限")
    args = parser.parse_args()
    contract = json.loads((HERE / "validation_contract.json").read_text())
    ROOT.mkdir(parents=True, exist_ok=True)
    output = args.output_dir or ROOT
    output.mkdir(parents=True, exist_ok=True)
    if args.space_request:
        from space_adapter import read_request
        read_request(output)
    python = prepare_environment()
    from cudnn_probe import child_environment, loader_directory_for, select_mode
    preflight_path = output / "cudnn-probe.json"
    run([python, HERE / "cudnn_probe.py", "--output", preflight_path], timeout=420)
    preflight = json.loads(preflight_path.read_text())
    if preflight.get("status") != "passed" or preflight.get("run_id") != os.environ.get("LINGBOT_VALIDATION_RUN_ID"):
        raise RuntimeError("cuDNN 预检报告无效，停止模型加载")
    mode = select_mode(preflight["attempts"])
    loader = loader_directory_for(output)
    if mode == "wheel" and preflight.get("loader_dir") != str(loader):
        raise RuntimeError("cuDNN 预检与模型进程的私有链接目录不一致")
    model_env = child_environment(mode, preflight["wheel_library_dir"], loader_directory=loader if mode == "wheel" else None)
    if args.space_request:
        runtime_report = output / "studio-runtime-probe.json"
        run([python, HERE / "studio_runtime_probe.py", "--test-write", "--output", runtime_report], env=model_env, timeout=90)
        runtime = json.loads(runtime_report.read_text())
        emit("studio_runtime_probe", facts=runtime["facts"], summary=runtime["summary"])
        if not runtime["summary"]["torch_cuda_available"] or runtime["facts"]["storage"]["persistent_workspace"]["write_test"] != "passed":
            raise RuntimeError("目标GPU或持久化目录验证失败")
    source_contract = json.loads((HERE / "space_source.json").read_text()) if args.space_request else contract["source"]
    source = prepare_source(source_contract, private_root=output if args.interactive or args.space_request else None)
    if not args.space_request:
        sample = source / "examples" / contract["smoke_workload"]["example"]
        shutil.copyfile(sample / "image.jpg", output / "input.jpg")
    emit("source_ready", revision=source_contract["revision"])
    # 源码不写入字节码，防止下一次缓存检查被 __pycache__ 干扰。
    model_env["PYTHONDONTWRITEBYTECODE"] = "1"
    for role in ("primary", "auxiliary"):
        component = contract[role]
        directory = ROOT / role / component["revision"]
        for item in component["files"]:
            download(component, item, directory / safe_relative(item["path"]))
    emit("model_probe_start", backend=contract["backend"])
    model_env["HF_HUB_OFFLINE"] = "1"
    model_env["TRANSFORMERS_OFFLINE"] = "1"
    if args.interactive:
        run([python, HERE / "interactive_worker.py", "--root", ROOT, "--source", source, "--output-dir", output, "--chunks", args.interactive_chunks, "--accept-experimental-config"], env=model_env, timeout=7200)
    else:
        options = ["--space-request"] if args.space_request else []
        run([python, HERE / "model_probe.py", "--root", ROOT, "--source", source, "--output-dir", output, "--repetitions", args.repetitions, "--accept-experimental-config", *options], env=model_env, timeout=1800 if args.space_request else 1200)
    emit("model_probe_finished")


if __name__ == "__main__":
    main()
