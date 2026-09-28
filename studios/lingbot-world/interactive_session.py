"""CPU侧会话管理及私有邮箱协议；不导入Torch，不持有GPU管理凭据。"""
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from interactive_pipeline import MAX_CHUNKS, checked_keys

SESSION_ROOT = Path("/tmp/lingbot-validation/interactive")
STATES = {"idle", "preparing", "running", "stopping", "stopped", "finished", "failed"}
TERMINAL = {"stopped", "finished", "failed"}


def safe_directory(directory):
    directory = Path(directory)
    if not directory.is_absolute() or directory.is_symlink() or directory.resolve() != directory:
        raise ValueError("会话目录不安全")
    return directory


def atomic_json(path, value):
    safe_directory(path.parent)
    temporary = path.with_name(path.name + ".partial")
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError("拒绝链接文件")
    with temporary.open("w") as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
    temporary.replace(path)


def read_json(path, limit=65536):
    safe_directory(path.parent)
    if not path.is_file() or path.is_symlink():
        return {}
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("会话报告过大")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("会话报告类型错误")
    return value


def controls(directory, run_id, now=None):
    now = time.monotonic() if now is None else now
    data = read_json(directory / "control.json", 4096)
    if data.get("run_id") != run_id:
        raise ValueError("邮箱会话不匹配")
    stamp = data.get("updated")
    if type(stamp) not in (int, float) or not math.isfinite(stamp) or stamp > now:
        raise ValueError("心跳时间无效")
    age = now - stamp
    keys = checked_keys(data.get("keys"))
    played, seq = data.get("played"), data.get("seq")
    if type(played) is not int or not -1 <= played < MAX_CHUNKS or type(seq) is not int or not -1 <= seq < 2**53 or type(data.get("stop")) is not bool:
        raise ValueError("邮箱控制字段无效")
    return {"keys": keys if age <= 2 else [], "stop": data["stop"] or age > 20,
            "played": played, "seq": seq}


class SessionManager:
    def __init__(self, gpu_lock, media_root, environment, stop_process, relay, timeout=7200, root=SESSION_ROOT):
        self.gpu_lock, self.media_root, self.environment = gpu_lock, Path(media_root) / "interactive", environment
        self.stop_process, self.relay, self.timeout = stop_process, relay, timeout
        self.root = Path(root)
        self.lock = threading.RLock()
        self.active = None

    def create(self, owner):
        if not isinstance(owner, str) or not owner or len(owner) > 256:
            raise ValueError("缺少有效的浏览器会话")
        if not self.gpu_lock.acquire(blocking=False):
            raise RuntimeError("已有GPU任务运行，请稍后再试")
        try:
            with self.lock:
                self.cleanup_media()
                self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
                safe_directory(self.root)
                os.chmod(self.root, 0o700)
                self.cleanup_directories(self.root)
                run_id = uuid.uuid4().hex
                directory = safe_directory(self.root / run_id)
                directory.mkdir(parents=True, exist_ok=False, mode=0o700)
                now = time.monotonic()
                self.active = {"run_id": run_id, "owner": owner, "directory": directory, "created": now,
                               "updated": now, "accepted": 0.0, "seq": -1, "played": -1, "keys": [], "stop": False,
                               "stop_at": None, "published": -1, "state": "preparing", "ready_at": None, "pending_control": False}
                self.write_control()
                return run_id
        except BaseException:
            self.active = None
            self.gpu_lock.release()
            raise

    def write_control(self):
        data = self.active
        atomic_json(data["directory"] / "control.json", {k: data[k] for k in ("run_id", "updated", "seq", "played", "keys", "stop")})
        data["pending_control"] = False

    def event(self, owner, run_id, payload):
        if not isinstance(payload, dict) or set(payload) != {"run_id", "seq", "keys", "played", "stop"}:
            raise ValueError("控制事件格式无效")
        if not isinstance(payload["run_id"], str) or not re.fullmatch(r"[0-9a-f]{32}", payload["run_id"]):
            raise ValueError("控制会话ID无效")
        keys = checked_keys(payload["keys"])
        sequence, played = payload["seq"], payload["played"]
        if type(sequence) is not int or not 0 <= sequence < 2**53 or type(played) is not int or not -1 <= played < MAX_CHUNKS or type(payload["stop"]) is not bool:
            raise ValueError("控制序号无效")
        with self.lock:
            data = self.active
            if data is None or (data["owner"], data["run_id"]) != (owner, run_id) or payload["run_id"] != run_id:
                return False
            if sequence <= data["seq"] or played < data["played"] or played > data["published"]:
                return False
            now = time.monotonic()
            if data["state"] in TERMINAL:
                return False
            data.update(seq=sequence, played=played, keys=[] if data["stop"] else keys, updated=now, pending_control=True)
            if payload["stop"]:
                self.request_stop(now)
            # 网络抖动导致连续事件靠拢时只合并最新快照，不因限流丢失松键。
            if payload["stop"] or now - data["accepted"] >= 0.1:
                data["accepted"] = now
                self.write_control()
            return True

    def request_stop(self, now):
        data = self.active
        if data["stop_at"] is None:
            data["stop_at"] = now
        data.update(stop=True, keys=[], state="stopping")

    def cleanup_media(self):
        root = safe_directory(self.media_root)
        root.mkdir(parents=True, exist_ok=True)
        self.cleanup_directories(root)

    @staticmethod
    def cleanup_directories(root):
        directories = [p for p in root.iterdir() if re.fullmatch(r"[0-9a-f]{32}", p.name) and p.is_dir() and not p.is_symlink()]
        directories.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for index, directory in enumerate(directories):
            if index >= 3 or time.time() - directory.stat().st_mtime > 3600:
                # 仅删除本应用生成的有界媒体目录；不跟随符号链接。
                shutil.rmtree(directory)

    def publish(self, report):
        data = self.active
        segments = report.get("segments", [])
        if not isinstance(segments, list) or len(segments) > MAX_CHUNKS:
            raise ValueError("片段清单无效")
        if len(segments) - 1 < data["published"]:
            raise ValueError("片段报告回退")
        if len(segments) - (data["played"] + 1) > 2:
            raise RuntimeError("播放背压协议失效")
        safe_directory(data["directory"])
        directory = safe_directory(self.media_root / data["run_id"])
        directory.mkdir(parents=True, exist_ok=True)
        public = []
        for index, item in enumerate(segments):
            expected = 9 if index == 0 else 12
            if not isinstance(item, dict) or type(item.get("index")) is not int or item["index"] != index or item.get("verified") is not True or type(item.get("frames")) is not int or item["frames"] != expected:
                raise ValueError("片段校验标记无效")
            size = item.get("bytes")
            if type(size) is not int or not 0 < size <= 16 * 1024 * 1024:
                raise ValueError("片段大小无效")
            name = f"chunk-{index:03d}.mp4"
            source, target = data["directory"] / name, directory / name
            if source.is_symlink() or not source.is_file() or source.stat().st_size != size or target.is_symlink():
                raise ValueError("片段文件不安全")
            if not target.exists():
                partial = target.with_suffix(".partial")
                if partial.is_symlink():
                    raise ValueError("媒体暂存链接无效")
                shutil.copyfile(source, partial)
                partial.replace(target)
            if index > data["played"]:
                public.append({"index": index, "path": str(target), "frames": expected})
        data["published"] = len(segments) - 1
        return public

    def snapshot(self):
        with self.lock:
            data = self.active
            now = time.monotonic()
            if data["pending_control"] and now - data["accepted"] >= 0.1:
                data["accepted"] = now
                self.write_control()
            report = read_json(data["directory"] / "interactive.json")
            if report and report.get("run_id") != data["run_id"]:
                raise ValueError("worker报告会话不匹配")
            stage = report.get("stage")
            stage = stage if stage in ("loading", "equivalence", "conditioning", "chunk", "waiting", "done") else "loading"
            state = data["state"]
            if state != "stopping" and report.get("status") in STATES - {"idle"}:
                state = report["status"]
            data["state"] = state
            if state == "running" and data["ready_at"] is None:
                data["ready_at"] = time.monotonic()
            result = {"run_id": data["run_id"], "status": state, "stage": stage,
                      "pending_keys": list(data["keys"]) if time.monotonic() - data["updated"] <= 2 else [], "segments": self.publish(report),
                      "applied_keys": checked_keys(report.get("applied_keys", [])), "ack_seq": data["seq"]}
            for name in ("chunk_seconds", "peak_allocated_gib", "chunks_done"):
                value = report.get(name)
                if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                    result[name] = value
            return result, report

    def check_deadlines(self, report, now):
        """由监督线程执行，不依赖浏览器继续消费生成器。"""
        with self.lock:
            data = self.active
            if now - data["created"] > self.timeout:
                raise TimeoutError("准备及会话总超时")
            chunk_started = report.get("chunk_started")
            if report.get("stage") == "chunk" and type(chunk_started) in (int, float) and now - chunk_started > 120:
                raise TimeoutError("单片段生成超时")
            ready_at = report.get("ready_at", data["ready_at"])
            expired = type(ready_at) in (int, float) and now - ready_at >= 300
            if now - data["updated"] > 20 or expired:
                self.request_stop(now)
                self.write_control()
            return data["stop_at"] is not None and now - data["stop_at"] >= 15

    def supervise(self, run_id, box, done):
        process, offset = None, 0
        directory = self.active["directory"]
        result = {"run_id": run_id, "status": "failed", "segments": [], "stage": "done"}
        try:
            env = self.environment()
            env["LINGBOT_VALIDATION_RUN_ID"] = run_id
            argv = [sys.executable, str(Path(__file__).with_name("prepare_validation.py")), "--accept-experimental-config", "--interactive", "--output-dir", str(directory)]
            with (directory / "worker.log").open("w") as log:
                process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while process.poll() is None:
                    current, report = self.snapshot()
                    if self.check_deadlines(report, time.monotonic()):
                        self.stop_process(process)
                        break
                    offset = self.relay(directory / "worker.log", offset)
                    if current["status"] in TERMINAL:
                        current["status"] = "stopping"
                    with self.lock:
                        box["value"] = current
                    time.sleep(0.2)
                self.relay(directory / "worker.log", offset)
            current, report = self.snapshot()
            if self.active["stop"]:
                current["status"] = "stopped"
            elif process.returncode != 0 or report.get("status") not in ("finished", "stopped"):
                raise RuntimeError("交互worker未正常结束")
            else:
                current["status"] = report["status"]
            result = current
        except Exception as error:
            print("LINGBOT_INTERACTIVE_ERROR=" + type(error).__name__, flush=True)
            import traceback
            traceback.print_exc()
        finally:
            try:
                self.stop_process(process)
            finally:
                with self.lock:
                    box["value"] = result
                    self.active = None
                    self.gpu_lock.release()
                    done.set()

    def run(self, owner):
        run_id = self.create(owner)
        done = threading.Event()
        box = {"value": {"run_id": run_id, "status": "preparing", "stage": "loading", "segments": []}}
        supervisor = threading.Thread(target=self.supervise, args=(run_id, box, done), daemon=True)
        try:
            supervisor.start()
        except BaseException:
            with self.lock:
                self.active = None
                self.gpu_lock.release()
            raise
        try:
            while True:
                with self.lock:
                    current, finished = dict(box["value"]), done.is_set()
                yield current
                if finished:
                    return
                done.wait(0.5)
        finally:
            with self.lock:
                if self.active and self.active["run_id"] == run_id and not done.is_set():
                    self.request_stop(time.monotonic())
                    self.write_control()
            # 监督线程拥有进程组及锁，断连也继续完成清理；不阻塞控制事件。
            supervisor.join(timeout=0.1)
