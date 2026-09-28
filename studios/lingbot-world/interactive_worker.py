"""独立GPU会话；先做原生等价性对照，再读取实时邮箱续推。"""
import argparse
import gc
import json
import os
import time
import traceback
from pathlib import Path

from interactive_pipeline import InteractivePipeline, MAX_CHUNKS, verify_equivalence
from interactive_session import SESSION_ROOT, atomic_json, controls, safe_directory
from model_probe import build_pipeline, prepare_runtime, write_verified_mp4


def execute(args, report, save):
    import numpy as np
    import torch
    from PIL import Image
    from cudnn_probe import require_single_source
    torch.cuda.reset_peak_memory_stats()
    contract = prepare_runtime(args, report, save)
    pipe = build_pipeline(args.root / "primary" / contract["primary"]["revision"],
                          args.root / "auxiliary" / contract["auxiliary"]["revision"],
                          contract["model_config"], report, save)
    require_single_source("interactive_loaded")
    if controls(args.output_dir, report["run_id"])["stop"]:
        report.update(status="stopped", stage="done")
        save()
        return
    sample = args.source / "examples/00"
    with Image.open(sample / "image.jpg") as original:
        image = original.convert("RGB")
    prompt = (sample / "prompt.txt").read_text().strip()
    report.update(stage="equivalence")
    save()
    with torch.inference_mode():
        report["equivalence"] = verify_equivalence(pipe, image, prompt, sample)
    require_single_source("interactive_equivalence")
    if controls(args.output_dir, report["run_id"])["stop"]:
        report.update(status="stopped", stage="done")
        save()
        return
    report.update(stage="conditioning")
    save()
    with torch.inference_mode():
        session = InteractivePipeline(pipe, image, prompt, np.load(sample / "intrinsics.npy"), chunks=args.chunks)
    require_single_source("interactive_conditioned")
    started = time.monotonic()
    report["ready_at"] = started
    try:
        report.update(status="running", stage="waiting")
        save()
        while session.chunk < args.chunks:
            action = controls(args.output_dir, report["run_id"])
            if action["stop"]:
                report.update(status="stopped", stage="done")
                break
            if time.monotonic() - started >= 300:
                report.update(status="finished", stage="done")
                break
            # 已生成数量 - 已播放完成数量不得超过“正在播放+一段待播”。
            if session.chunk - (action["played"] + 1) >= 2:
                time.sleep(0.1)
                continue
            chunk_started = time.monotonic()
            report.update(stage="chunk", chunk_started=chunk_started, applied_keys=action["keys"], input_seq=action["seq"])
            save()
            with torch.inference_mode():
                video, latent = session.step(action["keys"])
            require_single_source("interactive_chunk_" + str(session.chunk))
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            if peak >= torch.cuda.get_device_properties(0).total_memory:
                raise RuntimeError("交互显存达到可见上限")
            frames = ((video.permute(1, 2, 3, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            index = session.chunk - 1
            meta = write_verified_mp4(args.output_dir / f"chunk-{index:03d}.mp4", frames, 16)
            report["segments"].append({"index": index, "verified": meta["mp4_verified"], "frames": meta["decoded_frame_count"], "bytes": meta["file_bytes"]})
            report.update(stage="waiting", chunks_done=session.chunk, chunk_seconds=time.monotonic() - chunk_started,
                          peak_allocated_gib=peak / 1024**3,
                          peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
                          cache_shapes=session.cache_shapes(), vae_cache_shapes=session.decoder.cache_shapes())
            save()
            del video, latent, frames
        else:
            report.update(status="finished", stage="done")
        save()
    finally:
        session.close()
        pipe._t5_cache.clear()
        del session, pipe
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--accept-experimental-config", action="store_true", required=True)
    parser.add_argument("--chunks", type=int, choices=(2, 8, MAX_CHUNKS), default=MAX_CHUNKS)
    args = parser.parse_args()
    directory = safe_directory(args.output_dir)
    run_id = os.environ.get("LINGBOT_VALIDATION_RUN_ID")
    if directory.parent != SESSION_ROOT or directory.name != run_id:
        raise ValueError("交互输出目录不匹配")
    report = {"run_id": run_id, "status": "preparing", "stage": "loading", "segments": []}

    def save():
        atomic_json(directory / "interactive.json", report)
        # 独立短事件，避免平台截断长JSON；私有报告保留完整诊断。
        event = {k: report[k] for k in ("run_id", "status", "stage", "chunks_done", "chunk_seconds", "peak_allocated_gib", "equivalence") if k in report}
        print("LINGBOT_INTERACTIVE=" + json.dumps(event, ensure_ascii=False), flush=True)

    try:
        save()
        execute(args, report, save)
    except Exception:
        report.update(status="failed", stage="done", traceback=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
