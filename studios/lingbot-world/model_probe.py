"""严格加载原始权重并生成 MP4；单段或重复测试不等于实时交互验证通过。"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent


class PerformanceTimer:
    """互斥阶段计时；CUDA 流区间含主机提交间隙，不等于纯 kernel 执行时间。"""
    def __init__(self, cuda=None, clock=None):
        self.cuda = cuda
        self.clock = clock or time.perf_counter
        self.entries = []
        self.active = False

    @contextmanager
    def measure(self, stage, gpu=False, chunk=None):
        if self.active:
            raise RuntimeError("性能阶段不能嵌套，以免重复计时")
        if gpu:
            self.cuda.synchronize()
            begin, end = self.cuda.Event(enable_timing=True), self.cuda.Event(enable_timing=True)
        started = self.clock()
        self.active = True
        completed = False
        try:
            if gpu:
                begin.record()
            yield
            if gpu:
                end.record()
                self.cuda.synchronize()
            completed = True
        finally:
            elapsed = self.clock() - started
            self.active = False
            item = {"stage": stage, "wall_seconds": elapsed, "completed": completed, "chunk": chunk}
            if completed and gpu:
                item["cuda_stream_seconds"] = begin.elapsed_time(end) / 1000
            self.entries.append(item)

    def summary(self, chunk=None):
        stages = {}
        for item in self.entries:
            if not item["completed"] or (chunk is not None and item["chunk"] != chunk):
                continue
            entry = stages.setdefault(item["stage"], {"calls": 0, "wall_seconds": 0.0})
            entry["calls"] += 1
            entry["wall_seconds"] += item["wall_seconds"]
            if "cuda_stream_seconds" in item:
                entry["cuda_stream_seconds"] = entry.get("cuda_stream_seconds", 0.0) + item["cuda_stream_seconds"]
        return stages


@contextmanager
def temporary_attribute(obj, name, value):
    """只在独立 worker 的计时窗口替换，异常时也恢复实例与模块的原始属性。"""
    owned = name in vars(obj)
    original = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if owned:
            setattr(obj, name, original)
        else:
            delattr(obj, name)


@contextmanager
def profile_generation(pipe, timer, chunk_size, validate, progress, max_chunks=2):
    """按契约限制块数的可恢复仪表；不复制或改写上游生成算法。"""
    state = {"calls": 0, "chunk": None, "chunks": []}

    def timed(original, stage, gpu=True, in_chunk=False):
        def call(*args, **kwargs):
            with timer.measure(stage, gpu=gpu, chunk=state["chunk"] if in_chunk else None):
                return original(*args, **kwargs)
        return call

    encoder = pipe.text_encoder

    class TimedEncoder:
        def __getattr__(self, name):
            return getattr(encoder, name)

        def __call__(self, *args, **kwargs):
            with timer.measure("t5_encode", gpu=True):
                return encoder(*args, **kwargs)

    original_forward = pipe.model.forward
    chunk_started = None

    def forward(*args, **kwargs):
        nonlocal chunk_started
        start, frame = kwargs.get("current_start"), kwargs.get("frame_seqlen")
        if type(start) is not int or type(frame) is not int or start < 0 or frame <= 0:
            raise RuntimeError("无法识别原生 chunk 计时边界")
        chunk, offset = divmod(start, chunk_size * frame)
        if offset or chunk != state["calls"] // 5 or chunk >= max_chunks:
            raise RuntimeError("chunk数量超出契约，或不是每块4步去噪及1步缓存更新")
        state["chunk"] = chunk
        step = state["calls"] % 5
        if step == 0:
            chunk_started = timer.clock()
        stage = "dit_cache_update" if step == 4 else "dit_denoise"
        with timer.measure(stage, gpu=True, chunk=chunk):
            output = original_forward(*args, **kwargs)
        with timer.measure("dit_finite_check", chunk=chunk):
            validate(output)
        state["calls"] += 1
        with timer.measure("progress_report", chunk=chunk):
            progress(state["calls"])
        if step == 4:
            state["chunks"].append({"chunk": chunk, "forward_window_seconds": timer.clock() - chunk_started})
        return output

    with ExitStack() as stack:
        stack.enter_context(temporary_attribute(pipe, "text_encoder", TimedEncoder()))
        stack.enter_context(temporary_attribute(pipe.model, "forward", forward))
        targets = [(encoder.model, "to", "t5_to_device", False),
                   (pipe.vae, "encode", "vae_encode", False),
                   (pipe.vae, "decode", "vae_decode", False),
                   (pipe, "_initialize_self_kv_cache", "self_kv_init", False),
                   (pipe, "_initialize_crossattn_cache", "cross_kv_init", False),
                   (pipe, "_convert_flow_pred_to_x0", "flow_conversion", True),
                   (pipe.scheduler, "add_noise", "scheduler_add_noise", True)]
        for obj, name, stage, in_chunk in targets:
            stack.enter_context(temporary_attribute(obj, name, timed(getattr(obj, name), stage, in_chunk=in_chunk)))
        module = sys.modules[type(pipe).__module__]
        for name in ("get_Ks_transformed", "interpolate_camera_poses", "compute_relative_poses", "get_plucker_embeddings"):
            stack.enter_context(temporary_attribute(module, name, timed(getattr(module, name), "camera_condition_ops", gpu=name == "get_plucker_embeddings")))
        yield state


def cache_run_metadata(pipe, prompt, iteration):
    return {"run_kind": "first_in_process" if iteration == 0 else "repeat_same_instance",
            "t5_cache_hit_before": hashlib.sha256(prompt.encode("utf-8")).hexdigest() in pipe._t5_cache,
            "kv_cache_reused_between_sequences": False}


def emit_performance(run_id, iteration, performance):
    # 小型独立日志行，避免完整模型报告被平台截断后丢失性能数据。
    common = {"run_id": run_id, "iteration": iteration}
    totals = {key: value for key, value in performance.items() if key not in ("generation_stages", "postprocess_stages", "chunks")}
    print("LINGBOT_PERFORMANCE=" + json.dumps({**common, **totals}, ensure_ascii=False), flush=True)
    for group in ("generation_stages", "postprocess_stages"):
        for stage, values in performance[group].items():
            print("LINGBOT_PERFORMANCE=" + json.dumps({**common, "group": group, "stage": stage, **values}), flush=True)
    for chunk in performance["chunks"]:
        totals = {key: value for key, value in chunk.items() if key != "stages"}
        print("LINGBOT_PERFORMANCE=" + json.dumps({**common, **totals}), flush=True)
        for stage, values in chunk.get("stages", {}).items():
            print("LINGBOT_PERFORMANCE=" + json.dumps({**common, "chunk": chunk["chunk"], "stage": stage, **values}), flush=True)


def compare_shapes(expected, actual):
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatched = {key: {"expected": list(expected[key]), "actual": list(actual[key])} for key in set(expected) & set(actual) if tuple(expected[key]) != tuple(actual[key])}
    if missing or unexpected or mismatched:
        raise ValueError(json.dumps({"missing": missing, "unexpected": unexpected, "mismatched": mismatched}, ensure_ascii=False))
    return len(expected)


def verify_runtime(contract):
    import torch
    result = {"python": platform.python_version(), "packages": {}}
    for name in ("torch", "torchvision", "flash-attn", "diffusers", "transformers", "tokenizers", "huggingface-hub"):
        result["packages"][name] = importlib.metadata.version(name)
    for name in ("torch", "torchvision", "flash-attn"):
        if result["packages"][name].split("+")[0] != contract["runtime"][name]:
            raise RuntimeError(f"平台二进制版本不匹配: {name}")
    if not torch.cuda.is_available() or torch.version.cuda != contract["runtime"]["cuda"]:
        raise RuntimeError("CUDA 不可用或版本不匹配")
    props = torch.cuda.get_device_properties(0)
    result.update(gpu=props.name, compute_capability=f"{props.major}.{props.minor}", visible_vram_gib=props.total_memory / 1024**3, cuda=torch.version.cuda)
    return result


def strict_load(primary, config, report, save):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from wan.modules.model_fast import WanModelFast

    started = time.perf_counter()
    index = json.loads((primary / "model.safetensors.index.json").read_text())["weight_map"]
    state = {}
    actual = {}
    for shard in sorted(set(index.values())):
        report["stage"] = "read_weight_shard:" + shard
        save()
        if Path(shard).name != shard or not shard.endswith(".safetensors"):
            raise ValueError("不安全的分片文件名")
        with safe_open(primary / shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in actual or index.get(key) != shard:
                    raise ValueError("分片与索引不一致: " + key)
                actual[key] = handle.get_slice(key).get_shape()
        state.update(load_file(primary / shard, device="cpu"))
    if set(index) != set(actual):
        raise ValueError("索引与张量集合不一致")
    with torch.device("meta"):
        model = WanModelFast(**config)
    expected = {name: value.shape for name, value in model.state_dict().items()}
    count = compare_shapes(expected, actual)
    loaded = model.load_state_dict(state, strict=True, assign=True)
    if loaded.missing_keys or loaded.unexpected_keys or any(p.is_meta for p in model.parameters()):
        raise RuntimeError("严格加载后仍有缺失参数")
    report["strict_load_cpu"] = {"tensor_count": count, "missing_keys": [], "unexpected_keys": [], "shape_mismatches": []}
    report["stage"] = "move_model_to_cuda"
    save()
    model = model.eval().requires_grad_(False).to(device="cuda", dtype=torch.float32)
    # freqs 在上游不属于注册 buffer；meta 构造后须按原始公式重新创建。
    from wan.modules.model import rope_params
    d = config["dim"] // config["num_heads"]
    model.freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))], dim=1).to("cuda")
    del state
    gc.collect()
    torch.cuda.synchronize()
    report["strict_load"] = {"tensor_count": count, "missing_keys": [], "unexpected_keys": [], "shape_mismatches": [], "parameter_count": sum(p.numel() for p in model.parameters()), "weight_dtype": "float32", "seconds": time.perf_counter() - started}
    return model


def build_pipeline(primary, auxiliary, config, report, save):
    import copy
    import torch
    from wan import WanI2VCausal
    from wan.configs.wan_i2v_A14B import i2v_A14B
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae2_1 import Wan2_1_VAE
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    # 只适配上游构造阶段的 checkpoint 目录与严格加载；直接复用其 generate。
    pipe = WanI2VCausal.__new__(WanI2VCausal)
    pipe.config = copy.deepcopy(i2v_A14B)
    pipe.config.update(config)
    pipe.device = torch.device("cuda:0")
    pipe.rank = 0
    pipe.sp_size = 1
    pipe.t5_cpu = False
    pipe.init_on_cpu = True
    pipe.infer_mode = "causal_fast"
    pipe.num_train_timesteps = pipe.config.num_train_timesteps
    pipe.boundary = pipe.config.boundary
    pipe.param_dtype = torch.bfloat16
    pipe.pipe_dtype = torch.bfloat16
    pipe.local_attn_size = config["local_attn_size"]
    pipe.sink_size = config["sink_size"]
    pipe.vae_stride = pipe.config.vae_stride
    pipe.patch_size = config["patch_size"]
    pipe.sample_neg_prompt = pipe.config.sample_neg_prompt
    pipe._t5_cache = {}
    pipe._cross_attn_initialized = False
    loading = PerformanceTimer(torch.cuda)
    with loading.measure("load_main", gpu=True):
        pipe.model = strict_load(primary, config, report, save)
    report["stage"] = "load_t5"
    save()
    with loading.measure("load_t5", gpu=True):
        pipe.text_encoder = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device=torch.device("cpu"), checkpoint_path=str(auxiliary / "models_t5_umt5-xxl-enc-bf16.pth"), tokenizer_path=str(auxiliary / "google/umt5-xxl"))
    report["stage"] = "load_vae"
    save()
    with loading.measure("load_vae", gpu=True):
        pipe.vae = Wan2_1_VAE(vae_pth=str(auxiliary / "Wan2.1_VAE.pth"), device=pipe.device)
    report["load_stages"] = loading.summary()
    pipe.scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=pipe.num_train_timesteps, shift=1, use_dynamic_shifting=False)
    return pipe


def write_verified_mp4(output_path, frames, fps, timer=None):
    import imageio.v2 as imageio
    timer = timer or PerformanceTimer()
    temporary = output_path.with_name(output_path.stem + ".partial.mp4")
    with timer.measure("mp4_encode"):
        imageio.mimwrite(temporary, frames, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1, ffmpeg_params=["-movflags", "+faststart"])
    count = 0
    with timer.measure("mp4_readback"):
        reader = imageio.get_reader(temporary, format="FFMPEG")
        try:
            for frame in reader:
                if frame.shape != frames[0].shape:
                    raise RuntimeError("MP4 回读画面尺寸不一致")
                count += 1
        finally:
            reader.close()
        if count != len(frames) or temporary.stat().st_size == 0:
            raise RuntimeError("MP4 回读帧数不一致或文件为空")
    with timer.measure("mp4_publish"):
        temporary.replace(output_path)
    return {"mp4_verified": True, "decoded_frame_count": count, "file_bytes": output_path.stat().st_size}


def prepare_runtime(args, report, save):
    import torch
    from cudnn_probe import require_single_source

    contract = json.loads((HERE / "validation_contract.json").read_text())
    report["runtime"] = verify_runtime(contract)
    report["cudnn_sources"] = {"runtime": require_single_source("runtime")}
    sys.path.insert(0, str(args.source))
    from wan.modules import attention as attention_module
    report["cudnn_sources"]["wan_import"] = require_single_source("wan_import")
    # L20 采用已探测的 FlashAttention 2；不因其他包存在而选用 FA3。
    attention_module.FLASH_ATTN_3_AVAILABLE = False
    with torch.inference_mode():
        query = torch.randn(1, 32, 12, 128, device="cuda", dtype=torch.bfloat16)
        output = attention_module.flash_attention(query, query, query, version=2)
        reference = torch.nn.functional.scaled_dot_product_attention(query.transpose(1, 2), query.transpose(1, 2), query.transpose(1, 2)).transpose(1, 2)
        if not torch.isfinite(output).all() or not torch.allclose(output, reference, atol=0.03, rtol=0.03):
            raise RuntimeError("实际 FlashAttention2 算子与 SDPA 对照失败")
        report["flash_attention_forward"] = {"passed": True, "max_abs_diff": (output - reference).abs().max().item()}
    del query, output, reference
    save()
    return contract


def execute(args, report, save):
    import torch
    from PIL import Image
    from cudnn_probe import require_single_source

    space_request = getattr(args, "space_request", False)
    if space_request:
        from space_adapter import prepare_input
        prompt, image, sample, workload = prepare_input(args.source, args.output_dir)
    contract = prepare_runtime(args, report, save)
    primary = args.root / "primary" / contract["primary"]["revision"]
    auxiliary = args.root / "auxiliary" / contract["auxiliary"]["revision"]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    report["stage"] = "load_pipeline"
    save()
    pipe = build_pipeline(primary, auxiliary, contract["model_config"], report, save)
    report["cudnn_sources"]["pipeline_loaded"] = require_single_source("pipeline_loaded")
    report["load_seconds"] = time.perf_counter() - started
    save()
    for stage, values in report["load_stages"].items():
        print("LINGBOT_PERFORMANCE=" + json.dumps({"run_id": report.get("run_id"), "group": "load_stages", "stage": stage, **values}), flush=True)

    def check_forward(outputs):
        if not all(bool(torch.isfinite(tensor).all()) for tensor in outputs):
            raise RuntimeError("DiT forward 输出包含 NaN/Inf")

    def progress(count):
        report["stage"] = f"dit_forward:{count}/{expected_calls}"
        save()

    if not space_request:
        workload = dict(contract["smoke_workload"])
        sample = args.source / "examples" / workload["example"]
        prompt = (sample / "prompt.txt").read_text().strip()
        image = Image.open(sample / "image.jpg").convert("RGB").resize((workload["width"], workload["height"]))
    workload["repetitions"] = args.repetitions
    report["workload"] = workload
    chunks = ((workload["frame_num"] - 1) // 4 + 1) // workload["chunk_size"]
    expected_calls = chunks * 5
    report["input"] = {"example": workload.get("example", "uploaded"), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "source_revision": workload.get("source_revision", contract["source"]["revision"])}
    report["sequences"] = []
    baseline = None
    for iteration in range(workload["repetitions"]):
        report["stage"] = f"generate_sequence:{iteration + 1}"
        save()
        if not space_request:
            torch.cuda.reset_peak_memory_stats()
        else:
            # 原生KV为每次generate的局部变量；这里只限制持久文本缓存。
            key = hashlib.sha256(prompt.encode()).hexdigest()
            pipe._t5_cache = {key: pipe._t5_cache[key]} if key in pipe._t5_cache else {}
            gc.collect()
            torch.cuda.empty_cache()
        sequence_started = time.perf_counter()
        generation = PerformanceTimer(torch.cuda)
        postprocess = PerformanceTimer(torch.cuda)
        performance = cache_run_metadata(pipe, prompt, iteration)
        with profile_generation(pipe, generation, workload["chunk_size"], check_forward, progress, max_chunks=chunks) as state:
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                video = pipe.generate(input_prompt=prompt, img=image, action_path=str(sample), chunk_size=workload["chunk_size"], max_area=workload["width"] * workload["height"], frame_num=workload["frame_num"], timesteps_index=workload.get("timesteps_index", [0, 250, 500, 750]), shift=workload.get("shift", 5.0), seed=workload["seed"], offload_model=False)
            torch.cuda.synchronize()
            performance["generate_seconds"] = time.perf_counter() - started
        with postprocess.measure("cudnn_source_check"):
            report["cudnn_sources"][f"sequence_{iteration + 1}"] = require_single_source("after_generation")
        seconds = time.perf_counter() - started
        with postprocess.measure("video_validation", gpu=True):
            if video.ndim != 4 or video.shape[:2] != (3, workload["frame_num"]) or not torch.isfinite(video).all():
                raise RuntimeError("视频形状或有限值检查失败")
            if state["calls"] != expected_calls:
                raise RuntimeError("模型调用次数与本次chunk契约不一致")
            result = {"iteration": iteration + 1, "seconds": seconds, "shape": list(video.shape), "dit_forward_calls": state["calls"], "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3, "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3, "finite": True, "std": video.std().item(), "t5_cache_entries": len(pipe._t5_cache)}
        with postprocess.measure("video_to_cpu", gpu=True):
            cpu_video = video.cpu()
        with postprocess.measure("video_statistics"):
            if baseline is not None:
                result["repeat_mean_abs_diff"] = (cpu_video - baseline).abs().mean().item()
            else:
                baseline = cpu_video.clone()
            result["temporal_mean_abs_diff"] = (cpu_video[:, 1:] - cpu_video[:, :-1]).abs().mean().item()
        with postprocess.measure("frames_to_uint8"):
            frames = ((cpu_video.permute(1, 2, 3, 0).clamp(-1, 1) + 1) * 127.5).to(torch.uint8).numpy()
        output_path = args.output_dir / f"smoke-{iteration + 1}.mp4"
        report["stage"] = "encode_and_verify_mp4"
        with postprocess.measure("progress_report"):
            save()
        result.update(write_verified_mp4(output_path, frames, workload["fps"], timer=postprocess))
        result["video"] = str(output_path)
        if space_request:
            import resource
            result["full_task_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
            result["full_task_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
            result["host_peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
            if result["full_task_peak_reserved_gib"] > workload["peak_reserved_limit_gib"]:
                report["resource_rejected"] = result
                raise RuntimeError("全任务峰值显存超过40GiB验收门槛")
        report["sequences"].append(result)
        del video, cpu_video, frames
        with postprocess.measure("result_report"):
            save()
        performance.update(generation_stages=generation.summary(), postprocess_stages=postprocess.summary(), chunks=state["chunks"])
        performance["generation_unattributed_seconds"] = max(0.0, performance["generate_seconds"] - sum(item["wall_seconds"] for item in performance["generation_stages"].values()))
        performance["sequence_total_seconds"] = time.perf_counter() - sequence_started
        performance["profiled_frames_per_second"] = result["decoded_frame_count"] / performance["generate_seconds"]
        for chunk in performance["chunks"]:
            chunk["stages"] = generation.summary(chunk=chunk["chunk"])
        result["performance"] = performance
        emit_performance(report.get("run_id"), iteration + 1, performance)
        save()
    report["stage"] = "finished"
    report["status"] = "experimental_smoke_passed"
    report["not_verified"] = ["官方1.3B配置确认", "画质与动作方向人工验收", "SGLang运行", "实时键盘/流协议", "长时会话", "重启后缓存恢复"]
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repetitions", type=int, choices=(1, 2), default=2)
    parser.add_argument("--space-request", action="store_true")
    parser.add_argument("--accept-experimental-config", action="store_true", required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir or args.root
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "run_id": os.environ.get("LINGBOT_VALIDATION_RUN_ID"), "backend": "official_wan_native", "config_status": "experimental_not_official_1.3b_config"}

    def save():
        text = json.dumps(report, ensure_ascii=False, indent=2)
        temporary = args.output_dir / "model-probe.json.tmp"
        temporary.write_text(text + "\n")
        temporary.replace(args.output_dir / "model-probe.json")
        print("LINGBOT_MODEL_PROBE=" + json.dumps(report, ensure_ascii=False), flush=True)

    try:
        execute(args, report, save)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        from cudnn_probe import loaded_cudnn
        report["loaded_cudnn"] = loaded_cudnn()
        save()
        for line in report["traceback"].splitlines():
            print("LINGBOT_VALIDATION_STAGE=" + json.dumps({"stage": "model_trace", "line": line}, ensure_ascii=False), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
