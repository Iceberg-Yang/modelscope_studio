"""Runtime GPU detection and per-compute-capability SageAttention2 build."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAGE_SOURCE = PROJECT_ROOT / "third_party" / "SageAttention"
STATUS_PATH = Path(
    os.environ.get("ABOT_RUNTIME_STATUS", "/tmp/abot-l20n-runtime.json")
)
EXTENSION_CACHE = Path(
    os.environ.get(
        "ABOT_EXTENSION_CACHE",
        "/mnt/workspace/abot-world/extensions",
    )
)
SUPPORTED_CC = {(12, 0)}


def _set_sageattention_enabled(enabled: bool):
    os.environ["ABOT_DISABLE_SAGEATTENTION"] = "0" if enabled else "1"


def _write_status(**values):
    status = {}
    if STATUS_PATH.exists():
        try:
            status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except Exception:
            status = {}
    status.update(values)
    status["updated_at"] = time.time()
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATUS_PATH)


def _verify_sageattention(env):
    """Verify attention/FP8 extensions and execute kernels in a fresh process.

    SageAttention's core module intentionally catches extension import errors.
    Import each binary explicitly so an ABI/SM mismatch remains visible in the
    container log instead of being reduced to ``SM89_ENABLED=False``.
    """
    verification = r"""
import importlib
import torch
import triton

# Exercise the exact FlashAttention2 varlen forward API used by ABot's
# cross-attention.  Use the real first-block sequence lengths and head shape so
# an ABI, SM120, or kernel-launch mismatch fails before the model is downloaded.
flash_attn = importlib.import_module("flash_attn")
flash_attn_varlen_func = getattr(flash_attn, "flash_attn_varlen_func", None)
if not callable(flash_attn_varlen_func):
    raise RuntimeError("flash_attn.flash_attn_varlen_func is not callable")
with torch.inference_mode():
    q_len, kv_len, num_heads, head_dim = 3920, 512, 24, 128
    flash_q = torch.randn(
        (q_len, num_heads, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    flash_k = torch.randn(
        (kv_len, num_heads, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    flash_v = torch.randn_like(flash_k)
    flash_cu_q = torch.tensor(
        [0, q_len],
        device="cuda",
        dtype=torch.int32,
    )
    flash_cu_k = torch.tensor(
        [0, kv_len],
        device="cuda",
        dtype=torch.int32,
    )
    flash_output = flash_attn_varlen_func(
        q=flash_q,
        k=flash_k,
        v=flash_v,
        cu_seqlens_q=flash_cu_q,
        cu_seqlens_k=flash_cu_k,
        max_seqlen_q=q_len,
        max_seqlen_k=kv_len,
        dropout_p=0.0,
        causal=False,
    )
    if (
        flash_output.shape != flash_q.shape
        or not torch.isfinite(flash_output).all()
    ):
        raise RuntimeError(
            "FlashAttention2 varlen smoke test failed: "
            f"shape={tuple(flash_output.shape)}"
        )
    del flash_q, flash_k, flash_v, flash_output
    torch.cuda.synchronize()
print(
    "[BOOT][L20N][VERIFY] "
    f"FlashAttention2 {flash_attn.__version__} SM120 varlen smoke test passed",
    flush=True,
)

sageattention = importlib.import_module("sageattention")
print(
    "[BOOT][L20N][VERIFY] "
    f"package={sageattention.__file__}, triton={triton.__version__}",
    flush=True,
)
for extension in ("_fused", "_qattn_sm80", "_qattn_sm89"):
    module = importlib.import_module(f"sageattention.{extension}")
    print(f"[BOOT][L20N][VERIFY] {extension}={module.__file__}", flush=True)

core = importlib.import_module("sageattention.core")
print(
    "[BOOT][L20N][VERIFY] "
    f"SM80_ENABLED={core.SM80_ENABLED}, SM89_ENABLED={core.SM89_ENABLED}",
    flush=True,
)
if not callable(getattr(sageattention, "sageattn", None)):
    raise RuntimeError("sageattention.sageattn is not callable")
if not core.SM89_ENABLED:
    raise RuntimeError("SageAttention SM89 CUDA path is not enabled for SM120")

with torch.inference_mode():
    # Match ABot's wrapper layout: [batch, sequence, heads, head_dim].
    q = torch.randn((1, 128, 2, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    output = sageattention.sageattn(
        q,
        k,
        v,
        tensor_layout="NHD",
        is_causal=False,
    )
    if isinstance(output, tuple):
        output = output[0]
    if output.shape != q.shape:
        raise RuntimeError(
            f"SageAttention smoke test shape mismatch: {output.shape} != {q.shape}"
        )
    torch.cuda.synchronize()
print("[BOOT][L20N][VERIFY] CUDA smoke test passed", flush=True)

# Exercise the exact dynamic FP8 path used by generator Linear modules:
# Triton per-token quantization followed by torch._scaled_mm.
from quantizer.quant.modules.linear import FP8DynamicLinear
from quantizer.quant.quant_func import fp8_per_token_group_quant

with torch.inference_mode():
    weight = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
    qweight, weight_scale = fp8_per_token_group_quant(weight, weight.shape[-1])
    fp8_linear = FP8DynamicLinear(
        weight=qweight,
        weight_scale=weight_scale.t(),
        bias=None,
        native_fp8_support=True,
        quant_type="fp8-per-token",
    )
    fp8_input = torch.randn(
        (1, 32, 128),
        device="cuda",
        dtype=torch.bfloat16,
    )
    fp8_output = fp8_linear(fp8_input)
    if fp8_output.shape != fp8_input.shape or not torch.isfinite(fp8_output).all():
        raise RuntimeError(
            f"FP8 smoke test failed: shape={tuple(fp8_output.shape)}"
        )
    torch.cuda.synchronize()
print("[BOOT][L20N][VERIFY] FP8 per-token GEMM smoke test passed", flush=True)

# Exercise the two global Triton replacements before loading the full model.
from wan.modules.helios_kernels.triton_norm import flash_rms_layernorm
from wan.modules.helios_kernels.triton_rope import Flash_RoPE_Transposed

with torch.inference_mode():
    norm = torch.nn.RMSNorm(
        128,
        eps=1e-6,
        device="cuda",
        dtype=torch.bfloat16,
    )
    norm_input = torch.randn(
        (2, 16, 128),
        device="cuda",
        dtype=torch.bfloat16,
    )
    norm_output = flash_rms_layernorm(norm, norm_input)
    rope_input = torch.randn(
        (1, 128, 2, 128),
        device="cuda",
        dtype=torch.bfloat16,
    )
    rope_freqs = torch.randn(
        (1, 128, 256),
        device="cuda",
        dtype=torch.bfloat16,
    )
    rope_output = Flash_RoPE_Transposed.apply(rope_input, rope_freqs)
    if (
        norm_output.shape != norm_input.shape
        or rope_output.shape != rope_input.shape
        or not torch.isfinite(norm_output).all()
        or not torch.isfinite(rope_output).all()
    ):
        raise RuntimeError("Helios Triton norm/RoPE smoke test failed")
    torch.cuda.synchronize()
print("[BOOT][L20N][VERIFY] Triton RMSNorm/RoPE smoke tests passed", flush=True)
"""
    _run([sys.executable, "-c", verification], env)


def _run(command, env):
    display_command = list(map(str, command))
    if "-c" in display_command:
        inline_index = display_command.index("-c") + 1
        if inline_index < len(display_command):
            display_command[inline_index] = "<inline verification script>"
    print("[BOOT][L20N] running:", " ".join(display_command), flush=True)
    subprocess.run(command, check=True, env=env)


def _sage_source_revision() -> str:
    digest = hashlib.sha256()
    suffixes = {".py", ".cu", ".cuh", ".cpp", ".h", ".hpp"}
    for path in sorted(
        item
        for item in SAGE_SOURCE.rglob("*")
        if item.is_file() and item.suffix in suffixes
    ):
        digest.update(str(path.relative_to(SAGE_SOURCE)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def prepare_runtime() -> bool:
    started = time.monotonic()
    _write_status(
        phase="detecting_gpu",
        flashattention2_available=False,
        sageattention_available=False,
    )

    if not torch.cuda.is_available():
        _set_sageattention_enabled(False)
        _write_status(
            phase="failed",
            error="CUDA is not available",
            torch_version=torch.__version__,
            torch_cuda=torch.version.cuda,
        )
        print("[BOOT][L20N] CUDA is not available.", flush=True)
        return False

    props = torch.cuda.get_device_properties(0)
    cc = (props.major, props.minor)
    sm = f"sm{props.major}{props.minor}"
    system_ram_gib = None
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        memtotal_line = next(
            line for line in meminfo.splitlines() if line.startswith("MemTotal:")
        )
        system_ram_gib = round(
            int(memtotal_line.split()[1]) * 1024 / (1024**3),
            2,
        )
    except Exception:
        pass
    hardware = {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "sm": sm,
        "total_gib": round(props.total_memory / (1024**3), 2),
        "system_ram_gib": system_ram_gib,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    print(
        "[BOOT][L20N] "
        f"name={props.name}, vram={hardware['total_gib']}GiB, "
        f"ram={hardware['system_ram_gib']}GiB, "
        f"cc={hardware['compute_capability']}, torch={torch.__version__}, "
        f"cuda={torch.version.cuda}",
        flush=True,
    )
    _write_status(phase="gpu_detected", hardware=hardware)

    if cc not in SUPPORTED_CC:
        _set_sageattention_enabled(False)
        message = (
            f"Expected the verified L20N CC 12.0 target, got "
            f"{props.major}.{props.minor}; "
            "refusing to build an SM80 or guessed CUDA extension"
        )
        print(f"[BOOT][L20N] {message}", flush=True)
        _write_status(phase="unsupported_cc", error=message)
        return False

    env = os.environ.copy()
    env["TORCH_CUDA_ARCH_LIST"] = f"{props.major}.{props.minor}"
    env["MAX_JOBS"] = env.get("MAX_JOBS", "4")
    env["EXT_PARALLEL"] = env.get("EXT_PARALLEL", env["MAX_JOBS"])
    env["NVCC_APPEND_FLAGS"] = env.get("NVCC_APPEND_FLAGS", "--threads 4")

    cache_dir = (
        EXTENSION_CACHE
        / "sageattention2"
        / (
            f"torch-{torch.__version__.split('+')[0]}-cu{torch.version.cuda}-"
            f"{sm}-{_sage_source_revision()}"
        )
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    wheels = sorted(cache_dir.glob("sageattention-*.whl"))

    try:
        _write_status(
            phase="installing_sageattention",
            target_arch=env["TORCH_CUDA_ARCH_LIST"],
            wheel_cache=str(cache_dir),
        )
        if not wheels:
            _write_status(phase="compiling_sageattention")
            _run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    str(SAGE_SOURCE),
                    "--no-build-isolation",
                    "--no-deps",
                    "--wheel-dir",
                    str(cache_dir),
                ],
                env,
            )
            wheels = sorted(cache_dir.glob("sageattention-*.whl"))
        if not wheels:
            raise RuntimeError("SageAttention build completed without a wheel")

        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--force-reinstall",
                str(wheels[-1]),
            ],
            env,
        )
        _write_status(phase="testing_sageattention")
        _verify_sageattention(env)

        elapsed = time.monotonic() - started
        _write_status(
            phase="ready",
            flashattention2_available=True,
            flashattention2_version="2.8.3",
            sageattention_available=True,
            target_arch=env["TORCH_CUDA_ARCH_LIST"],
            wheel=str(wheels[-1]),
            bootstrap_seconds=round(elapsed, 2),
        )
        print(
            "[BOOT][L20N] SageAttention2 ready: "
            f"SAGE_ATTN_AVAILABLE: True, cc={props.major}.{props.minor}",
            flush=True,
        )
        _set_sageattention_enabled(True)
        return True
    except Exception as exc:
        _set_sageattention_enabled(False)
        traceback.print_exc()
        _write_status(
            phase="sageattention_failed",
            sageattention_available=False,
            error=f"{type(exc).__name__}: {exc}",
            bootstrap_seconds=round(time.monotonic() - started, 2),
        )
        print(
            "[BOOT][L20N] SageAttention2 failed; the app will expose the "
            "failure in /healthz and use a safe attention fallback.",
            flush=True,
        )
        return False


if __name__ == "__main__":
    raise SystemExit(0 if prepare_runtime() else 1)
