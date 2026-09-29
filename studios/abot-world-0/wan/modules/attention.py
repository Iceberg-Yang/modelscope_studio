# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import statistics
import torch

try:
    import flash_attn_interface

    def is_hopper_gpu():
        """flashattn-hopper 仅支持 Hopper (H100)，不支持 Blackwell，故只检测 Hopper。"""
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name or "hopper" in device_name
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    if os.environ.get("ABOT_DISABLE_SAGEATTENTION", "0") == "1":
        raise ModuleNotFoundError("SageAttention disabled by L20N bootstrap")
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False

try:
    from sageattn3 import sageattn3_blackwell
    SAGE_ATTN_3_BLACKWELL_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_3_BLACKWELL_AVAILABLE = False

from .sla_attn import get_block_map
from .sla_kernel import _attention


# FLASH_ATTN_3_AVAILABLE = False

import warnings

# 优先级从高到低，模块加载时按此顺序选取第一个可用的作为默认后端
# ATTN_BACKEND_PRIORITY = ['flash_attn_3', 'flash_attn_2', 'sdpa']
ATTN_BACKEND_PRIORITY = ['sla_triton', 'sageattn', 'sageattn3', 'flash_attn_3', 'flash_attn_2', 'sdpa']
# ATTN_BACKEND_PRIORITY = ['flash_attn_3', 'flash_attn_2', 'sdpa']
print(f"Attention backend priority: {ATTN_BACKEND_PRIORITY}")
print(f"SAGE_ATTN_3_BLACKWELL_AVAILABLE: {SAGE_ATTN_3_BLACKWELL_AVAILABLE}")
print(f"SAGE_ATTN_AVAILABLE: {SAGE_ATTN_AVAILABLE}")
print(f"FLASH_ATTN_3_AVAILABLE: {FLASH_ATTN_3_AVAILABLE}")
print(f"FLASH_ATTN_2_AVAILABLE: {FLASH_ATTN_2_AVAILABLE}")


def _normalize_backend_name(name):
    aliases = {
        "flash_attn": "flash_attn_2",
        "flashattention2": "flash_attn_2",
        "sageattention": "sageattn",
        "sageattention2": "sageattn",
    }
    normalized = str(name or "auto").strip().lower()
    return aliases.get(normalized, normalized)


REQUESTED_ATTN_BACKEND = _normalize_backend_name(
    os.environ.get("ABOT_SELF_ATTN_BACKEND", "auto")
)
_ATTN_BACKEND_CALL_COUNTS = {}


def _resolve_default_attn_backend(requested=None):
    """根据优先级数组和当前环境可用性，解析出默认的 attention 后端。"""
    requested = _normalize_backend_name(
        REQUESTED_ATTN_BACKEND if requested is None else requested
    )
    if requested != "auto":
        if not _is_backend_available(requested):
            raise RuntimeError(
                f"Requested attention backend {requested!r} is unavailable"
            )
        return requested
    availability = {
        'sageattn3': SAGE_ATTN_3_BLACKWELL_AVAILABLE,
        'sageattn': SAGE_ATTN_AVAILABLE,
        'flash_attn_3': FLASH_ATTN_3_AVAILABLE,
        'flash_attn_2': FLASH_ATTN_2_AVAILABLE,
        'sla_triton': False,
        'sdpa': True,  # 始终可用
    }
    for name in ATTN_BACKEND_PRIORITY:
        if availability.get(name, False):
            return name
    return 'sdpa'


def _is_power_of_two(x):
    return x > 0 and (x & (x - 1)) == 0


def _is_backend_available(backend):
    availability = {
        'sageattn3': SAGE_ATTN_3_BLACKWELL_AVAILABLE,
        'sageattn': SAGE_ATTN_AVAILABLE,
        'flash_attn_3': FLASH_ATTN_3_AVAILABLE,
        'flash_attn_2': FLASH_ATTN_2_AVAILABLE,
        'sla_triton': True,
        'sdpa': True,
    }
    return availability.get(backend, False)


def _can_use_backend(
    backend,
    q,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    causal=False,
    window_size=(-1, -1),
):
    # Layout in this module: [B, L, H, D]
    b, _, _, d = q.shape
    has_varlen = q_lens is not None or k_lens is not None
    has_window = window_size != (-1, -1)
    has_dropout = dropout_p > 0

    if backend == 'sla_triton':
        # Current SLA Triton path does not consume varlen/causal/window/dropout args,
        # and Triton kernel in sla_attn.py requires power-of-two D.
        return (
            not has_varlen
            and not has_window
            and not causal
            and not has_dropout
            and b == 1
            and _is_power_of_two(d)
        )
    if backend in ('sageattn', 'sageattn3'):
        # Sage kernels in this file do not support varlen/window/dropout controls.
        return not has_varlen and not has_window and not has_dropout
    if backend == 'flash_attn_3':
        # flash-attn3 path here does not support dropout/window_size.
        return not has_window and not has_dropout
    if backend == 'flash_attn_2':
        return True
    if backend == 'sdpa':
        return True
    return False


def _resolve_auto_backend(
    q,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    causal=False,
    window_size=(-1, -1),
):
    candidates = [DEFAULT_ATTN_BACKEND] + [b for b in ATTN_BACKEND_PRIORITY if b != DEFAULT_ATTN_BACKEND]
    for backend in candidates:
        if not _is_backend_available(backend):
            continue
        if _can_use_backend(
            backend=backend,
            q=q,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
        ):
            return backend
    return 'sdpa'


# 模块初始化时确定默认 attention 后端
DEFAULT_ATTN_BACKEND = _resolve_default_attn_backend()


def set_default_attn_backend(backend):
    """Select the process-wide self-attention backend after a runtime benchmark."""
    global DEFAULT_ATTN_BACKEND
    DEFAULT_ATTN_BACKEND = _resolve_default_attn_backend(backend)
    print(
        f"[ATTN_SELECT] requested={REQUESTED_ATTN_BACKEND} "
        f"selected={DEFAULT_ATTN_BACKEND}",
        flush=True,
    )
    return DEFAULT_ATTN_BACKEND


def attention_backend_status():
    return {
        "requested_backend": REQUESTED_ATTN_BACKEND,
        "default_backend": DEFAULT_ATTN_BACKEND,
        "call_counts": dict(_ATTN_BACKEND_CALL_COUNTS),
    }


@torch.inference_mode()
def benchmark_attention_backends(
    *,
    num_heads,
    head_dim,
    first_query_tokens,
    steady_query_tokens,
    steady_kv_tokens,
    dtype=torch.bfloat16,
    warmup=1,
    repeats=3,
    include_sdpa=False,
    min_speedup_ratio=0.03,
):
    """Benchmark the two production backends with ABot's real self-attn shapes."""
    if not torch.cuda.is_available():
        return {"selected": DEFAULT_ATTN_BACKEND, "results": {}}

    candidates = [
        name
        for name in ("sageattn", "flash_attn_2")
        if _is_backend_available(name)
    ]
    if include_sdpa:
        candidates.append("sdpa")
    shapes = {
        "first": (int(first_query_tokens), int(first_query_tokens)),
        "steady": (int(steady_query_tokens), int(steady_kv_tokens)),
    }
    results = {}
    device = torch.device("cuda")
    benchmark_generator = torch.Generator(device=device)
    benchmark_generator.manual_seed(20260730)

    for backend in candidates:
        backend_results = {}
        try:
            for shape_name, (q_tokens, kv_tokens) in shapes.items():
                q = torch.randn(
                    (1, q_tokens, int(num_heads), int(head_dim)),
                    device=device,
                    dtype=dtype,
                    generator=benchmark_generator,
                )
                k = torch.randn(
                    (1, kv_tokens, int(num_heads), int(head_dim)),
                    device=device,
                    dtype=dtype,
                    generator=benchmark_generator,
                )
                v = torch.randn(
                    k.shape,
                    device=device,
                    dtype=dtype,
                    generator=benchmark_generator,
                )
                for _ in range(max(0, int(warmup))):
                    attention(q, k, v, backend=backend, _count_backend=False)
                torch.cuda.synchronize()
                samples = []
                for _ in range(max(1, int(repeats))):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    attention(q, k, v, backend=backend, _count_backend=False)
                    end.record()
                    end.synchronize()
                    samples.append(float(start.elapsed_time(end)))
                backend_results[shape_name] = round(
                    statistics.median(samples), 3
                )
                del q, k, v
            # Steady state dominates a long interactive session.
            backend_results["score_ms"] = round(
                backend_results["first"] + 4.0 * backend_results["steady"],
                3,
            )
            results[backend] = backend_results
            print(
                f"[ATTN_BENCH] backend={backend} "
                f"first={backend_results['first']:.3f}ms "
                f"steady={backend_results['steady']:.3f}ms "
                f"score={backend_results['score_ms']:.3f}",
                flush=True,
            )
        except Exception as exc:
            results[backend] = {"error": repr(exc)}
            print(
                f"[ATTN_BENCH] backend={backend} failed={exc!r}",
                flush=True,
            )
        finally:
            torch.cuda.empty_cache()

    valid = {
        name: values
        for name, values in results.items()
        if "score_ms" in values
    }
    selected = DEFAULT_ATTN_BACKEND
    if REQUESTED_ATTN_BACKEND == "auto" and valid:
        candidate = min(valid, key=lambda name: valid[name]["score_ms"])
        baseline = valid.get(DEFAULT_ATTN_BACKEND)
        if (
            candidate == DEFAULT_ATTN_BACKEND
            or baseline is None
            or valid[candidate]["score_ms"]
            <= baseline["score_ms"] * (1.0 - float(min_speedup_ratio))
        ):
            selected = candidate
        set_default_attn_backend(selected)
    else:
        set_default_attn_backend(REQUESTED_ATTN_BACKEND)
    return {
        "selected": selected,
        "results": results,
        "shapes": shapes,
        "min_speedup_ratio": float(min_speedup_ratio),
    }

__all__ = [
    'ATTN_BACKEND_PRIORITY',
    'DEFAULT_ATTN_BACKEND',
    'REQUESTED_ATTN_BACKEND',
    'attention_backend_status',
    'benchmark_attention_backends',
    'set_default_attn_backend',
    'flash_attention',
    'sage_attention',
    'sage_attention3_blackwell',
    'attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([_u[:_v] for _u, _v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([_u[:_v] for _u, _v in zip(k, k_lens)]))
        v = half(torch.cat([_u[:_v] for _u, _v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def sage_attention(
    q,
    k,
    v,
    dropout_p=0.,
    softmax_scale=None,
    q_scale = None,
    causal=False,
    dtype=torch.bfloat16,
    smooth_k=True,
):
    """
    使用 SageAttention 的 attention，输入输出与 flash_attention 兼容。
    q, k, v: [B, L, N, C] (NHD: batch, seq_len, num_heads, head_dim)。
    不支持 q_lens/k_lens（变长序列），此类场景请用 flash_attention 或 attention(backend='flash_attn')。
    """
    assert SAGE_ATTN_AVAILABLE

    if q_scale is not None:
        q = q * q_scale

    half_dtypes = (torch.float16, torch.bfloat16)
    out_dtype = q.dtype
    if q.dtype not in half_dtypes:
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)

    # sageattn 支持 NHD: (batch_size, seq_len, head_num, head_dim)，与当前 [B, L, N, C] 一致，无需转置
    kwargs = dict(tensor_layout="NHD", is_causal=causal, smooth_k=smooth_k)
    if softmax_scale is not None:
        kwargs["sm_scale"] = softmax_scale
    attn_output = sageattn(q, k, v, **kwargs)
    return attn_output.type(out_dtype)


def sage_attention3_blackwell(
    q,
    k,
    v,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    dtype=torch.bfloat16,
):
    """
    使用 SageAttention3 Blackwell (FP4) 的 attention，输入输出与 flash_attention 兼容。
    q, k, v: [B, L, N, C] (NHD: batch, seq_len, num_heads, head_dim)。
    内部自动转置为 sageattn3_blackwell 所需的 HND layout，调用方无需感知。
    仅支持 Blackwell GPU (sm120, RTX 5090 等)，需单独安装 sageattn3 包。
    不支持 q_lens/k_lens（变长序列）、window_size（滑动窗口）、dropout。
    """
    assert SAGE_ATTN_3_BLACKWELL_AVAILABLE

    if q_scale is not None:
        q = q * q_scale

    half_dtypes = (torch.float16, torch.bfloat16)
    out_dtype = q.dtype
    if q.dtype not in half_dtypes:
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)

    # sageattn3_blackwell 内部固定使用 HND layout: [B, H, L, D]
    # 输入为 NHD: [B, L, N, C]，transpose 为零拷贝，contiguous 由内部 pad_128 统一处理
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    attn_output = sageattn3_blackwell(q, k, v, is_causal=causal)

    return attn_output.transpose(1, 2).contiguous().type(out_dtype)

def sla_triton(
    q,
    k,
    v,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    **kwargs,
):
    sparsity_ratio = 0.8
    topk = 1 - sparsity_ratio

    # (B, L, H, D) -> (B, H, L, D)
    B, L, H, D = q.shape

    # 根据设备 shared memory 能力与 head_dim 动态选择 BLOCK 大小，避免 OOR。
    # 参考 sla_kernel._attention 中的约束：BLOCK_{M,N} ∈ {64, 128}
    try:
        props = torch.cuda.get_device_properties(q.device)
        max_smem = getattr(props, "shared_memory_per_block", 0)
    except Exception:
        max_smem = 0

    # 粗略估计 forward kernel 的 shared memory 需求：
    # main buffer 近似 ~ (3 * BLOCK_M * D + 2 * BLOCK_N * D) * 4 Bytes
    # 为安全起见给一点冗余。
    def _estimate_smem(block_m, block_n, d):
        elems = (3 * block_m * d + 2 * block_n * d)
        return elems * 4

    # 默认尝试较大的 block，若超出显存再退回 64。
    blk_m, blk_n = 128, 128
    if max_smem and _estimate_smem(blk_m, blk_n, D) > max_smem:
        blk_m, blk_n = 64, 64

    BLKQ, BLKK = blk_m, blk_n
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()

    sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio=topk, BLKQ=BLKQ, BLKK=BLKK)

    out = _attention.apply(q, k, v, sparse_map, lut, real_topk, BLKQ, BLKK)
    out = out.transpose(1, 2)

    return out


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
    backend='auto',
    smooth_k=True,
    _count_backend=True,
):
    """
    backend: 'auto' | 'sageattn3' | 'sageattn' | 'flash_attn_3' | 'flash_attn_2' | 'flash_attn' | 'sdpa'
      - 'auto': 按 ATTN_BACKEND_PRIORITY 在模块初始化时选定的默认后端
      - 其他: 强制使用对应后端
    注意：sageattn3/sageattn 不支持 q_lens/k_lens/window_size，遇到此类参数会自动回退到 flash 或 sdpa。
    """
    if backend == 'auto':
        backend = _resolve_auto_backend(
            q=q,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
        )

    if _count_backend:
        _ATTN_BACKEND_CALL_COUNTS[backend] = (
            _ATTN_BACKEND_CALL_COUNTS.get(backend, 0) + 1
        )

    if backend == 'sageattn':
        return sage_attention(
            q=q, k=k, v=v,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            dtype=dtype,
            smooth_k=smooth_k,
        )
    elif backend == 'flash_attn_2' or backend == 'flash_attn_3':
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=2 if backend == 'flash_attn_2' else 3,
        )
    elif backend == 'sageattn3':
        return sage_attention3_blackwell(
            q=q, k=k, v=v,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            dtype=dtype,
        )
    elif backend == 'sla_triton':
        return sla_triton(
            q=q,
            k=k,
            v=v,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p, scale=softmax_scale)

        out = out.transpose(1, 2).contiguous()
        return out
