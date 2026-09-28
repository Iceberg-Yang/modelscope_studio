"""Lazy Turbo-LoRA switching for the DiffSynth MiniMax-H3 NF4 pipeline."""

from __future__ import annotations

import os
import gc

import torch

LARRY_REPO = os.environ.get("H3_LORA_REPO", "larryvrh/MiniMax-H3-Turbo-Lora")
LARRY_FILE = os.environ.get("H3_LORA", "minimax_h3_turbo_v4_step600_ema.safetensors")
LIGHTX_REPO = os.environ.get("H3_LIGHTX_REPO", "lightx2v/Minimax-h3-Turbo")
LIGHTX_FILE = os.environ.get("H3_LIGHTX_FILE", "minimax_h3_fl2v_turbo_4step_v0.1.safetensors")
LARRY_STRENGTH = float(os.environ.get("H3_LORA_STRENGTH", "1.0"))
LIGHTX_ALPHA = float(os.environ.get("H3_LIGHTX_ALPHA", "8"))

_ACTIVE = "off"
_CACHE: dict[str, dict[str, torch.Tensor]] = {}


def _download(model_id: str, file_path: str, cache_dir: str) -> str:
    from modelscope.hub.file_download import model_file_download

    print(f"[lora] downloading {model_id}/{file_path}", flush=True)
    return model_file_download(model_id=model_id, file_path=file_path, cache_dir=cache_dir)


def _load(path: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(path, device="cpu")


def _strip_lightx_prefix(name: str) -> str:
    for prefix in ("base_model.model.", "transformer.", "diffusion_model."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name


def _lightx_to_diffsynth(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert the Diffusers-native LightX LoRA to DiffSynth's reference module tree."""
    suffix_a = ".lora_A.default.weight"
    suffix_b = ".lora_B.default.weight"
    bases = sorted(key[: -len(suffix_a)] for key in state if key.endswith(suffix_a))
    pairs = {
        _strip_lightx_prefix(base): (state[base + suffix_a], state[base + suffix_b])
        for base in bases
        if base + suffix_b in state
    }
    converted: dict[str, torch.Tensor] = {}

    def emit(target: str, a: torch.Tensor, b: torch.Tensor):
        converted[target + ".lora_A.weight"] = a
        converted[target + ".lora_B.weight"] = b

    # Fuse every Q/K/V triplet first. Alphabetical traversal visits `to_k`
    # before `to_q`, which previously emitted 52 stray `to_k`/`to_v` targets.
    consumed = set()
    for base in sorted(item for item in pairs if item.endswith(".attn.to_q")):
        prefix = base[: -len("to_q")]
        triplet = [prefix + "to_q", prefix + "to_k", prefix + "to_v"]
        if not all(item in pairs for item in triplet):
            raise KeyError(f"Incomplete q/k/v LoRA set under {prefix}")
        factors = [pairs[item] for item in triplet]
        ranks = [item[0].shape[0] for item in factors]
        a_fused = torch.cat([item[0] for item in factors], dim=0)
        out_sizes = [item[1].shape[0] for item in factors]
        b_fused = torch.zeros(
            sum(out_sizes), sum(ranks), dtype=factors[0][1].dtype, device=factors[0][1].device
        )
        out_offset = rank_offset = 0
        for (_, b_part), out_size, rank in zip(factors, out_sizes, ranks):
            b_fused[out_offset : out_offset + out_size, rank_offset : rank_offset + rank] = b_part
            out_offset += out_size
            rank_offset += rank
        target = prefix.replace("transformer_blocks.", "blocks.") + "qkv_proj"
        target = target.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.")
        emit(target, a_fused, b_fused)
        consumed.update(triplet)

    for base, (a, b) in pairs.items():
        if base in consumed:
            continue

        target = base.replace("transformer_blocks.", "blocks.")
        target = target.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.")
        target = target.replace(".attn.to_out.0", ".attn.out_proj")
        target = target.replace(".ff.net.2", ".mlp.fc2")
        target = target.replace("norm_out.linear", "final_layer.adaln_proj.linear")
        if target.endswith(".ff.net.0.proj"):
            # Diffusers stores [value, gate]; the reference/DiffSynth tree stores [gate, value].
            value, gate = b.chunk(2, dim=0)
            b = torch.cat([gate, value], dim=0)
            target = target.replace(".ff.net.0.proj", ".mlp.fc1")
        emit(target, a, b)
        consumed.add(base)

    if not converted:
        raise ValueError("No supported LightX LoRA tensors were found")
    return converted


def _state(name: str, cache_dir: str) -> tuple[dict[str, torch.Tensor], float]:
    if name in _CACHE:
        state = _CACHE[name]
    elif name == "larry":
        state = _load(_download(LARRY_REPO, LARRY_FILE, cache_dir))
        _CACHE.clear()
        _CACHE[name] = state
    elif name == "lightx":
        raw = _load(_download(LIGHTX_REPO, LIGHTX_FILE, cache_dir))
        state = _lightx_to_diffsynth(raw)
        del raw
        _CACHE.clear()
        _CACHE[name] = state
    else:
        raise ValueError(f"Unknown LoRA: {name}")

    if name == "larry":
        return state, LARRY_STRENGTH
    # Fusing q/k/v triples creates rank 3r factors, while ordinary targets retain
    # rank r.  LightX's alpha is defined against the original rank r.
    rank = min(value.shape[0] for key, value in state.items() if key.endswith(".lora_A.weight"))
    return state, LIGHTX_ALPHA / rank


def _validate_state(pipe, name: str, state: dict[str, torch.Tensor]) -> dict[str, int]:
    """Require complete LoRA pairs with exact target and matrix-shape coverage."""
    suffix_a = ".lora_A.weight"
    suffix_b = ".lora_B.weight"
    targets_a = {key.removesuffix(suffix_a) for key in state if key.endswith(suffix_a)}
    targets_b = {key.removesuffix(suffix_b) for key in state if key.endswith(suffix_b)}
    if not targets_a:
        raise RuntimeError(f"LoRA `{name}` does not contain DiffSynth LoRA A tensors")

    missing_a = sorted(targets_b - targets_a)
    missing_b = sorted(targets_a - targets_b)
    if missing_a or missing_b:
        raise RuntimeError(
            f"LoRA `{name}` has incomplete A/B pairs: "
            f"missing_A={missing_a[:5]}, missing_B={missing_b[:5]}"
        )

    modules = dict(pipe.dit.named_modules())
    unmatched = sorted(targets_a - modules.keys())
    if unmatched:
        raise RuntimeError(
            f"LoRA `{name}` only partially matches DiffSynth: "
            f"matched={len(targets_a) - len(unmatched)}/{len(targets_a)}, "
            f"unmatched={unmatched[:8]}"
        )

    shape_errors = []
    ranks = []
    for target in sorted(targets_a):
        tensor_a = state[target + suffix_a]
        tensor_b = state[target + suffix_b]
        module = modules[target]
        if tensor_a.ndim != 2 or tensor_b.ndim != 2:
            shape_errors.append(f"{target}: A{tuple(tensor_a.shape)} B{tuple(tensor_b.shape)}")
            continue
        rank = tensor_a.shape[0]
        ranks.append(rank)
        expected_in = getattr(module, "in_features", None)
        expected_out = getattr(module, "out_features", None)
        if (
            tensor_b.shape[1] != rank
            or (expected_in is not None and tensor_a.shape[1] != expected_in)
            or (expected_out is not None and tensor_b.shape[0] != expected_out)
        ):
            shape_errors.append(
                f"{target}: A{tuple(tensor_a.shape)} B{tuple(tensor_b.shape)} "
                f"target=({expected_out},{expected_in})"
            )
    if shape_errors:
        raise RuntimeError(f"LoRA `{name}` has incompatible tensor shapes: {shape_errors[:8]}")

    return {
        "pairs": len(targets_a),
        "rank_min": min(ranks),
        "rank_max": max(ranks),
    }


def set_active(pipe, name: str, cache_dir: str) -> str:
    global _ACTIVE
    name = name if name in ("larry", "lightx", "off") else "off"
    if name == _ACTIVE:
        return name

    pipe.clear_lora(verbose=0)
    if name != "off":
        _CACHE.clear()
        gc.collect()
        state, alpha = _state(name, cache_dir)
        coverage = _validate_state(pipe, name, state)
        pipe.load_lora(pipe.dit, state_dict=state, alpha=alpha, hotload=True, verbose=1)
        print(
            f"[lora] active={name}; validated_pairs={coverage['pairs']}; "
            f"rank={coverage['rank_min']}..{coverage['rank_max']}; alpha={alpha:g}; mode=hotload",
            flush=True,
        )
    _ACTIVE = name
    return name
