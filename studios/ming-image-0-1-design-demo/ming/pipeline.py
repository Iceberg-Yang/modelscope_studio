"""Hand-rolled inference pipeline for `inclusionAI/Ming-Image-0.1-Design`.

Follows vLLM-Omni PR 8021 (`vllm_omni/diffusion/models/ming_image/`, pinned to
a62d2ec999ae8fa669e8c22194c67575c0f2d3dc), the reference integration for this
checkpoint.

    raw prompt + <image> <imagePatch>x256 </image>   (no chat template)
    BailingMoeV2 "thinker", sequential positions for every token
      query  : final hidden state at the 256 learned queries
               -> proj_in -> bidirectional Qwen2 connector -> proj_out   (2560)
      direct : hidden states of the prompt tokens before blocks 5 and 12 and
               after the final norm, concatenated -> RMSNorm -> Linear   (3840)
    ZImageTransformer2DModel: caption = cap_embedder(query) ++ direct,
      alignment padding zeroed and masked out of attention
    FlowMatchEulerDiscreteScheduler (dynamic shifting, mu from image_seq_len)
    CFG whenever guidance_scale > 0, against zero query and zero direct conditioning
    AutoencoderKLQwenImage (z_dim 16, 4-channel / RGBA in+out, scaling 8.0064)
"""

import ctypes
import json
import os
import threading
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders import AutoencoderKLQwenImage
from safetensors import safe_open
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from .configuration_bailing_moe_v2 import BailingMoeV2Config
from .modeling_bailing_moe_v2 import BailingMoeV2ForCausalLM
from .transformer_z_image import ZImageTransformer2DModel

MODEL_ID = os.environ.get("MING_MODEL_ID", "inclusionAI/Ming-Image-0.1-Design")
VAE_SCALE_FACTOR = 8  # AutoencoderKLQwenImage: 3 down/up stages -> 8x spatial

_model_dir = None


def resolve_cache_dir():
    """ModelScope cache directory, preferring storage that survives restarts."""
    env = os.environ.get("MING_CACHE_DIR")
    if env:
        os.makedirs(env, exist_ok=True)
        return env
    base = "/mnt/workspace"
    if os.path.isdir(base) and os.access(base, os.W_OK):
        cache = os.path.join(base, ".ming_modelscope_cache")
        os.makedirs(cache, exist_ok=True)
        return cache
    return None


def resolve_model_dir():
    """Fetch the checkpoint from ModelScope and return its local snapshot dir.

    ModelScope xGPU instances cannot reach huggingface.co.  The repo id is
    identical on both hubs, so the ModelScope copy is used directly; the HF
    mirror stays as a fallback for local development.
    """
    global _model_dir
    if _model_dir is not None:
        return _model_dir
    try:
        from modelscope import snapshot_download

        _model_dir = str(snapshot_download(MODEL_ID, cache_dir=resolve_cache_dir()))
        print(f"[ming] checkpoint resolved from ModelScope: {_model_dir}", flush=True)
    except Exception as exc:
        print(f"[ming] ModelScope download failed ({exc}); trying the HF mirror", flush=True)
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from huggingface_hub import snapshot_download as hf_snapshot_download

        _model_dir = str(hf_snapshot_download(MODEL_ID))
    return _model_dir


def _download(path):
    return os.path.join(resolve_model_dir(), path)


def load_tokenizer():
    return PreTrainedTokenizerFast.from_pretrained(resolve_model_dir(), subfolder="mllm")


def load_thinker():
    """Build a meta-resident thinker and record its prompt-encoder shards."""
    with open(_download("mllm/config.json")) as f:
        mllm_cfg = json.load(f)
    llm_raw = dict(mllm_cfg["llm_config"])
    llm_raw.pop("_attn_implementation", None)
    llm_raw["_attn_implementation"] = "eager"
    cfg = BailingMoeV2Config(**llm_raw)

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            model = BailingMoeV2ForCausalLM(cfg)
    finally:
        torch.set_default_dtype(prev_dtype)
    model.lm_head = nn.Identity()
    model.eval()
    model.requires_grad_(False)

    with open(_download("mllm/model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    expected_keys = set(model.state_dict())
    by_shard = {}
    loaded_keys = set()
    for key, shard in weight_map.items():
        if not key.startswith("model.") or key.startswith("model.lm_head."):
            continue
        name = key[len("model."):]
        if name not in expected_keys:
            continue
        by_shard.setdefault(shard, []).append((key, name))
        loaded_keys.add(name)

    real_missing = sorted(expected_keys - loaded_keys)
    if real_missing:
        raise RuntimeError(f"Thinker checkpoint is missing keys: {real_missing[:20]}")
    model._ming_shards = tuple((shard, tuple(keys)) for shard, keys in sorted(by_shard.items()))
    print(f"[ming] thinker: indexed {len(loaded_keys)} tensors for CUDA paging", flush=True)
    return model, mllm_cfg

def load_mlp_module():
    """proj_in / proj_out / query tokens from `mlp/model.safetensors`."""
    with open(_download("mlp/config.json")) as f:
        cfg = json.load(f)
    with safe_open(_download("mlp/model.safetensors"), framework="pt") as f:
        sd = {k: f.get_tensor(k) for k in f.keys()}
    return cfg, sd


def load_connector():
    model = AutoModelForCausalLM.from_pretrained(
        resolve_model_dir(), subfolder="connector", dtype=torch.bfloat16
    )
    # Full bidirectional self-attention, exactly as the reference does.  In
    # transformers v5 this flag alone is not enough (see `encode_prompt`, which
    # passes an explicit per-layer additive mask); it is kept to mirror the
    # reference and to stay correct on older versions too.
    for layer in model.model.layers:
        layer.self_attn.is_causal = False
    model.eval()
    model.requires_grad_(False)
    return model


def _materialize_meta_tensors(model):
    """Replace any parameter / buffer still living on the `meta` device.

    diffusers' `from_pretrained` builds the module under `init_empty_weights`
    (accelerate) and then fills it from the checkpoint.  Keys *absent* from the
    checkpoint are therefore never materialised and the later
    `module.to("cuda")` dies with

        NotImplementedError: Cannot copy out of meta tensor; no data!

    For `ZImageTransformer2DModel` the only such keys are `x_pad_token` /
    `cap_pad_token`, which we zero anyway.  Materialising them here keeps the
    (memory-friendly) low-cpu-memory loading path instead of falling back to
    `low_cpu_mem_usage=False`.
    """
    patched = 0
    for module in model.modules():
        for name, param in list(module._parameters.items()):
            if param is not None and param.is_meta:
                module._parameters[name] = torch.nn.Parameter(
                    torch.zeros(param.shape, dtype=param.dtype, device="cpu"),
                    requires_grad=param.requires_grad,
                )
                patched += 1
        for name, buf in list(module._buffers.items()):
            if buf is not None and buf.is_meta:
                module._buffers[name] = torch.zeros(buf.shape, dtype=buf.dtype, device="cpu")
                patched += 1
    if patched:
        print(f"[ming] materialized {patched} meta tensor(s)", flush=True)
    return model


def load_transformer():
    model = ZImageTransformer2DModel.from_pretrained(resolve_model_dir(), subfolder="transformer", dtype=torch.bfloat16)
    _materialize_meta_tensors(model)
    # `x_pad_token` / `cap_pad_token` are `torch.empty` in __init__ and are absent
    # from the released checkpoint (they only pad sequences to SEQ_MULTI_OF, which
    # never bites at the supported resolutions).  Zero them so an unused garbage
    # value can never propagate a NaN.
    with torch.no_grad():
        model.x_pad_token.zero_()
        model.cap_pad_token.zero_()
    return model.eval()


def load_vae():
    vae = AutoencoderKLQwenImage.from_pretrained(resolve_model_dir(), subfolder="vae", dtype=torch.bfloat16)
    _materialize_meta_tensors(vae)
    return vae.eval()


def load_scheduler():
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(resolve_model_dir(), subfolder="scheduler")
    # ZImageLoss does exactly this before sampling: resolution-dependent shifting.
    # (`config` is a FrozenDict, so go through register_to_config.)
    scheduler.register_to_config(use_dynamic_shifting=True)
    return scheduler


def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _log_vram(tag):
    free, total = torch.cuda.mem_get_info()
    print(f"[ming] vram {tag}: {free / 1e9:.1f} GB free of {total / 1e9:.1f} GB", flush=True)


def _read_cgroup(name):
    for base in ("/sys/fs/cgroup", "/sys/fs/cgroup/memory"):
        try:
            raw = open(f"{base}/{name}").read().strip()
        except OSError:
            continue
        if raw == "max":
            return None
        try:
            return int(raw)
        except ValueError:
            return raw
    return None


def _release_host_memory():
    """Hand freed tensor pages back to the kernel.

    Paging the 31.9 GB thinker off the card allocates a whole second copy on
    the CPU side before the card copy is released.  glibc keeps those pages in
    its arena instead of returning them, so RSS ratchets up until the
    container's cgroup OOM killer fires -- a hard kill that leaves no
    traceback.  Forcing mmap semantics plus an explicit trim keeps RSS
    tracking real usage.
    """
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.mallopt(-3, 131072)  # M_MMAP_THRESHOLD: keep the heap arena flat
        libc.mallopt(-1, 131072)  # M_TRIM_THRESHOLD: trim on free
        libc.malloc_trim(0)
    except Exception:
        pass


def _memory_watch(tag):
    """Sample cgroup memory from a daemon thread while modules are paged.

    A cgroup OOM kill leaves nothing behind in the Python log, so the sampler
    prints the running peak and the kernel's own oom_kill counter; whatever it
    flushes before the kill is the only evidence left.
    """
    stop = threading.Event()
    limit = _read_cgroup("memory.max") or _read_cgroup("memory.limit_in_bytes") or 0
    last = [0, 0.0]

    def loop():
        while not stop.wait(2.0):
            cur = _read_cgroup("memory.current")
            if cur is None:
                cur = _read_cgroup("memory.usage_in_bytes")
            if not isinstance(cur, int):
                continue
            now = time.monotonic()
            if abs(cur - last[0]) < 1e9 and now - last[1] < 10:
                continue
            last[0], last[1] = cur, now
            peak = _read_cgroup("memory.peak")
            events = _read_cgroup("memory.events") or ""
            oom = next(
                (ln for ln in str(events).splitlines() if ln.startswith("oom_kill")), ""
            )
            free, _ = torch.cuda.mem_get_info()
            print(
                f"[ming] mem {tag}: cgroup {cur / 1e9:.1f}/{limit / 1e9:.1f} GB"
                + (f" peak {peak / 1e9:.1f} GB" if isinstance(peak, int) else "")
                + f" vram_free {free / 1e9:.1f} GB {oom}".rstrip(),
                flush=True,
            )

    threading.Thread(target=loop, daemon=True).start()
    return stop


def log_host_mem(tag):
    """Report host and cgroup memory headroom at a paging boundary."""
    vals = {}
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                vals[key] = rest.split()
    except OSError:
        return
    try:
        with open("/proc/self/status") as fh:
            rss = next(line for line in fh if line.startswith("VmRSS"))
        rss_gb = f"{int(rss.split()[1]) / 1e6:.1f} GB"
    except (OSError, StopIteration):
        rss_gb = "?"
    limit = _read_cgroup("memory.max") or _read_cgroup("memory.limit_in_bytes")
    current = _read_cgroup("memory.current") or _read_cgroup("memory.usage_in_bytes")
    peak = _read_cgroup("memory.peak")
    events = _read_cgroup("memory.events") or ""
    oom = next((line for line in str(events).splitlines() if line.startswith("oom_kill")), "")
    limit_text = f"{limit / 1e9:.1f} GB" if isinstance(limit, int) else "none"
    current_text = f"{current / 1e9:.1f} GB" if isinstance(current, int) else "?"
    peak_text = f", peak {peak / 1e9:.1f} GB" if isinstance(peak, int) else ""
    print(
        f"[ming] host mem {tag}: rss {rss_gb}, "
        f"available {int(vals['MemAvailable'][0]) / 1e6:.1f} GB of "
        f"{int(vals['MemTotal'][0]) / 1e6:.1f} GB, cgroup {current_text}/{limit_text}{peak_text} {oom}",
        flush=True,
    )


class MingImagePipeline:
    """Text-to-image pipeline for Ming-Image-0.1-Design (RGBA output)."""

    def __init__(self, thinker, mllm_cfg, connector, transformer, vae, scheduler, tokenizer, mlp_cfg, mlp_sd, device="cuda"):
        self.thinker = thinker
        self.thinker_cfg = mllm_cfg
        self.thinker_shards = thinker._ming_shards
        self.connector = connector
        self.transformer = transformer
        self.vae = vae
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.device = device
        self.llm_config_obj = thinker.config

        scales = list(mlp_cfg.get("img_gen_scales", [16]))
        if len(scales) != 1:
            raise ValueError(f"Ming-Image expects a single learned-query scale, got {scales}")
        self.num_queries = scales[0] * scales[0]
        self.selected_layers = list(mlp_cfg["selected_hidden_states_layers"])
        self.connector_norm = bool(mlp_cfg.get("connector_norm", False))
        self.text_encoder_norm = bool(mlp_cfg.get("text_encoder_norm", False))
        c_in = int(mlp_cfg.get("diffusion_c_input_dim", 2560))
        inner = int(mlp_cfg["diffusion_inner_dim"])
        hidden = thinker.config.hidden_size
        conn_hidden = connector.config.hidden_size

        dtype = torch.bfloat16
        self.proj_in = nn.Linear(hidden, conn_hidden).to(dtype)
        self.proj_in.load_state_dict(
            {"weight": mlp_sd["proj_in.weight"], "bias": mlp_sd["proj_in.bias"]}, strict=True
        )
        self.proj_out = nn.Linear(conn_hidden, c_in).to(dtype)
        self.proj_out.load_state_dict(
            {"weight": mlp_sd["proj_out.weight"], "bias": mlp_sd["proj_out.bias"]}, strict=True
        )
        direct_in = hidden * len(self.selected_layers)
        self.direct_projector = nn.Sequential(
            nn.RMSNorm(direct_in, eps=1e-5), nn.Linear(direct_in, inner, bias=True)
        ).to(dtype)
        self.direct_projector.load_state_dict(
            {
                "0.weight": mlp_sd["proj_directvlm.0.weight"],
                "1.weight": mlp_sd["proj_directvlm.1.weight"],
                "1.bias": mlp_sd["proj_directvlm.1.bias"],
            },
            strict=True,
        )
        self.query_tokens = mlp_sd[f"query_tokens_dict.{scales[0]}x{scales[0]}"].to(device=device, dtype=dtype)

        for module in (self.proj_in, self.proj_out, self.direct_projector):
            module.eval()
            module.requires_grad_(False)
            module.to(device)

    @torch.no_grad()
    def _hydrate_thinker(self):
        """Adopt each bf16 safetensor directly onto CUDA-owned parameters."""
        loaded = 0
        for shard, keys in self.thinker_shards:
            with safe_open(_download(f"mllm/{shard}"), framework="pt", device=torch.cuda.current_device()) as f:
                state = {name: f.get_tensor(source) for source, name in keys}
            self.thinker.load_state_dict(state, strict=False, assign=True)
            loaded += len(state)
            del state
            print(f"[ming] thinker: hydrated {loaded} tensors on cuda", flush=True)

        for module in self.thinker.modules():
            inv_freq = getattr(module, "inv_freq", None)
            if isinstance(inv_freq, torch.Tensor) and inv_freq.is_meta:
                inv_freq, attention_scaling = module.rope_init_fn(module.config, self.device)
                module.inv_freq = inv_freq
                module.original_inv_freq = inv_freq
                module.attention_scaling = attention_scaling
        torch.cuda.synchronize()

    @torch.no_grad()
    def _unload_thinker(self):
        """Release thinker GPU storage without allocating a CPU copy."""
        self.thinker.to_empty(device="meta")
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _thinker_stage(self, prompt):
        """Run the thinker over the prompt and hand back its raw conditioning."""
        if any(token in prompt for token in ("<image>", "<imagePatch>", "</image>", "<IMAGE>")):
            raise ValueError("Prompts cannot contain Ming's reserved image tokens.")
        cfg = self.llm_config_obj
        n = self.num_queries
        suffix = "<image>" + "<imagePatch>" * n + "</image>"
        input_ids = self.tokenizer(prompt + suffix, return_tensors="pt")["input_ids"].to(self.device)
        expected = [cfg.image_start_token] + [cfg.image_patch_token] * n + [cfg.image_end_token]
        if input_ids[0, -(n + 2):].tolist() != expected:
            raise ValueError("Tokenizer did not preserve Ming's learned-query suffix.")
        direct_end = input_ids.shape[1] - (n + 2)

        embeds = self.thinker.get_input_embeddings()(input_ids).clone()
        embeds[0, -(n + 1):-1] = self.query_tokens.to(embeds.dtype)
        out = self.thinker(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            position_ids=None,
            past_key_values=None,
            inputs_embeds=embeds,
            image_grid_thw=None,
            use_cache=False,
            image_mask=input_ids == cfg.image_patch_token,
            audio_mask=None,
            output_hidden_states=True,
        )
        states = out.hidden_states
        query = states[-1][:, -(n + 1):-1].contiguous()
        direct = torch.cat([states[layer][:, :direct_end] for layer in self.selected_layers], dim=-1)
        del out, states, embeds
        return query, direct

    @torch.no_grad()
    def _connector_stage(self, query, direct):
        """Project the thinker's output into the DiT's conditioning space."""
        query = self.proj_in(query)
        mask = torch.zeros((1, 1, query.shape[1], query.shape[1]), device=query.device, dtype=query.dtype)
        query = self.connector(
            inputs_embeds=query,
            attention_mask={"full_attention": mask},
            output_hidden_states=True,
        ).hidden_states[-1]
        query = self.proj_out(query)
        if self.connector_norm:
            query = F.normalize(query, dim=-1)
        if self.text_encoder_norm:
            query = query * 1000.0
        direct = self.direct_projector(direct)
        return query[0], direct[0]

    @torch.no_grad()
    def _encode_prompt_staged(self, prompt):
        """Trade the connector for a transient, shard-hydrated thinker."""
        log_host_mem("start")
        watch = _memory_watch("encode")
        self.connector.to("cpu")
        torch.cuda.empty_cache()
        _release_host_memory()
        _log_vram("connector parked")
        log_host_mem("connector parked")
        try:
            self._hydrate_thinker()
            _log_vram("thinker hydrated")
            log_host_mem("thinker hydrated")
            query, direct = self._thinker_stage(prompt)
            print("[ming] thinker forward done", flush=True)
        finally:
            self._unload_thinker()
            _log_vram("thinker unloaded")
            self.connector.to(self.device)
            _release_host_memory()
            torch.cuda.empty_cache()
            _log_vram("connector resident")
            log_host_mem("connector resident")
            watch.set()
        return self._connector_stage(query, direct)

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        height=1024,
        width=1024,
        num_inference_steps=12,
        guidance_scale=1.0,
        generator=None,
        callback=None,
    ):
        cap_feats, direct = self._encode_prompt_staged(prompt)

        # Mirrors the reference `ZImagePipeline.prepare_latents`: after this,
        # `height`/`width` are *latent* dimensions (the VAE has an 8x spatial
        # factor and the DiT patches latents 2x2, hence the 16), NOT pixels.
        vae_scale = VAE_SCALE_FACTOR * 2
        height = 2 * (int(height) // vae_scale)
        width = 2 * (int(width) // vae_scale)

        latents = torch.randn(
            (1, self.transformer.in_channels, height, width),
            generator=generator,
            device="cpu" if generator is not None and generator.device.type == "cpu" else self.device,
            dtype=torch.float32,
        ).to(self.device)

        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        self.scheduler.sigma_min = 0.0
        self.scheduler.set_timesteps(num_inference_steps, device=self.device, mu=mu)
        timesteps = self.scheduler.timesteps

        do_cfg = guidance_scale > 0
        pos_cap, pos_direct = cap_feats, direct
        neg_cap, neg_direct = torch.zeros_like(cap_feats), torch.zeros_like(direct)

        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0])
            timestep = (1000 - timestep) / 1000

            if do_cfg:
                lat_in = torch.cat([latents, latents], dim=0).to(self.transformer.dtype)
                caps = [pos_cap, neg_cap]
                directs = [pos_direct, neg_direct]
                ts = timestep.repeat(2)
            else:
                lat_in = latents.to(self.transformer.dtype)
                caps = [pos_cap]
                directs = [pos_direct]
                ts = timestep

            lat_in = lat_in.unsqueeze(2)
            model_out = self.transformer(
                list(lat_in.unbind(0)), ts, caps, cap_feats_2=directs, return_dict=False
            )[0]

            if do_cfg:
                pos = model_out[0].float()
                neg = model_out[1].float()
                noise_pred = (pos + guidance_scale * (pos - neg)).unsqueeze(0)
            else:
                noise_pred = torch.stack([o.float() for o in model_out], dim=0)

            noise_pred = -noise_pred.squeeze(2)
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if callback is not None:
                callback(i + 1, len(timesteps))

        latents = latents.to(self.vae.dtype)
        latents = (latents / float(self.vae.config.get("scaling_factor", 8.0064))) + float(
            self.vae.config.get("shift_factor", 0.0) or 0.0
        )
        image = self.vae.decode(latents.unsqueeze(2), return_dict=False)[0][:, :, 0]
        return image


def decode_to_pil(image):
    """image: [1, 4, H, W] in [-1, 1] -> RGBA PIL image."""
    from PIL import Image as PILImage

    img = (image / 2 + 0.5).clamp(0, 1)
    img = img[0].permute(1, 2, 0).float().cpu().numpy()
    img = (img * 255).round().astype("uint8")
    return PILImage.fromarray(img, mode="RGBA" if img.shape[-1] == 4 else "RGB")
