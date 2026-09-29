import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKLQwenImage,
    BitsAndBytesConfig,
    QwenImageEditPipeline,
    QwenImageTransformer2DModel,
)
from huggingface_hub import hf_hub_download
from matplotlib import colormaps
from peft import LoraConfig, prepare_model_for_kbit_training
from PIL import Image
from safetensors.torch import load_file

BASE_MODEL_URI = "Qwen/Qwen-Image-Edit-2509"
MODEL_URI = "huawei-bayerlab/marigold-v2-0"

# Output name -> (model subfolder, kind of the decoded RGB).
CHECKPOINTS = {
    "Depth": ("depth/Log-stage2", "depth"),
    "See-through Depth": ("depth/Log-layered", "depth"),
    "Normals": ("normals", "normals"),
    "Albedo": ("albedo", "albedo"),
}
PROMPT_EMBEDS = {
    "depth": "qwen_edit_2509_qwen_depth_realimg512",
    "normals": "qwen_edit_2509_qwen_normals_dummy512",
    "albedo": "qwen_edit_2509_qwen_albedo_rgb_dummy512",
}
LORA_TARGET_MODULES = [
    "img_in",
    "txt_in",
    "norm_out.linear",
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "img_mlp.net.0.proj",
    "img_mlp.net.2",
    "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
]
PROCESSING_MAX_LONG_SIDE = 1536
SEED = 2026


def resolve_file(uri: str, filename: str) -> str:
    if os.path.isdir(uri):
        return os.path.join(uri, filename)
    return hf_hub_download(uri, filename)


def load_vae(uri_base: str, device: torch.device) -> AutoencoderKLQwenImage:
    return AutoencoderKLQwenImage.from_pretrained(
        uri_base,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
        device_map=device,
        use_safetensors=True,
    )


def load_transformer(uri_base: str, device: torch.device) -> QwenImageTransformer2DModel:
    nf4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
    )
    transformer = QwenImageTransformer2DModel.from_pretrained(
        uri_base,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        quantization_config=nf4,
        device_map=device,
    )
    # Training kept the non-quantized parameters in fp32 through this call; mirror it for identical numerics.
    prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)
    transformer.add_adapter(
        LoraConfig(r=128, lora_alpha=128, target_modules=LORA_TARGET_MODULES)
    )
    for name, param in transformer.named_parameters():
        if "lora_" in name:
            param.data = param.data.to(torch.bfloat16)
    transformer.requires_grad_(False)
    return transformer


def load_prompt_embeds(uri_model: str, prefix: str, device: torch.device):
    embeds = torch.load(resolve_file(uri_model, f"qwen_text_embeddings/{prefix}_prompt_embeds.pt"))
    mask = torch.load(resolve_file(uri_model, f"qwen_text_embeddings/{prefix}_prompt_mask.pt"))
    return embeds[:1].to(device, torch.bfloat16), mask[:1].to(device) > 0


def preprocess(image: Image.Image, device: torch.device) -> torch.Tensor:
    scale = min(1.0, PROCESSING_MAX_LONG_SIDE / max(image.size))
    size = tuple(math.ceil(round(side * scale) / 16) * 16 for side in image.size)
    rgb = np.asarray(image.resize(size, Image.LANCZOS), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)


def latent_stats(vae: AutoencoderKLQwenImage, like: torch.Tensor):
    shape = (1, vae.config.z_dim, 1, 1, 1)
    mean = torch.tensor(vae.config.latents_mean, device=like.device, dtype=like.dtype)
    std_inv = 1.0 / torch.tensor(vae.config.latents_std, device=like.device, dtype=like.dtype)
    return mean.view(shape), std_inv.view(shape)


def encode(rgb: torch.Tensor, vae: AutoencoderKLQwenImage, generator: torch.Generator) -> torch.Tensor:
    latents = vae.encode(rgb.to(vae.dtype).unsqueeze(2)).latent_dist.sample(generator)
    mean, std_inv = latent_stats(vae, latents)
    return (latents - mean) * std_inv


def decode(latents: torch.Tensor, vae: AutoencoderKLQwenImage) -> torch.Tensor:
    mean, std_inv = latent_stats(vae, latents)
    return vae.decode(latents / std_inv + mean).sample[:, :, 0]


def flow_step(
    latents: torch.Tensor,
    transformer: QwenImageTransformer2DModel,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
) -> torch.Tensor:
    batch, channels, _, height, width = latents.shape
    packed = QwenImageEditPipeline._pack_latents(
        latents[:, :, 0], batch, channels, height, width
    ).to(torch.bfloat16)
    # bf16 arithmetic on purpose: this is how the timestep was computed during training.
    timestep = torch.full((batch,), 499.0, device=latents.device, dtype=torch.bfloat16) / 1000.0
    velocity = transformer(
        hidden_states=packed,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_mask=prompt_mask,
        img_shapes=[[(1, height // 2, width // 2)]] * batch,
        return_dict=False,
    )[0]
    velocity = QwenImageEditPipeline._unpack_latents(velocity, height * 8, width * 8, vae_scale_factor=8)
    return latents - velocity.to(latents.dtype)


def to_image(rgb: torch.Tensor) -> Image.Image:
    return Image.fromarray((rgb.clamp(0, 1) * 255).round().permute(1, 2, 0).byte().cpu().numpy())


def visualize(pixels: torch.Tensor, kind: str) -> Image.Image:
    if kind == "normals":
        return to_image((F.normalize(pixels, dim=0) + 1) / 2)
    # The model predicts linear-RGB reflectance; encode it as sRGB for display.
    albedo = ((pixels + 1) / 2).clamp(0, 1)
    return to_image(torch.where(albedo <= 0.0031308, 12.92 * albedo, 1.055 * albedo ** (1 / 2.4) - 0.055))


def colorize_depth(depth: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> Image.Image:
    value = ((depth - low) / (high - low)).cpu().numpy()
    return Image.fromarray((colormaps["Spectral"](value)[..., :3] * 255).round().astype(np.uint8))


def align_depth(source: torch.Tensor, target: torch.Tensor, band: float = 0.05, proposals: int = 3000, points: int = 40000):
    # Affine map of source onto target fitted on their consensus pixels (opaque surfaces): the line with the
    # most support in the joint value space, refined by least squares on its supporters.
    generator = torch.Generator(source.device).manual_seed(SEED)
    valid = (source.abs() < 0.98) & (target.abs() < 0.98)
    x, y = source[valid], target[valid]
    pick = torch.randint(len(x), (points,), device=x.device, generator=generator)
    x, y = x[pick], y[pick]
    i, j = torch.randint(points, (2, proposals), device=x.device, generator=generator)
    a = (y[j] - y[i]) / (x[j] - x[i])
    b = y[i] - a * x[i]
    keep = ((x[j] - x[i]).abs() > 0.1) & (a > 0.1) & (a < 10)
    a, b = a[keep], b[keep]
    support = torch.cat([((a[k : k + 500, None] * x + b[k : k + 500, None] - y).abs() < band).sum(dim=1) for k in range(0, len(a), 500)])
    best = support.argmax()
    inliers = (a[best] * x + b[best] - y).abs() < band
    for _ in range(3):
        x_mean, y_mean = x[inliers].mean(), y[inliers].mean()
        a = ((x[inliers] - x_mean) * (y[inliers] - y_mean)).sum() / ((x[inliers] - x_mean) ** 2).sum()
        b = y_mean - a * x_mean
        inliers = (a * x + b - y).abs() < band
    return a, b


def visualize_all(pixels: dict[str, torch.Tensor]) -> dict[str, Image.Image]:
    depth = {name: pixels[name].mean(dim=0) for name, (_, kind) in CHECKPOINTS.items() if kind == "depth"}
    # Show both depth views on one color scale, matched where the scene is opaque.
    a, b = align_depth(depth["Depth"], depth["See-through Depth"])
    depth["Depth"] = a * depth["Depth"] + b
    low, high = min(d.min() for d in depth.values()), max(d.max() for d in depth.values())
    return {
        name: colorize_depth(depth[name], low, high) if kind == "depth" else visualize(pixels[name], kind)
        for name, (_, kind) in CHECKPOINTS.items()
    }


class MarigoldV2:
    def __init__(self, uri_base: str, uri_model: str, device: torch.device):
        self.device = device
        self.vae = load_vae(uri_base, device)
        self.transformer = load_transformer(uri_base, device)
        self.prompt_embeds = {
            prefix: load_prompt_embeds(uri_model, prefix, device) for prefix in set(PROMPT_EMBEDS.values())
        }
        self.checkpoints = {
            name: load_file(resolve_file(uri_model, f"{subfolder}/trainables.safetensors"))
            for name, (subfolder, _) in CHECKPOINTS.items()
        }
        # Some checkpoints ship a fine-tuned VAE decoder; keep the stock one to restore for those that do not.
        self.vae_decoder_state = {
            k: v.cpu() for k, v in self.vae.state_dict().items() if k.startswith(("decoder.", "post_quant_conv."))
        }

    def load_checkpoint(self, name: str):
        state_dict = self.checkpoints[name]
        lora_state = {k.removeprefix("Diffuser."): v for k, v in state_dict.items() if k.startswith("Diffuser.")}
        missing, unexpected = self.transformer.load_state_dict(lora_state, strict=False)
        missing_lora = [k for k in missing if "lora_" in k]
        if unexpected or missing_lora:
            raise ValueError(f"{name}: unexpected keys {unexpected}, missing LoRA keys {missing_lora}")
        vae_state = dict(self.vae_decoder_state)
        vae_state.update({k.removeprefix("VAE."): v for k, v in state_dict.items() if k.startswith("VAE.")})
        self.vae.load_state_dict(vae_state, strict=False)

    @torch.no_grad()
    def __call__(self, image: Image.Image) -> dict[str, Image.Image]:
        rgb = preprocess(image, self.device)
        generator = torch.Generator(self.device).manual_seed(SEED)
        pixels = {}
        timings = {}
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            latents_rgb = encode(rgb, self.vae, generator)
            for name, (_, kind) in CHECKPOINTS.items():
                start = time.perf_counter()
                self.load_checkpoint(name)
                latents = flow_step(latents_rgb, self.transformer, *self.prompt_embeds[PROMPT_EMBEDS[kind]])
                decoded = decode(latents, self.vae)
                pixels[name] = F.interpolate(decoded.float(), size=image.size[::-1], mode="bilinear", align_corners=False)[0]
                timings[name] = time.perf_counter() - start
        outputs = visualize_all(pixels)
        print(
            f"Processed {image.size[0]}x{image.size[1]} at {rgb.shape[-1]}x{rgb.shape[-2]}: "
            + ", ".join(f"{name} {seconds:.1f}s" for name, seconds in timings.items()),
            flush=True,
        )
        return outputs
