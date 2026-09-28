"""TIPSv2 Feature Explorer for ModelScope xGPU Studio."""

import colorsys
import gc
import os
import types
from pathlib import Path

# Fail closed if any third-party model code accidentally tries another hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from fast_pytorch_kmeans import KMeans as TorchKMeans
from modelscope.hub.snapshot_download import snapshot_download
from sklearn.decomposition import PCA
from torchvision import transforms
from transformers import AutoConfig, AutoModel

# ── Constants ───────────────────────────────────────────────────────────────

DEFAULT_IMAGE_SIZE = 896
PATCH_SIZE = 14
RESOLUTIONS = [224, 336, 448, 672, 896, 1120, 1372, 1792]

ZEROSEG_IMAGE_SIZE = 1372

VARIANTS = {
    "TIPS v2 — B/14": {
        "dpt": "google/tipsv2-b14-dpt",
        "backbone": "google/tipsv2-b14",
    },
    "TIPS v2 — L/14": {
        "dpt": "google/tipsv2-l14-dpt",
        "backbone": "google/tipsv2-l14",
    },
    "TIPS v2 — SO400m/14": {
        "dpt": "google/tipsv2-so400m14-dpt",
        "backbone": "google/tipsv2-so400m14",
    },
    "TIPS v2 — g/14": {
        "dpt": "google/tipsv2-g14-dpt",
        "backbone": "google/tipsv2-g14",
    },
}
DEFAULT_VARIANT = "TIPS v2 — L/14"
MODELSCOPE_MODEL_REVISION = os.environ.get("MODELSCOPE_MODEL_REVISION", "master")
MODEL_CACHE_DIR = os.environ.get(
    "MODELSCOPE_CACHE",
    "/mnt/workspace/.cache/modelscope"
    if Path("/mnt/workspace").is_dir()
    else str(Path.home() / ".cache" / "modelscope"),
)


def _model_dtype():
    """Use half precision on xGPU by default; allow an FP32 compatibility mode."""
    dtype_name = os.environ.get("TIPS_DTYPE", "float16").lower()
    if dtype_name in {"float32", "fp32"} or not torch.cuda.is_available():
        return torch.float32
    if dtype_name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    return torch.float16


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Pascal Context (59 classes) ─────────────────────────────────────────────

TCL_PROMPTS = [
    "itap of a {}.",
    "a bad photo of a {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
    "a photo of many {}.",
    "a photo of {}s.",
]

PASCAL_CONTEXT_CLASSES = (
    "aeroplane",
    "bag",
    "bed",
    "bedclothes",
    "bench",
    "bicycle",
    "bird",
    "boat",
    "book",
    "bottle",
    "building",
    "bus",
    "cabinet",
    "car",
    "cat",
    "ceiling",
    "chair",
    "cloth",
    "computer",
    "cow",
    "cup",
    "curtain",
    "dog",
    "door",
    "fence",
    "floor",
    "flower",
    "food",
    "grass",
    "ground",
    "horse",
    "keyboard",
    "light",
    "motorbike",
    "mountain",
    "mouse",
    "person",
    "plate",
    "platform",
    "pottedplant",
    "road",
    "rock",
    "sheep",
    "shelves",
    "sidewalk",
    "sign",
    "sky",
    "snow",
    "sofa",
    "table",
    "track",
    "train",
    "tree",
    "truck",
    "tvmonitor",
    "wall",
    "water",
    "window",
    "wood",
)

ADE20K_CLASSES = (
    "wall",
    "building",
    "sky",
    "floor",
    "tree",
    "ceiling",
    "road",
    "bed",
    "windowpane",
    "grass",
    "cabinet",
    "sidewalk",
    "person",
    "earth",
    "door",
    "table",
    "mountain",
    "plant",
    "curtain",
    "chair",
    "car",
    "water",
    "painting",
    "sofa",
    "shelf",
    "house",
    "sea",
    "mirror",
    "rug",
    "field",
    "armchair",
    "seat",
    "fence",
    "desk",
    "rock",
    "wardrobe",
    "lamp",
    "bathtub",
    "railing",
    "cushion",
    "base",
    "box",
    "column",
    "signboard",
    "chest_of_drawers",
    "counter",
    "sand",
    "sink",
    "skyscraper",
    "fireplace",
    "refrigerator",
    "grandstand",
    "path",
    "stairs",
    "runway",
    "case",
    "pool_table",
    "pillow",
    "screen_door",
    "stairway",
    "river",
    "bridge",
    "bookcase",
    "blind",
    "coffee_table",
    "toilet",
    "flower",
    "book",
    "hill",
    "bench",
    "countertop",
    "stove",
    "palm",
    "kitchen_island",
    "computer",
    "swivel_chair",
    "boat",
    "bar",
    "arcade_machine",
    "hovel",
    "bus",
    "towel",
    "light",
    "truck",
    "tower",
    "chandelier",
    "awning",
    "streetlight",
    "booth",
    "television",
    "airplane",
    "dirt_track",
    "apparel",
    "pole",
    "land",
    "bannister",
    "escalator",
    "ottoman",
    "bottle",
    "buffet",
    "poster",
    "stage",
    "van",
    "ship",
    "fountain",
    "conveyer_belt",
    "canopy",
    "washer",
    "plaything",
    "swimming_pool",
    "stool",
    "barrel",
    "basket",
    "waterfall",
    "tent",
    "bag",
    "minibike",
    "cradle",
    "oven",
    "ball",
    "food",
    "step",
    "tank",
    "trade_name",
    "microwave",
    "pot",
    "animal",
    "bicycle",
    "lake",
    "dishwasher",
    "screen",
    "blanket",
    "sculpture",
    "hood",
    "sconce",
    "vase",
    "traffic_light",
    "tray",
    "ashcan",
    "fan",
    "pier",
    "crt_screen",
    "plate",
    "monitor",
    "bulletin_board",
    "shower",
    "radiator",
    "glass",
    "clock",
    "flag",
)

NUM_ADE20K_CLASSES = 150
ADE20K_PALETTE = np.zeros((NUM_ADE20K_CLASSES + 1, 3), dtype=np.uint8)
for i in range(1, NUM_ADE20K_CLASSES + 1):
    hue = (i * 0.618033988749895) % 1.0
    saturation = 0.65 + 0.35 * ((i * 7) % 5) / 4.0
    value = 0.70 + 0.30 * ((i * 11) % 3) / 2.0
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    ADE20K_PALETTE[i] = [int(r * 255), int(g * 255), int(b * 255)]

# ── Model state (one model loaded at a time) ───────────────────────────────

_model = {
    "name": None,
    "vision": None,
    "text": None,
    "temperature": None,
    "ade20k_embs": None,
    "dpt": None,
    "backbone": None,
}


def _release_model():
    """Release the previous variant before loading another one."""
    for key in tuple(_model):
        _model[key] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sdpa_attention_forward(module, x, attn_bias=None):
    """Memory-efficient attention for the custom TIPSv2 vision encoder."""
    if attn_bias is not None:
        raise ValueError("Nested attention inputs are not used by this Studio")
    batch, tokens, channels = x.shape
    qkv = module.qkv(x).reshape(
        batch,
        tokens,
        3,
        module.num_heads,
        channels // module.num_heads,
    )
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    output = output.transpose(1, 2).reshape(batch, tokens, channels)
    output = module.proj(output)
    return module.proj_drop(output)


def _enable_sdpa(model):
    """Replace the explicit N x N attention fallback when xFormers is absent."""
    patched = 0
    for module in model.modules():
        if (
            module.__class__.__name__ in {"Attention", "MemEffAttention"}
            and hasattr(module, "qkv")
            and hasattr(module, "num_heads")
        ):
            module.forward = types.MethodType(_sdpa_attention_forward, module)
            patched += 1
    return patched


def _download_model(model_id):
    """Download exclusively from ModelScope and return the local snapshot path."""
    return snapshot_download(
        model_id=model_id,
        revision=MODELSCOPE_MODEL_REVISION,
        cache_dir=MODEL_CACHE_DIR,
    )


def _patch_b14_dpt_snapshot(snapshot_dir):
    """Make the legacy B/14 DPT module resolve its sibling code locally."""
    snapshot_dir = Path(snapshot_dir)
    modeling_path = snapshot_dir / "modeling_dpt.py"
    head_path = snapshot_dir / "dpt_head.py"
    if not modeling_path.is_file() or not head_path.is_file():
        raise FileNotFoundError(
            "B/14 DPT ModelScope snapshot must contain modeling_dpt.py and "
            "dpt_head.py"
        )

    source = modeling_path.read_text(encoding="utf-8")
    legacy_loader = '        dpt_mod = _load_sibling("dpt_head", repo_id)\n\n'
    legacy_references = (
        "dpt_mod.DPTDepthHead",
        "dpt_mod.DPTNormalsHead",
        "dpt_mod.DPTSegmentationHead",
    )

    # A previously patched cache is already safe and needs no further writes.
    if legacy_loader not in source and not any(
        reference in source for reference in legacy_references
    ):
        return

    config_import = "from .configuration_dpt import TIPSv2DPTConfig\n"
    head_import = (
        "from .dpt_head import (\n"
        "    DPTDepthHead,\n"
        "    DPTNormalsHead,\n"
        "    DPTSegmentationHead,\n"
        ")\n"
    )
    if config_import not in source:
        raise RuntimeError("Unsupported B/14 DPT modeling_dpt.py import layout")
    if head_import not in source:
        source = source.replace(config_import, config_import + head_import, 1)

    source = source.replace(legacy_loader, "", 1)
    source = source.replace("dpt_mod.DPTDepthHead", "DPTDepthHead")
    source = source.replace("dpt_mod.DPTNormalsHead", "DPTNormalsHead")
    source = source.replace("dpt_mod.DPTSegmentationHead", "DPTSegmentationHead")

    backbone_load = (
        "AutoModel.from_pretrained(self.config.backbone_repo, "
        "trust_remote_code=True)"
    )
    local_backbone_load = (
        "AutoModel.from_pretrained(\n"
        "                self.config.backbone_repo,\n"
        "                trust_remote_code=True,\n"
        "                local_files_only=True,\n"
        "            )"
    )
    if backbone_load in source:
        source = source.replace(backbone_load, local_backbone_load, 1)

    if legacy_loader in source or any(
        reference in source for reference in legacy_references
    ):
        raise RuntimeError("Failed to apply the B/14 DPT compatibility patch")

    modeling_path.write_text(source, encoding="utf-8")
    print("Applied ModelScope-only B/14 DPT compatibility patch.")


def _patch_b14_backbone_snapshot(snapshot_dir):
    """Make every legacy B/14 backbone dependency load from its snapshot."""
    snapshot_dir = Path(snapshot_dir)
    required_files = (
        "modeling_tips.py",
        "image_encoder.py",
        "text_encoder.py",
        "tokenizer.model",
    )
    missing = [name for name in required_files if not (snapshot_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"B/14 ModelScope snapshot is missing: {', '.join(missing)}"
        )

    modeling_path = snapshot_dir / "modeling_tips.py"
    source = modeling_path.read_text(encoding="utf-8")
    legacy_encoder_load = (
        '        repo_id = getattr(config, "_name_or_path", None)\n'
        '        ie = _load_sibling("image_encoder", repo_id)\n'
        '        te = _load_sibling("text_encoder", repo_id)\n\n'
        "        build_fn = getattr(ie, config.vision_fn)\n"
    )
    if legacy_encoder_load not in source:
        return

    config_import = "from .configuration_tips import TIPSv2Config\n"
    local_imports = (
        "from .image_encoder import vit_base as _b14_vision_builder\n"
        "from .text_encoder import TextEncoder, Tokenizer\n"
    )
    if config_import not in source:
        raise RuntimeError("Unsupported B/14 modeling_tips.py import layout")
    source = source.replace(config_import, config_import + local_imports, 1)
    source = source.replace(
        legacy_encoder_load,
        "        build_fn = _b14_vision_builder\n",
        1,
    )
    source = source.replace("te.TextEncoder(", "TextEncoder(", 1)
    source = source.replace("        self._te_mod = te\n", "", 1)

    legacy_tokenizer = (
        "        tok_path = _this_dir / \"tokenizer.model\"\n"
        "        if not tok_path.exists():\n"
        "            tok_path = hf_hub_download(self.name_or_path, "
        "\"tokenizer.model\")\n"
        "        return self._te_mod.Tokenizer(str(tok_path))\n"
    )
    local_tokenizer = (
        "        tok_path = Path(self.config._name_or_path) / "
        "\"tokenizer.model\"\n"
        "        if not tok_path.is_file():\n"
        "            raise FileNotFoundError(\n"
        "                f\"Missing ModelScope tokenizer: {tok_path}\"\n"
        "            )\n"
        "        return Tokenizer(str(tok_path))\n"
    )
    if legacy_tokenizer not in source:
        raise RuntimeError("Unsupported B/14 tokenizer loader layout")
    source = source.replace(legacy_tokenizer, local_tokenizer, 1)

    if '_load_sibling("image_encoder"' in source or "self._te_mod" in source:
        raise RuntimeError("Failed to apply the B/14 backbone compatibility patch")

    modeling_path.write_text(source, encoding="utf-8")
    print("Applied ModelScope-only B/14 backbone compatibility patch.")


def load_variant(name):
    """Load a DPT variant and encoder backbone from ModelScope."""
    global _model
    if _model["name"] == name:
        return
    _release_model()
    repos = VARIANTS[name]
    dpt_dir = _download_model(repos["dpt"])
    backbone_dir = _download_model(repos["backbone"])
    if repos["dpt"] == "google/tipsv2-b14-dpt":
        _patch_b14_dpt_snapshot(dpt_dir)
        _patch_b14_backbone_snapshot(backbone_dir)
    dtype = _model_dtype()
    dpt_config = AutoConfig.from_pretrained(
        dpt_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    # Older DPT code resolves this config separately. Keep that lookup local.
    if hasattr(dpt_config, "backbone_repo"):
        dpt_config.backbone_repo = backbone_dir
    dpt = AutoModel.from_pretrained(
        dpt_dir,
        config=dpt_config,
        trust_remote_code=True,
        # The DPT depth head creates FP32 depth-bin centers internally. Keep
        # this model in FP32 so its einsum does not mix FP16 and FP32 tensors.
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    dpt.eval()
    backbone = AutoModel.from_pretrained(
        backbone_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
        local_files_only=True,
    )
    backbone.eval()
    _enable_sdpa(dpt)
    _enable_sdpa(backbone)
    _model.update(
        name=name,
        dpt=dpt,
        backbone=backbone,
        vision=backbone.vision_encoder,
        text=backbone.text_encoder,
        temperature=getattr(
            backbone.config,
            "temperature",
            getattr(backbone.config, "temperature_init_value", 0.01),
        ),
        ade20k_embs=None,
    )
    print(f"Loaded {name}")


def _move_models_to_device():
    """Move the active models onto the xGPU device."""
    dev = _device()
    if _model["vision"] is not None:
        _model["vision"].to(dev)
    if _model["text"] is not None:
        _model["text"].to(dev)
    if _model["dpt"] is not None:
        _model["dpt"].to(dev)


def _ensure_ade20k_embs():
    """Pre-compute Pascal Context text embeddings if not yet done (must run on GPU)."""
    if _model["ade20k_embs"] is not None:
        return
    backbone = _model["backbone"]
    all_embs = []
    for template in TCL_PROMPTS:
        prompts = [template.format(c) for c in PASCAL_CONTEXT_CLASSES]
        with torch.no_grad():
            embs = backbone.encode_text(prompts)
        all_embs.append(embs.float().cpu().numpy())
    _model["ade20k_embs"] = l2_normalize(np.mean(all_embs, axis=0))
    print("Pascal Context text embeddings computed.")


def _init_model(name=None):
    """Load the selected ModelScope snapshot and prepare it for inference."""
    load_variant(name or _model["name"] or DEFAULT_VARIANT)
    _move_models_to_device()
    _ensure_ade20k_embs()


# ── Preprocessing & helpers ─────────────────────────────────────────────────


def preprocess(img, size=DEFAULT_IMAGE_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ]
    )(img)


def l2_normalize(x, axis=-1):
    return x / np.linalg.norm(x, ord=2, axis=axis, keepdims=True).clip(min=1e-3)


def upsample(arr, h, w, mode="bilinear"):
    """Upsample (H, W, C) or (H, W) numpy array to (h, w, ...)."""
    t = torch.from_numpy(arr).float()
    if t.ndim == 2:
        t = t.unsqueeze(-1)
    t = t.permute(2, 0, 1).unsqueeze(0)
    kwargs = dict(align_corners=False) if mode == "bilinear" else {}
    up = F.interpolate(t, size=(h, w), mode=mode, **kwargs)
    return up[0].permute(1, 2, 0).numpy()


def to_uint8(x):
    return (x * 255).clip(0, 255).astype(np.uint8)


# ── Feature extraction (GPU-accelerated) ────────────────────────────────────


@torch.no_grad()
def extract_features(image_np, resolution=DEFAULT_IMAGE_SIZE):
    """Return spatial features (sp, sp, D) as numpy. sp = resolution // 14."""
    dev = _device()
    img = Image.fromarray(image_np).convert("RGB")
    tensor = preprocess(img, resolution).unsqueeze(0).to(dev, dtype=_model_dtype())
    _, _, patch_tokens = _model["vision"](tensor)
    sp = resolution // PATCH_SIZE
    return patch_tokens.float().cpu().reshape(sp, sp, -1).numpy()


@torch.no_grad()
def extract_features_value_attention(image_np, resolution=ZEROSEG_IMAGE_SIZE):
    """Return spatial features (sp, sp, D) using Value Attention on GPU.

    This follows the Colab reference implementation: run all blocks except the
    last normally, then for the last block extract V from QKV and manually
    apply out_proj, layer scale, residual, norm2, MLP + layer scale, second
    residual, and final norm.
    """
    dev = _device()
    model_image = _model["vision"]
    img = Image.fromarray(image_np).convert("RGB")
    tensor = preprocess(img, resolution).unsqueeze(0).to(dev, dtype=_model_dtype())

    x = model_image.prepare_tokens_with_masks(tensor)

    for blk in model_image.blocks[:-1]:
        x = blk(x)

    blk = model_image.blocks[-1]
    num_reg = getattr(model_image, "num_register_tokens", 1)

    b_dim, n_dim, c_dim = x.shape
    num_heads = blk.attn.num_heads
    qkv = blk.attn.qkv(blk.norm1(x))
    qkv = qkv.reshape(b_dim, n_dim, 3, num_heads, c_dim // num_heads)
    qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D_head)

    v = qkv[2]  # (B, H, N, D_head)
    v_out = v.transpose(1, 2).reshape(b_dim, n_dim, c_dim)
    v_out = blk.attn.proj(v_out)
    v_out = blk.ls1(v_out)
    x_val = v_out + x

    y_val = blk.norm2(x_val)
    y_val = blk.ls2(blk.mlp(y_val))
    x_val = x_val + y_val

    x_val = model_image.norm(x_val)

    patch_tokens = x_val[:, 1 + num_reg :, :]
    sp = resolution // PATCH_SIZE
    spatial = patch_tokens.float().cpu().reshape(sp, sp, -1).numpy()
    return spatial


# ── PCA Visualisations ──────────────────────────────────────────────────────


def vis_pca(spatial):
    """PCA of spatial features → RGB image."""
    feat = spatial.reshape(-1, spatial.shape[-1])
    pca = PCA(n_components=3, whiten=True)
    h, w = spatial.shape[0], spatial.shape[1]
    rgb = pca.fit_transform(feat).reshape(h, w, 3)
    rgb = 1 / (1 + np.exp(-2.0 * rgb))
    return to_uint8(rgb)


def vis_depth(spatial):
    """1st PCA component visualized with inferno colormap."""
    feat = spatial.reshape(-1, spatial.shape[-1])
    h, w = spatial.shape[0], spatial.shape[1]
    depth = PCA(n_components=1).fit_transform(feat).reshape(h, w)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    colored = plt.get_cmap("inferno")(depth)[:, :, :3].astype(np.float32)
    return to_uint8(colored)


def vis_kmeans(spatial, h, w, n_clusters=6):
    """K-means clustering of spatial features."""
    sp_h, sp_w = spatial.shape[:2]
    feat = torch.from_numpy(spatial.reshape(-1, spatial.shape[-1])).to(_device())
    km = TorchKMeans(n_clusters=n_clusters, max_iter=20)
    km.fit(feat)
    dists = -torch.cdist(feat, km.centroids)  # (H*W, k)
    scores = dists.cpu().numpy().reshape(sp_h, sp_w, n_clusters)
    scores_up = upsample(scores, h, w, mode="bilinear")
    labels = scores_up.argmax(axis=-1)
    palette = plt.cm.tab20(np.linspace(0, 1, n_clusters))[:, :3]
    seg = palette[labels].astype(np.float32)
    return to_uint8(seg)


# ── Zero-shot Segmentation ──────────────────────────────────────────────────


def vis_custom_semseg(spatial, orig_image, classes, class_embs):
    """Zero-shot semantic segmentation with user-defined classes."""
    h, w = orig_image.shape[:2]
    sp_h, sp_w = spatial.shape[:2]
    n = len(classes)

    feat = l2_normalize(spatial.reshape(-1, spatial.shape[-1]))
    sim = feat @ class_embs.T
    sim_map = sim.reshape(sp_h, sp_w, n)

    sim_up = upsample(sim_map, h, w, mode="bilinear")
    labels = sim_up.argmax(axis=-1)

    palette = (plt.cm.tab20(np.linspace(0, 1, max(n, 2)))[:n, :3] * 255).astype(
        np.uint8
    )

    seg_rgb = palette[labels].astype(np.float32) / 255.0
    mask_img = to_uint8(seg_rgb)

    blend = 0.1 * orig_image.astype(np.float32) / 255.0 + 0.9 * seg_rgb
    blend_img = Image.fromarray(to_uint8(blend))

    unique_ids, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    unique_ids, counts = unique_ids[order], counts[order]
    total = counts.sum()

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            60,
        )
    except OSError:
        font = ImageFont.load_default()

    n_legend = min(len(unique_ids), 10)
    row_h = 80
    swatch_w = 60
    pad = 12
    legend_w = 450

    legend_h = max(h, n_legend * row_h + pad * 2)
    canvas = Image.new("RGB", (w + legend_w, legend_h), (255, 255, 255))
    canvas.paste(blend_img, (0, 0))
    draw = ImageDraw.Draw(canvas)

    for i in range(n_legend):
        cid = unique_ids[i]
        color = tuple(palette[cid].tolist())
        y_top = pad + i * row_h
        draw.rectangle(
            [w + pad, y_top, w + pad + swatch_w, y_top + swatch_w],
            fill=color,
            outline=(0, 0, 0),
        )
        draw.text(
            (w + pad + swatch_w + 8, y_top + 6),
            classes[cid],
            fill="black",
            font=font,
        )

    overlay_out = np.array(canvas)

    detected_parts, minor_parts = [], []
    for i, cid in enumerate(unique_ids):
        pct = counts[i] / total * 100
        if pct >= 2:
            detected_parts.append(f"{classes[cid]} ({pct:.1f}%)")
        else:
            minor_parts.append(f"{classes[cid]} ({pct:.1f}%)")
    absent = [
        f"{classes[i]} (0.0%)" for i in range(n) if i not in set(unique_ids.tolist())
    ]
    detected_str = ", ".join(detected_parts)
    undetected_str = ", ".join(minor_parts + absent)
    return overlay_out, mask_img, detected_str, undetected_str


# ── DPT Depth Inference ─────────────────────────────────────────────────────


def vis_depth_dpt(depth_map, h, w):
    """Colour a depth map with the turbo colormap → PIL Image."""
    d = depth_map.squeeze()
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    colored = plt.get_cmap("turbo")(d)[:, :, :3].astype(np.float32)
    return to_uint8(upsample(colored, h, w))


def vis_normals_dpt(normals_map, h, w):
    """Map normals from [-1, 1] to [0, 1] and resize to original size."""
    n = normals_map.float().cpu().numpy()
    n = (n + 1.0) / 2.0
    n = np.transpose(n, (1, 2, 0))  # (H, W, 3)
    return to_uint8(upsample(n, h, w))


def vis_segmentation_dpt(seg_map, orig_image):
    """Colour a segmentation map with the ADE20K colormap + legend."""
    h, w = orig_image.shape[:2]
    logits = seg_map.float().cpu().numpy().transpose(1, 2, 0)  # (H, W, 150)
    logits_up = upsample(logits, h, w, mode="bilinear")
    pred = logits_up.argmax(axis=-1)  # (h, w)
    seg_rgb = ADE20K_PALETTE[pred.astype(np.int32) + 1].astype(np.float32) / 255.0

    blend = 0.15 * orig_image.astype(np.float32) / 255.0 + 0.85 * seg_rgb
    blend_img = Image.fromarray(to_uint8(blend))

    unique_ids, counts = np.unique(pred, return_counts=True)
    total_pixels = counts.sum()
    order = np.argsort(-counts)
    unique_ids, counts = unique_ids[order], counts[order]

    pcts = counts / total_pixels * 100
    mask = pcts >= 2.0
    unique_ids, counts, pcts = unique_ids[mask], counts[mask], pcts[mask]

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            36,
        )
    except OSError:
        font = ImageFont.load_default()

    n_legend = min(len(unique_ids), 10)
    row_h, swatch_w, pad, legend_w = 50, 40, 10, 450
    legend_h = max(h, n_legend * row_h + pad * 2)
    canvas = Image.new("RGB", (w + legend_w, legend_h), (255, 255, 255))
    canvas.paste(blend_img, (0, 0))
    draw = ImageDraw.Draw(canvas)

    for i in range(n_legend):
        cid = unique_ids[i]
        color = tuple(ADE20K_PALETTE[cid + 1].tolist())
        name = ADE20K_CLASSES[cid] if cid < len(ADE20K_CLASSES) else f"class_{cid}"
        y_top = pad + i * row_h
        draw.rectangle(
            [w + pad, y_top, w + pad + swatch_w, y_top + swatch_w],
            fill=color,
            outline=(0, 0, 0),
        )
        draw.text(
            (w + pad + swatch_w + 8, y_top + 4),
            name,
            fill="black",
            font=font,
        )

    return np.array(canvas)


# ── Gradio callbacks ────────────────────────────────────────────────────────


def on_variant_change(variant_name):  # noqa: ARG001
    # Only clear stale outputs. Loading remains lazy so the Studio can start
    # without downloading every model variant.
    return (
        None,
        None,
        None,  # pca_out, depth_out, kmeans_out
        None,  # pca_state
        None,
        None,
        "",
        "",  # custom outputs
    )


def on_pca_extract(image, variant, resolution, _pca_state):
    if image is None:
        return None, None, None, None
    _init_model(variant)
    resolution = int(resolution)
    spatial = extract_features(image, resolution)
    h, w = image.shape[:2]
    pca = vis_pca(spatial)
    depth = vis_depth(spatial)
    kmeans = vis_kmeans(spatial, h, w)
    state = {
        "spatial": spatial,
        "orig_image": image,
        "variant": _model["name"],
        "resolution": resolution,
    }
    return pca, depth, kmeans, state


def on_recluster(image, variant, resolution, n_clusters, pca_state):
    if image is None:
        gr.Warning("请先上传图像。")
        return None, pca_state
    _init_model(variant)
    resolution = int(resolution)
    if (
        pca_state is not None
        and pca_state.get("variant") == _model["name"]
        and pca_state.get("resolution") == resolution
    ):
        spatial = pca_state["spatial"]
    else:
        spatial = extract_features(image, resolution)
        pca_state = {
            "spatial": spatial,
            "orig_image": image,
            "variant": _model["name"],
            "resolution": resolution,
        }
    h, w = image.shape[:2]
    return vis_kmeans(spatial, h, w, int(n_clusters)), pca_state


def on_zeroseg_custom(image, variant, resolution, class_names_str):
    if image is None or not class_names_str or not class_names_str.strip():
        gr.Warning("请上传图像，并至少输入一个类别名称。")
        return None, None, "", ""
    _init_model(variant)
    resolution = int(resolution)
    classes = [c.strip() for c in class_names_str.split(",") if c.strip()]
    if not classes:
        return None, None, "", ""

    all_embs = []
    for template in TCL_PROMPTS:
        prompts = [template.format(c) for c in classes]
        with torch.no_grad():
            embs = _model["backbone"].encode_text(prompts)
        all_embs.append(embs.float().cpu().numpy())
    class_embs = l2_normalize(np.mean(all_embs, axis=0))

    spatial = extract_features_value_attention(image, resolution)
    overlay, mask, detected, undetected = vis_custom_semseg(
        spatial,
        image,
        classes,
        class_embs,
    )
    return overlay, mask, detected, undetected


def on_depth_normals_predict(image, dpt_variant, resolution):
    """Run DPT depth and normals prediction."""
    if image is None:
        return None, None
    _init_model(dpt_variant)
    dev = _device()

    h, w = image.shape[:2]
    img = Image.fromarray(image).convert("RGB")
    tensor = preprocess(img, int(resolution)).unsqueeze(0).to(
        dev,
        dtype=torch.float32,
    )

    depth_map = _model["dpt"].predict_depth(tensor)
    normals_map = _model["dpt"].predict_normals(tensor)

    return (
        vis_depth_dpt(depth_map[0, 0].float().cpu().numpy(), h, w),
        vis_normals_dpt(normals_map[0], h, w),
    )


def on_segmentation_predict(image, dpt_variant, resolution):
    """Run DPT segmentation prediction."""
    if image is None:
        return None
    _init_model(dpt_variant)
    dev = _device()

    img = Image.fromarray(image).convert("RGB")
    tensor = preprocess(img, int(resolution)).unsqueeze(0).to(
        dev,
        dtype=torch.float32,
    )

    seg_map = _model["dpt"].predict_segmentation(tensor)
    return vis_segmentation_dpt(seg_map[0], image)


# ── UI ──────────────────────────────────────────────────────────────────────

custom_css = """
#pca_output_image img, #depth_output_image img {
    image-rendering: pixelated;
    object-fit: contain;
}
"""

with gr.Blocks(title="TIPSv2 Feature Explorer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## TIPSv2 Feature Explorer")

    with gr.Row():
        variant_dd = gr.Dropdown(
            choices=list(VARIANTS.keys()),
            value=DEFAULT_VARIANT,
            label="模型规格",
        )
        resolution_dd = gr.Dropdown(
            choices=RESOLUTIONS,
            value=DEFAULT_IMAGE_SIZE,
            label="输入分辨率（越高细节越丰富，但速度越慢）",
        )

    # ── PCA / Feature Visualization Tab ─────────────────────────────────
    with gr.Tab("🎨 PCA 与特征可视化"):
        pca_state = gr.State(None)

        with gr.Row():
            with gr.Column():
                pca_input = gr.Image(type="numpy", label="输入图像")
                pca_btn = gr.Button("提取特征", variant="primary")

            with gr.Column():
                with gr.Tabs():
                    with gr.Tab("PCA"):
                        pca_out = gr.Image(
                            label="PCA（3 个主成分 → RGB）",
                            height=448,
                            elem_id="pca_output_image",
                        )
                    with gr.Tab("PCA（第一主成分）"):
                        depth_out = gr.Image(
                            label="PCA 第一主成分",
                            height=448,
                            elem_id="depth_output_image",
                        )
                    with gr.Tab("K-means 聚类"):
                        n_clusters = gr.Slider(
                            2,
                            20,
                            value=6,
                            step=1,
                            label="聚类数量",
                        )
                        recluster_btn = gr.Button("重新聚类")
                        kmeans_out = gr.Image(label="K-means 聚类结果")

        gr.Markdown("上传图像，探索其空间特征。")

    # ── Zero-shot Segmentation Tab ──────────────────────────────────────
    with gr.Tab("✏️ 零样本分割"):
        gr.Markdown(
            "自定义希望识别的类别，使用英文逗号分隔。"
            "为获得更稳定的结果，建议输入英文类别名称。",
        )

        with gr.Row():
            with gr.Column():
                custom_input = gr.Image(type="numpy", label="输入图像", height=448)
                custom_classes = gr.Textbox(
                    label="类别名称（使用英文逗号分隔）",
                    value="class1, class2, class3",
                    placeholder="例如：cat, dog, sky, grass",
                )
                custom_btn = gr.Button("开始分割", variant="primary")

            with gr.Column():
                with gr.Tabs():
                    with gr.Tab("叠加图"):
                        custom_overlay = gr.Image(
                            label="分割叠加图",
                            height=448,
                        )
                    with gr.Tab("分割掩码"):
                        custom_mask = gr.Image(
                            label="分割掩码",
                            height=448,
                        )
                custom_detected = gr.Textbox(
                    label="已检出类别（按面积排序）",
                    lines=2,
                )
                custom_undetected = gr.Textbox(label="未检出类别", lines=2)

        gr.Markdown("上传图像，并输入使用英文逗号分隔的类别名称。")

    # ── Depth/Normals Visualization Tab ─────────────────────────────────
    with gr.Tab("🏔️ 深度与表面法线"):
        gr.Markdown(
            "基于冻结的 **TIPSv2 视觉编码器**和 **DPT（密集预测 Transformer）**，"
            "进行单目深度与表面法线估计。模型在 **NYU Depth V2** 数据集上训练。",
        )

        with gr.Row():
            with gr.Column():
                depth_input = gr.Image(type="numpy", label="输入图像", height=448)
                depth_btn = gr.Button("预测深度与法线", variant="primary")

            with gr.Column():
                dpt_depth_out = gr.Image(label="DPT 深度图", height=448)

            with gr.Column():
                dpt_normals_out = gr.Image(
                    label="DPT 表面法线",
                    height=448,
                )

        gr.Markdown("上传图像，预测单目深度和表面法线。")

    # ── Supervised Segmentation Tab ──────────────────────────────────────
    with gr.Tab("🎭 监督式语义分割"):
        gr.Markdown(
            "基于冻结的 **TIPSv2 视觉编码器**和 **DPT（密集预测 Transformer）**，"
            "进行 ADE20K 语义分割，共支持 150 个类别。",
        )

        with gr.Row():
            with gr.Column():
                seg_input = gr.Image(type="numpy", label="输入图像", height=448)
                seg_btn = gr.Button("开始分割", variant="primary")

            with gr.Column():
                seg_out = gr.Image(label="DPT 语义分割结果（ADE20K）", height=448)

        gr.Markdown("上传图像，运行 ADE20K 语义分割。")

    # ── Wiring ──────────────────────────────────────────────────────────

    variant_dd.change(
        fn=on_variant_change,
        inputs=[variant_dd],
        outputs=[
            pca_out,
            depth_out,
            kmeans_out,
            pca_state,
            custom_overlay,
            custom_mask,
            custom_detected,
            custom_undetected,
        ],
    )

    pca_btn.click(
        fn=on_pca_extract,
        inputs=[pca_input, variant_dd, resolution_dd, pca_state],
        outputs=[pca_out, depth_out, kmeans_out, pca_state],
    )
    recluster_btn.click(
        fn=on_recluster,
        inputs=[pca_input, variant_dd, resolution_dd, n_clusters, pca_state],
        outputs=[kmeans_out, pca_state],
    )

    depth_btn.click(
        fn=on_depth_normals_predict,
        inputs=[depth_input, variant_dd, resolution_dd],
        outputs=[dpt_depth_out, dpt_normals_out],
    )

    seg_btn.click(
        fn=on_segmentation_predict,
        inputs=[seg_input, variant_dd, resolution_dd],
        outputs=[seg_out],
    )

    custom_btn.click(
        fn=on_zeroseg_custom,
        inputs=[custom_input, variant_dd, resolution_dd, custom_classes],
        outputs=[custom_overlay, custom_mask, custom_detected, custom_undetected],
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        css=custom_css,
    )
