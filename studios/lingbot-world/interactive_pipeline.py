"""固定原生 causal_fast 的有界步进适配；不改上游源码和权重。"""
import hashlib
import math

KEYS = frozenset("wasdijkl")
MAX_CHUNKS = 32


def checked_keys(value):
    if not isinstance(value, list) or len(value) > 8 or any(type(k) is not str or k not in KEYS for k in value):
        raise ValueError("按键列表无效")
    return sorted(set(value))


class CameraState:
    """OpenCV坐标；跨chunk保留绝对位姿，只输出相邻帧相对运动。"""
    def __init__(self):
        import numpy as np
        self.pose = np.eye(4, dtype=np.float32)
        self.yaw = self.pitch = 0.0
        self.frames = 0

    def advance(self, keys, count=3):
        import numpy as np
        keys = checked_keys(keys)
        dx = int("d" in keys) - int("a" in keys)
        dz = int("w" in keys) - int("s" in keys)
        dyaw = (int("l" in keys) - int("j" in keys)) * math.radians(2)
        dpitch = (int("i" in keys) - int("k" in keys)) * math.radians(2)
        delta = np.array([dx, 0, dz], dtype=np.float32)
        delta /= max(1.0, float(np.linalg.norm(delta)))
        result = []
        for _ in range(count):
            previous = self.pose.copy()
            if self.frames:
                self.yaw += dyaw
                self.pitch = max(-math.pi / 3, min(math.pi / 3, self.pitch + dpitch))
                cy, sy, cp, sp = math.cos(self.yaw), math.sin(self.yaw), math.cos(self.pitch), math.sin(self.pitch)
                rotation = np.array([[cy, sy * sp, sy * cp], [0, cp, -sp], [-sy, cy * sp, cy * cp]], dtype=np.float32)
                self.pose[:3, 3] += previous[:3, :3] @ delta
                self.pose[:3, :3] = rotation
            relative = np.eye(4, dtype=np.float32)
            relative[:3, :3] = previous[:3, :3].T @ self.pose[:3, :3]
            relative[:3, 3] = previous[:3, :3].T @ (self.pose[:3, 3] - previous[:3, 3])
            result.append(relative)
            self.frames += 1
        return np.stack(result)


class StreamingDecoder:
    def __init__(self, vae):
        self.vae = vae
        self.vae.model.clear_cache()
        self.frames = 0
        self.cache_signature = None

    def decode(self, latent):
        import torch
        model, scale = self.vae.model, self.vae.scale
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=self.vae.dtype):
            z = latent.unsqueeze(0)
            z = z / scale[1].view(1, model.z_dim, 1, 1, 1) + scale[0].view(1, model.z_dim, 1, 1, 1)
            x = model.conv2(z)
            frames = []
            for i in range(x.shape[2]):
                model._conv_idx = [0]
                frames.append(model.decoder(x[:, :, i:i + 1], feat_cache=model._feat_map, feat_idx=model._conv_idx))
            video = torch.cat(frames, dim=2).float().clamp_(-1, 1).squeeze(0)
        expected = latent.shape[1] * 4 - (3 if self.frames == 0 else 0)
        if video.shape[1] != expected or not torch.isfinite(video).all():
            raise RuntimeError("增量VAE帧数或有限值校验失败")
        signature = self.cache_shapes()
        if self.cache_signature is not None and signature != self.cache_signature:
            raise RuntimeError("VAE缓存尺寸发生变化")
        self.cache_signature = signature
        self.frames += video.shape[1]
        return video

    def cache_shapes(self):
        import torch
        result = []
        for item in self.vae.model._feat_map:
            if torch.is_tensor(item):
                if item.ndim != 5 or item.shape[2] > 2:
                    raise RuntimeError("VAE缓存超过原生时间窗口")
                result.append(tuple(item.shape))
            elif item is None or isinstance(item, str) and item == "Rep":
                result.append(None)
            else:
                raise RuntimeError("VAE缓存格式错误")
        return result

    def close(self):
        self.vae.model.clear_cache()


class InteractivePipeline:
    def __init__(self, pipe, image, prompt, intrinsics, chunks=MAX_CHUNKS, seed=42):
        import numpy as np
        import torch
        import torchvision.transforms.functional as TF
        from wan.utils.cam_utils import get_Ks_transformed
        if type(chunks) is not int or not 2 <= chunks <= MAX_CHUNKS:
            raise ValueError("片段数量越界")
        if pipe.local_attn_size != 18 or pipe.sink_size != 6 or pipe.sp_size != 1:
            raise ValueError("交互适配仅支持固定单卡KV窗口")
        self.pipe, self.chunks, self.chunk = pipe, chunks, 0
        self.camera = CameraState()
        self.chunk_size = 3
        self.lat_f = chunks * 3
        self.frame_count = (self.lat_f - 1) * 4 + 1
        image = TF.to_tensor(image).sub_(0.5).div_(0.5).to(pipe.device)
        ratio = image.shape[1] / image.shape[2]
        self.lat_h = round(np.sqrt(416 * 240 * ratio) // pipe.vae_stride[1] // pipe.patch_size[1] * pipe.patch_size[1])
        self.lat_w = round(np.sqrt(416 * 240 / ratio) // pipe.vae_stride[2] // pipe.patch_size[2] * pipe.patch_size[2])
        self.h, self.w = self.lat_h * pipe.vae_stride[1], self.lat_w * pipe.vae_stride[2]
        self.frame_seqlen = self.lat_h * self.lat_w // 4
        self.rng = torch.Generator(device=pipe.device).manual_seed(seed)
        self.noise = torch.randn(16, self.lat_f, self.lat_h, self.lat_w, generator=self.rng, device=pipe.device, dtype=torch.float32)
        pipe.scheduler.set_timesteps(pipe.num_train_timesteps, shift=5.0)
        self.timesteps = pipe.scheduler.timesteps[[0, 250, 500, 750]]
        key = hashlib.sha256(prompt.encode()).hexdigest()
        if key not in pipe._t5_cache:
            pipe.text_encoder.model.to(pipe.device)
            pipe._t5_cache[key] = pipe.text_encoder([prompt], pipe.device)
        self.context = pipe._t5_cache[key]
        self.intrinsics = get_Ks_transformed(torch.as_tensor(intrinsics).float(), 480, 832, self.h, self.w, self.h, self.w)[0].to(pipe.device)
        mask = torch.ones(1, self.frame_count, self.lat_h, self.lat_w, device=pipe.device)
        mask[:, 1:] = 0
        mask = torch.cat([torch.repeat_interleave(mask[:, :1], 4, dim=1), mask[:, 1:]], dim=1)
        mask = mask.view(1, self.lat_f, 4, self.lat_h, self.lat_w).transpose(1, 2)[0]
        with torch.no_grad():
            first = torch.nn.functional.interpolate(image[None].cpu(), size=(self.h, self.w), mode="bicubic").transpose(0, 1)
            encoded = pipe.vae.encode([torch.cat([first, torch.zeros(3, self.frame_count - 1, self.h, self.w)], dim=1).to(pipe.device)])[0]
        self.condition = torch.cat([mask, encoded])
        if not torch.isfinite(self.condition).all() or not torch.isfinite(self.noise).all():
            raise RuntimeError("初始化条件或噪声非有限值")
        config = pipe.model.config
        self.kv_size = self.frame_seqlen * pipe.local_attn_size
        self.kv = pipe._initialize_self_kv_cache(config.num_layers, [1, self.kv_size, config.num_heads, config.dim // config.num_heads], pipe.pipe_dtype, pipe.device)
        self.cross = pipe._initialize_crossattn_cache(config.num_layers, [1, 512, config.num_heads, config.dim // config.num_heads], pipe.pipe_dtype, pipe.device)
        self.cross_initialized = False
        self.decoder = StreamingDecoder(pipe.vae)
        self.initial_cache_shapes = self.cache_shapes()

    def cache_shapes(self):
        return [(tuple(x["k"].shape), tuple(x["v"].shape)) for x in self.kv + self.cross]

    def camera_embedding(self, poses):
        import torch
        from einops import rearrange
        from wan.utils.cam_utils import get_plucker_embeddings
        poses = torch.as_tensor(poses, device=self.pipe.device, dtype=torch.float32)
        rays = get_plucker_embeddings(poses, self.intrinsics.repeat(3, 1), self.h, self.w)
        rays = rearrange(rays, "f (h c1) (w c2) c -> (f h w) (c c1 c2)", c1=self.h // self.lat_h, c2=self.w // self.lat_w)[None]
        return rearrange(rays, "b (f h w) c -> b c f h w", f=3, h=self.lat_h, w=self.lat_w).to(self.pipe.param_dtype)

    def step(self, keys, poses=None):
        import torch
        if self.chunk >= self.chunks:
            raise RuntimeError("会话已达到片段上限")
        keys = checked_keys(keys)
        pipe, start = self.pipe, self.chunk * 3
        camera = self.camera_embedding(self.camera.advance(keys) if poses is None else poses)
        current = self.noise[:, start:start + 3]
        kwargs = dict(context=[self.context[0]], seq_len=3 * self.frame_seqlen,
                      y=[self.condition[:, start:start + 3]], dit_cond_dict={"c2ws_plucker_emb": camera.chunk(1, dim=0)},
                      kv_cache=self.kv, crossattn_cache=self.cross, current_start=start * self.frame_seqlen,
                      max_attention_size=self.kv_size, frame_seqlen=self.frame_seqlen)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=pipe.param_dtype):
            for index, timestep in enumerate(self.timesteps):
                prediction = pipe.model(x=[current.to(pipe.device)], t=torch.stack([timestep]).to(pipe.device),
                                        cross_attn_first_call=not self.cross_initialized, **kwargs)[0]
                if not torch.isfinite(prediction).all():
                    raise RuntimeError("DiT输出非有限值")
                self.cross_initialized = True
                x0 = pipe._convert_flow_pred_to_x0(prediction, current, timestep, pipe.scheduler)
                if index < len(self.timesteps) - 1:
                    current = pipe.scheduler.add_noise(x0, torch.randn(x0.shape, generator=self.rng, device=x0.device, dtype=x0.dtype), self.timesteps[index + 1])
            result = pipe.model(x=[x0], t=torch.stack([self.timesteps[-1] * 0.0]).to(pipe.device), cross_attn_first_call=False, **kwargs)
            if not all(torch.isfinite(x).all() for x in result) or not torch.isfinite(x0).all():
                raise RuntimeError("缓存更新输出非有限值")
        if self.cache_shapes() != self.initial_cache_shapes:
            raise RuntimeError("KV缓存尺寸发生变化")
        video = self.decoder.decode(x0)
        if video.shape[0] != 3 or tuple(video.shape[2:]) != (self.h, self.w):
            raise RuntimeError("交互视频尺寸错误")
        self.chunk += 1
        return video, x0

    def close(self):
        self.decoder.close()
        self.kv.clear()
        self.cross.clear()
        self.noise = self.condition = self.context = None


def verify_equivalence(pipe, image, prompt, sample):
    """在真实GPU中用同seed对照固定上游两chunk；失败不得开始键盘会话。"""
    import numpy as np
    import torch
    from wan.utils.cam_utils import interpolate_camera_poses, compute_relative_poses
    reference_latents, reference_conditions, reference_camera, reference_inputs = [], [], [], []
    original = pipe.model.forward
    owned = "forward" in pipe.model.__dict__
    calls = 0

    def capture(*args, **kwargs):
        nonlocal calls
        calls += 1
        reference_inputs.append((kwargs["x"][0].detach().cpu().clone(), kwargs["t"].detach().cpu().clone(), kwargs["current_start"]))
        if calls % 5 == 0:
            reference_latents.append(kwargs["x"][0].detach().cpu().clone())
            reference_conditions.append(kwargs["y"][0].detach().cpu().clone())
            reference_camera.append(kwargs["dit_cond_dict"]["c2ws_plucker_emb"][0].detach().cpu().clone())
        result = original(*args, **kwargs)
        if not all(torch.isfinite(item).all() for item in result):
            raise RuntimeError("原生对照forward非有限值")
        return result

    pipe.model.forward = capture
    try:
        reference = pipe.generate(input_prompt=prompt, img=image, action_path=str(sample), chunk_size=3,
                                  max_area=416 * 240, frame_num=21, timesteps_index=[0, 250, 500, 750],
                                  shift=5.0, seed=42, offload_model=False)
    finally:
        if owned:
            pipe.model.forward = original
        else:
            del pipe.model.forward
    if calls != 10 or len(reference_latents) != 2:
        raise RuntimeError("上游forward调用契约不匹配")
    poses = np.load(sample / "poses.npy")[:21]
    relative = compute_relative_poses(interpolate_camera_poses(np.arange(21), poses[:, :3, :3], poses[:, :3, 3], np.linspace(0, 20, 6)), framewise=True)
    session = InteractivePipeline(pipe, image, prompt, np.load(sample / "intrinsics.npy"), chunks=2)
    videos, latent_error, compared_calls = [], 0.0, 0

    def compare_inputs(*args, **kwargs):
        nonlocal compared_calls
        expected_x, expected_t, expected_start = reference_inputs[compared_calls]
        if (kwargs["current_start"] != expected_start
                or kwargs["t"].dtype != expected_t.dtype
                or not torch.equal(kwargs["t"].cpu(), expected_t)
                or not torch.allclose(kwargs["x"][0].cpu(), expected_x, atol=1e-5, rtol=1e-5)):
            raise RuntimeError("预生成噪声、调度RNG或全局位置对照失败")
        compared_calls += 1
        return original(*args, **kwargs)

    pipe.model.forward = compare_inputs
    try:
        for index in range(2):
            if not torch.allclose(session.condition[:, index * 3:(index + 1) * 3].cpu(), reference_conditions[index], atol=1e-5, rtol=1e-5):
                raise RuntimeError("I2V条件对照失败")
            if not torch.allclose(session.camera_embedding(relative[index * 3:(index + 1) * 3])[0].cpu(), reference_camera[index], atol=1e-5, rtol=1e-5):
                raise RuntimeError("相机条件对照失败")
            video, latent = session.step([], relative[index * 3:(index + 1) * 3])
            latent_error = max(latent_error, (latent.cpu() - reference_latents[index]).abs().max().item())
            if not torch.allclose(latent.cpu(), reference_latents[index], atol=1e-5, rtol=1e-5):
                raise RuntimeError("原生步进latent对照失败")
            videos.append(video)
        joined = torch.cat(videos, dim=1)
        error = (joined - reference).abs().max().item() if joined.shape == reference.shape else float("inf")
        if compared_calls != 10 or not math.isfinite(error) or error > 1e-3:
            raise RuntimeError("增量VAE与原生视频对照失败")
        return {"passed": True, "latent_max_error": latent_error, "video_max_error": error, "frames": joined.shape[1]}
    finally:
        if owned:
            pipe.model.forward = original
        else:
            del pipe.model.forward
        session.close()
