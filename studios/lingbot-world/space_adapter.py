"""HF Space 的输入与相机适配；模型始终走现有魔搭独立 worker。"""
import importlib.util
import json
import math
import secrets
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORDED_PATHS = {
    "Recorded · dragon flight (official)": "00",
    "Recorded · Stonehenge sweep (official)": "01",
    "Recorded · city walk (official)": "02",
    "Recorded · lakeside drift (official)": "03",
    "Recorded · Great Wall walk (official)": "04",
}
CAMERA_CHOICES = [
    "Dolly forward", "Dolly backward", "Truck left", "Truck right", "Crane up", "Crane down",
    "Pan left", "Pan right", "Tilt up", "Tilt down", "Orbit left", "Orbit right",
    "Fly forward, pan left", "Fly forward, pan right", "Hold still", *RECORDED_PATHS,
]
CAMERA_LABELS = [
    "向前推进", "向后拉远", "向左平移", "向右平移", "向上升起", "向下降低",
    "向左摇镜", "向右摇镜", "向上俯仰", "向下俯仰", "向左环绕", "向右环绕",
    "前进并左转", "前进并右转", "镜头静止", "录制轨迹 · 巨龙飞行", "录制轨迹 · 巨石阵",
    "录制轨迹 · 城市漫步", "录制轨迹 · 湖畔", "录制轨迹 · 长城",
]
REQUEST_KEYS = {"prompt", "camera_motion", "num_frames", "turn_rate", "seed"}
PUBLIC_MAX_FRAMES = 81


def integer(value, name, minimum, maximum):
    if type(value) not in (int, float) or not math.isfinite(value) or int(value) != value or not minimum <= value <= maximum:
        raise ValueError(f"{name}必须为 {minimum}～{maximum} 的整数。")
    return int(value)


def validate_parameters(prompt, camera_motion, num_frames, turn_rate, seed, randomize_seed=False, *, maximum=PUBLIC_MAX_FRAMES):
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2000:
        raise ValueError("请填写 1～2000 字符的场景提示词。")
    if not isinstance(camera_motion, str) or camera_motion not in CAMERA_CHOICES:
        raise ValueError("请选择预设镜头轨迹。")
    frames = integer(num_frames, "帧数", 45, maximum)
    if (frames - 45) % 12:
        raise ValueError("帧数须从45起、每次增加12。")
    if type(turn_rate) not in (int, float) or not math.isfinite(turn_rate) or not 1 <= turn_rate <= 30:
        raise ValueError("转向速度必须为 1～30°/s。")
    seed = integer(seed, "Seed", 0, 2**31 - 1)
    if type(randomize_seed) is not bool:
        raise ValueError("随机种子选项必须为布尔值。")
    if randomize_seed:
        seed = secrets.randbelow(2**31)
    return {"prompt": prompt.strip(), "camera_motion": camera_motion, "num_frames": frames,
            "turn_rate": float(turn_rate), "seed": seed}


def normalize_image(image):
    from PIL import Image, ImageOps
    if not isinstance(image, Image.Image):
        raise ValueError("请上传图片或选择示例。")
    width, height = image.size
    if min(width, height) < 64 or width * height > 16 * 1024 * 1024 or not 0.25 <= width / height <= 4:
        raise ValueError("图片短边至少64像素、至多1600万像素，宽高比须在1:4至4:1之间。")
    image = ImageOps.exif_transpose(image).convert("RGB")
    if max(image.size) > 1280:
        scale = 1280 / max(image.size)
        image = image.resize(tuple(round(size * scale) for size in image.size), Image.Resampling.LANCZOS)
    # 新建像素对象，避免将上传文件的元数据带入任务目录。
    return Image.frombytes("RGB", image.size, image.tobytes())


def read_request(directory, *, maximum=PUBLIC_MAX_FRAMES):
    path = Path(directory) / "request.json"
    if path.is_symlink() or path.stat().st_size > 16384:
        raise ValueError("任务参数文件无效")
    request = json.loads(path.read_text())
    if not isinstance(request, dict) or set(request) != REQUEST_KEYS:
        raise ValueError("任务参数字段无效")
    return validate_parameters(**request, maximum=maximum)


def prepare_input(source, directory, *, maximum=PUBLIC_MAX_FRAMES):
    """再次验证参数；只执行已校验离线包中的相机数学，不调用其临时目录函数。"""
    import numpy as np
    from PIL import Image
    request = read_request(directory, maximum=maximum)
    image_path = directory / "input.png"
    if image_path.is_symlink() or image_path.stat().st_size > 12 * 1024 * 1024:
        raise ValueError("任务图片无效")
    with Image.open(image_path) as uploaded:
        image = normalize_image(uploaded)
    spec = importlib.util.spec_from_file_location("lingbot_space_camera", source / "camera.py")
    camera = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(camera)
    if camera.CAMERA_CHOICES != CAMERA_CHOICES:
        raise ValueError("相机选项与固定上游不一致")
    motion, frames = request["camera_motion"], request["num_frames"]
    action = directory / "camera"
    action.mkdir(exist_ok=False)
    if motion in RECORDED_PATHS:
        sample = source / "examples" / RECORDED_PATHS[motion]
        poses = np.load(sample / "poses.npy", allow_pickle=False)
        intrinsics = np.load(sample / "intrinsics.npy", allow_pickle=False)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4) or intrinsics.ndim != 2 or intrinsics.shape[1] != 4:
            raise ValueError("录制相机数据形状无效")
        frames = min(frames, len(poses), len(intrinsics))
        frames = 45 + ((frames - 45) // 12) * 12
        if frames < 45:
            raise ValueError("录制轨迹不足45帧")
        poses, intrinsics = poses[:frames], intrinsics[:frames]
    else:
        poses = camera.synthesize_poses(motion, frames, request["turn_rate"])
        intrinsics = np.tile(camera.DEFAULT_INTRINSICS, (frames, 1))
    if not np.isfinite(poses).all() or not np.isfinite(intrinsics).all():
        raise ValueError("相机数据含非有限值")
    np.save(action / "poses.npy", poses)
    np.save(action / "intrinsics.npy", intrinsics)
    workload = json.loads((HERE / "space_workload.json").read_text())
    workload.update(frame_num=frames, seed=request["seed"], camera_motion=motion,
                    turn_rate=request["turn_rate"], requested_frames=request["num_frames"])
    return request["prompt"], image, action, workload


def load_examples(media_root):
    """页面启动仅校验和展开小型离线资产，不下载权重、不触发 GPU。"""
    import tempfile
    from PIL import Image, ImageOps
    from prepare_validation import prepare_source
    source = json.loads((HERE / "space_source.json").read_text())
    target = Path(media_root) / "examples-792aa925"
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="lingbot-examples-") as private:
        extracted = prepare_source(source, private_root=Path(private))
        for index in ("00", "01", "02", "05", "03", "04"):
            sample = extracted / "examples" / index
            filename = f"{index}.jpg"
            image = target / filename
            if image.is_symlink():
                raise ValueError("示例媒体不能为链接")
            # 仅对通过固定源码包校验的内置图片先缩图，上传图片仍执行原有像素上限。
            with Image.open(sample / "image.jpg") as original:
                preview = ImageOps.exif_transpose(original)
                preview.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                normalize_image(preview).save(image, format="JPEG", quality=95)
            prompt = (sample / "prompt.txt").read_text().strip() if index in ("00", "01", "02", "05") else {
                "03": "A lone willow tree standing in the shallow water of an alpine lake, snow-capped mountains behind it and bright cumulus clouds drifting across a deep blue sky.",
                "04": "A first-person walk along the stone battlements of the Great Wall of China, watchtowers climbing the forested ridge under clear autumn light.",
            }[index]
            motion = next((name for name, value in RECORDED_PATHS.items() if value == index), "Orbit left")
            rows.append([str(image), prompt, motion])
    return rows
