---
domain:
  - cv
license: Apache License 2.0
models:
  - TencentARC/SCoPE
  - lightx2v/Wan2.2-Lightning
  - Qwen/Qwen2-VL-2B-Instruct
tags:
  - image-to-video
  - camera-control
  - wan2.2
deployspec:
  entry_file: app.py
---

# SCoPE Camera-Controlled Video Generation

基于 SCoPE 的相机轨迹可控图生视频 Demo：上传首帧、选择三维相机运动并填写场景描述，即可生成具有指定镜头运动的短视频。

| 资源 | 地址 |
|---|---|
| 在线体验 | [ModelScope Studio](https://modelscope.cn/studios/TencentARC/scope-camera-video-generation) |
| 相机控制模型 | [TencentARC/SCoPE](https://modelscope.cn/models/TencentARC/SCoPE) |
| 四步加速 LoRA | [lightx2v/Wan2.2-Lightning](https://modelscope.cn/models/lightx2v/Wan2.2-Lightning) |
| 自动提示词模型 | [Qwen/Qwen2-VL-2B-Instruct](https://modelscope.cn/models/Qwen/Qwen2-VL-2B-Instruct) |

## 功能

- 从单张图片生成 `832×480`、81 帧、16 FPS 的视频。
- 提供 Dolly、Truck、Orbit、Crane、Flyover、Spiral 等 16 种相机轨迹预设。
- 调节相机平移幅度、水平视场角、采样步数和随机种子。
- 在生成前以三维图预览相机路径、朝向及起止位置。
- 自动将非 16:9 图片居中裁剪并缩放到推理分辨率。
- 场景描述为空时，使用 Qwen2-VL 根据首帧自动生成视频提示词。
- 提供多组首帧、提示词和镜头运动示例，可一键载入参数。

## 推理方案

```text
首帧 ───────────────┬──────────────────────────────────────────────┐
                    └─> Qwen2-VL（提示词为空时）─> 场景描述       │
相机轨迹 [81,3,4] ─> Plücker camera rays ─> SCoPE Q/K 条件注入     │
场景描述 ─> INT8 UMT5-XXL 文本编码                                │
                                                                  v
FP8 Wan2.2-I2V-A14B 双 DiT + Wan2.2-Lightning 四步 LoRA ─> 视频 latent
                                                                  │
                                                                  v
                                                    tiled Wan VAE ─> MP4
```

1. 相机预设以 OpenCV camera-to-world 矩阵表示，形状为 `[81, 3, 4]`，并转换成相对首帧的相机路径。
2. SCoPE 将归一化后的 Plücker camera rays 经过门控编码，注入 DiT 自注意力的 Query 和 Key，使相机轨迹直接参与去噪。
3. 两个 14B Wan2.2 专家按高噪声与低噪声阶段切换；大尺寸线性层使用 FP8，SCoPE 的相机控制 MLP 保持 BF16。
4. Wan2.2-Lightning LoRA 在加载时融合到两个专家，默认执行 4 个去噪步骤并使用 `cfg_scale=1.0`。
5. 81 帧条件编码和最终解码启用空间 VAE tiling，生成帧随后编码为 MP4。

## 技术细节

| 项目 | 配置 |
|---|---|
| Web UI | Gradio `Blocks` |
| Studio 入口 | `app.py`，端口 `7860` |
| 基础生成模型 | Wan2.2-I2V-A14B 双专家 |
| 相机控制 | SCoPE Sightline-Coordinate Positional Encoding |
| 加速 | Wan2.2-Lightning 4-step LoRA |
| DiT 精度 | 大型投影 FP8，相机控制层 BF16 |
| 文本编码器 | UMT5-XXL，INT8 weight-only |
| 自动提示词 | Qwen2-VL-2B-Instruct，BF16，按需移入 GPU |
| 视频规格 | 832×480、81 帧、16 FPS |
| VAE | BF16、空间分块编码与解码 |
| 请求调度 | 推理锁串行执行，队列单并发 |
| 推荐硬件 | 48 GB xGPU |

模型采用分片流式加载：每个权重 shard 下载后立即装载、融合 LoRA、量化并删除临时文件，避免约 71 GB 的完整原始权重同时占用磁盘。完成首次构建后，融合并量化的两个专家会缓存到 `/mnt/workspace/scope-fp8-cache/`，后续启动优先直接恢复缓存。

## 目录结构

```text
scope-camera-video-generation/
├── app.py             # 模型加载、量化、推理和 Gradio 界面
├── scope/              # SCoPE 相机控制、权重与推理管线
├── diffsynth/          # 视频模型组件与 Wan panshot pipeline
├── trajectories/      # 16 组相机轨迹预设
├── configs/            # 推理配置与负面提示词
├── examples/           # 创空间示例首帧
└── requirements.txt    # Python 运行依赖
```

## 本地运行

该项目需要支持 FP8 的 CUDA GPU。建议使用 Linux、独立 Python 环境以及与 GPU 匹配的 PyTorch：

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch
pip install -r requirements.txt
python app.py
```

首次启动会下载并处理较大的模型分片，应为模型缓存、临时权重和量化缓存预留充足磁盘空间。服务启动后访问 `http://127.0.0.1:7860`。

## 环境变量

| 变量 | 用途 | 默认值 |
|---|---|---|
| `MODELSCOPE_PERSIST_ROOT` | 持久化数据根目录 | `/mnt/workspace`；不可用时回退 `/tmp` |
| `MODELSCOPE_CACHE` | ModelScope 下载缓存 | `<持久化根目录>/modelscope-cache` |
| `SCOPE_WEIGHT_DIR` | 分片下载和加载工作目录 | `<持久化根目录>/scope-weights` |
| `SCOPE_FP8_CACHE_DIR` | 融合并量化后的专家缓存 | `<持久化根目录>/scope-fp8-cache` |
| `MODELSCOPE_MODEL_REVISION` | 模型仓库 revision | `master` |

## ModelScope 部署

创空间使用 Gradio SDK，入口文件为 `app.py`，服务端口为 `7860`，并启用 Gradio MCP server。部署目标是 48 GB xGPU；`/mnt/workspace` 用于保存 ModelScope 下载缓存和融合后的 FP8 专家，从而缩短后续启动时间。模型权重不保存在代码仓库中，而是在首次启动时从 ModelScope 分片下载。
