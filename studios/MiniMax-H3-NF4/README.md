---
title: MiniMax H3 NF4 Turbo
emoji: 🎬
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
license: apache-2.0
---

# MiniMax-H3 NF4 Turbo ModelScope Studio

使用 DiffSynth-Studio 在单张 xGPU 上运行 MiniMax-H3 NF4，支持文生音视频、首尾帧约束，以及 Larry / LightX 两种 Turbo LoRA。

| 资源 | 地址 |
|---|---|
| 在线体验 | [ModelScope Studio](https://modelscope.cn/studios/MiniMax/MiniMax-H3-NF4) |
| NF4 模型 | [DiffSynth-Studio/MiniMax-H3-NF4](https://modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4) |
| Processor | [MiniMax/MiniMax-H3](https://modelscope.cn/models/MiniMax/MiniMax-H3) |

## 功能

- 文本生成带 32 kHz 同步音频的视频，并支持首帧、尾帧或首尾帧条件。
- 提供横屏、竖屏、方形及超宽屏等多组画布，时长范围 2–14 秒。
- 可选择 Larry Turbo v4、LightX2V Turbo v0.1 或关闭 LoRA 使用 NF4 基模。
- 自动裁剪关键帧、管理持久化输出，并公开兼容原 Space 的生成 API。

## 推理方案

```text
文本 + 可选首尾帧 ─> FL2VA Processor ───────────────────────┐
                                                           v
             MiniMax-H3 NF4 Audio-Video Pipeline
        ┌───────────────┼───────────────────┐
        │               │                   │
   NF4 DiT          Video VAE           Audio VAE
        │               │                   │
  可选 Turbo LoRA       └──── 同步解码 ──────┘
        │                                   │
        └──── DiffSynth disk/CPU/GPU offload ─> MP4
```

1. 输入图像按目标画幅 cover-crop，并作为第 0 帧、末帧或两端关键帧传入 pipeline。
2. NF4 DiT、文本编码器、视频 VAE 和音频 VAE 采用 BF16 计算，并由 DiffSynth 在磁盘、CPU 与 GPU 间调度。
3. Larry LoRA 直接映射至 DiffSynth 模块；LightX 的 PEFT 权重会转换命名并将 Q/K/V 融合为 `qkv_proj`。
4. LoRA 加载前校验完整 A/B 对、目标模块覆盖率和矩阵形状，以运行时残差方式挂载而不合并 NF4 权重。
5. 视频按 24 FPS、音频按 32 kHz 合并为 MP4；每次请求前后卸载模型并清理 CUDA cache。

## 技术细节

| 项目 | 配置 |
|---|---|---|
| 推理框架 | DiffSynth-Studio 2.1.1 `MiniMaxH3Pipeline` |
| 模型组成 | NF4 DiT、NF4 text encoder、NF4 video VAE、NF4 audio VAE |
| 计算精度 | NF4 权重，BF16 onload / preparation / computation |
| Turbo LoRA | Larry v4 默认 6 步；LightX v0.1 默认 4 步；基模默认 50 步 |
| LoRA 方式 | lazy download、严格结构校验、hotload runtime residual |
| 时序约束 | 24 FPS；帧数自动调整为满足 `frames % 17 == 5` |
| 输出音频 | 32 kHz，与视频同步封装为 MP4 |
| 显存管理 | 默认预留 4 GiB；DiffSynth disk → CPU → GPU offload |
| 并发与存储 | 单并发、队列 4；默认仅保留最新 5 个生成视频 |
| 模型缓存 | `/mnt/workspace/modelscope-cache` |

## 目录结构

```text
MiniMax-H3-NF4/
├── app.py          # Pipeline、显存与输出管理、Gradio UI
├── h3_lora.py      # Larry / LightX 转换、校验与热加载
├── examples/       # 首尾帧示例
└── requirements.txt
```

## 本地运行

建议使用 Linux、Python 3.11、CUDA GPU，并为约 52 GB 的首次模型下载预留空间。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

`H3_WORKSPACE` 可修改持久化目录，`H3_GPU_RESERVE_GB` 控制预留显存；各模型和 LoRA ID 也可通过 `H3_*` 环境变量覆盖。

## ModelScope 部署

创空间使用 Gradio，入口为 `app.py`。模型在后台线程加载；前端每 5 秒刷新状态，pipeline 就绪后才接受推理。默认监听 `0.0.0.0:7860`。
