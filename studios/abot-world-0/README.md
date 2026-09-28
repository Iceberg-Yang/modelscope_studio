---
title: ABot-World A10
sdk: docker
---

# ABot-World-0 ModelScope Studio

基于 ABot-World-0-5B-LF 的动作可控交互式世界模型 Demo，可从单张场景图开始，按键驱动连续视频生成。

| 资源 | 地址 |
|---|---|
| 在线体验 | [ModelScope Studio](https://modelscope.cn/studios/amap_cvlab/abot-world-0) |
| 模型 | [amap_cvlab/ABot-World-0-5B-LF](https://modelscope.cn/models/amap_cvlab/ABot-World-0-5B-LF) |

## 功能

- 以上传图片或内置场景作为首帧，持续生成可探索的动态世界。
- 通过 W/A/S/D 控制移动，通过 I/J/K/L 控制视角，并在生成过程中动态提交动作。
- 使用 LongForcing 模型进行开放式 rollout，分块解码并实时更新画面。
- 保存生成视频，维护参考图缓存和最近结果列表。

## 推理方案

```text
首帧 + 文本场景描述 + 键盘动作
                 │
                 v
      UMT5-XXL Text Encoder + Reference Encoder
                 │ text / image / action conditions
                 v
 ABot-World-0-5B-LF Causal DiT（4-step DMD, block rollout）
                 │ latent blocks
                 v
       Wan2.2 VAE / TAE 解码 ─> 流式帧 ─> H.264 MP4
```

1. 首帧缩放至参考分辨率并编码，文本由 BF16 UMT5-XXL 转换为条件向量。
2. 前端将互斥键组归一化为动作序列，推理 worker 在每个 causal block 读取最新控制状态。
3. LongForcing 配置使用 1000/750/500/250 四个去噪时间步、局部注意力窗口 21 和相对 RoPE。
4. 模型每个 latent block 生成 3 帧 latent；解码后标准输出 12 帧，首块包含参考帧并输出 9 帧。
5. UI 按设定 stride 抽样显示预览，完整帧以 12 FPS 编码为 H.264 视频。

## 技术细节

| 项目 | 配置 |
|---|---|
| 主模型 | ABot-World-0-5B-LF，causal image-to-video / action-conditioned world model |
| 推理框架 | PyTorch + 自定义 CausalInferencePipeline + Gradio |
| 文本编码 | UMT5-XXL BF16 |
| 生成策略 | LongForcing + 4-step DMD + block-wise causal rollout |
| 注意力 | local attention size 21、relative RoPE；A10 路径使用 FlashAttention 2 |
| 精度 | A10 Studio 使用 BF16；禁用 Blackwell 专用 FP8 GEMM/SageAttention3 路径 |
| Studio 分辨率 | 默认 480×832；latent 为 48×30×52 |
| 视频输出 | 12 FPS、H.264；UI 默认每 3 帧刷新一次 |
| 内存策略 | 低于 40 GB 空闲显存时对文本编码器与 VAE 启用动态交换 |
| 持久化 | checkpoint 位于 `/mnt/workspace/checkpoints`，输出位于 `/mnt/workspace/outputs` |

## 目录结构

```text
abot-world-0/
├── studio_app.py       # ModelScope checkpoint 下载与 Docker 入口
├── web_client/         # Gradio UI、动作状态、流式推理与视频写入
├── pipeline/           # causal block 推理主流程
├── wan/                # DiT、VAE、文本编码和 CUDA/Triton 组件
├── quantizer/          # 可选 FP8 量化实现
├── configs/            # LongForcing 与默认模型配置
└── Dockerfile          # CUDA 12.8 / PyTorch 2.10 A10 镜像
```

## 本地运行

Docker Studio 路径建议使用 NVIDIA GPU、CUDA 12.8 和可用的持久化目录：

```bash
docker build -t abot-world-studio .
docker run --gpus all -p 7860:7860 \
  -v "$PWD/workspace:/mnt/workspace" abot-world-studio
```

也可按 `requirements.txt` 安装完整环境后运行 `bash web_client/run.sh`。`ABOT_STREAM_HEIGHT`、`ABOT_STREAM_WIDTH` 和 `ABOT_UI_FRAME_STRIDE` 可调整 Studio 输出与预览频率。

## ModelScope 部署

创空间使用 Docker SDK，入口为 `studio_app.py`。容器首次启动会从 ModelScope 下载 checkpoint 并链接至项目目录；服务监听 `0.0.0.0:7860`。
