---
domain:
  - multi-modal
tags:
  - LingBot-World
  - image-to-video
  - camera-control
models:
  - Robbyant/lingbot-world-v2-1.3b-causal-fast
  - Wan-AI/Wan2.1-T2V-1.3B
license: CC-BY-NC-SA-4.0
---

# LingBot-World V2 1.3B ModelScope Studio

面向镜头轨迹控制的图生视频 Demo：上传首帧、输入场景描述并选择相机运动，生成完整 MP4。

| 资源 | 地址 |
|---|---|
| 在线体验 | [ModelScope Studio](https://modelscope.cn/studios/icebergyang/lingbot-world) |
| 主模型 | [Robbyant/lingbot-world-v2-1.3b-causal-fast](https://modelscope.cn/models/Robbyant/lingbot-world-v2-1.3b-causal-fast) |
| 辅助组件 | [Wan-AI/Wan2.1-T2V-1.3B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B) |

## 功能

- 从单张图片生成 45–81 帧视频，默认 81 帧、16 FPS，约 5.06 秒。
- 提供 15 种合成镜头运动和 5 种内置录制轨迹，可调转向速度与随机种子。
- 输出面积约 832×480，并根据输入宽高比对齐实际尺寸。
- 独立 worker 完成依赖准备、权重校验、生成、MP4 编码与回读校验。

## 推理方案

```text
首帧 ─> Wan2.1 VAE Encode ───────────────┐
Prompt ─> UMT5-XXL BF16 ────────────────┤
镜头预设 ─> 位姿 / 内参 ─> Plücker rays ┤
                                             v
        LingBot-World V2 1.3B Causal-Fast DiT
       3-latent chunks · KV cache · 4-step Flow
                                             │
                                             v
                    Wan2.1 VAE Decode ─> MP4
```

1. 上传图像会清除元数据、限制像素与宽高比，并按 480p 面积对齐模型尺寸。
2. 合成轨迹根据推进、平移、升降、摇镜、俯仰或环绕参数生成位姿；录制轨迹从校验过的离线包读取。
3. UMT5-XXL 编码 Prompt，首帧经 Wan2.1 VAE 形成图像条件，位姿和内参转换为 Plücker camera embedding。
4. Causal-Fast DiT 以 3 个 latent frame 为一块，使用 `[0, 250, 500, 750]` 四个时间步并复用 self/cross KV cache。
5. VAE 解码目标帧，worker 将结果按 16 FPS 编码并回读验证，验证通过后才发布下载链接。

## 技术细节

| 项目 | 配置 |
|---|---|
| 主模型 | LingBot-World V2 causal-fast，1.3B 配置，30 层、dim 1536、12 heads |
| 条件 | 首帧、最多 512 text tokens、相机 Plücker embedding |
| 推理模式 | causal-fast，chunk size 3，local attention 18，sink size 6 |
| 去噪 | Flow UniPC，4-step，默认 shift 10（公开 Space workload） |
| 精度 | 主权重保持 FP32；前向使用 BF16 autocast |
| 视频规格 | 默认 81 帧 / 16 FPS；公开上限 81 帧；输出面积约 832×480 |
| 缓存 | T5 prompt cache、30 层 self-attention KV cache、cross-attention KV cache |
| 模型组件 | 主权重约 6.8 GB；UMT5-XXL BF16、Wan2.1 VAE 与 tokenizer 来自辅助模型 |
| 进程隔离 | 独立 Python venv 和 worker；GPU 单任务互斥、2 小时超时 |
| 媒体安全 | 输入上限 12 MB；输出 MP4 回读校验后复制到独立公开目录 |

## 目录结构

```text
lingbot-world/
├── app.py                    # Gradio UI、任务监督与安全发布
├── prepare_validation.py     # 独立环境、源码包和模型缓存准备
├── model_probe.py            # Pipeline 组装、推理与 MP4 验证
├── space_adapter.py          # 公共参数、图像和相机轨迹适配
├── source_bundle.tar         # 固定推理源码与内置示例包
├── space_bundle.tar          # 固定参考 Space 资产包
├── validation_contract.json  # 模型文件、哈希、架构与运行契约
└── requirements-worker.txt   # worker 独立依赖
```

## 本地运行

建议使用 Linux、Python 3.12、CUDA 12.8 和 48 GB 级 NVIDIA GPU。页面运行环境与推理 worker 的依赖相互隔离。

```bash
python -m venv .venv
source .venv/bin/activate
pip install gradio modelscope pillow
export LINGBOT_ACCEPT_EXPERIMENTAL_CONFIG=1
python app.py
```

首次生成会创建 worker venv，并下载、逐文件校验约 18.73 GB 的主模型和辅助组件缓存。

## ModelScope 部署

创空间使用 Gradio，入口为 `app.py`，队列上限为 2、GPU 生成单并发。服务默认监听 `0.0.0.0:7860`，并要求设置 `LINGBOT_ACCEPT_EXPERIMENTAL_CONFIG=1` 后开放生成。
