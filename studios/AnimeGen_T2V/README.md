---
title: AnimeGen T2V
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.17.3
python_version: '3.11'
app_file: app.py
pinned: false
license: apache-2.0
short_description: Anime-style text-to-video generation powered by Wan2.2
tags:
  - video-generation
  - text-to-video
  - anime
  - Wan2.2
  - diffusers
---

# AnimeGen T2V ModelScope Studio

基于 Wan2.2 双 Transformer 架构的动漫风格文生视频 Demo，使用 AnimeGen 微调权重和 Lightning LoRA 缩短采样过程。

| 资源 | 地址 |
|---|---|
| 在线体验 | [ModelScope Studio](https://modelscope.cn/studios/icebergyang/AnimeGen_T2V) |
| 基座模型 | [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers) |
| 加速 LoRA | [lightx2v/Wan2.2-Lightning](https://modelscope.cn/models/lightx2v/Wan2.2-Lightning) |

## 功能

- 将文本描述生成约 3 秒、16 FPS 的动漫风格视频。
- 支持 832×480 横屏和 480×832 竖屏两种画幅。
- 使用固定动漫风格前缀、可编辑负向提示词和 8 步快速采样。
- 可选关键词过滤，以及通过 DashScope 或 OpenAI API 执行版权 IP 检测。

## 推理方案

```text
文本提示词 ─> Anime style 前缀 ─> 文本编码 ───────────────┐
                                                         v
AnimeGen High-noise Transformer + Lightning LoRA ─> 高噪声去噪
                                                         │
AnimeGen Low-noise Transformer  + Lightning LoRA ─> 低噪声去噪
                                                         │
                    FlowMatch Euler Scheduler（8 steps）
                                                         v
                              Wan VAE ─> 49 帧 ─> MP4
```

1. 应用先检查关键词，并可调用外部 LLM 判断提示词是否包含明确的版权角色或作品。
2. 两组 AnimeGen 单文件权重分别替换 Wan2.2 pipeline 的高噪声与低噪声 Transformer。
3. 两个 Transformer 分别加载 Lightning LoRA，adapter 权重设为 2.0 和 1.0。
4. Transformer 权重以 FP8 分层存储、BF16 计算，VAE 使用 FP32，并通过 CPU offload 控制显存占用。
5. FlowMatch Euler 调度器以 shift 3.0 完成 8 步去噪，VAE 解码 49 帧后按 16 FPS 写入 MP4。

## 技术细节

| 项目 | 配置 |
|---|---|
| 基座 | Wan2.2-T2V-A14B-Diffusers，双 Transformer 文生视频 pipeline |
| 微调权重 | `high_noise.safetensors` 与 `low_noise.safetensors` |
| 加速 | Wan2.2 Lightning 4-step LoRA；当前应用执行 8 个 inference steps |
| 调度器 | FlowMatchEulerDiscreteScheduler，shift 3.0 |
| 精度 | Transformer FP8 storage / BF16 compute；pipeline BF16；VAE FP32 |
| 显存策略 | Diffusers model CPU offload；生成前后清理 CUDA cache |
| 输出规格 | 832×480 或 480×832，49 帧，16 FPS，MP4 |
| 默认引导 | guidance scale 1.0；负向提示词支持自定义 |
| 模型缓存 | `/mnt/workspace/hf_cache`；优先 ModelScope，失败时回退 Hugging Face 镜像 |
| 内容检查 | 本地关键词；可选 DashScope `qwen-plus` 或 OpenAI `gpt-4o-mini` |

## 目录结构

```text
AnimeGen_T2V/
├── app.py                    # 权重装配、视频推理与 Gradio UI
├── copyright_classifier.py   # 可选的提示词版权 IP 检测
├── configuration.json        # ModelScope 模型元数据
└── requirements.txt          # Python 依赖
```

## 本地运行

建议使用 Linux、Python 3.11 和支持 BF16/FP8 的 CUDA GPU，并预留模型下载和缓存空间。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

版权检测为可选功能：设置 `DASHSCOPE_API_KEY` 或 `OPENAI_API_KEY` 后启用，也可用 `LLM_BACKEND` 选择后端。`NG_WORD` 与 `NG_WORD_JA` 接受 JSON 列表形式的过滤词。

## ModelScope 部署

创空间使用 Gradio，入口为 `app.py`，队列上限为 10。基础模型、AnimeGen 权重和 Lightning LoRA 均在启动时下载，代码仓库不保存模型权重。
