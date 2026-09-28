---
title: Anima 2.9B 动漫插画生成器
emoji: 🎨
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
license: other
models:
  - Gazingstars123/Anima-2.9B
  - circlestone-labs/Anima
---

# Anima 2.9B 动漫插画生成器

基于 Anima-2.9B 的动漫与非写实插画生成 Demo，通过固定版本的 ComfyUI 运行标准文生图工作流。

| 资源 | 地址 |
|---|---|
| 在线体验 | [ModelScope Studio](https://modelscope.cn/studios/icebergyang/Anima-2.9B) |
| 模型 | [Gazingstars123/Anima-2.9B](https://modelscope.cn/models/Gazingstars123/Anima-2.9B) |
| 基础组件 | [circlestone-labs/Anima](https://modelscope.cn/models/circlestone-labs/Anima) |

## 功能

- 生成动漫、插画和非写实艺术图像，支持正向与反向提示词。
- 提供纵向、横向、方形及最高 1536×1536 的实验分辨率。
- 可选 Euler、ER-SDE、Res Multistep 采样器和四种调度器。
- 支持 28–50 步、CFG 3.5–5.0、固定或随机种子，并展示实际生成参数。

## 推理方案

```text
正向 / 反向提示词 ─> Qwen3 0.6B Text Encoder ─> 条件向量 ─┐
空白 latent + seed ───────────────────────────────────────┤
                                                          v
                         Anima-2.9B BF16 + ComfyUI KSampler
                                                          │
                                                          v
                                Qwen Image VAE ─> PNG
```

1. 后台线程首次启动时安装固定提交的 ComfyUI，并从 ModelScope 下载三个模型文件。
2. Qwen3 0.6B 编码正、负提示词，`EmptyLatentImage` 按用户选择创建单张 latent。
3. ComfyUI `KSampler` 使用指定 seed、采样器、调度器、步数和 CFG 完成全量去噪。
4. Qwen Image VAE 解码 latent；Gradio 通过本机 ComfyUI HTTP API 轮询并读取 PNG。

## 技术细节

| 项目 | 配置 |
|---|---|
| 主模型 | Anima-2.9B Preview v1 BF16，约 5.84 GB |
| 文本编码器 | Qwen3 0.6B，`qwen_3_06b_base.safetensors` |
| 解码器 | Qwen Image VAE |
| 后端 | ComfyUI，固定提交 `5ab2f7a2d676c1fb7b410c22e82e2ed8f217b56c` |
| Workflow | UNETLoader → CLIPTextEncode → EmptyLatentImage → KSampler → VAEDecode |
| 默认采样 | 30 steps，CFG 4.0，Euler，SGM Uniform，denoise 1.0 |
| 分辨率 | 812×1216 至 1536×1536，共六种预设 |
| 运行方式 | ComfyUI 仅监听 127.0.0.1:8188；Gradio 单并发、队列 8 |
| 持久化 | `/mnt/workspace/anima-2.9b-studio` 保存运行时、模型、输出和日志 |

## 目录结构

```text
Anima-2.9B/
├── app.py             # Gradio UI 与参数入口
├── anima_backend.py   # ComfyUI 安装、模型下载、workflow 与结果轮询
├── assets/            # 项目封面素材
├── tests/             # 后端单元测试
└── requirements.txt
```

## 本地运行

建议使用 Linux、Python 3.11 和 CUDA GPU。首次启动需要联网安装 ComfyUI 并下载模型。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

可用 `ANIMA_RUNTIME_ROOT` 修改持久化目录，用 `COMFYUI_PORT` 修改内部服务端口。

## ModelScope 部署

创空间使用 Gradio，入口为 `app.py`，运行于 48 GB xGPU。初始化在后台进行，页面会显示模型状态；提前提交的任务会等待运行时就绪。
