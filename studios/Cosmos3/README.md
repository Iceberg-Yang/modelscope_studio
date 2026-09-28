---
title: Cosmos3 Edge Demo
emoji: 🌌
colorFrom: black
colorTo: green
sdk: gradio
sdk_version: "6.17.3"
app_file: app.py
pinned: false
license: openmdw-1.1
---

# Cosmos3-Edge ModelScope Studio

基于 NVIDIA Cosmos3-Edge Reasoner 的 Physical AI 多模态推理 Demo，支持对文本、图片和短视频进行场景理解、任务规划与物理推理。

| 资源 | 地址 |
|---|---|
| 在线体验 | [ModelScope Studio](https://modelscope.cn/studios/nv-community/Cosmos3) |
| 模型 | [nv-community/Cosmos3-Edge](https://modelscope.cn/models/nv-community/Cosmos3-Edge) |

## 功能

- 分析机器人和真实世界图片，生成任务分解、场景描述或空间定位结果。
- 均匀采样视频帧，理解车辆运动和周边交通场景。
- 无视觉输入时进行 Physical AI 文本推理。
- 切换思考模式，并流式展示模型输出。
- 调节最大生成长度、temperature 和 top-p。

## 推理方案

```text
图片 ───────────────────────────────┐
视频 ─> PyAV 解码 ─> 均匀采样 ≤16 帧 ┤
文本指令 ────────────────────────────┤
                                     v
       AutoProcessor.apply_chat_template(enable_thinking=...)
                                     │
                                     v
     Cosmos3-Edge Reasoner / Nemotron（BF16 + SDPA + CUDA）
                                     │
                                     v
      TextIteratorStreamer ─> Gradio 实时增量文本输出
```

1. 图片直接作为多模态消息输入；视频由 PyAV 解码后最多均匀采样 16 帧，以控制视觉 token 数量。
2. AutoProcessor 使用 Qwen3-VL 兼容消息格式组合视觉内容和用户指令，并传入思考模式开关。
3. Cosmos3-Edge Reasoner 以 BF16 加载到 CUDA，注意力固定使用 PyTorch SDPA。
4. 生成在后台线程执行，TextIteratorStreamer 持续输出新增 token，避免界面等待完整结果。
5. 单例 ModelManager 负责懒加载和线程锁，首次请求下载模型，后续请求复用同一实例。

## 技术细节

| 项目 | 配置 |
|---|---|
| 推理塔 | Cosmos3-Edge Reasoner，约 2B Nemotron |
| 输入 | 文本、单张图片或最多 16 帧视频 |
| 推理框架 | PyTorch + Transformers `Cosmos3EdgeForConditionalGeneration` + ModelScope |
| 精度与量化 | BF16；无量化 |
| 注意力 | PyTorch SDPA |
| 默认生成 | 最大 1024 tokens，temperature 0.6，top-p 0.95 |
| 视频处理 | imageio + PyAV，均匀帧采样 |
| 流式输出 | `TextIteratorStreamer` + 后台生成线程 |
| 硬件 | ModelScope L20 48GB xGPU |
| 缓存 | `/mnt/workspace/modelscope_cache` 持久化模型快照 |

该 Studio 只启用 Cosmos3-Edge 的 Reasoner tower，不提供模型完整的图像或视频生成能力。

## 目录结构

```text
Cosmos3/
├── app.py          # Gradio UI、运行环境修复与事件绑定
├── inference.py    # 模型懒加载、多模态预处理和流式生成
├── theme.py        # NVIDIA 风格主题与页面组件
├── examples/       # 图片和视频推理示例
└── requirements.txt
```

## 本地运行

建议使用 Linux、支持 BF16 的 NVIDIA GPU，以及与本机 CUDA 匹配的 PyTorch。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

模型默认优先从 ModelScope 下载，失败时回退到 Hugging Face 模型 ID。服务由 Gradio 启动，部署环境需开放平台指定端口。

## ModelScope 部署

创空间使用 Gradio 6.17.3，入口为 `app.py`，运行在 48GB xGPU 上。模型按首次请求懒加载，快照写入 `/mnt/workspace`；队列最多保留 12 个请求，单模型实例负责实际生成。
