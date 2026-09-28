---
title: Ornith-1.5-9B Reasoning Playground
emoji: 🐦
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
license: mit
---

# Ornith-1.5-9B Reasoning Playground

在 ModelScope 48GB xGPU 创空间直接运行 Ornith-1.5-9B，展示多轮对话与推理能力。

| 资源 | 地址 |
|---|---|
| 在线体验 | [ModelScope Studio](https://modelscope.cn/studios/icebergyang/ornith-1-5-9b) |
| 模型 | [ornith-ai/Ornith-1.5-9B](https://modelscope.cn/models/ornith-ai/Ornith-1.5-9B) |

## 功能

- 多轮文本对话，并以流式方式持续显示回答。
- 将模型输出中的 `<think>...</think>` 与最终正文分开展示。
- 提供通用推理与精确编码预设，可调整温度和最大输出长度。
- 显示输入、输出 token 数、推理耗时和模型加载状态。

## 推理方案

```text
系统提示词 + 多轮历史 + 当前问题
                 │
                 v
       AutoProcessor chat template
                 │ tokens / attention mask
                 v
  Ornith-1.5-9B BF16 · device_map=auto
                 │ TextIteratorStreamer
                 v
      <think> 推理过程 + 最终回答
```

1. 首次请求从 ModelScope 下载模型，并以 BF16、低 CPU 内存模式加载到可用设备。
2. Processor 将系统提示词和完整对话历史套入模型 chat template，并限制输入不超过 32K tokens。
3. `model.generate` 在后台线程执行，主线程通过 `TextIteratorStreamer` 逐段更新 Gradio 对话框。
4. 输出按 `</think>` 标记拆分为折叠的推理过程和最终回答，同时统计 token 与耗时。

## 技术细节

| 项目 | 配置 |
|---|---|
| 模型 | Ornith-1.5-9B，约 9.41B 参数、18.84 GB BF16 权重 |
| 加载方式 | `AutoModelForImageTextToText`，BF16，`device_map="auto"` |
| 输入上限 | 默认 32,768 tokens，可通过环境变量修改 |
| 输出上限 | UI 支持 128–4,096 tokens，默认 2,048 |
| 默认采样 | temperature 1.0，top-p 0.95，top-k 20，repetition penalty 1.0 |
| 精确编码预设 | temperature 0.6，最大输出 2,048 tokens |
| 流式输出 | 后台生成线程 + Transformers `TextIteratorStreamer` |
| 推理展示 | 实时解析 `<think>` 标签，推理过程与正文分栏 |
| 并发与硬件 | 单并发；ModelScope 48 GB xGPU |
| 模型缓存 | `/mnt/workspace/modelscope` |

## 目录结构

```text
ornith-1-5-9b/
├── app.py          # 模型加载、流式生成与 Gradio UI
├── assets/         # 项目封面素材
└── requirements.txt
```

## 本地启动

需要一张有足够显存且支持 BF16 的 NVIDIA GPU：

```bash
python -m pip install -r requirements.txt
python app.py
```

可通过 `MODEL_ID`、`MODEL_REVISION`、`MODELSCOPE_CACHE`、`MAX_INPUT_TOKENS` 和 `SYSTEM_PROMPT` 修改模型与运行参数。

## ModelScope 部署

创空间使用 Gradio，入口为 `app.py`，默认监听 `0.0.0.0:7860`。模型按首次请求懒加载，生成队列限制为单并发。
