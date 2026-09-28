---
domain: nlp
tags:
- minicpm5
- minicpm
- text-generation
- llama
- long-context
- thinking-mode
models:
- OpenBMB/MiniCPM5-2B
deployspec:
  entry_file: app.py
license: apache-2.0
---

# MiniCPM5-2B ModelScope Studio

面向 MiniCPM5-2B 的在线对话 Demo，支持多轮上下文、流式输出、思考模式切换和采样参数调节。

| 资源 | 地址 |
|---|---|
| 在线体验 | [ModelScope Studio](https://modelscope.cn/studios/OpenBMB/MiniCPM5-2B) |
| 模型 | [OpenBMB/MiniCPM5-2B](https://modelscope.cn/models/OpenBMB/MiniCPM5-2B) |

## 功能

- 多轮对话与流式增量输出。
- 开启或关闭 `enable_thinking`，并将思考内容与正式回答分区展示。
- 实时调节 `temperature` 和 `top_p`；`temperature=0` 时自动切换为贪婪解码。
- 支持 Markdown、代码块及行内和块级 LaTeX 渲染。
- 可停止正在进行的生成，或一键清空对话。
- 切换思考模式时自动清空旧历史，避免不同提示模板的上下文混用。

## 推理方案

```text
用户消息 + 多轮历史
        │
        v
tokenizer.apply_chat_template(enable_thinking=...)
        │
        v
MiniCPM5-2B / LlamaForCausalLM（BF16、CUDA）
        │
        v
TextIteratorStreamer ─> Gradio 流式更新 ─> Markdown / LaTeX / 思考段渲染
```

1. 启动时通过 ModelScope 下载 `OpenBMB/MiniCPM5-2B`，并将快照保存在持久化缓存目录。
2. `AutoTokenizer` 使用模型自带的 chat template，将系统消息、多轮历史和当前问题组合为提示词，并传入思考模式开关。
3. 标准 `LlamaForCausalLM` 以 BF16 加载到 CUDA；生成在后台线程运行，由 `TextIteratorStreamer` 持续输出新增文本。
4. 界面历史保留完整显示内容，模型上下文只回填去除思考段后的正式回答，避免隐藏推理内容污染后续轮次。
5. Gradio 原生 `reasoning_tags` 负责折叠思考段，生成队列限制为单并发、最多排队 16 个请求。

## 技术细节

| 项目 | 配置 |
|---|---|
| Web UI | Gradio 6.17.3 `Blocks` |
| Studio 入口 | `app.py`，监听 `0.0.0.0:7860` |
| 模型架构 | `LlamaForCausalLM`，约 2.517B 参数、42 层 |
| 上下文长度 | 128K |
| 注意力结构 | 16 attention heads、2 KV heads 的 GQA |
| 推理框架 | PyTorch + Transformers + ModelScope |
| 推理精度 | BF16 |
| 最大生成长度 | 4096 tokens |
| 默认采样 | `temperature=1.0`、`top_p=0.95` |
| 请求调度 | 单模型实例、单并发、最多排队 16 个请求 |

模型权重约 4.7 GiB。创空间默认将快照缓存到 `/mnt/workspace/.cache/modelscope`，以便重启后复用；如果持久化目录不可写，则自动回退到项目内的 `.model_cache/`。

## 目录结构

```text
MiniCPM5-2B/
├── app.py                  # 模型加载、流式生成与 Gradio 界面
├── utils_chatbot.py        # 多轮消息格式整理
├── requirements.txt        # Transformers、ModelScope 等运行依赖
└── vendor/
    └── openbmb-logo.png    # 页面 Logo
```

## 本地运行

建议使用带 CUDA 的 Linux 环境和 Python 3.12。需要预先安装与本机 CUDA 匹配的 PyTorch，并安装兼容版本的 Gradio：

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch gradio==6.17.3
pip install -r requirements.txt
python app.py
```

首次启动会从 ModelScope 下载模型。服务启动后访问 `http://127.0.0.1:7860`。

## 环境变量

| 变量 | 用途 | 默认值 |
|---|---|---|
| `MINICPM_CACHE_DIR` | 模型快照缓存目录 | `/mnt/workspace/.cache/modelscope` |
| `GRADIO_ANALYTICS_ENABLED` | Gradio 遥测开关 | 代码默认设为 `False` |

## ModelScope 部署

创空间使用 Gradio SDK 6.17.3，基础镜像为 `ubuntu22.04-py312-torch2.10.0-modelscope1.37.0`，入口文件为 `app.py`，监听端口为 `7860`。部署环境需要 CUDA GPU，并为模型缓存预留至少约 5 GiB 的持久化空间。模型权重不保存在代码仓库中，而是在首次启动时下载。
