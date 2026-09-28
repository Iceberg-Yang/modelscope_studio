---
title: Breeze TTS 2
emoji: 🎙️
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
short_description: Bilingual TTS with voice design, cloning, and direction
python_version: 3.12
colorFrom: blue
colorTo: indigo
startup_duration_timeout: 30m
---

# Breeze TTS 2 ModelScope Studio

面向中英文语音合成的在线 Demo，提供自然语言音色设计、参考音频克隆和带表演指令的音色引导三种模式。

| 资源 | 地址 |
|---|---|
| 在线体验 | [ModelScope Studio](https://modelscope.cn/studios/BreezeBlue/breeze-tts-2-demo) |
| 模型 | [BreezeBlue/Breeze-TTS-2](https://modelscope.cn/models/BreezeBlue/Breeze-TTS-2) |
| 参考音频转写 | [openai-mirror/whisper-tiny](https://modelscope.cn/models/openai-mirror/whisper-tiny) |

## 功能

- 使用自然语言描述音色、情绪、语速和表达风格。
- 从参考音频及其文本克隆音色，并可继续施加表演指令。
- 支持中文、英文，以及笑、叹气、咳嗽和清嗓等行内事件标记。
- 提供随机种子和 CFG 强度控制；参考音频可由 Whisper Tiny 自动转写后编辑。

## 推理方案

```text
目标文本 + 音色指令 ─> 模板与文本编码 ───────────────────┐
参考音频 ─> Qwen3 TTS Tokenizer ─> 音频 token ───────────┤
参考文本 / Whisper Tiny 转写 ─────────────────────────────┤
                                                           v
                        Breeze 自回归 Backbone + CFG
                                      │ 首个 codebook token
                                      v
                          Depth Decoder（其余 codebooks）
                                      │
                                      v
                         Audio Codec ─> PCM 16-bit WAV
```

1. Voice Design 使用 `tts_instruction` 模板组织目标文本与自然语言音色描述。
2. Voice Clone 和 Voice Direction 使用 `ref_edit_tata` 模板，将参考文本与音频 token 一同作为上下文。
3. 自回归 backbone 逐帧预测首个音频 codebook，depth decoder 再生成其余 31 个 codebook token。
4. 推理按 CFG 构造条件分支，默认启用采样、temperature 0.9、top-k 50，并对重复 token 施加 1.1 惩罚。
5. Codec 将生成 token 分块解码为波形，服务端拼接后写出 PCM 16-bit WAV。

## 技术细节

| 项目 | 配置 |
|---|---|
| 主模型 | BreezeBlue/Breeze-TTS-2 |
| 生成架构 | 文本编码器 + 自回归 backbone + depth decoder + audio codec |
| 音频表示 | 32 个离散 codebook；Qwen3 TTS Tokenizer 负责参考音频编码与生成音频解码 |
| 模板 | `tts_instruction`（音色设计）与 `ref_edit_tata`（克隆 / 引导） |
| 推理框架 | PyTorch + Transformers 4.57.3 + ModelScope + Gradio |
| 精度与注意力 | BF16，eager attention，模型常驻 CUDA |
| 生成上限 | 1500 个新 token，最大序列长度 2048 |
| 默认采样 | temperature 0.9，top-k 50，top-p 1.0，repetition penalty 1.1 |
| 参考转写 | Whisper Tiny 在 CPU 懒加载；输入重采样为 16 kHz，转写结果可编辑 |
| 运行策略 | 单请求推理；CUDA Graph 快速路径代码随仓库保留，但当前 Studio 配置未启用 |
| 模型缓存 | 默认使用 `/mnt/workspace/modelscope` 持久化缓存 |

## 目录结构

```text
breeze-tts-2-demo/
├── app.py              # 模型下载、三类任务入口与 Gradio UI
├── breeze_infer/       # 模板、输入准备、运行时和音频工具
├── models/             # Breeze 架构、生成逻辑与可选 CUDA Graph 实现
├── configs/fast.json   # 快速运行配置与图捕获桶定义
└── requirements.txt    # Python 依赖
```

## 本地运行

建议使用 Linux、Python 3.12 和支持 BF16 的 CUDA GPU。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

可通过 `MODEL_REVISION`、`ASR_MODEL_REVISION` 和 `BREEZE_CACHE_DIR` 调整模型版本与缓存位置。服务默认监听 `0.0.0.0:7860`。

## ModelScope 部署

创空间使用 Gradio，入口为 `app.py`。主模型启动时从 ModelScope 下载并加载到 GPU；Whisper Tiny 仅在请求自动转写参考音频时下载到缓存并加载到 CPU。

使用音色克隆时，请确保拥有参考声音和合成内容的合法使用权，不要用于冒充、欺诈或侵犯隐私。
