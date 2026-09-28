---
domain: audio
tags:
  - music-generation
  - text-to-music
models:
  - MiniMax/MiniMax-Music3
license: other
deployspec:
  entry_file: app.py
---

# MiniMax Music 3 ModelScope Studio

从结构化音乐描述与带段落标签的歌词生成完整歌曲，支持生成中分段播放以及最终 WAV 下载。

| 资源 | 地址 |
|---|---|
| 在线体验 | [ModelScope Studio](https://modelscope.cn/studios/MiniMax/minimax-music3) |
| 模型 | [MiniMax/MiniMax-Music3](https://modelscope.cn/models/MiniMax/MiniMax-Music3) |

## 功能

- 输入风格描述、歌词和歌曲时长，生成包含人声与伴奏的完整双声道音乐。
- Simple 模式可通过远程语言模型补全结构化描述和歌词；Studio 模式允许逐项编辑。
- 支持 seed、时长、Flow Matching 步数、guidance 和输出 headroom 控制。
- 推理过程中按窗口流式解码 PCM，浏览器可提前播放，完成后自动校正为完整 WAV。

## 推理方案

```text
Global Metadata + Vocal Details + Arrangement + Tagged Lyrics
                              │ tokenizer / CFG prompt pair
                              v
          8B Autoregressive LM + RVQ Depth Decoder
                              │ semantic / depth frame hidden states
                              v
         Condition Encoder ─> Flow Matching DiT（滑动窗口）
                              │ overlapped audio latents
                              v
                 Vocoder ─> 44.1 kHz stereo PCM ─> WAV
```

1. 应用按官方特殊 token 拼接结构化描述和歌词，并建立条件/无条件两路输入用于 AR CFG。
2. 语言模型自回归采样语义 code，RVQ depth decoder 为每帧补全深层 code，并将 hidden states 反馈至下一步。
3. hidden states 以 200 帧窗口、100 帧 hop 送入 condition encoder 和 Flow Matching Transformer。
4. 相邻窗口复用 172 latent 帧作为连续性条件，通过 overlap noise mixing 避免拼接断裂。
5. Vocoder 解码为双声道 PCM；服务端按绝对 sample offset 流式传输并最终写出完整 WAV。

## 技术细节

| 项目 | 配置 |
|---|---|
| Pipeline | Diffusers ModularPipeline，MiniMax-Music3 组件全部从本地快照加载 |
| 模型组成 | tokenizer、8B language model、RVQ depth decoder、condition encoder、DiT、vocoder |
| 精度 | 全组件 BF16、GPU 常驻；启动要求默认至少 40 GiB 显存 |
| AR 采样 | 语义词表约束、top-k 与 CFG；KV cache 逐帧复用 |
| 音频扩散 | Flow Matching，窗口级 CFG，默认 guidance 1.7 |
| 流式窗口 | 200 frames / 100 hop；跨窗口携带 latent 与 condition overlap |
| 输出 | 44.1 kHz、16-bit、双声道 WAV；支持浏览器 PCM 分块播放 |
| 可选编译 | `LM_COMPILE=1` 使用 StaticCache 和 `torch.compile`，按 1024–8192 bucket 预热 |
| 模型 revision | `652e7b018386b56e755551832f9efc6a0009d6e1` |
| 模型缓存 | `/mnt/workspace/models`，按推理组件选择性下载约 28.5 GB |

## 目录结构

```text
minimax-music3/
├── app.py            # 模型加载、AR/DiT/vocoder 推理、流式播放器与 Gradio UI
├── examples/         # 内置歌曲示例及 manifest
├── vendor/           # 与模型实现匹配的 Diffusers wheel
├── cover.png         # 项目封面
└── requirements.txt
```

## 本地运行

建议使用 Linux、Python 3.11 和至少 48 GB 显存的 CUDA GPU。

```bash
python -m venv .venv
source .venv/bin/activate
pip install vendor/diffusers-0.40.0.dev0-py3-none-any.whl
pip install -r requirements.txt
python app.py
```

可用 `MODEL_PATH` 跳过下载，或通过 `MODEL_ID`、`MODEL_REVISION`、`MODEL_CACHE_DIR` 和 `MIN_VRAM_GIB` 修改运行配置。歌词辅助可配置 ModelScope Token，未配置时 Simple 模式会使用内置模板。

## ModelScope 部署

创空间使用 Gradio，入口为 `app.py`，默认监听 `0.0.0.0:7860`。模型启动时从 ModelScope 下载到持久化目录，生成文件保存在临时歌曲目录。
