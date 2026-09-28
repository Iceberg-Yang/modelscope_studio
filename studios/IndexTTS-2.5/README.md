---
title: IndexTTS-2.5 Demo
emoji: 🎙️
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: other
preload_from_hub:
  - IndexTeam/IndexTTS-2.5
  - amphion/MaskGCT
  - funasr/campplus
  - facebook/w2v-bert-2.0
  - nv-community/bigvgan_v2_22khz_80band_256x
---

# IndexTTS-2.5 ModelScope Studio

面向 IndexTTS-2.5 的交互式语音合成 Demo，支持零样本音色克隆、多语言合成、情感控制、语速调节和细粒度发音标注。

| 资源 | 地址 |
|---|---|
| 在线体验 | [ModelScope Studio](https://modelscope.cn/studios/IndexTeam/IndexTTS-2.5) |
| 模型 | [IndexTeam/IndexTTS-2.5](https://modelscope.cn/models/IndexTeam/IndexTTS-2.5) |

## 功能

- 使用一段参考音频进行零样本音色克隆。
- 合成中文、英文、日语、西班牙语和阿拉伯语。
- 通过音色参考、独立情感参考音频、八维情感向量或情感描述文本控制表达。
- 使用 `0.5–2.0` 的时长系数调节语速；数值越小越快，数值越大越慢。
- 支持中文拼音、英文 CMU 音素和日语假名发音标注。
- 内置多语言文本与参考音频示例，生成结果可直接试听和下载。

发音标注示例：

```text
他在银<行|XING2>里<行|HANG2>走了半天。
He had a <minute|M IH1 . N AH0 T> to examine the details.
彼は料理が<上手|じょうず>だ。
```

## 推理方案

```text
参考音频 ─┬─> Wav2Vec2-BERT + semantic codec ─> 语义条件 ─┐
          └─> CAMPPlus ───────────────────────> 音色条件 ─┤
文本 ─> 规范化与分词 ─> GPT Text-to-Semantic ─> 语义 token ├─> S2M / CFM ─> Mel
情感音频、向量或文本 ─────────────────────────> 情感条件 ─┘                  │
                                                                          v
                                                                     BigVGAN ─> WAV
```

1. `webui.py` 校验模型目录；缺少文件时，通过 ModelScope 或 Hugging Face 镜像准备主模型与辅助模型。
2. 文本经过语言相关的规范化和分词，参考音频分别提取语义特征、音色嵌入和可选情感信息。
3. GPT 自回归模块生成低帧率语义 token；IndexTTS-2.5 使用 25 Hz semantic codec 表示语音内容。
4. Zipformer 系列的 Semantic-to-Mel 模块结合时长与条件信息生成 Mel 频谱，随后由 BigVGAN 声码器还原波形。
5. Gradio 将 WAV 文件写入输出目录并返回播放器；后台清理器按时间、文件数量和总容量回收旧结果。

## 技术细节

| 项目 | 配置 |
|---|---|
| Web UI | Gradio 5.49.1 |
| Studio 入口 | `app.py`，监听 `0.0.0.0:7860` |
| 主模型 | `IndexTeam/IndexTTS-2.5` |
| 文本到语义 | GPT 自回归生成 |
| 语音条件 | Wav2Vec2-BERT、MaskGCT semantic codec、CAMPPlus |
| 语义到频谱 | Zipformer S2M / Conditional Flow Matching |
| 声码器 | BigVGAN |
| 推理精度 | GPU 上启用 BF16；代码参数沿用兼容名称 `--fp16` |
| 推荐环境 | Python 3.11、PyTorch 2.8、CUDA 12.4、48 GB GPU |
| 请求调度 | 单模型实例、单并发推理、最多排队 20 个任务 |

主模型默认保存在 `/mnt/workspace/IndexTTS-2.5`，使 ModelScope Studio 重启后能够复用持久化权重。本地运行时则使用项目内的 `checkpoints/`。辅助模型统一准备到模型目录下的 `hf_cache/`，避免在每次推理时重复下载。

## 目录结构

```text
IndexTTS-2.5/
├── app.py                 # ModelScope Studio 入口与队列配置
├── webui.py               # Gradio 页面、参数校验和推理调用
├── studio_storage.py      # 输出目录与自动清理策略
├── indextts/              # 模型加载、文本处理和推理实现
├── examples/              # 演示文本与参考音频
├── assets/                # README 和界面素材
├── requirements.txt       # Python 依赖
└── pyproject.toml         # 包与开发工具配置
```

## 本地运行

建议使用带 CUDA 的 Linux 环境和独立 Python 3.11 虚拟环境。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

首次启动需要下载模型文件，耗时取决于网络和磁盘速度。服务启动后访问 `http://127.0.0.1:7860`。

## 环境变量

| 变量 | 用途 | 默认值 |
|---|---|---|
| `INDEXTTS_MODEL_DIR` | 主模型和辅助模型目录 | Studio 中为 `/mnt/workspace/IndexTTS-2.5`，本地为 `checkpoints/` |
| `INDEXTTS_OUTPUT_DIR` | 生成音频目录 | `outputs/` |
| `INDEXTTS_OUTPUT_RETENTION_SECONDS` | 输出保留秒数 | `21600`（6 小时） |
| `INDEXTTS_OUTPUT_MAX_BYTES` | 输出目录容量上限 | `2147483648`（2 GiB） |
| `INDEXTTS_OUTPUT_MAX_FILES` | 输出文件数量上限 | `1000` |
| `INDEXTTS_OUTPUT_CLEANUP_INTERVAL_SECONDS` | 后台清理间隔 | `300` |

## ModelScope 部署

创空间使用 Gradio SDK，入口为 `app.py`，端口为 `7860`。部署环境需要能够访问模型仓库，并为 `/mnt/workspace` 提供足够的持久化空间。代码仓库不包含主模型权重；首次启动会自动下载所需文件，之后优先复用本地缓存。

使用语音克隆功能时，请确保拥有参考声音的合法使用权，不要将生成内容用于冒充、欺诈、侵犯隐私或其他违法用途。
