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

基于 IndexTTS-2.5 的语音合成 Demo，支持零样本音色克隆、多语言合成、情感控制和语速调节。

| 资源 | 地址 |
|---|---|
| 在线体验 | [ModelScope Studio](https://modelscope.cn/studios/IndexTeam/IndexTTS-2.5) |
| 模型 | [IndexTeam/IndexTTS-2.5](https://modelscope.cn/models/IndexTeam/IndexTTS-2.5) |

## 功能

- 使用参考音频进行零样本音色克隆。
- 支持中文、英文、日语、西班牙语和阿拉伯语。
- 通过情感参考音频、八维情感向量或自然语言描述控制表达。
- 使用 `0.5–2.0` 时长系数调节语速。
- 支持中文拼音、英文 CMU 音素和日语假名发音标注。

## 推理方案

```text
参考音频 ─┬─> Wav2Vec2-BERT + 25 Hz semantic codec ─> 语义条件 ─┐
          └─> CAMPPlus ─────────────────────────────> 音色条件 ─┤
文本 ─> 规范化与分词 ─> GPT Text-to-Semantic ──────> 语义 token ├─> Zipformer S2M / CFM
情感音频、向量或文本 ─> 情感特征 ───────────────────────────────┘          │
                                                                             v
                                                                        Mel 频谱
                                                                             │
                                                                             v
                                                                          BigVGAN
                                                                             │
                                                                             v
                                                                            WAV
```

1. 文本按语言进行规范化和分词；参考音频分别提取语义特征与 CAMPPlus 音色嵌入。
2. 情感参考、八维向量或文本分析结果形成独立情感条件，与音色信息解耦。
3. GPT 自回归模块根据文本、音色和情感条件生成 25 Hz 语义 token。
4. Zipformer S2M 与 Conditional Flow Matching 将语义 token 转换为 Mel 频谱，并结合时长系数控制节奏。
5. BigVGAN 将 Mel 频谱还原为 WAV；Gradio 返回音频并由后台任务清理旧输出。

## 技术细节

| 项目 | 配置 |
|---|---|
| 文本到语义 | GPT 自回归生成，25 Hz semantic codec |
| 语音条件 | Wav2Vec2-BERT、MaskGCT semantic codec、CAMPPlus |
| 情感控制 | 参考音频、8D 向量或 QwenEmotion 文本分析 |
| 语义到频谱 | Zipformer S2M + Conditional Flow Matching |
| 声码器 | BigVGAN |
| 精度与量化 | BF16；无量化 |
| 输出 | WAV |
| 推荐环境 | Python 3.11、PyTorch 2.8、CUDA 12.4、48 GB GPU |
| UI 与并发 | Gradio 5.49.1，单并发，最多排队 20 个任务 |

主模型默认保存在 `/mnt/workspace/IndexTTS-2.5`，辅助模型统一放入其 `hf_cache/`，后续启动优先复用缓存。

## 目录结构

```text
IndexTTS-2.5/
├── app.py              # ModelScope Studio 入口
├── webui.py            # Gradio 页面与推理调用
├── studio_storage.py   # 输出目录和自动清理
├── indextts/           # 模型、文本处理与推理实现
├── examples/           # 示例文本与参考音频
└── requirements.txt    # Python 依赖
```

## 本地运行

建议使用带 CUDA 的 Linux 环境和 Python 3.11。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

首次启动会下载主模型和辅助模型，服务默认监听 `0.0.0.0:7860`。

## 环境变量

| 变量 | 默认值 | 用途 |
|---|---|---|
| `INDEXTTS_MODEL_DIR` | Studio 中为 `/mnt/workspace/IndexTTS-2.5` | 模型与缓存目录 |
| `INDEXTTS_OUTPUT_DIR` | `outputs/` | 生成音频目录 |
| `INDEXTTS_OUTPUT_RETENTION_SECONDS` | `21600` | 输出保留时间 |
| `INDEXTTS_OUTPUT_MAX_BYTES` | `2147483648` | 输出目录容量上限 |
| `INDEXTTS_OUTPUT_MAX_FILES` | `1000` | 输出文件数量上限 |

## ModelScope 部署

创空间使用 Gradio 5.49.1，入口为 `app.py`，端口为 `7860`。模型权重不保存在代码仓库中，首次启动时下载到 `/mnt/workspace` 并在后续启动中复用。

使用语音克隆功能时，请确保拥有参考声音的合法使用权，不要将生成内容用于冒充、欺诈或侵犯隐私。
