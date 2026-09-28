---
title: FireRedTTS3
emoji: 🔥
colorFrom: red
colorTo: pink
license: apache-2.0
short_description: Voice cloning, voice design and speech editing
---

# FireRedTTS3 ModelScope Studio

基于 FireRedTTS3 的统一语音生成与编辑 Demo，支持零样本音色克隆、自然语言音色设计以及语义和声学编辑。

| 资源 | 地址 |
|---|---|
| 在线体验 | [ModelScope Studio](https://modelscope.cn/studios/FireRedTeam/FireRedTTS3) |
| 模型 | [FireRedTeam/FireRedTTS3](https://modelscope.cn/models/FireRedTeam/FireRedTTS3) |

## 功能

- 使用短参考音频进行零样本音色克隆，支持 24 种语言和 21 种中文方言。
- 根据自然语言描述直接设计音色，无需参考音频。
- 按指令插入、删除或替换语音内容。
- 调整已有语音的语速、音高或音量。
- 使用 fastText 自动识别语言，并通过本地 WeText 完成中英文文本规范化。

## 推理方案

```text
文本 ─> 语言识别 + 文本规范化 + tokenizer ───────────────┐
参考/待编辑音频 ─> RedAE Encoder ─> 25 Hz audio latent ──┤
参考音频 ─> CAM++ ─> speaker embedding（克隆模式）───────┤
自然语言指令 ─> ChatML task tokens（设计/编辑模式）───────┤
                                                          v
                                         Qwen3-1.7B 自回归骨干 + KV cache
                                                          │
                                                          v
                                       DiT Flow Matching + CFG 逐 patch 生成
                                                          │
                                                          v
                                             RedAE Decoder ─> 24 kHz WAV
```

1. 文本先进行语言识别、规范化和分词；输入音频统一重采样并限制在 20 秒以内。
2. RedAE 将参考或待编辑音频编码为 25 Hz 连续 latent；克隆模式额外提取 CAM++ 说话人嵌入。
3. Qwen3-1.7B 根据文本、任务指令、音频 latent 和音色条件自回归地产生每个音频 patch 的条件状态。
4. DiT flow-matching head 在每个 patch 内执行多步 CFG 去噪，依次生成连续音频 latent。
5. RedAE 将 latent 解码为 24 kHz 波形；长文本分段生成后通过 cross-fade 拼接。

## 技术细节

| 项目 | 配置 |
|---|---|
| 模型变体 | `fireredtts3_base` + `fireredtts3_instruct` |
| 自回归骨干 | Qwen3-1.7B，KV cache |
| 音频表示 | RedAE，24 kHz 波形 ↔ 25 Hz 连续 latent |
| 音色条件 | CAM++ speaker embedding |
| 生成头 | 11 层 DiT + Flow Matching + CFG |
| 默认采样 | 10 个 flow steps；Base CFG 2.0，Instruct CFG 1.2 |
| 精度与量化 | BF16；无量化 |
| 权重规模 | 约 20.8 GB；Base 与 Instruct 共享同一个 RedAE 实例 |
| 推荐硬件 | NVIDIA L20 48GB xGPU |
| UI 与并发 | Gradio 6.24.0，单并发 |

模型启动时从 ModelScope 下载到 `/mnt/workspace/fireredtts3_cache`。Base 与 Instruct 复用同一个 RedAE，避免重复加载约 3.8 GB 权重；fastText `lid.176` 已随代码提供，不需要额外下载或外部 API。

## 目录结构

```text
FireRedTTS3/
├── app.py              # 四类能力的 Gradio 界面与推理入口
├── fireredtts3/
│   ├── core.py         # 文本前端、切句、推理封装与音频拼接
│   ├── llm/            # Qwen3 + DiT Base/Instruct 实现
│   ├── redae/          # 连续音频自编码器
│   ├── campp/          # 说话人嵌入模型
│   └── utils/          # tokenizer、语言识别与文本规范化
├── examples/           # 中英文参考音频
└── requirements.txt    # Python 依赖
```

## 本地运行

建议使用 Linux、Python 3.11 和支持 BF16 的 CUDA GPU，并为模型缓存预留至少约 21 GB 磁盘空间。

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchaudio
pip install -r requirements.txt
python app.py
```

首次启动会从 ModelScope 下载模型，服务默认监听端口 `7860`。

## ModelScope 部署

创空间使用 Gradio 6.24.0，入口为 `app.py`，部署在 48 GB xGPU 上并限制为单并发。模型权重不保存在代码仓库中，首次启动下载到 `/mnt/workspace`，后续启动复用持久化缓存。

使用音色克隆和语音编辑功能时，请确保拥有输入声音及相关内容的合法使用权，不要用于冒充、欺诈或侵犯隐私。
