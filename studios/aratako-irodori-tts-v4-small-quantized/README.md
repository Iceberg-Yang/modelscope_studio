# Irodori-TTS v4.1 Anime ModelScope Studio

面向动漫风格日语语音合成的在线 Demo，支持风格描述、控制 Emoji 和基于参考音频的零样本音色克隆。

| 资源 | 地址 |
|---|---|
| 在线体验 | [ModelScope Studio](https://modelscope.cn/studios/icebergyang/aratako-irodori-tts-v4-small-quantized) |
| 模型 | [phasefield-audio/Irodori-TTS-v4.1-Anime](https://modelscope.cn/models/phasefield-audio/Irodori-TTS-v4.1-Anime) |
| Codec | [Aratako/Semantic-DACVAE-Japanese-32dim](https://modelscope.cn/models/Aratako/Semantic-DACVAE-Japanese-32dim) |

## 功能

- 日语文本转 48 kHz 语音，并自动预测输出时长。
- 使用日语 caption 控制音色、情绪和表达方式。
- 拼接一段或多段参考音频进行零样本音色克隆。
- 在文本中插入 Emoji 控制耳语、叹息或轻笑等表现。
- 调节采样步数、三路 CFG、时长和 linear/sway 时间步计划。

## 推理方案

```text
日语文本 ─> tokenizer + duration predictor ──────────────┐
风格 caption ─> caption condition ───────────────────────┤
参考音频 ─> DACVAE Encoder ─> speaker/latent condition ─┤
                                                          v
                           766M Rectified-Flow DiT + independent CFG
                                                          │
                                                          v
                              Semantic-DACVAE latent ─> 48 kHz waveform
```

1. 文本规范化后由 tokenizer 编码，模型自动预测时长，也可由用户指定秒数或缩放比例。
2. 参考音频依次拼接、响度归一化，并由 Semantic-DACVAE 编码为说话人和声学条件。
3. Rectified-Flow DiT 在连续 codec latent 上采样，分别接收文本、风格 caption 和说话人条件。
4. 三路独立 CFG 默认分别为 3.0、4.0 和 5.0，只在指定时间区间启用；推理复用 context KV cache。
5. DACVAE 将 latent 解码为 48 kHz 波形，并裁剪生成尾部后返回 Gradio 播放器。

## 技术细节

| 项目 | 配置 |
|---|---|
| 主模型 | Irodori-TTS-v4.1-Anime，约 766M 参数 |
| 生成架构 | Rectified-Flow Diffusion Transformer |
| 音频表示 | Semantic-DACVAE-Japanese，32 维连续 latent，48 kHz 输出 |
| 推理框架 | PyTorch 2.10 + Transformers + ModelScope |
| 精度与量化 | 模型和 codec 均使用 BF16；当前 Studio 路径不启用量化 |
| 默认采样 | 40 steps，independent CFG，linear schedule，CFG 区间 0.5–1.0 |
| 音色条件 | 多段参考音频顺序拼接；无参考时自由生成音色 |
| 加速策略 | context KV cache；模型与 codec 常驻 CUDA |
| 硬件与并发 | L20 48GB xGPU；单模型实例、单并发 |
| 模型缓存 | 优先使用 `/mnt/workspace` 持久化缓存 |

## 目录结构

```text
aratako-irodori-tts-v4-small-quantized/
├── app.py              # 推理参数封装与 Gradio UI
├── irodori_tts/        # 模型、RF 采样、时长预测、codec 与 tokenizer
├── dacvae/             # 随仓库提供的 DACVAE 运行代码
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

可通过 `MODEL_REPO` 替换模型 ID，通过 `GRADIO_SERVER_NAME` 和 `GRADIO_SERVER_PORT` 修改监听地址。服务默认监听 `0.0.0.0:7860`。

## ModelScope 部署

创空间使用 Gradio，入口为 `app.py`，基础镜像提供 PyTorch 2.10 和 ModelScope 1.37，运行于 48GB xGPU 并限制为单并发。模型与 codec 在启动时从 ModelScope 下载，代码仓库不保存权重。

使用音色克隆时，请确保拥有参考声音和合成内容的合法使用权，不要用于冒充、欺诈或侵犯隐私。
