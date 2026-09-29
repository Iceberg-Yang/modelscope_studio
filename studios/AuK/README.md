---
domain: [audio]
tags: [text-to-speech, speech-editing, voice-cloning]
models:
  - Tencent-Hunyuan/AuK
  - Tencent-Hunyuan/AuK-Flash
  - Qwen/Qwen2.5-Omni-3B
deployspec:
  entry_file: app.py
license: Apache License 2.0
---

# AuK

面向指令驱动语音生成与编辑的 AuK 创空间，支持文本转语音、零样本音色迁移，以及基于自然语言的语音内容、风格和非语言声音编辑。

[在线体验](https://www.modelscope.ai/studios/Tencent-Hunyuan/AuK) · [AuK](https://www.modelscope.ai/models/Tencent-Hunyuan/AuK) · [AuK-Flash](https://www.modelscope.ai/models/Tencent-Hunyuan/AuK-Flash)

## 推理流程

```text
自然语言指令 + 可选参考音频
  └─ Prompt Enhancer：任务识别、参数整理、时长估计
      └─ Qwen2.5-Omni-3B 编码文本与参考音频条件
          └─ CFMEdit / Flux2Edit 潜空间生成
              ├─ AuK Base：可调 NFE 与 CFG
              └─ AuK-Flash：固定 4 步、关闭 CFG
                  └─ BigVGAN Flow VAE 解码
                      └─ PCM16 WAV 输出
```

1. 应用从 ModelScope 下载 AuK Base、AuK-Flash 与共享的 Qwen2.5-Omni-3B 编码器。
2. Prompt Enhancer 将自然语言需求归类为 TTS 或语音编辑任务，并补齐模型输入与目标时长；也可关闭后手动填写参数。
3. Qwen2.5-Omni Thinker 保留文本与音频能力、移除未使用的视觉塔，形成多模态条件。
4. CFMEdit 以 Flux2Edit 为骨干在 BigVGAN Flow VAE 潜空间采样；Base 默认 32 NFE、CFG 2.0，Flash 锁定四步时间网格并关闭 CFG。
5. 潜变量经 VAE 解码，完成响度处理后以 PCM16 WAV 返回。

## 技术要点

| 项目 | 实现 |
|---|---|
| 模型变体 | `Tencent-Hunyuan/AuK` 与蒸馏版 `Tencent-Hunyuan/AuK-Flash` |
| 条件编码 | `Qwen/Qwen2.5-Omni-3B` 同时处理文本和参考音频，移除视觉模块节省显存 |
| 生成骨干 | `CFMEdit` + `Flux2Edit`，在音频 VAE 潜空间执行条件流匹配 |
| 音频编解码 | BigVGAN Flow VAE，参考音频先编码，生成段再反归一化并解码 |
| Base 采样 | NFE 和 CFG 可调，界面默认 `32 / 2.0` |
| Flash 采样 | DMD 蒸馏配方，固定四步时间网格、CFG=0 |
| 时长控制 | 支持显式时长；Prompt Enhancer 可结合文本、参考音频、ASR/VAD 自动估计 |
| 兼容处理 | SoundFile 适配新 TorchAudio/TorchCodec，输出统一为 PCM16 WAV |
| 资源策略 | 两个变体常驻 CUDA；配置一致时共享 Qwen 编码器，并校验 VAE 后复用 |

## 目录

```text
.
├── app.py                    # ModelScope 下载、运行时适配和 Gradio 入口
├── src/auk/infer/            # 推理、Prompt Enhancer 与界面逻辑
├── src/auk/model/            # CFMEdit、Flux2Edit 与 VAE
├── examples/                 # 按任务组织的示例音频
├── tests/                    # 运行时与界面测试
├── .env.example              # 可选服务配置示例
└── requirements.txt          # Python 依赖
```

## 本地运行

该应用需要 CUDA 环境，模型体积和显存需求较高。Prompt Enhancer 依赖兼容 OpenAI API 的 LLM 配置；未配置时仍可使用手动模式。云端 ASR 未配置时可回退到本地识别。

```bash
pip install -r requirements.txt
python app.py
```

常用环境变量见 `.env.example`。请通过环境变量或 ModelScope Secrets 注入凭据，不要写入代码仓库。
