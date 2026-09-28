# ModelScope Studio Collection

ModelScope 创空间是魔搭社区提供的在线应用形态，可将开源模型封装为可以直接体验的交互式 Demo。

本仓库整理一系列开发维护的 ModelScope 创空间，每个创空间作为 `studios/` 下的独立目录保存，并补充模型来源、推理方案、关键技术、运行条件和在线体验链接。

## 项目列表

| Studio | 任务 | 主要模型 | 推理技术 | ModelScope | 代码 |
|---|---|---|---|---|---|
| LTX2.5 | 文生视频 / 图生视频 / 同步音频 | Lightricks/LTX-2.5 | DistilledPipeline · BF16 · CPU offload | [在线体验](https://modelscope.cn/studios/icebergyang/LTX2.5) | [`studios/LTX2.5`](studios/LTX2.5) |
| IndexTTS-2.5 | 零样本语音克隆 / 多语言 TTS / 情感控制 | IndexTeam/IndexTTS-2.5 | GPT T2S · Zipformer S2M · BigVGAN · BF16 | [在线体验](https://modelscope.cn/studios/IndexTeam/IndexTTS-2.5) | [`studios/IndexTTS-2.5`](studios/IndexTTS-2.5) |
| MiniCPM5-2B | 多轮对话 / 流式生成 / 思考模式 | OpenBMB/MiniCPM5-2B | Transformers · LlamaForCausalLM · BF16 | [在线体验](https://modelscope.cn/studios/OpenBMB/MiniCPM5-2B) | [`studios/MiniCPM5-2B`](studios/MiniCPM5-2B) |
| scope-camera-video-generation | 相机轨迹可控图生视频 | TencentARC/SCoPE · Wan2.2-I2V-A14B | SCoPE · FP8 · Lightning LoRA · VAE tiling | [在线体验](https://modelscope.cn/studios/TencentARC/scope-camera-video-generation) | [`studios/scope-camera-video-generation`](studios/scope-camera-video-generation) |
| FireRedTTS3 | 音色克隆 / 音色设计 / 语音编辑 | FireRedTeam/FireRedTTS3 | Qwen3-1.7B · DiT Flow Matching · RedAE · BF16 | [在线体验](https://modelscope.cn/studios/FireRedTeam/FireRedTTS3) | [`studios/FireRedTTS3`](studios/FireRedTTS3) |

更多创空间将逐个完成整理后加入。

## 目录结构

```text
modelscope_studio/
├── README.md
└── studios/
    └── <studio-name>/
        ├── README.md
        └── ...
```

## 使用说明

每个子项目有独立的依赖和运行要求。请进入对应目录阅读 README，不要假设所有创空间可以安装在同一个 Python 环境中。

模型权重通常不保存在本仓库，而是在首次运行时从 ModelScope 下载。ModelScope 创空间使用的 Secrets 和 `/mnt/workspace` 运行数据也不会复制到这里。
