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
| YuE2-3B | 文生歌曲 / ABC 乐谱规划 / 翻唱 | m-a-p/YuE2-3B · YuE2-Vae | AR/NAR MoT · Flow Matching · CUDA Graph · BF16 | [在线体验](https://www.modelscope.ai/studios/m-a-p/YuE2-3B) | [`studios/YuE2-3B`](studios/YuE2-3B) |
| LLaDA-Image | 文生图 / 指令图像编辑 | inclusionAI/LLaDA-Image 6B | LLaDA2 MoE · SigVQ · Flow Matching · FP8/BF16 | [在线体验](https://modelscope.cn/studios/icebergyang/LLaDA-Image) | [`studios/LLaDA-Image`](studios/LLaDA-Image) |
| SenseNova-U1.5-8B-MoT-Preview | 信息图生成 / 指令图像编辑 | SenseNova-U1.5-8B-MoT-Preview | NEO-Unify MoT · RGB Flow Matching · BF16 | [在线体验](https://modelscope.cn/studios/SenseNova/sensenova-u1.5-preview) | [`studios/sensenova-u1.5-preview`](studios/sensenova-u1.5-preview) |
| Irodori-TTS v4.1 Anime | 日语 TTS / 风格控制 / 音色克隆 | phasefield-audio/Irodori-TTS-v4.1-Anime | RF-DiT · Semantic-DACVAE · BF16 | [在线体验](https://modelscope.cn/studios/icebergyang/aratako-irodori-tts-v4-small-quantized) | [`studios/aratako-irodori-tts-v4-small-quantized`](studios/aratako-irodori-tts-v4-small-quantized) |
| SenseNova-U1.5-8B-MoT | 信息图生成 / 指令图像编辑 | SenseNova-U1.5-8B-MoT | NEO-Unify MoT · RGB Flow Matching · BF16 | [在线体验](https://modelscope.cn/studios/SenseNova/sensenova-u1.5) | [`studios/sensenova-u1.5`](studios/sensenova-u1.5) |
| Cosmos3-Edge | 图片 / 视频 / Physical AI 推理 | nv-community/Cosmos3-Edge | Nemotron Reasoner · SDPA · BF16 · Streaming | [在线体验](https://modelscope.cn/studios/nv-community/Cosmos3) | [`studios/Cosmos3`](studios/Cosmos3) |
| TIPSv2 Feature Explorer | 特征可视化 / 分割 / 深度 / 法线 | Google TIPSv2 + DPT | ViT · SDPA · PCA · K-means · FP16 | [在线体验](https://modelscope.cn/studios/icebergyang/TIPSv2) | [`studios/TIPSv2`](studios/TIPSv2) |
| Ming-Image 0.1 Design | 文生设计图 / 文字渲染 / RGBA | inclusionAI/Ming-Image-0.1-Design | BailingMoE-v2 · Z-Image Flow DiT · BF16 | [在线体验](https://modelscope.cn/studios/inclusionAI/ming-image-0-1-design-demo) | [`studios/ming-image-0-1-design-demo`](studios/ming-image-0-1-design-demo) |
| Breeze TTS 2 | 音色设计 / 音色克隆 / 音色引导 | BreezeBlue/Breeze-TTS-2 | AR Backbone · Depth Decoder · 32 Codebooks · BF16 | [在线体验](https://modelscope.cn/studios/BreezeBlue/breeze-tts-2-demo) | [`studios/breeze-tts-2-demo`](studios/breeze-tts-2-demo) |
| AnimeGen T2V | 动漫风格文生视频 | Wan2.2-T2V-A14B · AnimeGen-T2V | Dual Transformer · Lightning LoRA · FP8/BF16 | [在线体验](https://modelscope.cn/studios/icebergyang/AnimeGen_T2V) | [`studios/AnimeGen_T2V`](studios/AnimeGen_T2V) |
| MiniMax-H3 NF4 Turbo | 文生音视频 / 首尾帧约束 | DiffSynth-Studio/MiniMax-H3-NF4 | DiffSynth · NF4/BF16 · Turbo LoRA · Offload | [在线体验](https://modelscope.cn/studios/MiniMax/MiniMax-H3-NF4) | [`studios/MiniMax-H3-NF4`](studios/MiniMax-H3-NF4) |
| Anima 2.9B | 动漫 / 插画文生图 | Gazingstars123/Anima-2.9B | ComfyUI · Qwen3 Encoder · Qwen Image VAE · BF16 | [在线体验](https://modelscope.cn/studios/icebergyang/Anima-2.9B) | [`studios/Anima-2.9B`](studios/Anima-2.9B) |
| Ornith-1.5-9B | 多轮推理对话 / 流式生成 | ornith-ai/Ornith-1.5-9B | Transformers · BF16 · 32K Context · Streaming | [在线体验](https://modelscope.cn/studios/icebergyang/ornith-1-5-9b) | [`studios/ornith-1-5-9b`](studios/ornith-1-5-9b) |
| ABot-World-0 | 动作可控交互世界生成 | amap_cvlab/ABot-World-0-5B-LF | Causal DiT · LongForcing · 4-step DMD · BF16 | [在线体验](https://modelscope.cn/studios/amap_cvlab/abot-world-0) | [`studios/abot-world-0`](studios/abot-world-0) |
| MiniMax Music 3 | 文生歌曲 / 歌词生成 / 流式播放 | MiniMax/MiniMax-Music3 | AR LM · RVQ · Flow Matching DiT · BF16 | [在线体验](https://modelscope.cn/studios/MiniMax/minimax-music3) | [`studios/minimax-music3`](studios/minimax-music3) |

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
