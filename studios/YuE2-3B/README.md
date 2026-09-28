# YuE2-3B ModelScope Studio

基于 YuE2-3B 的完整歌曲生成 Demo，可由风格描述和结构化歌词生成带人声、伴奏及 ABC 乐谱的 48 kHz 立体声歌曲，也支持根据外部旋律乐谱进行翻唱。

| 资源 | 地址 |
|---|---|
| 在线体验 | [ModelScope Studio](https://www.modelscope.ai/studios/m-a-p/YuE2-3B) |
| 主模型 | [m-a-p/YuE2-3B](https://www.modelscope.ai/models/m-a-p/YuE2-3B) |
| 音频解码器 | [m-a-p/YuE2-Vae](https://www.modelscope.ai/models/m-a-p/YuE2-Vae) |

## 功能

- 输入音乐风格与带段落标记的歌词，生成完整歌曲、可播放音频和 ABC 乐谱。
- 提供 `full`、`melody`、`off` 三种规划模式，分别控制旋律与和弦规划、仅旋律规划或直接生成。
- 接收外部旋律 ABC 乐谱，在保留旋律条件的同时改变歌词和目标风格。
- 支持固定随机种子和调整 CFG scale，结果可直接试听和下载。

## 推理方案

```text
风格描述 + 结构化歌词 + seed
              │
              v
      YuE2 tokenizer / CoT 规划
              │
              ├─ full / melody ─> 自回归生成或接收外部 ABC 乐谱
              └─ off ──────────> 跳过乐谱规划
              │
              v
  AR Mixture-of-Transformers ─> semantic audio tokens（可选 CFG）
              │
              v
  NAR Flow Matching（32-step midpoint）─> 64 维 acoustic latents
              │
              v
     YuE2-Vae FP32 分块解码 ─> 48 kHz stereo audio
```

1. 文本 tokenizer 将风格、歌词和控制标记组织为模型前缀；`full` 或 `melody` 模式先生成 ABC 符号规划，也可直接使用用户提供的乐谱。
2. 同一 AR/NAR Mixture-of-Transformers 骨干以自回归方式生成 semantic audio tokens；设置 CFG 时额外构造负向分支增强文本条件。
3. NAR 阶段复用模型骨干，通过 32 步 midpoint flow matching 将 semantic tokens 合成为 64 维连续声学 latent。
4. YuE2-Vae 保持 FP32，并以带 halo 的分块方式解码长序列，最后输出 48 kHz 双声道波形。
5. Gradio 将音频、生成耗时和 ABC 乐谱一并返回；模型实例常驻显存，队列限制为单并发。

## 技术细节

| 项目 | 配置 |
|---|---|
| 主模型 | YuE2-3B，28 层、hidden size 2048、16 attention heads / 8 KV heads |
| 推理组件 | `yue2_infer` 0.1.5 + YuE2-Vae |
| 推理框架 | PyTorch 2.10 + Transformers 4.57.6 + ModelScope |
| 精度与量化 | AR/NAR 使用 BF16，VAE 使用 FP32；默认不量化 |
| 生成策略 | ABC CoT planning + semantic AR + 32-step NAR flow matching |
| 加速方式 | PyTorch fused SDPA + CUDA Graph；不依赖外部 FlashAttention 扩展 |
| 上下文与输出 | 24,576 token context；64 维 latent；48 kHz stereo |
| 显存与硬件 | 官方单请求测试峰值约 11–14 GiB；建议 24GB 以上 BF16 NVIDIA GPU，Studio 使用 L20 48GB xGPU |
| 缓存与并发 | `/mnt/workspace/yue2-models` 持久化缓存；单模型实例、单并发 |

启动时分别从 ModelScope 下载主模型和 VAE，合计约 7.3 GiB。应用优先使用模型快照内的 `yue2_infer` wheel，并保留仓库根目录中的同版本 wheel 作为离线回退；权重和建模代码随后均从本地快照加载。

## 目录结构

```text
YuE2-3B/
├── app.py                            # 模型下载、推理封装与 Gradio 界面
├── yue2_infer-0.1.5-py3-none-any.whl # YuE2 推理包离线回退
├── examples/tonight-awake.json       # 内置歌曲示例
├── assets/yue-logo.png               # 页面 Logo
├── requirements.txt                  # Python 运行依赖
├── LICENSE                           # 模型权重许可
└── THIRD_PARTY_NOTICES.md            # 第三方组件声明
```

## 本地运行

建议使用 Linux、Python 3.10+ 和支持 BF16 的 NVIDIA GPU。PyTorch、Gradio 与 ModelScope 通常由 Studio 基础镜像提供；本地环境需要另行安装与 CUDA 匹配的 PyTorch。

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch modelscope "gradio>=6,<7"
pip install -r requirements.txt
python app.py
```

服务默认监听 `0.0.0.0:7860`。可通过 `YUE2_WORKSPACE` 修改模型缓存根目录，并通过 `GRADIO_SERVER_NAME`、`GRADIO_SERVER_PORT` 或 `PORT` 修改监听地址和端口。

## ModelScope 部署

创空间使用 Gradio 6.x，入口为 `app.py`，基础镜像为 `ubuntu22.04-py312-torch2.10.0-modelscope1.37.0`，部署在 48GB xGPU 上并限制为单并发。后台线程会在界面启动后预热模型，首次启动需下载模型，后续重启复用 `/mnt/workspace` 中的持久化缓存。
