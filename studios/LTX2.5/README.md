# LTX-2.5 ModelScope Studio

基于 LTX-2.5 的文生视频、图生视频 Demo，可联合生成视频画面和同步音频。

| 资源 | 地址 |
|---|---|
| 在线体验 | [ModelScope Studio](https://modelscope.cn/studios/icebergyang/LTX2.5) |
| 模型 | [Lightricks/LTX-2.5](https://modelscope.cn/models/Lightricks/LTX-2.5) |

## 功能

- 根据文本生成视频，或上传图片作为首帧进行图生视频。
- 同步生成画面与音频，并封装为 24 FPS MP4。
- 支持 3:2、2:3、1:1 画幅以及 3 秒、5 秒两种时长。
- 支持固定或随机种子，并显示耗时与 GPU 峰值显存。

## 推理方案

```text
文本 ─> Gemma 4 12B 文本编码器 ───────────────┐
可选首帧 ─> 中心裁剪与缩放 ───────────────────┤
                                               v
                                LTX-2.5 Distilled Transformer
                                低分辨率联合音视频扩散
                                               │
                                               v
                                    2× latent 空间上采样
                                               │
                                               v
                                      目标分辨率细化去噪
                                         ┌─────┴─────┐
                                         v           v
                                  Conv Video VAE  Audio VAE
                                         └─────┬─────┘
                                               v
                                             MP4
```

1. 文本由 LTX 专用 Gemma 4 12B 编码；图生视频时，首帧同时作为视觉条件。
2. `DistilledPipeline` 先在低分辨率 latent 中联合生成视频和音频。
3. latent 经 2× 空间上采样后，在目标分辨率执行第二阶段细化去噪。
4. Video VAE 与 Audio VAE 分别解码画面和声音，最后合成为 MP4。
5. Pipeline 首次请求时延迟加载，随后在进程内复用；队列与服务锁限制为单任务。

## 技术细节

| 项目 | 配置 |
|---|---|
| 推理实现 | LTX `DistilledPipeline` |
| 核心组件 | 22B distilled Transformer、Gemma 4 12B、Video VAE、Audio VAE、2× upsampler |
| 推理阶段 | 低分辨率扩散 → latent 上采样 → 高分辨率细化 |
| 精度与量化 | BF16；无量化 |
| 注意力后端 | PyTorch SDPA |
| 显存策略 | ≥75 GiB 全 GPU；44–75 GiB 使用 block-streaming CPU offload |
| 输出规格 | 73/121 帧，24 FPS，视频与同步音频 |
| 推荐硬件 | NVIDIA A100 80GB；48 GB GPU 使用 CPU offload |
| UI 与并发 | Gradio 5.49.1，单并发 |

运行时仅下载五个必需模型组件，默认缓存在 `/mnt/workspace/models/ltx-2.5`。每次任务结束后会清理 CUDA cache；输出目录仅保留最近 20 个 MP4。

## 目录结构

```text
LTX2.5/
├── app.py              # Gradio 界面与启动入口
├── inference.py        # 模型下载、Pipeline 和输出管理
├── requirements.txt    # 运行依赖
├── tests/              # 配置测试
└── vendor/             # 固定版本的 LTX 推理源码与 wheel
```

## 本地运行

建议使用 Linux、Python 3.12、支持 BF16 的 NVIDIA GPU，并为模型缓存预留约 80 GiB 磁盘空间。

```bash
python -m pip install -r requirements.txt
python app.py
```

首次推理会从 ModelScope 下载模型组件，服务默认监听 `0.0.0.0:7860`。

## 环境变量

| 变量 | 默认值 | 用途 |
|---|---|---|
| `LTX_PERSISTENT_ROOT` | `/mnt/workspace` | 持久化缓存根目录 |
| `LTX_FALLBACK_ROOT` | `/tmp/ltx25-workspace` | 持久化目录不可用时的回退目录 |
| `LTX_MODEL_DIR` | `<persistent-root>/models/ltx-2.5` | 模型组件目录 |
| `LTX_OUTPUT_DIR` | `<persistent-root>/outputs/ltx25` | 视频输出目录 |
| `LTX_OFFLOAD` | `auto` | `auto`、`none`、`cpu` 或 `disk` |
| `PORT` | `7860` | Gradio 端口 |

## ModelScope 部署

创空间使用 Gradio 5.49.1，入口为 `app.py`，基础镜像为 `ubuntu22.04-py312-torch2.10.0-modelscope1.37.0`，端口为 `7860`。模型权重不保存在代码仓库中，首次运行时下载到 `/mnt/workspace` 并在后续启动中复用。
