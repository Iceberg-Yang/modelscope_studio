# LTX-2.5 ModelScope Studio

一个面向 ModelScope xGPU 的 LTX-2.5 文生视频、图生视频 Demo。它使用 Lightricks 官方 `DistilledPipeline`，生成带同步音频的短视频，并提供简化的 Gradio 交互界面。

| 资源 | 链接 |
|---|---|
| 在线体验 | [ModelScope Studio：icebergyang/LTX2.5](https://modelscope.cn/studios/icebergyang/LTX2.5) |
| 模型 | [ModelScope：Lightricks/LTX-2.5](https://modelscope.cn/models/Lightricks/LTX-2.5) |

## 功能

- 文本生成视频，或上传一张图片作为首帧进行图生视频。
- 同时生成视频画面和同步音频，并封装为 MP4。
- 支持横屏 3:2、竖屏 2:3、方形 1:1 三种画幅。
- 支持 3 秒和 5 秒视频，对应 73 帧和 121 帧，输出帧率为 24 FPS。
- 支持固定随机种子或自动随机种子。
- 在界面中返回实际种子、分辨率、帧数、耗时和 PyTorch 统计的 GPU 峰值显存。

## 推理方案

```text
文本提示词 ───────────────┐
                          ├─> Gemma 4 12B 文本编码器
可选首帧 ─> 缩放并中心裁剪 ┘
                                  │
                                  ▼
                    LTX-2.5 Distilled Transformer
                     第一阶段：低分辨率联合音视频扩散
                                  │
                                  ▼
                          2× latent 空间上采样
                                  │
                                  ▼
                     第二阶段：目标分辨率细化去噪
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
              Conv Video VAE               Audio VAE
                    └─────────────┬─────────────┘
                                  ▼
                         PyAV 编码为 MP4
```

运行时按需从 `Lightricks/LTX-2.5` 下载五个拆分组件：

1. 22B distilled Transformer BF16 权重。
2. LTX 专用 Gemma 4 12B 文本编码器及投影层。
3. 卷积 Video VAE。
4. Audio VAE。
5. 2× latent 空间上采样器。

模型文件不进入 Git 仓库。默认缓存在 `/mnt/workspace/models/ltx-2.5`，以便 ModelScope 创空间重启后复用。

## 技术细节

| 项目 | 当前实现 |
|---|---|
| UI / SDK | Gradio 5.49.1，入口 `app.py` |
| 推理实现 | Lightricks 官方 `DistilledPipeline` |
| 权重与计算精度 | BF16 |
| 量化 | 无；`quantization=None` |
| 推理阶段 | distilled 两阶段：低分辨率生成、2× latent 上采样与目标分辨率细化 |
| 注意力后端 | 未额外安装 FlashAttention；官方代码在此环境下自动回退到 PyTorch SDPA |
| Video VAE | 卷积版 Video VAE，避免依赖 NATTEN |
| 并发 | Gradio 队列与服务内部锁均限制为单任务 |
| 模型下载 | `modelscope.snapshot_download`，仅拉取所需五个文件 |
| 输出 | 24 FPS MP4，包含视频流和模型生成音频 |

### 显存与 offload

`LTX_OFFLOAD=auto` 根据 PyTorch 实际可见显存选择策略：

- 可见显存不低于 75 GiB：使用 `OffloadMode.NONE`。DiT Transformer 在单个扩散阶段内完整驻留 GPU；官方 Pipeline 会在阶段结束时释放相应模型。
- 44–75 GiB：使用 `OffloadMode.CPU`，由官方 block streaming 机制逐层从 CPU 向 GPU 流送权重。
- 低于 44 GiB：当前应用会拒绝启动推理。

本项目的目标部署配置是单卡 NVIDIA A100 80GB、PyTorch 2.10.0、CUDA 12.8。代码保留了 48GB 级 GPU 的 CPU offload 路径，但本次 GitHub 快照整理没有重新进行 48GB 或 A100 GPU 端到端实测，因此不能把代码阈值理解为完整的性能保证。

### 模型生命周期

- `LTX25Service` 在首次请求时延迟下载模型并构造 Pipeline。
- Pipeline 对象在进程内复用。
- 官方 Pipeline 的各计算阶段按需构建、释放模型；这里没有实现跨阶段永久驻留。
- 每次任务前后会清理 Python GC 和 PyTorch CUDA cache，并记录 `max_memory_allocated()`。
- 输出目录最多保留最近 20 个 MP4；上传首帧产生的临时裁剪图在任务结束后删除。

## 目录结构

```text
LTX2.5/
├── app.py                 # Gradio UI、输入校验、队列和启动入口
├── inference.py           # 模型下载、Pipeline 构建、推理与输出管理
├── requirements.txt       # ModelScope 目标运行环境依赖
├── tests/test_config.py   # 帧数、路径与 offload 配置单元测试
└── vendor/
    ├── UPSTREAM.md        # 固定上游 commit 与 wheel 校验值
    ├── LICENSE.LTX-2.md   # LTX-2 Community License
    ├── ltx-core/          # 固定版本的官方核心推理源码
    ├── ltx-pipelines/     # 固定版本的官方 Pipeline 源码
    └── wheels/            # 由上述源码构建的纯 Python wheel
```

Vendored 源码和 wheel 用于避免 ModelScope 构建阶段依赖 GitHub 网络。两个 wheel 的来源与 SHA-256 见 [`vendor/UPSTREAM.md`](vendor/UPSTREAM.md)。

## 本地运行

### 环境要求

- Linux
- Python 3.12
- 支持 BF16 的 NVIDIA CUDA GPU
- 目标环境为 80GB 显存；48GB 级 GPU 需要 CPU offload
- 至少约 80 GiB 可用磁盘空间，用于模型缓存和生成结果

安装依赖：

```bash
python -m pip install -r requirements.txt
```

启动：

```bash
python app.py
```

默认监听 `0.0.0.0:7860`。首次推理会从 ModelScope 下载模型组件，耗时取决于网络和缓存状态。

## 环境变量

所有配置均为可选项，不需要 API Key 或其他 Secret。

| 名称 | 默认值 | 用途 |
|---|---|---|
| `LTX_PERSISTENT_ROOT` | `/mnt/workspace` | 持久化缓存根目录 |
| `LTX_FALLBACK_ROOT` | `/tmp/ltx25-workspace` | 持久化目录不可写时的回退目录 |
| `LTX_MODEL_DIR` | `<persistent-root>/models/ltx-2.5` | 模型组件目录 |
| `LTX_OUTPUT_DIR` | `<persistent-root>/outputs/ltx25` | 生成结果目录 |
| `LTX_OFFLOAD` | `auto` | `auto`、`none`、`cpu` 或 `disk` |
| `PORT` | `7860` | Gradio 监听端口 |
| `LOG_LEVEL` | `INFO` | Python 日志级别 |

如果 `/mnt/workspace` 不存在或不可写，应用自动使用 `LTX_FALLBACK_ROOT`。回退目录通常不具备跨重启持久化能力。

## ModelScope 部署配置

| 配置 | 建议值 |
|---|---|
| SDK | Gradio 5.49.1 |
| 入口文件 | `app.py` |
| 基础镜像 | `ubuntu22.04-py312-torch2.10.0-modelscope1.37.0` |
| 目标硬件 | NVIDIA A100 80GB |
| 端口 | `7860` |
| 持久化目录 | `/mnt/workspace` |
| 并发 | 1 |

`requirements.txt` 固定 PyTorch、Torchaudio、Transformers、NumPy 和 ModelScope 版本，以减少平台构建时的依赖漂移。若目标节点的 CUDA/PyTorch 组合与上述镜像不同，应先重新验证依赖兼容性。
