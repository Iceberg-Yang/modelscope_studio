---
title: LLaDA-Image 图像生成与编辑
colorFrom: purple
colorTo: green
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
license: apache-2.0
---

# LLaDA-Image ModelScope Studio

基于 LLaDA-Image 6B 统一图像模型的在线 Demo，在同一套推理管线中提供中英文文生图和参考图指令编辑，并支持 Turbo 与 Base 两种生成档位。

| 资源 | 地址 |
|---|---|
| 在线体验 | [ModelScope Studio](https://modelscope.cn/studios/icebergyang/LLaDA-Image) |
| 模型 | [inclusionAI/LLaDA-Image](https://modelscope.cn/models/inclusionAI/LLaDA-Image) |

## 功能

- 根据中英文提示词生成图像，支持负向提示词、随机种子、尺寸、步数和 CFG 调节。
- 输入参考图和编辑指令，联合图像语义与源图 latent 完成指令编辑。
- Turbo 使用 4 步蒸馏推理，Base 使用 50 步 CFG 推理。
- 在同一进程中按需切换两个模型变体，并复用持久化下载与转换缓存。

## 推理方案

```text
文本提示词 ─> LLaDA2 MoE Text Encoder ─> QueryFormer + Text Projection ─┐
                                                                        │
参考图（编辑模式）─> SigVQ semantic features ───────────────────────────┤
                  └> Flux2 VAE Encoder ─> source latents ───────────────┤
                                                                        v
随机噪声 ─> 30-layer LLaDA Flow-Matching Transformer + CFG ─> image latents
                                                                        │
                                                                        v
                                              Flux2 VAE Decoder ─> PIL 图像
```

1. LLaDA2 MoE 文本编码器处理提示词，QueryFormer 追加可学习查询，Text Projection 将隐藏状态映射为去噪器的 caption features。
2. 编辑模式额外用 SigVQ 提取参考图语义特征，并用 Flux2 VAE 编码、patchify 和归一化源图 latent。
3. 推理从随机噪声开始，FlowMatchEulerDiscreteScheduler 按模型配置构造 sigma 时间表。
4. 30 层 Transformer 联合文本条件、编辑条件和源图 latent 预测 flow；Base 默认以 CFG 5.0 运行 50 步，Turbo 默认以 guidance 1.0 运行 4 步。
5. 去噪后的 latent 经反归一化和 unpatchify，再由 Flux2 VAE 解码为最终图像。

## 技术细节

| 项目 | 配置 |
|---|---|
| 模型组件 | LLaDA2 MoE、QueryFormer、Text Projection、SigVQ、Flow Transformer、Flux2 VAE |
| 去噪器 | 30 层、hidden size 3840、30 attention heads、128-channel patchified latent |
| 推理框架 | PyTorch 2.8 + Diffusers 0.39.0 + Transformers 4.57.6 + ModelScope |
| 模型变体 | Turbo-FP8：4 steps / guidance 1.0；Base-FP8：50 steps / CFG 5.0 |
| 精度路径 | FP8 checkpoint；DiT 按 128×128 block 反量化为 BF16，其他视觉组件使用 BF16 |
| 编辑条件 | SigVQ semantic features + Flux2 VAE source latents |
| 调度与输出 | Flow Matching scheduler；常用约 1MP 输出，文生图尺寸为 16 的倍数、编辑尺寸为 32 的倍数 |
| 显存策略 | 单变体驻留 GPU；切换时卸载或暂存另一变体；OOM 时文本编码器回退 CPU |
| 硬件与并发 | ModelScope L20 48GB xGPU；单进程、单并发 |
| 缓存 | 模型位于 `/mnt/workspace/modelscope-cache`，反量化 DiT 位于 `/mnt/workspace/llada-converted` |

应用默认下载约 27.6 GB 的单个 FP8 变体。首次加载会把 FP8 DiT 分片逐块反量化、拆分融合的 QKV/MoE 权重并写入 BF16 转换缓存，后续启动直接复用；两个档位默认不会同时占用 GPU。

## 目录结构

```text
LLaDA-Image/
├── app.py                              # 模型下载、FP8 转换、显存调度与 Gradio UI
├── src/
│   ├── models/transformer_llada_image.py # Transformer、QueryFormer、Projection 与 SigVQ
│   └── pipelines/pipeline_llada_image.py # 文生图与编辑推理管线
├── requirements.txt                    # 推理依赖
└── LEGAL.md                            # 法律声明
```

## 本地运行

建议使用 Linux、Python 3.11、CUDA 12.4 和支持 BF16 的 NVIDIA GPU。PyTorch 需根据本机 CUDA 环境单独安装。

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch gradio==6.17.3
pip install -r requirements.txt
python app.py
```

服务默认监听 `0.0.0.0:7860`。可通过 `LLADA_PRECISION`、`LLADA_MOE_BACKEND`、`LLADA_KEEP_BOTH`、`LLADA_PREFETCH_ALL` 和 `LLADA_WORKSPACE` 调整精度、MoE 后端、变体驻留及缓存目录。

## ModelScope 部署

创空间使用 Gradio 6.17.3，入口为 `app.py`，基础镜像提供 PyTorch 2.8 与 CUDA 12.4，部署在 L20 48GB xGPU 上。后台线程会优先预热 Turbo 模型，Base 在首次选择时加载；生成队列限制为单并发，模型权重和转换产物均保存在 `/mnt/workspace`。
