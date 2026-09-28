# Ming-Image 0.1 Design ModelScope Studio

面向 UI、信息图和海报等文字密集型视觉设计的文生图 Demo，可直接生成包含标题文字的完整画面，并输出带透明通道的 RGBA 图像。

| 资源 | 地址 |
|---|---|
| 在线体验 | [ModelScope Studio](https://modelscope.cn/studios/inclusionAI/ming-image-0-1-design-demo) |
| 模型 | [inclusionAI/Ming-Image-0.1-Design](https://modelscope.cn/models/inclusionAI/Ming-Image-0.1-Design) |

## 功能

- 根据设计需求生成 UI 界面、信息图、海报和编辑视觉稿。
- 强调标题、标签和数据文字的准确渲染。
- 使用 4 通道 VAE 输出 RGBA 图像，保留透明背景。
- 提供 1024×1024 和 2048×2048 两档分辨率。
- 调节采样步数、CFG 和随机种子。

## 推理方案

```text
原始提示词 + 256 learnable queries
                │
                v
       BailingMoE-v2 Thinker（20 层 / 256 experts）
                ├─ query hidden states ─> proj_in ─> Qwen2 connector ─> proj_out
                └─ layer 5 / 12 / final text states ─> direct VLM projection
                                      │
                                      v
随机 latent ─> Z-Image Flow-Matching DiT（30 层）+ CFG ─> RGBA latent
                                      │
                                      v
                         AutoencoderKLQwenImage ─> RGBA PNG
```

1. Tokenizer 编码原始提示词，并在序列末尾追加 256 个可学习 query token。
2. BailingMoE-v2 thinker 产生两路条件：query 流负责整体视觉语义，direct 流拼接第 5 层前、第 12 层前和最终文本状态以保留精确文字信息。
3. 两路条件分别投影到 2560 和 3840 维后拼接，并在 padding 位置清零和屏蔽。
4. 30 层 Z-Image DiT 使用动态 shifting 的 FlowMatchEulerDiscreteScheduler 去噪；CFG 大于 0 时使用全零负条件。
5. 4 通道 Qwen-Image VAE 解码 latent，最终以 RGBA PNG 返回。

## 技术细节

| 项目 | 配置 |
|---|---|
| 模型规模 | 约 6B；完整 BF16 权重约 53 GB |
| 文本条件 | BailingMoE-v2 thinker，20 层、256 experts、256 query tokens |
| 生成器 | Z-Image Flow-Matching DiT，dim 3840、30 层、16-channel latent |
| 解码器 | `AutoencoderKLQwenImage`，4 通道 RGBA |
| 推理框架 | PyTorch + Transformers 5.17 + Diffusers 0.40 + ModelScope |
| 精度与量化 | BF16；无量化 |
| 默认采样 | 12 steps，CFG 1.0，动态 scheduler shifting |
| 显存策略 | Transformer、connector 和 VAE 常驻 GPU；thinker 仅在提示词编码阶段从 meta 分片载入 CUDA，完成后释放 |
| 硬件与并发 | L20 48GB xGPU；单并发 |
| 缓存 | 权重下载至 `/mnt/workspace` 持久卷 |

## 目录结构

```text
ming-image-0-1-design-demo/
├── app.py                         # 分阶段加载、生成入口与 Gradio UI
├── ming/
│   ├── pipeline.py                # 双路文本条件、显存调度和采样流程
│   ├── modeling_bailing_moe_v2.py # Thinker 架构
│   └── transformer_z_image.py     # Z-Image Flow Transformer
├── assets/                        # RGBA 与设计示例
└── requirements.txt
```

## 本地运行

建议使用 Linux、Python 3.12、64GB 以上主机内存和 48GB 级 CUDA GPU。

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch modelscope
pip install -r requirements.txt
python app.py
```

服务监听 `0.0.0.0:7860`，模型首次启动需要下载约 53GB 权重。

## ModelScope 部署

创空间使用 Gradio，入口为 `app.py`，部署在 L20 48GB xGPU 上。默认限制单并发、最多排队 4 个请求；1024 分辨率为推荐档，2048 分辨率需要更长推理时间和更高峰值显存。
