---
domain: [cv]
tags: [depth-estimation, normal-estimation, intrinsic-image]
models:
  - huawei-bayerlab/marigold-v2-0
  - Qwen/Qwen-Image-Edit-2509
deployspec:
  entry_file: app.py
license: Apache License 2.0
related_arxiv_id: [2609.08084]
---

# Marigold V2

基于 Marigold V2 的单图几何与材质预测创空间。上传一张图片后，可同时得到深度、透视深度、表面法线和反照率四类结果。

[在线体验](https://www.modelscope.ai/studios/huawei-bayerlab/Marigold-V2) · [模型权重](https://www.modelscope.ai/models/huawei-bayerlab/marigold-v2-0)

## 推理流程

```text
输入图像
  └─ 缩放至最长边 ≤ 1536，并对齐到 16 的倍数
      └─ Qwen Image VAE 编码
          └─ 依次加载 Depth / Layered Depth / Normals / Albedo 权重
              └─ Qwen Image Edit Transformer 单步 Flow 推理
                  └─ VAE 解码与任务专用可视化
                      └─ 四项预测结果
```

1. 应用从 ModelScope 下载 Qwen Image Edit 2509 的 Transformer、VAE，以及 Marigold V2 的任务权重。
2. 输入图像保持宽高比缩放，编码为 Qwen Image VAE 潜变量。
3. 四个任务共享一套骨干网络，推理时顺序切换 LoRA 和可选的 VAE Decoder 权重。
4. 每个任务在固定时间步 `t=0.499` 预测一次速度场，并以 `latents - velocity` 得到输出潜变量。
5. 深度结果统一色标，法线归一化到可视范围，反照率从线性 RGB 转为 sRGB。

## 技术要点

| 项目 | 实现 |
|---|---|
| 基础模型 | `Qwen/Qwen-Image-Edit-2509` |
| 任务权重 | `huawei-bayerlab/marigold-v2-0` |
| 精度与量化 | Transformer 使用 NF4，BF16 计算；VAE 使用 BF16 |
| 参数适配 | Rank 128、Alpha 128 的 LoRA，覆盖图像/文本注意力与 MLP 模块 |
| 文本条件 | 直接加载预计算 Prompt Embedding，不常驻文本编码器 |
| 推理步数 | 固定时间步的一次 Flow 更新 |
| 多任务复用 | 共享 Transformer/VAE，逐项热切换任务权重以降低常驻显存 |
| 可视化 | 双深度仿射对齐并共用色标；法线与反照率分别做物理量映射 |

## 目录

```text
.
├── app.py                    # 模型下载、Gradio 界面和服务入口
├── marigoldv2_inference.py   # 四任务加载、单步推理与可视化
├── gradio_dualvision/        # 对比查看组件
├── example_images/           # 示例图片
├── assets/                   # 页面字体与图标
└── requirements.txt          # Python 依赖
```

## 本地运行

需要支持 BF16 与 bitsandbytes 4-bit 量化的 CUDA 环境。模型权重体积较大，首次启动会从 ModelScope 下载。

```bash
pip install -r requirements.txt
python app.py
```

服务默认监听 `0.0.0.0:7860`。模型目录可通过 `QWEN_IMAGE_EDIT_URI` 和 `MARIGOLD_V2_URI` 调整。
