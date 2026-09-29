---
domain: [cv]
tags: [reflection-removal, image-to-image]
models:
  - huawei-bayerlab/windowseat-reflection-removal-v1-0
  - Qwen/Qwen-Image-Edit-2509
deployspec:
  entry_file: app.py
license: Apache License 2.0
related_arxiv_id: [2512.05000]
---

# WindowSeat Reflection Removal

用于去除隔窗拍摄照片中反射影像的交互式创空间。用户上传图片后，可通过滑块放大并对比原图与清理结果。

[在线体验](https://www.modelscope.ai/studios/huawei-bayerlab/windowseat-reflection-removal) · [模型权重](https://www.modelscope.ai/models/huawei-bayerlab/windowseat-reflection-removal-v1-0)

## 推理流程

```text
含反射的输入照片
  └─ 尺寸规划与短边分块
      └─ Qwen Image VAE 编码
          └─ Qwen Image Edit Transformer + WindowSeat LoRA
              └─ 固定时间步的单步 Flow 更新
                  └─ 分块融合与 VAE 解码
                      └─ 去反射图像
```

1. 启动时从 ModelScope 准备 Qwen Image Edit 2509 的 Transformer/VAE 和 WindowSeat LoRA。
2. 输入图像按模型配置确定处理分辨率；大幅面图像采用短边分块路径，避免一次性生成超大潜变量。
3. VAE 将图像编码到潜空间，预计算的文本条件直接送入 Transformer。
4. WindowSeat LoRA 驱动一次固定时间步的速度预测，完成潜空间反射分离。
5. 各分块融合后由 VAE 解码，输出与原始尺寸对应的结果图。

## 技术要点

| 项目 | 实现 |
|---|---|
| 基础模型 | `Qwen/Qwen-Image-Edit-2509` |
| 适配权重 | `huawei-bayerlab/windowseat-reflection-removal-v1-0` |
| 精度与量化 | Transformer 使用 NF4，BF16 计算；VAE 使用 BF16 |
| 参数适配 | PEFT LoRA 注入 Qwen Image Transformer |
| 文本条件 | 权重包携带预计算 Embedding，无需常驻文本编码器 |
| 采样方式 | 固定时间步的单次 Flow 更新 |
| 大图处理 | 短边分块推理与重叠区域融合，兼顾显存占用和接缝质量 |
| 交互界面 | Gradio DualVision 提供拖动对比和局部缩放 |

## 目录

```text
.
├── app.py                       # 模型下载、界面与服务入口
├── windowseat_inference.py      # LoRA 加载、分块推理与图像融合
├── gradio_dualvision/           # 图像对比组件
├── example_images/              # 示例图片
├── assets/                      # 页面字体与图标
└── requirements.txt             # Python 依赖
```

## 本地运行

建议使用支持 BF16 与 bitsandbytes 4-bit 量化的 CUDA 环境。首次启动会从 ModelScope 下载模型。

```bash
pip install -r requirements.txt
python app.py
```

服务默认监听 `0.0.0.0:7860`。
