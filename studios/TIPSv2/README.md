# TIPSv2 Feature Explorer ModelScope Studio

基于 Google TIPSv2 视觉表征模型的特征探索 Demo，统一提供 PCA/K-means 可视化、零样本分割、ADE20K 分割、深度估计和表面法线预测。

| 资源 | 地址 |
|---|---|
| 在线体验 | [ModelScope Studio](https://modelscope.cn/studios/icebergyang/TIPSv2) |
| 默认模型 | [google/tipsv2-l14](https://modelscope.cn/models/google/tipsv2-l14) |

## 功能

- 在 B/14、L/14、SO400m/14 和 g/14 四种视觉编码器之间切换。
- 将 patch features 投影为 PCA 颜色图或进行 GPU K-means 聚类。
- 输入自定义类别名称进行开放词汇零样本分割。
- 使用对应 DPT head 预测 ADE20K 语义分割、相对深度和表面法线。
- 在 224–1792 像素范围内选择推理分辨率。

## 推理方案

```text
输入图像 ─> resize / normalize ─> TIPSv2 ViT encoder ─> patch features ─┬─> PCA / K-means
                                                                         ├─> text similarity ─> zero-shot mask
                                                                         └─> DPT head ─┬─> depth
                                                                                       ├─> normals
                                                                                       └─> ADE20K logits
```

1. 图像按选定分辨率预处理，TIPSv2 ViT 将其编码为全局和空间 patch features。
2. 特征探索路径对空间特征做 PCA，或使用 `fast_pytorch_kmeans` 在 GPU 上聚类并映射为颜色。
3. 零样本分割将类别提示模板编码为文本向量，与归一化图像 patch 特征计算相似度并上采样到原图。
4. 稠密预测路径把对应 backbone 与 DPT head 组合，分别输出深度、法线或 ADE20K 类别 logits。
5. 切换模型变体时主动释放旧模型和 CUDA cache，避免多个大模型同时占用显存。

## 技术细节

| 项目 | 配置 |
|---|---|
| 模型族 | TIPSv2 B/14、L/14、SO400m/14、g/14 及对应 DPT checkpoints |
| Patch size | 14 |
| 推理框架 | PyTorch + Transformers remote code + ModelScope |
| 精度 | 默认 FP16；可通过环境变量切换 BF16 或 FP32 |
| 注意力 | 将兼容模块替换为 PyTorch fused SDPA，降低高分辨率注意力峰值 |
| 特征分析 | scikit-learn PCA + GPU K-means |
| 零样本分割 | 多模板文本 embedding 与 patch cosine similarity |
| 稠密任务 | DPT depth / normals / ADE20K 150-class segmentation |
| 硬件与并发 | L20 48GB xGPU；单变体驻留、单并发 |
| 缓存 | `/mnt/workspace/.cache/modelscope` |

模型仅通过 ModelScope SDK 下载并从本地 snapshot 加载；应用同时设置 Transformers 离线模式，避免运行阶段意外访问其他模型托管站点。

## 目录结构

```text
TIPSv2/
├── app.py          # 模型加载、兼容补丁、特征分析和 Gradio UI
├── assets/         # 创空间封面图
└── requirements.txt
```

## 本地运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

环境变量包括 `MODELSCOPE_CACHE`、`MODELSCOPE_MODEL_REVISION`、`TIPS_DTYPE` 和 `PORT`。服务默认监听 `0.0.0.0:7860`。

## ModelScope 部署

创空间使用 Gradio 6.17.3，入口为 `app.py`，推荐 L20 48GB xGPU。默认加载 L/14；其他 backbone 和 DPT checkpoint 在首次选择时下载，队列限制为单并发。
