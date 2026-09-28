---
title: SenseNova U1.5
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.2.0
python_version: '3.12'
app_file: app.py
pinned: false
short_description: Image generation & editing with SenseNova-U1.5
startup_duration_timeout: 1h
---

# SenseNova-U1.5-8B-MoT ModelScope Studio

基于 SenseNova-U1.5-8B-MoT 的统一图像生成与编辑 Demo，面向中英文海报、图表、知识插图等文字密集型内容，并支持自然语言指令编辑。

| 资源 | 地址 |
|---|---|
| 在线体验 | [ModelScope Studio](https://modelscope.cn/studios/SenseNova/sensenova-u1.5) |
| 模型 | [SenseNova/SenseNova-U1.5-8B-MoT](https://modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT) |

## 功能

- 根据中英文描述生成信息图、图表、海报和知识插图。
- 根据参考图和自然语言指令修改文字、内容、风格或布局。
- 提供 7 种文生图宽高比，最高边可达 2720 像素。
- 编辑模式自动保持输入图宽高比，并将输入和输出控制在约 4MP 像素预算内。
- 支持固定或随机 seed，便于复现和探索不同结果。

## 推理方案

```text
文本提示词 ─> tokenizer ─> Qwen3-style MoT 理解/生成骨干 ─> 条件 KV cache ─┐
空提示词 ───────────────────────────────────────────────> 无条件 KV cache ─┤
参考图（编辑模式）─> NEO Vision 16×16 patch encoder ─> 图像条件 KV cache ─┤
                                                                            v
RGB 高斯噪声 ─> generation expert + Flow-Matching head + CFG ─> Euler 更新
                                                                            │
                                                                            v
                                         反归一化 RGB tensor ─> PNG / PIL 图像
```

1. 文本经对话模板和 tokenizer 编码；`think_mode=False` 时直接插入空思考段并进入图像生成标记。
2. 文生图建立条件与无条件两套 KV cache；编辑模式额外编码参考图，并建立完整条件、仅图像条件和全无条件分支。
3. 模型从按分辨率动态缩放的 RGB 高斯噪声开始，以 32×32 生成 token 网格反复提取图像特征和时间步嵌入。
4. MoT 生成专家与 Flow-Matching head 预测速度场，按 timestep shift 3.0 的时间表执行 50 步 Euler 更新。
5. CFG 4.0 融合条件分支，并以全局范数抑制过强引导；最终 RGB tensor 反归一化后直接转为图像，不经过独立 VAE。

## 技术细节

| 项目 | 配置 |
|---|---|
| 模型架构 | NEO-Unify 8B MoT，Qwen3-style language backbone + NEO Vision + Flow-Matching modules |
| 专家路径 | 理解专家处理文本/参考图上下文，生成专家负责图像 token 与速度场预测 |
| 图像表示 | 原生 RGB latent；16×16 vision patch，经 0.5 downsample 后形成 32×32 生成网格 |
| 推理框架 | PyTorch + Transformers 4.57.1 + ModelScope |
| 精度 | BF16，全模型加载到 CUDA；无量化 |
| 默认采样 | 50 steps、text CFG 4.0、image CFG 1.0、global CFG norm、timestep shift 3.0 |
| 注意力 | 自动选择 FlashAttention；未安装时回退 PyTorch SDPA |
| 文生图尺寸 | 2048×2048、2720×1536、1536×2720 等 7 个训练宽高比桶 |
| 编辑规格 | 输入与输出约 2048×2048 像素预算，尺寸对齐 32 的倍数 |
| UI 与调度 | Gradio 6.2.0；模型单实例常驻 CUDA |

应用启动时优先从 ModelScope 下载模型；若 `/mnt/workspace` 可用，则缓存到 `/mnt/workspace/modelscope-cache`。模型架构注册代码随仓库提供，因此 `AutoConfig` 和 `AutoModel` 可以直接加载 NEO-Unify checkpoint。

## 目录结构

```text
sensenova-u1.5/
├── app.py                       # 模型加载、生成/编辑入口与 Gradio UI
├── sensenova_u1/
│   ├── models/neo_unify/        # NEO-Unify、Qwen3 MoT、Vision 与 Flow Matching
│   ├── prompt_enhance/          # 可复用的提示词增强模块
│   └── utils/                   # LoRA、模型统计和性能分析工具
├── assets/                      # 编辑示例图
├── cover.png                    # 创空间封面
└── requirements.txt             # Python 依赖
```

## 本地运行

建议使用 Linux、Python 3.12 和支持 BF16 的 CUDA GPU，并预先安装与 CUDA 匹配的 PyTorch 和 Gradio。

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch gradio==6.2.0 modelscope
pip install -r requirements.txt
python app.py
```

可通过 `MODELSCOPE_CACHE` 指定模型缓存目录。服务由 Gradio 启动；在 ModelScope 环境中会直接使用本地 CUDA，不依赖 Hugging Face ZeroGPU 装饰器。

## ModelScope 部署

创空间使用 Gradio 6.2.0，入口为 `app.py`，启动时以 BF16 将完整模型加载到 CUDA。模型权重不保存在代码仓库中，首次启动从 ModelScope 下载并写入持久化缓存；生成和编辑均为高分辨率 50 步推理，应使用具备充足显存的 xGPU 环境。
