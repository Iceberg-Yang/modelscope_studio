---
title: ABot-World L20N
sdk: docker
---

# ABot-World-0

基于 ABot-World-0-5B-LF 的动作可控交互式世界模型创空间。应用从单张场景图开始持续生成视频，并通过键盘动作实时控制移动和视角。

[在线体验](https://modelscope.cn/studios/amap_cvlab/abot-world-0) · [模型权重](https://modelscope.cn/models/amap_cvlab/ABot-World-0-5B-LF)

## 推理流程

```text
首帧 + 场景描述 + W/A/S/D 与 I/J/K/L 动作
  └─ UMT5-XXL 文本编码 + 参考图像编码
      └─ ABot-World Causal DiT
          └─ LongForcing + 4-step DMD 分块生成
              └─ Wan2.2 VAE / TAE 流式解码
                  └─ WebSocket JPEG 帧流 + 视频输出
```

1. 启动器检测实际 GPU、显存和 Compute Capability，并对 L20N 的 CC 12.0 运行环境执行门禁检查。
2. 首帧和文本形成参考条件，浏览器持续提交互斥的移动与视角动作。
3. Causal DiT 采用 LongForcing，以 `1000/750/500/250` 四个时间步逐块扩展潜变量序列。
4. 默认使用 FP8 per-token Linear；自注意力在 SageAttention2 与 FlashAttention2 之间根据实测结果自动选择。
5. VAE/TAE 将每个生成块解码为画面，经有界队列和 WebSocket 发送到浏览器，并保留完整视频结果。

## 技术要点

| 项目 | 配置 |
|---|---|
| 主模型 | `amap_cvlab/ABot-World-0-5B-LF`，动作条件 Causal DiT |
| 生成策略 | LongForcing · 4-step DMD · block-wise causal rollout |
| 运行硬件 | L20N 部署配置，启动时验证 CC 12.0 和 CUDA 扩展可用性 |
| 精度与量化 | 默认动态 FP8 per-token GEMM；保留 BF16 与 Hybrid A/B 模式 |
| 注意力 | FlashAttention2 SM120；SageAttention2 在实际 GPU 上编译并持久化缓存 |
| 默认分辨率 | `704 × 1280`，保持五个参考槽位和每块生成结构 |
| 流式链路 | FastAPI + Gradio + 二进制 WebSocket；自适应 10/11/12 FPS 播放与追帧 |
| 并发策略 | 单 GPU FIFO 会话租约，断开或停止后在当前推理块结束时释放 GPU |
| 运行诊断 | `/healthz` 暴露硬件、扩展和模型状态；`/metrics` 记录阶段耗时与生成 FPS |
| 持久化 | 模型缓存、SageAttention/Triton 编译产物和输出位于 `/mnt/workspace` |

## 目录

```text
.
├── entrypoint_l20n.py       # L20N 运行时门禁与服务启动
├── app.py                   # FastAPI、Gradio 和 WebSocket 入口
├── web_client/              # UI、会话调度、流式推理与指标
├── pipeline/                # Causal DiT 分块推理流程
├── wan/                     # 模型、VAE/TAE 与注意力实现
├── quantizer/               # FP8/Hybrid 量化实现
├── third_party/SageAttention/ # 随运行环境编译的注意力内核
├── scripts/                 # L20N 启动检查与传输策略验证
├── DEPLOY_L20N.md           # 部署参数和验收说明
└── Dockerfile               # CUDA 12.8 / PyTorch 2.10 镜像
```

## 本地运行

该镜像针对已经验证的 L20N CC 12.0 环境构建；默认开启严格启动门禁，不会在扩展不可用时静默降级。

```bash
docker build -t abot-world-l20n .
docker run --gpus all -p 7860:7860 \
  -v "$PWD/workspace:/mnt/workspace" abot-world-l20n
```

服务监听 `0.0.0.0:7860`。详细的运行时验证、注意力选择、量化 A/B 和流式参数见 `DEPLOY_L20N.md`。
