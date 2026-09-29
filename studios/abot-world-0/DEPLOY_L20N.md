# ABot-World L20N Docker 部署

该目录基于官方 `amap-cvlab/ABot-World` 的 5090 版本，保留原始
FP8 动态量化路径，并加入 ModelScope Docker、SM120 扩展编译、
WebSocket 低延迟显示和单 GPU 公平排队。

基础源码快照为官方仓库 `main` 的
`e2a406a176195d994bcaf5ebb92b2047edcd3c2b`，其中 FP8 即为默认配置。
Docker 只安装
`requirements-l20n.txt`；根目录的原始 `requirements.txt` 不参与镜像构建。

## 首次启动

ModelScope 的 Docker 构建阶段通常看不到 GPU。这个 L20N 镜像已知部署
目标是 CC 12.0，因此构建阶段会从官方源码预先编译
`flash-attn==2.8.3`，并通过 `FLASH_ATTN_CUDA_ARCHS=120` 只生成 SM120
wheel。它提供 ABot 交叉注意力直接使用的 `flash_attn_varlen_func`。

SageAttention 仍然依据容器实际获得的 GPU 编译。容器得到 GPU 后，
`entrypoint_l20n.py` 会：

1. 读取卡名、显存和 compute capability。
2. 仅接受本次已经实测的 CC 12.0；其他 CC 明确失败，避免加载错误 wheel。
3. 在真实 GPU 上运行 FlashAttention2 SM120 varlen forward 自检。
4. 设置 `TORCH_CUDA_ARCH_LIST=12.0`，由 SageAttention 的 Blackwell
   构建规则生成对应内核。
5. 编译 SageAttention2，并把 wheel 缓存到
   `/mnt/workspace/abot-world/extensions`。
6. 启动 Web UI，随后后台下载并加载模型。

第一次运行会比后续运行慢。只要 `/mnt/workspace` 保留，完全相同的
Torch、CUDA、CC 和 SageAttention 源码会复用缓存 wheel。
编译期间端口已经开放，`/healthz` 的 `service_phase` 为
`l20n_preflight`，并持续显示 GPU 信息和当前编译阶段。

生产配置默认 `ABOT_REQUIRE_SAGEATTENTION=1`：实际卡不是 CC 12.0，
或者 SageAttention2 编译/自检失败时，容器会明确失败，不会悄悄以慢速
后端上线。仅排障时才可临时设为 `0`；错误仍会保留在 `/healthz` 和日志中。

启动门禁会实际运行 FlashAttention2 varlen forward、SageAttention2、
FP8 per-token Linear、Triton RMSNorm 和 Triton RoPE 的 CUDA 测试。
FlashAttention 使用首次真实推理的 3920/512 序列长度、24 heads 和
128 head dimension；Triton JIT 产物持久化在
`/mnt/workspace/abot-world/extensions/triton`，重启后可复用。

## `/healthz` 验收

部署后访问：

```bash
curl -sS https://<studio-domain>/healthz
```

硬件与扩展验收至少满足：

```text
gpu.name: 实际 L20N 卡名
gpu.total_gib: 与容器内实际分配一致
gpu.compute_capability: 12.0
runtime_bootstrap.phase: ready
runtime_bootstrap.flashattention2_available: true
runtime_bootstrap.flashattention2_version: 2.8.3
runtime_bootstrap.sageattention_available: true
attention.sageattention2: true
attention.sageattention3: false
attention.default_backend: sageattn 或 flash_attn_2（由实测选择）
```

模型加载完成后继续满足：

```text
model_ready: true
inference.fp8_gemm: true
inference.quantized_linear_layers: 大于 0
```

本性能版还会在模型加载后使用 ABot 的实际 attention 形状比较
SageAttention2 与 FlashAttention2：

```text
[ATTN_BENCH] backend=sageattn first=...ms steady=...ms score=...
[ATTN_BENCH] backend=flash_attn_2 first=...ms steady=...ms score=...
[ATTN_SELECT] requested=auto selected=<实测较快后端>
```

`/healthz` 的 `attention.benchmark` 会保留完整结果，`requested_backend`
和 `default_backend` 分别表示配置值与最终实际值。需要固定后端进行复测时，
只修改环境变量并重启，无需重建镜像：

```text
ABOT_SELF_ATTN_BACKEND=auto
ABOT_SELF_ATTN_BACKEND=sageattn
ABOT_SELF_ATTN_BACKEND=flash_attn_2
```

模型缓存只有在主模型、T5、Wan VAE、TAEW2.2、配置和 tokenizer 均存在且
大文件尺寸合理时才会被视为完整；中断的快照会自动继续下载，而不是进入
永久的“模型加载中”状态。

启动日志也应出现：

```text
[BUILD][VERIFY] flash-attn 2.8.3 varlen_func=True
[BOOT][L20N][VERIFY] FlashAttention2 2.8.3 SM120 varlen smoke test passed
FLASH_ATTN_2_AVAILABLE: True
SAGE_ATTN_AVAILABLE: True
cc=12.x
fp8_gemm=True
quantized linear layers: <正整数>
```

如果实际 CC 不是 12.0，启动器不会猜测架构，也不会退回编译 SM80；
错误原因会记录在 `runtime_bootstrap` 中。

## 帧率基准

首轮保持：

```text
ABOT_DISPLAY_FPS=8
ABOT_WS_SEND_FPS=8
```

在 `/metrics` 观察 `generated_fps`。只有生成速度能够持续高于目标帧率时，
才同时调整两个变量：

```text
ABOT_DISPLAY_FPS=12
ABOT_WS_SEND_FPS=12
```

如果生成速度在完整会话中仍稳定高于 16 FPS，再尝试 16。不要只提高显示
帧率，否则会耗尽浏览器缓冲并增加丢帧。

`/metrics` 会在会话结束后继续保留最近 120 个 block，并输出
`recent_summary` 中 diffusion、VAE、后处理、总 block 时间和生成 FPS 的
p50/p95。`gpu_runtime` 同时包含 GPU 利用率、显存控制器利用率、SM 时钟、
实时功耗和功耗上限。不要再根据单个瞬时 FPS 判断优化是否有效。

## 量化 A/B（下一轮）

同一个镜像支持以下运行模式，默认保持上游 FP8：

```text
ABOT_QUANT_MODE=fp8
ABOT_QUANT_MODE=bf16
ABOT_QUANT_MODE=hybrid
```

`hybrid` 只量化权重元素数达到阈值的大型 Linear，默认阈值：

```text
ABOT_HYBRID_FP8_MIN_WEIGHT_NUMEL=16000000
```

RTX PRO 5000 72GB 有足够空间运行 BF16 对照。比较时必须保持相同的 seed、
prompt、参考缓存和动作序列，至少预热 5 个 block，再比较后续 20 个 block
的 `recent_summary`。FP8、BF16 和 hybrid 只保留实测最快且画质验收通过的
模式。

## 已实施的同步优化

- KV cache 的 `global_end_index`、`local_end_index` 和 `ref_token_len`
  使用 Python 整数，不再在每层调用 CUDA tensor `.item()`。
- 每次模型 forward 的帧 token 数只在入口计算一次，不再由每个
  Transformer layer 从 CUDA `grid_sizes` 读取。
- `generate_next_block()` 末尾不再强制 `torch.cuda.synchronize()`；
  diffusion 与 VAE 使用 CUDA Event，在帧复制回 CPU 的自然同步点读取。

这些修改不改变分辨率、五参考槽位、四步 denoise、每块 latent 帧数或动作
条件。

## 后续优化顺序

1. 先根据本版本的 attention 基准和 p50/p95，确定 Sage2 或 FA2。
2. 再做 `fp8`、`bf16`、`hybrid` 三组 A/B，并调整 hybrid 阈值。
3. 仅当 `vae_ms` 持续超过 `block_ms` 的 25% 时，单独优化或替换 VAE。
4. 上述动态同步点稳定后，才评估稳态 KV 窗口的 CUDA Graph。
5. SageAttention3 FP4、减少 denoise 步数、减少参考槽位和降低分辨率都属于
   画质变化实验，不进入默认生产配置。

## 公共空间策略

默认配置：

| 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `ABOT_SESSION_MAX_BLOCKS` | `120` | 单会话 GPU 租约 |
| `ABOT_SESSION_TIMEOUT` | `90` | 无连接/无活动会话回收秒数 |
| `ABOT_FRAME_QUEUE_SIZE` | `16` | 服务端帧队列 |
| `ABOT_WS_SEND_FPS` | `12` | 队列较浅时的常规发送帧率 |
| `ABOT_WS_CATCHUP_FPS` | `14` | 队列积压时的临时追帧速率 |
| `ABOT_WS_CATCHUP_QUEUE_THRESHOLD` | `4` | 启用追帧的队列深度 |
| `ABOT_WS_CATCHUP_MIN_GENERATED_FPS` | `11.5` | 低于该生成速度时禁用追帧，避免周期性断粮 |
| `ABOT_WS_MAX_INFLIGHT` | `3` | 未 ACK 的在途帧上限，覆盖约 189ms RTT 下的 14 FPS |
| `ABOT_CLIENT_JITTER_PRIME` | `2` | 浏览器首播所需帧数 |
| `ABOT_CLIENT_JITTER_TARGET` | `3` | 浏览器目标缓冲帧数 |
| `ABOT_CLIENT_JITTER_MAX` | `6` | 浏览器缓冲硬上限 |

会话按照 FIFO 顺序等待 GPU。排队位置会显示在画面上，并写入
`/metrics`。浏览器断开或点击停止后，会话立即标记停止；当前 CUDA
推理块结束后释放 GPU，并唤醒下一位。

服务端只在队列达到阈值时使用追帧速率，降到阈值以下后自动恢复常规
12 FPS；浏览器仍按 10/11/12 FPS 自适应播放。该策略只移动传输和播放
缓冲，不改变模型分块、生成帧数或画质。验收时应同时比较
`client_display_age_ms_p50`、`client_control_response_ms_p50` 和
`client_buffer_underruns`；若延迟下降但缓冲耗尽率明显超过 3%，应先把
`ABOT_CLIENT_JITTER_PRIME` 回调到 3，而不是继续提高追帧速率。

浏览器还会读取每个块携带的实际生成 FPS；当供给临时低于 11 FPS 时，
缓冲较浅阶段提前降到 10 FPS，避免播放端持续快于生成端。生成恢复后会
自动回到 11/12 FPS，不需要重启会话。

重新构建镜像前可以在无 GPU、无 Torch 的本机运行传输策略仿真：

```bash
python3 scripts/validate_stream_policy.py
```

脚本使用线上实测的约 1012ms/block 和 189ms RTT，对比旧版 3/5/10
缓冲与新版自适应追帧 2/3/6 缓冲。它用于提前排除明显不合理的队列参数，
不能替代部署后的真实网络 A/B。也可以用 `--rtt-ms`、`--catchup-fps`、
`--prime`、`--target` 和 `--max-buffer` 快速测试不同参数，无需重建镜像。

## SageAttention3

此版本只构建精度更稳妥的 SageAttention2。仓库保留 SageAttention3
FP4 源码，但不会安装或启用。只有完成相同 seed、prompt、动作序列的逐帧
画质比较后，才应单独制作实验版本开启 SageAttention3。
