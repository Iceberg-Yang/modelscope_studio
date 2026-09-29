FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MAX_JOBS=4 \
    NVCC_THREADS=4 \
    FLASH_ATTN_CUDA_ARCHS=120 \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    CUDA_MODULE_LOADING=LAZY \
    PORT=7860 \
    ABOT_STREAM_HEIGHT=704 \
    ABOT_STREAM_WIDTH=1280 \
    ABOT_DISPLAY_FPS=8 \
    ABOT_WS_SEND_FPS=8 \
    ABOT_FRAME_QUEUE_SIZE=16 \
    ABOT_JPEG_QUALITY=65 \
    ABOT_SESSION_MAX_BLOCKS=120 \
    ABOT_SESSION_TIMEOUT=90 \
    ABOT_WS_MAX_INFLIGHT=2 \
    ABOT_OUTPUT_DIR=/mnt/workspace/abot-world/outputs \
    ABOT_EXTENSION_CACHE=/mnt/workspace/abot-world/extensions \
    ABOT_RUNTIME_STATUS=/tmp/abot-l20n-runtime.json \
    ABOT_REQUIRE_SAGEATTENTION=1 \
    TRITON_CACHE_DIR=/mnt/workspace/abot-world/extensions/triton \
    MODELSCOPE_CACHE=/mnt/workspace/.cache/modelscope

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        ninja-build \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "${VIRTUAL_ENV}"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN pip install --upgrade pip setuptools wheel packaging ninja psutil \
    && pip install \
        torch==2.10.0 \
        torchvision==0.25.0 \
        --index-url https://download.pytorch.org/whl/cu128

# Compile FlashAttention2 from the official v2.8.3 source distribution during
# the image build.  The build host has no GPU, so pin the known deployment
# target explicitly and reject generic/prebuilt wheels that may omit SM120.
ARG FLASH_ATTN_VERSION=2.8.3
RUN mkdir -p /opt/flash-attn-wheels \
    && pip wheel \
        "flash-attn==${FLASH_ATTN_VERSION}" \
        --no-build-isolation \
        --no-deps \
        --no-binary=flash-attn \
        --wheel-dir /opt/flash-attn-wheels \
    && pip install \
        --no-deps \
        /opt/flash-attn-wheels/flash_attn-*.whl \
    && python -c "import flash_attn; from flash_attn import flash_attn_varlen_func; print('[BUILD][VERIFY] flash-attn', flash_attn.__version__, 'varlen_func=True')" \
    && rm -rf /opt/flash-attn-wheels

COPY requirements-l20n.txt /app/requirements-l20n.txt
RUN pip install -r /app/requirements-l20n.txt

# SageAttention is still compiled on the actual SM120 GPU at runtime. Its wheel
# is cached in persistent /mnt/workspace.
COPY . /app

# Performance policy lives after all expensive dependency layers so tuning these
# values does not invalidate the Torch/FlashAttention build cache.
ENV ABOT_QUANT_MODE=fp8 \
    ABOT_HYBRID_FP8_MIN_WEIGHT_NUMEL=16000000 \
    ABOT_DISPLAY_FPS=12 \
    ABOT_DISPLAY_FPS_MIN=10 \
    ABOT_DISPLAY_FPS_TARGET=11 \
    ABOT_WS_SEND_FPS=12 \
    ABOT_WS_CATCHUP_FPS=14 \
    ABOT_WS_CATCHUP_QUEUE_THRESHOLD=4 \
    ABOT_WS_CATCHUP_MIN_GENERATED_FPS=11.5 \
    ABOT_WS_MAX_INFLIGHT=3 \
    ABOT_CLIENT_JITTER_PRIME=2 \
    ABOT_CLIENT_JITTER_TARGET=3 \
    ABOT_CLIENT_JITTER_MAX=6 \
    ABOT_SELF_ATTN_BACKEND=auto \
    ABOT_ATTN_BENCHMARK=1 \
    ABOT_ATTN_BENCH_WARMUP=1 \
    ABOT_ATTN_BENCH_REPEATS=3 \
    ABOT_ATTN_AUTO_MIN_SPEEDUP=0.03 \
    ABOT_ATTN_BENCHMARK_SDPA=0 \
    ABOT_METRICS_HISTORY_BLOCKS=120

RUN mkdir -p \
        /mnt/workspace/.cache/modelscope \
        /mnt/workspace/abot-world/extensions \
        /mnt/workspace/abot-world/extensions/triton \
        /mnt/workspace/abot-world/outputs

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20m --retries=5 \
    CMD curl -fsS http://127.0.0.1:7860/healthz || exit 1

CMD ["/opt/venv/bin/python", "-u", "entrypoint_l20n.py"]
