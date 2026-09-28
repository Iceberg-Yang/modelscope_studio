FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_MODULE_LOADING=LAZY \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    MAX_JOBS=4 \
    NVCC_THREADS=4 \
    FLASH_ATTN_CUDA_ARCHS=80 \
    PORT=7860 \
    GRADIO_SHARE=false \
    ABOT_AUTO_DOWNLOAD=1 \
    ABOT_CHECKPOINT_ROOT=/mnt/workspace/checkpoints \
    ABOT_OUTPUT_DIR=/mnt/workspace/outputs \
    ABOT_STREAM_HEIGHT=480 \
    ABOT_STREAM_WIDTH=832 \
    ABOT_UI_FRAME_STRIDE=3

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        ninja-build \
        python3 \
        python3-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-a10.txt ./
RUN python3 -m pip install --upgrade pip setuptools wheel packaging ninja
RUN python3 -m pip install \
        torch==2.10.0+cu128 \
        torchvision==0.25.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128
RUN python3 -m pip install -r requirements-a10.txt
RUN python3 -m pip install --no-build-isolation flash-attn==2.8.1

COPY . .

EXPOSE 7860

CMD ["python3", "-u", "studio_app.py"]
