# Upstream provenance and local changes

## Source

- Hugging Face Space: <https://huggingface.co/spaces/TencentARC/scope-camera-video-generation>
- Retrieved revision: `a7a8de0926e5c76cf66d91ecb7a03be5d942ae9f`
- Reference implementation: <https://github.com/TencentARC/SCoPE>
- Source Space metadata license: Apache-2.0

The source Space did not contain a root `LICENSE` file. This migration includes the unmodified
standard Apache License 2.0 text matching the license declared by the Space and its three model
repositories.

## ModelScope migration changes

- Removed the Hugging Face `spaces.GPU` decorator and `spaces` import.
- Replaced `huggingface_hub` model-file downloads with ModelScope SDK downloads.
- Uses the same model IDs on ModelScope: `TencentARC/SCoPE`,
  `lightx2v/Wan2.2-Lightning`, and `Qwen/Qwen2-VL-2B-Instruct`.
- Defaults model downloads and reusable cache data to `/mnt/workspace`.
- Keeps the upstream FP8, int8, four-step Lightning LoRA, and shard-streamed loading behavior.
- Updated the Studio card and UI wording for ModelScope's free 48 GB xGPU runtime.
