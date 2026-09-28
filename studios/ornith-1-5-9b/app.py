import os
import threading
import time
from collections.abc import Generator

import gradio as gr
import torch
from modelscope import snapshot_download
from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer


MODEL_ID = os.getenv("MODEL_ID", "ornith-ai/Ornith-1.5-9B")
MODEL_REVISION = os.getenv("MODEL_REVISION", "master")
MODEL_CACHE = os.getenv("MODELSCOPE_CACHE", "/mnt/workspace/modelscope")
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "32768"))
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are Ornith-1.5-9B, a helpful reasoning assistant. Give accurate, clear, well-structured answers.",
)

_model = None
_processor = None
_load_error = None
_load_lock = threading.Lock()


def load_model() -> tuple[object, object]:
    global _model, _processor, _load_error
    if _model is not None and _processor is not None:
        return _model, _processor

    with _load_lock:
        if _model is not None and _processor is not None:
            return _model, _processor
        try:
            model_dir = snapshot_download(
                MODEL_ID,
                revision=MODEL_REVISION,
                cache_dir=MODEL_CACHE,
            )
            _processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
            _model = AutoModelForImageTextToText.from_pretrained(
                model_dir,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            ).eval()
            _load_error = None
            return _model, _processor
        except Exception as exc:
            _load_error = f"{type(exc).__name__}: {exc}"
            raise


def model_status() -> str:
    if _model is not None:
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        return f"✅ 模型已加载 · {gpu} · BF16"
    if _load_error:
        return f"❌ 上次加载失败：`{_load_error}`"
    return "⏳ 模型将在首次发送消息时加载（约 18.84GB，首次下载需要一些时间）"


def split_reasoning(raw_text: str) -> tuple[str, str]:
    if "</think>" in raw_text:
        reasoning, answer = raw_text.split("</think>", 1)
        reasoning = reasoning.replace("<think>", "", 1).strip()
        return reasoning, answer.lstrip()
    if "<think>" in raw_text:
        return raw_text.replace("<think>", "", 1).strip(), ""
    return "", raw_text


def preset_parameters(preset: str) -> tuple[float, int]:
    if preset == "精确编码":
        return 0.6, 2048
    return 1.0, 2048


def stream_reply(
    message: str,
    history: list[dict],
    preset: str,
    temperature: float,
    max_new_tokens: int,
) -> Generator[tuple[list[dict], str, str, str, str], None, None]:
    if not message.strip():
        yield history, "", "", "", model_status()
        return

    started_at = time.monotonic()
    new_history = history + [{"role": "user", "content": message}]
    assistant_item = {"role": "assistant", "content": "_正在加载模型…_"}
    output_history = new_history + [assistant_item]
    yield output_history, "", "", "正在准备模型…", model_status()

    try:
        model, processor = load_model()
    except Exception as exc:
        assistant_item["content"] = f"模型加载失败：`{type(exc).__name__}: {exc}`"
        yield output_history, "", "", "", model_status()
        return

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(
        {"role": item["role"], "content": item.get("content", "")}
        for item in new_history
        if item.get("role") in {"user", "assistant"}
    )

    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_length = inputs["input_ids"].shape[-1]
        if input_length > MAX_INPUT_TOKENS:
            assistant_item["content"] = (
                f"当前对话为 {input_length:,} tokens，超过本空间设置的 "
                f"{MAX_INPUT_TOKENS:,} tokens。请清空对话或缩短输入。"
            )
            yield output_history, "", "", f"输入 {input_length:,} tokens", model_status()
            return

        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        streamer = TextIteratorStreamer(
            processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": temperature > 0,
            "temperature": max(float(temperature), 0.01),
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": 1.0,
        }
        worker = threading.Thread(target=model.generate, kwargs=generation_kwargs, daemon=True)
        worker.start()

        raw_text = ""
        for chunk in streamer:
            raw_text += chunk
            reasoning, answer = split_reasoning(raw_text)
            assistant_item["content"] = answer or "_Ornith 正在思考…_"
            elapsed = time.monotonic() - started_at
            metrics = f"输入 {input_length:,} tokens · 已运行 {elapsed:.1f}s · {preset}"
            yield output_history, "", reasoning, metrics, model_status()
        worker.join()

        reasoning, final_answer = split_reasoning(raw_text)
        final_answer = final_answer.strip()
        assistant_item["content"] = final_answer or "模型已完成请求，但没有返回可显示的正文。"
        output_length = len(processor.tokenizer.encode(raw_text, add_special_tokens=False))
        elapsed = time.monotonic() - started_at
        metrics = (
            f"输入 {input_length:,} tokens · 输出 {output_length:,} tokens · "
            f"耗时 {elapsed:.1f}s · {preset}"
        )
        yield output_history, "", reasoning, metrics, model_status()
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        assistant_item["content"] = "GPU 显存不足。请清空对话、缩短输入或降低最大输出长度后重试。"
        yield output_history, "", "", "", model_status()
    except Exception as exc:
        assistant_item["content"] = f"推理失败：`{type(exc).__name__}: {exc}`"
        yield output_history, "", "", "", model_status()


def clear_chat() -> tuple[list, str, str, str]:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [], "", "", ""


with gr.Blocks(title="Ornith-1.5-9B Reasoning Playground") as demo:
    gr.Markdown("# 🐦 Ornith-1.5-9B")
    chatbot = gr.Chatbot(height=520, label="对话")

    with gr.Row():
        prompt = gr.Textbox(
            placeholder="输入你的问题……",
            lines=3,
            show_label=False,
            scale=8,
        )
        send = gr.Button("发送", variant="primary", scale=1)

    with gr.Accordion("生成参数", open=False):
        preset = gr.Radio(
            ["通用推理", "精确编码"],
            value="通用推理",
            label="参数预设",
        )
        temperature = gr.Slider(0.0, 1.5, value=1.0, step=0.1, label="Temperature")
        max_new_tokens = gr.Slider(128, 4096, value=2048, step=128, label="最大输出 tokens")
        preset.change(preset_parameters, inputs=preset, outputs=[temperature, max_new_tokens])

    gr.Examples(
        examples=[
            ["有 12 枚外观相同的硬币，其中一枚重量不同。如何用天平三次找出它并判断轻重？"],
            ["比较强化学习中的 on-policy 与 off-policy 方法，并分别给出适用场景。"],
            ["写一个带重试、指数退避和类型标注的 Python HTTP 请求函数。"],
        ],
        inputs=prompt,
    )
    with gr.Row():
        clear = gr.Button("清空对话")

    metrics = gr.Markdown("")
    with gr.Accordion("本轮推理过程", open=False):
        reasoning = gr.Markdown("")
    with gr.Accordion("运行状态", open=False):
        status = gr.Markdown(model_status())

    submit_inputs = [prompt, chatbot, preset, temperature, max_new_tokens]
    submit_outputs = [chatbot, prompt, reasoning, metrics, status]
    send.click(stream_reply, submit_inputs, submit_outputs)
    prompt.submit(stream_reply, submit_inputs, submit_outputs)
    clear.click(clear_chat, outputs=[chatbot, prompt, reasoning, metrics])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860)
