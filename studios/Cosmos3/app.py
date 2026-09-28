"""
Cosmos3-Edge Physical AI Reasoning Demo.

仅使用 Reasoner tower（2B Nemotron），支持：
- 图片/视频/文本输入 + 文本指令
- 思维链 (Chain-of-Thought)
- 流式文本输出（实时显示推理过程）
- Physical AI 推理示例（任务规划/场景分析/动作预测/空间定位/物理推理）

参考: https://huggingface.co/spaces/hugging-apps/nvidia-cosmos3-edge
"""

import os
import sys
import glob

# ---- cuDNN 库版本冲突修复 ----
# 系统 libcudnn_engines_runtime_compiled.so.9 缺少 libcudnn_graph.so.9 的符号
# 设置 LD_LIBRARY_PATH 指向 torch 自带 cuDNN 目录后重启进程
if not os.environ.get('_CUDNN_FIXED'):
    for _dir in sorted(glob.glob(
        '/usr/local/lib/python3.*/site-packages/nvidia/cudnn/lib'
    )):
        _ld = os.environ.get('LD_LIBRARY_PATH', '')
        if _dir not in _ld:
            os.environ['LD_LIBRARY_PATH'] = _dir + ':' + _ld
            os.environ['_CUDNN_FIXED'] = '1'
            os.execv(sys.executable, [sys.executable] + sys.argv)
        break

# ---- torch/CUDA 版本检测（调试用）----
import torch
print(f"[Cosmos3] torch={torch.__version__}, CUDA={torch.version.cuda}, cuDNN={torch.backends.cudnn.version()}")
print(f"[Cosmos3] CUDA available={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# ---- num2words: 创空间 pip 源不可用，尝试从镜像安装 ----
# Cosmos3 processor 处理视频时间戳需要此包，图片/文本推理不需要
# 安装失败不阻止应用启动，仅影响视频推理
try:
    import num2words  # noqa: F401
except ImportError:
    import subprocess
    print("[Cosmos3] Installing num2words...")
    _installed = False
    for _mirror in [
        "https://mirrors.aliyun.com/pypi/simple/",
        "https://pypi.tuna.tsinghua.edu.cn/simple/",
        "https://pypi.org/simple/",
    ]:
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "num2words",
                "--index-url", _mirror,
                "--timeout", "120",
                "--quiet",
            ])
            print(f"[Cosmos3] num2words installed from {_mirror}")
            _installed = True
            break
        except Exception as _e:
            print(f"[Cosmos3] Failed from {_mirror}: {_e}")
    if not _installed:
        print("[Cosmos3] WARNING: num2words not installed. Video reasoning may not work.")
        print("[Cosmos3] Image and text reasoning will still work.")

import gradio as gr

from theme import CUSTOM_CSS, FOOTER_HTML, HEADER_HTML, nvidia_theme
from inference import (
    MAX_NEW_TOKENS_DEFAULT,
    get_status,
    reason_stream,
)

# ---------------------------------------------------------------------------
# 预置示例（对齐 NVIDIA 官方 HuggingFace Space）
# 参考: https://huggingface.co/spaces/hugging-apps/nvidia-cosmos3-edge
# ---------------------------------------------------------------------------

# 图片示例 prompts
EX_ROBOT_PROMPT = (
    "The task is to put the flower into the red bottle. Generate a plan "
    "consisting of subtasks to accomplish the task."
)
EX_DRIVE_PROMPT = (
    "You are a driving-assistance system. Describe the road scene and list "
    "the key hazards a driver should watch for."
)

# 视频示例 prompt
EX_VIDEO_PROMPT = (
    "This is a dashcam clip from an autonomous vehicle. Describe the vehicle's "
    "motion and the surrounding traffic scene."
)

# 文本示例 prompts
EX_TEXT_ACTION = "What is the next immediate action the robot should take?"
EX_TEXT_GROUNDING = "Locate the accurate bounding box of the objects. Return a json."
EX_TEXT_PHYSICS = "Can the countertop support the weight of the objects on it?"
EX_TEXT_CAPTION = "Caption the image in detail."


def refresh_status() -> str:
    """刷新状态栏。"""
    status, message = get_status()
    color = {
        "idle": "#888",
        "loading": "#ffb000",
        "ready": "#76b900",
        "error": "#ff4444",
    }.get(status, "#888")
    return f'<span style="color:{color}">● {message}</span>'


# ---------------------------------------------------------------------------
# 包装函数：避免在 inputs 列表中传 None（Gradio 6.x 不支持）
# ---------------------------------------------------------------------------
def _reason_image(prompt, image, enable_thinking, max_new_tokens, temperature, top_p):
    """图片推理包装：image 有值，video=None。"""
    yield from reason_stream(
        prompt, image=image, video=None,
        enable_thinking=enable_thinking,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

def _reason_video(prompt, video, enable_thinking, max_new_tokens, temperature, top_p):
    """视频推理包装：video 有值，image=None。"""
    yield from reason_stream(
        prompt, image=None, video=video,
        enable_thinking=enable_thinking,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

def _reason_text(prompt, enable_thinking, max_new_tokens, temperature, top_p):
    """文本推理包装：image=None, video=None。"""
    yield from reason_stream(
        prompt, image=None, video=None,
        enable_thinking=enable_thinking,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )


with gr.Blocks(
    title="Cosmos3-Edge — Physical AI Reasoning",
) as demo:
    gr.HTML(HEADER_HTML)

    # 状态栏
    status_box = gr.Markdown(value=refresh_status, elem_classes="status-bar")

    # ---- 主区域 ----
    gr.HTML('<div class="section-label">// Physical AI Reasoning — 理解物理世界</div>')
    with gr.Row():
        # ---- 左侧：输入区（分 Tab）----
        with gr.Column(scale=1, elem_classes="panel-card"):
            with gr.Tabs(elem_classes="tabs"):
                # ==================== Tab 1: 图片推理 ====================
                with gr.Tab("图片推理", id="image"):
                    image_input = gr.Image(
                        label="输入图片",
                        type="pil",
                        height=240,
                    )
                    prompt_img = gr.Textbox(
                        label="指令 / 问题",
                        lines=3,
                        placeholder="输入推理指令或问题...（建议使用英文，Physical AI 场景效果最佳）",
                    )
                    thinking_img = gr.Checkbox(
                        label="思维链 (Chain-of-Thought)",
                        value=True,
                        info="启用后模型先输出推理过程，再给出最终答案",
                    )
                    with gr.Accordion("高级设置", open=False):
                        max_tokens_img = gr.Slider(
                            label="最大生成长度 (tokens)",
                            minimum=128, maximum=4096,
                            value=MAX_NEW_TOKENS_DEFAULT, step=64,
                        )
                        temperature_img = gr.Slider(
                            label="Temperature",
                            minimum=0.0, maximum=1.5, value=0.6, step=0.05,
                        )
                        top_p_img = gr.Slider(
                            label="Top-p",
                            minimum=0.1, maximum=1.0, value=0.95, step=0.05,
                        )
                    btn_img = gr.Button("推理", variant="primary", elem_classes="primary-btn")

                    with gr.Group(elem_id="img-examples"):
                        gr.Examples(
                            examples=[
                                ["examples/robot_scene.png", EX_ROBOT_PROMPT, True],
                                ["examples/driving_scene.jpg", EX_DRIVE_PROMPT, True],
                            ],
                            inputs=[image_input, prompt_img, thinking_img],
                            label="图片示例",
                        )

                # ==================== Tab 2: 视频推理 ====================
                with gr.Tab("视频推理", id="video"):
                    video_input = gr.Video(
                        label="输入视频",
                        height=240,
                    )
                    prompt_vid = gr.Textbox(
                        label="指令 / 问题",
                        lines=3,
                        placeholder="输入推理指令或问题...（建议使用英文，Physical AI 场景效果最佳）",
                    )
                    thinking_vid = gr.Checkbox(
                        label="思维链 (Chain-of-Thought)",
                        value=True,
                        info="启用后模型先输出推理过程，再给出最终答案",
                    )
                    with gr.Accordion("高级设置", open=False):
                        max_tokens_vid = gr.Slider(
                            label="最大生成长度 (tokens)",
                            minimum=128, maximum=4096,
                            value=MAX_NEW_TOKENS_DEFAULT, step=64,
                        )
                        temperature_vid = gr.Slider(
                            label="Temperature",
                            minimum=0.0, maximum=1.5, value=0.6, step=0.05,
                        )
                        top_p_vid = gr.Slider(
                            label="Top-p",
                            minimum=0.1, maximum=1.0, value=0.95, step=0.05,
                        )
                    btn_vid = gr.Button("推理", variant="primary", elem_classes="primary-btn")

                    with gr.Group(elem_id="vid-examples"):
                        gr.Examples(
                            examples=[
                                ["examples/driving_clip.mp4", EX_VIDEO_PROMPT, True],
                            ],
                            inputs=[video_input, prompt_vid, thinking_vid],
                            label="视频示例",
                        )

                # ==================== Tab 3: 文本推理 ====================
                with gr.Tab("文本推理", id="text"):
                    gr.Markdown(
                        "无需图片或视频，直接输入问题进行推理。"
                        "也可以先在「图片推理」上传图片后再切换到此处。"
                    )
                    prompt_txt = gr.Textbox(
                        label="指令 / 问题",
                        lines=3,
                        placeholder="输入推理指令或问题...",
                    )
                    thinking_txt = gr.Checkbox(
                        label="思维链 (Chain-of-Thought)",
                        value=True,
                        info="启用后模型先输出推理过程，再给出最终答案",
                    )
                    with gr.Accordion("高级设置", open=False):
                        max_tokens_txt = gr.Slider(
                            label="最大生成长度 (tokens)",
                            minimum=128, maximum=4096,
                            value=MAX_NEW_TOKENS_DEFAULT, step=64,
                        )
                        temperature_txt = gr.Slider(
                            label="Temperature",
                            minimum=0.0, maximum=1.5, value=0.6, step=0.05,
                        )
                        top_p_txt = gr.Slider(
                            label="Top-p",
                            minimum=0.1, maximum=1.0, value=0.95, step=0.05,
                        )
                    btn_txt = gr.Button("推理", variant="primary", elem_classes="primary-btn")

                    # 示例按钮（替代 gr.Examples，避免纯文本 Tab 渲染异常）
                    gr.Markdown("**Physical AI 示例**")
                    with gr.Row():
                        btn_ex1 = gr.Button("动作预测", size="sm")
                        btn_ex2 = gr.Button("空间定位", size="sm")
                    with gr.Row():
                        btn_ex3 = gr.Button("物理推理", size="sm")
                        btn_ex4 = gr.Button("场景描述", size="sm")

        # ---- 右侧：输出区（共享）----
        with gr.Column(scale=1, elem_classes="panel-card"):
            gr.HTML('<div class="section-label">// Output</div>')
            output = gr.Textbox(
                label="推理输出",
                lines=22,
                elem_classes="output-media",
                interactive=False,
                placeholder="推理结果将在此实时显示...",
            )

    gr.HTML(FOOTER_HTML)

    # ---- 切换图片/视频时自动清空输出 ----
    def _clear_output():
        return gr.update(value="")

    image_input.change(fn=_clear_output, outputs=[output])
    video_input.change(fn=_clear_output, outputs=[output])

    # ---- 文本示例按钮 ----
    def _set_text_example(prompt, thinking):
        return prompt, thinking, ""

    btn_ex1.click(fn=lambda: _set_text_example(EX_TEXT_ACTION, True),
                   outputs=[prompt_txt, thinking_txt, output])
    btn_ex2.click(fn=lambda: _set_text_example(EX_TEXT_GROUNDING, False),
                   outputs=[prompt_txt, thinking_txt, output])
    btn_ex3.click(fn=lambda: _set_text_example(EX_TEXT_PHYSICS, True),
                   outputs=[prompt_txt, thinking_txt, output])
    btn_ex4.click(fn=lambda: _set_text_example(EX_TEXT_CAPTION, False),
                   outputs=[prompt_txt, thinking_txt, output])

    # ---- 事件绑定 ----
    # 图片推理：image 有值，video=None
    btn_img.click(
        fn=_reason_image,
        inputs=[prompt_img, image_input, thinking_img, max_tokens_img, temperature_img, top_p_img],
        outputs=[prompt_img, output],
    ).then(refresh_status, outputs=status_box)

    # 视频推理：video 有值，image=None
    btn_vid.click(
        fn=_reason_video,
        inputs=[prompt_vid, video_input, thinking_vid, max_tokens_vid, temperature_vid, top_p_vid],
        outputs=[prompt_vid, output],
    ).then(refresh_status, outputs=status_box)

    # 文本推理：image=None, video=None
    btn_txt.click(
        fn=_reason_text,
        inputs=[prompt_txt, thinking_txt, max_tokens_txt, temperature_txt, top_p_txt],
        outputs=[prompt_txt, output],
    ).then(refresh_status, outputs=status_box)


if __name__ == "__main__":
    demo.queue(max_size=12).launch(
        theme=nvidia_theme(),
        css=CUSTOM_CSS,
        head=HEADER_HTML,
    )
