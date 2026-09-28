# MiniCPM5-2B Demo — ModelScope 创空间版
#
# 模型: OpenBMB/MiniCPM5-2B (Apache-2.0)
#
# gradio 6.17.3 API 注意事项（核对 components/*.pyi、blocks.py）：
#   - Chatbot 移除 type 参数，messages 成为唯一格式（传 type= 会 TypeError）
#   - Blocks 构造器移除 theme / css / title，theme 与 css 移到 launch()
#   - Textbox 新增 submit_btn / stop_btn，停止按钮经 Textbox.stop(cancels=[...]) 接线
#
# 注：think 与 im_end 这类尖括号标记在本文件中一律用 chr() 拼接构造，
#     避免被编辑工具的标签解析误截断。

import logging
import os
import sys
import threading

# 必须在 import gradio 之前设置：gradio 在导入时即初始化遥测开关。
# 这同时消除了线上日志里的 GET https://api.gradio.app/pkg-version 外部请求。
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from modelscope.hub.snapshot_download import snapshot_download  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

from utils_chatbot import organize_messages  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = "OpenBMB/MiniCPM5-2B"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 创空间默认每次重启数据丢失，只有 /mnt/workspace 持久化。
# 模型约 4.7 GiB，缓存不落持久化目录会导致每次重启都重新下载并可能启动超时。
PERSISTENT_DIR = "/mnt/workspace"
CACHE_DIR = os.environ.get("MINICPM_CACHE_DIR") or os.path.join(
    PERSISTENT_DIR, ".cache", "modelscope"
)

# 生成长度：思考模式下最多输出 4096 token
MAX_NEW_TOKENS = 4096

# 模型输出中的特殊标记。
# 用 chr() 拼接而非字面量书写，理由见文件头注释。
_LT = chr(60)
_GT = chr(62)
THINK_OPEN = _LT + "think" + _GT
THINK_CLOSE = _LT + "/think" + _GT
EOS_MARKER = _LT + "|im_end|" + _GT

# ---------------------------------------------------------------------------
# 视觉样式：通过 gradio 原生组件 + 自定义 CSS 实现
#
# 设计语言包含 indigo 渐变气泡、非对称圆角、左竖线
# 思考块、fadeIn 入场、4px 细滚动条）。这些是纯样式，与渲染管线无关，所以在
# Blocks 模式下依然可以还原，分两层落地：
#
#   1) THEME —— 用 gradio 官方语义变量驱动输入框/按钮/滑块/开关/代码块/圆角/阴影。
#      不依赖任何 DOM class，gradio 升级不会失效。
#   2) CUSTOM_CSS —— 补 theme 表达不了的部分。核对 themes/base.py 中 Theme.set()
#      的 163 个可设置变量后确认：chatbot 相关只有 chatbot_text_size，没有任何
#      气泡或 thought 变量，因此气泡渐变、非对称圆角、思考块左竖线、入场动画、
#      滚动条只能用 CSS 覆盖。
#
# CSS 选择器取自 gradio 6.17.3 前端 bundle（templates/frontend/assets 下 Chatbot
# 组件的 Index-*.css）实测：.user-row.bubble 与 .bot-row.bubble 是同一元素上的双
# class，.bubble.pending 表示生成中，.thought-group .thought 是 reasoning_tags
# 渲染出的思考块容器。刻意不使用 .svelte-xxxxx 这类 scoped hash——每次构建都会变。
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* 头部 logo：40px 圆角 + 靛蓝辉光 */
.hero {
    text-align: center;
}
.hero img {
    width: 40px;
    height: 40px;
    border-radius: 8px;
    display: block;
    margin: 0 auto 10px;
    filter: drop-shadow(0 0 8px rgba(99, 102, 241, 0.25));
}

/* 布局：固定 1000px 居中（比原 900 略宽），窄屏回退满宽避免横向溢出 */
.gradio-container {
    width: 1000px !important;
    max-width: 1000px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
@media (max-width: 1040px) {
    .gradio-container { width: 100% !important; }
}

/* 消息滚动区：4px 细滚动条 */
.message-wrap {
    scrollbar-width: thin;
    scrollbar-color: #cbd5e1 transparent;
    scroll-behavior: smooth;
}
.message-wrap::-webkit-scrollbar { width: 4px; }
.message-wrap::-webkit-scrollbar-track { background: transparent; }
.message-wrap::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 10px;
}

/* 气泡入场动画 */
.message-row {
    animation: ms-fade-in 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}
@keyframes ms-fade-in {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* 用户气泡：gradio 6.x 真正画底色的是行内层 .user 元素（.user-row.bubble
   只是透明的对齐容器），渐变必须涂在内层，否则白字会落在默认浅色底上 */
.user-row.bubble {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    width: 100% !important;
}
.user-row.bubble .user {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    border: none !important;
    border-radius: 20px 20px 4px 20px !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.2) !important;
    padding: 12px 18px !important;
    width: fit-content !important;
    max-width: 85% !important;
}
/* 渐变底上强制白字，压过 theme 的 body_text_color */
.user-row.bubble .user,
.user-row.bubble .user .message-content,
.user-row.bubble .user .prose,
.user-row.bubble .user .prose p,
.user-row.bubble .user .prose li,
.user-row.bubble .user .prose strong,
.user-row.bubble .user .prose em {
    color: #ffffff !important;
}
.user-row.bubble .user .prose code {
    color: #ffffff !important;
    background: rgba(255, 255, 255, 0.18) !important;
}
.user-row.bubble .user a {
    color: #ffffff !important;
    text-decoration: underline;
}

/* 机器人气泡：gradio 6.x 真正画底色的是行内层 .bot 元素，浅灰底 + 左下尖角 */
.bot-row.bubble {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    width: 100% !important;
}
.bot-row.bubble .bot {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 20px 20px 20px 4px !important;
    box-shadow: none !important;
    padding: 12px 18px !important;
    width: fit-content !important;
    max-width: 85% !important;
}

/* 思考段：gradio 6.x 的思考块外层即 .thought-group（内部无 .thought 元素），
   左竖线浅靛蓝块 */
.thought-group {
    background: #eef2ff !important;
    border: none !important;
    border-left: 3px solid #4f46e5 !important;
    border-radius: 4px 12px 12px 4px !important;
    padding: 12px 16px !important;
    margin-bottom: 12px !important;
    font-size: 14px !important;
    font-style: italic;
}
.thought-group,
.thought-group .prose,
.thought-group .prose p,
.thought-group .message-content,
.thought-group .duration,
.thought-group [role="button"],
.thought-group .title {
    color: #64748b !important;
}
/* 生成中的思考段做呼吸（pending 时 gradio 在 .bubble 上加 .pending 类） */
.bubble.pending {
    animation: ms-think-pulse 2s ease-in-out infinite;
}
@keyframes ms-think-pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.72; }
}

/* 代码块 */
.prose pre {
    background: #f1f5f9 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 12px !important;
    overflow-x: auto;
}
.bot-row.bubble .prose code {
    color: #4f46e5 !important;
    background: rgba(79, 70, 229, 0.07) !important;
}
.bot-row.bubble .prose pre code {
    color: #1e293b !important;
    background: transparent !important;
}

/* LaTeX 独立公式留白并允许横向滚动，避免长公式撑破气泡 */
.katex-display {
    overflow-x: auto;
    overflow-y: hidden;
    padding: 4px 0;
}

/* 输入区：胶囊外观。gradio 在显示 submit/stop 按钮时
   会移除 textarea 边框，故把边框/阴影加在 .input-container 外层容器上 */
.chat-input .input-container {
    border: 1px solid #e2e8f0 !important;
    border-radius: 2rem !important;
    background: #ffffff !important;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06) !important;
    padding: 6px 6px 6px 18px !important;
    align-items: center !important;
    transition: border-color .3s ease, box-shadow .3s ease;
}
.chat-input .input-container:focus-within {
    border-color: rgba(79, 70, 229, 0.4) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
}
/* 发送按钮：实心靛蓝圆形 */
.chat-input .submit-button {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 9999px !important;
    width: 40px !important;
    height: 40px !important;
    min-width: 40px !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.25) !important;
}
.chat-input .submit-button:hover {
    filter: brightness(1.05);
}
/* 停止按钮：描边红，生成中才显眼 */
.chat-input .stop-button {
    background: #ffffff !important;
    color: #ef4444 !important;
    border: 1px solid #fecaca !important;
    border-radius: 9999px !important;
    width: 40px !important;
    height: 40px !important;
    min-width: 40px !important;
}
"""

# primary_hue=indigo 对应 #4f46e5（indigo-600），
# neutral_hue=slate 对应文字与边框的 slate-800/500/200/50
THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="indigo",
    neutral_hue="slate",
    radius_size="lg",
    # Soft 默认带 LocalFont("Montserrat")，但 Montserrat 无中文字形，中文会 fallback
    # 到系统字体导致中英混排不一致。改用系统字体栈
    font=(
        "-apple-system",
        "BlinkMacSystemFont",
        "Segoe UI",
        "Roboto",
        "PingFang SC",
        "Hiragino Sans GB",
        "Microsoft YaHei",
        "Helvetica Neue",
        "Arial",
        "sans-serif",
    ),
    font_mono=(
        "ui-monospace",
        "SFMono-Regular",
        "Menlo",
        "Consolas",
        "monospace",
    ),
)

# 下列变量名已逐个对照 themes/base.py 中 Theme.set() 的 kwonly 白名单核对，
# 该方法不接受 **kwargs，写错名字会直接 TypeError
THEME.set(
    # 全局基调：白底、slate 文字、indigo 强调
    body_background_fill="#ffffff",
    body_text_color="#1e293b",
    body_text_color_subdued="#64748b",
    background_fill_primary="#ffffff",
    border_color_primary="#e2e8f0",
    color_accent="#4f46e5",
    link_text_color="#4f46e5",
    link_text_color_hover="#6366f1",
    loader_color="#4f46e5",
    # 输入框：圆角边框与 focus-within 光环
    input_background_fill="#ffffff",
    input_border_color="#e2e8f0",
    input_border_color_hover="#cbd5e1",
    input_border_color_focus="rgba(79, 70, 229, 0.4)",
    input_shadow="none",
    input_shadow_focus="0 0 0 3px rgba(99, 102, 241, 0.1)",
    input_placeholder_color="#94a3b8",
    input_radius="20px",
    # 主按钮：渐变 + 投影 + hover 微放大
    button_primary_background_fill="linear-gradient(135deg, #4f46e5, #6366f1)",
    button_primary_background_fill_hover="linear-gradient(135deg, #6366f1, #818cf8)",
    button_primary_border_color="transparent",
    button_primary_border_color_hover="transparent",
    button_primary_text_color="#ffffff",
    button_primary_shadow="0 4px 14px rgba(79, 70, 229, 0.2)",
    button_primary_shadow_hover="0 6px 18px rgba(79, 70, 229, 0.28)",
    button_transform_hover="scale(1.03)",
    button_transition="all 0.3s ease",
    button_medium_radius="100px",
    button_medium_text_weight="600",
    # 次按钮（清空对话）：描边风格，不与主按钮抢视觉
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_hover="#f1f5f9",
    button_secondary_border_color="#e2e8f0",
    button_secondary_border_color_hover="#cbd5e1",
    button_secondary_text_color="#64748b",
    button_secondary_shadow="none",
    button_secondary_shadow_hover="none",
    # 开关（思考模式）：选中态渐变
    checkbox_background_color_selected="#4f46e5",
    checkbox_border_color_selected="#4f46e5",
    checkbox_border_color_focus="#4f46e5",
    checkbox_label_background_fill_selected="linear-gradient(135deg, #4f46e5, #6366f1)",
    checkbox_label_border_color_selected="#4f46e5",
    checkbox_label_text_color_selected="#ffffff",
    # 滑块：轨道用 accent
    slider_color="#4f46e5",
    # 容器：扁平无投影，靠 1px slate 边框分区
    block_background_fill="#ffffff",
    block_border_color="#e2e8f0",
    block_radius="16px",
    block_shadow="none",
    panel_background_fill="#ffffff",
    panel_border_color="#e2e8f0",
    accordion_text_color="#1e293b",
    code_background_fill="#f1f5f9",
    chatbot_text_size="16px",
    # ---- 深色模式对齐 ----
    # 浅色主题下无任何 dark 变体；而 gradio 的
    # theme.css 会额外生成 :root.dark 块，templates/frontend/index.html 第 50 行
    # 还有 @media (prefers-color-scheme: dark) 把 body 背景直接设成
    # body_background_fill_dark（Soft 默认为 #0a0f1e 近黑）。若不处理，系统深色
    # 模式下会出现近黑 body 配浅色气泡的割裂观感。
    # 这里把全部定制变量的 _dark 版本一并设为浅色值，等价于强制 light 外观。
    # 这 52 项由脚本从线上 theme.css 的 :root 块自动提取，并已逐个对照
    # Theme.set() 的 kwonly 白名单校验；又经比对 neutral/primary/secondary 三个
    # 色阶在 light 与 dark 块中定义完全一致，故其中 var() 引用在深色下解析相同。
    body_background_fill_dark="#ffffff",
    body_text_color_dark="#1e293b",
    color_accent_soft_dark="var(--primary-50)",
    background_fill_primary_dark="#ffffff",
    background_fill_secondary_dark="var(--neutral-50)",
    border_color_accent_dark="var(--primary-300)",
    border_color_primary_dark="#e2e8f0",
    link_text_color_dark="#4f46e5",
    link_text_color_active_dark="var(--secondary-600)",
    link_text_color_hover_dark="#6366f1",
    link_text_color_visited_dark="var(--secondary-500)",
    body_text_color_subdued_dark="#64748b",
    accordion_text_color_dark="#1e293b",
    block_background_fill_dark="#ffffff",
    block_border_color_dark="#e2e8f0",
    block_label_background_fill_dark="var(--primary-100)",
    panel_background_fill_dark="#ffffff",
    panel_border_color_dark="#e2e8f0",
    code_background_fill_dark="#f1f5f9",
    checkbox_background_color_dark="var(--background-fill-primary)",
    checkbox_background_color_selected_dark="#4f46e5",
    checkbox_border_color_dark="var(--neutral-100)",
    checkbox_border_color_focus_dark="#4f46e5",
    checkbox_border_color_hover_dark="var(--neutral-300)",
    checkbox_border_color_selected_dark="#4f46e5",
    checkbox_border_width_dark="1px",
    checkbox_label_background_fill_selected_dark="linear-gradient(135deg, #4f46e5, #6366f1)",
    checkbox_label_border_color_selected_dark="#4f46e5",
    checkbox_label_text_color_selected_dark="#ffffff",
    error_background_fill_dark="#fef2f2",
    input_background_fill_dark="#ffffff",
    input_border_color_dark="#e2e8f0",
    input_border_color_focus_dark="rgba(79, 70, 229, 0.4)",
    input_border_color_hover_dark="#cbd5e1",
    input_placeholder_color_dark="#94a3b8",
    slider_color_dark="#4f46e5",
    stat_background_fill_dark="var(--primary-300)",
    table_border_color_dark="var(--neutral-300)",
    table_even_background_fill_dark="white",
    table_odd_background_fill_dark="var(--neutral-50)",
    button_primary_background_fill_dark="linear-gradient(135deg, #4f46e5, #6366f1)",
    button_primary_background_fill_hover_dark="linear-gradient(135deg, #6366f1, #818cf8)",
    button_primary_border_color_dark="transparent",
    button_primary_border_color_hover_dark="transparent",
    button_primary_shadow_hover_dark="0 6px 18px rgba(79, 70, 229, 0.28)",
    button_primary_shadow_active_dark="var(--shadow-inset)",
    button_secondary_background_fill_dark="#ffffff",
    button_secondary_background_fill_hover_dark="#f1f5f9",
    button_secondary_border_color_dark="#e2e8f0",
    button_secondary_border_color_hover_dark="#cbd5e1",
    button_secondary_shadow_hover_dark="none",
    button_secondary_shadow_active_dark="var(--shadow-inset)",
)


def resolve_cache_dir() -> str:
    """优先使用持久化目录；不可写时回退到本地目录，避免启动直接失败。"""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        probe = os.path.join(CACHE_DIR, ".write_probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        return CACHE_DIR
    except OSError as exc:
        fallback = os.path.join(BASE_DIR, ".model_cache")
        logger.warning(
            "持久化缓存目录 %s 不可用(%s)，回退到 %s（重启后需重新下载模型）",
            CACHE_DIR,
            exc,
            fallback,
        )
        os.makedirs(fallback, exist_ok=True)
        return fallback


def load_model(path: str) -> AutoModelForCausalLM:
    """transformers 5.x 把 torch_dtype 重命名为 dtype，这里做双向兼容。"""
    for kwargs in ({"dtype": torch.bfloat16}, {"torch_dtype": torch.bfloat16}):
        try:
            return AutoModelForCausalLM.from_pretrained(path, **kwargs)
        except TypeError:
            logger.info("from_pretrained 不接受 %s，尝试另一种写法", next(iter(kwargs)))
    raise RuntimeError("无法加载模型：from_pretrained 既不接受 dtype 也不接受 torch_dtype")


def extract_answer(full_text: str, thinking_enabled: bool) -> str:
    """取出正式回复部分，取出思考段之后的正式回复。

    显示层的思考段折叠由 Chatbot(reasoning_tags=...) 原生处理，这里只负责为
    回喂 prompt 的历史取出干净的 answer：历史中只存
    中的 finalAnswer 同样是不含思考段的 answer。

    JS 的 String.replace(字符串, 空串) 只替换首个匹配，这里用 count=1 对齐该行为。
    """
    text = full_text.replace(THINK_OPEN, "", 1).replace(EOS_MARKER, "", 1)
    if not thinking_enabled:
        return text.strip()
    pos = text.find(THINK_CLOSE)
    if pos == -1:
        # 思考段尚未闭合，还没有正式回复
        return ""
    return text[pos + len(THINK_CLOSE):].strip()


logger.info(
    "环境: python=%s gradio=%s transformers=%s torch=%s cuda_available=%s",
    sys.version.split()[0],
    gr.__version__,
    transformers.__version__,
    torch.__version__,
    torch.cuda.is_available(),
)
if torch.cuda.is_available():
    logger.info(
        "GPU: %s | 可见显存 %.2f GiB",
        torch.cuda.get_device_name(0),
        torch.cuda.get_device_properties(0).total_memory / 1024**3,
    )

cache_dir = resolve_cache_dir()
logger.info("开始从 ModelScope 下载/加载 %s，缓存目录 %s", MODEL_ID, cache_dir)
model_path = snapshot_download(MODEL_ID, cache_dir=cache_dir)
logger.info("模型快照就绪: %s", model_path)

# 仓库内无 .py 建模文件，标准 LlamaForCausalLM，无需 trust_remote_code
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = load_model(model_path).to("cuda")
model.eval()
logger.info("模型已加载到 GPU，占用显存 %.2f GiB", torch.cuda.memory_allocated() / 1024**3)


def stream_generate(messages, thinking_mode, temperature, top_p):
    """后台线程跑 model.generate，主线程按 token 产出累计文本。"""
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking_mode,
    )
    # 不加 truncation：llama 分词器默认 truncation_side 为 right，一旦触发会截掉
    # 末尾的 generation prompt，导致输出异常
    model_inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda")

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,
    )

    gen_kwargs = dict(
        **model_inputs,
        streamer=streamer,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    if temperature > 0:
        gen_kwargs.update(temperature=temperature, top_p=top_p, do_sample=True)
    else:
        # temperature=0 走贪婪解码；此时不能下发 temperature/top_p，
        # 否则 transformers 会警告这些参数被忽略
        gen_kwargs.update(do_sample=False)

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    # chat template 在 enable_thinking 时把开头 <think> 写进了 generation prompt，
    # 被 TextIteratorStreamer(skip_prompt=True) 丢弃，导致流式文本缺少开标签、
    # Chatbot 的 reasoning_tags 正则匹配不上而把 </think>/<|im_end|> 原样显示。
    # 这里给显示文本补回开标签；同时去掉结尾的 <|im_end|>（EOS 在气泡里无意义）。
    prefix = THINK_OPEN if thinking_mode else ""
    full_text = ""
    for new_token_text in streamer:
        if not new_token_text:
            continue
        full_text += new_token_text
        yield prefix + full_text.replace(EOS_MARKER, "")

    thread.join()
    logger.info(
        "生成完成: 输入 %d token, 输出 %d 字符, 峰值显存 %.2f GiB",
        model_inputs["input_ids"].shape[-1],
        len(full_text),
        torch.cuda.max_memory_allocated() / 1024**3,
    )


def respond(message, chat_history, raw_history, thinking_mode, temperature, top_p):
    """Chatbot 的流式回调。gradio 6.x 中 messages 是唯一的历史格式。"""
    history = list(chat_history or [])
    raw = list(raw_history or [])
    text = (message or "").strip()
    if not text:
        yield history, "", raw
        return

    # 复用 organize_messages（吃 [[user, assistant], ...] 结构）
    messages = organize_messages(text, raw)

    history.append({"role": "user", "content": message})
    yield list(history), "", raw

    history.append({"role": "assistant", "content": ""})
    full_text = ""
    for full_text in stream_generate(messages, thinking_mode, temperature, top_p):
        # 思考段的折叠展示交给 Chatbot(reasoning_tags=...)，这里直接透传原始输出
        history[-1] = {"role": "assistant", "content": full_text}
        yield list(history), "", raw

    # 只在未中断且确有正式回复时才把本轮写入历史；
    # 写入的是去思考段的 answer，避免思考内容污染后续上下文
    answer = extract_answer(full_text, thinking_mode)
    if answer:
        raw.append([text, answer])
    yield list(history), "", raw


def on_thinking_mode_change(chat_history, message, raw_history):
    """切换思考模式时清空历史：think/no-think 的 prompt 模板不同，
    混合历史会产生格式不一致的上下文。gradio 无 confirm 对话框，改为直接清空并提示。"""
    if not raw_history:
        return chat_history, message, raw_history
    gr.Info("已切换思考模式，对话历史已清空")
    return [], "", []


# gradio 6.0 起 theme / css 从 Blocks 构造器移到了 launch()，Blocks 也不再接受 title；
# 浏览器标题由平台引导页的 control_page_title 控制
with gr.Blocks() as demo:
    gr.Markdown(
        "![OpenBMB](/gradio_api/file=vendor/openbmb-logo.png)\n\n"
        "# MiniCPM5-2B",
        elem_classes=["hero"],
    )

    chatbot = gr.Chatbot(
        label="MiniCPM5-2B",
        height=520,
        layout="bubble",
        render_markdown=True,
        placeholder="发送一条消息开始对话。",
        # LaTeX 分隔符：$$ 与 $
        # gradio 默认只配 $$，不显式传入则行内 $...$ 不会被渲染
        latex_delimiters=[
            {"left": "$$", "right": "$$", "display": True},
            {"left": "$", "right": "$", "display": False},
        ],
        # gradio 6.x 原生思考段支持：标记之间的内容自动抽取为独立可折叠消息，
        # 未闭合时前端显示 pending 状态，闭合后转 done
        reasoning_tags=[(THINK_OPEN, THINK_CLOSE)],
    )

    # 与显示历史分离：chatbot 存模型原始输出（含思考标记），raw_state 只存正式回复，
    # 回喂 prompt 时只用 raw_state，避免思考内容塞进上下文
    raw_state = gr.State([])

    msg = gr.Textbox(
        placeholder="Ask MiniCPM5...",
        lines=1,
        max_lines=6,
        autofocus=True,
        show_label=False,
        # 用图标按钮（True）而非文字：文字按钮在次级配色下对比度过低不明显，
        # 图标配合 CUSTOM_CSS 的实心圆钮
        submit_btn=True,
        stop_btn=True,
        elem_classes=["chat-input"],
    )

    with gr.Accordion("思考模式与采样参数", open=True):
        thinking_mode = gr.Checkbox(label="思考模式（enable_thinking）", value=True)
        with gr.Row():
            # temp 0~1 step 0.05 默认 1.0，
            # top_p 0~1 step 0.01 默认 0.95
            temperature = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="temperature")
            top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="top_p")
            clear_btn = gr.Button("清空对话", variant="secondary")

    inputs = [msg, chatbot, raw_state, thinking_mode, temperature, top_p]
    outputs = [chatbot, msg, raw_state]

    submit_event = msg.submit(respond, inputs=inputs, outputs=outputs, api_name="predict")

    # 停止接线方式取自 gradio 官方 ChatInterface._setup_events；
    # 停止生成
    msg.stop(None, None, None, cancels=[submit_event], queue=False)

    clear_btn.click(
        lambda c, m, r: ([], "", []),
        inputs=[chatbot, msg, raw_state],
        outputs=outputs,
    )

    thinking_mode.change(
        on_thinking_mode_change,
        inputs=[chatbot, msg, raw_state],
        outputs=[chatbot, msg, raw_state],
    )

# 单并发排队：常驻 GPU 下并发推理会叠加显存占用与首 token 延迟
demo.queue(default_concurrency_limit=1, max_size=16)

if __name__ == "__main__":
    # theme / css 在 gradio 6.0 起由 launch() 接收，不能再传给 Blocks 构造器。
    # 必须显式绑定 0.0.0.0:7860：gradio 未收到 server_name 且平台未设
    # GRADIO_SERVER_NAME 时默认只绑 127.0.0.1，平台反向代理将无法访问
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=THEME,
        css=CUSTOM_CSS,
        # 头部 logo 需要走 gradio 的 /file= 路由；默认不在允许列表内会返回
        # 403 File not allowed。该参数文档明确要求绝对路径
        allowed_paths=[os.path.join(BASE_DIR, "vendor")],
        show_error=True,
    )
