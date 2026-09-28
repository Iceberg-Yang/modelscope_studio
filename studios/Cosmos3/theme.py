# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVIDIA 黑绿科技主题系统。

包含:
- nvidia_theme(): 自定义 Gradio Theme（NVIDIA 绿 + 深黑中性色）
- CUSTOM_CSS: 科技感全局样式（网格背景 / 霓虹发光 / 脉冲动画）
- HEADER_HTML / FOOTER_HTML: 品牌头部与页脚
"""

from __future__ import annotations

import gradio as gr

# ---------------------------------------------------------------------------
# NVIDIA 配色常量
# ---------------------------------------------------------------------------
NV_GREEN = "#76B900"        # NVIDIA 官方绿
NV_GREEN_BRIGHT = "#9BE22D"  # 高亮绿
NV_GREEN_DIM = "#3d6300"     # 暗绿（边框/次要）
BG_BLACK = "#0d0d0d"         # 主背景
BG_PANEL = "#161616"         # 面板背景
BG_PANEL_ALT = "#1c1c1c"     # 次级面板
TEXT_MAIN = "#e8e8e8"        # 主文字
TEXT_DIM = "#8a8a8a"         # 次要文字


def nvidia_theme() -> gr.themes.Base:
    """构建 NVIDIA 黑绿风格的 Gradio 主题。"""
    theme = gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#f4fcea",
            c100="#e6f9d1",
            c200="#cdf0a3",
            c300="#ade46b",
            c400="#9BE22D",
            c500=NV_GREEN,
            c600="#5f9500",
            c700="#4a7400",
            c800=NV_GREEN_DIM,
            c900="#2b4a00",
            c950="#1a2e00",
        ),
        secondary_hue=gr.themes.Color(
            c50="#f5f5f5",
            c100="#e5e5e5",
            c200="#d4d4d4",
            c300="#a3a3a3",
            c400="#737373",
            c500="#525252",
            c600="#404040",
            c700="#2e2e2e",
            c800="#1c1c1c",
            c900="#161616",
            c950=BG_BLACK,
        ),
        neutral_hue=gr.themes.Color(
            c50="#f5f5f5",
            c100="#e5e5e5",
            c200="#d4d4d4",
            c300="#a3a3a3",
            c400="#737373",
            c500="#525252",
            c600="#404040",
            c700="#2e2e2e",
            c800="#1c1c1c",
            c900="#161616",
            c950=BG_BLACK,
        ),
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    )
    theme.set(
        body_background_fill=BG_BLACK,
        body_text_color=TEXT_MAIN,
        block_background_fill=BG_PANEL,
        background_fill_secondary=BG_PANEL_ALT,
        block_label_background_fill=BG_PANEL_ALT,
        block_border_width="1px",
        block_border_color=NV_GREEN_DIM,
        block_title_text_color=TEXT_MAIN,
        block_label_text_color=TEXT_MAIN,
        input_background_fill=BG_PANEL_ALT,
        input_border_color=NV_GREEN_DIM,
        input_border_color_focus=NV_GREEN,
        button_primary_background_fill=f"linear-gradient(135deg, {NV_GREEN} 0%, {NV_GREEN_BRIGHT} 100%)",
        button_primary_background_fill_hover=f"linear-gradient(135deg, {NV_GREEN_BRIGHT} 0%, {NV_GREEN} 100%)",
        button_primary_text_color="#0d0d0d",
        button_primary_text_color_hover="#0d0d0d",
        button_primary_border_color=NV_GREEN,
        button_secondary_background_fill=BG_PANEL_ALT,
        button_secondary_background_fill_hover="#242424",
        button_secondary_text_color=TEXT_MAIN,
        button_secondary_border_color=NV_GREEN_DIM,
        border_color_accent=NV_GREEN_DIM,
        shadow_drop="0 4px 24px rgba(118, 185, 0, 0.08)",
        panel_background_fill=BG_PANEL,
        panel_border_color=NV_GREEN_DIM,
    )
    # Chatbot 专用主题属性（若 Gradio 版本支持则生效）
    for _prop, _val in {
        "chatbot_message_background_fill": BG_PANEL,
        "chatbot_user_message_background_fill": f"rgba(118,185,0,0.08)",
        "chatbot_bot_message_background_fill": BG_PANEL_ALT,
        "chatbot_message_text_color": TEXT_MAIN,
    }.items():
        try:
            theme.set(**{_prop: _val})
        except Exception:
            pass  # 属性不存在则静默跳过
    return theme


# ---------------------------------------------------------------------------
# 全局自定义 CSS（科技感）
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* ===== 全局背景：暗色网格 + 顶部绿色辉光 ===== */
.gradio-container {
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(118,185,0,0.13) 0%, transparent 60%),
        linear-gradient(rgba(118,185,0,0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(118,185,0,0.035) 1px, transparent 1px),
        #0d0d0d !important;
    background-size: 100% 100%, 42px 42px, 42px 42px, 100% 100% !important;
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 24px !important;
    margin: 0 !important;
}

/* ===== 顶部品牌区 ===== */
.nv-header {
    text-align: center;
    padding: 34px 16px 18px;
    border-bottom: 1px solid rgba(118,185,0,0.25);
    margin-bottom: 8px;
    position: relative;
}
.nv-header::after {
    content: "";
    position: absolute;
    left: 50%; bottom: -1px;
    transform: translateX(-50%);
    width: 220px; height: 2px;
    background: linear-gradient(90deg, transparent, #76B900, transparent);
    box-shadow: 0 0 14px #76B900;
}
.nv-eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; letter-spacing: 3px; text-transform: uppercase;
    color: #76B900;
    border: 1px solid rgba(118,185,0,0.4);
    border-radius: 999px;
    padding: 4px 14px;
    background: rgba(118,185,0,0.07);
}
.nv-eyebrow .dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #76B900;
    box-shadow: 0 0 8px #76B900;
    animation: nv-pulse 1.6s ease-in-out infinite;
}
.nv-title {
    font-family: 'Orbitron', 'Inter', sans-serif;
    font-size: clamp(30px, 5vw, 46px);
    font-weight: 800;
    letter-spacing: 4px;
    margin: 14px 0 6px;
    color: #e8e8e8;
    text-shadow:
        0 0 18px rgba(118,185,0,0.55),
        0 0 46px rgba(118,185,0,0.28);
}
.nv-title .accent { color: #76B900; }
.nv-subtitle {
    font-size: 14px; color: #8a8a8a;
    letter-spacing: 1px;
    margin-bottom: 16px;
}
.nv-badges { display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; }
.nv-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; letter-spacing: 1px;
    color: #9BE22D;
    background: rgba(118,185,0,0.08);
    border: 1px solid rgba(118,185,0,0.35);
    border-radius: 6px;
    padding: 4px 12px;
}

/* ===== 模型状态栏 ===== */
.nv-status {
    display: flex; align-items: center; justify-content: center; gap: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; letter-spacing: 1px;
    color: #8a8a8a;
    border: 1px solid rgba(118,185,0,0.25);
    border-radius: 8px;
    background: rgba(22,22,22,0.85);
    padding: 8px 16px;
    margin: 10px auto 4px;
    max-width: 760px;
}
.nv-status .led {
    width: 9px; height: 9px; border-radius: 50%;
    background: #525252; flex: none;
}
.nv-status.ready .led { background: #76B900; box-shadow: 0 0 10px #76B900; }
.nv-status.loading .led {
    background: #9BE22D; box-shadow: 0 0 10px #9BE22D;
    animation: nv-pulse 0.9s ease-in-out infinite;
}
.nv-status.idle .led { background: #525252; }

/* ===== Tab 导航 ===== */
.tabs > .tab-nav {
    background: transparent !important;
    border-bottom: 1px solid rgba(118,185,0,0.25) !important;
    gap: 6px;
}
.tabs > .tab-nav > button {
    font-family: 'Orbitron', 'Inter', sans-serif !important;
    font-size: 13px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase;
    color: #8a8a8a !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    border-bottom: none !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 22px !important;
    transition: all .25s ease;
}
.tabs > .tab-nav > button.selected {
    color: #9BE22D !important;
    background: rgba(118,185,0,0.08) !important;
    border-color: rgba(118,185,0,0.4) !important;
    box-shadow: 0 -2px 18px rgba(118,185,0,0.12);
}

/* ===== 面板卡片 ===== */
.panel-card {
    border: 1px solid rgba(118,185,0,0.22) !important;
    border-radius: 12px !important;
    background: linear-gradient(160deg, rgba(28,28,28,0.92), rgba(18,18,18,0.96)) !important;
    box-shadow: 0 0 0 1px rgba(118,185,0,0.04), 0 8px 32px rgba(0,0,0,0.5) !important;
}
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; letter-spacing: 2px; text-transform: uppercase;
    color: #76B900 !important;
}

/* ===== 主按钮：发光 ===== */
.primary-btn button, button.primary {
    font-family: 'Orbitron', 'Inter', sans-serif !important;
    letter-spacing: 2px !important;
    text-transform: uppercase;
    box-shadow: 0 0 18px rgba(118,185,0,0.35) !important;
    transition: all .25s ease !important;
}
.primary-btn button:hover, button.primary:hover {
    box-shadow: 0 0 30px rgba(155,226,45,0.55) !important;
    transform: translateY(-1px);
}

/* ===== 输出媒体区：绿色描边 ===== */
.output-media, .output-media video, .output-media img {
    border-radius: 10px !important;
}
.output-media {
    border: 1px solid rgba(118,185,0,0.3) !important;
    box-shadow: inset 0 0 30px rgba(118,185,0,0.05), 0 0 22px rgba(118,185,0,0.08) !important;
}

/* ===== Chatbot 暗色覆盖（panel layout） ===== */
/* Gradio 6.x Chatbot 使用 Svelte scoped CSS，需要 !important + 精准选择器 */
#reason-chat,
#reason-chat gradio-chatbot {
    background: #141414 !important;
    --background-fill: #141414 !important;
    --block-background-fill: #141414 !important;
}
/* 所有子元素背景清除 / 覆盖 */
#reason-chat > *,
#reason-chat gradio-chatbot *,
#reason-chat [class*="message"],
#reason-chat [class*="bubble"],
#reason-chat [class*="content"],
#reason-chat [class*="row"] {
    background: transparent !important;
}
/* 消息气泡 — 暗色底 */
#reason-chat [class*="message"]:not([class*="row"]):not([class*="list"]):not([class*="wrap"]),
#reason-chat [class*="bubble"],
#reason-chat [data-testid*="message"],
#reason-chat .panel-message,
#reason-chat .message-bubble-border {
    background: #1c1c1c !important;
    color: #e8e8e8 !important;
    border: 1px solid rgba(118,185,0,0.15) !important;
    border-radius: 8px !important;
}
/* 用户消息 — 绿色微光 */
#reason-chat [class*="message"][class*="user"],
#reason-chat [data-role="user"],
#reason-chat .user-row [class*="message"],
#reason-chat [class*="user"][class*="message"] {
    background: rgba(118,185,0,0.08) !important;
    border-color: rgba(118,185,0,0.3) !important;
    color: #cfe8a8 !important;
}
/* AI 消息 — 深色面板 */
#reason-chat [class*="message"][class*="bot"],
#reason-chat [class*="message"][class*="assistant"],
#reason-chat [data-role="assistant"],
#reason-chat [data-role="bot"],
#reason-chat .bot-row [class*="message"],
#reason-chat [class*="assistant"][class*="message"] {
    background: #1a1a1a !important;
    border-color: rgba(118,185,0,0.2) !important;
    color: #e8e8e8 !important;
}
/* 消息内所有文字继承颜色 */
#reason-chat [class*="message"] *,
#reason-chat [class*="bubble"] *,
#reason-chat [class*="content"] p,
#reason-chat [class*="content"] span {
    color: inherit !important;
}

/* ===== Dropdown / Slider 暗色覆盖 ===== */
/* Dropdown 选项列表 */
.gradio-container .dropdown-list,
.gradio-container [class*="dropdown"] ul,
.gradio-container [class*="dropdown"] li {
    background: #1c1c1c !important;
    color: #e8e8e8 !important;
}
.gradio-container [class*="dropdown"] li:hover,
.gradio-container [class*="dropdown"] li.selected {
    background: rgba(118,185,0,0.15) !important;
    color: #9BE22D !important;
}
/* Dropdown 显示框 */
.gradio-container [class*="dropdown"] input,
.gradio-container [class*="dropdown"] [class*="wrap"] {
    background: #1c1c1c !important;
    color: #e8e8e8 !important;
    border-color: rgba(118,185,0,0.3) !important;
}
/* Slider 轨道与手柄 */
.gradio-container input[type=range] {
    background: #1c1c1c !important;
}
.gradio-container [class*="slider"] [class*="track"] {
    background: #2e2e2e !important;
}
.gradio-container [class*="slider"] [class*="thumb"],
.gradio-container input[type=range]::-webkit-slider-thumb {
    background: #76B900 !important;
    box-shadow: 0 0 8px rgba(118,185,0,0.5) !important;
}
.gradio-container input[type=range]::-webkit-slider-runnable-track {
    background: #2e2e2e !important;
    border: 1px solid rgba(118,185,0,0.2) !important;
}
/* 数字输入框 */
.gradio-container input[type=number],
.gradio-container input[type=text],
.gradio-container textarea {
    background: #1c1c1c !important;
    color: #e8e8e8 !important;
    border-color: rgba(118,185,0,0.25) !important;
}
.gradio-container input:focus,
.gradio-container textarea:focus {
    border-color: #76B900 !important;
    box-shadow: 0 0 0 2px rgba(118,185,0,0.12) !important;
}
/* 标签文字 */
.gradio-container label,
.gradio-container [class*="label"] {
    color: #e8e8e8 !important;
}
/* info 提示文字 */
.gradio-container [class*="info"],
.gradio-container [class*="caption"] {
    color: #8a8a8a !important;
}

/* ===== Examples 暗色化（强制覆盖 Svelte scoped CSS）===== */
/* 步骤 1: 清除所有子元素背景 */
#img-examples *,
#vid-examples *,
.gradio-container [data-testid*="example"] * {
    background: transparent !important;
    background-color: transparent !important;
}
/* 步骤 2: 恢复表格背景 */
#img-examples table,
#vid-examples table {
    background: rgba(28,28,28,0.5) !important;
    border-color: rgba(118,185,0,0.12) !important;
}
/* 步骤 3: 恢复按钮背景 */
#img-examples button,
#vid-examples button {
    background: rgba(118,185,0,0.06) !important;
    border: 1px solid rgba(118,185,0,0.25) !important;
    color: #cfe8a8 !important;
    border-radius: 8px !important;
    transition: all .2s ease;
}
#img-examples button:hover,
#vid-examples button:hover {
    background: rgba(118,185,0,0.14) !important;
    box-shadow: 0 0 12px rgba(118,185,0,0.2);
}
/* 步骤 4: 文字颜色 */
#img-examples td,
#img-examples th,
#vid-examples td,
#vid-examples th,
#img-examples label,
#vid-examples label,
#img-examples span,
#vid-examples span,
#img-examples p,
#vid-examples p {
    color: #e8e8e8 !important;
    border-color: rgba(118,185,0,0.08) !important;
}
#img-examples [class*="label"],
#vid-examples [class*="label"] {
    color: #76B900 !important;
}
/* 步骤 5: 行 hover */
#img-examples tr:hover td,
#vid-examples tr:hover td {
    background: rgba(118,185,0,0.06) !important;
}
/* 步骤 6: 图片圆角 */
#img-examples img,
#vid-examples img,
#img-examples video,
#vid-examples video {
    border-radius: 6px !important;
    background: transparent !important;
}

/* ===== 页脚 ===== */
.nv-footer {
    text-align: center;
    margin-top: 26px;
    padding: 18px 12px 8px;
    border-top: 1px solid rgba(118,185,0,0.2);
    color: #525252;
    font-size: 12px;
    letter-spacing: .5px;
    line-height: 1.9;
}
.nv-footer .nv-green { color: #76B900; }
.nv-footer a { color: #76B900; text-decoration: none; }
.nv-footer a:hover { color: #9BE22D; text-decoration: underline; }

/* ===== 动画 ===== */
@keyframes nv-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.25; }
}

/* 加载遮罩文字 */
.generating {
    font-family: 'JetBrains Mono', monospace;
    color: #9BE22D !important;
    letter-spacing: 1px;
}
"""


# ---------------------------------------------------------------------------
# 头部 / 页脚 HTML
# ---------------------------------------------------------------------------
HEADER_HTML = """
<div class="nv-header">
  <span class="nv-eyebrow"><span class="dot"></span>NVIDIA · PHYSICAL AI · WORLD FOUNDATION MODEL</span>
  <h1 class="nv-title">COSMOS<span class="accent">3</span> EDGE</h1>
  <div class="nv-subtitle">全模态世界模型 · 理解物理世界 · 生成未来画面 · 输出机器人动作</div>
  <div class="nv-badges">
    <span class="nv-badge">4B PARAMS</span>
    <span class="nv-badge">OMNIMODAL</span>
    <span class="nv-badge">480P VIDEO</span>
    <span class="nv-badge">REAL-TIME REASONING</span>
    <span class="nv-badge">OPENMDW-1.1</span>
  </div>
</div>
"""

FOOTER_HTML = """
<div class="nv-footer">
  <span class="nv-green">NVIDIA Cosmos3-Edge</span> · OpenMDW-1.1 · Powered by ModelScope xGPU
</div>
"""
