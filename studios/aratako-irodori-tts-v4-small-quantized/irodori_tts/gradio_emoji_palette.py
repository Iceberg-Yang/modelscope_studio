from __future__ import annotations

from dataclasses import dataclass
from html import escape

import gradio as gr


@dataclass(frozen=True)
class EmojiPaletteItem:
    emoji: str
    label: str
    description: str


EMOJI_PALETTE_CSS = """
.emoji-palette {
    max-width: 100%;
}

.emoji-palette-grid {
    display: flex;
    gap: 3px;
    flex-wrap: wrap;
    align-items: flex-start;
    max-height: 124px;
    overflow-y: auto;
}

.emoji-palette-button {
    flex: 0 0 28px !important;
    min-width: 28px !important;
    max-width: 28px !important;
}

.emoji-palette-button button {
    width: 28px !important;
    min-width: 28px !important;
    height: 28px !important;
    min-height: 28px !important;
    border-radius: 4px;
    font-size: 17px;
    line-height: 1;
    padding: 0 !important;
}

button.emoji-palette-button {
    width: 28px;
    min-width: 28px;
    height: 28px;
    min-height: 28px;
    border-radius: 4px;
    font-size: 17px;
    line-height: 1;
    padding: 0;
    cursor: pointer;
}
"""


EMOJI_PALETTE_ITEMS: tuple[EmojiPaletteItem, ...] = (
    EmojiPaletteItem("👂", "耳语", "耳畔音"),
    EmojiPaletteItem("😮‍💨", "吐息", "叹息、睡息"),
    EmojiPaletteItem("⏸️", "停顿", "沉默"),
    EmojiPaletteItem("🤭", "嬉笑", "嘿嘿笑、含笑"),
    EmojiPaletteItem("🥵", "喘声", "呻吟、喘鸣"),
    EmojiPaletteItem("📢", "回声", "混响"),
    EmojiPaletteItem("😏", "调侃", "撒娇般地"),
    EmojiPaletteItem("🥺", "颤声", "缺乏自信地"),
    EmojiPaletteItem("🌬️", "气短", "粗糙的呼吸声"),
    EmojiPaletteItem("😮", "倒吸气", "倒吸一口气"),
    EmojiPaletteItem("👅", "舔音", "咀嚼音、水声"),
    EmojiPaletteItem("💋", "唇音", "啵嘴声"),
    EmojiPaletteItem("🫶", "温柔", "柔情地"),
    EmojiPaletteItem("😭", "哭声", "啜泣、悲伤"),
    EmojiPaletteItem("😱", "尖叫", "喊叫、绝叫"),
    EmojiPaletteItem("😪", "困倦", "慵懒地"),
    EmojiPaletteItem("😴", "梦话", "鼾声"),
    EmojiPaletteItem("⏩", "快速", "一口气、急促地"),
    EmojiPaletteItem("📞", "电话音", "扬声器音质"),
    EmojiPaletteItem("🐢", "慢速", "缓慢地"),
    EmojiPaletteItem("🥤", "吞咽", "咽口水声"),
    EmojiPaletteItem("🤧", "咳鼻", "咳嗽、吸鼻涕"),
    EmojiPaletteItem("😒", "啮舌", "啧啧声"),
    EmojiPaletteItem("😰", "慌张", "动摇、紧张、口吃"),
    EmojiPaletteItem("😆", "喜悦", "开心地"),
    EmojiPaletteItem("💥", "用力", "强有力的气势"),
    EmojiPaletteItem("😠", "愤怒", "不满、闹别扭"),
    EmojiPaletteItem("😲", "惊讶", "惊叹"),
    EmojiPaletteItem("🥱", "哈欠", "打哈欠"),
    EmojiPaletteItem("😖", "痛苦", "痛苦地"),
    EmojiPaletteItem("😟", "担心", "不安地"),
    EmojiPaletteItem("🫣", "害羞", "害羞地"),
    EmojiPaletteItem("🙄", "无语", "无可奈何地"),
    EmojiPaletteItem("😊", "愉快", "高兴地"),
    EmojiPaletteItem("😎", "得意", "自信满满地"),
    EmojiPaletteItem("👌", "附和", "点头声"),
    EmojiPaletteItem("🙏", "恳求", "请求般地"),
    EmojiPaletteItem("🥴", "醉酒", "醉醺地"),
    EmojiPaletteItem("🎵", "哼唱", "哼哼歌"),
    EmojiPaletteItem("🤐", "捂嘴", "闷住的声音"),
    EmojiPaletteItem("😌", "安心", "满足地"),
    EmojiPaletteItem("🤔", "疑问", "质疑地"),
    EmojiPaletteItem("💪", "有力", "注入力量地"),
    EmojiPaletteItem("👃", "嗅音", "嗅闻声"),
    EmojiPaletteItem("📖", "朗读", "旁白般地"),
)


_INSERT_EMOJI_ON_POINTER_DOWN = (
    "event.preventDefault();"
    "const root=this.closest('[data-irodori-emoji-palette]');"
    "const input=root?document.querySelector(root.dataset.target):null;"
    "if(!input)return;"
    "const emoji=this.dataset.emoji;"
    "const text=input.value||'';"
    "const focused=document.activeElement===input;"
    "const start=focused&&typeof input.selectionStart==='number'?input.selectionStart:text.length;"
    "const end=focused&&typeof input.selectionEnd==='number'?input.selectionEnd:text.length;"
    "const next=text.slice(0,start)+emoji+text.slice(end);"
    "const caret=start+emoji.length;"
    "input.value=next;"
    "input.focus({preventScroll:true});"
    "input.setSelectionRange(caret,caret);"
    "input.dispatchEvent(new Event('input',{bubbles:true}));"
    "input.dispatchEvent(new Event('change',{bubbles:true}));"
)


def _textbox_selector(textbox: gr.Textbox) -> str:
    elem_id = getattr(textbox, "elem_id", None)
    root_selector = f"#{elem_id}" if elem_id else f"#component-{textbox._id}"
    return f"{root_selector} textarea, {root_selector} input:not([type='hidden'])"


def _emoji_palette_html(textbox: gr.Textbox) -> str:
    target = escape(_textbox_selector(textbox), quote=True)
    handler = escape(_INSERT_EMOJI_ON_POINTER_DOWN, quote=True)
    buttons = []
    for item in EMOJI_PALETTE_ITEMS:
        emoji = escape(item.emoji, quote=True)
        title = escape(f"{item.label}: {item.description}", quote=True)
        buttons.append(
            '<button type="button" '
            'class="emoji-palette-button" '
            f'data-emoji="{emoji}" '
            f'title="{title}" '
            f'aria-label="{title}" '
            f'onpointerdown="{handler}">'
            f"{emoji}</button>"
        )
    return (
        '<div class="emoji-palette-grid" '
        'data-irodori-emoji-palette="true" '
        f'data-target="{target}">'
        f"{''.join(buttons)}</div>"
    )


def build_emoji_palette(textbox: gr.Textbox, *, open: bool = True) -> None:
    with gr.Accordion("表情符号调色板", open=open, elem_classes=["emoji-palette"]):
        gr.HTML(_emoji_palette_html(textbox))
