/* 本地前端：最新按键快照、有限播放队列；不加载CDN。 */
function createKeyState() {
    const held = new Map();
    const aliases = {KeyW: 'w', KeyA: 'a', KeyS: 's', KeyD: 'd', KeyI: 'i', KeyJ: 'j', KeyK: 'k', KeyL: 'l', ArrowUp: 'i', ArrowDown: 'k', ArrowLeft: 'j', ArrowRight: 'l'};
    return {
        update(code, down, source = code) {
            if (!aliases[code]) return false;
            const changed = down ? !held.has(source) : held.has(source);
            if (down) held.set(source, aliases[code]); else held.delete(source);
            return changed;
        },
        keys() { return [...new Set(held.values())].sort(); },
        clear() { held.clear(); }
    };
}

class SegmentQueue {
    constructor() { this.reset(); }
    reset() { this.played = -1; this.current = null; this.pending = null; }
    accept(items) {
        for (const item of items) {
            if (!Number.isInteger(item.index) || item.index <= this.played || item.index === this.current?.index || item.index === this.pending?.index) continue;
            if (this.current === null && item.index === this.played + 1) this.current = item;
            else if (this.current && item.index === this.current.index + 1 && this.pending === null) this.pending = item;
        }
    }
    ended(index) {
        if (this.current?.index !== index) return false;
        this.played = index;
        this.current = this.pending;
        this.pending = null;
        return true;
    }
}

function mountKeyboard(element, trigger, props, watch) {
    const root = element.querySelector('.keyboard-root');
    if (!root) return;
    const keys = createKeyState(), queue = new SegmentQueue();
    root.innerHTML = `<div class="keyboard-controls"><div class="keypad" aria-label="移动"></div><div class="viewer"><video muted playsinline preload="auto"></video><video muted playsinline preload="auto"></video><button class="resume" hidden>点击播放片段</button><div class="empty">点击下方“开始键盘会话”，准备完成后操控</div></div><div class="keypad turn" aria-label="转向"></div></div><div class="live-status" role="status"></div><div class="key-status"></div><div class="live-actions"><button class="stop">停止</button><button class="reset">重置画面</button></div>`;
    for (const [selector, codes] of [['.keypad', ['w', 'a', 's', 'd']], ['.turn', ['i', 'j', 'k', 'l']]]) {
        for (const key of codes) {
            const button = document.createElement('button');
            button.textContent = key.toUpperCase();
            button.dataset.code = 'Key' + key.toUpperCase();
            root.querySelector(selector).appendChild(button);
        }
    }
    const image = document.createElement('img');
    image.className = 'example-image'; image.alt = '固定示例00输入图片';
    if (typeof props.example_image === 'string' && props.example_image.startsWith('data:image/jpeg;base64,')) image.src = props.example_image;
    root.querySelector('.viewer').prepend(image);
    const videos = [...root.querySelectorAll('video')];
    const status = root.querySelector('.live-status'), keyStatus = root.querySelector('.key-status');
    const resume = root.querySelector('.resume'), empty = root.querySelector('.empty');
    let runId = null, active = false, seq = 0, lastSend = -Infinity, dirty = true, stopping = false, resetPending = false;
    let visibleSlot = 0, playingIndex = null, latestReport = {}, awaitingSeq = -1, stopSent = false;
    const abort = new AbortController();
    const on = (node, event, handler) => node.addEventListener(event, handler, {signal: abort.signal});
    function clearKeys() { keys.clear(); dirty = true; renderKeys(); }
    function releaseKeys() { clearKeys(); send(true); }
    function renderKeys() {
        const current = keys.keys();
        root.querySelectorAll('[data-code]').forEach(button => button.classList.toggle('held', current.includes(button.dataset.code.slice(3).toLowerCase())));
        keyStatus.textContent = `已应用：${(latestReport.applied_keys || []).join(' ').toUpperCase() || '无'} · 待生效：${current.join(' ').toUpperCase() || '无'}（下一片段采样）`;
    }
    function send(force = false) {
        const now = performance.now();
        if (!active || !runId || now - lastSend < 100 || (!force && !dirty && now - lastSend < 1000)) return;
        if (awaitingSeq > (latestReport.ack_seq ?? -1) && !(stopping && !stopSent)) return;
        lastSend = now; dirty = false; awaitingSeq = seq++;
        if (stopping) stopSent = true;
        trigger('control', {run_id: runId, seq: awaitingSeq, keys: keys.keys(), played: queue.played, stop: stopping});
    }
    function clearVideo() {
        queue.reset(); playingIndex = null;
        for (const video of videos) { video.pause(); video.removeAttribute('src'); video.load(); video.hidden = true; video.dataset.index = ''; }
        empty.hidden = false; resume.hidden = true;
    }
    function mediaURL(item) {
        const expected = `/tmp/lingbot-public-media/interactive/${runId}/chunk-${String(item.index).padStart(3, '0')}.mp4`;
        if (item.path !== expected) throw new Error('媒体路径不匹配');
        return `./gradio_api/file=${encodeURIComponent(expected)}`;
    }
    function load(video, item) {
        if (video.dataset.index !== String(item.index)) {
            video.dataset.index = String(item.index); video.src = mediaURL(item); video.load();
        }
    }
    function preloadPending() {
        // 只有当前片段真正开始显示，另一个video才能用于预加载。
        if (queue.pending && videos[visibleSlot].dataset.index === String(playingIndex)) load(videos[1 - visibleSlot], queue.pending);
    }
    function playNext() {
        if (!queue.current || playingIndex === queue.current.index) return;
        const old = videos[visibleSlot], nextSlot = 1 - visibleSlot, next = videos[nextSlot];
        load(next, queue.current);
        playingIndex = queue.current.index;
        // 新片段有画面后才隐藏旧片段，等待期间保留末帧。
        const index = playingIndex, session = runId;
        const show = () => {
            if (session !== runId || index !== playingIndex || next.dataset.index !== String(index)) return;
            old.hidden = true; next.hidden = false; visibleSlot = nextSlot; empty.hidden = true; preloadPending();
        };
        next.onplaying = show;
        next.play().then(() => { if (session === runId && index === playingIndex) resume.hidden = true; }).catch(() => {
            if (session !== runId || index !== playingIndex) return;
            resume.hidden = false; status.textContent = '浏览器暂停自动播放，请点击播放片段。';
        });
    }
    on(resume, 'click', () => {
        const video = videos.find(v => v.dataset.index === String(playingIndex));
        video?.play().then(() => { resume.hidden = true; }).catch(() => { status.textContent = '播放失败，请停止并由维护者检查。'; });
    });
    for (const video of videos) {
        on(video, 'ended', () => {
            const index = Number(video.dataset.index);
            if (!queue.ended(index)) return;
            playingIndex = null; dirty = true; send(); playNext();
        });
        on(video, 'error', () => {
            if (!video.hasAttribute('src')) return;
            status.textContent = '片段播放失败，已请求停止。'; stopping = true; clearKeys(); send(true);
        });
    }
    on(root.querySelector('.viewer'), 'pointerdown', () => root.focus());
    on(root, 'keydown', event => {
        if (!active || stopping || event.repeat || event.ctrlKey || event.metaKey || event.altKey || /INPUT|TEXTAREA|SELECT/.test(event.target.tagName) || event.target.isContentEditable) return;
        if (keys.update(event.code, true)) { event.preventDefault(); if (!event.repeat) dirty = true; renderKeys(); }
    });
    on(root, 'keyup', event => {
        if (keys.update(event.code, false)) {
            if (!event.ctrlKey && !event.metaKey && !event.altKey && !/INPUT|TEXTAREA|SELECT/.test(event.target.tagName) && !event.target.isContentEditable) event.preventDefault();
            dirty = true; renderKeys();
        }
    });
    root.querySelectorAll('[data-code]').forEach(button => {
        on(button, 'pointerdown', event => { if (!active || stopping) return; event.preventDefault(); root.focus(); button.setPointerCapture(event.pointerId); keys.update(button.dataset.code, true, 'pointer-' + event.pointerId); dirty = true; renderKeys(); });
        for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) on(button, name, event => { keys.update(button.dataset.code, false, 'pointer-' + event.pointerId); dirty = true; renderKeys(); });
    });
    on(window, 'blur', releaseKeys);
    on(document, 'visibilitychange', () => { if (document.hidden) releaseKeys(); });
    on(root, 'focusout', event => { if (!root.contains(event.relatedTarget)) releaseKeys(); });
    on(root.querySelector('.stop'), 'click', () => { stopping = true; clearKeys(); send(true); });
    on(root.querySelector('.reset'), 'click', () => {
        clearKeys();
        if (active) { resetPending = true; stopping = true; send(true); }
        else clearVideo();
    });
    function update() {
        let report;
        try { report = JSON.parse(props.value || '{}'); } catch { return; }
        if (!report.run_id) return;
        if (report.run_id !== runId) {
            clearVideo(); clearKeys(); runId = report.run_id; seq = 0; stopping = false; resetPending = false; lastSend = -Infinity; awaitingSeq = -1; stopSent = false;
        }
        latestReport = report;
        active = ['preparing', 'running', 'stopping'].includes(report.status);
        const labels = {preparing: '准备模型与连续性对照', running: '会话运行中', stopping: '正在停止', stopped: '已停止，可重新开始', finished: '已达到会话上限，可重新开始', failed: '生成失败，请维护者查看日志'};
        const stages = {loading: '加载', equivalence: '原生等价性校验', conditioning: '准备图像条件', chunk: '生成片段', waiting: '等待下一片段', done: '结束'};
        status.textContent = `${labels[report.status] || '等待开始'} · ${stages[report.stage] || ''} · 完成 ${report.chunks_done || 0}/32 段${report.chunk_seconds !== undefined ? ' · 上段 ' + report.chunk_seconds.toFixed(2) + ' 秒' : ''}`;
        if (!active) clearKeys();
        queue.accept(report.segments || []);
        playNext();
        preloadPending();
        if (resetPending && !active) { clearVideo(); resetPending = false; }
        renderKeys(); send();
    }
    watch('value', update); update();
    const timer = setInterval(() => {
        if (!element.isConnected) { clearInterval(timer); abort.abort(); for (const v of videos) v.pause(); return; }
        if (root.getClientRects().length === 0) clearKeys();
        send();
    }, 100);
}

if (typeof module !== 'undefined' && module.exports) module.exports = {createKeyState, SegmentQueue, mountKeyboard};
if (typeof element !== 'undefined') mountKeyboard(element, trigger, props, watch);
