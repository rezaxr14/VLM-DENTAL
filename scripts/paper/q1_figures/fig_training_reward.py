"""Figure 4: SFT on verified traces, GRPO with live rollouts, the reward, dual-adapter RL."""
from common import *
import sys


def build(path):
    out.clear(); warns.clear()
    W, H = 1000, 622
    meta = json.load(open(SC + 'showcase.json'))
    gt = meta['flagship']['ground_truth']
    opg = Image.open(SC + 'flagship/image.jpg').convert('RGB')

    def sq(img, where='center', n=64):
        w, h = img.size; m = min(w, h); x0 = (w - m) // 2
        y0 = {'top': 0, 'center': (h - m) // 2}[where]
        return img.crop((x0, y0, x0 + m, y0 + m)).resize((n, n), Image.LANCZOS)
    cb = Image.open(SC + 'flagship/crop_before.jpg').convert('RGB')
    ca = Image.open(SC + 'flagship/crop_after.jpg').convert('RGB')
    tl = [Image.open(SC + f'params_dentex/tool_{i}.jpg').convert('RGB') for i in (1, 5, 6)]
    thumbs = {'a': sq(cb), 'b': sq(ca), 'c': sq(tl[0], 'top'), 'd': sq(tl[1], 'top'), 'e': sq(tl[2], 'top')}
    GOOD, MID = GR, '#E0A030'

    add(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Helvetica, Arial, \'Liberation Sans\', sans-serif">')
    add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')
    rect(16, 16, 420, 318, GRAY, rx=14); rect(452, 16, 532, 318, GRAY, rx=14)
    rect(16, 348, 968, 256, GRAY, rx=14)

    # ======================= 1 SFT =======================
    add('<g id="sft">')
    circ_num(36, 36, 1); text(54, 42, 'SFT on verified traces', 15, True, maxw=300)
    text(32, 66, '1,847 with-tools traces · QLoRA on Qwen3.5-9B', 12.5, False, 'start', MUTE, maxw=390)
    MASK, TH, CALL, ANS = '#D5D8DE', '#4F7FC4', '#1F4E96', '#2E9A68'
    kx = 32
    for col, lab, pre in [(MASK, 'input, masked', None), (TH, 'thought', 'loss on:'), (CALL, 'tool call', None), (ANS, 'answer', None)]:
        if pre: text(kx, 90, pre, 12, True, 'start', INK); kx += tw(pre, 12, True) + 8
        rect(kx, 80, 12, 11, (col, col if col != MASK else '#9AA0A8'), rx=3); text(kx + 17, 90, lab, 12, False, 'start', INK)
        kx += 17 + tw(lab, 12) + 14
    def run(x, y, n, col, s=11, gap=2):
        for i in range(n):
            add(f'<rect x="{x + i * (s + gap)}" y="{y}" width="{s}" height="{s}" rx="2.5" fill="{col}"/>')
        return x + n * (s + gap)
    def patch(x, y, w, h, im, nx, ny):
        image(x, y, w, h, im.resize((w * 4, h * 4), Image.LANCZOS))
        for c_ in range(1, nx): add(f'<line x1="{x + c_ * w / nx:.1f}" y1="{y}" x2="{x + c_ * w / nx:.1f}" y2="{y + h}" stroke="#FFFFFF" stroke-opacity="0.85" stroke-width="1"/>')
        for r_ in range(1, ny): add(f'<line x1="{x}" y1="{y + r_ * h / ny:.1f}" x2="{x + w}" y2="{y + r_ * h / ny:.1f}" stroke="#FFFFFF" stroke-opacity="0.85" stroke-width="1"/>')
        add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="#9AA0A8" stroke-width="1.4"/>')
    y1 = 108; x = 32
    x = run(x, y1 + 25, 5, MASK) + 4
    patch(x, y1, 92, 60, opg.crop((0, 40, 760, 560)), 6, 4); xp = x; x += 96
    x = run(x, y1 + 25, 7, TH) + 4
    x = run(x, y1 + 25, 3, CALL)
    text(32, y1 + 76, 'prompt', 11.5, False, 'start', MUTE); text(xp + 46, y1 + 76, 'image patches', 11.5, False, 'middle', MUTE)
    y2 = 206; x = 32
    patch(x, y2, 48, 60, cb, 3, 4); xc1 = x; x += 52
    x = run(x, y2 + 25, 5, TH) + 4
    x = run(x, y2 + 25, 3, CALL) + 4
    patch(x, y2, 48, 60, ca, 3, 4); xc2 = x; x += 52
    x = run(x, y2 + 25, 3, TH) + 4
    x = run(x, y2 + 25, 4, ANS)
    text(xc1 + 24, y2 + 76, 'crop', 11.5, False, 'middle', MUTE); text(xc2 + 24, y2 + 76, 'crop', 11.5, False, 'middle', MUTE)
    text(32, 308, 'loss on thought, tool call and answer only;', 12.5, False, 'start', INK, maxw=380)
    text(32, 324, 'images and tool outputs are inputs, never targets', 12.5, False, 'start', INK, maxw=380)
    # the SFT adapter sits beside the rows, level with the gap between them, and feeds GRPO
    rect(352, 150, 80, 44, BLUE, rx=14); flame(368, 172, 0.95); text(402, 168, 'SFT', 12.5, True, 'middle', INK); text(402, 183, 'adapter', 12.5, True, 'middle', INK)
    arrow([(434, 172), (450, 172)], head=6)
    add('</g>')

    # ======================= 2 GRPO =======================
    add('<g id="grpo">')
    circ_num(472, 36, 2); text(490, 42, 'GRPO with live tool rollouts', 15, True, maxw=300)
    image(466, 92, 118, 62, opg.resize((472, 248), Image.LANCZOS)); frame(466, 92, 118, 62, '#6B7280', 1.2)
    text(466, 176, 'one image,', 12.5, False, 'start', MUTE); text(466, 192, 'K = 4 rollouts', 12.5, False, 'start', MUTE)
    text(466, 214, 'real tools,', 12.5, False, 'start', MUTE); text(466, 230, 'same dispatcher', 12.5, False, 'start', MUTE)
    text(598, 76, 'rollouts', 12.5, True, 'start', INK); text(758, 76, 'reward', 12.5, True, 'start', INK); text(920, 76, 'advantage', 12.5, True, 'middle', INK)
    rewards = [0.92, 0.55, 0.78, 0.20]
    m = sum(rewards) / 4; sd = (sum((r - m) ** 2 for r in rewards) / 4) ** 0.5
    adv = [(r - m) / (sd + 1e-4) for r in rewards]
    lanes = [list('ocaoc') + ['b'], list('ocaoc') + ['c'], list('ocaoc') + ['d'], list('occccccc')]
    finals = [GOOD, MID, GOOD, RD]
    for i in range(4):
        ly = 100 + i * 36
        add(f'<path d="M586,123 C596,123 592,{ly} 598,{ly}" fill="none" stroke="#C9CCD3" stroke-width="1.4"/>')
        add(f'<line x1="598" y1="{ly}" x2="738" y2="{ly}" stroke="#D5D8DE" stroke-width="1.6"/>')
        x = 604
        for s in lanes[i]:
            if s == 'o':
                add(f'<circle cx="{x}" cy="{ly}" r="4.6" fill="#FFFFFF" stroke="#8A8F98" stroke-width="1.5"/>'); x += 13
            elif s == 'c':
                add(f'<rect x="{x - 4.5}" y="{ly - 4.5}" width="9" height="9" rx="2" fill="{BL}"/>'); x += 13 if i < 3 else 12
            elif s == 'a':
                image(x - 2, ly - 12, 24, 24, thumbs['a']); frame(x - 2, ly - 12, 24, 24, '#8A8F98', 1); x += 31
            else:
                image(x - 2, ly - 12, 24, 24, thumbs[s]); frame(x - 2, ly - 12, 24, 24, '#8A8F98', 1); x += 31
        add(f'<circle cx="{x + 4}" cy="{ly}" r="6" fill="{finals[i]}"/>')
        bw = 70 * rewards[i]
        add(f'<rect x="758" y="{ly - 5}" width="70" height="10" rx="3" fill="#E3E5EA"/><rect x="758" y="{ly - 5}" width="{bw:.1f}" height="10" rx="3" fill="{finals[i]}"/>')
        text(834, ly + 4, f'{rewards[i]:.2f}', 12, True, 'start', INK)
        a = adv[i]; k = 22
        col = GR if a > 0 else RD
        x1 = 920 + a * k
        add(f'<rect x="{min(920, x1):.1f}" y="{ly - 6}" width="{abs(x1 - 920):.1f}" height="12" rx="2" fill="{col}"/>')
    add('<line x1="920" y1="88" x2="920" y2="230" stroke="#6B7280" stroke-width="1.4"/>')
    text(598, 244, 'illustrative rewards', 11.5, False, 'start', MUTE, italic=True)
    text(598, 262, 'A = (r − mean) / (std + ε)', 12.5, False, 'start', INK, maxw=170)
    arrow([(920, 234), (920, 242)], head=6)
    rect(776, 244, 196, 30, BLUE, rx=15); flame(796, 259, 0.95); text(810, 264, 'update policy adapter', 12.5, True, 'start', INK, maxw=156)
    text(598, 288, 'KL to the frozen SFT adapter', 12, False, 'start', MUTE, maxw=170)
    # colour legend for the rollouts, rewards and advantages
    lx = 468; ly_ = 306
    add(f'<circle cx="{lx + 5}" cy="{ly_ - 4}" r="4.6" fill="#FFFFFF" stroke="#8A8F98" stroke-width="1.5"/>'); text(lx + 14, ly_, 'thought', 12, False, 'start', INK); lx += 14 + tw('thought', 12) + 12
    add(f'<rect x="{lx}" y="{ly_ - 9}" width="9" height="9" rx="2" fill="{BL}"/>'); text(lx + 14, ly_, 'tool call', 12, False, 'start', INK); lx += 14 + tw('tool call', 12) + 12
    image(lx, ly_ - 12, 14, 14, thumbs['a']); frame(lx, ly_ - 12, 14, 14, '#8A8F98', 1); text(lx + 19, ly_, 'tool output crop', 12, False, 'start', INK); lx += 19 + tw('tool output crop', 12) + 14
    text(lx, ly_, 'answer, reward:', 12, True, 'start', INK); lx += tw('answer, reward:', 12, True) + 8
    for col_, lab_ in [(GOOD, 'good'), (MID, 'partial'), (RD, 'wrong')]:
        add(f'<circle cx="{lx + 5}" cy="{ly_ - 4}" r="5" fill="{col_}"/>'); text(lx + 14, ly_, lab_, 12, False, 'start', INK); lx += 14 + tw(lab_, 12) + 10
    add(f'<rect x="468" y="317" width="14" height="9" rx="2" fill="{GR}"/><rect x="485" y="317" width="14" height="9" rx="2" fill="{RD}"/>')
    text(505, 326, 'advantage: above / below the group mean', 12, False, 'start', INK)
    add('</g>')

    # ======================= 3 REWARD =======================
    add('<g id="reward" transform="translate(0,32)">')
    circ_num(36, 340, 3); text(54, 346, 'Reward: ground-truth anchored, correction-friendly', 15, True, maxw=520)
    text(968, 346, 'R = 1.0 R_acc + 0.2 R_fmt + 0.2 R_tool + 0.1 R_eff', 13, True, 'end', INK, maxw=380)
    add('<line x1="412" y1="362" x2="412" y2="556" stroke="#D5D8DE" stroke-width="1.4"/><line x1="608" y1="362" x2="608" y2="556" stroke="#D5D8DE" stroke-width="1.4"/>')
    # --- R_acc
    text(32, 380, 'R_acc: matched findings, no LLM judge', 13, True, 'start', INK, maxw=372)
    Y0, Y1 = 150, 480; DW, DH = 368, 110
    crop = opg.crop((0, Y0, 1100, Y1)).resize((DW * 2, DH * 2), Image.LANCZOS)
    image(32, 390, DW, DH, crop); frame(32, 390, DW, DH, '#6B7280', 1.2)
    k0 = 1100 / 2869
    def to_px(b):
        x, y, w, h = b
        return 32 + x * k0 * DW / 1100, 390 + (y * k0 - Y0) * DH / (Y1 - Y0), w * k0 * DW / 1100, h * k0 * DH / (Y1 - Y0)
    def ov(b, col, dash=None, sw=2):
        px, py, pw, ph = to_px(b)
        ds = f' stroke-dasharray="{dash}"' if dash else ''
        add(f'<rect x="{px:.1f}" y="{py:.1f}" width="{pw:.1f}" height="{ph:.1f}" fill="none" stroke="#000000" stroke-opacity="0.55" stroke-width="{sw + 2}"{ds}/>')
        add(f'<rect x="{px:.1f}" y="{py:.1f}" width="{pw:.1f}" height="{ph:.1f}" fill="none" stroke="{col}" stroke-width="{sw}"{ds}/>')
    for i in (0, 7): ov(gt[i]['bbox'], '#E5E7EB', '5 3', 1.8)           # missed
    for i in (1, 3, 4, 5, 6, 8): ov(gt[i]['bbox'], '#22C55E')           # exact
    ov(gt[2]['bbox'], '#F59E0B')                                         # right tooth, wrong class
    ov((1360, 900, 150, 290), '#EF4444')                                 # invented
    def key(x, y, col, dash, label):
        ds = ' stroke-dasharray="3 2"' if dash else ''
        add(f'<rect x="{x}" y="{y}" width="14" height="11" fill="none" stroke="{col}" stroke-width="2.4"{ds}/>')
        text(x + 19, y + 10, label, 12, False, 'start', INK)
    key(32, 512, '#22C55E', False, 'exact'); key(96, 512, '#F59E0B', False, 'wrong class')
    key(190, 512, '#9AA0A8', True, 'missed'); key(262, 512, '#EF4444', False, 'invented')
    text(32, 540, 'pair score = 0.25 quadrant + 0.25 tooth + 0.50 diagnosis', 12, False, 'start', INK, maxw=374)
    text(32, 556, 'image score = F1 of matched recall and precision', 12, False, 'start', INK, maxw=374)
    # --- R_fmt / R_tool
    text(428, 380, 'R_fmt and R_tool', 13, True, 'start', INK, maxw=170)
    check(436, 404, 8); text(450, 409, 'one valid JSON answer', 12.5, False, 'start', INK, maxw=150)
    text(450, 425, 'R_fmt = 1 if it parses', 11.5, False, 'start', MUTE, maxw=150)
    check(436, 452, 8); add(f'<text x="450" y="457" font-size="12" font-family="Courier, \'Liberation Mono\', monospace" fill="{INK}">zoom_crop(bbox)</text>')
    add(f'<circle cx="436" cy="486" r="8" fill="{RD}"/><path d="M432,482 l8,8 M440,482 l-8,8" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round"/>')
    add(f'<text x="450" y="491" font-size="12" font-family="Courier, \'Liberation Mono\', monospace" fill="{INK}">localize_tooth()</text>')
    text(450, 507, 'unknown tool → error', 11.5, False, 'start', MUTE, maxw=152)
    text(428, 536, 'R_tool = valid / attempted', 12.5, True, 'start', INK, maxw=172)
    text(428, 552, 'calls', 12.5, True, 'start', INK)
    # --- R_eff
    text(624, 380, 'R_eff: complexity-scaled, no cost for correction', 13, True, 'start', INK, maxw=350)
    seq = 'EccEcEcccEcEc'
    for i, s in enumerate(seq):
        tx = 624 + i * 26
        if s == 'E':
            add(f'<rect x="{tx}" y="394" width="20" height="22" rx="4" fill="#FDEBD9" stroke="{OR}" stroke-width="1.6" stroke-dasharray="3 2"/>')
        elif s == 'F':
            add(f'<rect x="{tx}" y="394" width="20" height="22" rx="4" fill="#E8F0FB" stroke="#9DB8E0" stroke-width="1.4"/>')
        else:
            add(f'<rect x="{tx}" y="394" width="20" height="22" rx="4" fill="{BL}"/>')
    key(624, 428, BL, False, 'counted call')
    add(f'<rect x="716" y="428" width="14" height="11" fill="#FDEBD9" stroke="{OR}" stroke-width="1.8" stroke-dasharray="3 2"/>')
    text(735, 438, 'exempt: locate_tooth, nudge_crop', 12, False, 'start', INK, maxw=240)
    # plot
    px0, py0, pw, ph = 650, 470, 306, 54
    add(f'<line x1="{px0}" y1="{py0 + ph}" x2="{px0 + pw}" y2="{py0 + ph}" stroke="#6B7280" stroke-width="1.5"/><line x1="{px0}" y1="{py0}" x2="{px0}" y2="{py0 + ph}" stroke="#6B7280" stroke-width="1.5"/>')
    Bg = 6.0; cx_ = lambda c: px0 + c * pw / 24; cy_ = lambda r: py0 + ph - r * ph
    pts = []
    for c in [i * 0.5 for i in range(0, 49)]:
        r = 1.0 if c <= Bg else 1.0 / (1.0 + (c - Bg) / Bg)
        pts.append(f'{cx_(c):.1f},{cy_(r):.1f}')
    add(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{OR}" stroke-width="2.6" stroke-linejoin="round"/>')
    add(f'<line x1="{cx_(Bg):.1f}" y1="{py0 - 4}" x2="{cx_(Bg):.1f}" y2="{py0 + ph}" stroke="#8A8F98" stroke-width="1.3" stroke-dasharray="4 3"/>')
    text(cx_(Bg), py0 - 7, 'budget: 6 calls per located tooth', 12, False, 'middle', INK, maxw=250)
    text(px0 - 6, py0 + 5, '1', 12, False, 'end', MUTE); text(px0 - 6, py0 + ph + 4, '0', 12, False, 'end', MUTE)
    text(px0 + pw / 2, py0 + ph + 17, 'counted calls', 12, False, 'middle', MUTE)
    text(796, 556, 'past the budget: 1 / (1 + overage / budget)', 12, False, 'middle', MUTE, italic=True, maxw=340)
    add('</g>')

    add('</svg>')
    write(path)
