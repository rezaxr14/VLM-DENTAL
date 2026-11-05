"""Figure 9: experimental design, a 2x2 of training (SFT, SFT + GRPO) by tool access (none, tools).

One plane, four conditions, each with a small picture of what the model does. H1 reads down the columns (do tools
help?), H2 across the rows (does RL help?), H3 compares the full system with a supervised specialist. The zero-shot
references are bars on one F1 scale. Results are pending the first hardware run; the protocol is frozen.
"""
from common import *
import json


def build(path):
    out.clear(); warns.clear()
    W, H = 1000, 646
    meta = json.load(open(SC + 'showcase.json'))
    f = meta['flagship']
    pano = Image.open(SC + 'flagship/image.jpg').convert('RGB')
    crop = Image.open(SC + 'flagship/crop_after.jpg').convert('RGB')
    REF = '#22C55E'; OD = '#9A5410'
    add(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Helvetica, Arial, \'Liberation Sans\', sans-serif">')
    add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

    def refbox(x, y, w, h, pad=0.8, sw=1.4, dash='3 2'):
        x -= pad; y -= pad; w += 2 * pad; h += 2 * pad
        add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" stroke="#06210F" stroke-opacity="0.65" stroke-width="{sw + 1.4}" stroke-dasharray="{dash}"/>')
        add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" stroke="{REF}" stroke-width="{sw}" stroke-dasharray="{dash}"/>')

    def pic(x, y, w, im, col='#8A8F98', sw=1.1):
        h = round(w * im.size[1] / im.size[0], 1)
        image(x, y, w, h, im); frame(x, y, w, h, col, sw)
        return h

    def pill(x, y, w, h, label):
        rect(x, y, w, h, ('#E4F3E9', GR), rx=h / 2, sw=1.5)
        text(x + w / 2, y + h / 2 + 4, label, 12, True, 'middle', '#256B3E', maxw=w - 6)

    # ================= the 2x2 plane =================
    rect(16, 16, 718, 446, GRAY, rx=14)
    CW = 260; CX = [118, 452]; RY = [62, 262]
    for ci, (lab, cx) in enumerate((('SFT', CX[0]), ('SFT + GRPO', CX[1]))):
        mid = cx + CW / 2; tw_ = tw(lab, 15, True)
        text(mid + 14, 44, lab, 15, True, 'middle', INK, maxw=150)
        ix = mid + 14 - tw_ / 2 - 26
        if ci == 0:
            for j, wl in enumerate((16, 11, 14)):
                add(f'<line x1="{ix:.1f}" y1="{33 + j * 5}" x2="{ix + wl:.1f}" y2="{33 + j * 5}" stroke="{BL}" stroke-width="2.4" stroke-linecap="round"/>')
        else:
            add(f'<path d="M{ix:.1f},{46} C{ix + 6:.1f},{46} {ix + 8:.1f},{30} {ix + 18:.1f},{30}" fill="none" stroke="{OR}" stroke-width="2.6" stroke-linecap="round"/>')
            add(f'<circle cx="{ix + 18:.1f}" cy="30" r="2.6" fill="{OR}"/>')
    for ri, lab in enumerate(('no tools', 'with tools')):
        text(32, RY[ri] + 104, lab, 14, True, 'start', INK, maxw=84)

    names = [[('SFT only', 'no tools, 1,847 single-turn traces'), ('SFT + GRPO', 'no tools, same recipe, single-turn')],
             [('SFT + tools', '1,847 with-tools traces'), ('VLM-DENTAL', 'SFT + GRPO + tools, live tool rollouts')]]
    gx = f['focus']['gt_box']
    for ri in range(2):
        for ci in range(2):
            x0, y0 = CX[ci], RY[ri]
            full = (ri == 1 and ci == 1)
            if full:
                rect(x0 - 10, y0 - 8, CW + 20, 176, ('#E4F3E9', GR), rx=12, sw=2.2)
            text(x0 + 4, y0 + 18, names[ri][ci][0], 15, True, 'start', '#256B3E' if full else INK, maxw=CW - 8)
            text(x0 + 4, y0 + 36, names[ri][ci][1], 12.5, False, 'start', MUTE, maxw=CW - 6)
            ym = y0 + 108
            if ri == 0:
                h = pic(x0 + 8, ym - 34, 128, pano)
                arrow([(x0 + 144, ym - 34 + h / 2), (x0 + 196, ym - 34 + h / 2)], head=8)
                text(x0 + 170, ym - 34 + h / 2 - 8, 'one pass', 11.5, False, 'middle', MUTE, maxw=60)
                pill(x0 + 200, ym - 34 + h / 2 - 12, 54, 24, 'answer')
            else:
                pw = 96; h = pic(x0 + 8, ym - 26, pw, pano)
                k = pw / 2869
                bx, by, bw, bh = x0 + 8 + gx[0] * k, ym - 26 + gx[1] * k, gx[2] * k, gx[3] * k
                refbox(bx, by, bw, bh)
                cw_ = 44; cx0 = x0 + 118; chh = pic(cx0, ym - 32, cw_, crop)
                for p, q in (((bx + bw, by), (cx0, ym - 32)), ((bx + bw, by + bh), (cx0, ym - 32 + chh))):
                    add(f'<line x1="{p[0]:.1f}" y1="{p[1]:.1f}" x2="{q[0]}" y2="{q[1]:.1f}" stroke="#9DB8E0" stroke-width="1" stroke-dasharray="2 2"/>')
                arrow([(cx0 + cw_ / 2, ym - 34), (cx0 + cw_ / 2, ym - 48), (x0 + 8 + pw / 2, ym - 48), (x0 + 8 + pw / 2, ym - 29)], col=OR, sw=2, dash='4 3', head=7)
                text(x0 + 8 + pw / 2 + 8, ym - 53, 'look again', 11.5, True, 'start', OD, maxw=80)
                arrow([(cx0 + cw_ + 6, ym - 32 + chh / 2), (x0 + 202, ym - 32 + chh / 2)], head=8)
                pill(x0 + 204, ym - 32 + chh / 2 - 12, 52, 24, 'answer')
    # H2 across (RL), H1 down (tools), as questions
    for ri in range(2):
        ym = RY[ri] + 108
        arrow([(CX[0] + CW + 12, ym - 12), (CX[1] - 14, ym - 12)], col=OR, sw=3.2, head=10)
        text((CX[0] + CW + CX[1]) / 2 - 1, ym - 24, 'H2: RL', 12.5, True, 'middle', OD, maxw=60)
        text((CX[0] + CW + CX[1]) / 2 - 1, ym + 8, 'helps?', 12, False, 'middle', OD, maxw=60)
    for ci in range(2):
        xm = CX[ci] + CW / 2
        arrow([(xm, RY[0] + 150), (xm, RY[1] - (12 if ci == 1 else 8))], col=BL, sw=3.2, head=10)
        text(xm + 12, RY[0] + 182, 'H1: tools help?', 12.5, True, 'start', BL, maxw=110)
    add(f'<rect x="32" y="440" width="22" height="12" rx="3" fill="#FFFFFF" stroke="{RD}" stroke-width="1.3" stroke-dasharray="3 2"/>')
    text(62, 451, 'F1 and invented findings of all four: pending the first hardware run', 12.5, False, 'start', RD, maxw=520)

    # ================= zero-shot references =================
    rect(750, 16, 234, 232, GRAY, rx=14)
    text(766, 41, 'Zero-shot references', 14, True, maxw=200)
    text(766, 58, 'exact F1', 12, False, 'start', MUTE, maxw=200)
    for i, (lab, v) in enumerate((('best frontier VLM', 0.222), ('Qwen3.5-9B', 0.112), ('majority floor', 0.038))):
        y = 84 + i * 48
        text(766, y, lab, 12.5, False, 'start', INK, maxw=180)
        bw = v / 0.25 * 150
        add(f'<rect x="766" y="{y + 7}" width="{bw:.1f}" height="14" rx="3" fill="#6B7280"/>')
        text(766 + bw + 8, y + 19, f'{v:.3f}', 13, True, 'start', INK, maxw=50)
    text(766, 232, 'the four conditions: pending', 12.5, True, 'start', RD, maxw=210)

    # ================= H3 =================
    rect(750, 262, 234, 176, ORANGE, rx=14, sw=1.8)
    text(766, 288, 'H3: supervised specialist', 14, True, maxw=206)
    text(766, 308, 'HierarchicalDet on DENTEX', 12.5, False, 'start', MUTE, maxw=206)
    text(766, 334, 'AP 0.372', 13, True, 'start', INK, maxw=206)
    text(766, 352, 'AP50 0.601 · AR 0.653', 13, True, 'start', INK, maxw=206)
    text(766, 392, 'a fair test,', 12.5, True, 'start', OD, maxw=206)
    text(766, 408, 'not a sure win', 12.5, True, 'start', OD, maxw=206)
    arrow([(CX[1] + CW + 14, RY[1] + 80), (748, RY[1] + 80)], col='#6B7280', sw=2.6, head=10)

    # ================= protocol =================
    rect(16, 476, 968, 156, GRAY, rx=14)
    text(32, 501, 'Evaluation protocol, frozen', 14.5, True, maxw=400)
    def head(x, y, s, col=BL): text(x, y, s, 13, True, 'start', col, maxw=200)
    head(32, 528, 'test')
    text(32, 547, 'DENTEX official validation, used once', 12.5, False, 'start', INK, maxw=230)
    BW = 220
    add(f'<rect x="32" y="558" width="{BW * 46 / 50:.1f}" height="22" rx="4" fill="{BL}"/>')
    add(f'<rect x="{32 + BW * 46 / 50:.1f}" y="558" width="{BW * 4 / 50:.1f}" height="22" rx="4" fill="#9DB8E0"/>')
    text(32 + BW * 46 / 100, 574, '46 pathological', 12.5, True, 'middle', '#FFFFFF', maxw=150)
    text(32 + BW * 46 / 50 + BW * 2 / 50, 574, '4', 12.5, True, 'middle', INK, maxw=14)
    text(32, 600, '50 images, 182 target teeth', 12.5, False, 'start', MUTE, maxw=230)
    head(290, 528, 'dev')
    for i, s in enumerate(('about 70 training images', 'one fixed-seed draw', 'every choice is made here')):
        text(290, 547 + i * 17, s, 12.5, False, 'start', INK, maxw=200)
    head(520, 528, 'metrics')
    for i, s in enumerate(('pooled F1, set-matched, tooth-only', 'invented findings on healthy images', 'no-answer rate, ECE')):
        text(520, 547 + i * 17, s, 12.5, False, 'start', INK, maxw=220)
    head(290, 612, 'uncertainty')
    text(380, 612, 'bootstrap over images, paired', 12.5, False, 'start', INK, maxw=180)
    head(776, 528, 'probes', OD)
    text(776, 547, 'on the trained checkpoint,', 12.5, False, 'start', INK, maxw=200)
    text(776, 564, 'nothing retrained:', 12.5, False, 'start', INK, maxw=200)
    for i, s in enumerate(('tool removal', 'grounding perturbation', 'group size K in {1,2,4,8,16}')):
        text(776, 581 + i * 17 - 0, s, 12.5, False, 'start', INK, maxw=200)

    add('</svg>')
    write(path)
