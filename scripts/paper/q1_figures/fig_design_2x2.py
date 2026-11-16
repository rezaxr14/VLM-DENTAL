"""Figure 9: experimental design.

The four conditions form a diamond: tools and RL are two directions, so every comparison is a pair of parallel edges
(blue = H1, tools help?; orange = H2, RL helps?). The harness is drawn as rings around the full system, the probes
as satellites on the trained checkpoint, the supervised specialist (H3) and the zero-shot references on the sides, and
the data splits as the order in which images are used. F1 and invented findings of the four conditions are pending.
"""
from common import *
import json, math


def build(path):
    out.clear(); warns.clear()
    W, H = 1000, 730
    meta = json.load(open(SC + 'showcase.json'))
    pano = Image.open(SC + 'flagship/image.jpg').convert('RGB')
    tooth = Image.open(SC + 'flagship/crop_after.jpg').convert('RGB')
    arch = pano.crop((350, 130, 750, 530))               # the dental arch of the panoramic view
    zoom = tooth.crop((0, 30, 300, 330))                 # a zoomed tooth crop
    BLU, ORN, GRN, PUR = '#3B6BB5', '#D9822B', '#2E8B57', '#7B5EA7'
    add(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Helvetica, Arial, \'Liberation Sans\', sans-serif">')
    add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')
    cid = [0]

    def node(cx, cy, r, im, ring, rw=3.2):
        cid[0] += 1
        add(f'<defs><clipPath id="nc{cid[0]}"><circle cx="{cx}" cy="{cy}" r="{r}"/></clipPath></defs>')
        add(f'<image x="{cx - r}" y="{cy - r}" width="{2 * r}" height="{2 * r}" preserveAspectRatio="none" clip-path="url(#nc{cid[0]})" xlink:href="data:image/jpeg;base64,{b64(im)}"/>')
        add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{ring}" stroke-width="{rw}"/>')

    def edge(a, b, col, badge):
        (ax, ay, ar), (bx, by, br) = a, b
        L = math.hypot(bx - ax, by - ay); ux, uy = (bx - ax) / L, (by - ay) / L
        p0 = (ax + ux * (ar + 8), ay + uy * (ar + 8)); p1 = (bx - ux * (br + 9), by - uy * (br + 9))
        arrow([p0, p1], col=col, sw=3.4, head=11)
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        add(f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="13" fill="{col}" stroke="#FFFFFF" stroke-width="2.4"/>')
        text(round(mx, 1), round(my + 4.5, 1), badge, 11.5, True, 'middle', '#FFFFFF')

    S = (440, 490, 44); T = (270, 350, 44); G = (610, 350, 44); V = (440, 210, 56); B = (150, 490, 32)

    # ---------------- harness rings around the full system (drawn first, behind everything)
    for r_, dash in ((94, None), (80, '6 4'), (66, '2 4')):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        add(f'<circle cx="{V[0]}" cy="{V[1]}" r="{r_}" fill="#F6F2FB" fill-opacity="0.55" stroke="{PUR}" stroke-width="2"{d}/>')
    for r_, ang, lab in ((94, 34, '+ all five skills'), (80, 14, '+ skill index'), (66, -6, 'rules only, a control')):
        px, py = V[0] + r_ * math.cos(math.radians(ang)), V[1] - r_ * math.sin(math.radians(ang))
        add(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.4" fill="{PUR}"/>')
        add(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{px + 12:.1f}" y2="{py:.1f}" stroke="{PUR}" stroke-width="1.3"/>')
        text(round(px + 16, 1), round(py + 4, 1), lab, 12, False, 'start', '#5A3F86', maxw=170)
    text(566, 112, 'harness around the checkpoint', 12.5, True, 'start', PUR, maxw=220)
    text(566, 128, 'dev split only; GRPO is also trained', 11.5, False, 'start', MUTE, italic=True, maxw=250)
    text(566, 142, 'with the harness off and with rules + skills', 11.5, False, 'start', MUTE, italic=True, maxw=250)

    # ---------------- edges of the diamond
    arrow([(B[0] + B[2] + 8, B[1]), (S[0] - S[2] - 9, B[1])], col='#6B7280', sw=3.0, head=10)
    text((B[0] + B[2] + S[0] - S[2]) / 2, B[1] - 12, 'SFT', 12.5, True, 'middle', '#4B5563')
    edge(S, T, BLU, 'H1'); edge(G, V, BLU, 'H1')
    edge(S, G, ORN, 'H2'); edge(T, V, ORN, 'H2')
    text(440, 336, 'blue edges: H1, do tools help?', 13, True, 'middle', BLU, maxw=210)
    text(440, 356, 'orange edges: H2, does RL help?', 13, True, 'middle', '#B8620F', maxw=210)
    text(440, 376, 'every comparison is a pair of parallel edges', 11.5, False, 'middle', MUTE, italic=True, maxw=240)

    # ---------------- nodes
    node(*B[:2], B[2], arch, '#9AA3AE'); node(*S[:2], S[2], arch, '#9AA3AE')
    node(*T[:2], T[2], zoom, GRN); node(*G[:2], G[2], arch, '#9AA3AE'); node(*V[:2], V[2], zoom, GRN, 4)
    def label(c, name, sub, bold_col=INK, dy=18):
        text(c[0], c[1] + c[2] + dy, name, 14, True, 'middle', bold_col, maxw=230)
        text(c[0], c[1] + c[2] + dy + 16, sub, 12, False, 'middle', MUTE, maxw=230)
    label(S, 'SFT only', 'no tools, 1,847 single-turn traces')
    label(T, 'SFT + tools', '1,847 with-tools traces')
    label(G, 'SFT + GRPO', 'no tools, same recipe')
    text(440, 76, 'VLM-DENTAL', 16, True, 'middle', '#256B3E'); text(440, 94, 'SFT + GRPO + tools, live tool rollouts', 12, False, 'middle', MUTE, maxw=260)
    text(B[0], B[1] + B[2] + 18, 'Qwen3.5-9B, zero-shot', 13, True, 'middle', INK, maxw=190)
    text(B[0], B[1] + B[2] + 34, 'exact F1 0.112', 12, False, 'middle', MUTE)
    text(B[0], B[1] + B[2] + 52, 'best frontier VLM 0.222', 11.5, False, 'middle', MUTE, maxw=190)
    text(B[0], B[1] + B[2] + 67, 'majority floor 0.038', 11.5, False, 'middle', MUTE, maxw=190)
    text(24, 40, 'F1 and invented findings of the four conditions: pending the first hardware run', 12.5, True, 'start', RD, maxw=520)

    # ---------------- probes on the trained checkpoint
    text(342, 128, 'probes on the trained checkpoint', 12.5, True, 'end', '#9A5410', maxw=210)
    text(342, 143, 'nothing retrained', 11.5, False, 'end', MUTE, italic=True, maxw=140)
    for ang, lab in ((152, 'tool removal'), (180, 'grounding perturbation'), (208, 'group size K in {1,2,4,8,16}')):
        px, py = V[0] + 112 * math.cos(math.radians(ang)), V[1] - 112 * math.sin(math.radians(ang))
        add(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{V[0] - 96 * abs(math.cos(math.radians(ang))):.1f}" y2="{V[1] - 96 * math.sin(math.radians(ang)):.1f}" stroke="#C9B28F" stroke-width="1.3"/>')
        add(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="#FFFFFF" stroke="#9A5410" stroke-width="2"/>')
        text(round(px - 11, 1), round(py + 4, 1), lab, 12, False, 'end', INK, maxw=210)

    # ---------------- H3: the supervised specialist
    HX, HY = 880, 236
    arrow([(V[0] + 98, HY), (HX - 44, HY)], col='#6B7280', sw=2.6, dash='6 4', head=10)
    text(700, HY - 9, 'H3', 13, True, 'middle', '#4B5563')
    add(f'<circle cx="{HX}" cy="{HY}" r="34" fill="#FDEBD9" stroke="{ORN}" stroke-width="2.6"/>')
    text(HX, HY + 5, 'AP', 15, True, 'middle', '#8A4A08')
    text(HX, HY + 54, 'supervised specialist', 13, True, 'middle', INK, maxw=180); text(HX, HY + 70, 'HierarchicalDet on DENTEX', 12, False, 'middle', MUTE, maxw=180)
    text(HX, HY + 88, 'AP 0.372 · AP50 0.601 · AR 0.653', 12, True, 'middle', INK, maxw=190)
    text(HX, HY + 106, 'a fair test, not a sure win', 12, True, 'middle', '#9A5410', maxw=190)

    # ---------------- metrics
    text(700, 440, 'metrics', 13, True, 'start', BLU)
    for i, s_ in enumerate(('pooled F1, set-matched, tooth-only', 'invented findings on healthy images', 'no-answer rate and ECE', 'bootstrap over images, paired')):
        text(700, 459 + i * 17, s_, 12.5, False, 'start', INK, maxw=270)

    # ---------------- data: the order in which images are used
    piles = [(86, 4, '1,847 images', 'trains SFT and GRPO'), (296, 2, '~70 images', 'dev: carved from train, every choice'),
             (506, 2, '50 images', 'official validation split'), (716, 3, '250 images', 'official test split, used once')]
    text(24, 608, 'data', 13, True, 'start', BLU)
    for cx, n, a1, a2 in piles:
        for k in range(n):
            x = cx - 32 + k * 7; y = 624 + (n - 1 - k) * 0 - k * 4 + 12
            image(x, y, 64, 34, pano); frame(x, y, 64, 34, '#FFFFFF', 1.6); frame(x - 0.5, y - 0.5, 65, 35, '#9AA3AE', 0.8)
        text(cx, 686, a1, 13, True, 'middle', INK, maxw=190); text(cx, 702, a2, 11.5, False, 'middle', MUTE, maxw=200)
    for (x0, _, _, _), (x1, _, _, _) in zip(piles, piles[1:]):
        arrow([(x0 + 56, 656), (x1 - 56, 656)], col='#9AA3AE', sw=2.2, head=8)
    text(950, 656, 'order', 11.5, False, 'end', MUTE, italic=True) if False else None
    add('</svg>')
    write(path)
