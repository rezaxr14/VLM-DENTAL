"""Figure 10: correction behaviour by perturbation tier (Table 5 numbers).

(a) P(corrective nudge_crop) on the ground-truth-anchored subset, with Wilson 95% intervals, and the rate over all
locate returns (diamonds); the real episode behind each tier sits under its bar.
(b) Paired mean IoU with the expert box before and after correction, over the corrected locates (95% interval of the
mean); one real recovery (DENTEX image 127, FDI 48) underneath.
Numbers come from paper/assets_summary.json (nudge_by_tier), the data behind Table 5.
"""
from common import *
import json, math


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def build(path):
    out.clear(); warns.clear()
    W, H = 1000, 560
    S = json.load(open(str(ROOT) + '/paper/assets_summary.json'))['nudge_by_tier']
    G, ALL = S['gt_anchored'], S['overall']
    meta = json.load(open(SC + 'showcase.json'))
    REF = '#22C55E'
    TIER = ['none', 'small', 'big']
    TCOL = {'none': '#9AA3AE', 'small': '#E3A15A', 'big': '#C94F4F'}
    SHARE = {'none': '45%', 'small': '25%', 'big': '30%'}
    add(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Helvetica, Arial, \'Liberation Sans\', sans-serif">')
    add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

    def refbox(x, y, w, h, pad=0.8, sw=1.5, dash='4 3'):
        x -= pad; y -= pad; w += 2 * pad; h += 2 * pad
        add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" stroke="#06210F" stroke-opacity="0.65" stroke-width="{sw + 1.6}" stroke-dasharray="{dash}"/>')
        add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" stroke="{REF}" stroke-width="{sw}" stroke-dasharray="{dash}"/>')

    def sbox(x, y, w, h, col, sw=2):
        add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" stroke="#0B1220" stroke-opacity="0.45" stroke-width="{sw + 1.3}"/>')
        add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" stroke="{col}" stroke-width="{sw}"/>')

    # ---------------- real episodes behind each tier ----------------
    def episode(key, clean=False, w=104):
        v = meta[key]; f = v['focus']
        ctx = Image.open(SC + f'{key}/context.jpg').convert('RGB')
        rx0, ry0, rx1, ry1 = f['context_region']
        s = ctx.size[0] / (rx1 - rx0); k = w / ctx.size[0]
        h = round(w * ctx.size[1] / ctx.size[0], 1)
        def m(b): return ((b[0] - rx0) * s * k, (b[1] - ry0) * s * k, b[2] * s * k, b[3] * s * k)
        return ctx, h, m(f['gt_box']), m(f['returned_box']), f

    # ================= (a) =================
    rect(16, 16, 484, 530, GRAY, rx=14)
    text(32, 41, 'P(corrective nudge) by perturbation tier', 14.5, True, maxw=450)
    text(32, 59, 'expert box as reference', 12, False, 'start', MUTE, maxw=300)
    PX0, PX1, PY0, PY1 = 78, 484, 90, 318
    ymax = 50.0
    yv = lambda p: PY1 - p / ymax * (PY1 - PY0)
    for gv in (0, 10, 20, 30, 40, 50):
        add(f'<line x1="{PX0}" y1="{yv(gv):.1f}" x2="{PX1}" y2="{yv(gv):.1f}" stroke="#DADDE3" stroke-width="1"/>')
        text(PX0 - 8, yv(gv) + 4, f'{gv}%', 11.5, False, 'end', MUTE, maxw=40)
    cx = [PX0 + 66 + i * 136 for i in range(3)]
    BW = 84
    for i, t in enumerate(TIER):
        g = G[t]; n = g['n']; k_ = round(g['corrected_rate'] * n)
        p = g['corrected_rate'] * 100; lo, hi = [v * 100 for v in wilson(k_, n)]
        add(f'<rect x="{cx[i] - BW / 2}" y="{yv(p):.1f}" width="{BW}" height="{PY1 - yv(p):.1f}" rx="3" fill="{TCOL[t]}"/>')
        add(f'<line x1="{cx[i]}" y1="{yv(lo):.1f}" x2="{cx[i]}" y2="{yv(hi):.1f}" stroke="#1F2937" stroke-width="1.8"/>')
        for yy in (lo, hi): add(f'<line x1="{cx[i] - 6}" y1="{yv(yy):.1f}" x2="{cx[i] + 6}" y2="{yv(yy):.1f}" stroke="#1F2937" stroke-width="1.8"/>')
        text(cx[i], yv(hi) - 22, f'{p:.1f}%', 14, True, 'middle', INK, maxw=70)
        text(cx[i], yv(hi) - 7, f'n = {n:,}', 11.5, False, 'middle', MUTE, maxw=80)
        pa = ALL[t]['corrected_rate'] * 100
        add(f'<path d="M{cx[i]},{yv(pa) - 8:.1f} l8,8 l-8,8 l-8,-8 z" fill="#FFFFFF" stroke="#1F2937" stroke-width="1.8"/>')
    add(f'<line x1="{PX0}" y1="{PY1}" x2="{PX1}" y2="{PY1}" stroke="#4B5563" stroke-width="1.6"/>')
    add(f'<path d="M{PX0 + 18},{PY0 + 8} l7,7 l-7,7 l-7,-7 z" fill="#FFFFFF" stroke="#1F2937" stroke-width="1.8"/>')
    text(PX0 + 32, PY0 + 20, 'all locate returns', 12, False, 'start', INK, maxw=140)
    # tier names and the real episode under each bar
    for i, t in enumerate(TIER):
        text(cx[i], PY1 + 22, t, 14, True, 'middle', INK, maxw=80)
        text(cx[i], PY1 + 39, SHARE[t] + ' of shown boxes', 11.5, False, 'middle', MUTE, maxw=124)
        key = 'small_tier' if t == 'small' else 'flagship'
        c0 = Image.open(SC + f'{key}/context.jpg'); w_ = round(min(104, 128 * c0.size[0] / c0.size[1]), 1)
        ctx, h, gt, rt, f = episode(key, w=w_)
        x = cx[i] - w_ / 2; y = PY1 + 50
        image(x, y, w_, h, ctx); frame(x, y, w_, h, '#8A8F98', 1.2)
        if t == 'none':
            sbox(x + gt[0], y + gt[1], gt[2], gt[3], BL, 2.2)
        else:
            sbox(x + rt[0], y + rt[1], rt[2], rt[3], BL, 2.2)
        refbox(x + gt[0], y + gt[1], gt[2], gt[3])
    # legend for boxes
    add(f'<line x1="32" y1="526" x2="56" y2="526" stroke="{REF}" stroke-width="3" stroke-dasharray="5 3"/>'); text(62, 530, 'expert box', 12, False, 'start', INK, maxw=90)
    add(f'<line x1="162" y1="526" x2="186" y2="526" stroke="{BL}" stroke-width="3"/>'); text(192, 530, 'box shown to the teacher', 12, False, 'start', INK, maxw=200)

    # ================= (b) =================
    rect(516, 16, 468, 530, GRAY, rx=14)
    text(532, 41, 'IoU with the expert box, before and after', 14.5, True, maxw=440)
    text(532, 59, 'corrected locates, same episodes', 12, False, 'start', MUTE, maxw=300)
    QX0, QX1 = 574, 970
    yi = lambda v: PY1 - v * (PY1 - PY0)
    for gv in (0, 0.25, 0.5, 0.75, 1.0):
        add(f'<line x1="{QX0}" y1="{yi(gv):.1f}" x2="{QX1}" y2="{yi(gv):.1f}" stroke="#DADDE3" stroke-width="1"/>')
        text(QX0 - 8, yi(gv) + 4, f'{gv:.2f}', 11.5, False, 'end', MUTE, maxw=40)
    cx2 = [QX0 + 66 + i * 132 for i in range(3)]
    TAB = {'none': ('1.00', '0.27', '-0.73'), 'small': ('0.45', '0.45', '+0.00'), 'big': ('0.09', '0.24', '+0.16')}
    GREEN = '#2E9B5B'
    for i, t in enumerate(TIER):
        g = G[t]; nc = g['n_corrected']
        b, a = g['iou_before_corrected_mean'], g['iou_after_mean']
        cb = 1.96 * g['iou_before_corrected_sd'] / math.sqrt(nc); ca = 1.96 * g['iou_after_sd'] / math.sqrt(nc)
        xb, xa = cx2[i] - 22, cx2[i] + 22
        add(f'<line x1="{xb}" y1="{yi(b):.1f}" x2="{xa}" y2="{yi(a):.1f}" stroke="#6B7280" stroke-width="2.2"/>')
        for x_, v, c_, col in ((xb, b, cb, '#6B7280'), (xa, a, ca, GREEN)):
            add(f'<line x1="{x_}" y1="{yi(min(1, v + c_)):.1f}" x2="{x_}" y2="{yi(max(0, v - c_)):.1f}" stroke="#1F2937" stroke-width="1.6"/>')
            for e in (v + c_, v - c_): add(f'<line x1="{x_ - 4}" y1="{yi(min(1, max(0, e))):.1f}" x2="{x_ + 4}" y2="{yi(min(1, max(0, e))):.1f}" stroke="#1F2937" stroke-width="1.6"/>')
            add(f'<circle cx="{x_}" cy="{yi(v):.1f}" r="7" fill="{col}" stroke="#FFFFFF" stroke-width="2"/>')
        text(xb - 12, yi(b) + 4, TAB[t][0], 12.5, True, 'end', '#4B5563', maxw=40)
        text(xa + 12, yi(a) + 4, TAB[t][1], 12.5, True, 'start', '#1F6E41', maxw=40)
        dcol = '#1F6E41' if TAB[t][2].startswith('+0.1') else ('#9B2C2C' if TAB[t][2].startswith('-') else '#4B5563')
        text(cx2[i], PY1 + 22, t, 14, True, 'middle', INK, maxw=80)
        text(cx2[i], PY1 + 39, f'change {TAB[t][2]}', 12.5, True, 'middle', dcol, maxw=110)
    dot(QX0 + 8, PY0 - 12, '#6B7280', 5); text(QX0 + 18, PY0 - 8, 'before', 12, False, 'start', INK, maxw=70)
    dot(QX0 + 94, PY0 - 12, GREEN, 5); text(QX0 + 104, PY0 - 8, 'after the nudges', 12, False, 'start', INK, maxw=130)
    # one real recovery
    f = meta['flagship']['focus']
    cb_ = Image.open(SC + 'flagship/crop_before.jpg').convert('RGB'); ca_ = Image.open(SC + 'flagship/crop_after.jpg').convert('RGB')
    EY = PY1 + 52
    text(532, EY + 12, 'one recovery', 13.5, True, 'start', INK, maxw=140)
    text(532, EY + 29, 'DENTEX 127, FDI 48', 12, False, 'start', MUTE, maxw=140)
    hb = 96
    wb = round(hb * cb_.size[0] / cb_.size[1], 1); wa = round(hb * ca_.size[0] / ca_.size[1], 1)
    x1 = 700; image(x1, EY, wb, hb, cb_); frame(x1, EY, wb, hb, BL, 2.2)
    x2 = x1 + wb + 56; image(x2, EY, wa, hb, ca_); frame(x2, EY, wa, hb, OR, 2.2)
    arrow([(x1 + wb + 4, EY + hb / 2), (x2 - 4, EY + hb / 2)], col=OR, sw=2.4, head=9)
    text(x1 + wb / 2, EY + hb + 17, f"IoU {f['iou_before']:.3f}", 12.5, True, 'middle', BL, maxw=90)
    text(x2 + wa / 2, EY + hb + 17, f"IoU {f['iou_after']:.3f}", 12.5, True, 'middle', '#B8620F', maxw=90)

    add('</svg>')
    write(path)
