"""Results figure: exact F1 of every condition on the validation set against the zero-shot references, the official
250-image test split (conditions with and without the harness), the external MMOral-OPG-Bench score next to the published
OralGPT-Omni numbers, and the per-class F1. Numbers are read from paper/figures/assets/results.json; a null entry is drawn
as a dashed 'pending' placeholder, so the figure is final in layout and only needs the numbers.
"""
from common import *
import json


def build(path):
    out.clear(); warns.clear()
    W, H = 1000, 616
    R = json.load(open(str(ROOT) + '/paper/figures/assets/results.json'))
    BLD, ORD = '#3B6BB5', '#D9822B'
    add(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Helvetica, Arial, \'Liberation Sans\', sans-serif">')
    add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

    def pending(x, y, w, h):
        add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="#FFFFFF" stroke="{RD}" stroke-width="1.3" stroke-dasharray="4 3"/>')
        text(x + w / 2, y + h / 2 + 4, 'pending', 11.5, True, 'middle', RD, maxw=w - 4)

    def dots(x_of, rows, label_x, y0, pitch, label_w, bold_first=False, colors=None, fmt='{:.3f}'):
        for i, (nm, d) in enumerate(rows):
            y = y0 + i * pitch; v = d.get('value')
            text(label_x, y + 4, nm, 12.5, bold_first and i == 0, 'start', INK, maxw=label_w)
            if v is None: pending(x_of(0.03 if x_of(1) < 700 else 4) if False else x_of.pend, y - 9, 130 if x_of.pend < 600 else 100, 18)
            else:
                lo, hi = d.get('ci_low'), d.get('ci_high')
                if lo is not None: add(f'<line x1="{x_of(lo):.1f}" y1="{y}" x2="{x_of(hi):.1f}" y2="{y}" stroke="#1F2937" stroke-width="1.8"/>')
                add(f'<circle cx="{x_of(v):.1f}" cy="{y}" r="6" fill="{(colors or {}).get(nm, "#6B7280")}" stroke="#FFFFFF" stroke-width="1.6"/>')
                text(x_of(v) + 12, y + 4, fmt.format(v), 12, True, 'start', INK, maxw=48)

    # ---------------------------------------------------------- left: exact F1 on the validation set
    rect(16, 16, 590, 584, GRAY, rx=14)
    text(32, 41, 'Exact F1 on the DENTEX validation set', 14.5, True, maxw=420)
    text(32, 59, 'our conditions, the untrained base, and the zero-shot references', 12, False, 'start', MUTE, maxw=440)
    AX0, AX1 = 252, 580
    class A:
        pend = AX0 + 14
        def __call__(self, v): return AX0 + v / 0.5 * (AX1 - AX0)
    ax = A()
    f1 = R['f1']; names = list(f1.keys())
    g0 = ['VLM-DENTAL', 'SFT + tools', 'SFT + GRPO']; g1 = ['Qwen3.5-9B, zero-shot', 'Qwen3.5-9B + tools, untrained']
    g2 = [n for n in names if n not in g0 + g1]
    Y0, P = 94, 34
    ypos = {}
    for gi, g in enumerate((g0, g1, g2)):
        for k, nm in enumerate(g): ypos[nm] = Y0 + (k + (0 if gi == 0 else len(g0) + 0.7 if gi == 1 else len(g0) + len(g1) + 1.4)) * P
    ybot = max(ypos.values()) + 24
    for v in (0, 0.1, 0.2, 0.3, 0.4, 0.5):
        add(f'<line x1="{ax(v):.1f}" y1="{Y0 - 16}" x2="{ax(v):.1f}" y2="{ybot}" stroke="#DADDE3" stroke-width="1"/>')
        text(ax(v), ybot + 20, f'{v:g}', 11.5, False, 'middle', MUTE, maxw=30)
    for gi, ysep in ((1, Y0 + len(g0) * P - 12), (2, Y0 + (len(g0) + len(g1)) * P + 0.7 * P - 2 + 0)):
        add(f'<line x1="32" y1="{ysep:.1f}" x2="590" y2="{ysep:.1f}" stroke="#C9CDD4" stroke-width="1" stroke-dasharray="3 3"/>')
    for nm in names:
        y = ypos[nm]; d = f1[nm]; v = d.get('value')
        text(32, y + 4, nm, 12.5, nm == 'VLM-DENTAL', 'start', INK, maxw=206)
        if v is None: pending(ax.pend, y - 9, 130, 18)
        else:
            col = ORD if nm in g0 else (BLD if nm in g1 else '#6B7280')
            add(f'<circle cx="{ax(v):.1f}" cy="{y}" r="6" fill="{col}" stroke="#FFFFFF" stroke-width="1.6"/>')
            text(ax(v) + 12, y + 4, f'{v:.3f}', 12, True, 'start', INK, maxw=48)

    # ---------------------------------------------------------- right top: official test split and external benchmark
    rect(622, 16, 362, 390, GRAY, rx=14)
    text(638, 41, 'Official DENTEX test split, 250 images', 14.5, True, maxw=340)
    text(638, 59, 'exact F1, with and without our harness', 12, False, 'start', MUTE, maxw=340)
    TX0, TX1 = 842, 968
    class T:
        pend = TX0 + 4
        def __call__(self, v): return TX0 + v / 0.5 * (TX1 - TX0)
    tx = T()
    rows = list(R['test250'].items())[1:]
    for v in (0, 0.25, 0.5):
        add(f'<line x1="{tx(v):.1f}" y1="76" x2="{tx(v):.1f}" y2="{76 + len(rows) * 25 + 4}" stroke="#DADDE3" stroke-width="1"/>')
        text(tx(v), 76 + len(rows) * 25 + 20, f'{v:g}', 11.5, False, 'middle', MUTE, maxw=30)
    for i, (nm, d) in enumerate(rows):
        y = 92 + i * 25; v = d.get('value')
        text(638, y + 4, nm, 12.5, i == 0, 'start', INK, maxw=196)
        if v is None: pending(tx.pend, y - 9, 96, 18)
        else:
            add(f'<circle cx="{tx(v):.1f}" cy="{y}" r="6" fill="{ORD if i == 0 else BLD}" stroke="#FFFFFF" stroke-width="1.6"/>')
            text(tx(v) + 12, y + 4, f'{v:.3f}', 12, True, 'start', INK, maxw=48)
    add('<line x1="638" y1="260" x2="968" y2="260" stroke="#C9CDD4" stroke-width="1" stroke-dasharray="3 3"/>')
    text(638, 284, 'External benchmark: MMOral-OPG-Bench', 14.5, True, maxw=340)
    text(638, 302, 'overall score; published rows use other judges', 12, False, 'start', MUTE, maxw=340)
    MX0, MX1 = 842, 968
    mx = lambda v: MX0 + v / 100 * (MX1 - MX0)
    for v in (0, 50, 100):
        add(f'<line x1="{mx(v):.1f}" y1="318" x2="{mx(v):.1f}" y2="376" stroke="#DADDE3" stroke-width="1"/>')
        text(mx(v), 394, f'{v}', 11.5, False, 'middle', MUTE, maxw=30)
    ours = R['mmoral']['scored_by_us']
    for i, (nm, d) in enumerate(ours.items()):
        y = 332 + i * 22; v = d.get('value')
        text(638, y + 4, nm, 12.5, True, 'start', INK, maxw=196)
        if v is None: pending(mx(3), y - 9, 96, 18)
        else:
            add(f'<circle cx="{mx(v):.1f}" cy="{y}" r="6" fill="{ORD}" stroke="#FFFFFF" stroke-width="1.6"/>'); text(mx(v) + 12, y + 4, f'{v:.1f}', 12, True, 'start', INK, maxw=40)
    for i, (nm, v) in enumerate(R['mmoral']['published'].items()):
        y = 358 + i * 22
        text(638, y + 4, nm, 12.5, False, 'start', INK, maxw=196)
        add(f'<circle cx="{mx(v):.1f}" cy="{y}" r="5" fill="#9AA3AE" stroke="#FFFFFF" stroke-width="1.4"/>')
        text(mx(v) - 10, y + 4, f'{v:.2f}', 11.5, True, 'end', INK, maxw=40)

    # ---------------------------------------------------------- right bottom: per class
    rect(622, 418, 362, 182, GRAY, rx=14)
    text(638, 443, 'Exact F1 by diagnosis class', 14.5, True, maxw=300)
    dot(644, 462, '#6B7280', 5); text(654, 466, 'best zero-shot', 12, False, 'start', INK, maxw=110)
    dot(764, 462, ORD, 5); text(774, 466, 'VLM-DENTAL', 12, False, 'start', INK, maxw=110)
    text(968, 466, 'dashed: pending', 11.5, True, 'end', RD, maxw=110)
    CX0, CX1 = 790, 966
    cx = lambda v: CX0 + v / 0.5 * (CX1 - CX0)
    for i, (nm, d) in enumerate(R['per_class'].items()):
        y = 492 + i * 26
        text(638, y + 6, nm, 12.5, False, 'start', INK, maxw=140)
        for k, (key, col) in enumerate((('zero_shot_best', '#6B7280'), ('vlm_dental', ORD))):
            v = d.get(key); yy = y - 2 + k * 12
            if v is None: add(f'<rect x="{CX0}" y="{yy - 4}" width="120" height="8" rx="3" fill="#FFFFFF" stroke="{col}" stroke-width="1.3" stroke-dasharray="4 3"/>')
            else: add(f'<rect x="{CX0}" y="{yy - 4}" width="{cx(v) - CX0:.1f}" height="8" rx="3" fill="{col}"/>')
    add('</svg>')
    write(path)
