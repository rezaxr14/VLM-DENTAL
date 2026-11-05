"""Figure 11: six verified with-tools episodes (five pathology, one healthy negative control).

One IoU axis runs through the figure: for each pathology episode the real context (expert box, box returned by
locate_tooth, box after the nudge_crop chain), the crops the agent inspected before and after correcting, and a
before -> after IoU pair on the shared axis. The healthy control has no box to correct and returns an empty list.
"""
from common import *
import json


def build(path):
    out.clear(); warns.clear()
    W, H = 1000, 856
    meta = json.load(open(SC + 'showcase.json'))
    REF = '#22C55E'
    TCOL = {'big': '#C94F4F', 'small': '#E3A15A'}
    add(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Helvetica, Arial, \'Liberation Sans\', sans-serif">')
    add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')
    rect(16, 16, 968, H - 32, GRAY, rx=14)

    def refbox(x, y, w, h, pad=0.8, sw=1.5, dash='4 3'):
        x -= pad; y -= pad; w += 2 * pad; h += 2 * pad
        add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" stroke="#06210F" stroke-opacity="0.65" stroke-width="{sw + 1.6}" stroke-dasharray="{dash}"/>')
        add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" stroke="{REF}" stroke-width="{sw}" stroke-dasharray="{dash}"/>')

    def sbox(x, y, w, h, col, sw=2):
        add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" stroke="#0B1220" stroke-opacity="0.45" stroke-width="{sw + 1.3}"/>')
        add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" stroke="{col}" stroke-width="{sw}"/>')

    def fit(im, bw, bh):
        s = min(bw / im.size[0], bh / im.size[1]); return round(im.size[0] * s, 1), round(im.size[1] * s, 1)

    # shared IoU axis
    AX0, AX1 = 662, 962
    ax = lambda v: AX0 + v * (AX1 - AX0)
    text(240, 52, 'in context', 12.5, True, 'start', MUTE, maxw=140)
    text(440, 52, 'returned box', 12.5, True, 'middle', BL, maxw=110)
    text(548, 52, 'after nudges', 12.5, True, 'middle', '#B8620F', maxw=110)
    text(AX0, 52, 'IoU with the expert box', 12.5, True, 'start', MUTE, maxw=220)
    rows = [('three_step', 'a', 'DENTEX 62, FDI 16', 'caries'), ('breadth', 'b', 'DENTEX 58, FDI 25', 'caries'),
            ('deep_caries', 'c', 'DENTEX 291, FDI 36', 'deep caries'), ('tufts', 'd', 'Tufts 975, FDI 36', 'periapical lesion'),
            ('small_tier', 'e', 'DENTEX 84, FDI 16', 'caries')]
    RH = 116; Y0 = 66
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        add(f'<line x1="{ax(v):.1f}" y1="{Y0 + 4}" x2="{ax(v):.1f}" y2="{Y0 + RH * 5 - 6}" stroke="#E3E6EB" stroke-width="1"/>')
        text(ax(v), Y0 + RH * 5 + 8, f'{v:g}', 11.5, False, 'middle', MUTE, maxw=30)
    for i, (key, lab, title, dx) in enumerate(rows):
        e = meta[key]; f = e['focus']; y0 = Y0 + i * RH
        if i: add(f'<line x1="32" y1="{y0 - 3}" x2="968" y2="{y0 - 3}" stroke="#DADDE3" stroke-width="1"/>')
        circ_num(44, y0 + 18, lab, '#4B5563')
        text(62, y0 + 23, title, 13.5, True, 'start', INK, maxw=172)
        tier = f['logged_tier']; nn = len(f['nudges'])
        text(36, y0 + 41, dx, 12.5, False, 'start', MUTE, maxw=196)
        dot(46, y0 + 58, TCOL[tier], 5)
        text(58, y0 + 63, f"{tier} offset, {nn} nudge{'s' if nn > 1 else ''}", 12.5, False, 'start', INK, maxw=170)
        text(36, y0 + 82, f"{e['n_findings']} findings, {e['n_turns']} turns", 12, False, 'start', MUTE, maxw=196)
        text(36, y0 + 98, f"{e['n_calls']} tool calls", 12, False, 'start', MUTE, maxw=196)
        # context with the three boxes
        ctx = Image.open(SC + f'{key}/context.jpg').convert('RGB')
        cw, ch = fit(ctx, 140, 104)
        cx0, cy0 = 240, y0 + 4 + (104 - ch) / 2
        image(cx0, cy0, cw, ch, ctx); frame(cx0, cy0, cw, ch, '#8A8F98', 1.2)
        rx0, ry0, rx1, ry1 = f['context_region']; s = cw / (rx1 - rx0)
        m = lambda b: (cx0 + (b[0] - rx0) * s, cy0 + (b[1] - ry0) * s, b[2] * s, b[3] * s)
        refbox(*m(f['gt_box']), 0.8, 1.6)
        sbox(*m(f['returned_box']), BL, 2)
        sbox(*m(f['nudges'][-1]), OR, 2)
        # crops before / after
        for im_name, xc, col in (('crop_before', 440, BL), ('crop_after', 548, OR)):
            im = Image.open(SC + f'{key}/{im_name}.jpg').convert('RGB')
            w_, h_ = fit(im, 80, 104)
            x_, y_ = xc - w_ / 2, y0 + 4 + (104 - h_) / 2
            image(x_, y_, w_, h_, im); frame(x_, y_, w_, h_, col, 2.2)
        arrow([(483, y0 + 56), (505, y0 + 56)], col=OR, sw=2.4, head=8)
        # IoU pair on the shared axis
        yy = y0 + 56; b, a = f['iou_before'], f['iou_after']
        add(f'<line x1="{ax(b):.1f}" y1="{yy}" x2="{ax(a):.1f}" y2="{yy}" stroke="#6B7280" stroke-width="2.4"/>')
        arrow([(ax(b) + 8, yy), (ax(a) - 9, yy)], col='#6B7280', sw=2.4, head=8)
        add(f'<circle cx="{ax(b):.1f}" cy="{yy}" r="7" fill="{BL}" stroke="#FFFFFF" stroke-width="2"/>')
        add(f'<circle cx="{ax(a):.1f}" cy="{yy}" r="7" fill="{OR}" stroke="#FFFFFF" stroke-width="2"/>')
        text(ax(b), yy - 14, f'{b:.2f}', 13, True, 'middle', BL, maxw=40)
        text(ax(a), yy - 14, f'{a:.2f}', 13, True, 'middle', '#B8620F', maxw=40)
    # healthy negative control
    y0 = Y0 + 5 * RH + 30
    add(f'<line x1="32" y1="{y0 - 12}" x2="968" y2="{y0 - 12}" stroke="#DADDE3" stroke-width="1"/>')
    h = meta['healthy']
    circ_num(44, y0 + 12, 'f', '#4B5563')
    text(62, y0 + 17, 'DENTEX healthy 622', 13.5, True, 'start', INK, maxw=172)
    dot(46, y0 + 38, GR, 5); text(58, y0 + 43, 'negative control', 12.5, False, 'start', INK, maxw=170)
    text(36, y0 + 66, f"{h['n_findings']} findings, {h['n_turns']} turns", 12, False, 'start', MUTE, maxw=196)
    text(36, y0 + 82, f"{h['n_calls']} tool calls", 12, False, 'start', MUTE, maxw=196)
    im = Image.open(SC + 'healthy/image.jpg').convert('RGB')
    w_ = 250; h_ = round(w_ * im.size[1] / im.size[0], 1)
    image(240, y0 + 2, w_, h_, im); frame(240, y0 + 2, w_, h_, '#8A8F98', 1.2)
    text(AX0, y0 + 38, 'findings: [ ]', 20, True, 'start', '#256B3E', maxw=240)
    text(AX0, y0 + 60, 'cleared, no hallucination', 12.5, False, 'start', INK, maxw=290)
    text(AX0, y0 + 77, 'a partially edentulous scan with restorations', 12, False, 'start', MUTE, maxw=300)
    # legend
    ly = H - 36
    add(f'<line x1="32" y1="{ly - 4}" x2="58" y2="{ly - 4}" stroke="{REF}" stroke-width="3" stroke-dasharray="5 3"/>'); text(64, ly, 'expert box', 12.5, False, 'start', INK, maxw=100)
    add(f'<line x1="190" y1="{ly - 4}" x2="216" y2="{ly - 4}" stroke="{BL}" stroke-width="3"/>'); text(222, ly, 'box returned by locate_tooth', 12.5, False, 'start', INK, maxw=200)
    add(f'<line x1="440" y1="{ly - 4}" x2="466" y2="{ly - 4}" stroke="{OR}" stroke-width="3"/>'); text(472, ly, 'box after the nudge_crop chain', 12.5, False, 'start', INK, maxw=220)
    add('</svg>')
    write(path)
