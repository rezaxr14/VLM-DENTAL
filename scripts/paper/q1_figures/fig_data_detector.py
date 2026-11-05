"""Figure 6: data sources, grounding pool, YOLOv8m detector, its validation, candidates, corpus cohorts.

One left-to-right data path (two real sources -> grounding pool -> trained detector -> locate_tooth box and crop),
with the detector's cross-validation and held-out numbers on the right and the unused candidate datasets and the
corpus cohorts underneath. Real radiographs with real expert boxes (DENTEX image 127, Tufts image 975).
"""
from common import *
import json, math


def build(path):
    out.clear(); warns.clear()
    W, H = 1000, 756
    meta = json.load(open(SC + 'showcase.json'))
    dx_img = Image.open(SC + 'flagship/image.jpg').convert('RGB')
    tf_img = Image.open(SC + 'tufts/image.jpg').convert('RGB')
    ctx = Image.open(SC + 'flagship/context.jpg').convert('RGB')
    crop = Image.open(SC + 'flagship/crop_after.jpg').convert('RGB')
    REF = '#22C55E'
    BLD, ORD = '#3B6BB5', '#D9822B'
    TR = '#1F5BFF'
    SEG = {'dp': '#3B6BB5', 'dh': '#9DB8E0', 'to': '#D9822B', 'th': '#F2B97F', 'ta': '#A85A12'}

    add(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Helvetica, Arial, \'Liberation Sans\', sans-serif">')
    add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

    def refbox(x, y, w, h, pad=1.2, sw=1.7, dash='4 3'):
        x -= pad; y -= pad; w += 2 * pad; h += 2 * pad
        add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" stroke="#06210F" stroke-opacity="0.65" stroke-width="{sw + 1.8}" stroke-dasharray="{dash}"/>')
        add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" stroke="{REF}" stroke-width="{sw}" stroke-dasharray="{dash}"/>')

    def shade(c, f):
        c = c.lstrip('#'); r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
        t = 255 if f > 0 else 0; f = abs(f)
        return '#%02X%02X%02X' % tuple(int(v + (t - v) * f) for v in (r, g, b))

    def block3d(x, y, w, h, d, col):
        k = 0.8
        add(f'<polygon points="{x},{y} {x + d},{y - d * k:.1f} {x + w + d},{y - d * k:.1f} {x + w},{y}" fill="{shade(col, 0.4)}" stroke="#FFFFFF" stroke-width="0.8"/>')
        add(f'<polygon points="{x + w},{y} {x + w + d},{y - d * k:.1f} {x + w + d},{y + h - d * k:.1f} {x + w},{y + h}" fill="{shade(col, -0.2)}" stroke="#FFFFFF" stroke-width="0.8"/>')
        add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{col}" stroke="#FFFFFF" stroke-width="0.8"/>')

    def arc(cx, cy, R, a0, a1, col, sw):
        p = lambda a: (cx + R * math.cos(math.radians(a)), cy + R * math.sin(math.radians(a)))
        x0, y0 = p(a0); x1, y1 = p(a1)
        large = 1 if (a1 - a0) % 360 > 180 else 0
        add(f'<path d="M{x0:.2f},{y0:.2f} A{R},{R} 0 {large} 1 {x1:.2f},{y1:.2f}" fill="none" stroke="{col}" stroke-width="{sw}"/>')

    # ============================ sources ============================
    rect(16, 16, 310, 504, GRAY, rx=14)
    text(32, 41, 'Two sources in use', 15, True, maxw=280)

    def source(y0, im, gt, nat_w, name, count, l1, l2, status, scol):
        cylinder(41, y0 + 1, 18, 20, BLD)
        text(58, y0 + 6, name, 14, True, maxw=150)
        text(310, y0 + 6, count, 13, False, 'end', BLD, maxw=110)
        iy = y0 + 18; iw = 262; ih = iw * im.size[1] / im.size[0]
        image(32, iy, iw, round(ih, 1), im); frame(32, iy, iw, round(ih, 1), '#8A8F98', 1.2)
        k = iw / nat_w
        for g in gt:
            b = g['bbox']; refbox(32 + b[0] * k, iy + b[1] * k, b[2] * k, b[3] * k)
        text(32, iy + ih + 20, l1, 12.5, False, 'start', MUTE, maxw=282)
        text(32, iy + ih + 36, l2, 12.5, False, 'start', MUTE, maxw=282)
        dot(37, iy + ih + 52, scol, 4.5); text(47, iy + ih + 56, status, 12.5, False, 'start', INK, maxw=262)
        return iy, ih

    f = meta['flagship']; t = meta['tufts']
    iy1, ih1 = source(64, dx_img, f['ground_truth'], f['native_w'], 'DENTEX', '2,332 images',
                      'quadrant, enumeration, diagnosis; 4 classes', 'only panoramic set with position + all 4 classes',
                      'loader complete', GR)
    iy2, ih2 = source(300, tf_img, t['ground_truth'], t['native_w'], 'Tufts Dental Database', '1,000 images',
                      'per-tooth boxes, expert pathology layer', 'periapical maps to DENTEX; no caries layer',
                      'loader partial, 1 open question', OR)
    c1 = iy1 + ih1 / 2; c2 = iy2 + ih2 / 2

    # ============================ grounding pool (vertical stacked bar) ============================
    text(340, 41, 'Grounding pool and detector', 15, True, maxw=300)
    BX, BW, BY, BH = 340, 84, 96, 340
    h1 = BH * 1339 / 2339
    rect(BX, BY, BW, h1, BLUE, rx=6, sw=1.6)
    rect(BX, BY + h1, BW, BH - h1, ORANGE, rx=6, sw=1.6)
    for yy, nm, im_, bx_ in ((BY + h1 / 2, 'DENTEX', '1,339 images', '21,624 boxes'), (BY + h1 + (BH - h1) / 2, 'Tufts', '1,000 images', '25,184 boxes')):
        text(BX + BW / 2, yy - 8, nm, 12.5, True, 'middle', maxw=BW - 6)
        text(BX + BW / 2, yy + 8, im_, 12, False, 'middle', MUTE, maxw=BW - 4)
        text(BX + BW / 2, yy + 23, bx_, 12, False, 'middle', MUTE, maxw=BW - 4)
    text(BX + BW / 2, BY + BH + 22, '2,339 images', 13, True, 'middle', maxw=110)
    text(BX + BW / 2, BY + BH + 39, '46,808 boxes', 13, True, 'middle', maxw=110)
    arrow([(298, c1), (BX, c1)], head=8)
    arrow([(298, c2), (BX, c2)], head=8)

    # ============================ detector (3D block, trained) ============================
    DX, DY, DW, DH, DD = 448, 216, 56, 100, 18
    arrow([(BX + BW, BY + BH / 2), (DX, BY + BH / 2)], head=8)
    add(f'<rect x="{DX - 5}" y="{DY - DD * 0.8 - 5}" width="{DW + DD + 10}" height="{DH + DD * 0.8 + 10}" rx="9" fill="{TR}" fill-opacity="0.16"/>')
    block3d(DX, DY, DW, DH, DD, TR)
    flame(DX + DW / 2 + 4, DY - 22, 1.1)
    text(DX + (DW + DD) / 2, DY + DH + 22, 'YOLOv8m', 14, True, 'middle', maxw=100)
    text(DX + (DW + DD) / 2, DY + DH + 39, '32 FDI classes', 12.5, False, 'middle', MUTE, maxw=110)
    text(DX + (DW + DD) / 2, DY + DH + 55, '5-fold CV', 12.5, False, 'middle', MUTE, maxw=110)

    # ============================ locate_tooth output: real box and crop ============================
    OX, OY, OW = 556, 150, 84
    OH = OW * ctx.size[1] / ctx.size[0]
    text(OX + OW / 2, OY - 12, 'locate_tooth', 13.5, True, 'middle', INK, maxw=100)
    image(OX, OY, OW, round(OH, 1), ctx); frame(OX, OY, OW, round(OH, 1), '#8A8F98', 1.2)
    S = 0.6275; K = OW / 439
    gb = f['focus']['gt_box']
    bx_, by_, bw_, bh_ = (gb[0] - 503.3) * S * K, (gb[1] - 462.3) * S * K, gb[2] * S * K, gb[3] * S * K
    refbox(OX + bx_, OY + by_, bw_, bh_, 1.2, 1.7)
    CX, CY, CW = 560, 292, 76
    CH = CW * crop.size[1] / crop.size[0]
    image(CX, CY, CW, round(CH, 1), crop); frame(CX, CY, CW, round(CH, 1), '#8A8F98', 1.2)
    for (ax, ay), (cx_, cy_) in (((OX + bx_, OY + by_ + bh_), (CX, CY)), ((OX + bx_ + bw_, OY + by_ + bh_), (CX + CW, CY))):
        add(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{cx_}" y2="{cy_}" stroke="#9DB8E0" stroke-width="1.2" stroke-dasharray="3 3"/>')
    text(OX + OW / 2 + 2, CY + CH + 17, 'FDI 48: box, crop', 12, False, 'middle', MUTE, maxw=110)
    arrow([(DX + DW + DD + 4, DY + DH * 0.45), (538, DY + DH * 0.45), (538, OY + OH / 2), (OX - 2, OY + OH / 2)], head=8)

    text(448, 424, 'annotated tooth: exact box', 12, False, 'start', MUTE, maxw=196)
    text(448, 440, 'otherwise: detector box', 12, False, 'start', MUTE, maxw=196)
    flame(458, 482, 1.0)
    text(472, 487, 'trained', 12.5, False, 'start', INK, maxw=90)

    # ============================ detector validation (right) ============================
    rect(664, 16, 320, 292, GRAY, rx=14)
    text(680, 41, 'In-fold CV, target mAP50', 14, True, maxw=290)
    for yy, col, lab in ((62, BLD, 'DENTEX-only, mean 0.9508'), (80, ORD, 'DENTEX + Tufts, mean 0.9376')):
        dot(686, yy - 4, col, 5); text(698, yy, lab, 12.5, False, 'start', INK, maxw=270)
    yv = lambda v: 100 + (1.0 - v) * 933.3
    for gv in (0.85, 0.90, 0.95, 1.00):
        add(f'<line x1="700" y1="{yv(gv):.1f}" x2="968" y2="{yv(gv):.1f}" stroke="#DADDE3" stroke-width="1"/>')
        text(694, yv(gv) + 4, f'{gv:.2f}', 11.5, False, 'end', MUTE, maxw=30)
    dnl = [0.9534, 0.9405, 0.9614, 0.9564, 0.9423]
    dtf = [0.9676, 0.9502, 0.9384, 0.9504, 0.8814]
    for v, col in ((0.9508, BLD), (0.9376, ORD)):
        add(f'<line x1="700" y1="{yv(v):.1f}" x2="968" y2="{yv(v):.1f}" stroke="{col}" stroke-width="1.2" stroke-dasharray="5 3"/>')
    for i, (a, b) in enumerate(zip(dnl, dtf)):
        x = 724 + i * 54
        add(f'<line x1="{x}" y1="{yv(a):.1f}" x2="{x}" y2="{yv(b):.1f}" stroke="#9AA3AE" stroke-width="3"/>')
        add(f'<circle cx="{x}" cy="{yv(a):.1f}" r="6" fill="{BLD}" stroke="#FFFFFF" stroke-width="1.4"/>')
        add(f'<circle cx="{x}" cy="{yv(b):.1f}" r="6" fill="{ORD}" stroke="#FFFFFF" stroke-width="1.4"/>')
        text(x, 262, f'fold {i}', 12, False, 'middle', MUTE, maxw=50)
        text(x, 279, f'{a:.3f}', 12, True, 'middle', BLD, maxw=50)
        text(x, 295, f'{b:.3f}', 12, True, 'middle', ORD, maxw=50)

    # ============================ held-out + co-training ============================
    rect(664, 320, 320, 200, GRAY, rx=14)
    text(680, 345, 'Co-training with Tufts', 13.5, True, maxw=170)
    def pair(y, label, a, b, fa, fb):
        text(680, y, label, 12.5, False, 'start', MUTE, maxw=150)
        for k, (v, col, s) in enumerate(((a, BLD, fa), (b, ORD, fb))):
            yy = y + 8 + k * 14
            add(f'<rect x="680" y="{yy}" width="{v * 118:.1f}" height="10" rx="2" fill="{col}"/>')
            text(680 + v * 118 + 6, yy + 9, s, 11.5, True, 'start', col, maxw=48)
    pair(372, 'raw val mAP50', 0.5820, 0.8695, '0.5820', '0.8695')
    pair(422, 'precision', 0.9637, 0.9680, '0.9637', '0.9680')
    text(680, 474, 'protocol differs from the CV', 12, False, 'start', MUTE, italic=True, maxw=170)

    text(906, 345, 'Held-out val', 13.5, True, 'middle', maxw=120)
    for cx, v, lab, vs in ((876, 0.9593, 'mAP50', '0.9593'), (940, 0.9864, 'precision', '0.9864')):
        add(f'<circle cx="{cx}" cy="400" r="24" fill="none" stroke="#DADDE3" stroke-width="7"/>')
        arc(cx, 400, 24, -90, -90 + 360 * v - 0.01, TR, 7)
        text(cx, 444, vs, 12.5, True, 'middle', INK, maxw=60)
        text(cx, 460, lab, 12, False, 'middle', MUTE, maxw=62)
    text(680, 496, '50 images, 46 pathological, 182 target boxes', 12, False, 'start', MUTE, maxw=296)
    text(680, 512, 'best: DENTEX + Tufts, fold 1; deployed backend', 12, False, 'start', MUTE, maxw=296)

    # ============================ candidates, not used ============================
    rect(16, 534, 634, 206, GRAY, rx=14)
    text(32, 559, 'Eight candidate datasets, not used', 14, True, maxw=300)
    dot(446, 555, '#B7C3D6', 5); text(456, 559, 'grounding-only 7 of 8', 12, False, 'start', INK, maxw=190)
    dot(446, 573, '#E3A15A', 5); text(456, 577, 'diagnosis-labeled 1 of 8', 12, False, 'start', INK, maxw=190)
    cand = [('Dental Radiography', 1272, 'no confirmed diagnosis taxonomy', 0),
            ('DNS (UFBA-UESC)', 543, 'FDI boxes in COCO, on request (follow-up)', 0),
            ('Segmented Mandibles', 232, 'mandible-only masks, not per tooth', 0),
            ('TL-pano', 197, 'FDI + quadrant labels, NC license (follow-up)', 0),
            ('TK Tooth Number Code', 188, 'numbering only, no diagnosis', 0),
            ('Tunisia Panoramic', 180, 'in progress, 1 open FDI question', 0),
            ('Panoramic-Paraguay', 135, 'thin documentation in the review', 0),
            ('Panoramic-Caries-Seg', 75, 'binary caries masks, pixel level', 1)]
    for i, (nm, n, note, dg) in enumerate(cand):
        yy = 590 + i * 15
        text(32, yy + 10, nm, 12, True, 'start', INK, maxw=148)
        bw = n / 1272 * 120
        add(f'<rect x="184" y="{yy}" width="{bw:.1f}" height="11" rx="2" fill="{"#E3A15A" if dg else "#B7C3D6"}"/>')
        text(184 + bw + 6, yy + 10, f'{n:,}', 12, False, 'start', BLD, maxw=40)
        text(350, yy + 10, note, 12, False, 'start', MUTE, maxw=294)
    text(32, 731, 'promotion rule: annotation semantics verified end-to-end, per dataset', 12, False, 'start', MUTE, italic=True, maxw=600)

    # ============================ corpus cohorts (donut) ============================
    rect(664, 534, 320, 206, GRAY, rx=14)
    text(680, 559, 'Corpus cohorts', 14, True, maxw=200)
    cx, cy, R = 738, 648, 52
    vals = [('dp', 678), ('dh', 27), ('to', 202), ('th', 660), ('ta', 280)]
    tot = sum(v for _, v in vals); a = -90.0; gap = 1.2
    for key, v in vals:
        da = 360 * v / tot
        arc(cx, cy, R, a + gap / 2, a + da - gap / 2, SEG[key], 22)
        a += da
    text(cx, cy + 4, '1,847', 17, True, 'middle', INK, maxw=70)
    text(cx, cy + 21, 'per arm', 12, False, 'middle', MUTE, maxw=70)
    text(818, 590, 'DENTEX 705', 13, True, 'start', BLD, maxw=150)
    text(818, 656, 'Tufts 1,142', 13, True, 'start', ORD, maxw=150)
    for yy, key, lab in ((608, 'dp', '678 pathology'), (626, 'dh', '27 healthy'),
                         (674, 'to', '202 overlap'), (692, 'th', '660 healthy'), (710, 'ta', '280 all diseases')):
        dot(824, yy - 4, SEG[key], 5); text(836, yy, lab, 12.5, False, 'start', INK, maxw=140)

    finish_path = path
    add('</svg>')
    write(finish_path)
