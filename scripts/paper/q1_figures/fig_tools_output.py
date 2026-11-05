"""Figure 7: what the tools return, and how much effect the agent chooses.

The continuous parameters are the picture itself. Each sweep image is the project's own tool run on one real DENTEX
region (image 334) across its whole parameter range (scripts/paper/prepare_tool_sweeps.py): window_level on a
center x width map, denoise strength 0..1, contrast factor 1..3. Pins mark what the agent chose in two released
traces (DENTEX 334 in blue, Tufts 573 in orange); the outputs at those settings sit beside each sweep. The
data-only tools return coordinates.
"""
from common import *
import json


def build(path):
    out.clear(); warns.clear()
    W, H = 1000, 862
    P = SC + 'params_dentex/'; T = SC + 'params_tufts/'
    swm = Image.open(P + 'sweep_window.jpg').convert('RGB')
    sdn = Image.open(P + 'sweep_denoise.jpg').convert('RGB')
    sct = Image.open(P + 'sweep_contrast.jpg').convert('RGB')
    dt = [Image.open(P + f'tool_{i}.jpg').convert('RGB') for i in range(7)]
    tt = [Image.open(T + f'tool_{i}.jpg').convert('RGB') for i in range(5)]
    sw = json.load(open(P + 'sweeps.json'))
    BLD, ORD = '#3B6BB5', '#D9822B'
    WCOL = ['#2F63C8', '#E0731A', '#2E9B5B', '#8E44AD']

    def strip_crop(im, ratio=5.5):
        h = round(im.width / ratio); y0 = (im.height - h) // 2
        return im.crop((0, y0, im.width, y0 + h))
    sdn = strip_crop(sdn); sct = strip_crop(sct)

    add(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Helvetica, Arial, \'Liberation Sans\', sans-serif">')
    add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

    def pic(x, y, w, im, col='#8A8F98', sw_=1.2):
        h = round(w * im.size[1] / im.size[0], 1)
        image(x, y, w, h, im); frame(x, y, w, h, col, sw_)
        return h

    def pin(x, y0, y1, col):
        add(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" stroke="#0B1220" stroke-opacity="0.6" stroke-width="5"/>')
        add(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" stroke="{col}" stroke-width="2.4"/>')

    def scale(x0, x1, y, lo, hi, ticks, labels, marks):
        px = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
        add(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="#9AA3AE" stroke-width="3" stroke-linecap="round"/>')
        for tk, lab in zip(ticks, labels):
            add(f'<line x1="{px(tk):.1f}" y1="{y - 5}" x2="{px(tk):.1f}" y2="{y + 5}" stroke="#6B7280" stroke-width="1.6"/>')
            text(px(tk), y + 22, lab, 11.5, False, 'middle', MUTE, maxw=70)
        for v, col, lab, hollow in marks:
            fill = '#FFFFFF' if hollow else col
            add(f'<circle cx="{px(v):.1f}" cy="{y}" r="7.5" fill="{fill}" stroke="{col}" stroke-width="3"/>')
            text(px(v) + 11, y - 12, lab, 12.5, True, 'start', col, maxw=60)

    def result(x, y, w, im, col, lab, sub):
        h = pic(x, y, w, im, col, 2.4)
        text(x + w / 2, y + h + 16, lab, 12.5, True, 'middle', col, maxw=w + 40)
        text(x + w / 2, y + h + 31, sub, 12, False, 'middle', MUTE, maxw=w + 40)
        return h

    # ===================== window_level =====================
    rect(16, 16, 968, 300, GRAY, rx=14)
    text(32, 41, 'window_level', 15, True, maxw=150)
    text(150, 41, 'every center and width on one real tooth region', 13, False, 'start', MUTE, maxw=400)
    MX, MY, MW = 62, 76, 470
    MH = round(MW * swm.height / swm.width, 1)
    pic(MX, MY, MW, swm)
    cw = lambda c, w_: (MX + c / 255 * MW, MY + (w_ - sw['width_range'][0]) / (sw['width_range'][1] - sw['width_range'][0]) * MH)
    for v in (0, 128, 255):
        text(MX + v / 255 * MW, MY + MH + 17, str(v), 11.5, False, 'middle', MUTE, maxw=40)
    text(MX + MW / 2, MY + MH + 34, 'center', 12.5, False, 'middle', MUTE, maxw=80)
    for v in (50, 100, 150, 200):
        yy = cw(0, v)[1]
        text(MX - 6, yy + 4, str(v), 11.5, False, 'end', MUTE, maxw=30)
    text(MX - 30, MY - 8, 'width', 12.5, False, 'start', MUTE, maxw=60)
    pins = [(128, 100, 'bone', 'c 128, w 100'), (200, 100, 'enamel', 'c 200, w 100'),
            (80, 150, 'soft tissue', 'c 80, w 150'), (128, 200, 'override', 'c 128, w 200')]
    for i, (c, w_, nm, sub) in enumerate(pins):
        x, y = cw(c, w_)
        add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="13" fill="#FFFFFF" fill-opacity="0.9"/>')
        circ_num(round(x, 1), round(y, 1), i + 1, WCOL[i])
    text(600, 70, 'outputs at the four pins', 12.5, False, 'start', MUTE, maxw=250)
    for i, (c, w_, nm, sub) in enumerate(pins):
        x = 604 + i * 94
        h = pic(x, 82, 66, dt[1 + i], WCOL[i], 2.6)
        circ_num(x + 2, 82, i + 1, WCOL[i])
        text(x + 33, 82 + h + 17, nm, 12.5, True, 'middle', INK, maxw=90)
        text(x + 33, 82 + h + 32, sub, 12, False, 'middle', MUTE, maxw=90)

    # ===================== denoise =====================
    def sweep_panel(y0, name, sub, im, lo, hi, ticks, labels, d_val, t_val, d_pair, t_pair, d_lab, t_lab):
        rect(16, y0, 968, 196, GRAY, rx=14)
        text(32, y0 + 25, name, 15, True, maxw=150)
        text(32 + tw(name, 15, True) + 12, y0 + 25, sub, 13, False, 'start', MUTE, maxw=400)
        SX, SY, SW = 62, y0 + 44, 470
        sh = pic(SX, SY, SW, im)
        px = lambda v: SX + (v - lo) / (hi - lo) * SW
        sy = SY + sh + 38
        for v, col in ((d_val, BLD), (t_val, ORD)):
            pin(px(v), SY, SY + sh, col)
            add(f'<line x1="{px(v):.1f}" y1="{SY + sh}" x2="{px(v):.1f}" y2="{sy - 8}" stroke="{col}" stroke-width="1.6" stroke-dasharray="3 3"/>')
        scale(SX, SX + SW, sy, lo, hi, ticks, labels, [(d_val, BLD, d_lab, False), (t_val, ORD, t_lab, True)])
        # zoom_crop in, tool out: real before / after at each pin
        text(600, y0 + 40, 'zoom_crop in, tool out', 12.5, False, 'start', MUTE, maxw=250)
        for (b_im, a_im, x, w_, col, nm) in ((d_pair[0], d_pair[1], 604, 56, BLD, 'DENTEX 334'), (t_pair[0], t_pair[1], 790, 76, ORD, 'Tufts 573')):
            hb = pic(x, y0 + 50, w_, b_im)
            xa = x + w_ + 26
            pic(xa, y0 + 50, w_, a_im, col, 2.6)
            arrow([(x + w_ + 3, y0 + 50 + hb / 2), (xa - 3, y0 + 50 + hb / 2)], head=7)
            text(x + (xa + w_ - x) / 2, y0 + 50 + hb + 17, nm, 12.5, True, 'middle', col, maxw=170)
    sweep_panel(328, 'denoise', 'bilateral, strength 0 to 1', sdn, 0.0, 1.0, [0.0, 0.5, 1.0], ['0', '0.5', '1'],
                0.5, 0.7, (dt[0], dt[6]), (tt[0], tt[1]), '0.5', '0.7')
    sweep_panel(536, 'enhance_contrast', 'factor 1 to 3', sct, 1.0, 3.0, [1.0, 2.0, 3.0], ['1, unchanged', '2', '3'],
                1.8, 2.5, (dt[0], dt[5]), (tt[0], tt[3]), '1.8', '2.5')

    # ===================== data-only tools + mirror view =====================
    rect(16, 744, 968, 106, GRAY, rx=14)
    text(32, 769, 'Data-only tools return coordinates, not pixels', 14, True, maxw=420)
    cols = [(32, 'locate_tooth', 'FDI 48', 'box [807, 602, 230, 276]'),
            (270, 'nudge_crop', 'dx -0.57, dy 0.69 of the box', 'box [676, 793, 230, 276]'),
            (520, 'fdi_label', 'quadrant 4, position 8', 'FDI 48')]
    for x, nm, call, res in cols:
        text(x, 793, nm, 13.5, True, maxw=200)
        text(x, 810, call, 12.5, False, 'start', MUTE, maxw=230)
        text(x, 827, res, 13, True, 'start', BL, maxw=230)
    arrow([(180, 810), (260, 810)], head=8)
    text(740, 769, 'contralateral_compare, Tufts 573', 12.5, True, 'start', INK, maxw=220)
    lh = pic(740, 777, 110, tt[4])
    text(862, 805, 'region beside', 12, False, 'start', MUTE, maxw=80)
    text(862, 821, 'its mirror', 12, False, 'start', MUTE, maxw=80)

    add('</svg>')
    write(path)
