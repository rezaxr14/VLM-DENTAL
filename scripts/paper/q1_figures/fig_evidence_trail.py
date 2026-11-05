"""Figure 8: one image, two traces, and the tool use behind them.

Top: the evidence trail of DENTEX image 624 (FDI 34): the box the detector returned, three corrective nudges and
the expert box on the real radiograph, the walk magnified, and the two crops the agent saw before and after.
Middle: the no-tools trace (one turn) against the with-tools trace (turn lane, coloured by tool).
Bottom: the tool-call distribution of the verified corpus (same colours) and the exact 1:1 pairing of the two arms.
"""
from common import *


def build(path):
    out.clear(); warns.clear()
    W, H = 1000, 786
    A = A624
    opg = Image.open(A + 'opg_624.png').convert('RGB')
    inset = Image.open(A + 'inset_624.png').convert('RGB')
    c_loc = Image.open(A + 'crop_located_624.png').convert('RGB')
    c_fin = Image.open(A + 'crop_final_624.png').convert('RGB')
    REF = '#22C55E'
    TC = {'zoom_crop': '#3B6BB5', 'locate_tooth': '#1E8FA8', 'contralateral_compare': '#7B5EA7', 'nudge_crop': '#D9822B',
          'window_level': '#C99700', 'denoise': '#E6BE45', 'enhance_contrast': '#8F6B00', 'fdi_label': '#8A8F98'}
    NUD = ['#FFC287', '#FF9A4D', '#E8590C']
    add(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Helvetica, Arial, \'Liberation Sans\', sans-serif">')
    add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

    def refbox(x, y, w, h, pad=1.2, sw=1.8, dash='5 3'):
        x -= pad; y -= pad; w += 2 * pad; h += 2 * pad
        add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" stroke="#06210F" stroke-opacity="0.65" stroke-width="{sw + 1.8}" stroke-dasharray="{dash}"/>')
        add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" stroke="{REF}" stroke-width="{sw}" stroke-dasharray="{dash}"/>')

    def sbox(x, y, w, h, col, sw):
        add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" stroke="#0B1220" stroke-opacity="0.45" stroke-width="{sw + 1.4}"/>')
        add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" stroke="{col}" stroke-width="{sw}"/>')

    def pic(x, y, w, im, col='#8A8F98', sw=1.2):
        h = round(w * im.size[1] / im.size[0], 1)
        image(x, y, w, h, im); frame(x, y, w, h, col, sw)
        return h

    GT = (1621, 694, 84, 322)
    BOXES = [('locate', (1668, 917, 84, 322), BL), ('n1', (1685, 885, 50, 193), NUD[0]),
             ('n2', (1693, 818, 35, 135), NUD[1]), ('n3', (1696, 750, 28, 108), NUD[2])]
    REG = (1560, 629, 280, 659)

    # ========================= A: evidence trail =========================
    rect(16, 16, 968, 354, GRAY, rx=14)
    text(32, 41, 'Evidence trail, DENTEX image 624, FDI 34', 15, True, maxw=500)
    OX, OY, OW = 32, 58, 560
    oh = pic(OX, OY, OW, opg)
    k = OW / 2808
    def tb(b): return (OX + b[0] * k, OY + b[1] * k, b[2] * k, b[3] * k)
    rg = tb(REG)
    add(f'<rect x="{rg[0]:.1f}" y="{rg[1]:.1f}" width="{rg[2]:.1f}" height="{rg[3]:.1f}" fill="none" stroke="#FFFFFF" stroke-width="1.4"/>')
    g = tb(GT); refbox(*g, 1.2, 1.8)
    for nm, b, col in BOXES:
        sbox(*tb(b), col, 2.2 if nm != 'locate' else 2.4)
    # inset: the walk magnified
    IX, IY, IW = 628, 58, 112
    ih = pic(IX, IY, IW, inset)
    ks = IW / REG[2]
    def ib(b): return (IX + (b[0] - REG[0]) * ks, IY + (b[1] - REG[1]) * ks, b[2] * ks, b[3] * ks)
    refbox(*ib(GT), 1.2, 2.0)
    for nm, b, col in BOXES: sbox(*ib(b), col, 2.4)
    for p, q in (((rg[0] + rg[2], rg[1]), (IX, IY)), ((rg[0] + rg[2], rg[1] + rg[3]), (IX, IY + ih))):
        add(f'<line x1="{p[0]:.1f}" y1="{p[1]:.1f}" x2="{q[0]}" y2="{q[1]}" stroke="#9DB8E0" stroke-width="1.2" stroke-dasharray="3 3"/>')
    text(IX + IW / 2, IY + ih + 17, '2.6× zoom', 12, False, 'middle', MUTE, maxw=110)
    # the two crops the agent saw
    LX, LW = 760, 80
    lh = pic(LX, 58, LW, c_loc)
    text(LX + LW / 2, 58 + lh + 17, 'at locate_tooth', 12, False, 'middle', MUTE, maxw=110)
    FX, FW = 872, 76
    fh = pic(FX, 58, FW, c_fin, NUD[2], 2.4)
    text(FX + FW / 2, 58 + fh + 17, 'after 3 nudges', 12, False, 'middle', MUTE, maxw=110)
    arrow([(LX + LW + 3, 58 + lh / 2), (FX - 3, 58 + lh / 2)], col=OR, sw=2.4, head=8)
    # legend
    ly = OY + oh + 24
    add(f'<line x1="32" y1="{ly - 4}" x2="58" y2="{ly - 4}" stroke="{BL}" stroke-width="3"/>'); text(64, ly, 'locate_tooth box', 12.5, False, 'start', INK, maxw=130)
    for i, c in enumerate(NUD): add(f'<line x1="{214 + i * 9}" y1="{ly - 4}" x2="{222 + i * 9}" y2="{ly - 4}" stroke="{c}" stroke-width="3"/>')
    text(246, ly, 'nudges 1 to 3', 12.5, False, 'start', INK, maxw=110)
    add(f'<line x1="372" y1="{ly - 4}" x2="398" y2="{ly - 4}" stroke="{REF}" stroke-width="3" stroke-dasharray="5 3"/>'); text(404, ly, 'expert box', 12.5, False, 'start', INK, maxw=100)

    # ========================= B: two traces =========================
    rect(16, 382, 968, 214, GRAY, rx=14)
    text(32, 407, 'Same image, same label, two traces', 15, True, maxw=500)
    X0, X1 = 150, 944; step = (X1 - X0) / 22
    tx = lambda t: X0 + t * step
    # no-tools lane
    text(32, 450, 'No tools', 13.5, True, maxw=100)
    text(32, 466, '1 turn, 0 calls', 12, False, 'start', MUTE, maxw=110)
    add(f'<line x1="{X0}" y1="458" x2="{X1}" y2="458" stroke="#D0D4DA" stroke-width="2"/>')
    add(f'<circle cx="{tx(0)}" cy="458" r="8" fill="#9AA3AE" stroke="#FFFFFF" stroke-width="2"/>')
    text(tx(0) + 18, 444, 'one description of the full image: no crop, no second look, nothing to check', 12.5, False, 'start', MUTE, italic=True, maxw=620)
    check(tx(22), 458, 9, '#9AA3AE')
    # with-tools lane
    LY = 540
    text(32, 534, 'With tools', 13.5, True, maxw=100)
    text(32, 550, '23 turns, 22 calls', 12, False, 'start', MUTE, maxw=116)
    add(f'<line x1="{X0}" y1="{LY}" x2="{X1}" y2="{LY}" stroke="#C3C7CF" stroke-width="2"/>')
    for t in range(23): add(f'<line x1="{tx(t):.1f}" y1="{LY - 3}" x2="{tx(t):.1f}" y2="{LY + 3}" stroke="#B9BBC2" stroke-width="1"/>')
    calls = [(0, 'window_level', 'window_level', 1), (1, 'fdi_label', 'fdi_label', 2), (2, 'locate_tooth', 'locate_tooth', 1),
             (3, 'zoom_crop', 'zoom_crop', 2), (4, 'nudge_crop', 'nudge 1', 1), (6, 'nudge_crop', 'nudge 2', 1), (8, 'nudge_crop', 'nudge 3', 1),
             (9, 'zoom_crop', None, 0), (10, 'denoise', None, 0)]
    for t, tool, lab, row in calls:
        r = 8.5 if tool == 'nudge_crop' else 7
        add(f'<circle cx="{tx(t):.1f}" cy="{LY}" r="{r}" fill="{TC[tool]}" stroke="#FFFFFF" stroke-width="2"/>')
        if lab:
            if row == 1:
                if t == 0: text(tx(t) - 8, LY - 38, lab, 11.5, True, 'start', INK, maxw=80)
                else: text(tx(t), LY - 18, lab, 11.5, True, 'middle', INK, maxw=76)
            else:
                text(tx(t), LY + 26, lab, 11.5, True, 'middle', INK, maxw=72)
    text((tx(9) + tx(10)) / 2, LY + 26, 'zoom, denoise', 11.5, True, 'middle', INK, maxw=90)
    # T14..T21: compare, enhance, re-verify
    bx0, bx1 = tx(14), tx(21)
    add(f'<path d="M{bx0:.1f},{LY - 10} v-6 H{bx1:.1f} v6" fill="none" stroke="#9AA3AE" stroke-width="1.6"/>')
    for i, tool in enumerate(('contralateral_compare', 'enhance_contrast')):
        add(f'<circle cx="{(bx0 + bx1) / 2 - 14 + i * 28:.1f}" cy="{LY}" r="7" fill="{TC[tool]}" stroke="#FFFFFF" stroke-width="2"/>')
    text((bx0 + bx1) / 2, LY + 26, 'compare, enhance, re-verify', 11.5, True, 'middle', INK, maxw=190)
    check(tx(22), LY, 9, GR)
    text(tx(22), LY - 18, 'answer', 11.5, True, 'middle', GR, maxw=60)
    text(X0, 582, 'turns 0 to 22; the calls named in the trace are marked', 11.5, False, 'start', MUTE, italic=True, maxw=400)
    # ========================= C: distribution and pairing =========================
    rect(16, 608, 968, 166, GRAY, rx=14)
    text(32, 634, 'Tool calls, verified with-tools corpus', 14, True, maxw=330)
    text(32, 651, '44,350 calls over 1,847 traces', 12.5, False, 'start', MUTE, maxw=300)
    dist = [('zoom_crop', 17292), ('locate_tooth', 15663), ('contralateral_compare', 2730), ('nudge_crop', 2292),
            ('window_level', 2142), ('denoise', 1610), ('enhance_contrast', 1456), ('fdi_label', 1117)]
    for i, (tl, n) in enumerate(dist):
        y = 662 + i * 13.2
        text(176, y + 9, tl, 11.5, False, 'end', INK, maxw=150)
        bw = n / 17292 * 240
        add(f'<rect x="184" y="{y:.1f}" width="{bw:.1f}" height="10" rx="2" fill="{TC[tl]}"/>')
        text(184 + bw + 6, y + 9, f'{n:,}', 11.5, False, 'start', MUTE, maxw=50)
    # pairing: two equal stacked bars
    PX, PW = 560, 400
    text(PX, 634, 'Exact 1:1 pairing', 14, True, maxw=300)
    text(PX, 651, '5 cohorts, 1,847 + 1,847 = 3,694 traces', 12.5, False, 'start', MUTE, maxw=400)
    SEG = [('#3B6BB5', 678), ('#9DB8E0', 27), ('#D9822B', 202), ('#F2B97F', 660), ('#A85A12', 280)]
    for row, (lab, yy, sub) in enumerate((('with tools', 684, '44,350 calls'), ('no tools', 722, '0 calls'))):
        x = PX
        for col, n in SEG:
            w = PW * n / 1847
            add(f'<rect x="{x:.1f}" y="{yy}" width="{w:.1f}" height="22" fill="{col}" stroke="#FFFFFF" stroke-width="1"/>')
            if row == 0 and w > 26: text(x + w / 2, yy + 16, str(n), 11.5, True, 'middle', '#FFFFFF' if col not in ('#9DB8E0', '#F2B97F') else INK, maxw=w - 2)
            x += w
        text(PX, yy - 5, lab, 12.5, True, 'start', INK, maxw=100)
        text(PX + PW, yy - 5, sub, 12.5, False, 'end', MUTE, maxw=140)
    text(PX, 763, 'same image, same prompt, same label; only the tool channel differs', 12, False, 'start', MUTE, italic=True, maxw=404)

    add('</svg>')
    write(path)
