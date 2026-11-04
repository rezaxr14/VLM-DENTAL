"""Figure 5: Qwen3.5-9B opened up, read left to right.

One horizontal data path (inputs -> frozen ViT -> trained merger -> token sequence -> decoder layers -> MTP block
-> lm_head -> tokens) with everything frozen muted and everything trained vivid, the tool loop around it, and
three zoom-ins: the merger, a Gated DeltaNet layer with its MLP, and a full attention layer with RoPE.
"""
from common import *


def build(path):
    out.clear(); warns.clear()
    W, H = 1000, 812
    opg = Image.open(SC + 'flagship/image.jpg').convert('RGB')
    cb = Image.open(SC + 'flagship/crop_before.jpg').convert('RGB')
    ca = Image.open(SC + 'flagship/crop_after.jpg').convert('RGB')
    tl = {i: Image.open(SC + f'params_dentex/tool_{i}.jpg').convert('RGB') for i in (0, 1, 6)}
    # frozen = pale, trained = vivid
    TEAL_P, TEAL_D = '#9ED5CF', '#1E5F5B'; IND_P, IND_D = '#BDB8F2', '#3E3597'; GOLD_P, GOLD_D = '#F5DE98', '#8A6500'
    ICE = '#DCEAF8'; PALE = '#E6E9EE'; GH = '#B9BBC2'
    TR = '#1F5BFF'; TR_L = '#DCE6FF'; MTPC = '#8A3FE0'
    PUR = '#7B5EA7'; GRN = '#3E8E5A'; ORG = '#D9822B'

    def shade(c, f):
        c = c.lstrip('#'); r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
        t = 255 if f > 0 else 0; f = abs(f)
        r, g, b = (int(v + (t - v) * f) for v in (r, g, b))
        return f'#{r:02X}{g:02X}{b:02X}'
    def block3d(x, y, w, h, d, col):
        k = 0.8
        add(f'<polygon points="{x},{y} {x + d},{y - d * k:.1f} {x + w + d},{y - d * k:.1f} {x + w},{y}" fill="{shade(col, 0.4)}" stroke="#FFFFFF" stroke-width="0.8"/>')
        add(f'<polygon points="{x + w},{y} {x + w + d},{y - d * k:.1f} {x + w + d},{y + h - d * k:.1f} {x + w},{y + h}" fill="{shade(col, -0.2)}" stroke="#FFFFFF" stroke-width="0.8"/>')
        add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{col}" stroke="#FFFFFF" stroke-width="0.8"/>')
    def glow(x, y, w, h, rx=5, p=3):
        add(f'<rect x="{x - p}" y="{y - p}" width="{w + 2 * p}" height="{h + 2 * p}" rx="{rx + p}" fill="{TR}" fill-opacity="0.16"/>')
    def plug(x, y, w=18, col=TR, ghost=True):
        if ghost: add(f'<rect x="{x - 3}" y="{y + 2}" width="{w}" height="8" rx="2" fill="{GH}"/>')
        glow(x, y, w, 8, 2, 2.5)
        add(f'<rect x="{x}" y="{y}" width="{w}" height="8" rx="2" fill="{col}" stroke="#FFFFFF" stroke-width="0.8"/>')
        add(f'<line x1="{x + w / 2}" y1="{y + 8}" x2="{x + w / 2}" y2="{y + 13}" stroke="{col}" stroke-width="1.8"/>')
        flame(x + w / 2, y - 7, 0.78)
    def line(pts, col='#4B5563', sw=1.7):
        add(f'<path d="M{" L".join(f"{a},{b}" for a, b in pts)}" fill="none" stroke="{col}" stroke-width="{sw}" stroke-linejoin="round"/>')
    def circ(x, y, r, label, fs=11):
        add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="#FFFFFF" stroke="#4B5563" stroke-width="1.5"/>')
        if label == '⊙': add(f'<circle cx="{x}" cy="{y}" r="2.6" fill="#4B5563"/>')
        elif label == '+': add(f'<path d="M{x - r * 0.5:.1f},{y} h{r:.1f} M{x},{y - r * 0.5:.1f} v{r:.1f}" stroke="#4B5563" stroke-width="1.8" stroke-linecap="round"/>')
        elif label: text(x, y + fs * 0.36, label, fs, True, 'middle', INK)
    def tok(x, y, col, s=13):
        add(f'<rect x="{x}" y="{y}" width="{s}" height="{s}" rx="3" fill="{col}" stroke="#FFFFFF" stroke-width="0.8"/>')
    def bubble(x, y, w, h, stroke='#9DB8E0', sw=1.6):
        rect(x, y, w, h, ('#FAFBFD', stroke), rx=18, sw=sw, dash='7 4')
    def lin(x, y, name):
        """a projection: frozen weight W on top, the trained LoRA path A -> B beneath, summed"""
        cy = y + 32
        text(x + 53, y - 4, name, 11.5, False, 'middle', INK, maxw=104)
        add(f'<rect x="{x + 16}" y="{y + 3}" width="36" height="24" rx="3" fill="{PALE}" stroke="#B9BBC2" stroke-width="1.2"/>')
        add(f'<g stroke="#CDD1D8" stroke-width="1"><line x1="{x + 28}" y1="{y + 3}" x2="{x + 28}" y2="{y + 27}"/><line x1="{x + 40}" y1="{y + 3}" x2="{x + 40}" y2="{y + 27}"/><line x1="{x + 16}" y1="{y + 11}" x2="{x + 52}" y2="{y + 11}"/><line x1="{x + 16}" y1="{y + 19}" x2="{x + 52}" y2="{y + 19}"/></g>')
        snow(x + 45, y + 10, 4.5)
        line([(x, cy), (x + 8, cy), (x + 8, y + 15), (x + 16, y + 15)], '#6B7280', 1.5); line([(x + 52, y + 15), (x + 90, y + 15), (x + 90, cy - 7)], '#6B7280', 1.5)
        line([(x + 8, cy), (x + 8, y + 49), (x + 16, y + 49)], TR, 2); line([(x + 80, y + 49), (x + 90, y + 49), (x + 90, cy + 7)], TR, 2)
        glow(x + 16, y + 39, 64, 20, 5, 3)
        add(f'<rect x="{x + 16}" y="{y + 39}" width="28" height="20" rx="4" fill="{TR}"/><rect x="{x + 52}" y="{y + 39}" width="28" height="20" rx="4" fill="{TR}"/>')
        add(f'<line x1="{x + 44}" y1="{y + 49}" x2="{x + 52}" y2="{y + 49}" stroke="#FFFFFF" stroke-width="2"/>')
        text(x + 30, y + 53, 'A', 11.5, True, 'middle', '#FFFFFF'); text(x + 66, y + 53, 'B', 11.5, True, 'middle', '#FFFFFF')
        flame(x + 73, y + 36, 0.7); text(x + 40, y + 36, 'LoRA', 10, True, 'middle', TR)
        circ(x + 90, cy, 7, '+')
        arrow([(x + 97, cy), (x + 106, cy)], head=5)
    def tag_in(x, y, w, h, t):
        text(x + w / 2, y + h / 2 + 3.5, t, 9.5, True, 'middle', '#6B7280')

    add(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Helvetica, Arial, \'Liberation Sans\', sans-serif">')
    add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

    # ---- cones into the zoom-ins
    def cone(x1, x2, y1, X1, X2, Y2, col='#9DB8E0'):
        add(f'<polygon points="{x1},{y1} {x2},{y1} {X2},{Y2} {X1},{Y2}" fill="{col}" fill-opacity="0.16" stroke="{col}" stroke-width="1.3" stroke-dasharray="6 4"/>')
    cone(354, 390, 214, 16, 300, 330, TR)
    cone(518, 566, 214, 314, 690, 330)
    cone(662, 708, 214, 704, 984, 330)

    # ===================== tool loop =====================
    add('<g id="toolloop">')
    add('<path d="M881,152 V40 H100 V96" fill="none" stroke="#9AA0A8" stroke-width="2.2" stroke-dasharray="7 4"/><polygon points="100,98 94,88 106,88" fill="#9AA0A8"/>')
    add('<circle cx="190" cy="40" r="14" fill="#FFFFFF" stroke="#9AA0A8" stroke-width="1.8"/>')
    for a_ in range(0, 360, 45):
        ax, ay = math.cos(math.radians(a_)), math.sin(math.radians(a_))
        add(f'<line x1="{190 + ax * 6.5:.1f}" y1="{40 + ay * 6.5:.1f}" x2="{190 + ax * 10:.1f}" y2="{40 + ay * 10:.1f}" stroke="#6B7280" stroke-width="2.3" stroke-linecap="round"/>')
    add('<circle cx="190" cy="40" r="3.4" fill="none" stroke="#6B7280" stroke-width="1.7"/>')
    text(212, 32, 'dispatcher runs the tool;', 12, False, 'start', MUTE, italic=True); text(212, 60, 'crops and text return as tokens', 12, False, 'start', MUTE, italic=True)
    for x_, k in [(540, 0), (640, 1), (740, 6)]:
        image(x_, 4, 62, 74, tl[k].resize((186, 222), Image.LANCZOS)); frame(x_, 4, 62, 74, '#6B7280', 1.4)
    text(964, 33, 'tool call (JSON)', 12, False, 'end', MUTE, italic=True)
    add('</g>')

    # ===================== architecture line =====================
    add('<g id="line">')
    # inputs: big
    image(14, 100, 150, 78, opg.resize((600, 312), Image.LANCZOS)); frame(14, 100, 150, 78, '#6B7280', 1.3)
    image(14, 188, 72, 88, cb.resize((216, 264), Image.LANCZOS)); frame(14, 188, 72, 88, '#5B86C7', 2.3)
    image(92, 188, 72, 88, ca.resize((216, 264), Image.LANCZOS)); frame(92, 188, 72, 88, ORG, 2.3)
    text(90, 294, 'OPG and tool crops', 12, False, 'middle', MUTE, maxw=170)
    arrow([(168, 156), (180, 156)], head=6)
    add('<g transform="translate(-24,0)">')
    for r_ in range(3):
        for c_ in range(3): add(f'<rect x="{206 + c_ * 6}" y="{146 + r_ * 6}" width="5" height="5" rx="1" fill="#B8CFE8"/>')
    # frozen ViT blocks (pale, snowflake)
    for bx in (226, 264):
        block3d(bx, 130, 26, 76, 11, ICE); add(f'<rect x="{bx}" y="170" width="26" height="36" fill="#C5DAF0"/>'); snow(bx + 13, 148, 5.5)
    text(310, 172, '···', 20, True, 'middle', MUTE)
    block3d(324, 130, 26, 76, 11, ICE); add('<rect x="324" y="170" width="26" height="36" fill="#C5DAF0"/>'); snow(337, 148, 5.5)
    text(288, 232, 'ViT, 27 blocks, frozen', 12, False, 'middle', MUTE, maxw=170)
    arrow([(366, 168), (376, 168)], head=5)
    # trained merger: vivid, bigger
    add(f'<polygon points="376,124 416,144 416,192 376,212" fill="{TR}" fill-opacity="0.2" stroke="{TR}" stroke-width="7" stroke-linejoin="round" stroke-opacity="0.18"/>')
    add(f'<polygon points="378,128 414,146 414,190 378,208" fill="{TR_L}" stroke="{TR}" stroke-width="2.8" stroke-linejoin="round"/>'); flame(396, 168, 1.15)
    text(396, 228, 'merger', 12.5, True, 'middle', TR); text(396, 243, 'trained', 11.5, False, 'middle', TR)
    arrow([(418, 168), (428, 168)], head=5)
    seq = 'GGGBBPO'; cmap = {'P': PUR, 'B': '#5B86C7', 'G': GRN, 'O': ORG}
    for i, c in enumerate(seq): tok(430 + i * 14, 161, cmap[c], 12)
    add(f'<rect x="440" y="86" width="22" height="30" rx="3" fill="#F1EBF9" stroke="{PUR}" stroke-width="1.5"/><g stroke="{PUR}" stroke-width="1.5"><line x1="445" y1="94" x2="457" y2="94"/><line x1="445" y1="100" x2="457" y2="100"/><line x1="445" y1="106" x2="452" y2="106"/></g>')
    for k in range(3, -1, -1): add(f'<rect x="{470 + k * 4}" y="{88 - k * 2}" width="20" height="28" rx="3" fill="#F1EBF9" stroke="{PUR}" stroke-width="1.2"/>')
    text(476, 78, 'rules.md, skills', 11.5, True, 'middle', PUR, maxw=120)
    for xx in (452, 484):
        add(f'<path d="M{xx},120 V155" fill="none" stroke="{PUR}" stroke-width="1.8" stroke-dasharray="4 3"/><polygon points="{xx},158 {xx - 5},149 {xx + 5},149" fill="{PUR}"/>')
    rect(436, 196, 62, 22, ('#EEF0F3', '#B9BBC2'), rx=6, sw=1.3); snow(447, 207, 4.5); text(477, 211, 'embed', 11.5, False, 'middle', INK)
    arrow([(466, 194), (466, 178)], head=6)
    arrow([(530, 168), (540, 168)], head=6)
    # decoder: first group, frozen base pale, trained LoRA vivid
    for k in range(4):
        x0 = 542 + k * 48; full = k == 3
        block3d(x0 + 28, 140, 12, 62, 6, GOLD_P)
        block3d(x0, 126, 26, 76, 8, IND_P if full else TEAL_P)
        snow(x0 + 13, 166, 5.5); snow(x0 + 34, 160, 4)
        plug(x0 + 4, 106, 18); plug(x0 + 30, 120, 12)
        text(x0 + 17, 222, 'attention' if full else 'DeltaNet', 11, False, 'middle', IND_D if full else TEAL_D, maxw=50)
    text(742, 172, '···', 20, True, 'middle', MUTE)
    text(596, 262, 'same pattern, 8 groups = 32 layers', 11.5, True, 'middle', INK, maxw=240)
    # MTP block on the same line (trained): hidden state -> MTP block -> shared lm_head -> tokens
    arrow([(752, 168), (758, 168)], head=4)
    glow(758, 126, 26, 76, 5, 4)
    block3d(758, 126, 26, 76, 8, MTPC); block3d(786, 140, 12, 62, 6, '#C9A9F2')
    flame(771, 150, 1.0); plug(762, 106, 18, MTPC, ghost=False)
    text(782, 222, 'MTP block', 11.5, True, 'middle', MTPC)
    arrow([(808, 168), (818, 168)], head=5)
    block3d(820, 136, 20, 64, 8, '#D7DBE2'); snow(830, 150, 5)
    text(834, 121, 'lm_head', 11.5, False, 'middle', INK); text(834, 108, 'shared, frozen', 11, False, 'middle', MUTE, italic=True)
    arrow([(852, 168), (862, 168)], head=5)
    rect(864, 152, 82, 32, ('#E4F3E9', GRN), rx=9, sw=1.8); text(905, 172, 'next token', 12, True, 'middle', INK)
    for k, c in enumerate('ABC'):
        rect(952 + k * 20, 152, 17, 32, ('#F1EBF9', PUR), rx=4, sw=1.3, dash='3 2'); text(960.5 + k * 20, 172, c, 11.5, True, 'middle', MTPC)
    text(990, 200, 'draft tokens, target verifies', 11, False, 'end', MUTE)
    # the main prediction: the last hidden state also goes straight into lm_head
    line([(736, 206), (736, 238), (830, 238), (830, 206)], '#6B7280', 1.8); add('<polygon points="830,203 825,213 835,213" fill="#6B7280"/>')
    text(783, 254, 'main path: hidden state', 11, False, 'middle', MUTE)
    add('</g>')
    add('</g>')

    # ===================== zoom 1: the merger =====================
    add('<g id="zoom-merger">')
    bubble(16, 330, 284, 418, TR, 2.2)
    flame(36, 352, 0.95); text(50, 357, 'Merger, trained', 13.5, True, 'start', TR)
    # frozen ViT collapsed at the top
    for bx in (40, 60):
        block3d(bx, 372, 14, 30, 6, ICE); snow(bx + 7, 382, 3.4)
    text(90, 392, '···', 14, True, 'middle', MUTE)
    block3d(104, 372, 14, 30, 6, ICE); snow(111, 382, 3.4)
    text(136, 384, 'ViT blocks,', 11.5, False, 'start', MUTE); text(136, 398, 'frozen', 11.5, False, 'start', MUTE)
    arrow([(158, 408), (158, 420)], head=6); text(170, 418, 'patch tokens', 11.5, False, 'start', MUTE)
    image(98, 424, 120, 54, opg.crop((150, 130, 700, 380)).resize((480, 216), Image.LANCZOS)); frame(98, 424, 120, 54, '#6B7280', 1.2)
    for c_ in range(1, 12): add(f'<line x1="{98 + c_ * 10}" y1="424" x2="{98 + c_ * 10}" y2="478" stroke="#FFFFFF" stroke-opacity="0.8" stroke-width="1"/>')
    for r_ in range(1, 5): add(f'<line x1="98" y1="{424 + r_ * 10.8:.1f}" x2="218" y2="{424 + r_ * 10.8:.1f}" stroke="#FFFFFF" stroke-opacity="0.8" stroke-width="1"/>')
    add(f'<rect x="138" y="443" width="20" height="21.6" fill="{TR}" fill-opacity="0.25" stroke="{TR}" stroke-width="2.6"/>')
    text(244, 446, '2×2', 12, True, 'middle', TR); text(244, 460, 'group', 11.5, False, 'middle', MUTE)
    arrow([(148, 480), (148, 490)], head=5)
    for i in range(4): tok(118 + i * 15, 492, '#8FB0DD', 12)
    text(190, 502, 'concatenated', 11.5, False, 'start', MUTE)
    line([(117, 498), (98, 498), (98, 560)]); arrow([(98, 560), (106, 560)], head=5)
    lin(106, 528, 'merger fc1')
    add(f'<circle cx="158" cy="612" r="9" fill="#FFFFFF" stroke="#4B5563" stroke-width="1.5"/><path d="M153,615 L158,615 Q162,615 164,608" fill="none" stroke="#4B5563" stroke-width="1.6"/>')
    line([(212, 560), (228, 560), (228, 612)]); arrow([(228, 612), (169, 612)], head=5); text(250, 616, 'GELU', 11, False, 'middle', MUTE)
    line([(149, 612), (92, 612), (92, 674)]); arrow([(92, 674), (106, 674)], head=5)
    lin(106, 642, 'merger fc2')
    line([(212, 674), (234, 674), (234, 722)]); arrow([(234, 722), (206, 722)], head=5)
    for i in range(3): tok(150 + i * 17, 716, GRN, 13)
    text(158, 742, 'visual tokens to the sequence', 11.5, False, 'middle', MUTE, maxw=260)
    add('</g>')

    # ===================== zoom 2: Gated DeltaNet + MLP =====================
    add('<g id="zoom-delta">')
    bubble(314, 330, 376, 418)
    add(f'<rect x="318" y="346" width="4" height="196" rx="2" fill="{TEAL_P}"/><rect x="318" y="554" width="4" height="184" rx="2" fill="{GOLD_P}"/>')
    text(332, 356, 'Gated DeltaNet layer', 13.5, True, 'start', TEAL_D); text(332, 565, 'MLP', 13.5, True, 'start', GOLD_D)
    # mixer
    add(f'<rect x="320" y="412" width="13" height="22" rx="4" fill="#5B86C7" fill-opacity="0.25" stroke="#5B86C7" stroke-width="1.4"/>'); text(326.5, 427, 'h', 11, True, 'middle', INK)
    line([(326, 412), (326, 372), (674, 372), (674, 501)]); add('<polygon points="674,503 669,493 679,493" fill="#4B5563"/>')
    arrow([(334, 424), (346, 424)], head=5)
    lin(346, 392, 'in_proj_qkv')
    add('<rect x="466" y="413" width="26" height="22" rx="4" fill="#EEF0F3" stroke="#B9BBC2" stroke-width="1.3"/><path d="M469,424 q3,-9 6,0 t6,0 t6,0" fill="none" stroke="#9AA0A8" stroke-width="1.5"/>'); snow(487, 417, 3.5)
    text(479, 449, 'conv1d', 11, False, 'middle', MUTE)
    arrow([(456, 424), (464, 424)], head=4); arrow([(494, 424), (506, 424)], head=4)
    for r_ in range(5):
        for c_ in range(5):
            v = [0.15, 0.7, 0.35, 0.9, 0.5][(r_ * 2 + c_ * 3) % 5]
            add(f'<rect x="{510 + c_ * 9}" y="{402 + r_ * 9}" width="8" height="8" rx="1.5" fill="{TEAL_D}" fill-opacity="{0.15 + 0.75 * v:.2f}"/>')
    text(532, 396, 'state S', 11.5, True, 'middle', INK)
    add('<rect x="590" y="404" width="22" height="22" rx="3" fill="#EEF0F3" stroke="#B9BBC2" stroke-width="1.3"/>'); snow(606, 410, 3.5)
    text(601, 441, 'a, b gates', 11, False, 'middle', MUTE); arrow([(588, 415), (558, 415)], head=5)
    arrow([(532, 450), (532, 498)], head=5)
    line([(326, 434), (326, 508), (334, 508)]); arrow([(334, 508), (346, 508)], head=5)
    lin(346, 476, 'in_proj_z')
    circ(488, 508, 9, 'σ', 11); arrow([(456, 508), (478, 508)], head=5)
    circ(532, 508, 9, '⊙'); arrow([(498, 508), (522, 508)], head=5)
    lin(552, 476, 'out_proj'); arrow([(541, 508), (550, 508)], head=4)
    circ(674, 508, 8, '+'); line([(658, 508), (666, 508)])
    # MLP
    add(f'<rect x="320" y="605" width="13" height="22" rx="4" fill="#5B86C7" fill-opacity="0.25" stroke="#5B86C7" stroke-width="1.4"/>'); text(326.5, 620, 'h', 11, True, 'middle', INK)
    line([(326, 605), (326, 574), (674, 574), (674, 609)]); add('<polygon points="674,611 669,601 679,601" fill="#4B5563"/>')
    arrow([(334, 616), (346, 616)], head=5)
    lin(346, 584, 'gate_proj'); circ(488, 616, 11, 'SiLU', 8); arrow([(456, 616), (477, 616)], head=5)
    circ(532, 616, 9, '⊙'); arrow([(499, 616), (522, 616)], head=5)
    line([(326, 627), (326, 700), (334, 700)]); arrow([(334, 700), (346, 700)], head=5)
    lin(346, 668, 'up_proj'); line([(452, 700), (532, 700), (532, 625)]); add('<polygon points="532,623 527,633 537,633" fill="#4B5563"/>')
    lin(552, 584, 'down_proj'); arrow([(541, 616), (550, 616)], head=4)
    circ(674, 616, 8, '+'); line([(658, 616), (666, 616)])
    add('</g>')

    # ===================== zoom 3: full attention with RoPE =====================
    add('<g id="zoom-attn">')
    bubble(704, 330, 280, 418)
    add(f'<rect x="708" y="346" width="4" height="396" rx="2" fill="{IND_P}"/>')
    text(722, 356, 'Full attention layer', 13.5, True, 'start', IND_D)
    add(f'<rect x="716" y="468" width="14" height="22" rx="4" fill="#5B86C7" fill-opacity="0.25" stroke="#5B86C7" stroke-width="1.4"/>'); text(723, 483, 'h', 11, True, 'middle', INK)
    line([(730, 479), (738, 479)]); line([(738, 404), (738, 548)])
    for cy_ in (404, 476, 548): arrow([(738, cy_), (750, cy_)], head=4)
    lin(750, 372, 'q_proj'); lin(750, 444, 'k_proj'); lin(750, 516, 'v_proj')
    def rope(cx, cy):
        add(f'<circle cx="{cx}" cy="{cy}" r="12" fill="#FFFFFF" stroke="#C77D0A" stroke-width="1.8"/>')
        add(f'<path d="M{cx - 6},{cy + 3} A7,7 0 1 1 {cx + 5},{cy + 5}" fill="none" stroke="#C77D0A" stroke-width="1.9" stroke-linecap="round"/><polygon points="{cx + 7},{cy + 2} {cx + 3},{cy + 9} {cx + 10},{cy + 8}" fill="#C77D0A"/>')
        text(cx, cy + 25, 'RoPE', 11, True, 'middle', '#8A5300')
    rope(884, 404); rope(884, 476)
    arrow([(858, 404), (870, 404)], head=4); arrow([(858, 476), (870, 476)], head=4)
    arrow([(897, 404), (912, 414)], head=5); arrow([(897, 476), (912, 438)], head=5)
    for r_ in range(8):
        for c_ in range(8):
            if c_ <= r_:
                v = 0.25 + 0.7 * ((r_ * 7 + c_ * 3) % 5) / 4
                add(f'<rect x="{914 + c_ * 7}" y="{392 + r_ * 7}" width="6.4" height="6.4" rx="1.2" fill="#5A4FCF" fill-opacity="{v:.2f}"/>')
            else:
                add(f'<rect x="{914 + c_ * 7}" y="{392 + r_ * 7}" width="6.4" height="6.4" rx="1.2" fill="#EEF0F3"/>')
    text(942, 386, 'softmax(QK^T)', 11, True, 'middle', INK)
    arrow([(942, 450), (942, 538)], head=5); text(934, 500, 'causal', 11, False, 'end', MUTE)
    circ(942, 548, 9, '⊙'); arrow([(858, 548), (931, 548)], head=5); text(900, 540, '× V', 11, True, 'middle', INK)
    line([(942, 557), (942, 592), (736, 592), (736, 628)]); arrow([(736, 628), (748, 628)], head=5)
    lin(750, 596, 'o_proj')
    text(722, 700, 'KV cache grows with the context,', 11.5, False, 'start', MUTE); text(722, 715, 'in these 8 layers only;', 11.5, False, 'start', MUTE)
    text(722, 735, 'then the same MLP as in the DeltaNet layer', 11.5, True, 'start', INK, maxw=254)
    add('</g>')

    # ===================== legend =====================
    add('<g id="legend">')
    ly = 768
    snow(34, ly - 4, 8.5); text(50, ly, 'frozen', 13, True, 'start', INK)
    flame(122, ly - 4, 1.25); text(136, ly, 'trained', 13, True, 'start', TR)
    add(f'<rect x="206" y="{ly - 10}" width="14" height="10" rx="2" fill="{GH}"/><rect x="212" y="{ly - 10}" width="14" height="10" rx="2" fill="{TR}" stroke="#FFFFFF" stroke-width="0.8"/>')
    text(236, ly, 'LoRA: GRPO adapter in front of the frozen SFT adapter (KL reference)', 12.5, False, 'start', INK, maxw=520)
    add(f'<rect x="780" y="{ly - 10}" width="12" height="10" rx="2" fill="{MTPC}"/>'); text(798, ly, 'MTP block, fine-tuned', 12.5, False, 'start', INK, maxw=190)
    text(34, ly + 20, 'frozen: 4-bit base weights, ViT blocks, embeddings, lm_head, norms, conv1d, a/b gates', 11.5, False, 'start', MUTE, maxw=940)
    text(34, ly + 36, 'trained: LoRA on q, k, v, o, in_proj_qkv, in_proj_z, out_proj, gate, up, down; the two merger layers; the MTP block', 11.5, False, 'start', MUTE, maxw=940)
    add('</g>')

    add('</svg>')
    write(path)
