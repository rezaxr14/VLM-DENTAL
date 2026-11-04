"""Figure 1: VLM-DENTAL overview (offline data and training, inference loop, two optional extensions)."""
from common import *
import sys


def build(path):
    out.clear(); warns.clear()
    W, H = 1000, 612
    add(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Helvetica, Arial, \'Liberation Sans\', sans-serif">')
    add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

    # ===================== OFFLINE PANEL =====================
    add('<g id="offline">')
    rect(16,16,284,580,GRAY,rx=14)
    text(32,41,'Offline: data and training',15,True,maxw=260)
    box(32,56,252,46,BLUE,'Expert labels','DENTEX + Tufts, FDI notation')
    cylinder(262,79)
    arrow([(158,102),(158,120)])
    # teacher (orange: our perturbation)
    rect(32,120,252,124,ORANGE)
    text(158,142,'Teacher VLM + real tools',14,True,'middle',maxw=240)
    text(158,159,'label-directed LangGraph loop',13,False,'middle',MUTE,maxw=240)
    text(158,182,'tiered perturbation of the shown box',13,False,'middle','#9A5410',italic=True,maxw=240)
    bx,bw=44,228
    segs=[(0.45,'#E9E4DA','#5B6472','none','45%'),(0.25,'#F4B77A','#4A2A06','small','25%'),(0.30,'#E07B1F','#FFFFFF','big','30%')]
    x=bx
    for fr,fc,tc,lab,pc in segs:
        w=bw*fr
        add(f'<rect x="{x:.1f}" y="192" width="{w:.1f}" height="38" fill="{fc}" stroke="#FFFFFF" stroke-width="1"/>')
        text(x+w/2,208,lab,12.5,False,'middle',tc,maxw=w-4)
        text(x+w/2,224,pc,13,True,'middle',tc,maxw=w-4)
        x+=w
    arrow([(158,244),(158,262)])
    box(32,262,252,46,GREEN,'Cross-family verifier','label check, repair, decontamination')
    arrow([(158,308),(158,326)])
    box(32,326,252,50,GREEN,'Verified corpus (3,694 traces)','1,847 with tools + 1,847 without')
    arrow([(86,376),(86,420)])
    box(32,420,108,46,BLUE,'1 · SFT','verified traces')
    box(152,420,132,46,BLUE,'2 · GRPO','GT-anchored reward')
    arrow([(140,443),(152,443)],head=7)
    arrow([(218,466),(218,471),(93,471),(93,480)],head=7)
    arrow([(218,466),(218,480)],head=7)
    box(32,480,122,32,BLUE,'harness off',ts=13)
    box(162,480,122,32,PURPLE,'rules + skills',ts=13,dash='5 3')
    arrow([(93,512),(93,526)],head=7)
    arrow([(223,512),(223,526)],head=7)
    box(32,526,252,42,BLUE,'LoRA adapters','SFT, GRPO off, GRPO rules + skills')
    flame(97,538,0.95)
    snow(48,584,6.5); text(62,589,'frozen SFT adapter = KL reference',12.5,False,'start',MUTE,maxw=222)
    add('</g>')


    # ===================== HARNESS EXTENSION =====================
    add('<g id="harness">')
    rect(316,16,668,108,PURPLE,rx=14,dash='7 4')
    text(332,41,'Extension 1: rules and skills harness',14,True,'start','#5B3F86',maxw=330)
    box(332,52,208,58,WHITE,'rules.md','versioned system rules')
    box(560,52,408,58,WHITE,'Skill library (5 skills)','index in prompt; read_skill loads a body on demand')
    add('</g>')

    # ===================== INFERENCE LOOP =====================
    add('<g id="inference">')
    rect(316,134,668,278,GRAY,rx=14)
    text(332,156,'Inference loop',14,True,maxw=130)
    # image (real panoramic radiograph)
    import io, base64
    from PIL import Image as _I
    _opg=_I.open(SC + 'flagship/image.jpg').convert('RGB').resize((380,199))
    _b=io.BytesIO(); _opg.save(_b,'JPEG',quality=88)
    add(f'<image x="328" y="214" width="96" height="50" preserveAspectRatio="none" xlink:href="data:image/jpeg;base64,{base64.b64encode(_b.getvalue()).decode()}"/>')
    add('<rect x="328" y="214" width="96" height="50" fill="none" stroke="#6B7280" stroke-width="1.4"/>')
    text(376,207,'panoramic X-ray',12.5,False,'middle',MUTE,maxw=100)
    arrow([(424,239),(440,239)],head=7)
    # policy
    rect(440,190,160,140,BLUE)
    text(520,214,'Policy',16,True,'middle')
    snow(462,240,6.5); text(474,245,'Qwen3.5-9B',14,False,'start',maxw=118)
    flame(462,264,0.95); text(474,269,'+ LoRA adapters',14,False,'start',maxw=118)
    text(520,308,'thought, then tool call',12.5,False,'middle',MUTE,maxw=150)
    # call/obs
    arrow([(600,224),(656,224)])
    text(628,216,'call',12.5,False,'middle',MUTE)
    arrow([(656,258),(600,258)])
    text(628,274,'obs',12.5,False,'middle',MUTE)
    # tools
    rect(656,150,324,256,BLUE)
    text(668,171,'Tools (run on the native image)',13,True,maxw=300)
    tools=[('zoom_crop','window_level'),('denoise','enhance_contrast'),('contralateral_compare','locate_tooth'),('fdi_label','nudge_crop')]
    for i,(a,b) in enumerate(tools):
        y=182+i*28
        for j,nm in enumerate((a,b)):
            cx=664+j*158
            col = ORANGE if nm=='nudge_crop' else WHITE
            rect(cx,y,150,23,col,rx=5,sw=1.3)
            text(cx+75,y+16.5,nm,12.5,nm=='nudge_crop','middle',maxw=142)
    # grounding strip
    text(668,312,'grounding is a hypothesis',13,False,'start','#9A5410',italic=True,maxw=300)
    def tile(x,kind):
        rect(x,320,96,56,WHITE,rx=5,sw=1.2)
        cx=x+48
        add(f'<rect x="{cx-9}" y="326" width="18" height="44" rx="8" fill="#ECE6D6" stroke="#A39E8F" stroke-width="1.2"/>')
        if kind==1:
            add(f'<rect x="{cx-3}" y="344" width="24" height="26" fill="none" stroke="#3B6BB5" stroke-width="2.2"/>')
        elif kind==2:
            add(f'<rect x="{cx+3}" y="346" width="24" height="26" fill="none" stroke="#3B6BB5" stroke-width="1.6" stroke-dasharray="3 2"/>')
            arrow([(cx+14,352),(cx-8,336)],col='#D9822B',sw=2.2,head=7)
        else:
            add(f'<rect x="{cx-12}" y="325" width="24" height="46" fill="none" stroke="#D9822B" stroke-width="2.4"/>')
            add(f'<path d="M{cx+16},{344} l4,5 l8,-10" fill="none" stroke="#3E8E5A" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>')
    for x,k,lab in [(664,1,'locate'),(770,2,'nudge'),(876,3,'re-inspect')]:
        tile(x,k); text(x+48,392,lab,12.5,False,'middle',MUTE)
    arrow([(760,348),(768,348)],head=6); arrow([(866,348),(874,348)],head=6)
    # findings
    box(480,348,120,36,GREEN,'findings JSON',ts=13)
    arrow([(540,330),(540,348)],head=7)
    add('</g>')

    # ===================== SERVING EXTENSION =====================
    add('<g id="serving">')
    rect(316,440,350,132,PURPLE,rx=14,dash='7 4')
    text(332,464,'Extension 2: speculative decoding',14,True,'start','#5B3F86',maxw=320)
    box(332,478,96,70,WHITE,'MTP draft','fine-tuned','on target',ts=13.5,ss=12.5)
    arrow([(428,513),(440,513)],head=6)
    toks=[('A','#CFEAD7','#3E8E5A'),('B','#CFEAD7','#3E8E5A'),('C','#F6D2D0','#C0392B'),('D','#EEEEF0','#9AA0A8')]
    for i,(t,fc,sc) in enumerate(toks):
        x=442+i*29
        dash='3 2' if t=='D' else None
        rect(x,498,26,30,(fc,sc),rx=5,sw=1.5,dash=dash)
        text(x+13,518,t,13.5,True,'middle',sc)
    arrow([(560,513),(574,513)],head=6)
    box(574,478,80,70,WHITE,'Target','(policy)','verifies',ts=13.5,ss=12.5)
    text(500,562,'keep A, B; target corrects C',12.5,False,'middle',MUTE,maxw=200)
    add('</g>')

    # ===================== EVALUATION =====================
    add('<g id="evaluation">')
    rect(682,440,302,132,GRAY,rx=14)
    text(698,464,'Evaluation',14,True)
    lines=['Test: DENTEX validation, 50 images, once','Dev: ~70 training images, for selection','Pooled set-matched F1, bootstrap CIs','Probes: tool removal, box displacement']
    for i,l in enumerate(lines):
        y=488+i*21
        add(f'<circle cx="702" cy="{y-4}" r="2.4" fill="{MUTE}"/>')
        text(712,y,l,13,False,'start',INK,maxw=262)
    add('</g>')


    # ===================== CROSS-PANEL ARROWS (drawn last) =====================
    add('<g id="links">')
    # adapters -> policy
    arrow([(284,552),(308,552),(308,300),(440,300)],col='#3E8E5A',sw=1.8,dash='6 4')
    text(324,293,'adapters',12.5,False,'start','#2F7048',maxw=90)
    arrow([(470,110),(470,190)],col='#7B5EA7')
    text(478,160,'prompt',12.5,False,'start','#5B3F86')
    arrow([(580,110),(580,190)],col='#7B5EA7',both=True)
    text(588,160,'read_skill',12.5,False,'start','#5B3F86')
    arrow([(460,440),(460,330)],col='#7B5EA7',dash='6 4')
    text(468,402,'decode',12.5,False,'start','#5B3F86')
    arrow([(590,384),(590,424),(740,424),(740,440)])
    add('</g>')
    # ===================== LEGEND =====================
    add('<g id="legend">')
    def sq(x,col,dash=None): rect(x,586,14,14,col,rx=3,sw=1.4,dash=dash)
    items=[(316,BLUE,'data / model',None),(432,ORANGE,'our contribution',None),(568,GREEN,'verified output',None),(699,PURPLE,'optional extension','3 2')]
    for x,c,l,d in items:
        sq(x,c,d); text(x+20,598,l,12.5,False,'start',MUTE)
    flame(855,593,0.9); text(866,598,'trained',12.5,False,'start',MUTE)
    snow(934,593,6.5); text(946,598,'frozen',12.5,False,'start',MUTE)
    add('</g>')

    add('</svg>')
    write(path)
