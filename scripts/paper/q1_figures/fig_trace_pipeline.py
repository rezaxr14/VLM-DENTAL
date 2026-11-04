"""Figure 3: trace synthesis, verification and repair, decontamination and release."""
from common import *
import sys


def build(path):
    out.clear(); warns.clear()
    def sqcrop(path,where='center'):
        im=Image.open(path).convert('RGB'); w,h=im.size; m=min(w,h)
        x0=(w-m)//2
        y0={'top':0,'center':(h-m)//2}[where]
        return im.crop((x0,y0,x0+m,y0+m)).resize((96,96),Image.LANCZOS)
    def tracecard(x,yb,strip,flag=False):
        # small glyph of a trace: card with coloured header strip and text lines
        add(f'<rect x="{x}" y="{yb-34}" width="24" height="34" rx="3" fill="#FFFFFF" stroke="#8A8F98" stroke-width="1.3"/>')
        add(f'<rect x="{x}" y="{yb-34}" width="24" height="8" rx="3" fill="{strip}"/>')
        add(f'<g stroke="#C9CCD3" stroke-width="2"><line x1="{x+5}" y1="{yb-20}" x2="{x+19}" y2="{yb-20}"/><line x1="{x+5}" y1="{yb-13}" x2="{x+19}" y2="{yb-13}"/><line x1="{x+5}" y1="{yb-6}" x2="{x+14}" y2="{yb-6}"/></g>')
    def badge(x,y,n,col=BL):
        add(f'<circle cx="{x}" cy="{y}" r="9" fill="{col}"/>'); text(x,y+4.5,str(n),12.5,True,'middle','#FFFFFF')

    W,H=1000,760
    out.clear()
    add(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Helvetica, Arial, \'Liberation Sans\', sans-serif">')
    add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')
    rect(16,16,968,224,GRAY,rx=14); rect(16,256,540,488,GRAY,rx=14); rect(576,256,408,488,GRAY,rx=14)
    add('<polygon points="490,242 510,242 500,254" fill="#C3C7CF"/>')
    add('<polygon points="559,498 566,498 573,520 566,542 559,542 566,520" fill="#C3C7CF"/>')

    # ================= ROW A: GENERATE =================
    add('<g id="s1-generate">')
    circ_num(36,36,1); text(54,42,'Generate demonstrations',15,True,maxw=300)
    opg=Image.open(SC+'flagship/image.jpg').convert('RGB').resize((524,275),Image.LANCZOS)
    OX,OY,OW,OH=32,58,262,137
    image(OX,OY,OW,OH,opg); frame(OX,OY,OW,OH,'#6B7280',1.2)
    gt=json.load(open(SC+'showcase.json'))['flagship']['ground_truth']
    col={'Impacted':PU,'Deep Caries':'#E5484D','Caries':'#F2C20E'}
    for g in gt:
        x,y,w,h=g['bbox']; k=OW/2869
        add(f'<rect x="{OX+x*k:.1f}" y="{OY+y*k:.1f}" width="{w*k:.1f}" height="{h*k:.1f}" fill="none" stroke="#000000" stroke-opacity="0.55" stroke-width="3"/>')
        add(f'<rect x="{OX+x*k:.1f}" y="{OY+y*k:.1f}" width="{w*k:.1f}" height="{h*k:.1f}" fill="none" stroke="{col[g["diagnosis"]]}" stroke-width="1.8"/>')
    add('<rect x="38" y="63" width="78" height="17" rx="3" fill="#FFFFFF" fill-opacity="0.88"/>'); text(43,76,'expert labels',12,True,'start',INK)
    legend_item(34,216,PU,'impacted',12.5); legend_item(118,216,'#E5484D','deep caries',12.5); legend_item(212,216,'#F2C20E','caries',12.5)
    arrow([(298,126),(322,126)],head=8)
    text(332,70,'Teacher loop with real tools',13.5,True,maxw=290)
    cx,cy,R=420,142,38
    for a,_ in [(-90,0),(30,0),(150,0)]:
        add(f'<circle cx="{cx+R*math.cos(math.radians(a)):.1f}" cy="{cy+R*math.sin(math.radians(a)):.1f}" r="16" fill="#FFFFFF" stroke="{BL}" stroke-width="1.8"/>')
    arc_arrow(cx,cy,R,-66,6,BL,2); arc_arrow(cx,cy,R,54,126,BL,2); arc_arrow(cx,cy,R,174,246,BL,2)
    bx,by=cx,cy-R
    add(f'<ellipse cx="{bx}" cy="{by-2}" rx="9" ry="6.5" fill="none" stroke="{BL}" stroke-width="1.6"/><circle cx="{bx-6}" cy="{by+8}" r="2" fill="{BL}"/><circle cx="{bx-10}" cy="{by+12}" r="1.2" fill="{BL}"/>')
    gx,gy=cx+R*math.cos(math.radians(30)),cy+R*math.sin(math.radians(30))
    add(f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="5" fill="none" stroke="{BL}" stroke-width="1.8"/>')
    for a in range(0,360,45):
        ax,ay=math.cos(math.radians(a)),math.sin(math.radians(a))
        add(f'<line x1="{gx+ax*7:.1f}" y1="{gy+ay*7:.1f}" x2="{gx+ax*10:.1f}" y2="{gy+ay*10:.1f}" stroke="{BL}" stroke-width="2.4" stroke-linecap="round"/>')
    ex,ey=cx+R*math.cos(math.radians(150)),cy+R*math.sin(math.radians(150))
    add(f'<path d="M{ex-10:.1f},{ey:.1f} Q{ex:.1f},{ey-9:.1f} {ex+10:.1f},{ey:.1f} Q{ex:.1f},{ey+9:.1f} {ex-10:.1f},{ey:.1f} Z" fill="none" stroke="{BL}" stroke-width="1.6"/><circle cx="{ex:.1f}" cy="{ey:.1f}" r="3" fill="{BL}"/>')
    text(cx+22,cy-R+5,'thought',12.5,False,'start',MUTE)
    text(gx+22,gy+5,'tool call',12.5,False,'start',MUTE)
    text(ex,ey+34,'observation',12.5,False,'middle',MUTE)
    text(536,120,'box perturbation',12.5,True,'start','#9A5410',maxw=116)
    x=536
    for fr,fc,tc,lab in [(0.45,'#E9E4DA','#5B6472','45'),(0.25,'#F4B77A','#4A2A06','25'),(0.30,'#E07B1F','#FFFFFF','30')]:
        w=112*fr; add(f'<rect x="{x:.1f}" y="128" width="{w:.1f}" height="18" fill="{fc}" stroke="#FFFFFF" stroke-width="1"/>'); text(x+w/2,141,lab,12,True,'middle',tc); x+=w
    text(536,164,'on the shown box',12.5,False,'start',MUTE,italic=True,maxw=116)
    add(f'<path d="M{gx+16:.1f},{gy-6:.1f} C{gx+40:.1f},{gy-22:.1f} 510,150 532,142" fill="none" stroke="{OR}" stroke-width="1.6" stroke-dasharray="4 3"/>')
    text(672,70,'real tool outputs on one crop',12.5,True,'start',INK,maxw=290)
    tools=[(0,('zoom_','crop')),(1,('window_','level')),(5,('enhance_','contrast')),(6,('denoise',''))]
    for i,(t,(l1,l2)) in enumerate(tools):
        im=Image.open(SC+f'params_dentex/tool_{t}.jpg').convert('RGB')
        tx=672+i*74; image(tx,80,58,108,im); frame(tx,80,58,108,'#6B7280',1)
        text(tx+29,205,l1,12.5,False,'middle',INK,maxw=70)
        if l2: text(tx+29,219,l2,12.5,False,'middle',INK,maxw=70)
    add('</g>')

    # ================= ROW B / STAGE 2: VERIFY AND REPAIR =================
    add('<g id="s2-verify">')
    circ_num(36,280,2); text(54,286,'Verify and repair',15,True,maxw=300)
    rect(32,300,236,432,WHITE,rx=10,sw=1.5)
    text(44,319,'Trace: DENTEX image 624',12.5,True,'start',INK,maxw=210)
    text(44,334,'23 turns · 22 tool calls',12,False,'start',MUTE)
    rows=[('T0','window_level','survey the full field',BL,None),
          ('T1','fdi_label','orient to FDI notation',BL,None),
          ('T2','locate_tooth(34)','box [1668, 917, 84, 322]',BL,('crop_located_624.png','center')),
          ('T3','zoom_crop','“only bone and root … too low”',BL,None),
          ('T4','nudge_crop','dy −0.3 · scale 0.6',OR,None),
          ('T6','nudge_crop','dy −0.5 · scale 0.7',OR,None),
          ('T8','nudge_crop','dy −0.6 · scale 0.8',OR,None),
          ('T9–10','zoom_crop + denoise','“… clearly see tooth 34”',BL,('crop_final_624.png','top')),
          ('T14+','compare + enhance','contralateral, contrast',BL,('canon_compare_624.png','center')),
          ('T22','final answer','Q3 T4 · Caries · 0.82',GR,None)]
    for i,(tl,tool,sub,c,th) in enumerate(rows):
        y0=344+i*38
        if i>0: add(f'<line x1="40" y1="{y0-1}" x2="260" y2="{y0-1}" stroke="#EEF0F3" stroke-width="1.2"/>')
        text(42,y0+16,tl,11.5,True,'start',MUTE,maxw=36)
        wch=tw(tool,11.5,True)+12
        fill = {'#3B6BB5':'#E8F0FB','#D9822B':'#FDEBD9','#3E8E5A':'#E4F3E9'}[c]
        add(f'<rect x="82" y="{y0+5}" width="{wch:.1f}" height="17" rx="8" fill="{fill}" stroke="{c}" stroke-width="1.2"/>')
        text(82+wch/2,y0+17,tool,11.5,True,'middle',INK)
        text(82,y0+34,sub,11.5,False,'start',MUTE,maxw=136 if th else 178)
        if th:
            image(222,y0+2,34,34,sqcrop(A624+th[0],th[1])); frame(222,y0+2,34,34,'#8A8F98',1)
    # badges linking trace rows to the checks
    badge(268,344+2*38+19,1); badge(268,344+3*38+19,2)
    add(f'<path d="M264,{344+4*38+5} h7 v{3*38-10} h-7" fill="none" stroke="{BL}" stroke-width="1.8"/>'); badge(278,344+5*38+19,3)
    badge(268,344+9*38+19,4)
    checks=[('Format','valid FDI notation and JSON'),('No leakage','no leaked generator directive'),('Evidence','claims backed by shown crops'),('Diagnosis','matches the expert label')]
    for i,(t,sub) in enumerate(checks):
        cy_=300+i*50
        rect(300,cy_,240,44,WHITE,rx=8,sw=1.3)
        badge(318,cy_+22,i+1); text(336,cy_+19,t,13,True,'start',INK,maxw=190); text(336,cy_+35,sub,12,False,'start',MUTE,maxw=196)
    # pass / reject split
    text(540,518,'97.5% pass first time',12,True,'end',GR)
    arrow([(444,534),(548,534)],col=GR,sw=2.2,head=8)
    arrow([(326,500),(326,546)],col=RD,sw=2.2,head=8); text(318,526,'reject',12.5,True,'end',RD)
    cb=Image.open(SC+'flagship/crop_before.jpg').convert('RGB'); ca=Image.open(SC+'flagship/crop_after.jpg').convert('RGB')
    image(300,550,52,62,cb); frame(300,550,52,62,RD,2.2); text(326,630,'drifted',12.5,False,'middle',RD)
    arrow([(356,581),(384,581)],head=7)
    image(388,550,52,62,ca); frame(388,550,52,62,GR,2.2); text(414,630,'repaired',12.5,False,'middle',GR)
    for i,t in enumerate(['editor repairs','the trace, keeps','the valid','reasoning']): text(450,568+i*15,t,12,False,'start',INK,maxw=92)
    arrow([(404,546),(404,498)],col='#4B5563',sw=1.8,head=7); text(396,526,'re-verify',12,False,'end',MUTE)
    text(300,658,'Why traces are rejected',13,True,'start',INK,maxw=240)
    x=300
    for n,c in [(66,'#C0392B'),(15,'#E0A030'),(11,'#E07B1F')]:
        w=240*n/92; add(f'<rect x="{x:.1f}" y="668" width="{w:.1f}" height="14" fill="{c}" stroke="#FFFFFF" stroke-width="1"/>'); x+=w
    for i,(c,l) in enumerate([('#C0392B','66  invented findings'),('#E0A030','15  label contradictions'),('#E07B1F','11  uncorrected drifts')]):
        legend_item(300,704+i*17,c,l,12,maxw=240)
    add('</g>')

    # ================= STAGE 3: DECONTAMINATE AND RELEASE =================
    add('<g id="s3-release">')
    circ_num(596,280,3); text(614,286,'Decontaminate and release',15,True,maxw=360)
    def gate(gx_,l1,l2):
        add(f'<rect x="{gx_-20}" y="334" width="40" height="58" fill="#DCE8FA" fill-opacity="0.75"/>')
        add(f'<g stroke="#9DB8E0" stroke-width="1.2"><line x1="{gx_-20}" y1="348" x2="{gx_+20}" y2="348"/><line x1="{gx_-20}" y1="362" x2="{gx_+20}" y2="362"/><line x1="{gx_-20}" y1="376" x2="{gx_+20}" y2="376"/></g>')
        add(f'<rect x="{gx_-28}" y="328" width="8" height="64" rx="2" fill="{BL}"/><rect x="{gx_+20}" y="328" width="8" height="64" rx="2" fill="{BL}"/><rect x="{gx_-28}" y="322" width="56" height="9" rx="3" fill="{BL}"/>')
    gate(720,'regex sweep','')
    gate(870,'re-verify','')
    text(720,300,'regex sweep',12.5,True,'middle',INK); text(720,314,'leaked directives',12,False,'middle',MUTE,maxw=130)
    text(870,300,'re-verify flagged',12.5,True,'middle',INK,maxw=130); text(870,314,'traces (family B)',12,False,'middle',MUTE,maxw=130)
    arrow([(592,392),(962,392)],col='#8A8F98',sw=2.4,head=9)
    for xx in (610,636,662): tracecard(xx,392,BL)
    tracecard(766,392,BL); tracecard(800,392,RD)
    tracecard(920,392,GR)
    text(812,342,'flagged',12,True,'middle',RD)
    add(f'<path d="M870,400 C870,436 780,446 700,446 C650,446 602,436 600,404" fill="none" stroke="{RD}" stroke-width="1.8" stroke-dasharray="5 4"/>')
    add(f'<polygon points="600,396 594,408 606,408" fill="{RD}"/>')
    text(740,464,'excise and regenerate',12,False,'middle',RD,italic=True)
    text(592,484,'Released corpus: 3,694 traces',14,True,'start',INK,maxw=370)
    CLS={'Impacted':PU,'Deep Caries':'#E5484D','Caries':'#F2C20E','Periapical Lesion':'#2E9BD6'}
    meta=json.load(open(SC+'showcase.json'))
    cells=[('breadth','DENTEX · 15 findings','#3B6BB5'),('deep_caries','DENTEX · 13 findings','#3B6BB5'),('flagship','DENTEX · 9 findings','#3B6BB5'),
           ('healthy','DENTEX · healthy scan','#9DB8E0'),('tufts','Tufts · 3 findings','#2A8C86'),('params_tufts','Tufts · 3 findings','#2A8C86')]
    TW,TH=118,70
    for i,(key,cap,fc) in enumerate(cells):
        gx0=592+(i%3)*129; gy0=494+(i//3)*92
        im=Image.open(SC+f'{key}/image.jpg').convert('RGB'); w,h=im.size
        if w/h>TW/TH: cw,ch=int(TW/TH*h),h
        else: cw,ch=w,int(w*TH/TW)
        x0,y0=(w-cw)//2,(h-ch)//2
        im=im.crop((x0,y0,x0+cw,y0+ch)).resize((TW*3,TH*3),Image.LANCZOS)
        image(gx0,gy0,TW,TH,im)
        m=meta[key]
        if m.get('native_w'):
            r=w/m['native_w']; k=TW/cw
            for g in m['ground_truth']:
                bx,by,bw_,bh=g['bbox']; px=gx0+(bx*r-x0)*k; py=gy0+(by*r-y0)*k; pw=bw_*r*k; ph=bh*r*k
                c=CLS.get(g['diagnosis'],'#9AA0A8')
                add(f'<rect x="{px:.1f}" y="{py:.1f}" width="{pw:.1f}" height="{ph:.1f}" fill="none" stroke="#000000" stroke-opacity="0.55" stroke-width="2.2"/>')
                add(f'<rect x="{px:.1f}" y="{py:.1f}" width="{pw:.1f}" height="{ph:.1f}" fill="none" stroke="{c}" stroke-width="1.3"/>')
        frame(gx0,gy0,TW,TH,fc,2.2)
        text(gx0+TW/2,gy0+TH+14,cap,11.5,False,'middle',INK,maxw=TW+8)
    kx=592
    for nm,c in [('impacted',PU),('deep caries','#E5484D'),('caries','#F2C20E'),('periapical','#2E9BD6')]:
        legend_item(kx,684,c,nm,12); kx+=16+tw(nm,12)+12
    rect(592,694,376,42,WHITE,rx=8,sw=1.3)
    text(602,711,'with tools',12.5,False,'start',INK); text(602,727,'without tools',12.5,False,'start',INK)
    add(f'<line x1="706" y1="707" x2="906" y2="707" stroke="{BL}" stroke-width="1.6"/>')
    for i in range(12): add(f'<circle cx="{706+i*200/11:.1f}" cy="707" r="3.6" fill="{BL if i%2==0 else "#9DB8E0"}" stroke="#FFFFFF" stroke-width="0.8"/>')
    add(f'<circle cx="706" cy="723" r="3.6" fill="{BL}"/>'); text(718,727,'single turn',12,False,'start',MUTE,italic=True)
    text(958,711,'1,847',12.5,True,'end',INK); text(958,727,'1,847',12.5,True,'end',INK)
    add('</g>')
    finish(path)
