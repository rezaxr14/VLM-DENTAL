"""Figure 2: nudge_crop on a real episode, the perturbation roll, and the agent loop."""
from common import *
import sys


def build(path):
    out.clear(); warns.clear()
    A=SC+'flagship/'
    ctx=Image.open(A+'context.jpg').convert('RGB')          # 439x469, scale 0.6275 of native
    crop_before=Image.open(A+'crop_before.jpg').convert('RGB')
    crop_after=Image.open(A+'crop_after.jpg').convert('RGB')
    def b64(im,q=88):
        b=io.BytesIO(); im.save(b,'JPEG',quality=q); return base64.b64encode(b.getvalue()).decode()
    def image(x,y,w,h,im,name=''):
        add(f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="none" xlink:href="data:image/jpeg;base64,{b64(im)}"/>')
    def frame(x,y,w,h,col,sw=2.4): add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{col}" stroke-width="{sw}"/>')
    def chiptext(x,y,s,size=12.5,bold=False,fill=INK,anchor='start',pad=4):
        w=tw(s,size,bold); x0=x if anchor=='start' else (x-w/2 if anchor=='middle' else x-w)
        add(f'<rect x="{x0-pad:.1f}" y="{y-size:.1f}" width="{w+2*pad:.1f}" height="{size+6:.1f}" rx="3" fill="#FFFFFF" fill-opacity="0.88"/>')
        text(x,y,s,size,bold,anchor,fill)
    BL='#3B6BB5'; OR='#D9822B'; GR='#3E8E5A'
    REF='#22C55E'
    def refbox(x,y,w,h,pad,sw):
        x-=pad;y-=pad;w+=2*pad;h+=2*pad
        add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" stroke="#06210F" stroke-opacity="0.65" stroke-width="{sw+2.4}" stroke-dasharray="8 5"/>')
        add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" stroke="{REF}" stroke-width="{sw}" stroke-dasharray="8 5"/>')


    # native -> context px (0.6275) -> display
    S=0.6275
    def ctx_box(nb): return ((nb[0]-503.3)*S,(nb[1]-462.3)*S,nb[2]*S,nb[3]*S)
    GT=ctx_box((669,793,230,276)); SHOWN=ctx_box((807.3,602.3,230,276)); CORR=ctx_box((676.2,792.7,230,276))

    W,H=1000,556
    out.clear(); 
    add(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Helvetica, Arial, \'Liberation Sans\', sans-serif">')
    add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

    # ===================== (a) real episode geometry =====================
    add('<g id="a-geometry">')
    rect(16,16,356,524,GRAY,rx=14)
    text(32,41,'(a) nudge_crop on a real episode',15,True,maxw=330)
    X0,Y0,K=34,60,320/439
    image(X0,Y0,320,342,ctx); frame(X0,Y0,320,342,'#8A8F98',1.2)
    def d(b): return (X0+b[0]*K, Y0+b[1]*K, b[2]*K, b[3]*K)
    g=d(GT); s=d(SHOWN); c=d(CORR)
    refbox(g[0],g[1],g[2],g[3],4,3.2)
    add(f'<rect x="{s[0]:.1f}" y="{s[1]:.1f}" width="{s[2]:.1f}" height="{s[3]:.1f}" fill="none" stroke="{BL}" stroke-width="2.8"/>')
    add(f'<rect x="{c[0]:.1f}" y="{c[1]:.1f}" width="{c[2]:.1f}" height="{c[3]:.1f}" fill="none" stroke="{OR}" stroke-width="2.8"/>')
    sc=(s[0]+s[2]/2,s[1]+s[3]/2); cc=(c[0]+c[2]/2,c[1]+c[3]/2)
    arrow([sc,cc],col=OR,sw=3,head=11)
    add(f'<circle cx="{sc[0]:.1f}" cy="{sc[1]:.1f}" r="3.6" fill="{BL}" stroke="#FFFFFF" stroke-width="1.2"/>')
    # w / h dimension lines on the shown box
    ty=s[1]-9
    add(f'<g stroke="#FFFFFF" stroke-width="1.6"><line x1="{s[0]:.1f}" y1="{ty}" x2="{s[0]+s[2]:.1f}" y2="{ty}"/><line x1="{s[0]:.1f}" y1="{ty-5}" x2="{s[0]:.1f}" y2="{ty+5}"/><line x1="{s[0]+s[2]:.1f}" y1="{ty-5}" x2="{s[0]+s[2]:.1f}" y2="{ty+5}"/></g>')
    chiptext(s[0]+s[2]/2,ty-6,'w',13,True,INK,'middle')
    rx_=s[0]+s[2]+9
    add(f'<g stroke="#FFFFFF" stroke-width="1.6"><line x1="{rx_:.1f}" y1="{s[1]:.1f}" x2="{rx_:.1f}" y2="{s[1]+s[3]:.1f}"/><line x1="{rx_-5:.1f}" y1="{s[1]:.1f}" x2="{rx_+5:.1f}" y2="{s[1]:.1f}"/><line x1="{rx_-5:.1f}" y1="{s[1]+s[3]:.1f}" x2="{rx_+5:.1f}" y2="{s[1]+s[3]:.1f}"/></g>')
    chiptext(rx_+8,s[1]+s[3]/2+5,'h',13,True,INK,'start')
    chiptext(250,300,'dx  −0.57 · w',12.5,True,OR)
    chiptext(250,320,'dy  +0.69 · h',12.5,True,OR)
    chiptext(g[0]+g[2]/2,g[1]+g[3]+22,'tooth 48, impacted',12.5,False,INK,'middle')
    # legend
    def key(x,col,dash,label):
        ds=' stroke-dasharray="4 3"' if dash else ''
        add(f'<rect x="{x}" y="418" width="18" height="12" fill="none" stroke="{col}" stroke-width="{3 if dash else 2.2}"{ds}/>')
        text(x+24,429,label,12.5,False,'start',MUTE)
    key(36,REF,True,'reference'); key(138,BL,False,'shown'); key(220,OR,False,'corrected')
    text(34,456,'shift, rescale, clamp; returns coordinates only',13,False,'start',INK,maxw=320)
    rect(34,474,320,34,WHITE,rx=6,sw=1.3)
    add(f'<text x="44" y="496" font-size="11.5" font-family="Courier, \'Liberation Mono\', monospace" fill="{INK}">nudge_crop(bbox, dx_frac, dy_frac, scale)</text>')
    add('</g>')

    # ===================== (b) perturbation roll =====================
    add('<g id="b-roll">')
    rect(388,16,596,236,GRAY,rx=14)
    text(404,41,'(b) Perturbation roll: only the shown box changes',15,True,maxw=570)
    cx,cy,R,r=450,140,50,28
    cols=[('#CFCBC0','none'),('#F4B77A','small'),('#E07B1F','big')]
    fr=[0.45,0.25,0.30]; a=-math.pi/2
    for f,(col,_) in zip(fr,cols):
        b=a+2*math.pi*f; lg=1 if f>0.5 else 0
        p=lambda ang,rad:(cx+rad*math.cos(ang),cy+rad*math.sin(ang))
        x0,y0=p(a,R); x1,y1=p(b,R); x2,y2=p(b,r); x3,y3=p(a,r)
        add(f'<path d="M{x0:.1f},{y0:.1f} A{R},{R} 0 {lg} 1 {x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f} A{r},{r} 0 {lg} 0 {x3:.1f},{y3:.1f} Z" fill="{col}" stroke="#FFFFFF" stroke-width="2"/>')
        a=b
    text(cx,cy+5,'roll',13,True,'middle',MUTE)
    text(cx,212,'fixed rates',12.5,False,'middle',MUTE,italic=True)
    text(cx,229,'no verbal cue',12.5,False,'middle',MUTE,italic=True)
    arrow([(506,140),(524,140)],head=7)
    RX0,RY0,RW,RH=60,70,300,350
    region=ctx.crop((RX0,RY0,RX0+RW,RY0+RH)).resize((240,280),Image.LANCZOS)
    tiers=[('none 45%','no offset','#9C978A',(0,0)),('small 25%','12–28% offset','#D9822B',(0.22,-0.17)),('big 30%','45–75% offset','#C46A10',None)]
    for i,(lab,sub,col,off) in enumerate(tiers):
        tx=532+i*144; ty0=62
        image(tx,ty0,120,140,region); frame(tx,ty0,120,140,col,2.6)
        kk=120/RW
        g_=((GT[0]-RX0)*kk,(GT[1]-RY0)*kk,GT[2]*kk,GT[3]*kk)
        if off is None: sb=((SHOWN[0]-RX0)*kk,(SHOWN[1]-RY0)*kk,SHOWN[2]*kk,SHOWN[3]*kk)
        else: sb=(g_[0]+off[0]*g_[2],g_[1]+off[1]*g_[3],g_[2],g_[3])
        refbox(tx+g_[0],ty0+g_[1],g_[2],g_[3],2.5,2.4)
        add(f'<rect x="{tx+sb[0]:.1f}" y="{ty0+sb[1]:.1f}" width="{sb[2]:.1f}" height="{sb[3]:.1f}" fill="none" stroke="{BL}" stroke-width="2.4"/>')
        text(tx+60,222,lab,13,True,'middle',INK,maxw=140)
        text(tx+60,239,sub,12.5,False,'middle',MUTE,maxw=140)
    add('</g>')

    # ===================== (c) agent loop =====================
    add('<g id="c-agent">')
    rect(388,264,596,276,GRAY,rx=14)
    text(404,285,'(c) Agent: inspect, judge, correct',15,True,maxw=400)
    IY=330; IW,IH=130,156
    image(404,IY,IW,IH,crop_before); frame(404,IY,IW,IH,BL,3)
    image(672,IY,IW,IH,crop_after); frame(672,IY,IW,IH,OR,3)
    text(469,IY+IH+19,'zoom_crop: shown box',12.5,False,'middle',BL,maxw=150)
    text(737,IY+IH+19,'zoom_crop: after nudge',12.5,False,'middle',OR,maxw=150)
    # loop arc
    add(f'<path d="M737,{IY-4} C737,{IY-30} 469,{IY-30} 469,{IY-4}" fill="none" stroke="#8A8F98" stroke-width="1.8" stroke-dasharray="5 4"/>')
    add(f'<polygon points="469,{IY-3} 463,{IY-13} 475,{IY-13}" fill="#8A8F98"/>')
    text(603,IY-27,'repeat until the box is acceptable',12.5,False,'middle',MUTE,italic=True,maxw=240)
    # speech bubble
    add('<path d="M556,346 h92 a8,8 0 0 1 8,8 v38 a8,8 0 0 1 -8,8 h-92 a8,8 0 0 1 -8,-8 v-14 l-12,-8 l12,-4 v-12 a8,8 0 0 1 8,-8 z" fill="#FFFFFF" stroke="#8A8F98" stroke-width="1.4"/>')
    text(602,366,'“about one arch',12.5,False,'middle',INK,italic=True)
    text(602,382,'too high”',12.5,False,'middle',INK,italic=True)
    # nudge arrow + badge
    arrow([(540,424),(664,424)],col=OR,sw=2.6,head=9)
    rect(558,410,90,26,ORANGE,rx=13,sw=1.6)
    text(603,428,'nudge_crop',13,True,'middle',INK)
    text(603,452,'dx −0.57 · dy +0.69',12.5,False,'middle',MUTE,maxw=130)
    text(603,470,'IoU 0.07 → 0.94',12.5,True,'middle',GR)
    # commit
    arrow([(802,408),(832,408)],col=GR,sw=2.4,head=8)
    rect(832,372,148,72,GREEN,rx=12,sw=1.8)
    add(f'<circle cx="858" cy="408" r="11" fill="{GR}"/><path d="M852,408 l5,6 l9,-11" fill="none" stroke="#FFFFFF" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>')
    text(878,404,'commit',14,True,'start',INK)
    text(878,422,'Q4 T8 · impacted',12.5,False,'start',MUTE,maxw=98)
    add('</g>')
    finish(path)
