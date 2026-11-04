"""Shared drawing helpers for the Q1-style paper figures.

Every figure is hand-structured SVG (editable in Inkscape/Illustrator) produced
by plain Python: palette, text with a width check, boxes, arrows with explicit
arrowhead polygons (so svglib converts them to PDF faithfully) and embedded
real radiographs from paper/figures/assets.
"""
import base64, io, json, math
from pathlib import Path
from PIL import Image, ImageFont

ROOT = Path(__file__).resolve().parents[3]
SC = str(ROOT / "paper" / "figures" / "assets" / "showcase") + "/"
A624 = str(ROOT / "paper" / "figures" / "assets") + "/"

_FONT_CANDIDATES = {
    "n": ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "b": ["/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
}
FONT = {k: next((p for p in v if Path(p).exists()), None) for k, v in _FONT_CANDIDATES.items()}


def tw(s, size, bold=False):
    """Approximate rendered text width (px) used for the fit checks."""
    path = FONT["b" if bold else "n"]
    if path is None:
        return len(s) * size * 0.52
    return ImageFont.truetype(path, 100).getlength(s) * size / 100

INK='#1F2937'; MUTE='#5B6472'
BLUE=('#E8F0FB','#3B6BB5'); ORANGE=('#FDEBD9','#D9822B'); GREEN=('#E4F3E9','#3E8E5A')
PURPLE=('#F1EBF9','#7B5EA7'); GRAY=('#F5F5F7','#B9BBC2'); WHITE=('#FFFFFF','#B9BBC2')
out=[]; warns=[]
def add(s): out.append(s)
def esc(s): return s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

def rect(x,y,w,h,col,rx=8,sw=1.6,dash=None,extra=''):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{col[0]}" stroke="{col[1]}" stroke-width="{sw}"{d}{extra}/>')
def text(x,y,s,size=14,bold=False,anchor='start',fill=INK,italic=False,maxw=None):
    if maxw is not None and tw(s,size,bold) > maxw:
        warns.append(f'TEXT TOO WIDE ({tw(s,size,bold):.0f}>{maxw}): {s}')
    st = ' font-style="italic"' if italic else ''
    wt = ' font-weight="bold"' if bold else ''
    add(f'<text x="{x}" y="{y}" font-size="{size}"{wt}{st} text-anchor="{anchor}" fill="{fill}">{esc(s)}</text>')
def box(x,y,w,h,col,title,sub=None,sub2=None,ts=14,ss=13,rx=8,dash=None):
    rect(x,y,w,h,col,rx=rx,dash=dash)
    cx=x+w/2
    if sub is None:
        text(cx,y+h/2+5,title,ts,True,'middle',maxw=w-10)
    else:
        n = 2 + (1 if sub2 else 0)
        t0 = y + (h - (ts + (n-1)*ss*1.25))/2 + ts*0.82
        text(cx,t0,title,ts,True,'middle',maxw=w-10)
        text(cx,t0+ts*0.2+ss*1.15,sub,ss,False,'middle',MUTE,maxw=w-10)
        if sub2: text(cx,t0+ts*0.2+ss*2.4,sub2,ss,False,'middle',MUTE,maxw=w-10)
def arrow(pts,col='#4B5563',sw=1.8,dash=None,both=False,head=8):
    d='M'+' L'.join(f'{x},{y}' for x,y in pts)
    ds=f' stroke-dasharray="{dash}"' if dash else ''
    # shorten line end so the head sits at the endpoint
    def tip(p_from,p_to):
        dx,dy=p_to[0]-p_from[0],p_to[1]-p_from[1]; L=math.hypot(dx,dy); ux,uy=dx/L,dy/L
        bx,by=p_to[0]-ux*head,p_to[1]-uy*head; nx,ny=-uy,ux
        return (bx,by),f'{p_to[0]},{p_to[1]} {bx+nx*head*0.42:.1f},{by+ny*head*0.42:.1f} {bx-nx*head*0.42:.1f},{by-ny*head*0.42:.1f}'
    pts2=list(pts)
    b,poly_end=tip(pts[-2],pts[-1]); pts2[-1]=b
    heads=[poly_end]
    if both:
        b0,poly_start=tip(pts[1],pts[0]); pts2[0]=b0; heads.append(poly_start)
    d='M'+' L'.join(f'{x:.1f},{y:.1f}' for x,y in pts2)
    add(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{sw}"{ds} stroke-linejoin="round"/>')
    for h in heads: add(f'<polygon points="{h}" fill="{col}"/>')
def flame(cx,cy,s=1.0):
    add(f'<path transform="translate({cx},{cy}) scale({s})" d="M0,-8 C5,-3 6,2 0,8 C-6,2 -5,-3 0,-8 Z" fill="#E4572E"/>')
    add(f'<path transform="translate({cx},{cy}) scale({s})" d="M0,-2 C2.5,0 3,3 0,6 C-3,3 -2.5,0 0,-2 Z" fill="#FFC857"/>')
def snow(cx,cy,r=7,col='#3B6BB5'):
    for a in (0,60,120):
        dx,dy=r*math.cos(math.radians(a)),r*math.sin(math.radians(a))
        add(f'<line x1="{cx-dx:.1f}" y1="{cy-dy:.1f}" x2="{cx+dx:.1f}" y2="{cy+dy:.1f}" stroke="{col}" stroke-width="1.8" stroke-linecap="round"/>')
def cylinder(cx,cy,w=22,h=24,col='#3B6BB5'):
    add(f'<g fill="none" stroke="{col}" stroke-width="1.6"><ellipse cx="{cx}" cy="{cy-h/2+4}" rx="{w/2}" ry="4"/>'
        f'<path d="M{cx-w/2},{cy-h/2+4} V{cy+h/2-4} A{w/2},4 0 0 0 {cx+w/2},{cy+h/2-4} V{cy-h/2+4}"/>'
        f'<path d="M{cx-w/2},{cy} A{w/2},4 0 0 0 {cx+w/2},{cy}"/></g>')


def begin(W,H):
    out.clear(); warns.clear()
    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Helvetica, Arial, \'Liberation Sans\', sans-serif">')
    add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')
def finish(path):
    add('</svg>')
    write(path)
def chip(x,y,w,h,col,left,right,lc=INK,rc=MUTE):
    rect(x,y,w,h,col,rx=6,sw=1.4)
    text(x+10,y+h/2+5,left,13,True,'start',lc)
    text(x+w-10,y+h/2+5,right,12.5,False,'end',rc)
def diamond(cx,cy,hw,hh,col,label):
    add(f'<polygon points="{cx},{cy-hh} {cx+hw},{cy} {cx},{cy+hh} {cx-hw},{cy}" fill="{col[0]}" stroke="{col[1]}" stroke-width="1.6"/>')
    text(cx,cy+5,label,13,True,'middle')


# ---- helpers added for the multi-panel figures ----
def b64(im,q=88):
    b=io.BytesIO(); im.save(b,'JPEG',quality=q); return base64.b64encode(b.getvalue()).decode()
def image(x,y,w,h,im):
    add(f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="none" xlink:href="data:image/jpeg;base64,{b64(im)}"/>')
def frame(x,y,w,h,col,sw=1.4): add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{col}" stroke-width="{sw}"/>')
BL='#3B6BB5'; OR='#D9822B'; GR='#3E8E5A'; RD='#C0392B'; PU='#8E44AD'

def arc_arrow(cx,cy,R,a0,a1,col='#4B5563',sw=2,head=8):
    a1e=a1-math.degrees(head*0.9/R)
    large=1 if ((a1e-a0)%360)>180 else 0
    p=lambda a:(cx+R*math.cos(math.radians(a)),cy+R*math.sin(math.radians(a)))
    x0,y0=p(a0); x1,y1=p(a1e)
    add(f'<path d="M{x0:.1f},{y0:.1f} A{R},{R} 0 {large} 1 {x1:.1f},{y1:.1f}" fill="none" stroke="{col}" stroke-width="{sw}"/>')
    tx,ty=p(a1); t=(-math.sin(math.radians(a1)),math.cos(math.radians(a1))); n=(-t[1],t[0])
    bx,by=tx-t[0]*head,ty-t[1]*head
    add(f'<polygon points="{tx:.1f},{ty:.1f} {bx+n[0]*head*0.45:.1f},{by+n[1]*head*0.45:.1f} {bx-n[0]*head*0.45:.1f},{by-n[1]*head*0.45:.1f}" fill="{col}"/>')
def circ_num(x,y,n,col=BL):
    add(f'<circle cx="{x}" cy="{y}" r="11" fill="{col}"/>'); text(x,y+5,str(n),14,True,'middle','#FFFFFF')
def check(cx,cy,r=8,col=GR):
    add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{col}"/><path d="M{cx-r*0.45:.1f},{cy+r*0.02:.1f} l{r*0.32:.1f},{r*0.4:.1f} l{r*0.62:.1f},{-r*0.8:.1f}" fill="none" stroke="#FFFFFF" stroke-width="{r*0.27:.1f}" stroke-linecap="round" stroke-linejoin="round"/>')
def chevron(x,y,col='#C3C7CF'):
    add(f'<polygon points="{x},{y-26} {x+11},{y-26} {x+24},{y} {x+11},{y+26} {x},{y+26} {x+13},{y}" fill="{col}"/>')
def dot(x,y,col,r=5): add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{col}"/>')
def legend_item(x,y,col,label,size=12.5,maxw=None):
    dot(x+5,y-4,col); text(x+16,y,label,size,False,'start',INK,maxw=maxw)



def write(path):
    """Write the accumulated SVG and report any label that overflows its box."""
    Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")
    print("  " + ("fit ok" if not warns else "\n  ".join(warns)))
