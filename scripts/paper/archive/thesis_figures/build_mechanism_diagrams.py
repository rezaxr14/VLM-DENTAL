"""Approved harness/context and draft-training/speculation figures; no model execution."""
from pathlib import Path
from html import escape
import json,hashlib
from PIL import ImageFont
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'paper/figures/svg'
C={'ink':'#17324A','muted':'#536A7C','blue':'#2865A7','bluebg':'#EEF4FC','teal':'#087F83','tealbg':'#EAF7F4','orange':'#BD641B','orangebg':'#FFF2E3','red':'#BF4652','redbg':'#FCEEF0','green':'#337857','greenbg':'#EAF5ED','line':'#CBD8E1','grey':'#F2F5F7','white':'#FFFFFF','purple':'#76519A'}
from matplotlib import get_data_path
from functools import lru_cache
FONT_DIR=Path(get_data_path())/'fonts/ttf'
FONT=str(FONT_DIR/'DejaVuSans.ttf');BOLD=str(FONT_DIR/'DejaVuSans-Bold.ttf');MONO=str(FONT_DIR/'DejaVuSansMono.ttf')

@lru_cache(maxsize=1)
def prepare_fonts():
 from svglib.fonts import register_font
 for family,path,weight in [('DejaVu Sans',FONT,'normal'),('DejaVu Sans',BOLD,'bold'),('DejaVu Sans Mono',MONO,'normal'),('DejaVu Sans Mono',MONO,'bold')]:
  _,ok=register_font(family,path,weight=weight)
  if not ok:raise RuntimeError('Cannot register diagram font: '+family)

class Figure:
 def __init__(self,w,h,title,subtitle):
  prepare_fonts()
  self.w=w;self.h=h;self.parts=[];self.labels=[];self.dx=0;self.dy=0
  self.rect(5,5,w-10,h-10,'white','ink',2.5,12)
  self.text(36,53,title,35,'ink',True)
  self.text(36,93,subtitle,25,'muted')
  self.line(36,113,w-36,113,'line',1.5)
 def color(self,s):return C.get(s,s)
 def rect(self,x,y,w,h,fill='white',stroke='line',sw=1.5,r=6,dash=None):
  x+=self.dx;y+=self.dy
  self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{self.color(fill)}" stroke="{self.color(stroke)}" stroke-width="{sw}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
 def line(self,x1,y1,x2,y2,color='line',sw=1.5,dash=None):
  x1+=self.dx;x2+=self.dx;y1+=self.dy;y2+=self.dy
  self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{self.color(color)}" stroke-width="{sw}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
 def hatch(self,x1,y1,x2,y2):
  # Decorative mask hatch, not a connector/leader line.
  self.parts.append(f'<path d="M {x1+self.dx} {y1+self.dy} L {x2+self.dx} {y2+self.dy}" fill="none" stroke="{C["line"]}" stroke-width="1.5"/>')
 def text(self,x,y,s,size=26,color='ink',bold=False,anchor='start',mono=False):
  x+=self.dx;y+=self.dy
  fp=MONO if mono else BOLD if bold else FONT
  width=ImageFont.truetype(fp,size).getlength(s)
  x0=x-width/2 if anchor=='middle' else x-width if anchor=='end' else x
  assert x0>=12 and x0+width<=self.w-12,(s,x0,width,self.w)
  self.labels.append((x0,y-size*.85,x0+width,y+size*.2,s))
  self.parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="'+('DejaVu Sans Mono, monospace' if mono else 'DejaVu Sans, Arial, sans-serif')+f'" font-size="{size}" font-weight="'+('bold' if bold else 'normal')+f'" fill="{self.color(color)}">{escape(s)}</text>')
 def rows(self,x,y,rows,size=26,color='ink',bold=False,dy=35,mono=False):
  for i,s in enumerate(rows):self.text(x,y+i*dy,s,size,color,bold,mono=mono)
 def arrow(self,points,color='ink',sw=2.5,dash=None):
  from diagram_common import arrow
  points=[(x+self.dx,y+self.dy) for x,y in points]
  self.parts.append(arrow(points,color=self.color(color),sw=sw,dash=dash))
 def panel(self,x,y,w,h,letter,title,color='blue',fill='white'):
  self.rect(x,y,w,h,fill,color,1.8,9)
  self.rect(x+15,y+15,32,32,color,color,0,5)
  self.text(x+31,y+39,letter,23,'white',True,'middle')
  self.text(x+61,y+39,title,26,color,True)
 def token(self,x,y,s,fill='bluebg',stroke='blue',w=65,h=49,size=28):
  self.rect(x,y,w,h,fill,stroke,1.8,6);self.text(x+w/2,y+h/2+size*.34,s,size,stroke,True,'middle')
 def save(self,name,description):
  s=f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}" role="img" aria-labelledby="title desc"><title id="title">{escape(name)}</title><desc id="desc">{escape(description)}</desc>' + ''.join(self.parts)+'</svg>'
  return s

def rules_skills(_assets=None):
 f=Figure(1528,1090,'Rules and skills: how the harness changes the policy context',
          'Explicit contracts + progressive disclosure; the learned checkpoint and diagnostic tools are shared')
 f.panel(34,138,410,386,'a','Rule contract','blue','bluebg')
 f.text(54,220,'rules.md  |  v1.1.0',27,'blue',True,mono=True)
 f.text(54,265,'R007  Output structure',24,'blue',True)
 f.text(54,298,'One JSON object per turn.',24,'ink')
 f.line(54,318,424,318)
 f.text(54,353,'R018  Tooth identity',24,'blue',True)
 f.text(54,386,'Use FDI two-digit tooth numbers.',23,'ink')
 f.line(54,405,424,405)
 f.text(54,440,'R008  Final findings',24,'blue',True)
 f.text(54,473,'List all investigated findings.',24,'ink')
 f.text(54,505,'No target pathology: empty list [].',22,'ink')
 f.dx=28
 f.panel(517,138,466,386,'b','Policy context','ink','white')
 for y,title,sub,color,bg,dash in [
  (211,'System instructions','Existing rules, rendered verbatim','blue','bluebg',None),
  (286,'Skill index','Names + short descriptions','teal','tealbg',None),
  (361,'Episode state','Image + conversation + tool history','ink','grey',None),
  (436,'Requested body','Added only after a read_skill action','orange','orangebg','7 5')]:
  f.rect(536,y,428,67,bg,color,1.6,5,dash)
  f.text(552,y+29,title,25,color,True);f.text(552,y+55,sub,21,'muted')
 f.dx=0
 f.arrow([(444,331),(545,331)],'blue');f.text(494.5,309,'render',22,'blue',True,'middle')
 f.arrow([(1084,318),(1011,318)],'teal');f.text(1046,300,'index',19,'teal',True,'middle')
 f.dx=28
 f.panel(1056,138,410,386,'c','Skill library','teal','tealbg')
 f.text(1076,222,'skills/<name>/SKILL.md',23,'teal',True)
 names=['box-verification','caries-vs-deep-caries','periapical-lesion','impacted-tooth','healthy-scan-negative-survey']
 for i,n in enumerate(names):
  y=253+i*43
  f.rect(1076,y,370,37,'white','teal' if i==1 else 'line',2 if i==1 else 1,4)
  f.text(1090,y+26,n,21,'teal' if i==1 else 'ink',i==1)
 f.text(1076,506,'Each file: version, sources, body',22,'muted')
 f.dx=0
 f.text(34,594,'d   On-demand retrieval during an episode',29,'ink',True)
 f.panel(34,630,410,240,'1','Policy requests context','blue','white')
 f.rows(54,717,['tool: read_skill','args.name:','caries-vs-deep-caries'],23,dy=40,mono=True)
 f.text(54,844,'A harness action, not tool #9.',23,'muted')
 f.dx=28
 f.panel(517,630,466,240,'2','Dispatcher + session','teal','white')
 f.rows(537,717,['Validate the requested name.','Read the allow-listed skill file.','Return body, version and sources.'],23,dy=40)
 f.text(537,844,'No diagnostic tool is executed.',23,'muted')
 f.panel(1056,630,410,240,'3','Retrieved content','orange','orangebg')
 f.text(1076,716,'SKILL  caries-vs-deep-caries',21,'orange',True)
 f.rows(1076,753,['Caries and Deep Caries are','distinct permitted output','labels. [S2]'],24,dy=32)
 f.text(1076,850,'v1.2.0 + source references',22,'orange',True)
 f.dx=0
 f.arrow([(444,759),(545,759)],'blue');f.arrow([(1011,759),(1084,759)],'teal')
 # Skill body is appended to history: the new context layer is the actual intervention.
 f.arrow([(1289,630),(1289,566),(1046,566),(1046,469),(1011,469)],'orange',2.8)
 f.text(1120,554,'append to history',22,'orange',True)
 f.arrow([(778,870),(778,922)],'green',2.3)
 f.panel(34,923,1460,124,'e','Observable intervention','green','greenbg')
 f.text(54,999,'harness_events: listed / loaded; skill, version, content hash, turn, elapsed time and mode',25,'green')
 f.text(54,1031,'Rules-only = prompt identity control. On-demand = index first. Full injection = all five bodies first.',23,'ink')
 f.text(34,1074,'Source-grounded schematic: rules.md, skills.py and the shared dispatcher. Skill excerpt is from the existing file, not a new clinical rule.',20,'muted')
 return f.save('Rules and skills','Standalone approval draft: concrete versioned output, tooth-identity and final-findings rules, a five-file skill library, policy context layers, an actual caries-vs-deep-caries request and body excerpt, and disclosure logging. No crop workflow is repeated.')

def speculative(_assets=None):
 f=Figure(1500,1230,'Speculative decoding: propose a block, verify it, commit the valid prefix',
          'A trained draft proposes tokens; the fixed target checks them before they enter the output')
 f.panel(34,138,1432,204,'a','Propose several next tokens','purple','white')
 f.rows(56,236,['Committed','context'],26,'ink',True,dy=34)
 for i,t in enumerate(['p1','p2','pt']):f.token(328+i*77,218,t,'bluebg','blue',66,55,27)
 f.arrow([(548,245),(660,245)],'purple')
 f.rect(660,208,270,76,'#F2EDF7','purple',1.8,7)
 f.text(795,239,'Trained draft',26,'purple',True,'middle')
 f.text(795,268,'proposes the next block',21,'muted',anchor='middle')
 f.arrow([(930,245),(1004,245)],'purple')
 for i,t in enumerate(['A','B','C','D']):f.token(1004+i*100,218,t,'#F2EDF7','purple',76,55,29)
 f.text(750,319,'Draft first; verify the block with the target. Accepted tokens extend the committed context.',24,'ink',anchor='middle')
 # Panels b and c share a translated group; panel c is otherwise unchanged.
 f.dy=40
 f.panel(34,326,648,423,'b','Causal target verification','blue','white')
 f.text(60,400,'Predict',23,'blue',True)
 f.text(312,400,'Prefix cache',22,'blue',True,'middle')
 f.text(473,400,'Draft keys',23,'teal',True,'middle')
 x0=243;y0=451;cell=46
 for j,t in enumerate(['A','B','C','D']):f.text(x0+(j+3)*cell+22,y0-11,t,24,'ink',True,'middle')
 labels=['Token 1','Token 2','Token 3','Token 4','Bonus token']
 for i,label in enumerate(labels):
  y=y0+i*cell
  f.text(60,y+32,label,23,'ink',True)
  for j in range(7):
   active=j<3 or j-3<i
   f.rect(x0+j*cell,y,cell-4,cell-4,'blue' if j<3 else 'teal' if active else 'grey','white',1,3)
   if not active:f.hatch(x0+j*cell+9,y+34,x0+j*cell+34,y+9)
 f.text(54,707,'Predict four new tokens, plus a possible bonus.',20,'muted')
 f.text(54,735,'Colored cells = visible keys, not attention scores.',20,'muted')
 f.panel(710,326,756,423,'c','Accept / reject with greedy target output','green','white')
 xs=[919,1047,1175,1303]
 f.text(731,421,'Proposal',26,'purple',True)
 for i,(x,t) in enumerate(zip(xs,['A','B','C','D'])):f.token(x,385,t,'#F2EDF7' if i<2 else 'redbg','purple' if i<2 else 'red',82,52,29)
 f.text(731,512,'Target',26,'blue',True)
 for x,t in zip(xs,['A','B','X','-']):f.token(x,476,t,'bluebg','blue',82,52,29)
 for i,x in enumerate(xs):
  f.line(x+41,440,x+41,473,'green' if i<2 else 'red',2)
  f.text(x+41,567,'match' if i<2 else 'mismatch' if i==2 else 'discard',22,'green' if i<2 else 'red',True,'middle')
 f.text(731,634,'Commit',26,'green',True)
 for x,t,color,bg in [(919,'A','green','greenbg'),(1047,'B','green','greenbg'),(1175,'X','blue','bluebg')]:f.token(x,598,t,bg,color,82,52,29)
 f.text(731,690,'Keep A, B; use target correction X at the first mismatch.',24,'ink')
 f.text(731,726,'Drop C and dependent D. If all match, a bonus token is possible.',22,'muted')
 f.dy=0
 f.panel(34,817,1432,348,'d','Planned draft training and deployment','orange','orangebg')
 f.text(56,895,'TRAINING STAGE',22,'orange',True)
 # A planned train-derived supervision path, not a completed training result.
 boxes=[(56,922,263,129),(356,922,273,129),(666,922,348,129),(1051,922,391,129)]
 for x,y,w,h in boxes:f.rect(x,y,w,h,'white','orange' if x<1051 else 'purple',1.8,7)
 f.text(187,961,'With-tools traces',24,'ink',True,'middle')
 f.text(187,999,'Images + tool history',21,'muted',anchor='middle')
 f.text(187,1029,'Training split only',21,'muted',anchor='middle')
 f.text(492,961,'Frozen target',25,'blue',True,'middle')
 f.text(492,999,'Generates continuations',21,'muted',anchor='middle')
 f.text(492,1029,'Provides target tokens',21,'muted',anchor='middle')
 f.text(840,961,'Draft optimization',25,'orange',True,'middle')
 f.text(840,999,'Learn from target continuations',21,'muted',anchor='middle')
 f.text(840,1029,'Update draft weights',23,'orange',True,'middle')
 f.text(1246,961,'Trained draft weights',25,'purple',True,'middle')
 f.text(1246,999,'Load into proposal step (a)',22,'purple',anchor='middle')
 f.text(1246,1029,'Target remains the verifier',22,'muted',anchor='middle')
 for left,right in zip(boxes,boxes[1:]):
  x,y,w,h=left;xx,yy,ww,hh=right;f.arrow([(x+w,y+h/2),(xx,yy+hh/2)],'orange',2.4)
 # Loading the trained weights begins a distinct inference path below the training chain.
 f.arrow([(1246,1051),(1246,1073),(226,1073),(226,1102)],'purple',2.5)
 f.text(56,1091,'DEPLOY',19,'purple',True)
 serving=[(56,1102,340,45),(555,1102,350,45),(1051,1102,391,45)]
 for (x,y,w,h),title,color in zip(serving,['Draft proposes','Target verifies','Commit output to agent'],['purple','blue','green']):
  f.rect(x,y,w,h,'white',color,1.7,5);f.text(x+w/2,y+31,title,24,color,True,'middle')
 for left,right in zip(serving,serving[1:]):
  x,y,w,h=left;xx,yy,ww,hh=right;f.arrow([(x+w,y+h/2),(xx,yy+hh/2)],'purple',2.4)
 f.text(34,1202,'Symbolic token example. The trained draft proposes; only target-verified output enters the existing agent/tool loop.',23,'muted')
 return f.save('Speculative decoding','Standalone approval draft: a spaced proposal stage with a trained draft, explicit next-token prediction rows in the causal visibility matrix, the preserved first-mismatch acceptance example, and a planned training-to-deployment flow for draft weights with a frozen target. No measured training or speed result is asserted.')


BUILDERS={'fig_rules_skills_mechanism.svg':rules_skills,'fig_speculative_mechanism.svg':speculative}

def main():
 from svglib.svglib import svg2rlg
 from reportlab.graphics import renderPDF
 OUT.mkdir(parents=True,exist_ok=True)
 pdf_dir=ROOT/'paper/figures/pdf';pdf_dir.mkdir(parents=True,exist_ok=True)
 for name,builder in BUILDERS.items():
  source=OUT/name;source.write_text(builder(),encoding='utf-8')
  drawing=svg2rlg(str(source))
  if drawing is None:raise RuntimeError('Cannot convert '+name)
  renderPDF.drawToFile(drawing,str(pdf_dir/(source.stem+'.pdf')),invariant=1)
  print(source.name)

if __name__=='__main__':main()
