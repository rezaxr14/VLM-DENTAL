"""Source-grounded procedural and token-level mechanisms. Original diagrams are untouched."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import textwrap
from diagram_common import R,T,L,IMG,b64_of,arrow,connect,svg,MONO
ROOT=Path(__file__).resolve().parents[2]
NAVY='#17324D';BLUE='#34548A';TEAL='#16746B';GREEN='#2F7D45';ORANGE='#B8561F';RED='#B5383C'
INK='#303A45';GREY='#667787';PALE='#EDF3F8';LIGHT='#DCE6ED';WHITE='#FFFFFF'


def txt(x,y,s,size=14,color=INK,bold=False,mono=False,anchor='start'):
    size=size+2 if size in (12,13,14) else size
    return T(x,y,s,size,color,anchor=anchor,weight='bold' if bold else None,font=MONO if mono else 'Helvetica')


def lines(x,y,values,size=13,leading=21,**kwargs):
    return ''.join(txt(x,y+i*leading,s,size,**kwargs) for i,s in enumerate(values))


def head(x,y,letter,title,subtitle):
    return txt(x,y,letter,22,NAVY,True)+txt(x+32,y,title,17,NAVY,True)+txt(x+32,y+22,subtitle,11,GREY)


def doc(x,y,w,h,title):
    return R(x,y,w,h,rx=0,fill=WHITE,stroke=LIGHT)+R(x,y,w,28,rx=0,fill=PALE,stroke=LIGHT)+txt(x+12,y+19,title,12,BLUE,True,True)


def meta(path):return json.loads(path.read_text().split('---',2)[1])


def rules_skills(_assets=None):
    contract=ROOT/'dental_agent/harness/rules.md';data=meta(contract);rules={r['id']:r for r in data['rules']}
    skills=[meta(p) for p in sorted((ROOT/'dental_agent/harness/skills').glob('*/SKILL.md'))]
    p=[head(24,30,'A','Versioned behavioral contract','Existing instructions, implementation-linked IDs'),
       head(382,30,'B','Procedural context on demand','Progressive disclosure; no learned-weight update'),
       head(742,30,'C','Executable boundary','Shared routing; eight diagnostic tools')]
    p += [doc(24,68,332,280,'rules.md | v'+data['version']),txt(38,117,'R007  FORMAT',13,BLUE,True)]
    p += [lines(38,140,textwrap.wrap(rules['R007']['statement'],35),13,18),txt(38,202,'R002  BOX VERIFICATION',13,TEAL,True)]
    p += [lines(38,225,textwrap.wrap(rules['R002']['statement'].split('. After')[0]+'.',35),13,18)]
    p += [lines(38,288,['R009-R012 : reward bindings','R013      : verifier binding','AST/content digests detect drift'],12,18,mono=True)]
    p += [doc(382,68,334,155,'Skill index | names + descriptions'),lines(394,115,[s['name'] for s in skills],12,20,mono=True)]
    version=next(s['version'] for s in skills if s['name']=='box-verification')
    p += [doc(382,238,334,110,'box-verification | v'+version),lines(394,285,['Inspect the returned crop.','If misaligned: nudge, then re-inspect.','dx/dy are fractions of box width/height.'],12,19)]
    p += [doc(742,68,314,105,'Harness action (not tool #9)'),lines(754,112,['tool: read_skill','args: {name: box-verification}','returns: body, sources, version'],12,19,mono=True)]
    p += [txt(742,201,'Diagnostic registry (unchanged)',13,BLUE,True)]
    tools=[('locate_tooth','bbox'),('nudge_crop','revised bbox'),('fdi_label','FDI data'),('zoom_crop','image'),
           ('window_level','image'),('enhance_contrast','image'),('denoise','image'),('contralateral_compare','image')]
    for i,(name,kind) in enumerate(tools):
        p += [txt(748,224+i*17,name,12,mono=True),txt(1048,224+i*17,kind,11,TEAL if 'bbox' in kind else GREY,anchor='end')]
    p += [L(365,65,365,365,LIGHT),L(730,65,730,365,LIGHT),head(24,404,'D','What changes in an episode','Illustrative request/reply ordering through the existing shared dispatcher')]
    boxes=[(30,437,170,38),(294,437,190,38),(564,437,205,38),(850,437,205,38)]
    for box,title in zip(boxes,['Policy context','Harness session','Shared dispatcher','Native-image tools']):
        p += [R(*box,rx=0,fill=NAVY,stroke=NAVY),txt(box[0]+box[2]/2,462,title,13,WHITE,True,anchor='middle')]
    xs=[x+w/2 for x,y,w,h in boxes]
    for x in xs:p += [L(x,482,x,739,LIGHT,dash='3 4')]
    def message(a,b,y,s,color=BLUE):
        return arrow([(xs[a],y),(xs[b],y)],color=color,sw=1.1,free='both')+txt((xs[a]+xs[b])/2,y-8,s,12,color,anchor='middle')
    p += [message(0,2,507,'read_skill(name)'),message(2,1,542,'route to session.read(args)',TEAL),
          message(1,0,577,'body + citations enter conversation',TEAL),message(0,2,612,'separate locate / zoom / nudge turns'),
          message(2,3,647,'execute on base_image'),message(3,0,682,'bbox data or image observation',GREEN)]
    p += [R(294,707,475,50,rx=0,fill='#F3F8F5',stroke='#D0DFD4'),lines(306,727,['read_skill does not consume a diagnostic-call slot.','The existing episode-turn bound still applies.'],12,18)]
    p += [head(24,802,'E','Controlled comparisons and audit trail','Same checkpoint, images, tools and budgets; context is the intervention')]
    rows=[('OFF / baseline','existing prompt'),('rules only','byte-identical rendered prompt'),('index + on demand','index first; requested bodies later'),('full injection','all five bodies in initial context')]
    for i,(a,b) in enumerate(rows):p += [txt(24,843+i*22,a,13,BLUE,True),txt(240,843+i*22,b,13)]
    p += [doc(665,835,390,108,'Optional harness_events'),lines(677,878,['event: listed | loaded  turn: <index>','elapsed_seconds  name  version','sha256  mode  repeat'],12,20,mono=True)]
    p += [txt(24,965,'Prompt identity is a control, not an accuracy claim. Disclosure is logged rather than inferred.',12,GREY),
          txt(24,987,'Source: rules.md, skills.py, agent/loop.py, tool_dispatch.py | rules SHA-256 '+hashlib.sha256(contract.read_bytes()).hexdigest()[:12],11,GREY)]
    p += [L(24,1010,1056,1010,LIGHT),head(24,1051,'F','Visual re-inspection and box-relative geometry','Existing teacher training example: DENTEX image 624; not a harness-effectiveness measurement')]
    assets=ROOT/'paper/figures/assets'
    p += [txt(30,1098,'Initial view',13,BLUE,True),txt(210,1098,'After nudges',13,TEAL,True),
          IMG(30,1114,87,199,b64_of(assets/'crop_located_624.png')),IMG(222,1114,59,199,b64_of(assets/'crop_final_624.png')),
          R(30,1114,87,199,rx=0,fill='none',stroke=LIGHT),R(222,1114,59,199,rx=0,fill='none',stroke=LIGHT)]
    p += [connect((30,1114,87,199),'r',(222,1114,59,199),'l',color=TEAL,sw=1),txt(169,1200,'re-inspect',11,TEAL,anchor='middle')]
    p += [lines(390,1104,['nudge_crop changes coordinates, not pixels.','zoom_crop must show the revised box before diagnosis.'],14,24,bold=True),
          lines(390,1166,['cx_new = x + w/2 + dx_frac * w','cy_new = y + h/2 + dy_frac * h',
                         'w_new, h_new = max(scale,0.1) * (w,h)','Clamp size and corner to native-image bounds.'],13,25,mono=True),
          lines(390,1285,['The same native source image feeds every diagnostic call.','A prior crop never silently replaces base_image.'],13,23,color=GREY)]
    return svg(1080,1330,''.join(p))


def tile(x,y,w,h,s,fill=PALE,color=INK):
    return R(x,y,w,h,rx=2,fill=fill,stroke=LIGHT)+txt(x+w/2,y+h/2+5,s,14,color,mono=True,anchor='middle')


def speculative(_assets=None):
    p=[head(24,30,'A','One numerical target; two proposal mechanisms','Greedy verification with symbolic tokens; scheduling is illustrative, not a measured trace')]
    p += [txt(24,88,'Committed prefix',13,NAVY,True)]
    for i,s in enumerate(['x(t-3)','x(t-2)','x(t-1)','x(t)']):p += [tile(24+i*96,103,85,40,s)]
    p += [txt(24,166,'Reusable target KV cache for committed context',12,GREY),
          txt(450,88,'Native MTP',14,ORANGE,True),lines(450,111,['Use preserved mtp.* tensors;','propose d1, d2, d3 in a draft loop.'],13,20),
          txt(790,88,'N-gram lookup',14,BLUE,True),lines(790,111,['Match a context suffix;','reuse following tokens as proposals.'],13,20)]
    p += [L(24,190,1056,190,LIGHT),head(24,232,'B','Parallel target verification','Target distributions follow the same prefix and the proposed causal token history')]
    x0,y0,cell=125,297,38
    for j,c in enumerate(['K1','K2','K3','d1','d2','d3']):p += [txt(x0+j*cell+cell/2,y0-12,c,12,mono=True,anchor='middle')]
    for i,s in enumerate(['p1','p2','p3','p4']):
        p += [txt(76,y0+i*cell+25,s,14,BLUE,True,True)]
        for j in range(6):p += [R(x0+j*cell,y0+i*cell,cell-3,cell-3,rx=0,fill=BLUE if j<3 or j-3<i else '#F3F5F7',stroke=WHITE)]
    p += [txt(24,479,'Visibility mask, not attention scores',12,GREY),lines(24,505,['K1-K3 symbolize the cached prefix.','Each p(i) predicts a next token.','New query positions are target-batched.','Backend cache scheduling may differ.'],13,21)]
    p += [txt(505,285,'Draft',13,ORANGE,True),txt(505,350,'Target argmax',13,BLUE,True),txt(505,428,'Commit',13,GREEN,True)]
    for j in range(3):
        x=655+j*123
        p += [tile(x,258,95,42,'d'+str(j+1),'#FFF3E9',ORANGE),tile(x,323,95,42,'g'+str(j+1),'#EDF3FB',BLUE),
              L(x+45,302,x+45,320,GREEN if j<2 else RED,2),txt(x+45,390,'match' if j<2 else 'mismatch',12,GREEN if j<2 else RED,True,anchor='middle'),
              tile(x,403,95,42,'g'+str(j+1),'#ECF5EE' if j<2 else '#EDF3FB',GREEN if j<2 else BLUE)]
    p += [lines(505,482,['Accept the longest agreeing prefix; replace at the first mismatch.','Discard the rejected and dependent draft suffix.','All accepted? The target can supply a bonus token.','Only verified, parsed output can dispatch a tool.'],13,23)]
    p += [L(24,590,1056,590,LIGHT),head(24,632,'C','What can become faster, and what stays fixed','Separate image-prefill, token decoding and real tool execution')]
    p += [txt(24,687,'Baseline',13,NAVY,True)]
    for j,s in enumerate(['target g1','target g2','target g3']):p += [tile(140+j*148,662,132,40,s)]
    p += [txt(24,745,'Speculative',13,NAVY,True),tile(140,720,132,40,'draft block','#FFF3E9',ORANGE),tile(288,720,280,40,'target verifies the block','#EDF3FB',BLUE),
          txt(24,794,'Logical operations: widths do not encode latency or a claimed speedup.',12,GREY)]
    p += [doc(630,661,426,150,'Recorded per episode / request'),lines(642,704,['wall_seconds       tool_seconds','prefill_seconds    decode_seconds','generated_tokens   accepted / drafted','GPU peaks   raw outputs + tool calls'],12,23,mono=True)]
    p += [head(24,854,'D','Evidence gates and execution envelope','Baseline and speculation use the same merged checkpoint; never compare across numeric targets')]
    p += [lines(24,898,['BEFORE: compare mtp.* keys, shapes, dtypes and bytes with the original native base.',
                       'AFTER: compare decoded turns, tool calls, final answers and pooled dev exact F1.',
                       'FAILURE: retain divergences; suppress equivalence-qualified speedup claims.'],13,23),
          lines(24,982,['Requested: dual T4 = FP16 / TP2; RTX 4090 and 48-GB RTX Pro = BF16 / TP1.',
                       'Fit/support need a real runtime probe. This runner does not supply TPU speculation.'],12,20,color=GREY)]
    return svg(1080,1020,''.join(p))


BUILDERS={'fig_rules_skills_mechanism.svg':rules_skills,'fig_speculative_mechanism.svg':speculative}


def main():
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF
    for name,builder in BUILDERS.items():
        source=ROOT/'paper/figures/svg'/name;source.write_text(builder(),encoding='utf-8')
        drawing=svg2rlg(str(source))
        if drawing is None:raise RuntimeError('Cannot convert '+name)
        renderPDF.drawToFile(drawing,str(ROOT/'paper/figures/pdf'/source.with_suffix('.pdf').name),invariant=1)
        print(source.name)


if __name__=='__main__':main()
