"""Measured dev-only publication panels; absent inputs do not create fabricated charts.

Uses 0050's evidence validator. The paper alone builds with committed diagrams/pending tables.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]


def finite(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)


def collect(harness=None,serving=None):
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    try:
        from generate_harness_assets import render_harness,render_serving,counts,serving_stats,serving_gate
    except ImportError as error:
        raise RuntimeError('Apply evaluation patch 0050 for the shared evidence validator') from error
    data={};sources={}
    for kind,path,validate in [('harness',harness,render_harness),('serving',serving,render_serving)]:
        if path:
            raw=Path(path).read_bytes();value=json.loads(raw);validate(value)
            data[kind]=value;sources[kind]={'sha256':hashlib.sha256(raw).hexdigest(),'images':len(value['image_ids'])}
    panels={};missing=[]
    if 'harness' in data:
        report=data['harness'];rows=[];base=counts(report['conditions']['baseline']['records'])[0]
        for name,comparison in report.get('comparisons',{}).items():
            metric=comparison.get('pooled_exact_f1',{});delta=metric.get('delta');ci=metric.get('ci95')
            if name not in report['conditions']:raise ValueError('Unknown comparison '+name)
            actual=counts(report['conditions'][name]['records'])[0]
            if base is not None and actual is not None and finite(delta) and abs(delta-(actual-base))>1e-8:
                raise ValueError('Stale paired F1 difference for '+name)
            if finite(delta) and isinstance(ci,list) and len(ci)==2 and all(finite(x) for x in ci) and ci[0]<=delta<=ci[1] and metric.get('valid_bootstraps',0)>0:
                rows.append((name,delta,ci))
            else:missing.append('paired_interval/'+name)
        if rows:panels['quality']=rows
    if 'serving' in data:
        report=data['serving'];entries=report['conditions'];base=serving_stats(entries['baseline'])
        latency=[];acceptance=[];speed=[];phases=[];memory=[]
        for name,entry in entries.items():
            if entry.get('status')!='complete':missing.append('incomplete/'+name);continue
            stats=serving_stats(entry);episodes=entry['episodes']
            passed=serving_gate(report,name)
            times=[e['measurements'].get('wall_seconds') for e in episodes]
            if times and all(finite(v) and v>0 for v in times):latency.append((name,times,passed))
            else:missing.append('wall_time/'+name)
            if name!='baseline' and stats['acceptance'] is not None and passed:acceptance.append((name,stats['acceptance']))
            elif name!='baseline':missing.append('acceptance_or_gate/'+name)
            if passed and base['wall'] and stats['wall']:speed.append((name,base['wall']/stats['wall']))
            ratios={}
            for key in ('prefill_seconds','decode_seconds','tool_seconds'):
                values=[e['measurements'].get(key) for e in episodes]
                if stats['wall'] and all(finite(v) and v>=0 for v in values):ratios[key]=sum(values)/stats['wall']
                else:missing.append(key+'/'+name)
            if ratios:phases.append((name,ratios))
            devices={}
            for e in episodes:
                for gpu in e.get('sampled_gpu_peak') or []:
                    v=gpu.get('used_mib')
                    if finite(v) and v>=0:devices[str(gpu['index'])]=max(devices.get(str(gpu['index']),0),v/1024)
            if devices:memory.append((name,devices))
        for key,rows in [('latency',latency),('acceptance',acceptance),('speedup',speed),('phases',phases),('memory',memory)]:
            if rows:panels[key]=rows
    return panels,{'sources':sources,'unmeasured':missing,'phase_note':'Components may overlap; ratios need not sum to one.',
                   'memory_note':'Sampled whole-device peaks per GPU, not pooled VRAM or allocator-only usage.'}


def generate(harness=None,serving=None,paper=None):
    if not harness and not serving:
        print('No measured inputs: diagrams and plots unchanged.');return None
    panels,manifest=collect(harness,serving)
    if not panels:raise ValueError('No measured panels; no chart created')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
                         'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    order=[k for k in ('quality','latency','acceptance','speedup','phases','memory') if k in panels]
    cols=2 if len(order)>1 else 1;rows=math.ceil(len(order)/cols)
    fig,axes=plt.subplots(rows,cols,figsize=(9.4 if cols==2 else 6.3,2.9*rows),squeeze=False,constrained_layout=True)
    label=lambda x:x.replace('rules_skills_','').replace('rules_only','rules').replace('ngram_','lookup ').replace('mtp_','MTP ')
    titles={'quality':'Paired dev exact-F1 difference','latency':'Episode latency distributions','acceptance':'Accepted / drafted tokens',
            'speedup':'Equivalence-qualified episode speedup','phases':'Measured phase / episode wall time','memory':'Sampled whole-device memory'}
    blue,orange,red='#34548A','#B8561F','#B5383C'
    for index,(key,ax) in enumerate(zip(order,axes.flat)):
        rows=panels[key];ax.set_title(f'({chr(97+index)}) {titles[key]}',loc='left',fontweight='bold')
        ax.grid(axis='y',color='#DDDDDD',linewidth=.5);ax.set_axisbelow(True)
        if key=='quality':
            names,delta,cis=zip(*rows);y=np.arange(len(names));ci=np.asarray(cis)
            ax.errorbar(delta,y,xerr=[np.asarray(delta)-ci[:,0],ci[:,1]-np.asarray(delta)],fmt='o',color=blue,capsize=3)
            ax.axvline(0,color='#888888',ls='--',lw=.8);ax.set_yticks(y,[label(n) for n in names]);ax.set_xlabel('Difference from baseline (95% image-bootstrap CI)')
        elif key=='latency':
            names,values,gates=zip(*rows);boxes=ax.boxplot(values,patch_artist=True,widths=.55)
            for box,passed in zip(boxes['boxes'],gates):box.set_facecolor(blue if passed else red);box.set_alpha(.4)
            ax.set_xticks(range(1,len(names)+1),[label(n)+(' *' if not g else '') for n,g in zip(names,gates)],rotation=40,ha='right')
            ax.set_ylabel('Seconds per episode');ax.set_ylim(bottom=0)
            if not all(gates):ax.text(.02,.98,'* decoded-output gate failed',va='top',transform=ax.transAxes,color=red,fontsize=7)
        elif key=='acceptance':
            for method,color,marker in [('mtp',orange,'o'),('ngram',blue,'s')]:
                points=sorted((int(n.split('_')[1]),v) for n,v in rows if n.startswith(method+'_'))
                if points:
                    x,y=zip(*points);ax.plot(x,y,marker=marker,color=color,label='MTP' if method=='mtp' else 'Lookup')
            ax.set_ylim(0,1);ax.set_xlabel('Proposed tokens');ax.set_ylabel('Acceptance fraction');ax.legend(frameon=False)
        elif key=='speedup':
            names,values=zip(*rows);ax.bar(range(len(names)),values,color=blue,width=.6)
            ax.axhline(1,color='#888888',ls='--',lw=.8);ax.set_xticks(range(len(names)),[label(n) for n in names],rotation=40,ha='right');ax.set_ylabel('Baseline wall / condition wall')
        elif key=='phases':
            names=[n for n,_ in rows];width=.23
            for j,(field,title,color) in enumerate([('prefill_seconds','Prefill',blue),('decode_seconds','Decode',orange),('tool_seconds','Tools','#2F7D45')]):
                points=[(i,v[field]) for i,(_,v) in enumerate(rows) if field in v]
                if points:
                    x,y=zip(*points);ax.bar(np.asarray(x)+(j-1)*width,y,width,label=title,color=color)
            ax.set_xticks(range(len(names)),[label(n) for n in names],rotation=40,ha='right');ax.set_ylabel('Measured seconds / episode seconds');ax.legend(frameon=False,fontsize=7)
        else:
            names=[n for n,_ in rows];devices=sorted({g for _,d in rows for g in d});width=.75/len(devices)
            for j,gpu in enumerate(devices):
                points=[(i,d[gpu]) for i,(_,d) in enumerate(rows) if gpu in d];x,y=zip(*points)
                ax.bar(np.asarray(x)+j*width,y,width,label='GPU '+gpu)
            ax.set_xticks(np.arange(len(names))+width*(len(devices)-1)/2,[label(n) for n in names],rotation=40,ha='right')
            ax.set_ylabel('GiB per GPU (sampled)');ax.legend(frameon=False,fontsize=7)
    for ax in list(axes.flat)[len(order):]:ax.set_visible(False)
    paper=Path(paper or ROOT/'paper')
    for kind in ('svg','pdf'):
        target=paper/'figures'/kind/('fig_harness_serving_results.'+kind);target.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(target,bbox_inches='tight')
    plt.close(fig)
    out=paper/'generated';out.mkdir(parents=True,exist_ok=True);manifest['panels']=order
    (out/'harness_serving_plot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'harness_serving_plot.tex').write_text(r'''\begin{figure}[tbp]
\centering
\includegraphics[width=\linewidth]{fig_harness_serving_results}
\caption{Measured dev-only procedural-context and serving results. Paired F1 differences resample
shared images. Serving conditions share a numerical target; speedups require decoded-output gates.
Missing counters are omitted, not imputed. Phase ratios may overlap; memory is sampled whole-device
usage. Input hashes, cohort sizes and omitted measurements are recorded in the plot manifest.}
\label{fig:harness-serving-results}
\end{figure}
''')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--harness-results',type=Path);p.add_argument('--serving-results',type=Path)
    p.add_argument('--paper-dir',type=Path,default=ROOT/'paper');a=p.parse_args();generate(a.harness_results,a.serving_results,a.paper_dir)


if __name__=='__main__':main()
