"""Clean, cross-platform TeX build: historical source .aux/.bbl never enter the build tree."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
ROOT=Path(__file__).resolve().parents[2]


def diagnostics(log):
    return {'citations':sorted(set(re.findall(r"Citation [`']([^`'\n]+)['`]",log))),
            'references':sorted(set(re.findall(r"Reference [`']([^`'\n]+)['`]",log))),
            'layout':[s for s in log.splitlines() if 'Overfull \\hbox' in s or 'Float too large' in s],
            'unresolved_summary':bool(re.search(r'There were undefined (?:citations|references)',log))}


def check_log(log,logs=None):
    found=diagnostics(log)
    if any(found.values()):
        detail='\n'.join(k+': '+', '.join(v) for k,v in found.items() if isinstance(v,list) and v)
        raise RuntimeError('Paper QA failed after isolated compilation.\n'+detail+'\nComplete logs: '+str(logs or 'main.log/main.blg'))


def stage_sources(paper,work):
    paper=Path(paper).resolve();work=Path(work).resolve()
    if work==paper or work.is_relative_to(paper):raise ValueError('Build outside paper/')
    result={}
    for p in sorted(paper.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(paper)
        include=p.suffix.lower() in {'.tex','.bib','.bst','.sty','.cls'}
        include=include or (rel.parts[0]=='figures' and p.suffix.lower() in {'.pdf','.png','.jpg','.jpeg','.eps'})
        if include:
            target=work/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
            result[rel.as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def validate_citations(work):
    text='\n'.join(p.read_text(encoding='utf-8') for p in Path(work).rglob('*.tex'))
    text=re.sub(r'(?<!\\)%.*','',text)
    citations=set()
    for group in re.findall(r'\\cite\w*\*?(?:\[[^\]]*\]){0,2}\{([^}]+)\}',text):citations.update(k.strip() for k in group.split(','))
    bibliography='\n'.join(p.read_text(encoding='utf-8') for p in Path(work).glob('*.bib'))
    keys=set(re.findall(r'@\w+\s*\{\s*([^,\s]+)',bibliography))
    if citations-keys:raise ValueError('Citations absent from bibliography: '+', '.join(sorted(citations-keys)))


def build(output):
    for tool in ('pdflatex','bibtex'):
        if not shutil.which(tool):raise RuntimeError('Missing '+tool+'; install MiKTeX or TeX Live and add it to PATH')
    output=Path(output).resolve();paper=ROOT/'paper'
    if output==paper or output.is_relative_to(paper):raise ValueError('Use an output folder outside paper/')
    output.mkdir(parents=True,exist_ok=True);logs=output/'logs';logs.mkdir(exist_ok=True);commands=[]
    with tempfile.TemporaryDirectory(prefix='tex_',dir=output) as temporary:
        work=Path(temporary);sources=stage_sources(paper,work);validate_citations(work)
        def run(command,label):
            commands.append(command);env=os.environ.copy()
            for key in ('TEXINPUTS','BIBINPUTS','BSTINPUTS'):env[key]='.'+os.pathsep+env.get(key,'')
            result=subprocess.run(command,cwd=work,env=env,text=True,encoding='utf-8',errors='replace',stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            (logs/(label+'.txt')).write_text(result.stdout,encoding='utf-8')
            for suffix in ('.log','.blg','.aux','.bbl'):
                path=work/('main'+suffix)
                if path.exists():shutil.copyfile(path,logs/path.name)
            if result.returncode:raise RuntimeError(command[0]+' failed; see '+str(logs/(label+'.txt')))
        latex=['pdflatex','-interaction=nonstopmode','-halt-on-error','-file-line-error','main.tex']
        run(latex,'01_latex');run(['bibtex','main'],'02_bibtex')
        previous=None
        for i in range(2,7):
            run(latex,f'{i+1:02}_latex_pass_{i}')
            log=(work/'main.log').read_text(errors='replace');found=diagnostics(log)
            state=hashlib.sha256(b''.join((work/('main'+ext)).read_bytes() for ext in ('.aux','.out') if (work/('main'+ext)).exists())).hexdigest()
            if state==previous and not (found['citations'] or found['references'] or found['unresolved_summary']):break
            previous=state
        check_log(log,logs)
        for name in ('main.pdf','main.log','main.blg','main.bbl'):shutil.copyfile(work/name,output/name)
    metadata={'commands':commands,'staged_clean':True,'sources_sha256':sources,
              'pdflatex':subprocess.check_output(['pdflatex','--version'],text=True).splitlines()[0]}
    (output/'build_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(output/'main.pdf');return output/'main.pdf'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'scratch/paper_build')
    build(parser.parse_args().output_dir)


if __name__=='__main__':main()
