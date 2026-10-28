import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import pytest
ROOT=Path(__file__).resolve().parents[2];SCRIPTS=ROOT/'scripts/paper'
sys.path.insert(0,str(SCRIPTS))


def module(name):
    spec=importlib.util.spec_from_file_location(name,SCRIPTS/(name+'.py'));value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


b=module('build_paper');d=module('build_mechanism_diagrams')


def test_original_figures_keep_their_original_sections_and_order():
    saved=json.loads(Path(__file__).with_name('original_layout.json').read_text())
    for name,expected in saved['figures'].items():
        text=(ROOT/'paper/sections'/name).read_text(encoding='utf-8-sig')
        assert re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',text)==expected


def test_frozen_protocol_is_verbatim():
    saved=json.loads(Path(__file__).with_name('original_layout.json').read_text())
    text=(ROOT/'paper/sections/04_experiments.tex').read_text(encoding='utf-8-sig')
    text=text[text.index('\\subsection{Evaluation protocol}'):text.index('\\subsection{Zero-shot baselines:')]
    assert hashlib.sha256(text.encode()).hexdigest()==saved['protocol_sha256']


def test_stage_excludes_stale_aux_and_bbl(tmp_path):
    paper=tmp_path/'paper';paper.mkdir()
    for name in ['main.tex','refs.bib','main.bbl','main.aux','main.pdf']:(paper/name).write_text('fixture')
    (paper/'figures/pdf').mkdir(parents=True);(paper/'figures/pdf/a.pdf').write_bytes(b'fixture')
    work=tmp_path/'work';work.mkdir();files=b.stage_sources(paper,work)
    assert set(files)=={'main.tex','refs.bib','figures/pdf/a.pdf'}
    assert not (work/'main.bbl').exists()


def test_diagnostics_identify_citation_and_reference_keys():
    log="Package natbib Warning: Citation `missing-key' on page 1 undefined.\nLaTeX Warning: Reference `fig:missing' on page 1 undefined."
    assert b.diagnostics(log)['citations']==['missing-key']
    assert b.diagnostics(log)['references']==['fig:missing']
    with pytest.raises(RuntimeError,match='missing-key'):b.check_log(log)


def test_preflight_rejects_missing_bibliography_entry(tmp_path):
    (tmp_path/'main.tex').write_text(r'\citep{absent}')
    (tmp_path/'refs.bib').write_text('@misc{present,title={Unit}}')
    with pytest.raises(ValueError,match='absent'):b.validate_citations(tmp_path)


@pytest.mark.skipif(not shutil.which('pdflatex') or not shutil.which('bibtex'),reason='TeX is unavailable')
def test_real_tex_build_ignores_poisoned_tracked_bbl(tmp_path,monkeypatch):
    root=tmp_path/'project';paper=root/'paper';paper.mkdir(parents=True)
    (paper/'main.tex').write_text(r'''\documentclass{article}\usepackage{natbib}
\begin{document}Cite \citep{good}; see Section~\ref{sec:ok}.
\section{Good}\label{sec:ok}\bibliographystyle{plainnat}\bibliography{refs}\end{document}''')
    (paper/'refs.bib').write_text('@misc{good,author={Author, Test},title={Correct bibliography},year={2026}}')
    (paper/'main.bbl').write_text('POISONED STALE BIBLIOGRAPHY')
    (paper/'main.aux').write_text('POISONED AUX')
    monkeypatch.setattr(b,'ROOT',root)
    output=b.build(tmp_path/'output');assert output.is_file()
    assert 'POISONED' not in output.with_suffix('.bbl').read_text()
    assert 'good' in output.with_suffix('.bbl').read_text()


def test_detailed_diagrams_are_source_grounded_and_deterministic():
    a=d.rules_skills();c=d.speculative();ET.fromstring(a);ET.fromstring(c)
    assert a==d.rules_skills() and c==d.speculative()
    for field in ['R007','R002','read_skill','harness_events','base_image','dx_frac']:assert field in a
    assert a.count('<image ')==2
    for field in ['Visibility mask','mtp.*','mismatch','KV cache','tool_seconds']:assert field in c


def test_no_measurements_create_no_plot(tmp_path):
    p=module('plot_harness_serving');assert p.generate(paper=tmp_path) is None
    assert not list(tmp_path.rglob('*.pdf'))


def test_nested_verifier_returns_failure_on_missing_input(tmp_path):
    script=tmp_path/'scripts/paper/verify_paper.py';script.parent.mkdir(parents=True);shutil.copy2(SCRIPTS/'verify_paper.py',script)
    paper=tmp_path/'paper';(paper/'generated').mkdir(parents=True)
    shutil.copy2(ROOT/'paper/assets_summary.json',paper/'assets_summary.json')
    (paper/'main.tex').write_text(r'\input{generated/unit}\ref{tab:unit}')
    (paper/'generated/unit.tex').write_text(r'\label{tab:unit}')
    assert subprocess.run([sys.executable,str(script)],capture_output=True).returncode==0
    (paper/'generated/unit.tex').unlink()
    assert subprocess.run([sys.executable,str(script)],capture_output=True).returncode==1


def test_measured_plot_pipeline_with_synthetic_unit_records(tmp_path):
    pytest.importorskip('generate_harness_assets',reason='Measured plots use the 0050 validator when installed')
    import copy
    record={'image_id':'unit-1','exact_tp':1,'exact_fp':1,'exact_fn':0,'format_ok':True,
            'turns':[{'raw_output':'unit','tool_calls':[]}],'final_answer':{'unit':True}}
    harness={'schema_version':1,'status':'complete','split':'dev','dev_split_sha256':'unit-only',
             'image_ids':['unit-1'],'conditions':{},'comparisons':{}}
    for name in ['baseline','rules_only','rules_skills_index','rules_skills_full']:
        harness['conditions'][name]={'records':[copy.deepcopy(record)]}
        if name!='baseline':harness['comparisons'][name]={'pooled_exact_f1':{'delta':0,'ci95':[0,0],'valid_bootstraps':2000}}
    serving={'schema_version':1,'status':'complete','settings':{'split':'dev'},'dev_split_sha256':'unit-only',
             'image_ids':['unit-1'],'conditions':{},'gates':{}}
    for name in ['baseline','mtp_1','mtp_2']:
        r=copy.deepcopy(record)
        r['measurements']={'wall_seconds':4,'request_wall_seconds':2,'generated_tokens':10,
                           'accepted_tokens':2,'drafted_tokens':4,'prefill_seconds':.5,'decode_seconds':1.5,'tool_seconds':1}
        r['sampled_gpu_peak']=[{'index':0,'used_mib':1024}]
        serving['conditions'][name]={'status':'complete','episodes':[r]}
        if name!='baseline':serving['gates'][name]={'passed':True,'exact_f1_unchanged':True,'divergences':[]}
    hp=tmp_path/'harness.json';sp=tmp_path/'serving.json'
    hp.write_text(json.dumps(harness));sp.write_text(json.dumps(serving))
    p=module('plot_harness_serving');manifest=p.generate(hp,sp,tmp_path/'unit_plot')
    assert set(manifest['panels'])=={'quality','latency','acceptance','speedup','phases','memory'}
    assert (tmp_path/'unit_plot/figures/svg/fig_harness_serving_results.svg').is_file()
    harness['comparisons']['rules_only']['pooled_exact_f1']['delta']=.9
    hp.write_text(json.dumps(harness))
    with pytest.raises(ValueError,match='Stale'):p.collect(hp,sp)

    harness['comparisons']['rules_only']['pooled_exact_f1']['delta']=0
    hp.write_text(json.dumps(harness))
    serving['conditions']['mtp_1']['episodes'][0]['turns'][0]['raw_output']='divergent despite claimed gate'
    sp.write_text(json.dumps(serving))
    panels,_=p.collect(hp,sp)
    assert 'mtp_1' not in [name for name,_ in panels['speedup']]
