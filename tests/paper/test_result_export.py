"""Paper-only tests use synthetic records, never clinical/test-split inputs."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import pytest

SCRIPTS=Path(__file__).resolve().parents[2]/'scripts/paper'
sys.path.insert(0,str(SCRIPTS))


def module(name):
    spec=importlib.util.spec_from_file_location(name,SCRIPTS/(name+'.py'))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


assets=module('generate_harness_assets')


def episode():
    return dict(image_id='synthetic-a',exact_tp=1,exact_fp=1,exact_fn=0,format_ok=True,
                turns=[dict(raw_output='synthetic answer',tool_calls=[])],final_answer={'synthetic':True},
                measurements=dict(wall_seconds=4,request_wall_seconds=2,generated_tokens=10,
                                  accepted_tokens=3,drafted_tokens=6,prefill_seconds=None,decode_seconds=None))


def result(kind):
    record_key='records' if kind=='harness' else 'episodes'
    names=['baseline','rules_only'] if kind=='harness' else ['baseline','mtp_1']
    return dict(schema_version=1,status='complete',split='dev',settings={'split':'dev'},
                dev_split_sha256='synthetic-fingerprint',image_ids=['synthetic-a'],
                conditions={k:{'status':'complete',record_key:[episode()]} for k in names},
                gates={'mtp_1':{'passed':True,'exact_f1_unchanged':True,'divergences':[]}})


def test_pending_is_not_zero():
    assert r'\todo{pending}' in assets.render_harness()
    assert r'\todo{pending}' in assets.render_serving()
    assert '0.000' not in assets.render_serving()


def test_harness_recomputes_counts():
    data=result('harness');data['conditions']['baseline']['summary']={'pooled_exact_f1':99}
    assert '0.667' in assets.render_harness(data)
    assert '0.500' in assets.render_harness(data)


@pytest.mark.parametrize('kind',['harness','serving'])
@pytest.mark.parametrize('failure',['test','duplicate','unpaired','nan','negative'])
def test_invalid_evidence_rejected(kind,failure):
    data=result(kind);key='records' if kind=='harness' else 'episodes'
    if failure=='test': data['split']='test';data['settings']['split']='test'
    elif failure=='duplicate': data['image_ids']*=2
    elif failure=='unpaired':data['conditions']['baseline'][key][0]['image_id']='other'
    elif failure=='nan':data['conditions']['baseline'][key][0]['exact_tp']=float('nan')
    elif failure=='negative':data['conditions']['baseline'][key][0]['exact_fp']=-1
    with pytest.raises(ValueError):
        (assets.render_harness if kind=='harness' else assets.render_serving)(data)


def test_serving_rates_and_missing_counters():
    text=assets.render_serving(result('serving'))
    assert '0.667' in text  # pooled exact F1 recomputed from episode counts
    assert '1.000' in text  # paired episode wall-time ratio
    assert text.count('Speculative')==1
    assert 'Lookup' not in text and 'MTP,' not in text
    assert '0.500' in text
    assert 'pass' in text
    assert 'Hit rate' in text


@pytest.mark.parametrize('field',['text','final','gate','score','format'])
def test_divergent_conditions_have_no_speedup(field):
    data=result('serving');candidate=data['conditions']['mtp_1']['episodes'][0]
    if field=='text': candidate['turns'][0]['raw_output']='different'
    elif field=='final':candidate['final_answer']={}
    elif field=='score':candidate['exact_tp']=0
    elif field=='format':candidate['format_ok']=False
    else:data['gates']['mtp_1']['passed']=False
    data['status']='failed_or_divergent'
    text=assets.render_serving(data)
    row=next(r for r in text.splitlines() if r.startswith('Speculative'))
    assert 'FAIL' in row and row.count(r'\todo{pending}')==1


def test_no_argument_build_preserves_measured_tables(tmp_path):
    source=tmp_path/'input.json';source.write_text(json.dumps(result('harness')))
    out=tmp_path/'generated';assets.generate(out,harness_path=source)
    before=(out/'table_harness_dev.tex').read_bytes()
    assets.generate(out)
    assert (out/'table_harness_dev.tex').read_bytes()==before
    assert json.loads((out/'harness_evidence.json').read_text())['harness']['status']=='result_file'


def test_failed_batch_does_not_partially_replace_tables(tmp_path):
    assets.generate(tmp_path)
    before=(tmp_path/'table_harness_dev.tex').read_bytes()
    good=tmp_path/'good.json';good.write_text(json.dumps(result('harness')))
    bad=tmp_path/'bad.json';bad.write_text('{}')
    with pytest.raises(ValueError):assets.generate(tmp_path,good,bad)
    assert (tmp_path/'table_harness_dev.tex').read_bytes()==before


def test_grid_requires_explicit_paper_selection():
    data=result('serving')
    data['conditions']['mtp_2']=copy.deepcopy(data['conditions']['mtp_1'])
    data['gates']['mtp_2']=copy.deepcopy(data['gates']['mtp_1'])
    with pytest.raises(ValueError,match='Select one'):assets.render_serving(data)
    text=assets.render_serving(data,'mtp_2')
    assert text.count('Speculative')==1
    assert '0.500' in text


def test_paper_selection_is_recorded(tmp_path):
    data=result('serving');path=tmp_path/'input.json';path.write_text(json.dumps(data))
    assets.generate(tmp_path/'out',serving_path=path,serving_condition='mtp_1')
    manifest=json.loads((tmp_path/'out/harness_evidence.json').read_text())
    assert manifest['serving']['paper_condition']=='mtp_1'
