"""Orchestration contracts, not GPU/TPU or diagnostic-performance validation."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
SCRIPTS=ROOT/'scripts/paper'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('workflow',SCRIPTS/'final_eval_workflow.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)


def config(tmp_path):
    return dict(profile='dual_t4',mode='dev',run_id='unit',stage='dentex_alone',selected_k=4,
                conditions=list(w.CONDITIONS),sizes=['canonical','original'],smoke_limit=None,
                model_id='/model',data_dir='/data',dev_split='/dev.json',output=str(tmp_path),
                adapters={c:str(tmp_path/c) for c in w.CONDITIONS if not c.startswith('base_')})


def test_profiles_and_exact_track_names():
    for name,tag in [('dual_t4','dual_t4_fp16'),('rtx4090','rtx4090_bf16'),('rtxpro6000_48gb','workstation48_lora_bf16'),('tpu','tpu_bf16')]:
        assert w.profile(name)['tag']==tag
        paths=w.names(name,'dentex_alone',4)
        assert all(tag in p for p in paths.values())
        assert paths['grpo_with_tools'].endswith('/grpo-with_tools-final')
        assert 'no_tools' in paths['sft_no_tools']


def test_primary_load_groups_keep_distinct_trained_tracks(tmp_path):
    c=config(tmp_path);commands=w.primary_commands(c)
    assert len(commands)==5
    for cmd in commands[1:]:
        condition=cmd[cmd.index('--conditions')+1]
        flag='--sft-adapter' if condition.startswith('sft') else '--grpo-adapter'
        assert cmd[cmd.index(flag)+1]==c['adapters'][condition]
        assert '--reuse-cached' not in cmd and '--loss-chunk-tokens' not in cmd
    assert '2048' in commands[0]


@pytest.mark.parametrize('bad',['same_track','partial_test','unknown_profile','duplicate','unsafe','missing'])
def test_invalid_plan_rejected_before_model_load(tmp_path,bad):
    c=config(tmp_path)
    if bad=='same_track':c['adapters']['sft_no_tools']=c['adapters']['sft_with_tools']
    elif bad=='partial_test':c.update(mode='test',smoke_limit=2)
    elif bad=='unknown_profile':c['profile']='unknown'
    elif bad=='duplicate':c['conditions']*=2
    elif bad=='unsafe':c['run_id']='../escape'
    else:c['adapters'].pop('sft_no_tools')
    with pytest.raises(ValueError):w.primary_commands(c)


def test_probes_are_dev_only(tmp_path):
    c=config(tmp_path)
    assert len(w.probe_commands(c,['nudge_crop'],[(.5,1.)]))==4
    c['mode']='test'
    with pytest.raises(ValueError):w.probe_commands(c,['nudge_crop'],[(.5,1.)])


def test_identity_lock_never_overwrites_different_run(tmp_path):
    path=tmp_path/'identity.json';w.lock(path,{'precision':'fp16'});w.lock(path,{'precision':'fp16'})
    with pytest.raises(ValueError):w.lock(path,{'precision':'bf16'})
    assert json.loads(path.read_text())=={'precision':'fp16'}


@pytest.mark.parametrize('failure',['missing','duplicate','split','prompt'])
def test_cohort_rejects_partial_or_mixed_evidence(failure):
    rows=[dict(image_id=i,split='dev',prompt='unified') for i in (1,2)]
    if failure=='missing':rows.pop()
    elif failure=='duplicate':rows.append(rows[0])
    elif failure=='split':rows[0]['split']='test'
    else:rows[0]['prompt']='zero_shot_cached'
    with pytest.raises(ValueError):w.cohort(rows,[1,2],'dev')


def test_strict_jsonl_and_failure_propagation(tmp_path):
    path=tmp_path/'rows.jsonl';path.write_text('{broken}')
    with pytest.raises(ValueError):w.strict_records(path)
    with pytest.raises(subprocess.CalledProcessError):w.run([[sys.executable,'-c','raise SystemExit(3)']],tmp_path/'logs')


def test_freeze_is_complete_dev_and_relocation_safe(tmp_path):
    c=config(tmp_path);done={f'{cond}@{size}':{} for cond in w.CONDITIONS for size in c['sizes']}
    path=tmp_path/'freeze.json';w.freeze(path,c,{'weights':'sha'},done)
    final=copy.deepcopy(c);final.update(mode='test',model_id='/new/path',output='/other')
    w.verify_frozen(path,final,{'weights':'sha'})
    final['selected_k']=8
    with pytest.raises(ValueError):w.verify_frozen(path,final,{'weights':'sha'})
    with pytest.raises(ValueError):w.freeze(tmp_path/'bad',c,{}, {'fake':{}})
    c['smoke_limit']=2
    with pytest.raises(ValueError):w.freeze(tmp_path/'bad',c,{},done)


def test_grpo_policy_metadata_and_hashes(tmp_path):
    root=tmp_path/'checkpoint';p=root/'grpo_policy';p.mkdir(parents=True)
    (p/'adapter_config.json').write_text('{}');(p/'adapter_model.safetensors').write_bytes(b'unit-not-a-model')
    (root/'checkpoint_meta.json').write_text(json.dumps({'extra':{'track':'with_tools','group_size':4}}))
    c=config(tmp_path);identity=w.adapter_identity(root,'grpo_with_tools',c)
    assert identity['policy_path']==str(p)
    assert all(not Path(k).is_absolute() for k in identity['files'])
    c['selected_k']=8
    with pytest.raises(ValueError):w.adapter_identity(root,'grpo_with_tools',c)


def test_notebook_schema_syntax_and_execution_boundaries():
    import nbformat
    n=nbformat.read(ROOT/'notebooks/VLM_Dental_Colab_Final_Evaluation.ipynb',as_version=4);nbformat.validate(n)
    for cell in n.cells:
        if cell.cell_type=='code':
            ast.parse(cell.source);assert cell.execution_count is None and cell.outputs==[]
    source='\n'.join(c.source for c in n.cells)
    assert 'pkill' not in source and "'git','pull'" not in source
    assert 'MAX_MODEL_LEN = 16384' in source and 'AGGREGATE_ONLY' in source
    assert 'scripts/run_zero_shot.py' not in source


def flags(path):
    out=set()
    for n in ast.walk(ast.parse((ROOT/path).read_text())):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument':
            for a in n.args:
                if isinstance(a,ast.Constant) and isinstance(a.value,str) and a.value.startswith('--'):
                    out.add(a.value)
                    if any(k.arg=='action' and isinstance(k.value,ast.Attribute) and k.value.attr=='BooleanOptionalAction' for k in n.keywords):out.add('--no-'+a.value[2:])
    return out


def test_flags_match_actual_runner_contract(tmp_path):
    allowed=set().union(*(flags(p) for p in ['scripts/run_eval_matrix.py','dental_agent/model/local_runtime.py','dental_agent/utils/canonical.py']))
    c=config(tmp_path)
    for cmd in w.primary_commands(c)+w.probe_commands(c,['nudge_crop'],[(.5,1.)]):
        assert all(arg in allowed for arg in cmd if arg.startswith('--'))
