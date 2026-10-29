"""CPU wiring/provenance tests; no model downloads, training, vLLM or live Hub calls."""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace,ModuleType

import pytest

from dental_agent.harness.configuration import (HARNESS_MODES,harness_options,harness_identity,
    assert_harness_identity,validate_harness_mode,write_target_manifest,require_harness_grpo_target)
from dental_agent.harness import notebook_eval as w
from dental_agent.harness.checkpoints import mtp_parameter_inventory,target_tensor_fingerprint

ROOT=Path(__file__).resolve().parents[1]
MODE='rules_skills_index'
_profile_spec=importlib.util.spec_from_file_location('experiment_profiles_cpu',ROOT/'dental_agent/training/experiment_profiles.py')
_profile_module=importlib.util.module_from_spec(_profile_spec);_profile_spec.loader.exec_module(_profile_module)
grpo_name=_profile_module.grpo_name


def load_script(name):
    path=ROOT/'scripts/bench'/name
    spec=importlib.util.spec_from_file_location(name.replace('.py',''),path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def safe_file(path,shapes):
    header={};offset=0
    for name,shape in shapes.items():
        size=1
        for d in shape:size*=d
        header[name]=dict(shape=shape,dtype='U8',data_offsets=[offset,offset+size]);offset+=size
    data=json.dumps(header).encode();data+=b' '*((-len(data))%8)
    path.write_bytes(struct.pack('<Q',len(data))+data+b'x'*offset)


def source_function(relative,name,namespace):
    node=next(n for n in ast.walk(ast.parse((ROOT/relative).read_text())) if isinstance(n,ast.FunctionDef) and n.name==name)
    for arg in [*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs]:arg.annotation=None
    node.returns=None
    exec(compile(ast.Module(body=[node],type_ignores=[]),relative,'exec'),namespace)
    return namespace[name]


def test_harness_options_and_checkpoint_names_are_distinct():
    assert harness_options('off') is None
    for mode,skill_mode in [(MODE,'index_only'),('rules_skills_full','full_inject')]:
        opt=harness_options(mode)
        assert opt.rules_from_file and opt.skills and opt.skills_mode==skill_mode
    names={m:grpo_name('with_tools',4,'dentex_alone','qlora','dual_t4_fp16',harness_mode=m) for m in HARNESS_MODES}
    assert len(set(names.values()))==3
    assert names['off']=='qwen3_5_9b_grpo_with_tools_k4_dentex_alone_qlora__dual_t4_fp16'
    with pytest.raises(ValueError,match='with_tools'):grpo_name('no_tools',4,'dentex_alone','qlora','tag',harness_mode=MODE)


def test_resume_rejects_other_harness_and_stale_rule_content():
    assert_harness_identity(None,'off')
    assert_harness_identity(harness_identity(MODE),MODE)
    with pytest.raises(ValueError):assert_harness_identity(None,MODE)
    with pytest.raises(ValueError):assert_harness_identity(harness_identity(MODE),'off')
    stale=harness_identity(MODE);stale['files']['rules.md']='changed'
    with pytest.raises(ValueError):assert_harness_identity(stale,MODE)


def test_actual_grpo_rollout_receives_harness_options():
    observed=[]
    def agent(**kwargs):observed.append(kwargs);return SimpleNamespace(to_dict=lambda:{'messages':[]})
    env=dict(run_agent=agent,combine_reward=lambda *a,**k:(1.0,{}))
    fn=source_function('dental_agent/training/grpo.py','collect_grpo_group_batched_with_tools',env)
    harness=harness_options(MODE)
    result=fn(SimpleNamespace(eval=lambda:None),object(),7,object(),[],object(),group_size=2,
              compute_old_log_probs=False,max_new_tokens=16384,harness=harness)
    assert len(result[0])==2
    assert all(x['harness'] is harness and x['max_new_tokens']==16384 for x in observed)


def test_standard_trained_adapter_eval_receives_harness_and_no_tools_refuses_it():
    env=dict(run_agent=lambda **k:k,run_agent_no_tools=lambda **k:k,harness_options=harness_options)
    fn=source_function('scripts/evaluate_models.py','run_condition_trajectory',env)
    args=SimpleNamespace(max_tool_calls=50,canonical_resize=True,dataset='dentex',max_new_tokens=16384,
                         max_turns=None,malformed_retries=0,harness_mode=MODE)
    result=fn(object(),object(),object(),9,object(),args)
    assert result['harness']==harness_options(MODE)
    with pytest.raises(ValueError,match='with_tools'):fn(object(),object(),object(),9,None,args)


def trained_files(mode):
    stem=grpo_name('with_tools',4,'dentex_alone','qlora','dual_t4_fp16',harness_mode=mode)
    root=f'grpo/{stem}/grpo-with_tools-final'
    prefix=root+'/grpo_policy'
    return {prefix+'/adapter_config.json':b'{}',prefix+'/adapter_model.safetensors':b'weights',
            root+'/checkpoint_meta.json':json.dumps({'extra':{'harness':harness_identity(mode)}}).encode()}


class Hub:
    def __init__(self,tmp):self.tmp=tmp;self.files={**trained_files('off'),**trained_files(MODE)}
    def model_info(self,*a,**k):
        return SimpleNamespace(sha='pinned',siblings=[SimpleNamespace(rfilename=n,blob_id=hashlib.sha256(b).hexdigest(),lfs=None) for n,b in self.files.items()])
    def download(self,repo,name,**kwargs):
        p=self.tmp/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(self.files[name]);return p
    def list_repo_files(self,*a,**k):return []


def test_discovery_selects_both_training_variants_but_mtp_only_final_harness_grpo(tmp_path):
    hub=Hub(tmp_path)
    session=w.NotebookEvaluations(api=hub,download=hub.download,kinds=['grpo','speculative'],grpo_harness_modes=['off',MODE],
        speculative_harness_mode=MODE,speculative_target_kind='grpo',track='with_tools')
    assert {j['harness_mode'] for j in session.jobs if j['kind']=='grpo'}=={'off',MODE}
    jobs=[j for j in session.jobs if j['kind']=='speculative']
    assert len(jobs)==1 and jobs[0]['harness_mode']==MODE and jobs[0]['checkpoint_kind']=='grpo'
    assert jobs[0]['adapter'].endswith('grpo-with_tools-final/grpo_policy')
    assert '--harness_'+MODE in jobs[0]['adapter']
    assert not session.status


def test_cohort_commands_carry_correct_harness_and_draft_identity(tmp_path):
    hub=Hub(tmp_path)
    session=w.NotebookEvaluations(api=hub,download=hub.download,kinds=['grpo','speculative'],grpo_harness_modes=['off',MODE],
        speculative_harness_mode=MODE,speculative_target_kind='grpo',track='with_tools',model_id='/same-base')
    session._saved_identity=lambda ident:None
    session._input=lambda split:dict(ids=list(range(50)),fingerprint='cohort')
    session.inputs={'test':{'data_dir':'/same-data'},'dev':{'data_dir':'/same-dev'}}
    session.dev_split=tmp_path/'dev.json';session.dev_split.write_text(json.dumps({'image_ids':[1,2]}))
    session._adapter=lambda job:'/exact/'+job['harness_mode']
    for job in session.jobs:
        job={**job,'size':'canonical'}
        if job['kind']=='speculative':
            session.spec_checkpoints={job['profile']:{job['adapter']:'/merged-final-harness-grpo'}}
            session._merged_identity=lambda path:{'exact':'target'}
        ident=session._identity(job)
        command,_,_=session._command(job,ident,tmp_path)
        if job['harness_mode']=='off':assert '--harness-mode' not in command
        else:assert command[command.index('--harness-mode')+1]==MODE
        if job['kind']=='speculative':
            assert ident['draft_variant']=='original'
            assert command[command.index('--checkpoint')+1]=='/merged-final-harness-grpo'
            assert command[command.index('--draft-variant')+1]=='original'
        if job['harness_mode']!='off':assert 'harness_'+MODE in w.result_path(ident)


def test_checkpoint_manifest_requires_final_harness_policy(tmp_path):
    prefix='grpo/experiment--harness_'+MODE+'/grpo-with_tools-final/grpo_policy'
    write_target_manifest(tmp_path,prefix,'grpo',MODE)
    assert require_harness_grpo_target(tmp_path,MODE)['source_adapter']==prefix
    with pytest.raises(ValueError):require_harness_grpo_target(tmp_path,'rules_skills_full')
    with pytest.raises(ValueError):write_target_manifest(tmp_path,'sft/other','sft',MODE)


def test_mtp_inventory_and_fingerprint_exclude_shared_target_weights(tmp_path):
    safe_file(tmp_path/'model.safetensors',{'mtp.fc.weight':[2,3],'model.weight':[5],'lm_head.weight':[3,4]})
    inventory=mtp_parameter_inventory(tmp_path)
    assert inventory['parameters']==6 and inventory['excludes_shared_embeddings_and_lm_head']
    before=target_tensor_fingerprint(tmp_path)
    safe_file(tmp_path/'model.safetensors',{'mtp.fc.weight':[4,3],'model.weight':[5],'lm_head.weight':[3,4]})
    assert target_tensor_fingerprint(tmp_path)==before
    safe_file(tmp_path/'model.safetensors',{'mtp.fc.weight':[4,3],'model.weight':[6],'lm_head.weight':[3,4]})
    assert target_tensor_fingerprint(tmp_path)!=before


def test_both_speculative_episode_calls_pass_the_same_harness():
    tree=ast.parse((ROOT/'scripts/bench/spec_decode_bench.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='evaluate_episode']
    assert len(calls)==2
    for call in calls:
        value=next(k.value for k in call.keywords if k.arg=='harness')
        assert isinstance(value,ast.Name) and value.id=='harness'


def test_acceptance_comparison_checks_only_requested_mtp_and_target_harness():
    mod=load_script('compare_acceptance.py')
    record=dict(image_id=1,final_answer=[],format_ok=True,turns=[dict(raw_output='same',tool_calls=[])],
                measurements={'drafted_tokens':10,'accepted_tokens':8})
    entry=dict(status='complete',episodes=[record],summary={'pooled_exact_f1':1.0})
    before=dict(settings=dict(conditions=['baseline','mtp_1'],checkpoint='before',output='a',draft_variant='original'),
                image_ids=[1],dev_split_sha256='dev',target_tensor_sha256='target',harness=harness_identity(MODE),
                conditions={'baseline':entry,'mtp_1':entry},gates={'mtp_1':{'passed':True}})
    after=json.loads(json.dumps(before));after['settings'].update(checkpoint='after',output='b',draft_variant='finetuned')
    result=mod.compare(before,after)
    assert set(result)=={'mtp_1'} and result['mtp_1']['status']=='valid'
    after['target_tensor_sha256']='different'
    with pytest.raises(ValueError,match='weights'):mod.compare(before,after)


def test_publish_filters_head_only_and_refuses_target_sized_artifacts(tmp_path):
    mod=load_script('publish_mtp.py')
    head=tmp_path/'checkpoints/checkpoint_best';head.mkdir(parents=True)
    stitched=tmp_path/'stitched';stitched.mkdir()
    (stitched/'mtp_draft_manifest.json').write_text(json.dumps(dict(draft_variant='finetuned',changed_mtp_tensors=['mtp.fc.weight'],mtp_inventory={'parameters':6},harness=harness_identity(MODE))))
    safe_file(head/'model.safetensors',{'mtp.fc.weight':[2,3]})
    (head/'optimizer.pt').write_bytes(b'do not upload')
    files,_=mod.draft_files(tmp_path)
    assert 'optimizer.pt' not in {n for n,_ in files} and 'mtp_draft_manifest.json' in {n for n,_ in files}
    safe_file(head/'model.safetensors',{'mtp.fc.weight':[2,3],'lm_head.weight':[100,100]})
    with pytest.raises(ValueError,match='extra weights'):mod.draft_files(tmp_path)


def test_grpo_notebook_exposes_two_runs_and_all_evals_receive_explicit_selectors():
    notebook=json.loads((ROOT/'notebooks/VLM_Dental_Colab_GRPO.ipynb').read_text())
    source='\n'.join(''.join(c['source']) for c in notebook['cells'])
    assert 'GRPO_HARNESS_MODES = ["off", "rules_skills_index"]' in source
    assert 'for _harness_mode in GRPO_HARNESS_MODES:' in source
    assert '"--harness-mode", _harness_mode' in source
    for short in ['SFT','GRPO','Rules_Skills','Speculative_Decoding','Final_Evaluation','ZeroShot']:
        source='\n'.join(''.join(c['source']) for c in json.loads((ROOT/f'notebooks/VLM_Dental_Colab_{short}.ipynb').read_text())['cells'])
        assert 'EVAL_GRPO_HARNESS_MODES' in source and 'speculative_harness_mode=SPEC_HARNESS_MODE' in source
        assert 'speculative_target_kind=SPEC_TARGET_KIND' in source


def test_regeneration_uses_final_target_harness_and_excludes_dev(tmp_path,monkeypatch):
    import pandas as pd
    script=load_script('regenerate_train.py')
    target=tmp_path/'final-target';target.mkdir()
    prefix='grpo/run--harness_'+MODE+'/grpo-with_tools-final/grpo_policy'
    write_target_manifest(target,prefix,'grpo',MODE)
    safe_file(target/'model.safetensors',{'mtp.fc.weight':[2,3],'model.weight':[5]})
    traces=tmp_path/'train_cot_traces_dentex.jsonl'
    traces.write_text(''.join(json.dumps(dict(image_id=i,dataset='dentex',track='with_tools'))+'\n' for i in [1,2,3]))
    out=tmp_path/'conversations.jsonl';calls=[]
    def module(name,**members):
        m=ModuleType(name);m.__dict__.update(members);monkeypatch.setitem(sys.modules,name,m)
    module('dental_agent.data.dentex',load_dentex_dataset=lambda **kw:(pd.DataFrame([dict(id=i,source_file='train') for i in [1,2,3]]),None,None))
    module('dental_agent.data.dev_split',load_dev_keys=lambda path:{('dentex',2)},assert_not_test_split=lambda frame:None,image_key=lambda dataset,i:(dataset,i))
    def atomic(path,obj):
        Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(obj))
    module('dental_agent.harness.evaluation',digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest(),atomic_json=atomic)
    import dental_agent.harness.specdec as specdec
    monkeypatch.setattr(specdec,'http_json',lambda url:{'data':[{'id':str(target.resolve())}]})
    class Reply:
        def __init__(self,*a,**k):self.requests=[{'finish_reason':'stop'}]
    monkeypatch.setattr(specdec,'ServingReply',Reply)
    def agent(iid,images,*args,**kwargs):
        calls.append((iid,kwargs))
        return SimpleNamespace(format_ok=True,messages=[dict(role='assistant',content='target response')])
    module('dental_agent.agent.loop',run_agent=agent)
    monkeypatch.setattr(sys,'argv',['regenerate_train.py','--traces',str(traces),'--output',str(out),
        '--checkpoint',str(target),'--harness-mode',MODE,'--max-new-tokens','16384'])
    script.main()
    assert {iid for iid,_ in calls}=={1,3}
    assert all(k['harness']==harness_options(MODE) and k['max_new_tokens']==16384 for _,k in calls)
    provenance=json.loads(Path(str(out)+'.provenance.json').read_text())
    assert provenance['harness']==harness_identity(MODE)
    assert provenance['target_manifest']['source_adapter']==prefix
    # The MTP planner accepts only these exact conversations/target/harness.
    trainer=load_script('mtp_finetune.py');workflow=tmp_path/'workflow'
    monkeypatch.setattr(sys,'argv',['mtp_finetune.py','--checkpoint',str(target),'--conversations',str(out),
        '--output-dir',str(workflow),'--harness-mode',MODE,'--stage','prepare'])
    trainer.main()
    plan=json.loads((workflow/'mtp_workflow.json').read_text())
    assert plan['mtp_inventory']['parameters']==6 and plan['harness']==harness_identity(MODE)
    # Altering the target, while keeping its path/name, must invalidate these conversations.
    safe_file(target/'model.safetensors',{'mtp.fc.weight':[2,3],'model.weight':[6]})
    with pytest.raises(ValueError,match='exact final target'):trainer.main()


def test_restored_head_requires_same_target_and_preserves_main_weights(tmp_path,monkeypatch):
    script=load_script('restore_mtp.py')
    target=tmp_path/'target';target.mkdir()
    prefix='grpo/run--harness_'+MODE+'/grpo-with_tools-final/grpo_policy'
    target_info=write_target_manifest(target,prefix,'grpo',MODE)
    safe_file(target/'model.safetensors',{'mtp.fc.weight':[2,3],'model.weight':[5]})
    head=tmp_path/'head';head.mkdir()
    data=dict(draft_variant='finetuned',target_manifest=target_info,harness=harness_identity(MODE),
              target_tensor_sha256=target_tensor_fingerprint(target),mtp_inventory=mtp_parameter_inventory(target))
    (head/'mtp_draft_manifest.json').write_text(json.dumps(data))
    monkeypatch.setattr(sys,'argv',['restore_mtp.py','--checkpoint',str(target),'--draft-dir',str(head),
        '--output',str(tmp_path/'new-output'),'--harness-mode',MODE])
    script.main()  # Plan only: no server, stitch or model load.
    assert not (tmp_path/'new-output').exists()
    data['target_tensor_sha256']='foreign';(head/'mtp_draft_manifest.json').write_text(json.dumps(data))
    with pytest.raises(ValueError,match='different target'):script.main()


def test_published_harness_comparison_is_enabled_minus_control(tmp_path,monkeypatch):
    """Check grouping/sign and upload routing, without mocking numerical correctness."""
    observed=[]
    matrix=ModuleType('dental_agent.evaluation.matrix')
    matrix.build_summary=lambda cells,**kw:dict(cells={n:{} for n in cells},comparisons=[])
    matrix.summary_markdown=lambda report:'summary'
    def paired(a,b,**kwargs):
        observed.append((a[0]['mode'],b[0]['mode'],kwargs['level']))
        return dict(mean_diff=1.0)
    matrix.paired_bootstrap_diff=paired
    monkeypatch.setitem(sys.modules,'dental_agent.evaluation.matrix',matrix)
    import huggingface_hub
    class API:
        def repo_info(self,*a,**k):return SimpleNamespace(sha='revision')
        def create_commit(self,*a,**k):return None
    monkeypatch.setattr(huggingface_hub,'HfApi',lambda **kw:API())
    status=[]
    for mode in ['off',MODE]:
        folder=tmp_path/mode;folder.mkdir()
        (folder/'results.jsonl').write_text(json.dumps(dict(image_id=1,mode=mode))+'\n')
        status.append(dict(status='complete',kind='grpo',profile='dual_t4',stage='dentex_alone',k=4,
                           condition='grpo_with_tools',size='canonical',path=str(folder),
                           results_repo='existing/evaluations',results_repo_type='dataset',
                           identity=dict(split='dev',cohort_sha256='cohort',harness_mode=mode)))
    published=w.publish_paired_summaries(status,output_dir=tmp_path/'summaries')
    assert observed==[(MODE,'off','exact'),(MODE,'off','fdi')]
    enabled=next(p for p in published if p['harness_mode']==MODE)
    report=json.loads((Path(enabled['local_path'])/'summary.json').read_text())
    assert report['comparisons'][0]['a']=='grpo_with_tools@canonical'
    assert report['comparisons'][0]['b']=='grpo_with_tools_harness_off@canonical'


def test_sft_preflight_naming_never_receives_grpo_harness_arguments():
    tree=ast.parse((ROOT/'dental_agent/training/preflight.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='sft_hub_subfolder']
    assert len(calls)==2
    assert all(not any(k.arg=='harness_mode' for k in call.keywords) for call in calls)


def test_evaluation_cli_bootstraps_repository_before_project_imports():
    tree=ast.parse((ROOT/'scripts/evaluate_models.py').read_text())
    bootstrap=next(n.lineno for n in tree.body if isinstance(n,ast.If) and 'repo_root not in sys.path' in ast.unparse(n.test))
    imports=[n for n in tree.body if isinstance(n,ast.ImportFrom) and (n.module or '').startswith('dental_agent')]
    assert imports and all(n.lineno>bootstrap for n in imports)
