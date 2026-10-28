"""Synthetic integration: real loop, parser, dispatcher and tool; scripted generation only."""
import json
from pathlib import Path
import pandas as pd
import pytest
from PIL import Image
from dental_agent.agent.loop import run_agent
from dental_agent.harness.skills import HarnessOptions
from dental_agent.tools.registry import ToolRegistry

def test_flags_off_golden_and_on_demand(tmp_path):
    image=tmp_path/'synthetic.png';Image.new('RGB',(120,80)).save(image)
    images=pd.DataFrame({'id':[99999],'local_path':[str(image)]})
    def run(options,replies):
        replies=iter(replies)
        def generate(*a,**kw):return next(replies),1,[1],None,None
        registry=ToolRegistry.create_default()
        traj=run_agent(99999,images,None,None,registry=registry,reply_fn=generate,harness=options,verbose=False)
        assert len(registry.list_tools())==8
        return traj
    final=json.dumps({'thought':'synthetic','final_answer':[]})
    assert run(None,[final]).to_dict()==run(HarnessOptions(),[final]).to_dict()
    action=json.dumps({'thought':'load','tool_calls':[{'tool':'read_skill','args':{'name':'box-verification'}}]})
    traj=run(HarnessOptions(rules_from_file=True,skills=True,skills_mode='index_only'),[action,final])
    assert traj.tool_calls==0 and traj.format_ok and traj.harness_events[-1]['event']=='loaded'
    assert any('box-verification' in str(m) for m in traj.messages)

def test_paired_metrics():
    from dental_agent.harness.evaluation import pooled,compare_paired
    a=[dict(image_id=9,exact_tp=1,exact_fp=0,exact_fn=0,format_ok=True)]
    assert pooled(a)['pooled_exact_f1']==1
    assert compare_paired(a,a,n_boot=10)['pooled_exact_f1']['delta']==0
    with pytest.raises(ValueError):compare_paired(a,[],n_boot=10)

def test_dev_episode_plumbing_end_to_end(tmp_path):
    from dental_agent.harness.evaluation import evaluate_episode,CONDITIONS,atomic_json
    image=tmp_path/'synthetic.png';Image.new('RGB',(64,48)).save(image)
    images=pd.DataFrame({'id':[99999],'local_path':[str(image)],'source_file':['synthetic_train.json']})
    annotations=pd.DataFrame(columns=['image_id','category_id_3'])
    final=json.dumps({'thought':'synthetic only','final_answer':[]})
    def generate(*a,**kw):return final,1,[1],None,None
    for name,options in CONDITIONS.items():
        record,_=evaluate_episode(99999,images,annotations,None,harness=options,reply_fn=generate,max_turns=1)
        assert record['format_ok'] and record['exact_fp']==0 and record['exact_fn']==0
        path=tmp_path/(name+'.json');atomic_json(path,record)
        assert json.loads(path.read_text())['final_answer']==[]

def test_dev_loader_requires_provenance(tmp_path,monkeypatch):
    from scripts import evaluate_models as ev
    from dental_agent.harness.evaluation import load_dev
    image=tmp_path/'synthetic.png';Image.new('RGB',(8,8)).save(image)
    frame=pd.DataFrame({'id':[99999],'local_path':[str(image)],'source_file':['synthetic_validation.json']})
    def load(args):
        assert args.split=='dev'
        return frame,pd.DataFrame(),pd.DataFrame()
    monkeypatch.setattr(ev,'load_eval_split',load)
    with pytest.raises(SystemExit,match='held-out test'):load_dev('data','fixed_dev.json',limit=1)
    frame['source_file']='synthetic_train.json'
    ids,_,_,_=load_dev('data','fixed_dev.json',limit=1)
    assert ids==[99999]
