"""CPU-only orchestration contracts. These do not certify CUDA/TPU generation or live Hub access."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('notebook_eval',ROOT/'dental_agent/harness/notebook_eval.py')
w = importlib.util.module_from_spec(spec);spec.loader.exec_module(w)


def metadata(files):
    return {n:SimpleNamespace(rfilename=n,blob_id=w.fingerprint(n),lfs=None) for n in files}


def trained_files():
    files=[]
    for name,p in w.PROFILES.items():
        if name=='dual_t4_fp16': continue
        for track in ('with_tools','no_tools'):
            for prefix in (f"sft/qwen3_5_9b_sft_{track}_dentex_alone_{p['precision']}__{p['tag']}",
                           f"grpo/qwen3_5_9b_grpo_{track}_k4_dentex_alone_{p['precision']}__{p['tag']}/grpo-{track}-final/grpo_policy"):
                files += [prefix+'/adapter_config.json',prefix+'/adapter_model.safetensors']
    return metadata(files)


class Hub:
    def __init__(self,tmp):self.tmp=tmp;self.files={};self.n=0;self.commits=[]
    def model_info(self,repo,**kwargs):return SimpleNamespace(sha=str(self.n),siblings=list(trained_files().values()))
    def list_repo_files(self,repo,**kwargs):
        assert kwargs['repo_type']=='model'
        return list(self.files)
    def download(self,repo,name,**kwargs):
        path=self.tmp/'cache'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(self.files[name]);return str(path)
    def create_commit(self,repo,**kwargs):
        assert kwargs['repo_type']=='model' and kwargs['parent_commit']==str(self.n)
        for op in kwargs['operations']:self.files[op.path_in_repo]=op.path_or_fileobj
        self.n+=1;self.commits.append(kwargs)


def ident(kind='baseline'):
    return dict(profile='dual_t4',kind=kind,condition='base_no_tools',split='test',image_ids=[90001,90002],base_model='Qwen/Qwen3.5-9B',adapter=None)


def row(i=90001):
    return dict(image_id=i,split='test',condition='base_no_tools',prompt='unified',eval_settings=dict(max_new_tokens=16384,adapter_path='/local'),predictions=[])


def test_profiles_aliases_and_all_no_duplicates():
    assert w.select_profiles(['rtx4090','rtx4090_qlora'])==['rtx4090_qlora']
    assert w.select_profiles('rtxpro6000_48gb')==['workstation48_lora']
    assert len(w.select_profiles('all'))==len(w.PROFILES)
    with pytest.raises(ValueError):w.select_profiles('made_up')


def test_discovery_both_tracks_all_profiles_and_reference_exclusion():
    files=trained_files()
    files.update(metadata(['grpo/ignored/grpo_reference/adapter_config.json']))
    jobs=w.discover(files,list(w.PROFILES))
    assert len(jobs)==16
    assert len(w.discover(files,['dual_t4']))==4
    assert {j['condition'] for j in jobs}==set(w.PRIMARY[2:])
    assert all('grpo_reference' not in j['adapter'] for j in jobs)
    assert not w.discover(files,['dual_t4'],stages='other')
    assert all(j['kind']=='sft' for j in w.discover(files,['dual_t4'],selected_k=8))
    assert len(w.discover(files,['dual_t4'],selected_k='all'))==4


def test_hardware_flags_do_not_silently_change_precision():
    t4=[dict(capability=(7,5),gib=15,name='T4')]*2
    assert w.hardware_reason('dual_t4','sft',t4) is None
    assert 'BF16' in w.hardware_reason('rtx4090_qlora','sft',t4)
    assert 'TPU' in w.hardware_reason('tpu','sft',[])
    assert '--qlora-compute-dtype' in w.runtime_flags('dual_t4')
    assert w.runtime_flags('workstation48_lora')[1]=='bf16'


def test_store_resume_longer_progress_and_atomic_identity_guards(tmp_path):
    h=Hub(tmp_path);store=w.HubResults('existing/models',h,h.download)
    payload={'results.jsonl':(json.dumps(row())+'\n').encode()}
    store.save(tmp_path,'evaluations/run',ident(),payload,1,False)
    store.save(tmp_path,'evaluations/run',ident(),payload,1,False)
    assert len(h.commits)==1
    out=tmp_path/'restore';store.restore(out,'evaluations/run',ident());assert (out/'results.jsonl').is_file()
    more={'results.jsonl':payload['results.jsonl']+(json.dumps(row(90002))+'\n').encode()}
    store.save(tmp_path,'evaluations/run',ident(),more,2,True)
    store.restore(out,'evaluations/run',ident());assert len(w.jsonl_rows(out/'results.jsonl'))==2
    with pytest.raises(ValueError,match='more progress'):store.save(tmp_path,'evaluations/run',ident(),payload,1,False)
    with pytest.raises(ValueError,match='identity'):store.restore(out,'evaluations/run',dict(ident(),profile='other'))
    corrupted=json.loads(more['results.jsonl'].splitlines()[0]);corrupted['predictions']=[{'invented':True}]
    bad={'results.jsonl':(json.dumps(corrupted)+'\n').encode()+more['results.jsonl'].splitlines(keepends=True)[1]}
    with pytest.raises(ValueError,match='Conflicting'):store.save(tmp_path,'evaluations/run',ident(),bad,2,True)


def test_remote_checksum_conflict_is_not_overwritten(tmp_path):
    h=Hub(tmp_path);store=w.HubResults('existing/models',h,h.download)
    store.save(tmp_path,'evaluations/run',ident(),{'results.jsonl':(json.dumps(row())+'\n').encode()},1,False)
    h.files['evaluations/run/results.jsonl']=b'corrupt'
    with pytest.raises(ValueError,match='checksum'):store.restore(tmp_path/'new','evaluations/run',ident())


def test_partial_tail_budget_duplicate_and_split_gates(tmp_path):
    file=tmp_path/'results.jsonl';file.write_text(json.dumps(row())+'\n{"image_id":')
    payload,n,done=w.validate_document('baseline',file,ident());assert n==1 and not done
    for bad in [dict(row(),split='dev'),dict(row(),prompt='zero_shot_cached'),dict(row(),eval_settings={'max_new_tokens':2048})]:
        file.write_text(json.dumps(bad)+'\n')
        with pytest.raises(ValueError):w.validate_document('baseline',file,ident())
    file.write_text((json.dumps(row())+'\n')*2)
    with pytest.raises(ValueError,match='Duplicate'):w.validate_document('baseline',file,ident())


def test_extension_resume_retains_path_changes_and_rejects_decoding_changes():
    args=dict(max_new_tokens=16384,precision='qlora',seed=42,output='/new',adapter_path='/new/adapter',existing_results='resume')
    saved=dict(settings={**args,'output':'/old','adapter_path':'/old'},image_ids=[90001],conditions={'baseline':{'records':[{'image_id':90001}]}})
    w.validate_resume_settings(saved,args,[90001])
    with pytest.raises(ValueError,match='max_new_tokens'):w.validate_resume_settings(saved,{**args,'max_new_tokens':2048},[90001])
    with pytest.raises(ValueError,match='cohort'):w.validate_resume_settings(saved,args,[90002])


def test_merged_partial_extension_documents_preserve_all_conditions():
    a=dict(status='running',conditions={'baseline':dict(records=[dict(image_id=90001)])})
    b=dict(status='running',conditions={'rules_only':dict(records=[dict(image_id=90001)])})
    result=json.loads(w.merge_evidence('results.json',json.dumps(a).encode(),json.dumps(b).encode()))
    assert set(result['conditions'])=={'baseline','rules_only'}


def test_complete_reuse_before_any_model_or_image_resolution(tmp_path,monkeypatch):
    hub=Hub(tmp_path)
    session=w.NotebookEvaluations(kinds=['baseline'],api=hub,download=hub.download,output_dir=tmp_path)
    session.jobs=session.jobs[:1]
    identity=ident();identity['kind']='baseline'
    session._identity=lambda job:identity
    session._input=lambda split:pytest.fail('complete evidence should not resolve/download images')
    session._command=lambda *a:pytest.fail('complete evidence should not load a checkpoint')
    prefix=w.result_path(identity)
    data=''.join(json.dumps(row(i))+'\n' for i in identity['image_ids']).encode()
    file=tmp_path/'canonical.jsonl';file.write_bytes(data)
    canonical,_,_=w.validate_document('baseline',file,identity)
    session.store.save(tmp_path,prefix,identity,canonical,2,True)
    status=session.run(execute=False)
    assert status[0]['status']=='complete_reused'


def test_source_checkpoint_identity_unaffected_by_result_repo_commits(tmp_path):
    h=Hub(tmp_path);session=w.NotebookEvaluations(kinds=['sft'],api=h,download=h.download)
    job=session.jobs[0];names=(job['adapter']+'/adapter_config.json',job['weight'])
    before={n:w._metadata_digest(session.files[n]) for n in names}
    h.n+=1
    h.files['evaluations/unrelated/results.jsonl']=b'other results'
    assert before=={n:w._metadata_digest(session.files[n]) for n in names}


def test_input_resolution_uses_all_official_images_and_local_assets(tmp_path):
    native=tmp_path/'dentex';native.mkdir();images=[];annotations=[]
    for i in range(90001,90051):
        name=f'val_{i}.png';Image.new('RGB',(8,6),color=i%255).save(native/name)
        images.append(dict(id=i,file_name=name,width=8,height=6));annotations.append(dict(image_id=i,category_id_3=0))
    source=native/'validation_triple.json';source.write_text(json.dumps(dict(images=images,annotations=annotations,categories=[])))
    fail=lambda *a,**k:pytest.fail('local cohort must not download')
    result=w.prepare_inputs(tmp_path,'test',None,SimpleNamespace(),fail,None,files=sorted(native.iterdir()))
    assert len(result['ids'])==50
    prepared=json.loads((Path(result['data_dir'])/'validation_triple.json').read_text())
    assert all(Path(i['file_name']).is_file() for i in prepared['images'])


def test_test_images_are_not_used_for_dev_numeric_ids(tmp_path):
    root=tmp_path/'source';root.mkdir();dev=tmp_path/'dev.json';dev.write_text(json.dumps({'image_ids':[90001]}))
    (root/'train.json').write_text(json.dumps(dict(images=[dict(id=90001,file_name='train_90001.png',width=8,height=6)],annotations=[dict(image_id=90001,category_id_3=0)],categories=[])))
    val=root/'validation_images';val.mkdir();Image.new('RGB',(8,6)).save(val/'90001.png')
    api=SimpleNamespace(dataset_info=lambda repo:SimpleNamespace(sha='x'),list_repo_files=lambda *a,**k:[])
    with pytest.raises(FileNotFoundError):w.prepare_inputs(tmp_path,'dev',dev,api,lambda *a,**k:None,None,files=[root/'train.json',val/'90001.png'])


def test_notebook_cells_schema_syntax_visible_profiles_and_no_paper_build():
    import nbformat
    from IPython.core.interactiveshell import InteractiveShell
    shell=InteractiveShell.instance()
    for name in ['SFT','GRPO','Rules_Skills','Speculative_Decoding','Final_Evaluation','ZeroShot']:
        nb=nbformat.read(ROOT/f'notebooks/VLM_Dental_Colab_{name}.ipynb',as_version=4);nbformat.validate(nb)
        source='\n'.join(c.source for c in nb.cells)
        assert 'EVAL_TARGET_PROFILES' in source and 'EVAL_PROFILES' in source
        assert 'EVAL_PROFILES = copy.deepcopy(PROFILES)' not in source
        for cell in nb.cells:
            if cell.cell_type != 'code': continue
            tree = ast.parse(shell.input_transformer_manager.transform_cell(cell.source))
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'EVAL_PROFILES' for t in node.targets):
                    assert isinstance(node.value, ast.Dict)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'NotebookEvaluations':
                    keywords = {k.arg:k.value for k in node.keywords}
                    for arg, variable in [('profiles','EVAL_PROFILES'), ('track','EVAL_TRACK'), ('results_repo','EVAL_RESULTS_REPO')]:
                        assert isinstance(keywords[arg], ast.Name) and keywords[arg].id == variable
        for cell in nb.cells:
            if cell.cell_type=='code':ast.parse(shell.input_transformer_manager.transform_cell(cell.source))
        if name in ['Rules_Skills','Speculative_Decoding','Final_Evaluation']:
            assert 'MAX_NEW_TOKENS = 16384' in source and 'MAX_NEW_TOKENS = 2048' not in source
        if name=='Final_Evaluation':
            for forbidden in ['build_paper.py','generate_paper_assets.py','BUILD_PAPER_PDF','create_repo(']:assert forbidden not in source
            assert all(kind in source for kind in w.KINDS)


def test_command_contracts_are_explicit_16k_and_separate_adapters(tmp_path):
    h=Hub(tmp_path);session=w.NotebookEvaluations(api=h,download=h.download,model_id='/base')
    session.inputs={'test':dict(data_dir='/input'),'dev':dict(data_dir='/dev')}
    session._adapter=lambda job:'/exact/'+job['condition']
    for job in session.jobs:
        if job['kind']=='speculative':continue
        identity=dict(split='dev' if job['kind']=='rules_skills' else 'test',size='canonical',checkpoint_kind=job.get('checkpoint_kind',job['kind']),conditions=['baseline','rules_only'])
        command,result,adapter=session._command(job,identity,tmp_path)
        assert '16384' in command and '--reuse-cached' not in command and '--limit' not in command
        if job['kind']=='zeroshot':assert '--data-dir' in command and '--git-sync-every' in command
        if job['adapter']:assert adapter=='/exact/'+job['condition']


def test_local_adapter_reuse_requires_exact_hub_contents(tmp_path):
    file=tmp_path/'adapter_config.json';file.write_bytes(b'{}')
    import hashlib
    blob=hashlib.sha1(b'blob 2\0{}').hexdigest()
    assert w.matches_hub_file(file,SimpleNamespace(lfs=None,blob_id=blob))
    assert w.matches_hub_file(file,SimpleNamespace(lfs={'sha256':w.digest(file)},blob_id=None))
    file.write_bytes(b'changed')
    assert not w.matches_hub_file(file,SimpleNamespace(lfs=None,blob_id=blob))


def test_adapter_signature_catches_weight_changes_and_grpo_reference(tmp_path):
    folder=tmp_path/'grpo_policy';folder.mkdir()
    (folder/'adapter_config.json').write_text('{}');(folder/'adapter_model.safetensors').write_bytes(b'policy')
    old=w.adapter_signature(tmp_path)
    (folder/'adapter_model.safetensors').write_bytes(b'new policy')
    assert w.adapter_signature(tmp_path)!=old
    with pytest.raises(ValueError,match='dev'):w.NotebookEvaluations(ablation={'remove_tools':['nudge_crop']},split='test')


def test_confirmed_parent_commit_conflict_rechecks_before_retry(tmp_path):
    h=Hub(tmp_path);original=h.create_commit;attempts=[]
    def commit(repo,**kwargs):
        attempts.append(1)
        if len(attempts)==1:
            h.n+=1
            error=RuntimeError('concurrent unrelated checkpoint commit');error.response=SimpleNamespace(status_code=409)
            raise error
        return original(repo,**kwargs)
    h.create_commit=commit
    store=w.HubResults('existing/models',h,h.download)
    store.save(tmp_path,'evaluations/run',ident(),{'results.jsonl':(json.dumps(row())+'\n').encode()},1,False)
    assert len(attempts)==2 and len(h.commits)==1


@pytest.mark.parametrize('track', ['with_tools', 'no_tools'])
def test_selected_track_filters_baselines_and_adapters(tmp_path, monkeypatch, track):
    monkeypatch.setattr(w.NotebookEvaluations, '_source_identity', lambda self: 'offline-source')
    hub = Hub(tmp_path)
    session = w.NotebookEvaluations(track=track, kinds=['baseline','sft','grpo'], api=hub, download=hub.download)
    assert {job['condition'] for job in session.jobs} == {'base_'+track, 'sft_'+track, 'grpo_'+track}
    assert all(job['track'] == track for job in session.jobs)
    assert not session.status


def test_sft_available_grpo_missing_is_reported_for_selected_track(tmp_path, monkeypatch):
    monkeypatch.setattr(w.NotebookEvaluations, '_source_identity', lambda self: 'offline-source')
    class SftOnly(Hub):
        def model_info(self, repo, **kwargs):
            return SimpleNamespace(sha='sft-sha', siblings=[v for k,v in trained_files().items() if k.startswith('sft/')])
    hub = SftOnly(tmp_path)
    session = w.NotebookEvaluations(track='with_tools', kinds=['baseline','sft','grpo'], api=hub, download=hub.download)
    assert {j['condition'] for j in session.jobs} == {'base_with_tools','sft_with_tools'}
    assert session.status == [dict(profile='dual_t4', kind='grpo', track='with_tools', condition='grpo_with_tools', status='missing_checkpoint', reason='No exact uploaded adapter for selected track/stage/K/profile')]


def test_cell_profile_edits_drive_discovery_identity_and_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(w.NotebookEvaluations, '_source_identity', lambda self: 'offline-source')
    profiles = copy.deepcopy(w.PROFILES)
    profiles['dual_t4'].update(tag='my_edited_tag', attention='eager', kv='off', threads=7, letterbox=True)
    class EditedHub(Hub):
        def model_info(self, repo, **kwargs):
            prefix = 'sft/qwen3_5_9b_sft_with_tools_dentex_alone_qlora__my_edited_tag'
            return SimpleNamespace(sha='edited', siblings=list(metadata([prefix+'/adapter_config.json', prefix+'/adapter_model.safetensors']).values()))
    hub = EditedHub(tmp_path)
    session = w.NotebookEvaluations(profiles=profiles, track='with_tools', kinds=['sft'], api=hub, download=hub.download, model_id='/base')
    assert len(session.jobs) == 1 and not session.status
    assert session.jobs[0]['adapter'].endswith('__my_edited_tag')
    session._saved_identity = lambda identity: None
    session._input = lambda split: dict(ids=list(range(50)), fingerprint='cohort')
    job = dict(session.jobs[0], size='canonical')
    identity = session._identity(job)
    assert identity['runtime'] == profiles['dual_t4'] and identity['track'] == 'with_tools'
    session.inputs = {'test': dict(data_dir='/inputs')}
    session._adapter = lambda job: '/adapter'
    command, _, _ = session._command(job, identity, tmp_path)
    for flag, value in [('--attn-implementation','eager'), ('--sdpa-repeat-kv','off'), ('--load-cpu-threads','7')]:
        assert command[command.index(flag)+1] == value
    assert '--canonical-letterbox' in command
    assert w.PROFILES['dual_t4']['tag'] == 'dual_t4_fp16'


def test_result_path_has_readable_profile_stage_condition_split_size_and_k():
    identity = dict(profile='dual_t4', runtime=w.PROFILES['dual_t4'], kind='grpo', condition='grpo_with_tools', split='test', stage='dentex_alone', k=4, size='canonical')
    path = w.result_path(identity)
    assert path.startswith('evaluations/dentex/test/dual_t4_fp16/dentex_alone/grpo/grpo_with_tools/k4/canonical/run_')
    assert w.result_path({**identity, 'k':8}) != path
    assert w.result_path({**identity, 'split':'dev'}) != path
    assert w.result_path({**identity, 'seed':99}) != path


def test_results_destination_is_separate_from_checkpoint_discovery(tmp_path, monkeypatch):
    monkeypatch.setattr(w.NotebookEvaluations, '_source_identity', lambda self: 'offline-source')
    monkeypatch.setenv('HF_ARTIFACT_REPO', 'owner/models')
    class SplitHub(Hub):
        def __init__(self, tmp): super().__init__(tmp); self.calls=[]
        def repo_info(self, repo, **kwargs):
            self.calls.append(('repo_info',repo,kwargs['repo_type']))
            return SimpleNamespace(sha='results-sha')
        def model_info(self, repo, **kwargs):
            assert repo == 'owner/models'
            return super().model_info(repo, **kwargs)
        def list_repo_files(self, repo, **kwargs):
            assert repo == 'owner/evaluations' and kwargs['repo_type'] == 'dataset'
            return list(self.files)
        def download(self, repo, name, **kwargs):
            assert repo == 'owner/evaluations' and kwargs['repo_type'] == 'dataset'
            return super().download(repo, name, **kwargs)
        def create_commit(self, repo, **kwargs):
            assert repo == 'owner/evaluations' and kwargs['repo_type'] == 'dataset'
            for op in kwargs['operations']: self.files[op.path_in_repo] = op.path_or_fileobj
            self.n += 1
    hub = SplitHub(tmp_path)
    session = w.NotebookEvaluations(kinds=['baseline'], api=hub, download=hub.download, results_repo='owner/evaluations', results_repo_type='dataset', track='no_tools')
    assert session.repo == 'owner/models' and session.results_repo == 'owner/evaluations'
    assert session.store.repo == 'owner/evaluations' and session.store.repo_type == 'dataset'
    session.list()
    assert session._saved_identity({'unmatched':True}) is None
    identity=ident(); prefix=w.result_path(identity)
    session.store.save(tmp_path, prefix, identity, {'test.txt':b'content'}, 1, False)
    session.store.restore(tmp_path/'restored', prefix, identity)
    assert (tmp_path/'restored/test.txt').read_bytes() == b'content'


def test_invalid_track_is_rejected_before_hub_calls():
    with pytest.raises(ValueError, match='EVAL_TRACK'):
        w.NotebookEvaluations(track='typo')
