"""CPU subprocess/upload contracts; no models, GPU inference or live Hub calls."""
import itertools
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from dental_agent.harness import notebook_eval as w

ROOT = Path(__file__).resolve().parents[1]


def notebook_source(name):
    notebook = json.loads((ROOT/'notebooks'/name).read_text())
    return [''.join(c['source']) for c in notebook['cells']]


def test_optional_mtp_comparison_imports_copy_before_using_it():
    source = next(s for s in notebook_source('VLM_Dental_Colab_Speculative_Decoding.ipynb') if 'AFTER_TARGETS = copy.deepcopy' in s)
    prefix = source[:source.index('    AFTER_TARGETS[')]
    original = {'profile': {'adapter': 'before'}}
    env = dict(RUN_MTP_FINETUNE=True, SPEC_CHECKPOINTS=original)
    exec(prefix, env)
    env['AFTER_TARGETS']['profile']['adapter'] = 'after'
    assert original['profile']['adapter'] == 'before'


@pytest.mark.parametrize('exit_code', [0, 7])
def test_grpo_launch_checks_real_subprocess_exit_code(exit_code):
    source = next(s for s in notebook_source('VLM_Dental_Colab_GRPO.ipynb') if 'subprocess.run(cmd, check=True)' in s)
    launch = 'import subprocess\n'+source.rsplit('import subprocess\n', 1)[1]
    env = dict(cmd=[sys.executable, '-c', f'raise SystemExit({exit_code})'])
    if exit_code:
        with pytest.raises(subprocess.CalledProcessError) as error: exec(launch, env)
        assert error.value.returncode == exit_code
    else: exec(launch, env)
    # Cleanup policy is intentionally unchanged.
    for pattern in ['multiprocessing.spawn', 'multiprocessing.resource_tracker', 'train_sft.py', 'run_grpo.py']:
        assert f'!pkill -9 -f "{pattern}"' in source
    assert '!fuser -k -9 /dev/vfio/*' in source


def session_with_store(store):
    session = w.NotebookEvaluations.__new__(w.NotebookEvaluations)
    session.store = store
    return session


def test_upload_retry_preserves_local_results_and_clears_pending_on_success(tmp_path, monkeypatch):
    attempts = []
    def save(*args):
        attempts.append(args)
        if len(attempts) < 3: raise ConnectionError('offline')
    session = session_with_store(SimpleNamespace(save=save))
    payloads = {'results.jsonl': b'local evidence\n'}
    file = tmp_path/'results.jsonl'; file.write_bytes(payloads['results.jsonl'])
    row = {}
    monkeypatch.setattr(w.time, 'sleep', lambda delay: None)
    assert not session._upload_results(tmp_path, 'evaluations/test', {}, payloads, 1, False, row)
    assert row['upload_status'] == 'pending' and (tmp_path/'upload_pending.json').is_file()
    assert file.read_bytes() == payloads['results.jsonl']
    assert session._upload_results(tmp_path, 'evaluations/test', {}, payloads, 1, True, row, attempts=3)
    assert len(attempts) == 3 and row['upload_status'] == 'synced'
    assert 'pending_upload' not in row and not (tmp_path/'upload_pending.json').exists()


@pytest.mark.parametrize('status,retryable', [(401,False),(403,False),(404,False),(409,False),(408,True),(429,True),(500,True),(503,True)])
def test_only_temporary_http_errors_are_deferred(status, retryable):
    error = RuntimeError('HTTP error'); error.response = SimpleNamespace(status_code=status)
    assert w.transient_upload_error(error) is retryable


@pytest.mark.parametrize('error', [ValueError('identity conflict'), PermissionError('disk denied'), RuntimeError('bad evidence')])
def test_upload_does_not_swallow_data_conflicts_or_local_errors(tmp_path, error):
    def save(*args): raise error
    session = session_with_store(SimpleNamespace(save=save))
    with pytest.raises(type(error), match=str(error)):
        session._upload_results(tmp_path, 'evaluations/test', {}, {}, 1, True, {})


@pytest.mark.parametrize('module,name', [('httpx','TransportError'),('httpx2','TransportError'),('requests.exceptions','ConnectionError'),('requests.exceptions','Timeout')])
def test_supported_transport_error_families(module,name):
    error_type = type(name,(Exception,),{'__module__':module})
    assert w.transient_upload_error(error_type('offline'))


def test_running_cpu_child_finishes_despite_persistent_upload_outage(tmp_path,monkeypatch):
    """Exercise run() with a real CPU child emitting two records, never a model."""
    identity = dict(profile='dual_t4',kind='baseline',condition='base_no_tools',split='test',
                    image_ids=[90001,90002],base_model='Qwen/Qwen3.5-9B',adapter=None,cohort_sha256='cohort')
    session = w.NotebookEvaluations.__new__(w.NotebookEvaluations)
    session.jobs = [dict(kind='baseline',condition='base_no_tools',profile='dual_t4',stage='base',adapter=None,k=None)]
    session.sizes = ('canonical',); session.status = []; session.output = tmp_path
    session.split = 'test'; session.profile_configs = w.PROFILES; session.seed = 42
    session.revision = 'checkpoint'; session.software = {}; session.model_id = '/existing-base'
    session.results_repo = 'existing/evaluations'; session.results_repo_type = 'dataset'
    session._identity = lambda job: identity
    session._input = lambda split: dict(fingerprint='cohort')
    session._save_inventory = lambda: None
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(Tensor=type('UnusedTensor',(),{}),cuda=SimpleNamespace(is_available=lambda:False)))
    # This test covers subprocess/upload control flow, not numerical scoring.
    monkeypatch.setitem(sys.modules,'dental_agent.evaluation.matrix',SimpleNamespace(summarize_cell=lambda rows,seed: {'episodes':len(rows)}))
    monkeypatch.setattr(w,'hardware_reason',lambda *args: None)
    monkeypatch.setattr(w.time,'sleep',lambda delay: None)
    clock = itertools.count(0,70); monkeypatch.setattr(w.time,'monotonic',lambda:next(clock))
    real_popen = subprocess.Popen; children = []; uploads = []
    class Child:
        def __init__(self,*args,**kwargs): self.proc = real_popen(*args,**kwargs); self.terminated=False; children.append(self)
        def poll(self): return self.proc.poll()
        def wait(self,timeout=None): return self.proc.wait(timeout=min(timeout or 0.05,0.05))
        @property
        def returncode(self): return self.proc.returncode
        def terminate(self): self.terminated=True; self.proc.terminate()
        def kill(self): self.terminated=True; self.proc.kill()
    monkeypatch.setattr(w.subprocess,'Popen',Child)
    def save(folder,prefix,ident,payloads,n,done):
        uploads.append((n,done,children[0].poll() is None))
        raise ConnectionError('temporary upload outage')
    session.store = SimpleNamespace(restore=lambda *args:None,save=save)
    def command(job,ident,folder):
        result = folder/'results.jsonl'
        record = dict(split='test',condition='base_no_tools',prompt='unified',predictions=[],
                      eval_settings=dict(max_new_tokens=16384,adapter_path=''))
        rows = [dict(record,image_id=i) for i in ident['image_ids']]
        code = ('import json,time\nfrom pathlib import Path\n'
                f'rows={rows!r}\n'
                f'with Path({str(result)!r}).open("w") as f:\n'
                ' for row in rows:\n  f.write(json.dumps(row)+"\\n"); f.flush(); time.sleep(0.2)\n')
        return [sys.executable,'-c',code],result,None
    session._command = command
    status = session.run()
    assert status[0]['status']=='complete' and status[0]['episodes']==2
    assert status[0]['upload_status']=='pending'
    assert children[0].returncode==0 and not children[0].terminated
    assert any(alive for _,_,alive in uploads), 'A periodic upload must fail while the child is still alive'
    folder = Path(status[0]['path'])
    assert len(w.jsonl_rows(folder/'results.jsonl'))==2
    assert (folder/'summary.json').is_file() and (folder/'upload_pending.json').is_file()
