import sys
from types import SimpleNamespace
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("profiles", Path(__file__).parents[1]/"dental_agent/training/experiment_profiles.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
PROFILE_TAGS, grpo_name, fetch_sft = module.PROFILE_TAGS, module.grpo_name, module.fetch_sft

def test_distinct_profile_names():
    names = [grpo_name('with_tools',4,'dentex_alone',p,PROFILE_TAGS[h]) for h,p in [('dual_t4','qlora'),('rtx4090','qlora'),('workstation48_lora','bf16'),('tpu','bf16')]]
    assert len(set(names)) == 4
    assert grpo_name('with_tools',4,'dentex_alone','qlora') == 'qwen3_5_9b_grpo_with_tools_k4_dentex_alone'

def test_exact_sft_no_fallback(tmp_path,monkeypatch):
    import pytest
    name='qwen3_5_9b_sft_with_tools_dentex_alone_qlora__dual_t4_fp16'
    calls=[]
    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)
    monkeypatch.setitem(sys.modules,'huggingface_hub',SimpleNamespace(snapshot_download=download))
    with pytest.raises(FileNotFoundError):
        fetch_sft('owner/models','with_tools','dentex_alone','qlora','dual_t4_fp16')
    folder=tmp_path/'sft'/name
    folder.mkdir(parents=True)
    (folder/'adapter_config.json').write_text('{}')
    assert fetch_sft('owner/models','with_tools','dentex_alone','qlora','dual_t4_fp16')==str(folder)
    assert calls[-1]['allow_patterns']==[f'sft/{name}/*']

def test_unsafe_tag():
    import pytest
    with pytest.raises(ValueError):grpo_name('with_tools',4,'dentex_alone','qlora','../bad')
