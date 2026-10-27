"""Local metadata/namespace regression tests. Hub APIs are fakes; no real uploads."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from huggingface_hub import ModelCard
from dental_agent.model.hub_artifacts import (prepare_hub_metadata,upload_adapter_folder,
    check_hub_write_access,sft_hub_subfolder,validate_base_model_id)

BASE='Qwen/Qwen3.5-9B'
LOCAL='/kaggle/input/datasets/rezanadimikj/qwen3-5-9b'


def checkpoint(folder,base=LOCAL):
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'adapter_config.json').write_text(json.dumps({'base_model_name_or_path':base,'r':32}))
    (folder/'adapter_model.safetensors').write_bytes(b'test fixture, not real model weights')
    (folder/'README.md').write_text(f'---\nbase_model: {base}\nlicense: apache-2.0\n---\nKeep this model description.\n')
    return folder


def test_repair_root_and_best_preserves_weights_and_other_metadata(tmp_path):
    root=checkpoint(tmp_path/'model');best=checkpoint(root/'best_adapter')
    before={p:p.read_bytes() for p in root.rglob('*.safetensors')}
    result=prepare_hub_metadata(root,BASE)
    assert result['model_cards']==2 and result['adapter_configs']==2
    for folder in [root,best]:
        card=ModelCard.load(str(folder/'README.md'))
        assert card.data.base_model==BASE and card.data.license=='apache-2.0'
        assert 'Keep this model description.' in card.content
        config=json.loads((folder/'adapter_config.json').read_text())
        assert config['base_model_name_or_path']==BASE and config['r']==32
    assert before=={p:p.read_bytes() for p in before}
    first={p:p.read_bytes() for p in root.rglob('*.md')}
    prepare_hub_metadata(root,BASE)
    assert first=={p:p.read_bytes() for p in first}
    assert not list(root.rglob('*.tmp'))


@pytest.mark.parametrize('bad',[LOCAL,r'C:\Users\rezax\model','../local-model'])
def test_no_override_omits_invalid_optional_card_field(tmp_path,bad):
    root=checkpoint(tmp_path/'model',bad)
    prepare_hub_metadata(root)
    assert ModelCard.load(str(root/'README.md')).data.to_dict().get('base_model') is None
    assert json.loads((root/'adapter_config.json').read_text())['base_model_name_or_path']==bad


def test_valid_existing_base_is_preserved_and_config_can_repair_card(tmp_path):
    root=checkpoint(tmp_path/'model',BASE)
    prepare_hub_metadata(root)
    assert ModelCard.load(str(root/'README.md')).data.base_model==BASE
    (root/'README.md').write_text(f'---\nbase_model: {LOCAL}\n---\nOriginal body\n')
    prepare_hub_metadata(root)
    assert ModelCard.load(str(root/'README.md')).data.base_model==BASE


def test_stage_precision_track_names_are_distinct_and_run_tags_are_optional():
    destinations={sft_hub_subfolder(track,stage,precision) for track in ['with_tools','no_tools']
                  for stage in ['dentex_alone','dentex_tufts_overlap','multicohort_all']
                  for precision in ['qlora','bf16','fp16']}
    assert len(destinations)==18
    path=sft_hub_subfolder('with_tools','dentex_alone','qlora')
    assert path=='sft/qwen3_5_9b_sft_with_tools_dentex_alone_qlora'
    assert sft_hub_subfolder('with_tools','dentex_alone','qlora','rtx4090_bf16')==path+'__rtx4090_bf16'
    for tag in ['../escape','a/b',r'a\b','..','a b']:
        with pytest.raises(ValueError):sft_hub_subfolder('with_tools','dentex_alone','bf16',tag)
    with pytest.raises(ValueError):sft_hub_subfolder('../escape','dentex_alone','bf16')


def test_metadata_is_fixed_before_upload_and_nested_grpo_cards_are_included(tmp_path,monkeypatch):
    root=tmp_path/'grpo';checkpoint(root/'reference');checkpoint(root/'grpo_policy')
    calls=[]
    class API:
        def upload_folder(self,**kwargs):
            calls.append(kwargs)
            for name in ['reference','grpo_policy']:
                assert 'base_model' not in ModelCard.load(str(root/name/'README.md')).data.to_dict()
            return 'mock-commit'
    monkeypatch.setattr('huggingface_hub.HfApi',API)
    assert upload_adapter_folder(root,'user/models','grpo/run/final')=='mock-commit'
    assert calls[0]['path_in_repo']=='grpo/run/final' and calls[0]['repo_type']=='model'


def test_write_check_proves_upload_attempt_and_reports_permission_failure(monkeypatch):
    captured=[]
    class API:
        def upload_file(self,**kwargs):
            payload=json.loads(kwargs['path_or_fileobj'].read())
            captured.append((kwargs,payload));return 'mock-check'
    monkeypatch.setattr('huggingface_hub.HfApi',API)
    check_hub_write_access('user/models','sft/run',metadata={'stage':'dentex_alone','precision':'qlora'})
    assert captured[0][0]['path_in_repo']=='sft/run/upload_check.json'
    assert captured[0][1]['precision']=='qlora'
    class Denied:
        def upload_file(self,**kwargs):raise PermissionError('403 token is read-only')
    monkeypatch.setattr('huggingface_hub.HfApi',Denied)
    with pytest.raises(RuntimeError,match='No model weights have been loaded'):
        check_hub_write_access('user/models','sft/run',metadata={})


def test_symlink_metadata_outside_checkpoint_is_rejected(tmp_path):
    root=tmp_path/'model';root.mkdir();outside=tmp_path/'outside.json';outside.write_text('{}')
    try:(root/'adapter_config.json').symlink_to(outside)
    except OSError:pytest.skip('Symlink creation unavailable')
    with pytest.raises(ValueError,match='outside'):prepare_hub_metadata(root,BASE)
    assert outside.read_text()=='{}'


def test_real_peft_save_metadata_can_be_repaired_without_mutating_live_model(tmp_path):
    from test_chunked_loss import _tiny
    from peft import LoraConfig,get_peft_model
    model=get_peft_model(_tiny(),LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj']))
    model.get_base_model().config._name_or_path=LOCAL
    model.peft_config['default'].base_model_name_or_path=LOCAL
    model.save_pretrained(tmp_path)
    original=(tmp_path/'adapter_model.safetensors').read_bytes()
    assert ModelCard.load(str(tmp_path/'README.md')).data.base_model==LOCAL
    prepare_hub_metadata(tmp_path,BASE)
    assert ModelCard.load(str(tmp_path/'README.md')).data.base_model==BASE
    assert (tmp_path/'adapter_model.safetensors').read_bytes()==original
    assert model.get_base_model().config._name_or_path==LOCAL
    assert model.peft_config['default'].base_model_name_or_path==LOCAL


def test_existing_checkpoint_cli_repairs_and_uses_saved_stage_precision(tmp_path,monkeypatch):
    import sys
    from scripts.upload_saved_checkpoint import main
    root=checkpoint(tmp_path/'recovered')
    (root/'sft_run_meta.json').write_text(json.dumps({'track':'with_tools','stage':'dentex_alone','precision':'qlora'}))
    calls=[]
    class API:
        def upload_folder(self,**kwargs):
            calls.append(kwargs)
            assert ModelCard.load(str(root/'README.md')).data.base_model==BASE
            return 'fake-upload'
    monkeypatch.setattr('huggingface_hub.HfApi',API)
    monkeypatch.setattr(sys,'argv',['upload_saved_checkpoint.py','--checkpoint-dir',str(root),
                                  '--hf-repo','user/models','--hub-base-model-id',BASE])
    main()
    assert calls[0]['path_in_repo']=='sft/qwen3_5_9b_sft_with_tools_dentex_alone_qlora'
    monkeypatch.setattr(sys,'argv',['upload_saved_checkpoint.py','--checkpoint-dir',str(root),
                                  '--hf-repo','user/models','--precision','bf16'])
    with pytest.raises(SystemExit):main()
    assert len(calls)==1


def test_prepare_only_cli_never_initializes_hub_api(tmp_path,monkeypatch):
    import sys
    from scripts.upload_saved_checkpoint import main
    root=checkpoint(tmp_path/'recovered')
    def no_api():raise AssertionError('prepare-only must not initialize an API client')
    monkeypatch.setattr('huggingface_hub.HfApi',no_api)
    monkeypatch.setattr(sys,'argv',['upload_saved_checkpoint.py','--checkpoint-dir',str(root),
                                  '--path-in-repo','sft/recovery','--hub-base-model-id',BASE,'--prepare-only'])
    main()
    assert ModelCard.load(str(root/'README.md')).data.base_model==BASE
