"""Execute notebook command-building code without launching models or accessing APIs.

Simulated platform flags here check CLI plumbing only, not GPU/TPU functionality.
"""
import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/'notebooks/VLM_Dental_Colab_SFT.ipynb'


def cells():return json.loads(NOTEBOOK.read_text())['cells']
def text(cell):return ''.join(cell['source'])


@pytest.mark.parametrize('profile,precision,compute,chunk',[
    ('dual_t4','qlora','fp16',256),('rtx4090_qlora','qlora','bf16',1024),
    ('workstation48_lora','bf16','auto',2048),('tpu','bf16','auto',0),
])
def test_full_configuration_and_training_command_match_actual_parser(tmp_path,monkeypatch,profile,precision,compute,chunk):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr('dental_agent.utils.inputs.resolve_qwen_model',lambda **kwargs:(str(tmp_path),'test-only'))
    from dental_agent.utils.inputs import LOCAL_BASE_MODEL_ENV
    monkeypatch.setenv(LOCAL_BASE_MODEL_ENV,'test-before')
    ns={'IS_TPU':profile=='tpu','torch':torch,'os':os,'sys':sys,'Path':Path}
    config=next(text(c) for c in cells() if 'HARDWARE_PROFILE = "dual_t4"' in text(c))
    config=config.replace('HARDWARE_PROFILE = "dual_t4"',f'HARDWARE_PROFILE = {profile!r}',1)
    exec(compile(config,str(NOTEBOOK),'exec'),ns)
    assert ns['PRECISION']==precision and ns['QLORA_COMPUTE_DTYPE']==compute and ns['LOSS_CHUNK_TOKENS']==chunk
    assert ns['TARGET_SEQ_LEN']==16384 and ns['SAVE_STRATEGY']=='both' and ns['PUSH_EVERY_STEPS']==5
    assert ns['HF_BASE_MODEL_ID']=='Qwen/Qwen3.5-9B'
    assert ns['hf_target_folder'].endswith('_'+precision)
    # Exercise normal training, custom local folder and optional tag, not just debug mode.
    ns.update(DEBUG_STEPS=0,OUTPUT_DIR=str(tmp_path/'custom_local'),HF_RUN_TAG='test_run')
    calls=[]
    monkeypatch.setattr(subprocess,'run',lambda command,**kwargs:(calls.append(command) or SimpleNamespace(returncode=0,stdout='simulated preflight',stderr='')))
    launch=next(text(c) for c in cells() if 'probe_cmd = [' in text(c))
    tree=ast.parse(launch)
    # Keep all command/probe/preflight branches; remove only the actual training Popen.
    tree.body=[node for node in tree.body if not(isinstance(node,ast.With) and 'subprocess.Popen' in ast.unparse(node.items[0].context_expr))]
    exec(compile(tree,str(NOTEBOOK),'exec'),ns)
    cmd=ns['cmd'];assert cmd[4]=='scripts/train_sft.py'
    monkeypatch.setattr(sys,'argv',[cmd[4],*cmd[5:]])
    from scripts.train_sft import parse_args
    args=parse_args()
    assert args.model_id==str(tmp_path) and args.hub_base_model_id=='Qwen/Qwen3.5-9B'
    assert args.hf_run_tag=='test_run' and args.hf_upload_check and args.output_dir==str(tmp_path/'custom_local')
    assert args.loss_chunk_tokens==chunk and args.precision==precision
    if profile=='tpu':
        assert args.tpu_loss_chunk_tokens==256
        assert any('scripts/probe_tpu_chunked_loss.py' in c for c in calls)
    else:
        assert args.tpu_loss_chunk_tokens==0 and '--tpu-loss-chunk-tokens' not in cmd
        probe=next(c for c in calls if 'scripts/probe_sft_attention.py' in c)
        assert '--loss-chunk-tokens' not in probe and '--loader-mode' not in probe and '--debug-steps' not in probe


def test_aot_uses_matching_tpu_chunk_and_lora_settings(monkeypatch):
    monkeypatch.chdir(ROOT)
    for key in ['TPU_PROCESS_ADDRESSES','TPU_PROCESS_COUNT','CLOUD_TPU_TASK_ID','PJRT_DEVICE','XLA_FLAGS','LIBTPU_INIT_ARGS']:
        monkeypatch.setenv(key,'test-stale-value')
    calls=[];shell=[]
    monkeypatch.setattr(subprocess,'run',lambda command,**kwargs:(calls.append(command) or SimpleNamespace(returncode=0)))
    ns={'IS_TPU':True,'os':os,'sys':sys,'Path':Path,'MODEL_ID':'/attached/local/model','TRACK':'with_tools',
        'PRECISION':'bf16','NUM_CORES':8,'TARGET_SEQ_LEN':16384,'PAD_VISION_TO_SLOTS':True,'VISION_SLOTS':(5,10,4),
        'TRIANGULAR_SHIM':True,'XLA_SPMD':True,'USE_FSDP':True,'LORA_TARGET_VISION':'projector','LORA_LINEAR_ATTN':True,
        'LORA_R':16,'LORA_ALPHA':32,'LORA_DROPOUT':.07,'LEARNING_RATE':1e-5,'XLA_PALLAS':True,
        'RUN_TPU_LOSS_PROBE':True,'TPU_LOSS_CHUNK_TOKENS':256,'TPU_LOSS_PROBE_PASSED':False,
        'get_ipython':lambda:SimpleNamespace(system=lambda command:shell.append(command))}
    from IPython.core.inputtransformer2 import TransformerManager
    aot=next(text(c) for c in cells() if 'warmup_cmd = [' in text(c))
    exec(compile(TransformerManager().transform_cell(aot),str(NOTEBOOK),'exec'),ns)
    cmd=ns['warmup_cmd'];monkeypatch.setattr(sys,'argv',cmd)
    # List starts with python, script; the parser expects argv[0] to be the script.
    monkeypatch.setattr(sys,'argv',cmd[cmd.index('scripts/warmup_xla_cache.py'):])
    from scripts.warmup_xla_cache import parse_args
    args=parse_args()
    assert args.tpu_loss_chunk_tokens==256 and args.lora_r==16 and args.lora_alpha==32
    assert args.lora_dropout==.07 and args.learning_rate==1e-5 and args.max_seq_len==16384
    assert ns['TPU_LOSS_PROBE_PASSED'] and 'scripts/probe_tpu_chunked_loss.py' in calls[0]
    assert '--loss-chunk-tokens' not in cmd
    assert 'PJRT_DEVICE' not in os.environ


def test_every_notebook_code_cell_compiles():
    from IPython.core.inputtransformer2 import TransformerManager
    t=TransformerManager()
    for i,cell in enumerate(cells()):
        if cell['cell_type']=='code':compile(t.transform_cell(text(cell)),f'{NOTEBOOK}:{i}','exec')
