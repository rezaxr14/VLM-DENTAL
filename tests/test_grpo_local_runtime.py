"""Small real CPU Qwen/PEFT checks; these do not establish GPU memory/speed."""
import copy
import ast
from pathlib import Path
import pytest
import torch
from test_chunked_loss import _tiny, _batch
from dental_agent.training.chunked_log_probs import token_log_probs
from dental_agent.training.grpo import compute_token_log_probs
from dental_agent.model.local_runtime import LocalRuntime, configure_model, generate_local, generation_kwargs, resolve_options

@pytest.mark.parametrize('chunk',[1,7,128])
def test_selected_log_probs_and_gradients(chunk):
    model = _tiny().train()
    batch = _batch(b=1,t=20,masked_prefix=8)
    def run(size):
        model.zero_grad()
        lp,mask=token_log_probs(model,batch,size)
        # Nonuniform PPO-like coefficients exercise individual token gradients.
        weights=torch.linspace(.1,1,lp.numel()).reshape_as(lp)
        (lp*weights*mask).sum().backward()
        return lp.detach(),{n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    want,wg=run(0); got,gg=run(chunk)
    torch.testing.assert_close(got,want,atol=2e-6,rtol=2e-5)
    assert wg.keys()==gg.keys()
    for name in wg: torch.testing.assert_close(gg[name],wg[name],atol=2e-5,rtol=2e-4,msg=name)


def test_checkpointed_peft_reference_restore_and_no_grad():
    from peft import LoraConfig,get_peft_model
    config=LoraConfig(r=2,lora_alpha=4,lora_dropout=0,target_modules=['q_proj','v_proj'])
    model=get_peft_model(_tiny(),config,adapter_name='grpo_policy')
    model.add_adapter('reference',copy.deepcopy(config))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    batch=_batch(b=1,t=16,masked_prefix=6)
    model.train()
    want,_=compute_token_log_probs(model,batch,logprob_chunk_tokens=0)
    ref,_=compute_token_log_probs(model,batch,use_reference=True,logprob_chunk_tokens=3)
    assert not ref.requires_grad and model.active_adapter=='grpo_policy'
    want.sum().backward()
    grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    got,_=compute_token_log_probs(model,batch,logprob_chunk_tokens=3)
    compute_token_log_probs(model,batch,use_reference=True,logprob_chunk_tokens=3)
    got.sum().backward()
    for n,p in model.named_parameters():
        if n in grads: torch.testing.assert_close(p.grad,grads[n],atol=1e-5,rtol=1e-4)
        if '.reference.' in n: assert p.grad is None and not p.requires_grad
    with torch.no_grad():
        assert not compute_token_log_probs(model,batch,logprob_chunk_tokens=3)[0].requires_grad


def test_chunks_bound_head_rows_and_saved_logits():
    model=_tiny().train(); batch=_batch(b=1,t=40,masked_prefix=5)
    rows=[];saved=[]
    model.lm_head.register_forward_hook(lambda m,i,o: rows.append(o.shape[0]))
    with torch.autograd.graph.saved_tensors_hooks(lambda t:(saved.append(tuple(t.shape)) or t),lambda t:t):
        lp,_=token_log_probs(model,batch,7)
    assert max(rows)<=7
    assert not any(len(s)>=2 and s[-1]==300 for s in saved)
    lp.sum().backward()


def test_real_qwen_cached_decode_and_last_token_logits():
    model=_tiny().eval(); configure_model(model,LocalRuntime(),torch.float32,'sdpa')
    batch=_batch(b=1,t=8,masked_prefix=0)
    args={k:v for k,v in batch.items() if k!='labels'}
    original=copy.deepcopy(model)
    stock=original.generate(**args,max_new_tokens=3,do_sample=False,use_cache=True,pad_token_id=0)
    rows=[]
    model.lm_head.register_forward_hook(lambda m,i,o: rows.append(o.shape[-2]))
    got=generate_local(model,**args,max_new_tokens=3,do_sample=False,pad_token_id=0)
    assert torch.equal(got,stock) and rows==[1,1,1]
    assert generation_kwargs(model)=={'use_cache':True,'logits_to_keep':1}


def test_tpu_rejects_dynamic_chunking():
    with pytest.raises(ValueError,match='TPU'):
        resolve_options(LocalRuntime(precision='bf16',logprob_chunk_tokens=128),is_tpu=True)


def test_remote_api_branches_unchanged():
    # Snapshot remote dispatch bodies, excluding chained local-HF elif bodies.
    import hashlib
    path=Path(__file__).resolve().parents[1]/'dental_agent/training/api_pool.py'
    tree=ast.parse(path.read_text())
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_call_llm_once')
    branches=[ast.dump(ast.If(test=n.test,body=n.body,orelse=[]),include_attributes=False)
              for n in ast.walk(fn) if isinstance(n,ast.If) and 'provider' in ast.unparse(n.test)
              and not any(x in ast.unparse(n.test) for x in ['local_hf','transformers'])]
    assert hashlib.sha256('\n'.join(branches).encode()).hexdigest()=='662e53cd46a6b6a514d713ad574e56fcdb2a3cda24fb5c65ff088ca6fb44b9c4'

def test_multimodal_checkpointed_projector_gradients():
    from peft import LoraConfig,get_peft_model
    plain=_tiny(); plain.config.image_token_id=299; plain.config.video_token_id=298; plain.config.vision_start_token_id=297
    model=get_peft_model(plain,LoraConfig(r=2,lora_alpha=4,lora_dropout=0,target_modules=['q_proj','v_proj','linear_fc1','linear_fc2']))
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train(); batch=_batch(b=1)
    batch['input_ids'].remainder_(290);batch['input_ids'][0,4]=297;batch['input_ids'][0,5:9]=299
    batch['labels']=batch['input_ids'].clone();batch['labels'][:,:25]=-100
    batch['mm_token_type_ids']=torch.zeros_like(batch['input_ids']);batch['mm_token_type_ids'][0,5:9]=1
    batch['image_grid_thw']=torch.tensor([[1,4,4]]);batch['pixel_values']=torch.randn(16,3*2*16*16)
    def run(c):
        model.zero_grad();lp,mask=token_log_probs(model,batch,c);(lp*mask).sum().backward()
        return lp.detach(),{n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    want,wg=run(0);got,gg=run(7)
    torch.testing.assert_close(got,want,atol=2e-6,rtol=2e-5)
    assert wg.keys()==gg.keys() and any('merger' in n for n in gg)
    for n in wg: torch.testing.assert_close(gg[n],wg[n],atol=2e-5,rtol=2e-4)


def test_agent_cache_preserves_full_span_offsets_and_final_sample(monkeypatch):
    from types import SimpleNamespace
    from dental_agent.model import inference
    calls=[]
    class Processor:
        tokenizer=SimpleNamespace(pad_token_id=0,eos_token_id=2)
        def apply_chat_template(self,*a,**k): return ''
        def __call__(self,**k):
            return {'input_ids':torch.tensor([[10,11,12,13,14,15]]),'attention_mask':torch.ones(1,6,dtype=torch.long)}
        def batch_decode(self,*a,**k): return ['answer']
    model=_tiny().eval()
    class Cache:
        def get_seq_length(self): return 3
    def fake_generate(model,**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(sequences=torch.cat([kwargs['input_ids'],torch.tensor([[20,21]])],dim=1),past_key_values=Cache())
    monkeypatch.setattr('dental_agent.model.local_runtime.generate_local',fake_generate)
    monkeypatch.setattr(inference,'safe_process_vision_info',lambda m:(None,None))
    old={'cached_input_ids':torch.tensor([[10,11,12,13]]),'last_msg_count':1}
    out=inference.generate_agent_reply(model,Processor(),[{'role':'user','content':'a'},{'role':'user','content':'b'}],past_key_values=Cache(),cache_state=old,return_cache=True,max_new_tokens=16384)
    assert calls[0]['input_ids'].tolist()==[[13,14,15]] # final sampled token wasn't in cache
    assert calls[0]['max_new_tokens']==16384 and calls[0]['attention_mask'].shape[1]==6
    assert out[1]==6 and out[2]==[20,21]
    assert old['cached_input_ids'].tolist()==[[10,11,12,13]]
    calls.clear()
    # Changed token prefix must prefill again, rather than trusting message count.
    old['cached_input_ids']=torch.tensor([[99,11,12,13]])
    inference.generate_agent_reply(model,Processor(),[{'role':'user','content':'a'},{'role':'user','content':'b'}],past_key_values=Cache(),cache_state=old)
    assert 'past_key_values' not in calls[0] and calls[0]['input_ids'].shape[1]==6


@pytest.mark.skipif(torch.cuda.device_count()<2,reason='requires two real CUDA GPUs')
def test_dispatched_grpo_head_backward():
    from accelerate import dispatch_model
    model=dispatch_model(_tiny().half().train(),device_map={'model':0,'lm_head':1})
    batch={k:v.cuda(0) for k,v in _batch(b=1).items()}
    lp,mask=token_log_probs(model,batch,7)
    (lp*mask).sum().backward()
    assert torch.isfinite(model.lm_head.weight.grad).all()


def test_pipeline_cache_keys_include_runtime_and_local_folder(monkeypatch):
    from dental_agent.training.api_pool import _POOL
    from types import SimpleNamespace
    folder=['/local/model-a'];loads=[]
    monkeypatch.setattr(_POOL,'_transformers_pipelines',{})
    monkeypatch.setattr('dental_agent.utils.inputs.local_model_alias',lambda name:folder[0])
    class Model:
        def eval(self): return self
    def load(cfg,**kw):
        loads.append((cfg.name,kw['runtime_options']))
        return Model(),SimpleNamespace()
    monkeypatch.setattr('dental_agent.model.backbone.load_model',load)
    first=_POOL.get_transformers_pipeline('alias',LocalRuntime(precision='fp16'))
    assert _POOL.get_transformers_pipeline('alias',LocalRuntime(precision='fp16'))[0] is first[0]
    _POOL.get_transformers_pipeline('alias',LocalRuntime(precision='bf16'))
    folder[0]='/local/model-b'
    _POOL.get_transformers_pipeline('alias',LocalRuntime(precision='fp16'))
    assert len(loads)==3 and loads[-1][0]=='/local/model-b'
