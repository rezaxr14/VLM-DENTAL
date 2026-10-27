"""CPU mathematical checks; XLA execution is covered by a separate real TPU probe."""
from argparse import Namespace
import subprocess
import sys
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
from dental_agent.training.tpu_chunked_loss import frozen_head_causal_loss, validate_tpu_loss_args, install_tpu_chunked_loss


def data(dtype=torch.float32, bias=False):
    torch.manual_seed(2026)
    head=torch.nn.Linear(12,31,bias=bias,dtype=dtype).requires_grad_(False)
    hidden=torch.randn(3,19,12,dtype=dtype,requires_grad=True)
    labels=torch.randint(0,31,(3,19));labels[:,:7]=-100;labels[1,12:]=-100;labels[2,:]=-100
    return hidden,head,labels


@pytest.mark.parametrize('chunk',[1,4,8,19,32])
@pytest.mark.parametrize('bias',[False,True])
def test_exact_shifted_mean_and_hidden_gradients(chunk,bias):
    h,head,labels=data(bias=bias)
    z=head(h)[:,:-1].float()
    want=F.cross_entropy(z.reshape(-1,31),labels[:,1:].reshape(-1))
    # Exercise arbitrary upstream scaling as in accumulation/gradient clipping.
    wg=torch.autograd.grad(want*2.7,h)[0]
    got=frozen_head_causal_loss(h,head,labels,chunk)
    gg=torch.autograd.grad(got*2.7,h)[0]
    torch.testing.assert_close(got,want,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(gg,wg,atol=2e-7,rtol=2e-5)
    assert torch.count_nonzero(gg[2])==0 and torch.count_nonzero(gg[:,-1])==0
    assert head.weight.grad is None


def test_bf16_chain_rule_matches_stock_and_all_masked():
    h,head,labels=data(torch.bfloat16)
    want=F.cross_entropy(head(h)[:,:-1].float().reshape(-1,31),labels[:,1:].reshape(-1))
    wg=torch.autograd.grad(want,h)[0]
    got=frozen_head_causal_loss(h,head,labels,8)
    gg=torch.autograd.grad(got,h)[0]
    torch.testing.assert_close(got,want,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(gg,wg,atol=2e-4,rtol=2e-2)
    labels[:]=-100
    zero=frozen_head_causal_loss(h,head,labels,8)
    assert zero.item()==0 and torch.count_nonzero(torch.autograd.grad(zero,h)[0])==0


def test_no_vocabulary_sized_activations_saved_for_backward():
    h,head,labels=data()
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:(saved.append(t.shape) or t),lambda t:t):
        loss=frozen_head_causal_loss(h,head,labels,4)
    # The frozen head itself is saved by reference, not a per-token logits array.
    assert not any(tuple(shape)==(3,4,31) for shape in saved)
    assert len(saved)==5
    loss.backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()


def test_input_labels_are_not_mutated_and_head_training_is_rejected():
    h,head,labels=data();original=labels.clone()
    frozen_head_causal_loss(h,head,labels,4)
    assert torch.equal(labels,original)
    head.weight.requires_grad_(True)
    with pytest.raises(ValueError,match='frozen'):
        frozen_head_causal_loss(h,head,labels,4)
    with pytest.raises(ValueError,match='frozen'):
        install_tpu_chunked_loss(torch.nn.Module(),4)


@pytest.mark.parametrize('override,is_tpu,match',[
    ({'tpu_loss_chunk_tokens':-1},True,'nonnegative'),
    ({},False,'real TPU'),
    ({'loss_chunk_tokens':128},True,'loss-chunk-tokens 0'),
    ({'xla_spmd':False},True,'spmd'),
    ({'pad_vision_to_slots':False},True,'fixed shapes'),
])
def test_configuration_fails_before_model_loading(override,is_tpu,match):
    ns=Namespace(tpu_loss_chunk_tokens=256,loss_chunk_tokens=0,xla_spmd=True,fsdp=True,pad_vision_to_slots=True)
    vars(ns).update(override)
    with pytest.raises(ValueError,match=match):validate_tpu_loss_args(ns,is_tpu=is_tpu)


def test_disabled_flag_has_no_new_requirements():
    validate_tpu_loss_args(Namespace(tpu_loss_chunk_tokens=0),is_tpu=False)
    validate_tpu_loss_args(Namespace(),is_tpu=False)


def test_real_qwen_multimodal_lora_gradients_match_without_gpu_helper_changes():
    from test_chunked_loss import _tiny,_batch
    from peft import LoraConfig,get_peft_model
    plain=_tiny();plain.config.image_token_id=299;plain.config.video_token_id=298;plain.config.vision_start_token_id=297
    model=get_peft_model(plain,LoraConfig(r=2,lora_alpha=4,lora_dropout=0,target_modules=['q_proj','v_proj','linear_fc1','linear_fc2']))
    model.enable_input_require_grads();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train();inputs=_batch(b=1)
    inputs['input_ids'].remainder_(290);inputs['input_ids'][0,4]=297;inputs['input_ids'][0,5:9]=299
    inputs['labels']=inputs['input_ids'].clone();inputs['labels'][:,:25]=-100
    inputs['mm_token_type_ids']=torch.zeros_like(inputs['input_ids']);inputs['mm_token_type_ids'][0,5:9]=1
    inputs['image_grid_thw']=torch.tensor([[1,4,4]]);inputs['pixel_values']=torch.randn(16,3*2*16*16)
    original=inputs['labels'].clone()
    def run(stock):
        model.zero_grad()
        if stock:loss=model(**inputs).loss
        else:
            base=model.get_base_model()
            body_inputs={k:v for k,v in inputs.items() if k!='labels'}
            h=base.model(**body_inputs,use_cache=False,return_dict=True)[0]
            loss=frozen_head_causal_loss(h,base.lm_head,inputs['labels'],7)
        loss.backward()
        return loss.detach(),{n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    expected,wg=run(True);got,gg=run(False)
    torch.testing.assert_close(got,expected,atol=1e-5,rtol=1e-5)
    assert wg.keys()==gg.keys() and any('merger' in n for n in gg)
    for n in wg:torch.testing.assert_close(gg[n],wg[n],atol=1e-5,rtol=1e-4,msg=n)
    assert torch.equal(original,inputs['labels'])


@pytest.mark.parametrize('module_name',['scripts.train_sft','scripts.warmup_xla_cache'])
def test_real_launcher_parsers_match_tpu_validator(module_name,monkeypatch):
    import importlib
    launcher=importlib.import_module(module_name)
    monkeypatch.setattr(sys,'argv',[module_name,'--track','with_tools','--tpu-loss-chunk-tokens','256','--spmd','--fsdp','--pad-vision-to-slots'])
    args=launcher.parse_args()
    validate_tpu_loss_args(args,is_tpu=True)
    assert args.tpu_loss_chunk_tokens==256
    assert getattr(args,'loss_chunk_tokens',0)==0
