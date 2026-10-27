"""Runtime controls shared by local GRPO rollouts and native Transformers inference.

No provider API settings, token budgets or sampling defaults are changed here.
Hardware-specific helpers are imported lazily to avoid training-package cycles.
"""
from __future__ import annotations

import contextlib
import inspect
from dataclasses import asdict, dataclass
from typing import Any
import torch


@dataclass(frozen=True)
class LocalRuntime:
    precision: str = 'auto'
    qlora_compute_dtype: str = 'auto'
    attn_implementation: str = 'auto'
    sdpa_repeat_kv: str = 'auto'
    loader_mode: str = 'auto'
    load_cpu_threads: int | None = None
    load_traceback_seconds: int = 0
    require_efficient_sdpa: bool = True
    logprob_chunk_tokens: int = 0
    allow_cpu_offload: bool = False


def runtime_from_args(args, *, precision_attr='precision') -> LocalRuntime:
    return LocalRuntime(**{name: getattr(args, precision_attr if name == 'precision' else name, field.default)
                          for name, field in LocalRuntime.__dataclass_fields__.items()})


def add_runtime_args(parser, *, precision_flag: str | None = None, scoring: bool = False):
    if precision_flag:
        parser.add_argument(precision_flag, dest='local_precision', choices=['auto','qlora','bf16','fp16','fp32'], default='auto',
                            help='Native Transformers backend only; API providers/vLLM server are unchanged.')
    parser.add_argument('--qlora-compute-dtype', choices=['auto','bf16','fp16'], default='auto')
    parser.add_argument('--attn-implementation', choices=['auto','sdpa','flash_attention_2'], default='auto')
    parser.add_argument('--sdpa-repeat-kv', choices=['auto','on','off'], default='auto')
    parser.add_argument('--loader-mode', choices=['auto','standard','staged'], default='auto')
    parser.add_argument('--load-cpu-threads', type=int, default=None, help='Omitted: 1 during auto T4 loading, otherwise preserve. 0 explicitly preserves.')
    parser.add_argument('--load-traceback-seconds', type=int, default=0)
    import argparse
    parser.add_argument('--require-efficient-sdpa', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--allow-cpu-offload', action='store_true', help='Explicitly permit slow inference CPU/disk offload; forbidden for GRPO training.')
    if scoring:
        parser.add_argument('--logprob-chunk-tokens', type=int, default=0,
                            help='GRPO CUDA/CPU scoring: full-vocabulary log probabilities on selected tokens in chunks; 0=stock path; TPU needs 0.')


def resolve_options(options: LocalRuntime | dict | None, *, load_in_4bit: bool = False, is_tpu: bool = False):
    from dental_agent.training.gpu_runtime import native_bf16, resolve_compute_dtype, select_attention_backend
    opts = options if isinstance(options, LocalRuntime) else LocalRuntime(**(options or {}))
    if opts.logprob_chunk_tokens < 0 or opts.load_traceback_seconds < 0 or (opts.load_cpu_threads is not None and opts.load_cpu_threads < 0):
        raise ValueError('Chunk size and loading diagnostics must be nonnegative')
    cuda = torch.cuda.is_available() and not is_tpu
    precision = opts.precision
    if precision == 'auto':
        precision = 'qlora' if load_in_4bit and cuda else ('bf16' if is_tpu or (cuda and native_bf16()) else ('fp16' if cuda else 'fp32'))
    if precision == 'qlora' and not cuda:
        raise ValueError('QLoRA requires CUDA, not CPU/TPU')
    dtype = torch.float32 if precision == 'fp32' else resolve_compute_dtype(precision, opts.qlora_compute_dtype)
    if cuda and dtype == torch.bfloat16 and not native_bf16():
        raise ValueError('Native BF16 unavailable: use --precision qlora --qlora-compute-dtype fp16 on T4')
    pre_ampere = cuda and any(torch.cuda.get_device_capability(i)[0] < 8 for i in range(torch.cuda.device_count()))
    mode = ('staged' if pre_ampere else 'standard') if opts.loader_mode == 'auto' else opts.loader_mode
    threads = (1 if pre_ampere and opts.loader_mode == 'auto' else 0) if opts.load_cpu_threads is None else opts.load_cpu_threads
    if is_tpu and (opts.logprob_chunk_tokens or opts.sdpa_repeat_kv == 'on' or mode != 'standard' or threads or opts.load_traceback_seconds):
        raise ValueError('TPU requires stock log probabilities, standard loader, no forced CUDA KV/thread/traceback overrides')
    backend = select_attention_backend(opts.attn_implementation, is_tpu=is_tpu)
    return opts, precision, dtype, backend, mode, threads


def configure_model(model, options, dtype, backend):
    opts = options if isinstance(options, LocalRuntime) else LocalRuntime(**(options or {}))
    placement = getattr(model, 'hf_device_map', {}) or {}
    if not opts.allow_cpu_offload and any(str(v) in {'cpu','disk'} for v in placement.values()):
        raise RuntimeError('Local model offloaded weights to CPU/disk. Select QLoRA or explicitly permit slow inference with --allow-cpu-offload.')
    model._dental_local_runtime = opts
    model._dental_compute_dtype = dtype
    model._dental_attention_backend = backend
    print(f'[LOCAL RUNTIME] compute={dtype}; attention={backend}; KV={opts.sdpa_repeat_kv}; chunks={opts.logprob_chunk_tokens}; placement={placement}', flush=True)
    return model


def runtime_for(model):
    for m in (model, getattr(model, 'module', None)):
        if m is not None and hasattr(m, '_dental_local_runtime'):
            return m._dental_local_runtime
    return None


def input_device(model):
    get_emb = getattr(model, 'get_input_embeddings', None)
    if callable(get_emb):
        emb = get_emb()
        if emb is not None and emb.weight.device.type != 'meta':
            return emb.weight.device
    dev = getattr(model, 'device', None)
    if dev is not None and str(dev) != 'meta':
        return torch.device(dev)
    return next(model.parameters()).device


@contextlib.contextmanager
def local_context(model, *, autocast: bool = True):
    opts = runtime_for(model)
    if opts is None:
        yield
        return
    from dental_agent.training.gpu_runtime import sdpa_gqa_policy
    from dental_agent.training.gpu_debug import gpu_forward_context, gpu_attention_context
    cuda = input_device(model).type == 'cuda'
    dtype = getattr(model, '_dental_compute_dtype', None)
    with contextlib.ExitStack() as stack:
        # Never change XLA policy; auto on modern CUDA preserves HF's policy.
        if cuda or opts.sdpa_repeat_kv == 'on':
            stack.enter_context(sdpa_gqa_policy(opts.sdpa_repeat_kv,report=False))
        if autocast and dtype in (torch.float16, torch.bfloat16):
            stack.enter_context(gpu_forward_context(cuda, dtype, opts.require_efficient_sdpa))
        else:
            stack.enter_context(gpu_attention_context(cuda, opts.require_efficient_sdpa))
        yield


def generation_kwargs(model) -> dict[str, Any]:
    """Decode needs the next-token distribution; scoring must not inherit this flag."""
    base = model.get_base_model() if callable(getattr(model, 'get_base_model', None)) else model
    forward = getattr(base, '_old_forward', base.forward)
    supported = 'logits_to_keep' in inspect.signature(forward).parameters
    if getattr(getattr(base, 'config', None), 'model_type', '') in {'qwen3_5','qwen3_5_moe'}:
        supported = True
    return {'use_cache': True, **({'logits_to_keep': 1} if supported else {})}


def generate_local(model, **kwargs):
    kwargs = {**generation_kwargs(model), **kwargs}
    with local_context(model), torch.inference_mode():
        return model.generate(**kwargs)


def runtime_record(model) -> dict:
    opts = runtime_for(model)
    if opts is None:
        return {}
    return {**asdict(opts), 'effective_compute_dtype': str(getattr(model, '_dental_compute_dtype', None)),
            'effective_attention': getattr(model, '_dental_attention_backend', None)}
