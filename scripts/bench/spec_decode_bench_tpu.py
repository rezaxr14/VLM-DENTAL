#!/usr/bin/env python3
"""TPU speculative-decoding benchmark: the same paired, fixed-dev, greedy dynamic-tool episodes as ``spec_decode_bench.py`` (vLLM, GPU),
run in-process on the TPU static-shape generator. Conditions ``baseline`` / ``ngram_2`` / ``ngram_3`` / ``ngram_5`` (the vLLM
``method=ngram`` configurations; ``mtp_*`` needs the native MTP head, which Hugging Face does not load, and is not available here).

Same result schema, same equivalence gate (decoded text, tool actions, final answer, pooled exact F1 must equal the baseline's).
Exactness of speculation on XLA is therefore CHECKED here, not assumed. Canonical black-bar letterbox views, as every TPU evaluation.
Needs the TPU runtime; loads the model once and switches the speculative configuration per condition (every condition is preceded by
warm-up episodes, because each ``k`` has its own verification program and XLA compiles it during the first rounds).
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CONDITIONS = ('baseline', 'ngram_2', 'ngram_3', 'ngram_5')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-id', required=True)
    p.add_argument('--adapter-path', required=True, help='SFT/GRPO adapter; merged into the base weights in memory (nothing is written)')
    p.add_argument('--checkpoint-kind', choices=['sft', 'grpo'], default='sft')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--dev-split', default='data/splits/dentex_dev.json')
    p.add_argument('--split', choices=['dev'], default='dev')
    p.add_argument('--limit', type=int)
    p.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(CONDITIONS))
    p.add_argument('--output', default='paper/generated/spec_decode_bench_tpu.json')
    p.add_argument('--existing-results', choices=['error', 'resume'], default='error')
    p.add_argument('--max-new-tokens', type=int, default=2048)
    p.add_argument('--max-tool-calls', type=int, default=50)
    p.add_argument('--max-turns', type=int)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--precision', choices=['bf16'], default='bf16')
    p.add_argument('--warmup-episodes', type=int, default=2, help='Full dev episodes per condition excluded from timing and gates')
    from dental_agent.harness.configuration import HARNESS_MODES
    p.add_argument('--harness-mode', choices=HARNESS_MODES, default='off')
    p.add_argument('--canonical-letterbox', action=argparse.BooleanOptionalAction, default=True)
    from dental_agent.model.local_runtime import add_runtime_args
    add_runtime_args(p)
    return p.parse_args(argv)


class TPUReply:
    """``reply_fn`` that generates through the loaded TPU model and records one measurement per request, shaped like ``ServingReply``."""

    def __init__(self, config):
        self.config, self.requests = config, []

    def __call__(self, model, processor, messages, **kwargs):
        from dental_agent.model.inference import generate_agent_reply
        self.config.stats.pop('last_call', None)
        started = time.perf_counter()
        out = generate_agent_reply(model, processor, messages, **kwargs)       # may raise GenerationTooLong / SlotOverflow: no record
        wall = time.perf_counter() - started
        last = self.config.stats.get('last_call')
        if last is None:
            raise RuntimeError('TPU generator recorded no measurements for this request')
        spec = last.get('speculative')
        drafted, accepted = (last['drafted'], last['accepted']) if spec else (None, None)
        self.requests.append(dict(
            input_sha256=hashlib.sha256(json.dumps([m.get('content') if isinstance(m.get('content'), str) else str(m.get('content')) for m in messages]).encode()).hexdigest(),
            wall_seconds=wall, prefill_seconds=last['prefill_seconds'], decode_seconds=last['decode_seconds'], ttft_ms=last['prefill_seconds'] * 1000,
            prompt_tokens=last['prompt_tokens'], generated_tokens=last['generated_tokens'], drafted_tokens=drafted, accepted_tokens=accepted,
            acceptance_rate=accepted / drafted if drafted and accepted is not None else None,
            verification_rounds=last.get('rounds'), plain_steps=last.get('plain_steps'), raw_metrics=dict(speculative_decoding=spec), finish_reason=None))
        return out


def main(argv=None):
    args = parse_args(argv)
    if not args.conditions or args.conditions[0] != 'baseline' or len(set(args.conditions)) != len(args.conditions):
        raise ValueError('Conditions must start with baseline and be unique')
    if args.warmup_episodes < 0:
        raise ValueError('warmup-episodes must be nonnegative')
    if not args.tpu_generation_chunk_tokens:
        raise SystemExit('The TPU benchmark needs --tpu-generation-chunk-tokens N (e.g. 512) and the cache tiers / vision slots of the TPU profile')
    from dental_agent.harness.configuration import harness_options, harness_identity
    from dental_agent.harness.specdec import episode_measurements, equivalence
    from dental_agent.harness.evaluation import atomic_json, load_dev, evaluate_episode, pooled, digest
    from dental_agent.model.tpu_inference import require_static_views, write_generation_report
    from dental_agent.model.tpu_speculative import parse_condition
    from dental_agent.model import tpu_generation as tg
    from dental_agent.model.local_runtime import runtime_from_args
    from dental_agent.harness.notebook_eval import adapter_signature, validate_resume_settings
    harness = harness_options(args.harness_mode)
    output = Path(args.output)
    saved = json.loads(output.read_text()) if output.exists() else None
    if saved and args.existing_results != 'resume':
        raise FileExistsError('Output exists; use --existing-results resume for this exact benchmark')
    if not Path(args.adapter_path).is_dir():
        raise FileNotFoundError(args.adapter_path)
    require_static_views(canonical_resize=True, letterbox=args.canonical_letterbox, what='spec_decode_bench_tpu')   # before the 9B loads
    adapter_sha256 = adapter_signature(args.adapter_path)
    if saved and saved.get('adapter_sha256') != adapter_sha256:
        raise ValueError('Saved benchmark adapter content differs or lacks provenance; evidence preserved')
    ids, images, annotations, categories = load_dev(args.data_dir, args.dev_split, args.limit, args.seed)
    if saved:
        validate_resume_settings(saved, vars(args), ids)
        if saved.get('status') in {'complete', 'failed_or_divergent'}:
            print('Terminal benchmark evidence retained:', output)
            if saved.get('status') == 'failed_or_divergent':
                raise SystemExit(2)
            return
    from scripts import evaluate_models as ev
    device, dtype, _ = ev.setup_eval_hardware(args.precision, 'auto', spmd=True)
    model, processor = ev.load_eval_model(args.model_id, args.adapter_path, device, dtype, args.checkpoint_kind + '_with_tools',
                                          quantize_4bit=False, runtime_options=runtime_from_args(args), max_new_tokens=args.max_new_tokens)
    config = tg.find_attr(model, tg.ATTR)
    if config is None:
        raise RuntimeError('The loaded model has no TPU generator installed')
    packages = {}
    for name in ['torch', 'torch_xla', 'transformers', 'peft']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = saved or dict(schema_version=1, status='running', settings=vars(args), software=packages, python=sys.version, platform=platform.platform(),
                           hardware=[dict(name='TPU v5e-8', tpu=True)], conditions={}, gates={}, measurements_pending=True,
                           backend='tpu_static_generation', commands={c: parse_condition(c).record() if c != 'baseline' else None for c in args.conditions})
    result.update(image_ids=ids, dev_split_sha256=digest(args.dev_split), adapter_sha256=adapter_sha256, harness=harness_identity(args.harness_mode),
                  draft_variant='ngram', settings=vars(args), status='running')
    atomic_json(output, result)
    report = output.with_name(output.stem + '_tpu_generation_report.json')

    def episode(iid, reply, tools=None):
        return evaluate_episode(iid, images, annotations, categories, model, processor, harness=harness, reply_fn=reply, seed=args.seed,
                                max_new_tokens=args.max_new_tokens, max_tool_calls=args.max_tool_calls, max_turns=args.max_turns,
                                letterbox=args.canonical_letterbox,
                                **({'tool_observer': lambda name, seconds: tools.append(dict(name=name, seconds=seconds))} if tools is not None else {}))

    for condition in args.conditions:
        entry = result['conditions'].setdefault(condition, dict(status='running', episodes=[]))
        if entry.get('status') == 'complete':
            continue
        entry['status'] = 'running'
        entry.pop('error', None)
        done = {int(r['image_id']) for r in entry['episodes']}
        if len(done) != len(entry['episodes']) or not done.issubset(ids):
            raise ValueError('Invalid saved benchmark cohort')
        try:
            config.speculative = parse_condition(condition)
            for j in range(args.warmup_episodes):                              # same warm-up episodes for every condition
                episode(ids[j % len(ids)], TPUReply(config))
            for iid in ids:
                if iid in done:
                    continue
                reply, tools = TPUReply(config), []
                start = time.perf_counter()
                record, _ = episode(iid, reply, tools)
                wall = time.perf_counter() - start
                record['measurements'] = episode_measurements(reply.requests, wall, sum(t['seconds'] for t in tools))
                record['requests'], record['tool_timings'], record['sampled_gpu_peak'] = reply.requests, tools, None
                entry['episodes'].append(record)
                atomic_json(output, result)
            entry['summary'] = pooled(entry['episodes'])
            entry['status'] = 'complete'
            if condition != 'baseline':
                gate = equivalence(result['conditions']['baseline']['episodes'], entry['episodes'])
                base_f1 = result['conditions']['baseline']['summary']['pooled_exact_f1']
                f1 = entry['summary']['pooled_exact_f1']
                gate['exact_f1_unchanged'] = base_f1 == f1 and base_f1 is not None
                gate['passed'] = gate['passed'] and gate['exact_f1_unchanged']
                result['gates'][condition] = gate
        except Exception as error:
            entry.update(status='failed', error=f'{type(error).__name__}: {error}')
            if condition == 'baseline':
                result['status'] = 'failed'
                atomic_json(output, result)
                raise
        finally:
            config.speculative = None
            write_generation_report(model, report)
        atomic_json(output, result)
    complete = all(e['status'] == 'complete' for e in result['conditions'].values())
    passed = complete and all(g['passed'] for g in result['gates'].values())
    result.update(status='complete' if passed else 'failed_or_divergent', measurements_pending=not complete)
    atomic_json(output, result)
    print('Result:', output, 'status:', result['status'])
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
