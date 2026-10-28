"""Final-evaluation orchestration; existing evaluators own inference, matching and bootstrap.

No eager model imports. No changes to running jobs or the frozen protocol. On 0048.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ('base_no_tools','base_with_tools','sft_no_tools','sft_with_tools','grpo_no_tools','grpo_with_tools')
RUNTIMES = {'dual_t4':('qlora','fp16','sdpa','on','staged',1),
            'rtx4090':('qlora','bf16','auto','auto','standard',0),
            'workstation48_lora':('bf16','auto','auto','auto','standard',0),
            'tpu':('bf16','auto','sdpa','off','standard',0)}
ALIASES = {'rtxpro6000_48gb':'workstation48_lora','rtx4090_qlora':'rtx4090'}


def leaf(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def profile(name):
    name = ALIASES.get(name, name)
    if name not in RUNTIMES: raise ValueError('Unknown profile: '+str(name))
    precision,compute,attn,kv,loader,threads = RUNTIMES[name]
    tags = leaf('dental_agent/training/experiment_profiles.py','_eval_profiles').PROFILE_TAGS
    return dict(name=name,tag=tags[name],precision=precision,compute=compute,attention=attn,
                kv=kv,loader=loader,threads=threads,letterbox=name=='tpu',efficient=name!='tpu')


def names(hardware, stage, k):
    p=profile(hardware)
    if not re.fullmatch(r'[A-Za-z0-9_]+',stage) or isinstance(k,bool) or not isinstance(k,int) or k<1:
        raise ValueError('Invalid stage or group size')
    inputs=leaf('dental_agent/utils/inputs.py','_eval_inputs')
    profiles=leaf('dental_agent/training/experiment_profiles.py','_eval_profiles')
    result={}
    for track in ('with_tools','no_tools'):
        result['sft_'+track]='sft/'+inputs.sft_adapter_name(track,stage,p['precision'])+'__'+p['tag']
        result['grpo_'+track]='grpo/'+profiles.grpo_name(track,k,stage,p['precision'],p['tag'])+'/grpo-'+track+'-final'
    return result


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=None
    try:
        with tempfile.NamedTemporaryFile('w',dir=path.parent,encoding='utf-8',delete=False) as stream:
            json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n');temp=Path(stream.name)
        temp.replace(path)
    finally:
        if temp and temp.exists():temp.unlink()


def lock(path,value):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError(f'Run identity changed: {path}. Use a new run_id; never mix results.')
    else:write(path,value)


def source_files():
    paths=list((ROOT/'dental_agent').rglob('*.py'))+list((ROOT/'scripts').rglob('*.py'))
    paths+=list((ROOT/'dental_agent/harness').rglob('*.md'))
    return {p.relative_to(ROOT).as_posix():digest(p) for p in sorted(paths)}


def runtime_flags(hardware):
    p=profile(hardware)
    return ['--precision',p['precision'],'--qlora-compute-dtype',p['compute'],'--attn-implementation',p['attention'],
            '--sdpa-repeat-kv',p['kv'],'--loader-mode',p['loader'],'--load-cpu-threads',str(p['threads']),
            '--require-efficient-sdpa' if p['efficient'] else '--no-require-efficient-sdpa']


def validate(cfg):
    profile(cfg['profile'])
    for key in ('stage','run_id'):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',cfg[key]):raise ValueError('Unsafe '+key)
    if cfg['mode'] not in ('dev','test'):raise ValueError('mode must be dev or test')
    conditions=cfg['conditions'];sizes=cfg['sizes'];limit=cfg.get('smoke_limit')
    if not conditions or len(set(conditions))!=len(conditions) or set(conditions)-set(CONDITIONS):raise ValueError('Invalid conditions')
    if not sizes or len(set(sizes))!=len(sizes) or set(sizes)-{'canonical','original'}:raise ValueError('Invalid sizes')
    if limit is not None and (isinstance(limit,bool) or not isinstance(limit,int) or limit<=0):raise ValueError('Invalid smoke_limit')
    if cfg['mode']=='test' and limit is not None:raise ValueError('Smoke tests belong to dev, not final test')
    for key,default in [('max_new_tokens',2048),('max_tool_calls',50),('n_boot',2000)]:
        if cfg.get(key,default)<=0:raise ValueError(key+' must be positive')
    for c in conditions:
        if not c.startswith('base_') and not cfg.get('adapters',{}).get(c):raise ValueError('Missing adapter: '+c)
    for stage in ('sft','grpo'):
        a,b=(cfg.get('adapters',{}).get(stage+'_'+track) for track in ('with_tools','no_tools'))
        if a and b and Path(a).resolve()==Path(b).resolve():raise ValueError(stage+': independently trained tracks require different adapters')


def adapter_identity(folder,condition,cfg=None):
    root=Path(folder).resolve()
    policy=root/'grpo_policy' if condition.startswith('grpo_') and (root/'grpo_policy').is_dir() else root
    weights=[policy/n for n in ('adapter_model.safetensors','adapter_model.bin') if (policy/n).is_file()]
    if not (policy/'adapter_config.json').is_file() or not weights or any(p.stat().st_size==0 for p in weights):
        raise FileNotFoundError(f'{condition}: missing adapter config/weights in {policy}')
    files=[policy/'adapter_config.json',*weights];metadata={}
    for parent in dict.fromkeys((root,policy,root.parent)):
        for name in ('sft_run_meta.json','checkpoint_meta.json'):
            p=parent/name
            if p.is_file():files.append(p);metadata[str(p)]=json.loads(p.read_text())
    if cfg:
        expected={'track':condition.split('_',1)[1],'stage':cfg['stage'],'precision':profile(cfg['profile'])['precision'],
                  'canonical_resize':True,'canonical_letterbox':profile(cfg['profile'])['letterbox'],'hf_run_tag':profile(cfg['profile'])['tag']}
        for item in metadata.values():
            item={**item,**(item.get('extra') or {})}
            for key,value in expected.items():
                if key in item and item[key]!=value:raise ValueError(f'{condition}: training metadata {key} mismatch')
            if condition.startswith('grpo_') and 'group_size' in item and item['group_size']!=cfg['selected_k']:
                raise ValueError('GRPO group size mismatch')
    portable={str(p.relative_to(root)) if p.is_relative_to(root) else '../'+p.name:digest(p) for p in sorted(set(files))}
    return dict(path=str(root),policy_path=str(policy),files=portable,metadata=metadata)


def matrix_command(cfg,conditions,output=None,extra=(),sizes=None):
    validate(cfg)
    if extra and cfg['mode']!='dev':raise ValueError('Ablations and sweep probes are dev-only')
    cmd=[cfg.get('python',sys.executable),'-u','scripts/run_eval_matrix.py','--conditions',*conditions,
         '--sizes',*(sizes or cfg['sizes']),'--model-id',cfg['model_id'],'--data-dir',cfg['data_dir'],
         '--dev-split',cfg['dev_split'],'--split',cfg['mode'],'--dataset','dentex','--output-dir',str(output or cfg['output']),
         '--existing-results','resume','--seed',str(cfg.get('seed',42)),'--n-boot',str(cfg.get('n_boot',2000)),
         '--max-new-tokens',str(cfg.get('max_new_tokens',2048)),'--max-tool-calls',str(cfg.get('max_tool_calls',50)),
         '--malformed-retries',str(cfg.get('malformed_retries',0)),'--iou-threshold',str(cfg.get('iou_threshold',.5)),
         *runtime_flags(cfg['profile']),'--canonical-letterbox' if profile(cfg['profile'])['letterbox'] else '--no-canonical-letterbox']
    if cfg.get('max_turns') is not None:cmd+=['--max-turns',str(cfg['max_turns'])]
    if cfg.get('smoke_limit') is not None:cmd+=['--limit',str(cfg['smoke_limit'])]
    for stage in ('sft','grpo'):
        chosen=[c for c in conditions if c.startswith(stage+'_')]
        if chosen:
            paths={cfg['adapters'][c] for c in chosen}
            if len(paths)!=1:raise ValueError('Run separately trained tracks separately')
            cmd+=['--'+stage+'-adapter',paths.pop()]
    return cmd+list(extra)


def primary_commands(cfg):
    validate(cfg);commands=[]
    base=[c for c in cfg['conditions'] if c.startswith('base_')]
    if base:commands.append(matrix_command(cfg,base))
    commands += [matrix_command(cfg,[c]) for c in cfg['conditions'] if not c.startswith('base_')]
    return commands


def probe_commands(cfg,tools,perturbations):
    if cfg['mode']!='dev':raise ValueError('Probes never run on test')
    out=[]
    for c in [c for c in cfg['conditions'] if c.endswith('with_tools') and not c.startswith('base_')]:
        out += [matrix_command(cfg,[c],extra=['--remove-tools',tool],sizes=['canonical']) for tool in tools]
        for shift,scale in perturbations:
            if not math.isfinite(shift) or not math.isfinite(scale) or shift<0 or scale<=0:raise ValueError('Invalid perturbation')
            out.append(matrix_command(cfg,[c],extra=['--locate-shift-frac',str(shift),'--locate-scale',str(scale),
                        '--ablation-seed',str(cfg.get('seed',42))],sizes=['canonical']))
    return out


def run(commands,logs):
    logs=Path(logs);logs.mkdir(parents=True,exist_ok=True)
    for command in commands:
        key=fingerprint(command)[:16];write(logs/(key+'.command.json'),command)
        env=os.environ.copy();env['PYTHONHASHSEED']='42';env['PYTHONIOENCODING']='utf-8'
        with (logs/(key+'.log')).open('a',encoding='utf-8') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                   text=True,encoding='utf-8',errors='replace')
            try:
                for line in child.stdout:print(line,end='',flush=True);log.write(line);log.flush()
                if child.wait():raise subprocess.CalledProcessError(child.returncode,command)
            finally:
                if child.poll() is None:
                    child.send_signal(signal.SIGINT) if os.name!='nt' else child.terminate()
                    try:child.wait(timeout=45)
                    except subprocess.TimeoutExpired:child.kill();child.wait()


def strict_records(path):
    result=[]
    for i,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
        if line.strip():
            try:result.append(json.loads(line))
            except ValueError as error:raise ValueError(f'{path}:{i}: malformed JSONL; retain the raw file before repair') from error
    ids=[str(r['image_id']) for r in result]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate image IDs in '+str(path))
    return result


def cohort(rows,ids,split):
    actual=[int(r['image_id']) for r in rows]
    if len(actual)!=len(set(actual)) or set(actual)!=set(map(int,ids)):raise ValueError('Incomplete, extra or duplicate cohort')
    if any(r.get('split')!=split or r.get('prompt')!='unified' for r in rows):raise ValueError('Mixed split or non-unified prompt')
    return rows


def completion(cfg,ids):
    from dental_agent.evaluation.matrix import cell_path
    result={}
    for c in cfg['conditions']:
        for size in cfg['sizes']:
            path=cell_path(cfg['output'],'dentex',cfg['mode'],c,size)
            rows=cohort(strict_records(path),ids,cfg['mode'])
            if any(r.get('condition')!=c or r.get('image_view')!=size for r in rows):raise ValueError('Mixed condition/geometry')
            result[c+'@'+size]=dict(images=len(rows),sha256=digest(path))
    return result


def selected(cfg):
    omit={'mode','output','run_id','smoke_limit','adapters','model_id','python','data_dir','dev_split'}
    return {k:v for k,v in cfg.items() if k not in omit}


def freeze(path,cfg,identity,done):
    if cfg['mode']!='dev' or cfg.get('smoke_limit') is not None:raise ValueError('Freeze requires a full dev run')
    expected={c+'@'+z for c in CONDITIONS for z in ('canonical','original')}
    if set(done)!=expected:raise ValueError('All twelve dev cells are required')
    lock(path,dict(schema_version=1,selection=selected(cfg),identity=identity,dev_completion=done))


def verify_frozen(path,cfg,identity):
    saved=json.loads(Path(path).read_text())
    if saved['selection']!=selected(cfg) or saved['identity']!=identity:
        raise ValueError('Settings/checkpoint/code identity differs from frozen dev selection')
    return saved


def calibration(output,split,destination):
    from dental_agent.evaluation.matrix import load_matrix
    from dental_agent.evaluation.metrics import finding_calibration_pairs,expected_calibration_error
    result={}
    for name,rows in load_matrix(output,'dentex',split).items():
        pairs=[];missing=0
        for row in rows:
            predictions=copy.deepcopy(row.get('predictions') or []);usable=[]
            for p in predictions:
                value=p.get('confidence');valid=isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and 0<=value<=1
                usable.append(valid)
                if not valid:missing+=1;p['confidence']=.5
            scored=finding_calibration_pairs(row.get('ground_truth') or [],predictions)
            pairs += [pair for pair,valid in zip(scored,usable) if valid]
        result[name]=dict(n=len(pairs),missing_confidence=missing,pairs=pairs,
            ece=expected_calibration_error([p[0] for p in pairs],[p[1] for p in pairs]) if len(pairs)>=5 else None,
            caveat='Shared legacy ECE excludes confidence exactly zero; invalid/missing confidences are not imputed into ECE.')
    write(destination,result);return result


def primary_tex(summary,path):
    fmt=lambda value:r'\todo{pending}' if value is None else f'{value:.3f}'
    lines=[r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule',r'Condition & Exact F1 & 95\% CI & Tooth F1 & Healthy invention \\',r'\midrule']
    for name,c in sorted(summary['cells'].items()):
        m=c['exact'];healthy=c['healthy'];rate=healthy['false_positive_image_rate'] if healthy['n_healthy'] else None
        lines.append(' & '.join([name.replace('_',r'\_').replace('@',' / '),fmt(m['f1']),
            '['+fmt(m.get('f1_ci_low'))+', '+fmt(m.get('f1_ci_high'))+']',fmt(c['fdi']['f1']),fmt(rate)])+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}'];Path(path).write_text('\n'.join(lines)+'\n')
