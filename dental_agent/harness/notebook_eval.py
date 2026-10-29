"""Shared, opt-in notebook evaluation coordination. No training imports or repo creation.

The existing evaluators own prompts, tools and scoring. This layer owns cohort/profile
identity, missing-only input resolution, and portable result transport in the existing HF_EVAL_REPO, separate from model checkpoints.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PROFILES = {
    'dual_t4': dict(tag='dual_t4_fp16', precision='qlora', compute='fp16', attention='sdpa', kv='on', loader='staged', threads=1, letterbox=False),
    'rtx4090_qlora': dict(tag='rtx4090_bf16', precision='qlora', compute='bf16', attention='auto', kv='auto', loader='standard', threads=0, letterbox=False),
    'workstation48_lora': dict(tag='workstation48_lora_bf16', precision='bf16', compute='auto', attention='auto', kv='auto', loader='standard', threads=0, letterbox=False),
    'tpu': dict(tag='tpu_bf16', precision='bf16', compute='auto', attention='sdpa', kv='off', loader='standard', threads=0, letterbox=True),
    'dual_t4_fp16': dict(tag='dual_t4_unquantized_fp16', precision='fp16', compute='auto', attention='sdpa', kv='on', loader='staged', threads=1, letterbox=False),
}
ALIASES = {'rtx4090': 'rtx4090_qlora', 'rtxpro6000_48gb': 'workstation48_lora'}
KINDS = ('zeroshot', 'baseline', 'sft', 'grpo', 'rules_skills', 'speculative')
PRIMARY = ('base_no_tools', 'base_with_tools', 'sft_no_tools', 'sft_with_tools', 'grpo_no_tools', 'grpo_with_tools')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def select_profiles(target, profiles=None):
    profiles = PROFILES if profiles is None else profiles
    values = list(profiles) if target == 'all' else ([target] if isinstance(target, str) else list(target))
    result = []
    for value in values:
        value = ALIASES.get(value, value)
        if value not in profiles: raise ValueError('Unknown EVAL_TARGET_PROFILES: ' + str(value))
        if value not in result: result.append(value)
    if not result: raise ValueError('Select at least one evaluation profile')
    return result


def runtime_flags(profile, profiles=None):
    profiles = PROFILES if profiles is None else profiles
    p = profiles[select_profiles(profile, profiles)[0]]
    return ['--precision', p['precision'], '--qlora-compute-dtype', p['compute'], '--attn-implementation', p['attention'],
            '--sdpa-repeat-kv', p['kv'], '--loader-mode', p['loader'], '--load-cpu-threads', str(p['threads']),
            '--no-require-efficient-sdpa' if profile == 'tpu' else '--require-efficient-sdpa']


def hardware_reason(profile, kind, devices, profiles=None):
    """Devices contain actual name/capability/memory; never emulate BF16 or pool TPU HBM."""
    if profile == 'tpu': return 'pending: existing evaluators have no sharded 9B TPU generation path'
    if not devices: return 'pending: CUDA evaluation runtime required'
    p = (PROFILES if profiles is None else profiles)[profile]
    if (p['precision'] == 'bf16' or p['compute'] == 'bf16') and any(d['capability'][0] < 8 for d in devices):
        return 'pending: this cohort requires native BF16 (Ampere or newer)'
    if p['precision'] != 'qlora' and sum(d['gib'] for d in devices) < 18:
        return 'pending: device memory is below the 16-bit 9B weight footprint, before activations'
    if kind == 'speculative' and any(d['capability'][0] < 8 for d in devices):
        return 'pending: no verified Turing backend for this configured Qwen3.5/MTP vLLM path; native FP16 evaluation remains available'
    return None


def discover(files, profiles, stages=None, selected_k=4, profile_configs=None):
    """Only separately trained root SFT adapters and final GRPO policies; never references."""
    profile_configs = PROFILES if profile_configs is None else profile_configs
    jobs = []
    stages = None if stages == 'all' or stages is None else ({stages} if isinstance(stages, str) else set(stages))
    for name in sorted(files):
        named_mode = re.search(r'--harness_(rules_skills_index|rules_skills_full)(?=/)', name)
        mode = named_mode[1] if named_mode else 'off'
        parsed_name = re.sub(r'--harness_(rules_skills_index|rules_skills_full)(?=/)', '', name)
        match = re.fullmatch(r'(sft|grpo)/(qwen3_5_9b_(?:sft|grpo)_(with_tools|no_tools)_(?:k(\d+)_)?(.+)_(qlora|bf16|fp16)__([A-Za-z0-9_-]+))(.*)/adapter_config\.json', parsed_name)
        if not match: continue
        kind, stem, track, k, stage, precision, tag, tail = match.groups()
        if mode != 'off' and (kind != 'grpo' or track != 'with_tools'): continue
        if stages is not None and stage not in stages: continue
        profile = next((p for p in profiles if profile_configs[p]['tag'] == tag and profile_configs[p]['precision'] == precision), None)
        if profile is None: continue
        if kind == 'sft' and (k or tail): continue
        if kind == 'grpo':
            if not k or (selected_k != 'all' and int(k) != int(selected_k)): continue
            expected = '/grpo-' + track + '-final'
            if tail not in (expected, expected + '/grpo_policy'): continue
            if tail == expected and name[:-len('/adapter_config.json')] + '/grpo_policy/adapter_config.json' in files: continue
        prefix = name.rsplit('/', 1)[0]
        weight = next((prefix + '/adapter_model' + ext for ext in ('.safetensors', '.bin') if prefix + '/adapter_model' + ext in files), None)
        if not weight: continue
        jobs.append(dict(kind=kind, condition=kind+'_'+track, track=track, profile=profile, stage=stage,
                         k=int(k) if k else None, adapter=prefix, weight=weight, harness_mode=mode))
    return jobs


def _metadata_digest(metadata):
    lfs = getattr(metadata, 'lfs', None)
    result = (lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)) or getattr(metadata, 'blob_id', None)
    if not result: raise ValueError('Hub file has no content identity')
    return result


def inventory_roots(data_dir):
    roots = [Path(data_dir), Path('/kaggle/input'), Path('/content/drive/MyDrive/VLM-DENTAL/data'),
             Path(os.environ.get('HF_HUB_CACHE') or Path(os.environ.get('HF_HOME', str(Path.home()/'.cache/huggingface'))) / 'hub')]
    for key in ('DENTEX_LOCAL_DIR', 'DENTEX_DATA_DIR', 'DENTEX_ROOT'):
        if os.environ.get(key): roots.insert(0, Path(os.environ[key]))
    files = set()
    for root in dict.fromkeys(p.expanduser().resolve() for p in roots if p.exists()):
        for folder, dirs, names in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in {'.git','__pycache__','evaluations','evaluation_inputs','invalidated_evaluations'})
            for name in names:
                if Path(name).suffix.lower() in {'.png','.jpg','.jpeg','.json','.zip'}: files.add(Path(folder)/name)
    return sorted(files)


def _image_ok(path, item):
    if not path or not Path(path).is_file(): return False
    from PIL import Image
    try:
        with Image.open(path) as im:
            ok = im.size == (int(item['width']), int(item['height'])); im.verify()
        return ok
    except (OSError, ValueError, KeyError, TypeError): return False


def prepare_inputs(data_dir, split, dev_split, api, download, token, files=None):
    """Resolve the entire cohort before any model load; train/test ID collisions stay separate."""
    data = Path(data_dir).expanduser().resolve()
    if split not in {'dev','test'}: raise ValueError('Only fixed dev and official test are supported')
    expected = set(map(int, json.loads(Path(dev_split).read_text())['image_ids'])) if split == 'dev' else None
    files = inventory_roots(data) if files is None else files
    candidates = []
    for path in files:
        name = path.name.lower()
        if split == 'test' and name != 'validation_triple.json': continue
        if split == 'dev' and not (name == 'train.json' or ('disease' in name and 'val' not in name)): continue
        try:
            payload = json.loads(path.read_text())
            ids = [int(i['id']) for i in payload['images']]
            if len(ids) != len(set(ids)) or not payload.get('annotations'): continue
            if split == 'test' and len(ids) != 50: continue
            if expected is not None and not expected.issubset(ids): continue
            richness = sum('category_id_3' in a for a in payload['annotations'])
            if not richness: continue
            candidates.append((richness, str(path), payload))
        except (ValueError, TypeError, KeyError): continue
    repo = os.environ.get('DENTEX_IMAGES_REPO', 'Reza-Nadimi/dentex-train-images')
    revision = None
    if candidates:
        _, name, coco = sorted(candidates, key=lambda x: (-x[0], x[1]))[0]; annotation = Path(name)
    else:
        revision = api.dataset_info(repo).sha
        annotation = Path(download(repo, 'train.json' if split == 'dev' else 'validation_triple.json', repo_type='dataset', revision=revision, token=token))
        coco = json.loads(annotation.read_text())
    ids = [int(i['id']) for i in coco['images']]
    if len(ids) != len(set(ids)): raise ValueError('Duplicate source image IDs')
    expected = set(ids) if expected is None else expected
    if split == 'test' and len(expected) != 50: raise ValueError('Official test must contain all 50 images')
    if not expected.issubset(ids): raise ValueError('Incomplete fixed dev membership')
    coco = copy.deepcopy(coco)
    coco['images'] = [i for i in coco['images'] if int(i['id']) in expected]
    coco['annotations'] = [a for a in coco['annotations'] if int(a['image_id']) in expected]
    paths = {}; hashes = {}
    image_files = [p for p in files if p.suffix.lower() in {'.png','.jpg','.jpeg'}]
    namespace = 'validation_images' if split == 'test' else 'images'
    for item in coco['images']:
        iid = int(item['id']); native = str(item.get('file_name') or '').replace('\\','/')
        local = str(item.get('local_path') or '').replace('\\','/')
        direct = [Path(native), Path(local), annotation.parent/native, annotation.parent/namespace/f'{iid}.png']
        hits = [p for p in direct if _image_ok(p,item)]
        if not hits:
            for p in image_files:
                parts = {v.lower() for v in p.parts}
                is_val = any('validation' in v or v == 'test' for v in parts)
                native_name = Path(native).name
                # Bare numeric IDs only in a proven split namespace.
                known = ((namespace in parts or any(('validation_data' if split=='test' else 'training_data') in v for v in parts)) and is_val == (split == 'test'))
                named = native_name and p.name == native_name and ('val' if split == 'test' else 'train') in p.name.lower()
                if (known and p.name in {f'{iid}.png',native_name} or named) and _image_ok(p,item): hits.append(p)
        if hits and len({digest(p) for p in hits}) == 1: paths[iid] = hits[0].resolve()
    # Extract only needed members from existing archives, never delete archives.
    for archive in [p for p in files if p.suffix == '.zip' and ('validation_data' if split == 'test' else 'training_data') in p.name]:
        if set(paths) == expected: break
        with zipfile.ZipFile(archive) as z:
            members = z.namelist()
            for item in coco['images']:
                iid = int(item['id'])
                if iid in paths: continue
                names = [n for n in members if Path(n).name == Path(str(item['file_name']).replace('\\','/')).name]
                if len(names) != 1: continue
                out = data/'evaluation_inputs'/'extracted'/split/f'{iid}.png'; out.parent.mkdir(parents=True, exist_ok=True)
                if not _image_ok(out,item): out.write_bytes(z.read(names[0]))
                if _image_ok(out,item): paths[iid] = out.resolve()
    missing = expected-set(paths)
    if missing:
        revision = revision or api.dataset_info(repo).sha
        remote = set(api.list_repo_files(repo, repo_type='dataset', revision=revision))
        for item in coco['images']:
            iid = int(item['id'])
            if iid not in missing: continue
            name = f'{namespace}/{iid}.png'
            if name not in remote: continue
            p = Path(download(repo,name,repo_type='dataset',revision=revision,token=token))
            if _image_ok(p,item): paths[iid] = p.resolve()
    if set(paths) != expected: raise FileNotFoundError('Cohort incomplete; no evaluation launched. Missing IDs: '+str(sorted(expected-set(paths))))
    hashes = {str(i):digest(paths[i]) for i in sorted(paths)}
    # Ignore mount/source_file differences in portable content identity.
    portable = copy.deepcopy(coco)
    for item in portable['images']:
        for key in ('file_name','source_file','local_path'): item.pop(key,None)
    key = fingerprint(dict(coco=portable,images=hashes,split=split))
    pathkey = fingerprint({str(i):str(p) for i,p in paths.items()})[:12]
    out = data/'evaluation_inputs'/split/key[:16]/pathkey
    (out/('validation_data' if split == 'test' else 'training_data')).mkdir(parents=True,exist_ok=True)
    for item in coco['images']:
        item['file_name'] = str(paths[int(item['id'])]); item.pop('source_file',None); item.pop('local_path',None)
    write(out/('validation_triple.json' if split == 'test' else 'train.json'),coco)
    return dict(data_dir=str(out), ids=sorted(expected), hashes=hashes, fingerprint=key, split=split)


def result_repo_type(api, repo, configured=None):
    if configured is not None:
        if configured not in {'model', 'dataset'}: raise ValueError('Results repo type must be model or dataset')
        api.repo_info(repo, repo_type=configured)
        return configured
    # Resolve an existing destination, without create_repo or broad exception fallback.
    if not hasattr(api, 'repo_info'): return 'model'  # Lightweight offline API doubles.
    from huggingface_hub.errors import RepositoryNotFoundError
    found = []
    for kind in ('dataset', 'model'):
        try: api.repo_info(repo, repo_type=kind)
        except RepositoryNotFoundError as error:
            if getattr(getattr(error, 'response', None), 'status_code', None) != 404: raise
        else: found.append(kind)
    if len(found) != 1:
        raise ValueError('Set EVAL_RESULTS_REPO_TYPE explicitly; expected one accessible existing results repository: '+repo)
    return found[0]


def result_path(identity):
    """Readable cohort names; content identity separates settings/checkpoint revisions."""
    runtime = identity.get('runtime') or {}
    parts = ['evaluations', 'dentex', identity['split'], runtime.get('tag', identity['profile']),
             identity.get('stage', 'base'), identity['kind'], identity['condition']]
    if identity.get('harness_mode','off') != 'off': parts.append('harness_'+identity['harness_mode'])
    if identity.get('k') is not None: parts.append('k'+str(identity['k']))
    parts += [identity.get('size', 'canonical'), 'run_'+fingerprint(identity)[:16]]
    if any(not re.fullmatch(r'[A-Za-z0-9_.-]+', str(part)) or str(part) in {'.','..'} for part in parts):
        raise ValueError('Unsafe evaluation path component')
    return '/'.join(map(str, parts))


class HubResults:
    """Result-only, atomic bundles in an EXISTING results repo; optimistic concurrency."""
    def __init__(self, repo, api, download, token=None, repo_type="model"):
        self.repo_type = repo_type
        self.repo, self.api, self.download, self.token = repo, api, download, token

    def remote(self, prefix):
        info = self.api.repo_info(self.repo, repo_type=self.repo_type) if hasattr(self.api, 'repo_info') else self.api.model_info(self.repo)
        names = set(self.api.list_repo_files(self.repo, repo_type=self.repo_type, revision=info.sha))
        name = prefix+'/bundle.json'
        if name not in names: return info.sha, None
        path = self.download(self.repo,name,repo_type=self.repo_type,revision=info.sha,token=self.token)
        return info.sha,json.loads(Path(path).read_text())

    def restore(self, folder, prefix, identity):
        folder = Path(folder); revision, bundle = self.remote(prefix)
        if bundle is None: return None
        if bundle['identity'] != identity: raise ValueError('Remote evaluation identity conflict; results preserved')
        payloads = {}
        for name, sha in bundle['files'].items():
            if PurePosixPath(name).name != name or '\\' in name: raise ValueError('Unsafe bundle path')
            file = Path(self.download(self.repo,prefix+'/'+name,repo_type=self.repo_type,revision=revision,token=self.token))
            data = file.read_bytes()
            if hashlib.sha256(data).hexdigest() != sha: raise ValueError('Remote evaluation checksum mismatch')
            payloads[name] = data
        folder.mkdir(parents=True,exist_ok=True)
        # Preserve live local progress; do not replace it with a shorter remote copy.
        for name,data in payloads.items():
            target = folder/name
            if not target.exists(): target.write_bytes(data)
            elif name.startswith('results'):
                merged = merge_evidence(name,target.read_bytes(),data)
                if merged != target.read_bytes():
                    temp = target.with_name(target.name+'.restore.tmp');temp.write_bytes(merged);temp.replace(target)
        return bundle

    def save(self, folder, prefix, identity, payloads, progress, complete, _attempt=0):
        from huggingface_hub import CommitOperationAdd
        revision, old = self.remote(prefix)
        digests = {name:hashlib.sha256(data).hexdigest() for name,data in payloads.items()}
        if old:
            if old['identity'] != identity: raise ValueError('Remote identity conflict; upload refused')
            if old['progress'] > progress: raise ValueError('Remote has more progress; upload refused, both copies preserved')
            for name,sha in old['files'].items():
                if name not in payloads or not name.startswith('results'): continue
                path = Path(self.download(self.repo,prefix+'/'+name,repo_type=self.repo_type,revision=revision,token=self.token))
                previous = path.read_bytes()
                if hashlib.sha256(previous).hexdigest()!=sha: raise ValueError('Remote checksum mismatch')
                before = episode_map(name,previous); after = episode_map(name,payloads[name])
                if any(after.get(key)!=value for key,value in before.items()):
                    raise ValueError('Conflicting or missing previously saved episodes; upload refused')
            if old['files'] == digests and old['complete'] == complete: return
        manifest = dict(schema_version=1,identity=identity,files=digests,progress=progress,complete=complete)
        payloads = {**payloads,'bundle.json':(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()}
        try:
            self.api.create_commit(self.repo,repo_type=self.repo_type,parent_commit=revision,
                operations=[CommitOperationAdd(path_in_repo=prefix+'/'+name,path_or_fileobj=data) for name,data in payloads.items()],
                commit_message=f"eval: {identity['profile']} {identity['kind']} {progress} completed episodes")
        except Exception as error:
            # Only retry a confirmed rejected parent commit, never an upload with unknown outcome.
            if getattr(getattr(error,'response',None),'status_code',None)==409 and _attempt<2:
                return self.save(folder,prefix,identity,{k:v for k,v in payloads.items() if k!='bundle.json'},progress,complete,_attempt+1)
            raise
        print('[HUB SAVED]',prefix,progress,'complete' if complete else 'partial',flush=True)


def jsonl_rows(path):
    if not Path(path).exists(): return []
    rows = []
    for line in Path(path).read_bytes().splitlines(keepends=True):
        if not line.endswith(b'\n'): break  # never upload a torn final record
        if line.strip(): rows.append(json.loads(line))
    ids = [int(r['image_id']) for r in rows]
    if len(ids) != len(set(ids)): raise ValueError('Duplicate evaluation image IDs; raw evidence preserved')
    return rows


def validate_document(kind, path, identity):
    """Return portable bytes/progress/completion. Failed/divergent benchmarks remain evidence."""
    ids = set(identity['image_ids'])
    if not Path(path).is_file(): return {},0,False
    if kind in {'zeroshot','baseline','sft','grpo'}:
        rows = jsonl_rows(path)
        for row in rows:
            if int(row['image_id']) not in ids: raise ValueError('Result contains an extra image')
            if row.get('split') != identity['split']: raise ValueError('Mixed evaluation splits')
            if kind != 'zeroshot':
                if row.get('condition') != identity['condition'] or row.get('prompt') != 'unified': raise ValueError('Wrong condition/prompt')
                settings = row.get('eval_settings') or {}
                if settings.get('max_new_tokens') != 16384: raise ValueError('Wrong evaluation budget')
                if settings.get('harness',{'mode':'off'}) != identity.get('harness',{'mode':'off'}): raise ValueError('Wrong evaluation harness')
            else:
                if row.get('provider') not in {'transformers','local_hf'}: raise ValueError('API zero-shot results are not local baselines')
                if (row.get('evaluation_settings') or {}).get('max_tokens') != 16384: raise ValueError('Zero-shot budget lacks 16K provenance')
            # Adapter path is runtime-specific; identity certifies the exact checkpoint hashes instead.
            if 'eval_settings' in row:
                row['eval_settings']['adapter_path'] = identity.get('adapter') or ''
                row['eval_settings']['model_id'] = identity['base_model']
            if 'adapter_path' in row: row['adapter_path'] = identity.get('adapter') or ''
            if 'model_id' in row: row['model_id'] = identity['base_model']
        payload = ''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows).encode()
        return {'results.jsonl':payload},len(rows),len(rows)==len(ids)
    doc = json.loads(Path(path).read_text())
    if doc.get('split',doc.get('settings',{}).get('split')) != 'dev': raise ValueError('Extensions must remain dev-only')
    if set(map(int,doc.get('image_ids',[]))) != ids: raise ValueError('Extension cohort mismatch')
    if doc.get('settings',{}).get('max_new_tokens') != 16384: raise ValueError('Extension budget mismatch')
    groups = doc.get('conditions') or {}; progress = 0
    for name,entry in groups.items():
        rows = entry.get('records',entry.get('episodes',[])); seen = [int(r['image_id']) for r in rows]
        if len(seen)!=len(set(seen)) or not set(seen).issubset(ids): raise ValueError('Extension cohort contains duplicates/extra images')
        progress += len(rows)
    complete = doc.get('status') == 'complete' and set(groups) == set(identity['conditions']) and all(
        len(e.get('records',e.get('episodes',[]))) == len(ids) for e in groups.values())
    # Preserve divergent benchmark reports as terminal evidence, but do not call them passed.
    terminal = complete or (kind == 'speculative' and doc.get('status') == 'failed_or_divergent' and all(
        len(e.get('episodes',[])) == len(ids) for e in groups.values()) and set(groups) == set(identity['conditions']))
    return {'results.json':(json.dumps(doc,sort_keys=True,allow_nan=False)+'\n').encode()},progress,terminal


def transient_upload_error(error):
    """Retry transport failures and temporary HTTP errors; data/auth conflicts stay fatal."""
    status = getattr(getattr(error, 'response', None), 'status_code', None)
    if status is not None: return status in {408, 429} or 500 <= status < 600
    if isinstance(error, (ConnectionError, TimeoutError)): return True
    return any((base.__module__.split('.')[0], base.__name__) in {
        ('httpx', 'TransportError'), ('httpx2', 'TransportError'),
        ('requests', 'ConnectionError'), ('requests', 'Timeout'),
    } for base in type(error).__mro__)


class NotebookEvaluations:
    """Inventory first, restore before weights, execute only missing compatible cohorts."""
    def __init__(self, *, target_profiles='dual_t4', kinds=KINDS, stages='all', selected_k=4,
                 data_dir='data', dev_split='data/splits/dentex_dev.json', output_dir='data/evaluations',
                 model_id=None, split='test', sizes=('canonical',), spec_checkpoints=None,
                 vllm_executable='vllm', gpu_indices=None, port=8910, seed=42, serving_context_length=32768, ablation=None, api=None, download=None, profiles=None,
                 results_repo=None, results_repo_type=None, track="all", grpo_harness_modes='off',
                 speculative_harness_mode=None, speculative_target_kind=None, adapter_prefixes=None, draft_variant='original'):
        if split not in {'test','dev'} or set(kinds)-set(KINDS) or not kinds: raise ValueError('Invalid split or kinds')
        if set(sizes)-{'canonical','original'} or not sizes: raise ValueError('Invalid sizes')
        if ablation and split!='dev': raise ValueError('Tool/grounding ablations belong to dev only')
        self.tracks = ('with_tools', 'no_tools') if track == 'all' else (track,)
        if any(t not in {'with_tools', 'no_tools'} for t in self.tracks):
            raise ValueError('EVAL_TRACK must be with_tools, no_tools, or all')
        from dental_agent.harness.configuration import HARNESS_MODES, validate_harness_mode
        self.grpo_harness_modes = HARNESS_MODES if grpo_harness_modes == 'all' else ((grpo_harness_modes,) if isinstance(grpo_harness_modes,str) else tuple(grpo_harness_modes))
        if not self.grpo_harness_modes: raise ValueError('Select at least one GRPO harness variant')
        for mode in self.grpo_harness_modes: validate_harness_mode(mode)
        if speculative_harness_mode is not None: validate_harness_mode(speculative_harness_mode)
        if speculative_target_kind not in {None,'sft','grpo'}: raise ValueError('Unknown speculative target kind')
        if draft_variant not in {'original','finetuned'}: raise ValueError('Unknown MTP draft variant')
        self.speculative_harness_mode,self.speculative_target_kind = speculative_harness_mode,speculative_target_kind
        self.adapter_prefixes = None if adapter_prefixes is None else set(adapter_prefixes)
        self.draft_variant = draft_variant
        self.ablation = copy.deepcopy(ablation)
        if serving_context_length<=16384: raise ValueError('Serving total context must exceed the unchanged 16384 generation budget to leave input/image headroom')
        self.serving_context_length=serving_context_length
        if split=='test' and selected_k=='all': raise ValueError('K sweep belongs to dev; select K before test')
        if api is None or download is None:
            from huggingface_hub import HfApi,hf_hub_download
            api = api or HfApi(token=os.environ.get('HF_TOKEN')); download = download or hf_hub_download
        self.repo = os.environ.get('HF_ARTIFACT_REPO','Reza-Nadimi/vlm-dental-models')
        self.results_repo = results_repo or os.environ.get('HF_EVAL_REPO','Reza-Nadimi/vlm-dental-evaluations')
        self.results_repo_type = result_repo_type(api, self.results_repo, results_repo_type)
        self.profile_configs = copy.deepcopy(PROFILES if profiles is None else profiles)
        self.api,self.download,self.token = api,download,os.environ.get('HF_TOKEN')
        self.profiles,self.kinds,self.stages,self.k = select_profiles(target_profiles, self.profile_configs),tuple(kinds),stages,selected_k
        self.data,self.dev_split,self.output = Path(data_dir),Path(dev_split),Path(output_dir)
        self.model_id,self.split,self.sizes = model_id,split,tuple(sizes)
        self.spec_checkpoints = spec_checkpoints or {}; self.vllm,self.gpu_indices,self.port,self.seed = vllm_executable,gpu_indices,port,seed
        self.store = HubResults(self.results_repo,api,download,self.token,self.results_repo_type)
        info = api.model_info(self.repo,files_metadata=True)
        self.revision,self.files = info.sha,{s.rfilename:s for s in info.siblings}
        self.inputs,self.jobs,self.status = {},[],[]
        self._inventory_files = None
        self._remote_manifests = None
        self.source = self._source_identity()
        self.software=software_versions()
        self._discover()

    def _source_identity(self):
        paths = []
        for folder in ('dental_agent/agent','dental_agent/model','dental_agent/tools','dental_agent/evaluation','dental_agent/harness'):
            paths += list((ROOT/folder).rglob('*.py'))
        paths += list((ROOT/'dental_agent/harness').rglob('*.md'))
        paths += [ROOT/'dental_agent/paper/prompts.py', ROOT/'dental_agent/utils/canonical.py', ROOT/'dental_agent/data/dentex.py']
        paths += [ROOT/p for p in ('scripts/evaluate_models.py','scripts/run_zero_shot.py','scripts/harness/run_dev.py','scripts/bench/spec_decode_bench.py')]
        return fingerprint({p.relative_to(ROOT).as_posix():digest(p) for p in sorted(set(paths))})

    def _discover(self):
        trained = discover(self.files,self.profiles,self.stages,self.k,self.profile_configs)
        trained = [job for job in trained if job['track'] in self.tracks and (job['kind']!='grpo' or job['harness_mode'] in self.grpo_harness_modes)]
        for profile in self.profiles:
            if 'zeroshot' in self.kinds:
                self.jobs.append(dict(kind='zeroshot',condition='zero_shot',profile=profile,stage='base',adapter=None,k=None))
            if 'baseline' in self.kinds:
                for track in self.tracks:
                    self.jobs.append(dict(kind='baseline',condition='base_'+track,track=track,profile=profile,stage='base',adapter=None,k=None))
            for job in trained:
                if job['profile'] != profile: continue
                if job['kind'] in self.kinds: self.jobs.append(job)
                if job['track']=='with_tools':
                    for kind in ('rules_skills','speculative'):
                        if kind not in self.kinds: continue
                        if kind == 'speculative':
                            if self.speculative_target_kind and job['kind'] != self.speculative_target_kind: continue
                            if self.speculative_harness_mode is not None and job['harness_mode'] != self.speculative_harness_mode: continue
                            if self.adapter_prefixes is not None and job['adapter'] not in self.adapter_prefixes: continue
                        self.jobs.append({**job,'kind':kind,'checkpoint_kind':job['kind']})
            for kind in ('sft','grpo','rules_skills','speculative'):
                for track in self.tracks:
                    if kind not in self.kinds: continue
                    if kind in {'rules_skills','speculative'} and track == 'no_tools':
                        self.status.append(dict(profile=profile,kind=kind,track=track,status='unsupported',reason='This experiment requires the with_tools track'))
                        continue
                    modes = tuple(m for m in self.grpo_harness_modes if track == 'with_tools' or m == 'off') if kind == 'grpo' else (None,)
                    for mode in modes:
                        if not any(j['profile']==profile and j['kind']==kind and j.get('track')==track and (mode is None or j.get('harness_mode','off')==mode) for j in self.jobs):
                            row = dict(profile=profile,kind=kind,track=track,condition=kind+'_'+track,status='missing_checkpoint',reason='No exact uploaded adapter for selected track/stage/K/profile')
                            if mode is not None and (mode != 'off' or len(self.grpo_harness_modes)>1): row['harness_mode']=mode
                            self.status.append(row)
        if self.ablation:
            self.jobs = [j for j in self.jobs if j['kind'] in {'baseline','sft','grpo'} and j['condition'].endswith('with_tools')]
        order = {kind:i for i,kind in enumerate(KINDS)}
        self.jobs.sort(key=lambda j:(order[j['kind']],self.profiles.index(j['profile']),j['condition'],j['stage'],j['k'] or 0))

    def list(self):
        # Inventory is cheap: no base/adapter/image downloads and no CUDA imports.
        result_files = self.api.list_repo_files(self.results_repo, repo_type=self.results_repo_type)
        remote = [name for name in result_files if name.startswith('evaluations/') and name.endswith('/bundle.json')]
        print('Checkpoint repository:', self.repo)
        print('Results repository:',self.results_repo,'| existing result bundles:',len(remote))
        managed = {name.rsplit('/', 1)[0] for name in remote}
        for name in sorted(result_files):
            if name.startswith('evaluations/') and name.rsplit('/', 1)[0] not in managed and name.endswith(('.jsonl','.zip')):
                print('[LEGACY UNVERIFIED]',name)
        for job in self.jobs: print(job['profile'],job['kind'],job['condition'],job['stage'],'K='+str(job['k']),job['adapter'] or '(base)')
        for row in self.status: print(row)
        return copy.deepcopy(self.jobs)

    def _input(self, split):
        if split not in self.inputs:
            self.inputs[split] = prepare_inputs(self.data,split,self.dev_split,self.api,self.download,self.token,files=self._local_files())
        return self.inputs[split]

    def _local_files(self):
        if self._inventory_files is None: self._inventory_files = inventory_roots(self.data)
        return self._inventory_files

    def _training_harness_metadata(self, job):
        mode = job.get('harness_mode','off')
        if mode == 'off' or job.get('checkpoint_kind',job['kind']) != 'grpo': return None
        from dental_agent.harness.configuration import assert_harness_identity
        name = job['adapter'].removesuffix('/grpo_policy')+'/checkpoint_meta.json'
        if name not in self.files: raise ValueError('Harness GRPO checkpoint has no training metadata: '+name)
        path = self.download(self.repo,name,repo_type='model',revision=self.revision,token=self.token)
        assert_harness_identity(json.loads(Path(path).read_text()).get('extra',{}).get('harness'),mode)
        return name

    def _identity(self, job):
        split = 'dev' if job['kind'] in {'rules_skills','speculative'} else self.split
        hashes = {} if not job['adapter'] else {name:_metadata_digest(self.files[name]) for name in (job['adapter']+'/adapter_config.json',job['weight'])}
        identity = dict(schema=1,profile=job['profile'],runtime=self.profile_configs[job['profile']],kind=job['kind'],condition=job['condition'],
            checkpoint_kind=job.get('checkpoint_kind',job['kind']),track=job.get('track','no_tools'),stage=job['stage'],k=job['k'],adapter=job['adapter'],checkpoint_files=hashes,
            base_model='Qwen/Qwen3.5-9B',software=self.software,split=split,dev_split_sha256=digest(self.dev_split) if split=='dev' else None,source_sha256=self.source,
            max_new_tokens=16384,max_tool_calls=50,max_turns=None,malformed_retries=0,seed=self.seed,size=job.get('size','canonical'))
        from dental_agent.harness.configuration import harness_identity
        mode = job.get('harness_mode','off')
        if job.get('checkpoint_kind',job['kind']) == 'grpo' or job['kind'] == 'speculative':
            identity['harness_mode'] = mode
            if mode != 'off': identity['harness'] = harness_identity(mode)
        meta_name = self._training_harness_metadata(job)
        if meta_name is not None: identity['checkpoint_files'][meta_name] = _metadata_digest(self.files[meta_name])
        if self.ablation: identity['ablation'] = self.ablation
        if job['kind']=='rules_skills': identity['conditions']=['baseline','rules_only','rules_skills_index','rules_skills_full']
        if job['kind']=='speculative':
            identity['conditions']=['baseline','mtp_1']
            identity['draft_variant']=self.draft_variant
            configured = self.spec_checkpoints.get(job['profile'],{})
            key = job['adapter']
            checkpoint = configured.get(key) if isinstance(configured,dict) else None
            # A merged target must be explicitly associated with its exact source adapter.
            identity['merged_files'] = self._merged_identity(checkpoint) if checkpoint else None
            identity['serving'] = dict(dtype='float16' if job['profile'].startswith('dual_t4') else 'bfloat16',max_model_len=self.serving_context_length,
                gpu_indices=self.gpu_indices,port=self.port,vllm_executable=self.vllm)
        # Reuse a fully identified prior cohort before downloading any images or model weights.
        saved = self._saved_identity(identity)
        if saved: return saved
        cohort = self._input(split)
        identity.update(image_ids=cohort['ids'],cohort_sha256=cohort['fingerprint'])
        return identity

    def _saved_identity(self, requested):
        if self._remote_manifests is None:
            self._remote_manifests = []
            info = self.api.repo_info(self.results_repo, repo_type=self.results_repo_type) if hasattr(self.api, 'repo_info') else self.api.model_info(self.results_repo)
            names = self.api.list_repo_files(self.results_repo, repo_type=self.results_repo_type, revision=info.sha)
            for name in sorted(names):
                if not name.startswith('evaluations/') or not name.endswith('/bundle.json'): continue
                path = self.download(self.results_repo,name,repo_type=self.results_repo_type,revision=info.sha,token=self.token)
                self._remote_manifests.append(json.loads(Path(path).read_text()))
        candidates = []
        for bundle in self._remote_manifests:
            old = bundle.get('identity') or {}
            compare = {k:v for k,v in old.items() if k not in {'image_ids','cohort_sha256'}}
            if compare != requested: continue
            if old.get('split')=='test' and len(set(old.get('image_ids',[])))!=50: continue
            if old.get('split')=='dev' and set(old.get('image_ids',[])) != set(json.loads(self.dev_split.read_text())['image_ids']): continue
            candidates.append(old)
        distinct = {fingerprint(v):v for v in candidates}
        if len(distinct)>1: raise ValueError('Multiple prior cohorts match settings; select an explicit input identity before reuse')
        return next(iter(distinct.values()),None)

    @staticmethod
    def _merged_identity(checkpoint):
        ck = Path(checkpoint).expanduser().resolve()
        from dental_agent.harness.checkpoints import tensor_headers
        headers = tensor_headers(ck)
        if not any(n.startswith('mtp.') for n in headers): raise ValueError('Merged target has no MTP tensors')
        # Cache hashes by immutable file size/mtime only within this coordinator instance's call.
        return merged_signature(ck)

    def _adapter(self,job):
        names = (job['adapter']+'/adapter_config.json',job['weight'])
        # Reuse a live local checkpoint only if both files match the exact pinned Hub contents.
        relative = Path(job['adapter']).relative_to(job['adapter'].split('/',1)[0])
        candidates = [self.data/folder/relative for folder in ('models','grpo','checkpoints')]
        for candidate in candidates:
            if all(matches_hub_file(candidate/Path(name).name,self.files[name]) for name in names):
                return str(candidate.resolve())
        # hf_hub_download reuses the pinned cache snapshot; no optimizers/base weights are requested.
        paths = [Path(self.download(self.repo,n,repo_type='model',revision=self.revision,token=self.token)) for n in names]
        return str(paths[0].parent)

    def _command(self,job,identity,folder):
        kind = job['kind']; cohort = self.inputs[identity['split']]
        adapter = self._adapter(job) if job['adapter'] else None
        if self.model_id is None:
            from dental_agent.utils.inputs import resolve_qwen_model
            self.model_id,_ = resolve_qwen_model()
        common = ['--model-id',str(self.model_id),'--data-dir',cohort['data_dir']]
        p = self.profile_configs[job['profile']]
        if kind in {'baseline','sft','grpo'}:
            command = [sys.executable,'-u','scripts/evaluate_models.py','--condition',job['condition'],*common,
                '--dataset','dentex','--split',identity['split'],'--dev-split',str(self.dev_split),'--output-dir',str(folder),
                '--existing-results','resume','--max-new-tokens','16384','--max-tool-calls','50','--malformed-retries','0',
                *runtime_flags(job['profile'], self.profile_configs), '--canonical-resize' if identity['size']=='canonical' else '--no-canonical-resize',
                '--canonical-letterbox' if p['letterbox'] else '--no-canonical-letterbox']
            if job.get('harness_mode','off') != 'off': command += ['--harness-mode',job['harness_mode']]
            if adapter: command += ['--adapter-path',adapter]
            suffix = '_canonical' if identity['size']=='canonical' else ''
            if self.ablation:
                from dental_agent.evaluation.tool_ablation import AblationSpec
                spec = AblationSpec(remove_tools=tuple(self.ablation.get('remove_tools',[])),locate_shift_frac=self.ablation.get('locate_shift_frac',0.0),locate_scale=self.ablation.get('locate_scale',1.0),seed=self.ablation.get('seed',0))
                if spec.remove_tools: command += ['--remove-tools',*spec.remove_tools]
                command += ['--locate-shift-frac',str(spec.locate_shift_frac),'--locate-scale',str(spec.locate_scale),'--ablation-seed',str(spec.seed)]
                suffix += spec.tag()
            result = folder/f"eval_dentex_{identity['split']}_{job['condition']}{suffix}.jsonl"
        elif kind == 'zeroshot':
            if identity['split'] != 'test': raise ValueError('Legacy zero-shot runner has no fixed-dev split; run this cohort on test')
            flags = runtime_flags(job['profile'], self.profile_configs); flags[0] = '--local-precision'
            command = [sys.executable,'-u','scripts/run_zero_shot.py','--provider','transformers','--model','Qwen/Qwen3.5-9B',
                '--dataset','dentex','--split','test','--data-dir',cohort['data_dir'],'--output',str(folder/'results.jsonl'),'--output-dir',str(folder),
                '--max-tokens','16384','--image-max-dim','0','--temperature','0','--pacing-delay','0','--git-sync-every','0','--no-retry-empty',*flags]
            result = folder/'results.jsonl'
        elif kind == 'rules_skills':
            command = [sys.executable,'-u','scripts/harness/run_dev.py',*common,'--adapter-path',adapter,
                '--checkpoint-kind',identity['checkpoint_kind'],'--dev-split',str(self.dev_split),'--split','dev',
                '--max-new-tokens','16384','--max-tool-calls','50','--seed',str(self.seed),'--conditions',*identity['conditions'],
                *runtime_flags(job['profile'], self.profile_configs),'--existing-results','resume','--output',str(folder/'results.json')]
            result = folder/'results.json'
        else:
            ck = self.spec_checkpoints[job['profile']][job['adapter']]
            indices = self.gpu_indices or '0'
            command = [sys.executable,'-u','scripts/bench/spec_decode_bench.py','--checkpoint',str(ck),'--data-dir',cohort['data_dir'],
                '--dev-split',str(self.dev_split),'--split','dev','--max-new-tokens','16384','--max-model-len',str(self.serving_context_length),
                '--max-tool-calls','50','--seed',str(self.seed),'--conditions',*identity['conditions'],'--vllm-executable',self.vllm,
                '--gpu-indices',indices,'--tensor-parallel',str(len(indices.split(','))),'--dtype',identity['serving']['dtype'],
                '--port',str(self.port),'--existing-results','resume','--output',str(folder/'results.json'),
                '--harness-mode',job.get('harness_mode','off'),'--draft-variant',self.draft_variant]
            result = folder/'results.json'
        return command,result,adapter

    def _upload_results(self, folder, prefix, identity, payloads, progress, complete, row, attempts=1):
        """Retry fresh Hub state; keep inference/local evidence alive during transport outages."""
        pending = folder/'upload_pending.json'
        for attempt in range(attempts):
            try:
                self.store.save(folder,prefix,identity,payloads,progress,complete)
            except Exception as error:
                if not transient_upload_error(error): raise
                row.update(upload_status='pending',pending_upload=str(pending))
                write(pending,dict(hub_path=prefix,progress=progress,complete=complete,
                    files=sorted(payloads),error_type=type(error).__name__))
                if attempt+1 < attempts:
                    time.sleep(min(2**attempt,2))
                    continue
                print('[UPLOAD DEFERRED]',prefix,type(error).__name__,'local results preserved; retry on the next sync or rerun',flush=True)
                return False
            else:
                row['upload_status']='synced'; row.pop('pending_upload',None)
                pending.unlink(missing_ok=True)
                return True

    def run(self, execute=True):
        devices = []
        if execute:
            import torch
            if torch.cuda.is_available():
                devices = [dict(name=torch.cuda.get_device_name(i),gib=torch.cuda.get_device_properties(i).total_memory/2**30,
                    capability=torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]
        for rawjob in self.jobs:
            for size in (self.sizes if rawjob['kind'] in {'baseline','sft','grpo'} else ('canonical',)):
                job = {**rawjob,'size':size}
                row = {**job,'status':'pending'}; self.status.append(row)
                if job['kind']=='zeroshot' and self.split=='dev': row.update(status='unsupported',reason='Zero-shot legacy prompt cohort is test-only'); continue
                identity = self._identity(job)
                prefix = result_path(identity)
                folder = self.output/Path(prefix).relative_to('evaluations')
                row.update(results_repo=self.results_repo, results_repo_type=self.results_repo_type)
                row.update(path=str(folder),hub_path=prefix,identity=identity)
                self.store.restore(folder,prefix,identity)
                marker = folder/'identity.json'
                if marker.exists() and json.loads(marker.read_text()) != identity: raise ValueError('Local evaluation identity conflict')
                write(marker,identity)
                portable = folder/('results.jsonl' if job['kind'] in {'zeroshot','baseline','sft','grpo'} else 'results.json')
                payloads,progress,complete = validate_document(job['kind'],portable,identity)
                if complete:
                    if (folder/'summary.json').is_file(): payloads['summary.json']=(folder/'summary.json').read_bytes()
                    if (folder/'runtime.json').is_file(): payloads['runtime.json']=(folder/'runtime.json').read_bytes()
                    self._upload_results(folder,prefix,identity,payloads,progress,True,row,attempts=3)
                    row.update(status=('failed_or_divergent_reused' if job['kind']=='speculative' and json.loads(portable.read_text()).get('status')=='failed_or_divergent' else 'complete_reused'),episodes=progress); self._save_inventory(); print('[SKIP FINISHED]',prefix); continue
                reason = hardware_reason(job['profile'],job['kind'],devices,self.profile_configs) if execute else None
                if job['kind']=='speculative' and not shutil.which(self.vllm): reason = 'pending: configure the compatible isolated VLLM_EXECUTABLE'
                if job['kind']=='speculative' and identity['merged_files'] is None: reason = 'pending: provide SPEC_CHECKPOINTS[profile][exact_adapter_prefix] for an audited merged target'
                if not execute or reason:
                    row.update(status='pending' if not execute else 'blocked_runtime',reason=reason or 'inventory only',episodes=progress); self._save_inventory(); continue
                cohort = self._input(identity['split'])
                if cohort['fingerprint'] != identity['cohort_sha256']: raise ValueError('Input contents changed since partial evaluation; old results preserved')
                if job['kind']=='speculative' and progress and (folder/'runtime.json').is_file():
                    previous=json.loads((folder/'runtime.json').read_text())
                    if previous.get('devices') != devices:
                        row.update(status='blocked_runtime',reason='Partial serving timings were recorded on different physical GPUs; preserved without mixing')
                        self._save_inventory(); continue
                command,result,adapter = self._command(job,identity,folder)
                if portable.exists() and result != portable and not result.exists(): shutil.copy2(portable,result)
                if result.suffix=='.jsonl' and result.exists() and job['kind']!='zeroshot':
                    rows = jsonl_rows(result)
                    for r in rows:
                        r['eval_settings']['adapter_path'] = adapter or ''
                        r['eval_settings']['model_id'] = str(self.model_id)
                    temp = result.with_suffix('.tmp'); temp.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8'); temp.replace(result)
                write(folder/'command.json',command)
                env = os.environ.copy(); env['PYTHONHASHSEED']=str(self.seed); env['PYTHONUNBUFFERED']='1'
                if job['kind']=='zeroshot':
                    env['QWEN35_9B_LOCAL_DIR']=str(self.model_id)
                row['status']='running'; self._save_inventory()
                write(folder/'runtime.json',dict(devices=devices,python=sys.version,software=self.software,command=command,checkpoint_revision=self.revision))
                child = subprocess.Popen(command,cwd=ROOT,env=env)
                last = time.monotonic()
                def sync(final=False):
                    payloads,n,done = validate_document(job['kind'],result,identity)
                    if n:
                        payloads['runtime.json']=(folder/'runtime.json').read_bytes()
                        if done:
                            if job['kind'] in {'zeroshot','baseline','sft','grpo'}:
                                from dental_agent.evaluation.matrix import summarize_cell
                                summary = summarize_cell(jsonl_rows(result),seed=self.seed)
                            else:
                                document = json.loads(result.read_text())
                                summary = dict(status=document.get('status'),conditions={c:e.get('summary') for c,e in document.get('conditions',{}).items()},gates=document.get('gates',{}),comparisons=document.get('comparisons',{}))
                            payloads['summary.json'] = (json.dumps(summary,sort_keys=True,allow_nan=False)+'\n').encode()
                        for name,data in payloads.items(): (folder/name).write_bytes(data) if folder/name != result else None
                        row['episodes']=n
                        self._upload_results(folder,prefix,identity,payloads,n,done,row,attempts=3 if final else 1)
                    return n,done
                try:
                    while child.poll() is None:
                        try: child.wait(timeout=5)
                        except subprocess.TimeoutExpired: pass
                        if time.monotonic()-last>=60: sync(); last=time.monotonic()
                    n,done = sync(final=True)
                    if child.returncode or not done: raise RuntimeError(f'Evaluation unfinished or failed ({child.returncode}); saved {n} episodes at {folder}')
                    row.update(status='complete',episodes=n)
                except BaseException:
                    row['status']='failed_or_canceled'; raise
                finally:
                    if child.poll() is None:
                        child.terminate()
                        try: child.wait(timeout=15)
                        except subprocess.TimeoutExpired: child.kill(); child.wait()
                    try: sync(final=True)
                    except Exception as error: print('[SYNC FAILED] Local evidence preserved:',type(error).__name__,str(error))
                    self._save_inventory()
        self._save_inventory()
        return copy.deepcopy(self.status)

    def _save_inventory(self):
        write(self.output/'inventory.json',dict(repo=self.repo,results_repo=self.results_repo,results_repo_type=self.results_repo_type,checkpoint_revision=self.revision,jobs=self.status))
        print('[INVENTORY]',self.output/'inventory.json')


def validate_resume_settings(saved, settings, ids):
    """Extension runners may move filesystem paths, but never decoding settings or cohort."""
    old = saved.get('settings') or {}
    portable_paths = {'output','adapter_path','model_id','data_dir','dev_split','checkpoint','vllm_executable','existing_results'}
    for key,value in settings.items():
        if key not in portable_paths and old.get(key) != value:
            raise ValueError('Resume settings changed: '+key)
    if list(saved.get('image_ids',[])) != list(ids): raise ValueError('Resume cohort/order changed')
    for entry in (saved.get('conditions') or {}).values():
        rows = entry.get('records',entry.get('episodes',[]))
        seen = [int(r['image_id']) for r in rows]
        if len(seen)!=len(set(seen)) or not set(seen).issubset(ids): raise ValueError('Saved episode IDs conflict')


def prepare_serving_targets(session, output_root='data/models/speculative', merge=False):
    """Explicit opt-in CPU merges into NEW folders; source adapters/base remain untouched."""
    from dental_agent.utils.inputs import resolve_qwen_model
    mapping = copy.deepcopy(session.spec_checkpoints)
    for job in session.jobs:
        if job['kind'] != 'speculative': continue
        session._training_harness_metadata(job)
        profile = job['profile']; prefix = job['adapter']
        if prefix in mapping.get(profile,{}): continue
        key = fingerprint({name:_metadata_digest(session.files[name]) for name in (prefix+'/adapter_config.json',job['weight'])})[:16]
        target = Path(output_root)/profile/key
        if merge:
            if session.model_id is None: session.model_id,_ = resolve_qwen_model()
            adapter = session._adapter(job)
            if not target.exists():
                subprocess.run([sys.executable,'scripts/bench/merge_adapter.py','--base',str(session.model_id),'--adapter',adapter,'--output',str(target)],cwd=ROOT,check=True)
            subprocess.run([sys.executable,'scripts/bench/check_mtp.py','--base',str(session.model_id),'--checkpoint',str(target),
                '--output',str(target.parent/(key+'_mtp_audit.json'))],cwd=ROOT,check=True)
        if target.is_dir():
            from dental_agent.harness.configuration import write_target_manifest
            write_target_manifest(target,prefix,job.get('checkpoint_kind',job['kind']),job.get('harness_mode','off'))
            mapping.setdefault(profile,{})[prefix] = str(target.resolve())
    session.spec_checkpoints = mapping
    return mapping


def summarize_inventory(status):
    """Read saved evidence; primary scoring stays in the existing shared matrix module."""
    output = []
    for item in status:
        row = {k:item.get(k) for k in ('profile','kind','track','condition','stage','k','size','status','reason','hub_path','upload_status','pending_upload','harness_mode')}
        folder = Path(item['path']) if item.get('path') else None
        if folder and (folder/'results.json').is_file():
            doc = json.loads((folder/'results.json').read_text())
            row['result_status'] = doc.get('status')
            row['conditions'] = {name:entry.get('summary') for name,entry in doc.get('conditions',{}).items()}
            row['gates'] = doc.get('gates',{})
            row['comparisons'] = doc.get('comparisons',{})
        elif folder and (folder/'results.jsonl').is_file():
            from dental_agent.evaluation.matrix import summarize_cell
            row['metrics'] = summarize_cell(jsonl_rows(folder/'results.jsonl'),seed=42)
        output.append(row)
    return output


def episode_map(name, data):
    if name.endswith('.jsonl'):
        return {('primary',int(row['image_id'])):row for row in [json.loads(line) for line in data.splitlines() if line.strip()]}
    doc = json.loads(data)
    return {(condition,int(row['image_id'])):row for condition,entry in doc.get('conditions',{}).items()
            for row in entry.get('records',entry.get('episodes',[]))}


def merge_evidence(name, local, remote):
    a,b = episode_map(name,local),episode_map(name,remote)
    if any(a[key]!=b[key] for key in a.keys() & b.keys()):
        raise ValueError('Local and Hub episodes conflict; neither is overwritten')
    if b.keys() <= a.keys(): return local
    if name.endswith('.jsonl'):
        return ''.join(json.dumps(row,sort_keys=True)+'\n' for row in (a|b).values()).encode()
    # Combine disjoint progress, then let the existing runner recompute summaries and gates.
    doc = json.loads(local); other = json.loads(remote)
    if a.keys() <= b.keys(): return remote
    for condition,entry in other.get('conditions',{}).items():
        if condition not in doc['conditions']: doc['conditions'][condition]=entry;continue
        field = 'records' if 'records' in entry else 'episodes'
        seen = {int(r['image_id']) for r in doc['conditions'][condition].get(field,[])}
        doc['conditions'][condition][field] += [r for r in entry[field] if int(r['image_id']) not in seen]
        if field=='episodes':doc['conditions'][condition]['status']='running'
    doc['status']='running'
    return (json.dumps(doc,sort_keys=True)+'\n').encode()


def publish_paired_summaries(status, output_dir='data/evaluations/summaries'):
    """JSON/Markdown statistics only; no figures, TeX assets or PDF compilation."""
    from dental_agent.evaluation.matrix import build_summary,summary_markdown
    groups = {}
    baselines = {}
    for item in status:
        if item.get('status') not in {'complete','complete_reused'} or item.get('kind') not in {'baseline','sft','grpo'}: continue
        identity = item['identity']; file = Path(item['path'])/'results.jsonl'
        if not file.exists(): continue
        block = (item['profile'],identity['split'],identity['cohort_sha256'],fingerprint(identity.get('ablation')))
        cell = item['condition']+'@'+item['size']
        rows = jsonl_rows(file)
        if item['kind']=='baseline': baselines.setdefault(block,{})[cell]=rows
        else: groups.setdefault((block,item['stage'],item.get('k'),identity.get('harness_mode','off')),{})[cell]=rows
    # Pair each SFT control with each matching GRPO K; no cross-profile or cross-cohort comparisons.
    for (block,stage,k,mode),cells in list(groups.items()):
        if k is not None:
            cells.update(groups.get((block,stage,None,'off'),{}))
        cells.update(baselines.get(block,{}))
        if mode != 'off':
            for cell,records in groups.get((block,stage,k,'off'),{}).items():
                if cell.startswith('grpo_'):
                    condition,size = cell.split('@')
                    cells[condition+'_harness_off@'+size]=records
    groups.update({(block,'base',None,'off'):cells for block,cells in baselines.items()})
    results=[]
    from huggingface_hub import HfApi
    api=HfApi(token=os.environ.get('HF_TOKEN'))
    destinations = {(item.get('results_repo', os.environ.get('HF_EVAL_REPO','Reza-Nadimi/vlm-dental-evaluations')), item.get('results_repo_type')) for item in status if item.get('status') in {'complete','complete_reused'}}
    if len(destinations) > 1: raise ValueError('Summary inputs use different result repositories')
    repo, configured_type = next(iter(destinations), (os.environ.get('HF_EVAL_REPO','Reza-Nadimi/vlm-dental-evaluations'), None))
    repo_type = result_repo_type(api, repo, configured_type)
    for key,cells in groups.items():
        report=build_summary(cells,n_boot=2000,seed=42)
        from dental_agent.evaluation.matrix import paired_bootstrap_diff
        for control in sorted(c for c in cells if '_harness_off@' in c):
            enabled = control.replace('_harness_off@','@')
            if enabled in cells:
                for level in ('exact','fdi'):
                    comparison = paired_bootstrap_diff(cells[enabled],cells[control],level=level,metric='f1',n_boot=2000,seed=42)
                    report['comparisons'].append(dict(label='GRPO harness on minus off',a=enabled,b=control,**comparison))
        digest_key=fingerprint(dict(group=key,records=cells))
        block, stage, k, mode = key
        relative = Path('dentex')/block[1]/block[0]/stage/('harness_'+mode)/('k'+str(k) if k is not None else 'controls')/('run_'+digest_key[:16])
        folder=Path(output_dir)/relative;folder.mkdir(parents=True,exist_ok=True)
        write(folder/'summary.json',report)
        (folder/'summary.md').write_text(summary_markdown(report),encoding='utf-8')
        from huggingface_hub import CommitOperationAdd
        prefix='evaluations/summaries/'+relative.as_posix()
        api.create_commit(repo,repo_type=repo_type,operations=[CommitOperationAdd(path_in_repo=prefix+'/'+name,path_or_fileobj=(folder/name).read_bytes()) for name in ('summary.json','summary.md')],commit_message='eval: paired cohort summaries')
        results.append(dict(profile=key[0][0],stage=key[1],k=key[2],harness_mode=key[3],hub_path=prefix,local_path=str(folder)))
    return results


def adapter_signature(path):
    root=Path(path)
    if (root/'grpo_policy').is_dir(): root=root/'grpo_policy'
    weights=next((root/name for name in ('adapter_model.safetensors','adapter_model.bin') if (root/name).is_file()),None)
    if weights is None or not (root/'adapter_config.json').is_file(): raise FileNotFoundError('Missing trained policy adapter')
    return {p.name:digest(p) for p in [root/'adapter_config.json',weights]}


def merged_signature(path):
    return {p.name:digest(p) for p in sorted(Path(path).glob('*')) if p.is_file() and p.suffix in {'.json','.safetensors'}}


def software_versions():
    import importlib.metadata
    result={}
    for name in ('torch','transformers','accelerate','peft','bitsandbytes','qwen-vl-utils'):
        try:result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result[name]=None
    return result


def matches_hub_file(path, metadata):
    path=Path(path)
    if not path.is_file(): return False
    lfs=getattr(metadata,'lfs',None)
    sha=lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)
    if sha:return digest(path)==sha
    blob=getattr(metadata,'blob_id',None)
    if not blob:return False
    content=path.read_bytes()
    return hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()==blob
