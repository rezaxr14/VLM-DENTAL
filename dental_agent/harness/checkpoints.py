"""Safetensors MTP audit and preservation without loading base weights into RAM."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import struct

MAX_HEADER_BYTES=100*1024*1024

def tensor_headers(folder):
    folder=Path(folder).resolve();result={}
    files=sorted(folder.glob('*.safetensors'))
    if not files:raise ValueError('No local safetensors shards found')
    for file in files:
        if not file.resolve().is_relative_to(folder):raise ValueError('Shard symlink escapes checkpoint')
        with file.open('rb') as f:
            raw=f.read(8)
            if len(raw)!=8:raise ValueError('Truncated safetensors header')
            n=struct.unpack('<Q',raw)[0]
            if n<=0 or n>MAX_HEADER_BYTES:raise ValueError('Invalid safetensors header length')
            header=json.loads(f.read(n))
        size=file.stat().st_size
        for name,value in header.items():
            if name=='__metadata__':continue
            lo,hi=value['data_offsets']
            if not (0<=lo<=hi and 8+n+hi<=size):raise ValueError('Invalid tensor offsets')
            if name in result:raise ValueError(f'Duplicate tensor {name}')
            result[name]=dict(file=file,dtype=value['dtype'],shape=value['shape'],offset=8+n+lo,length=hi-lo)
    index=folder/'model.safetensors.index.json'
    if index.exists():
        mapping=json.loads(index.read_text())['weight_map']
        if set(mapping)!=set(result):raise ValueError('Index/shard tensor set differs')
        for name,shard in mapping.items():
            if result[name]['file'].name!=shard:raise ValueError('Index points to wrong shard')
    return result

def tensor_hash(info):
    h=hashlib.sha256();remaining=info['length']
    with info['file'].open('rb') as f:
        f.seek(info['offset'])
        while remaining:
            block=f.read(min(remaining,8*1024*1024))
            if not block:raise ValueError('Truncated tensor')
            h.update(block);remaining-=len(block)
    return h.hexdigest()

def audit_mtp(base,merged,check_bytes=True):
    a=tensor_headers(base);b=tensor_headers(merged)
    expected={k:v for k,v in a.items() if k.startswith('mtp.')}
    if not expected:raise ValueError('Reference checkpoint has no mtp.* tensors; cannot certify provenance')
    missing=sorted(set(expected)-set(b));extra=sorted(k for k in b if k.startswith('mtp.') and k not in expected)
    changed=[]
    for name,old in expected.items():
        if name not in b:continue
        new=b[name]
        if (old['dtype'],old['shape'])!=(new['dtype'],new['shape']) or (check_bytes and tensor_hash(old)!=tensor_hash(new)):
            changed.append(name)
    return dict(ok=not(missing or extra or changed),tensor_count=len(expected),missing=missing,unexpected=extra,changed=changed,
                byte_comparison=check_bytes,
                provenance={name:dict(shape=v['shape'],dtype=v['dtype'],sha256=tensor_hash(v) if check_bytes else None) for name,v in expected.items()})

def preserve_mtp(base,merged):
    """Copy original serialized MTP bytes into a dedicated shard after an HF/PEFT merge."""
    base=Path(base).resolve();merged=Path(merged).resolve()
    if base==merged:raise ValueError('Never modify the source checkpoint')
    a=tensor_headers(base);b=tensor_headers(merged)
    mtp={k:v for k,v in a.items() if k.startswith('mtp.')}
    if not mtp:raise ValueError('No source MTP tensors')
    if any(k.startswith('mtp.') for k in b):
        report=audit_mtp(base,merged)
        if not report['ok']:raise ValueError('Existing MTP tensors differ; will not overwrite')
        return report
    target=merged/'mtp-preserved.safetensors'
    if target.exists():raise FileExistsError(target)
    header={};offset=0
    for name,info in sorted(mtp.items()):
        header[name]=dict(dtype=info['dtype'],shape=info['shape'],data_offsets=[offset,offset+info['length']]);offset+=info['length']
    raw=json.dumps(header,separators=(',',':')).encode();raw+=b' '*((-len(raw))%8)
    with target.open('xb') as out:
        out.write(struct.pack('<Q',len(raw)));out.write(raw)
        for name,info in sorted(mtp.items()):
            with info['file'].open('rb') as f:
                f.seek(info['offset']);remaining=info['length']
                while remaining:
                    block=f.read(min(remaining,8*1024*1024))
                    if not block:raise ValueError('Truncated source tensor')
                    out.write(block);remaining-=len(block)
    mapping={k:v['file'].name for k,v in b.items()};mapping.update({k:target.name for k in mtp})
    index=merged/'model.safetensors.index.json'
    index.write_text(json.dumps(dict(metadata=dict(total_size=sum(v['length'] for v in b.values())+offset),weight_map=mapping),indent=2)+'\n')
    return audit_mtp(base,merged)
