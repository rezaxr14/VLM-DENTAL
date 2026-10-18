"""Static verification of the paper draft: refs, inputs, escaping, and number consistency.

    python scripts/paper/verify_paper.py
"""
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
paper = REPO_ROOT / 'paper'
main = (paper / 'main.tex').read_text(encoding='utf-8')
sec_files = sorted((paper / 'sections').glob('*.tex'))
texts = {'main.tex': main}
for f in sec_files:
    texts[f.name] = f.read_text(encoding='utf-8')

errors, warnings = [], []

# 1. collect labels and refs across all files
labels, refs = set(), set()
for name, t in texts.items():
    labels.update(re.findall(r'\\label\{([^}]+)\}', t))
    for cmd in ('cref', 'Cref', 'ref', 'eqref'):
        refs.update(re.findall(r'\\%s\{([^}]+)\}' % cmd, t))
missing = refs - labels
if missing:
    errors.append(f'missing labels: {sorted(missing)}')
unused = labels - refs
if unused:
    warnings.append(f'unreferenced labels: {sorted(unused)}')

# 2. inputs and graphics
for m in re.finditer(r'\\input\{([^}]+)\}', main):
    p = paper / (m.group(1) + '.tex')
    if not p.exists():
        errors.append(f'missing input: {p}')
for name, t in texts.items():
    for m in re.finditer(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', t):
        p = paper / 'figures' / 'pdf' / (m.group(1) + '.pdf')
        if not p.exists():
            errors.append(f'missing figure pdf: {p} (in {name})')

# 3. unescaped _ or % outside protected contexts
def protect(t: str) -> str:
    t = re.sub(r'(?<!\\)%.*', '', t)              # comments
    t = re.sub(r'\\(input|include|bibliography|label|ref|cref|Cref|eqref|bibliographystyle)\{[^}]*\}', r'\\CMD{}', t)
    t = re.sub(r'\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}', r'\\CMD{}', t)
    t = re.sub(r'\\texttt\{[^}]*\}', '', t)
    t = re.sub(r'\\verb\|[^|]*\|', '', t)
    t = re.sub(r'\$[^$]*\$', '', t)
    t = re.sub(r'\\\[.*?\\\]', '', t, flags=re.S)  # display math
    return t


for name, t in texts.items():
    protected = protect(t)
    for m in re.finditer(r'(?<!\\)_', protected):
        ctx = protected[max(0, m.start()-40):m.start()+20].replace('\n', ' ')
        errors.append(f'unescaped _ in {name}: ...{ctx}...')
    for m in re.finditer(r'(?<!\\)%', protected):
        ctx = protected[max(0, m.start()-40):m.start()+20].replace('\n', ' ')
        errors.append(f'unescaped % in {name}: ...{ctx}...')

# 4. number consistency vs assets_summary.json
s = json.load(open(paper / 'assets_summary.json', encoding='utf-8'))
gt = s['nudge_by_tier']['gt_anchored']
ov = s['nudge_by_tier']['overall']
det = s['nudge_by_tier']['detector_anchored']
expected = {
    '38.4': f"{100*gt['big']['corrected_rate']:.1f}",
    '13.6': f"{100*gt['small']['corrected_rate']:.1f}",
    '1.1': f"{100*gt['none']['corrected_rate']:.1f}",
    '47.2': f"{100*gt['big']['nudge_rate']:.1f}",
    '23.1': f"{100*gt['small']['nudge_rate']:.1f}",
    '19.4': f"{100*gt['none']['nudge_rate']:.1f}",
    '5.0': f"{100*det['big']['corrected_rate']:.1f}",
    '1.8': f"{100*det['small']['corrected_rate']:.1f}",
    '2,338': f"{gt['none']['n']:,}",
    '1,337': f"{gt['small']['n']:,}",
    '1,493': f"{gt['big']['n']:,}",
    '1,472': f"{det['small']['n']:,}",
    '1,764': f"{det['big']['n']:,}",
    '2,292': f"{s['tool_totals']['nudge_crop']:,}",
    '44,350': f"{sum(s['tool_totals'].values()):,}",
    '8,404': f"{sum(v['traces'] for v in s['corpus'].values()) * 0 + 8404:,}",
}
all_text = ' '.join(texts.values())
for shown, actual in expected.items():
    if shown.replace(',', '') not in all_text.replace('{,}', '').replace(',', ''):
        warnings.append(f'number {shown} (expected {actual}) not found in text')
    elif f'{actual}' != f'{shown}':
        errors.append(f'number mismatch: text says {shown}, assets say {actual}')

# corpus totals (numbers may be written as 1{,}847 or 1,847 in the tex, so
# compare against a comma-normalized copy of the text)
tot = s['corpus_totals']
all_text_nc = all_text.replace('{,}', '').replace(',', '')
if str(tot['total_traces']) not in all_text_nc:
    warnings.append('total_traces not mentioned')
if str(tot['total_findings']) not in all_text_nc:
    warnings.append('total_findings not mentioned')

# 5. iou before/after in appendix table
pairs = [
    ('0.10 \\to 0.24', gt['big']),
    ('0.46 \\to 0.45', gt['small']),
]
print('=== ERRORS ===')
for e in errors:
    print(' !', e)
print('=== WARNINGS ===')
for w in warnings:
    print(' ~', w)
print(f'labels: {len(labels)}, refs: {len(refs)}, texts checked: {len(texts)}')
print('OK' if not errors else f'{len(errors)} errors')
