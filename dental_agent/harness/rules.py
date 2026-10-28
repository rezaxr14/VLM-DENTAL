"""Validate an extracted contract without importing training or evaluating source code."""
from __future__ import annotations
import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class HarnessConfig:
    rules_from_file: bool = False


def load_rules(path: str | Path | None = None) -> dict:
    path = Path(path) if path is not None else Path(__file__).with_name('rules.md')
    text = path.read_text(encoding='utf-8')
    if not text.startswith('---\n'):
        raise ValueError('Missing YAML front matter')
    data = json.loads(text.split('---\n', 2)[1])
    if data.get('version') not in {'1.0.0', '1.1.0'}:
        raise ValueError('Unsupported contract version')
    ids = []
    bound = set()
    for rule in data['rules']:
        rid = rule['id']
        if not re.fullmatch(r'R\d{3}', rid) or not rule.get('statement'):
            raise ValueError('Invalid rule')
        if rule['category'] not in {'format','orientation','tool-use','verification','safety'}:
            raise ValueError('Invalid category')
        if rule['checkable'] not in {'prompt-only','verifier-check','reward-term','tool-validation'}:
            raise ValueError('Invalid enforcement')
        ids.append(rid)
        if rule['checkable'] in {'verifier-check','reward-term'}:
            bound.add(rid)
    if len(ids) != len(set(ids)) or bound != set(data['implementations']):
        raise ValueError('Duplicate or unbound rule/implementation ID')
    for rule in data['rules']:
        if rule['checkable'] == 'prompt-only' and rule['statement'] not in data['prompt_template']:
            raise ValueError('Prompt rule differs from rendered template')
    return data


def validate_implementations(data: dict | None = None, root: Path | None = None) -> None:
    data = load_rules() if data is None else data
    root = Path(__file__).resolve().parents[2] if root is None else Path(root)
    symbols = set()
    for rid, binding in data['implementations'].items():
        path = (root / binding['path']).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Implementation path escapes source root')
        tree = ast.parse(path.read_text(encoding='utf-8'))
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == binding['symbol']]
        if len(nodes) != 1 or hashlib.sha256(ast.dump(nodes[0], include_attributes=False).encode()).hexdigest() != binding['sha256']:
            raise ValueError(f'{rid}: implementation drift')
        symbols.add(binding['symbol'])
    reward_bindings={(b['path'],b['symbol']) for b in data['implementations'].values() if b['symbol'].startswith('reward_') or b['symbol'].endswith('_reward')}
    inventory=set()
    for relative in ['dental_agent/rewards/components.py','dental_agent/rewards/judge.py','dental_agent/rewards/composite.py','dental_agent/training/rewards.py']:
        tree=ast.parse((root/relative).read_text())
        inventory.update((relative,n.name) for n in tree.body if isinstance(n,ast.FunctionDef) and (n.name.startswith('reward_') or n.name.endswith('_reward')))
    if inventory!=reward_bindings:
        raise ValueError('Reward implementation missing a rule ID, or stale implementation binding')
    verifier_tree = ast.parse((root/'dental_agent/training/trace_generation.py').read_text())
    verifier_prompt = next(n for n in verifier_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id == 'VERIFIER_SYSTEM_PROMPT' for t in n.targets))
    if hashlib.sha256(ast.literal_eval(verifier_prompt.value).encode()).hexdigest() != data['verifier_prompt_sha256']:
        raise ValueError('Verifier instructions drift')
    # The independent verifier API is intentionally one bound function; remote LLM judgments are not deterministic checks.
    if 'verify_trace' not in symbols:
        raise ValueError('Verifier missing rule binding')


def render_prompt(tools_description: str, objective_classes: str, classes_str: str) -> str:
    data = load_rules()
    validate_implementations(data)
    values = {'tools_description': tools_description, 'objective_classes': objective_classes, 'classes_str': classes_str}
    # One pass: braces and marker-like text supplied by tools remain literal.
    return re.sub(r'@@(tools_description|objective_classes|classes_str)@@',
                  lambda m: values[m.group(1)], data['prompt_template'])
