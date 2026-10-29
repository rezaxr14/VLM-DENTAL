"""Explicit harness experiment identity shared by training, evaluation and draft data."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

HARNESS_MODES = ('off', 'rules_skills_index', 'rules_skills_full')


def validate_harness_mode(mode, track='with_tools'):
    if mode not in HARNESS_MODES:
        raise ValueError('Unknown harness mode: '+str(mode))
    if mode != 'off' and track != 'with_tools':
        raise ValueError('Rules/skills require track=with_tools; no_tools is a separate control')
    return mode


def harness_options(mode):
    validate_harness_mode(mode)
    if mode == 'off': return None
    from dental_agent.harness.skills import HarnessOptions
    return HarnessOptions(rules_from_file=True, skills=True,
                          skills_mode='index_only' if mode == 'rules_skills_index' else 'full_inject')


def harness_identity(mode):
    validate_harness_mode(mode)
    if mode == 'off': return {'mode': 'off'}
    root = Path(__file__).resolve().parent
    files = [root/'rules.md', *sorted((root/'skills').glob('*/SKILL.md'))]
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return dict(mode=mode, files=hashes)


def assert_harness_identity(saved, mode):
    expected = harness_identity(mode)
    # Legacy checkpoints lacking this field are the original off control only.
    if (saved or {'mode':'off'}) != expected:
        raise ValueError('Harness configuration/content differs from the saved checkpoint or data')
    return expected


def write_target_manifest(checkpoint, source_adapter, checkpoint_kind, mode):
    """Record adapter provenance without modifying any model tensors."""
    checkpoint = Path(checkpoint)
    data = dict(source_adapter=source_adapter, checkpoint_kind=checkpoint_kind,
                training_harness=harness_identity(mode))
    path = checkpoint/'dental_target.json'
    if path.exists() and json.loads(path.read_text()) != data:
        raise ValueError('Merged target is already associated with a different adapter/harness')
    path.write_text(json.dumps(data, sort_keys=True, indent=2)+'\n', encoding='utf-8')
    return data


def require_harness_grpo_target(checkpoint, mode):
    """Draft training must use a final, harness-trained GRPO policy, never a reference."""
    validate_harness_mode(mode)
    if mode == 'off': raise ValueError('Select a rules/skills-enabled GRPO target for this MTP training workflow')
    path = Path(checkpoint)/'dental_target.json'
    if not path.is_file(): raise ValueError('Missing dental_target.json; prepare the exact merged GRPO target first')
    data = json.loads(path.read_text())
    source = data.get('source_adapter','')
    if data.get('checkpoint_kind') != 'grpo' or '/grpo-with_tools-final' not in source or 'reference' in source:
        raise ValueError('MTP training requires the final with_tools GRPO policy')
    assert_harness_identity(data.get('training_harness'), mode)
    return data
