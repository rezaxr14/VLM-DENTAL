"""Contract-only tests load modules by path to avoid initializing training dependencies."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj)
    return obj
rules=module('extracted_rules','dental_agent/harness/rules.py')
prompts=module('extracted_prompts','dental_agent/agent/prompts.py')

@pytest.mark.parametrize('dataset,classes',[('dentex',None),('tufts',None),('tufts_all',None),('dentex',['Custom {class}'])])
def test_prompt_golden(dataset,classes,monkeypatch):
    monkeypatch.setitem(sys.modules,'dental_agent.harness.rules',rules)
    original=module('golden_prompts','tests/fixtures/harness/prompts_before.py')
    tools='A tool with {braces} and @@classes_str@@'
    expected=original.build_agent_system_prompt(tools,dataset,classes)
    assert prompts.build_agent_system_prompt(tools,dataset,classes)==expected
    assert prompts.build_agent_system_prompt(tools,dataset,classes,harness=rules.HarnessConfig())==expected
    assert prompts.build_agent_system_prompt(tools,dataset,classes,harness=rules.HarnessConfig(True))==expected

def test_bindings():rules.validate_implementations()

def test_missing_or_extra_binding():
    data=rules.load_rules();data['implementations'].pop('R009')
    import json,tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'rules.md';path.write_text('---\n'+json.dumps(data)+'\n---\n')
        with pytest.raises(ValueError):rules.load_rules(path)

def test_changed_implementation(tmp_path):
    data=rules.load_rules();data['implementations']['R009']['sha256']='bad'
    with pytest.raises(ValueError,match='drift'):rules.validate_implementations(data)

def test_saved_synthetic_reward_regression():
    import json
    components=module('saved_components','dental_agent/rewards/components.py')
    cases=json.loads((ROOT/'tests/fixtures/harness/regression.json').read_text())
    for case in cases:
        t=case['trajectory']
        actual=[components.reward_format(t),components.reward_tool_validity(t),components.reward_efficiency(t),components.reward_accuracy(t,case['ground_truth'])]
        assert actual==case['rewards']

def test_saved_deterministic_verifier_gate():
    import ast,json
    tree=ast.parse((ROOT/'dental_agent/training/trace_generation.py').read_text())
    gate=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='grounded and is_healthy_ground_truth and (candidate_final_ans != [])')
    code=compile(ast.Module(body=[gate],type_ignores=[]),'existing_verifier_gate','exec')
    for healthy,answer,accepted in [(True,[],True),(True,[{}],False),(True,None,False),(False,[{}],True)]:
        state=dict(grounded=True,is_healthy_ground_truth=healthy,candidate_final_ans=answer,extracted_reason='accepted')
        exec(code,state)
        assert state['grounded']==accepted
