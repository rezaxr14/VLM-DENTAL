import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
# Leaf-loader tests do not need a model backend or the package's eager training imports.
def leaf(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/file);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
skills=leaf('dental_agent.harness._test_skills_leaf','dental_agent/harness/skills.py')

def test_library_index_and_sources():
    lib=skills.SkillLibrary();assert len(lib.all())==5
    assert 'dx_frac' not in lib.index()
    for skill in lib.all():assert skill.sources and skill.version and skill.sha256
    with pytest.raises(ValueError):lib.get('../rules')

def test_progressive_disclosure_logging():
    session=skills.HarnessSession(skills.HarnessOptions(skills=True,skills_mode='index_only'))
    prompt=session.augment('original')
    assert prompt.startswith('original') and 'read_skill' in prompt and 'dx_frac and dy_frac' not in prompt
    session.turn=3
    output=session.read({'name':'box-verification'})
    assert 'dx_frac' in output['body'] and session.events[-1]['turn']==3
    assert session.events[-1]['name']=='box-verification'
    assert skills.HarnessSession(skills.HarnessOptions()).augment('original')=='original'
    with pytest.raises(ValueError):session.read({'name':'../oops'})
    with pytest.raises(ValueError):skills.HarnessOptions(skills_mode='index_only')

def test_full_and_episode_isolation():
    a=skills.HarnessSession(skills.HarnessOptions(skills=True,skills_mode='full_inject'))
    assert 'dx_frac' in a.augment('prompt') and len(a.loaded)==5
    b=skills.HarnessSession(skills.HarnessOptions());assert not b.events and not b.loaded
    with pytest.raises(ValueError):a.read({'name':'box-verification'})

def test_shared_dispatcher_no_registry_addition():
    dispatch=leaf('test_dispatch_leaf','dental_agent/agent/tool_dispatch.py')
    registry=SimpleNamespace(execute=lambda *a,**kw:(a,kw))
    assert dispatch.execute_tool_call(registry,'fdi_label',{'quadrant':1},None)==(('fdi_label',),{'quadrant':1})
    s=skills.HarnessSession(skills.HarnessOptions(skills=True,skills_mode='index_only'))
    assert dispatch.execute_tool_call(registry,'read_skill',{'name':'box-verification'},None,harness=s)['skill']=='box-verification'
    assert 'read_skill' not in dispatch.IMAGE_CONSUMING_TOOLS

def test_symlink_escape(tmp_path):
    external=tmp_path/'elsewhere';external.mkdir();(external/'SKILL.md').write_text('not a skill')
    root=tmp_path/'library';root.mkdir()
    try:(root/'bad').symlink_to(external,target_is_directory=True)
    except OSError:pytest.skip('symlink unavailable')
    with pytest.raises(ValueError):skills.SkillLibrary(root)
