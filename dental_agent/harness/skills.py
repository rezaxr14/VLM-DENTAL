"""Episode-local progressive disclosure; registry remains the eight diagnostic tools."""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Literal
from .rules import HarnessConfig

@dataclass(frozen=True)
class HarnessOptions(HarnessConfig):
    skills: bool = False
    skills_mode: Literal['off', 'index_only', 'full_inject'] = 'off'

    def __post_init__(self):
        if self.skills_mode not in {'off','index_only','full_inject'}:
            raise ValueError('Unknown skills_mode')
        if not self.skills and self.skills_mode != 'off':
            raise ValueError('Non-off skills_mode requires skills=True')

@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    version: str
    sources: tuple[str, ...]
    body: str
    sha256: str

class SkillLibrary:
    def __init__(self, root: str | Path | None = None):
        self.root = (Path(root) if root is not None else Path(__file__).with_name('skills')).resolve()
        self._skills = {}
        for path in sorted(self.root.glob('*/SKILL.md')):
            if not path.resolve().is_relative_to(self.root):
                raise ValueError('Skill symlink escapes library')
            text = path.read_text(encoding='utf-8')
            if not text.startswith('---\n'):
                raise ValueError('Missing skill front matter')
            _, front, body = text.split('---\n', 2)
            meta = json.loads(front)
            name = meta['name']
            if not re.fullmatch(r'[a-z][a-z0-9-]*',name) or path.parent.name != name:
                raise ValueError('Invalid skill name')
            if not meta.get('sources') or not all(isinstance(x,str) and x for x in meta['sources']):
                raise ValueError('Skill requires explicit sources')
            if not meta.get('version') or not meta.get('description') or not body.strip():
                raise ValueError('Incomplete skill')
            self._skills[name] = Skill(name,meta['description'],meta['version'],tuple(meta['sources']),body.strip(),hashlib.sha256(text.encode()).hexdigest())
    def all(self):
        return tuple(self._skills[n] for n in sorted(self._skills))
    def get(self,name):
        # No path construction from model-provided names.
        if not isinstance(name,str) or name not in self._skills:
            raise ValueError('Unknown skill')
        return self._skills[name]
    def index(self):
        return '\n'.join(f'- {s.name}: {s.description}' for s in self.all())

@dataclass
class HarnessSession:
    options: HarnessOptions
    library: SkillLibrary = field(default_factory=SkillLibrary)
    events: list = field(default_factory=list)
    started: float = field(default_factory=time.monotonic)
    turn: int = 0
    loaded: set = field(default_factory=set)

    @property
    def active(self):
        return self.options.skills and self.options.skills_mode != 'off'
    def event(self,kind,**values):
        self.events.append(dict(event=kind,turn=self.turn,elapsed_seconds=time.monotonic()-self.started,**values))
    def augment(self,prompt):
        if not self.active:
            return prompt
        self.event('listed',skills=[dict(name=s.name,version=s.version,sha256=s.sha256) for s in self.library.all()])
        text = prompt + '\n\nOPTIONAL PROCEDURAL SKILLS (not diagnostic tools):\n' + self.library.index()
        if self.options.skills_mode == 'index_only':
            return text + '\nLoad one with {"thought":"...", "tool_calls":[{"tool":"read_skill","args":{"name":"<skill-name>"}}]}. This harness action does not execute a diagnostic tool.\n'
        for s in self.library.all():
            self.loaded.add(s.name)
            self.event('loaded',name=s.name,version=s.version,sha256=s.sha256,mode='full_inject')
            text += '\n\nSKILL '+s.name+'\n'+s.body+'\nSources:\n'+'\n'.join(f'[S{i}] {source}' for i,source in enumerate(s.sources,1))
        return text
    def read(self,args):
        if not self.active or self.options.skills_mode != 'index_only':
            raise ValueError('On-demand skill loading is disabled')
        if not isinstance(args,dict) or set(args) != {'name'}:
            raise ValueError('read_skill requires only name')
        skill = self.library.get(args['name'])
        self.event('loaded',name=skill.name,version=skill.version,sha256=skill.sha256,mode='on_demand',repeat=skill.name in self.loaded)
        self.loaded.add(skill.name)
        return dict(skill=skill.name,version=skill.version,sources=list(skill.sources),body=skill.body)
