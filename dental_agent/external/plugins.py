"""Plugins: modules that register conditions, tool packs and evaluators when imported.

The same module names go into ``ShardSpec.plugins`` so a hosted worker (which clones this repository) imports them before it runs,
and so does the driver before it scores. A plugin is ordinary repository code; there is no discovery magic.
"""
from __future__ import annotations

import importlib


def load_plugins(names) -> list[str]:
    loaded = []
    for name in names or []:
        importlib.import_module(name)
        loaded.append(name)
    return loaded
