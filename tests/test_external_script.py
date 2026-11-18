"""The command line wires the new features (hosted group, live run flags) without spending anything."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_kaggle_external.py"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("run_kaggle_external", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("argv", [["hosted", "push", "--help"], ["run", "--help"], ["estimate", "--help"]])
def test_help_parses(cli, argv, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert ("--shard-size" in out) if argv[0] == "hosted" else True
    assert ("--workers" in out and "--no-quota-guard" in out) if argv[0] == "run" else True


def test_run_refuses_to_spend_without_confirmation(cli):
    with pytest.raises(SystemExit, match="refusing to spend"):
        cli.main(["run", "--model", "x/y", "--condition", "kaggle_zero_shot"])


def test_hosted_status_uses_the_campaign_builder(cli, monkeypatch, tmp_path, capsys):
    seen = {}

    class Camp:
        def refresh(self): seen["refresh"] = True
        def report(self, out): return {"runs": {}, "cells": {}, "paused": [], "tasks": 0}

    monkeypatch.setattr(cli, "_load_split", lambda a: (None, None, None, None))
    monkeypatch.setattr(cli.hosted_driver, "build_campaign", lambda root, images, **kw: seen.update(kw) or Camp())
    cli.main(["hosted", "status", "--root", str(tmp_path), "--conditions", "kaggle_zero_shot", "--shard-size", "10", "--hf-results-repo", "o/r"])
    assert seen["conditions"] == ["kaggle_zero_shot"] and seen["shard_size"] == 10 and seen["hf_results_repo"] == "o/r" and seen["refresh"]
    assert json.loads(capsys.readouterr().out)["tasks"] == 0
