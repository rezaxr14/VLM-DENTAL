"""A private Kaggle dataset holding only a ``.env`` serves every notebook and every hosted run (no network, no real tokens)."""
from __future__ import annotations

from dental_agent import config


def _dataset(root, rel, body, name=".env"):
    folder = root.joinpath(*rel.split("/"))
    folder.mkdir(parents=True)
    (folder / name).write_text(body)
    return folder


def test_discovers_both_kaggle_layouts_and_reports_the_dataset_id(tmp_path):
    _dataset(tmp_path, "datasets/rezanadimikj/vlm-dental-env", "HF_TOKEN=a\n")          # interactive: owner/name
    _dataset(tmp_path, "hosted-env", "X=1\n", name="env")                                # hosted: name only, plain file name
    _dataset(tmp_path, "images-only", "")                                                # a dataset without a .env is ignored
    (tmp_path / "images-only" / ".env").unlink()
    found = {ds: p.name for ds, p in config.discover_env_datasets((str(tmp_path),))}
    assert found == {"rezanadimikj/vlm-dental-env": ".env", "hosted-env": "env"}


def test_values_fill_only_unset_keys_and_never_the_runners_own(tmp_path):
    _dataset(tmp_path, "datasets/o/n",
             "# c\nexport HF_TOKEN=\"hf_x\"\nHF_TRACES_REPO='Reza-Nadimi/vlm-dental-traces'\nGITHUB_TOKEN=gh # note\n"
             "MODEL_PROXY_API_KEY=evil\nKAGGLE_KEY=evil\nLLM_X=evil\nEMPTY=\nALREADY=new\n")
    env = {"MODEL_PROXY_API_KEY": "real", "ALREADY": "mine"}
    loaded = config.load_env_from_datasets((str(tmp_path),), env)
    assert loaded == {"o/n": ["GITHUB_TOKEN", "HF_TOKEN", "HF_TRACES_REPO"]}          # names only
    assert env["HF_TOKEN"] == "hf_x" and env["GITHUB_TOKEN"] == "gh"
    assert env["ALREADY"] == "mine" and env["MODEL_PROXY_API_KEY"] == "real" and "KAGGLE_KEY" not in env and "LLM_X" not in env


def test_nothing_happens_outside_kaggle(tmp_path):
    assert config.discover_env_datasets((str(tmp_path / "missing"),)) == []
    assert config.load_env_from_datasets((str(tmp_path / "missing"),), {}) == {}


def test_load_env_falls_back_to_the_dataset_and_an_explicit_path_skips_it(tmp_path, monkeypatch):
    _dataset(tmp_path, "datasets/o/n", "HF_EVAL_REPO=Reza-Nadimi/vlm-dental-evaluations\n")
    monkeypatch.setattr(config, "KAGGLE_INPUT", str(tmp_path))
    for key in ("HF_EVAL_REPO", "SOMETHING_ELSE"):                 # setenv first so monkeypatch restores them on teardown
        monkeypatch.setenv(key, "x")
        monkeypatch.delenv(key)
    explicit = tmp_path / "elsewhere.env"
    explicit.write_text("SOMETHING_ELSE=1\n")
    config.load_env(explicit)
    import os
    assert "HF_EVAL_REPO" not in os.environ                       # an explicit path means exactly that file
    monkeypatch.chdir(tmp_path)                                    # no repo .env up the tree from here
    assert config.load_env() is True
    assert os.environ["HF_EVAL_REPO"] == "Reza-Nadimi/vlm-dental-evaluations"
