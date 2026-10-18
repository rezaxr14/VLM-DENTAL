"""scripts/setup_gpu_server.sh decisions (dental_agent/utils/gpu_setup.py): no GPU, no network, no torch needed."""

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from dental_agent.utils import gpu_setup as gs

ADA = (8, 9)
BLACKWELL = (12, 0)
HOPPER = (9, 0)
AMPERE = (8, 0)


# ---- nvidia-smi parsing and GPU generation (compute capability, never the card name) ----------------------------------------
def test_parse_gpu_csv_reads_name_vram_capability_driver():
    text = "name, memory.total [MiB], compute_cap, driver_version\nNVIDIA GeForce RTX 4090, 24564 MiB, 8.9, 570.172.08\n"
    (g,) = gs.parse_gpu_csv(text)
    assert (g.name, g.vram_mib, g.cc, g.driver) == ("NVIDIA GeForce RTX 4090", 24564, ADA, "570.172.08")
    assert round(g.vram_gib, 1) == 24.0


def test_parse_gpu_csv_unreadable_capability_stops_with_a_clear_message():
    with pytest.raises(gs.SetupError, match="driver is too old"):
        gs.parse_gpu_csv("name, memory.total [MiB], compute_cap, driver_version\nX, 100 MiB, [N/A], 470.1\n")


def test_parse_cuda_header():
    assert gs.parse_cuda_header("| NVIDIA-SMI 570.172.08  Driver Version: 570.172.08  CUDA Version: 12.8 |") == "12.8"
    assert gs.parse_cuda_header("nothing") is None


@pytest.mark.parametrize("cc,word", [((8, 9), "Ada"), ((8, 0), "Ampere"), ((9, 0), "Hopper"), ((12, 0), "Blackwell"),
                                     ((10, 0), "Blackwell"), ((7, 5), "Turing"), ((7, 0), "Volta")])
def test_generation_comes_from_compute_capability(cc, word):
    assert word in gs.gpu_generation(cc)


def test_the_48gb_card_is_told_apart_by_capability_not_by_name():
    """'48 GB Pro 6000' may be an RTX 6000 Ada (8.9) or a Blackwell workstation card (12.0): the plan differs, the name does not."""
    assert gs.attention_plan((8, 9)).try_flash_attn2 is True
    assert gs.attention_plan((12, 0)).try_flash_attn2 is False


# ---- torch wheel index -------------------------------------------------------------------------------------------------------
def test_4090_with_driver_570_gets_cu126_because_cu130_needs_580():
    plan = gs.select_torch_index("570.172.08", ADA, "2.13.0")
    assert plan.index == "cu126"
    assert plan.index_url == "https://download.pytorch.org/whl/cu126"


def test_new_driver_gets_the_highest_stable_build_and_never_the_experimental_one():
    assert gs.select_torch_index("580.95.05", ADA, "2.13.0").index == "cu130"
    assert gs.select_torch_index("999.0", ADA, "2.13.0").index == "cu130"  # cu132 is experimental: never automatic


def test_blackwell_needs_cuda_128_or_newer_and_a_driver_that_supports_it():
    assert gs.select_torch_index("580.95.05", BLACKWELL, "2.13.0").index == "cu130"
    with pytest.raises(gs.SetupError, match="CUDA >= 12.8"):
        gs.select_torch_index("570.172.08", BLACKWELL, "2.13.0")  # cu126 cannot run it, cu130 needs driver 580.65.06


def test_blackwell_on_torch_211_may_use_cu128_with_driver_570_117():
    assert gs.select_torch_index("570.117", BLACKWELL, "2.11.0").index == "cu128"
    with pytest.raises(gs.SetupError):
        gs.select_torch_index("570.100", BLACKWELL, "2.11.0")  # below 570.117


def test_driver_too_old_for_every_build_says_what_to_change():
    with pytest.raises(gs.SetupError, match="driver"):
        gs.select_torch_index("470.1", ADA, "2.13.0")


def test_unknown_torch_minor_is_not_guessed():
    with pytest.raises(gs.SetupError, match="--cuda-index"):
        gs.select_torch_index("580.95.05", ADA, "9.9.0")


def test_override_is_checked_against_driver_architecture_and_the_published_builds():
    assert gs.select_torch_index("580.95.05", ADA, "2.13.0", "cu126").index == "cu126"
    with pytest.raises(gs.SetupError, match="no 'cu128' build"):
        gs.select_torch_index("580.95.05", ADA, "2.13.0", "cu128")
    with pytest.raises(gs.SetupError, match="driver >="):
        gs.select_torch_index("570.172.08", ADA, "2.13.0", "cu130")
    with pytest.raises(gs.SetupError, match="12.8"):
        gs.select_torch_index("580.95.05", BLACKWELL, "2.13.0", "cu126")
    plan = gs.select_torch_index("999.0", ADA, "2.13.0", "cu132")
    assert any("experimental" in w for w in plan.warnings)


def test_parse_torch_pins_reads_the_commented_lines_and_ignores_the_rest():
    text = "numpy\n# torch==2.13.0\n#torchvision == 0.28.0  # note\ntorchaudio\n"
    assert gs.parse_torch_pins(text) == {"torch": "2.13.0", "torchvision": "0.28.0"}


def test_the_repo_requirements_still_name_torch_and_torchvision():
    root = Path(__file__).resolve().parents[1]
    pins = gs.parse_torch_pins((root / "requirements.txt").read_text(encoding="utf-8"))
    assert set(pins) == {"torch", "torchvision"}


def test_commented_hints_of_requirements_gpu():
    text = '# pip install causal-conv1d==1.7.0 --no-build-isolation\n# pip install --no-deps "https://x.example/flash_attn-2.8.3+cu13torch2.13-cp312-linux_x86_64.whl"\n'
    assert gs.commented_pin(text, "causal-conv1d") == "1.7.0"
    assert gs.commented_wheel_url(text).endswith(".whl")
    assert gs.commented_wheel_url("# nothing\n") is None


# ---- attention / kernels -----------------------------------------------------------------------------------------------------
def test_flash_attention_2_covers_ampere_ada_hopper_and_fa3_is_never_installed():
    for cc in (AMPERE, ADA, HOPPER):
        plan = gs.attention_plan(cc)
        assert plan.try_flash_attn2 is True
        assert any("FlashAttention-3" in n for n in plan.notes)


def test_blackwell_and_old_gpus_use_sdpa_with_a_printed_notice():
    for cc in (BLACKWELL, (10, 0), (7, 5)):
        plan = gs.attention_plan(cc)
        assert plan.try_flash_attn2 is False
        assert plan.notes


def test_attention_outcome_mirrors_train_sft():
    assert gs.attention_outcome(8, True) == "flash_attention_2"
    assert gs.attention_outcome(8, False) == "sdpa"
    assert gs.attention_outcome(7, True) == "sdpa"
    assert gs.attention_outcome(12, False) == "sdpa"


def test_arch_supported_cubin_and_ptx_rules():
    assert gs.arch_supported(["sm_80", "sm_86", "sm_90"], ADA)  # sm_80 cubin runs on 8.9
    assert not gs.arch_supported(["sm_80", "sm_90"], BLACKWELL)
    assert gs.arch_supported(["sm_90", "compute_90"], BLACKWELL)  # PTX is JIT-compiled forward
    assert gs.arch_supported(["sm_120"], BLACKWELL)
    assert not gs.arch_supported(["sm_90"], ADA)


TAGS = gs.EnvTags("2.13.0+cu130", 13, "cp312", "TRUE")


def test_flash_attn_wheel_must_match_cuda_torch_python_and_abi():
    names = [
        "flash_attn-2.8.3+cu13torch2.13cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
        "flash_attn-2.8.2+cu13torch2.13cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
        "flash_attn-2.9.0+cu13torch2.12cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",  # wrong torch
        "flash_attn-2.9.0+cu12torch2.13cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",  # wrong CUDA
        "flash_attn-2.9.0+cu13torch2.13cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",  # wrong ABI
        "flash_attn-2.9.0+cu13torch2.13cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",  # wrong Python
        "notes.txt",
    ]
    assert gs.pick_flash_attn_asset(names, TAGS) == names[0]
    assert gs.pick_flash_attn_asset(names[2:], TAGS) is None


def test_causal_conv1d_wheel_url_is_built_like_its_setup_py():
    url, fname = gs.causal_conv1d_wheel_url("1.7.0", TAGS)
    assert fname == "causal_conv1d-1.7.0+cu13torch2.13cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
    assert url == f"https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/{fname}"
    with pytest.raises(gs.SetupError):
        gs.causal_conv1d_wheel_url("1.7.0", gs.EnvTags("2.13.0", 10, "cp312", "TRUE"))


# ---- .env ------------------------------------------------------------------------------------------------------------------
def test_placeholder_rule_matches_hub_token_resolver():
    for v in (None, "", "  ", "your_huggingface_token_here", "YOUR_TOKEN", "'your_x'", "/path/to/file"):
        assert gs.is_placeholder(v)
    for v in ("hf_abc123", "x"):
        assert not gs.is_placeholder(v)


def test_read_env_value_handles_export_quotes_comments_and_last_wins():
    text = "# HF_TOKEN=commented\nexport HF_TOKEN='first'\nHF_TOKEN=\"second\"  # trailing\nOTHER=1\n"
    assert gs.read_env_value(text, "HF_TOKEN") == "second"
    assert gs.read_env_value(text, "MISSING") is None
    assert gs.read_env_value("A=b # c\n", "A") == "b"


def test_set_env_value_changes_only_that_line():
    text = "# top\nHF_TOKEN=your_huggingface_token_here\nWANDB_API_KEY=keep # mine\n"
    new = gs.set_env_value(text, "HF_TOKEN", "hf_real")
    assert new == "# top\nHF_TOKEN=hf_real\nWANDB_API_KEY=keep # mine\n"
    assert gs.set_env_value("A=1", "B", "2") == "A=1\nB=2\n"
    assert gs.set_env_value("", "B", "x y") == 'B="x y"\n'


def test_env_status_never_exposes_the_value():
    assert gs.env_status("HF_TOKEN=your_x\n", "HF_TOKEN") == "placeholder"
    assert gs.env_status("HF_TOKEN=hf_secret\n", "HF_TOKEN") == "set"
    assert gs.env_status("", "HF_TOKEN") == "missing"


def _repo(tmp_path, template="HF_TOKEN=your_huggingface_token_here\nWANDB_API_KEY=your_wandb_key\n"):
    (tmp_path / ".env.example").write_text(template, encoding="utf-8")
    return tmp_path


def test_ensure_env_creates_from_template_with_mode_600(tmp_path):
    repo = _repo(tmp_path)
    res = gs.ensure_env_file(repo, environ={}, git_state=lambda r: (True, False))
    assert res.created and res.mode == "0o600"
    assert stat.S_IMODE((repo / ".env").stat().st_mode) == 0o600
    assert (repo / ".env").read_text(encoding="utf-8") == (repo / ".env.example").read_text(encoding="utf-8")
    assert res.statuses["HF_TOKEN"] == "placeholder" and res.hf_token_source == "none"
    assert res.git_ignored is True and res.git_tracked is False


def test_ensure_env_takes_the_token_from_the_environment_without_printing_it(tmp_path, capsys):
    repo = _repo(tmp_path)
    res = gs.ensure_env_file(repo, environ={"HF_TOKEN": "hf_SECRETVALUE"}, git_state=lambda r: (True, False))
    assert res.filled_hf_token and res.statuses["HF_TOKEN"] == "set" and res.hf_token_source == ".env"
    assert "hf_SECRETVALUE" not in repr(res) and "hf_SECRETVALUE" not in capsys.readouterr().out
    assert "HF_TOKEN=hf_SECRETVALUE" in (repo / ".env").read_text(encoding="utf-8")


def test_ensure_env_never_rewrites_an_existing_file_and_only_tightens_permissions(tmp_path):
    repo = _repo(tmp_path)
    env = repo / ".env"
    env.write_text("HF_TOKEN=hf_mine\n# my note\n", encoding="utf-8")
    env.chmod(0o644)
    res = gs.ensure_env_file(repo, environ={"HF_TOKEN": "hf_other"}, git_state=lambda r: (True, False))
    assert not res.created and res.chmod_fixed
    assert env.read_text(encoding="utf-8") == "HF_TOKEN=hf_mine\n# my note\n"
    assert stat.S_IMODE(env.stat().st_mode) == 0o600


def test_ensure_env_flags_an_env_that_git_would_commit(tmp_path):
    res = gs.ensure_env_file(_repo(tmp_path), environ={}, git_state=lambda r: (False, True))
    assert res.git_ignored is False and res.git_tracked is True


def test_ensure_env_without_a_template_writes_the_minimal_file(tmp_path):
    res = gs.ensure_env_file(tmp_path, environ={}, git_state=lambda r: (None, None))
    assert res.created and (tmp_path / ".env").read_text(encoding="utf-8") == gs.MINIMAL_ENV


def test_the_repo_env_example_and_gitignore_are_consistent_with_the_token_slots():
    root = Path(__file__).resolve().parents[1]
    example = (root / ".env.example").read_text(encoding="utf-8")
    assert gs.read_env_value(example, "HF_TOKEN") is not None
    assert any(ln.strip() in (".env", "/.env") for ln in (root / ".gitignore").read_text(encoding="utf-8").splitlines())


def test_check_hf_token_never_returns_the_token(monkeypatch):
    import huggingface_hub

    class Boom(Exception):
        response = type("R", (), {"status_code": 401})()

    class Api:
        def whoami(self, token=None):
            raise Boom("bad hf_SECRETVALUE")

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    status, detail = gs.check_hf_token("hf_SECRETVALUE")
    assert status == "rejected" and "hf_SECRETVALUE" not in detail

    class Ok:
        def whoami(self, token=None):
            return {"name": "reza"}

    monkeypatch.setattr(huggingface_hub, "HfApi", Ok)
    assert gs.check_hf_token("hf_x") == ("ok", "reza")


# ---- preflight arguments: discovery only, no training setting, no flag twice ------------------------------------------------
def test_preflight_argv_has_discovery_flags_only():
    argv = gs.build_preflight_argv("dentex_alone", "with_tools", "/m/Qwen3.5-9B", "user/ckpt")
    assert argv[0] == "scripts/train_sft.py" and argv[-1] == "--preflight"
    for forbidden in ("--precision", "--max-seq-len", "--lora-r", "--learning-rate", "--epochs", "--spmd", "--batch-size"):
        assert forbidden not in argv
    assert argv[argv.index("--model-id") + 1] == "/m/Qwen3.5-9B"
    assert argv[argv.index("--hf-repo") + 1] == "user/ckpt"


def test_preflight_argv_never_passes_a_flag_twice_and_the_user_wins():
    argv = gs.build_preflight_argv("s", "t", None, None, ["--track", "no_tools", "--no-canonical-resize", "--model-id=/x"])
    assert argv.count("--track") == 1 and argv[argv.index("--track") + 1] == "no_tools"
    assert "--canonical-resize" not in argv and "--no-canonical-resize" in argv
    assert not any(a == "--model-id" for a in argv)
    flags = [a for a in argv if a.startswith("--")]
    assert len(flags) == len(set(flags))


def test_preflight_flags_exist_in_train_sft():
    """Every flag the preflight passes is defined by train_sft.py (or by add_letterbox_arg, which it calls); a --no- twin counts as its base."""
    root = Path(__file__).resolve().parents[1]
    src = (root / "scripts" / "train_sft.py").read_text(encoding="utf-8") + (root / "dental_agent" / "utils" / "canonical.py").read_text(encoding="utf-8")
    for _names, argv in gs._PREFLIGHT_DEFAULTS:
        for a in (x for x in argv if x.startswith("--")):
            base = "--" + a[len("--no-"):] if a.startswith("--no-") else a
            assert f'"{base}"' in src, a
    assert '"--preflight"' in src


# ---- disk ------------------------------------------------------------------------------------------------------------------
def test_disk_blank_machine_needs_the_sum_of_the_allowances():
    a = gs.DISK_ALLOWANCE_GB
    v = gs.disk_plan(100, 100, True, {})
    assert v.ok and v.need_repo_gb == a["python_env"] + a["model"] + a["datasets"] + a["headroom"] + a["wheelhouse"] + 1.0


def test_disk_refuses_with_numbers_when_too_small():
    v = gs.disk_plan(20, 20, True, {})
    assert not v.ok and "20 GB free" in v.messages[0]


def test_disk_rerun_counts_only_what_is_missing():
    full = gs.disk_plan(40, 40, True, {})
    done = gs.disk_plan(40, 40, True, {"model": 19.0, "python_env": 10.0, "wheelhouse": 6.0, "datasets": 10.0})
    assert not full.ok and done.ok  # 40 GB is not enough for a blank machine but is for a finished one


def test_disk_separate_filesystems_are_checked_separately():
    v = gs.disk_plan(100, 0.5, False, {})
    assert not v.ok and any("at ~" in m for m in v.messages)
    assert gs.disk_plan(5, 100, False, {}, min_free_gb=3).ok  # --min-free-gb replaces the repo figure


# ---- constraints, state, summary -------------------------------------------------------------------------------------------
def test_constraints_pin_exactly_torch_and_torchvision(tmp_path):
    txt = gs.constraints_text("2.13.0+cu126", "0.28.0+cu126")
    assert [ln for ln in txt.splitlines() if not ln.startswith("#")] == ["torch==2.13.0+cu126", "torchvision==0.28.0+cu126"]
    conf = gs.pip_conf_text(tmp_path / "c.txt")
    assert "[install]" in conf and "[download]" in conf and str(tmp_path / "c.txt") in conf


def test_state_roundtrip_and_summary(tmp_path):
    st = tmp_path / "s.json"
    gs.record(st, "machine", "RTX 4090")
    gs.record(st, "missing", ["causal-conv1d kernel"])
    gs.record(st, "next_step", "open the notebook")
    out = gs.format_summary(gs.load_state(st))
    assert "RTX 4090" in out and "causal-conv1d kernel" in out and "NEXT STEP" in out and "open the notebook" in out
    assert "Missing / not done: nothing" in gs.format_summary({})
    assert gs.load_state(tmp_path / "absent.json") == {}


# ---- report with a fake nvidia-smi -----------------------------------------------------------------------------------------
def test_report_without_a_gpu_stops_with_status_no_gpu(tmp_path):
    def runner(argv):
        return (127, "")

    info, lines = gs.collect_report(tmp_path, tmp_path / ".venv", tmp_path / "wh", None, runner=runner)
    assert info["status"] == "no_gpu"
    assert any("nvidia-smi" in ln for ln in lines)


def test_report_with_a_4090_prints_capability_and_a_hint_that_changes_nothing(tmp_path):
    def runner(argv):
        if "--query-compute-apps=pid,used_memory,name" in argv:
            return (0, "")
        if any(a.startswith("--query-gpu") for a in argv):
            return (0, "name, memory.total [MiB], compute_cap, driver_version\nNVIDIA GeForce RTX 4090, 24564 MiB, 8.9, 570.172.08\n")
        return (0, "CUDA Version: 12.8")

    info, lines = gs.collect_report(tmp_path, tmp_path / ".venv", tmp_path / "wh", 1.0, runner=runner)
    text = "\n".join(lines)
    assert info["status"] == "ok" and tuple(info["cc"]) == ADA
    assert "compute capability 8.9" in text and "Ada" in text and "570.172.08" in text
    assert "nothing is changed" in info["precision_hint"] and "no memory peak has been measured" in info["precision_hint"]


# ---- command line: the shell script talks to the module through this --------------------------------------------------------
def test_cli_plan_and_summary_roundtrip(tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"report_cc": [12, 0], "report_driver": "580.95.05"}), encoding="utf-8")
    exe = [sys.executable, str(Path(gs.__file__)), ]
    out = subprocess.run([*exe, "plan", "--state", str(state), "--torch-version", "2.13.0"], capture_output=True, text=True, check=True)
    assert "cu130" in out.stdout
    got = subprocess.run([*exe, "get", "--state", str(state), "torch_index"], capture_output=True, text=True, check=True).stdout.strip()
    assert got == "cu130"
    fa = subprocess.run([*exe, "get", "--state", str(state), "plan_flash_attn2"], capture_output=True, text=True, check=True).stdout.strip()
    assert fa == "false"


def test_cli_setup_error_exits_2_with_a_clear_message(tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"report_cc": [12, 0], "report_driver": "570.172.08"}), encoding="utf-8")
    r = subprocess.run([sys.executable, str(Path(gs.__file__)), "plan", "--state", str(state), "--torch-version", "2.13.0"],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 2 and "[SETUP ERROR]" in r.stderr


def test_the_module_imports_without_torch_or_network():
    code = "import sys; sys.modules['torch']=None; import dental_agent.utils.gpu_setup as m; print('ok')"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]),
                       env={**os.environ, "HF_HUB_OFFLINE": "1"}, check=False)
    assert r.returncode == 0 and "ok" in r.stdout


# ---- the shell script itself -----------------------------------------------------------------------------------------------
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup_gpu_server.sh"


def test_script_is_valid_bash_and_never_commits_deletes_or_touches_training_settings():
    assert subprocess.run(["bash", "-n", str(SCRIPT)], check=False).returncode == 0
    code = "\n".join(ln for ln in SCRIPT.read_text(encoding="utf-8").splitlines() if not ln.lstrip().startswith("#"))
    for forbidden in ("git commit", "git push", "rm -rf", "rm -r ", "pkill", "kill -9", "tune2fs -m", "--precision", "--max-seq-len"):
        assert forbidden not in code, forbidden


def test_script_help_runs_without_a_gpu():
    r = subprocess.run(["bash", str(SCRIPT), "--help"], capture_output=True, text=True, check=False)
    assert r.returncode == 0 and "--report-only" in r.stdout


def test_script_stops_cleanly_on_a_machine_without_nvidia_smi(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "VLM_STATE_DIR": str(tmp_path / "st")}
    r = subprocess.run(["bash", str(SCRIPT), "--report-only"], capture_output=True, text=True, env=env, check=False)
    if r.returncode == 0:  # a real GPU machine running the tests
        pytest.skip("this machine has nvidia-smi")
    assert "no usable NVIDIA GPU" in r.stderr
