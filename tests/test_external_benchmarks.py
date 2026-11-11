"""MMOral-OPG-Bench adapter: official prompts and scoring, preparation, judge, resume safety, findings context, leakage, tables."""
import base64
import importlib
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dental_agent.benchmarks import judge as jg, leakage as lk, mmoral_opg as mm, runner as rn, tables as tb  # noqa: E402

# SHA-256 of the official 8-example judge head, verified byte-for-byte against build_mmoral_opg_gpt4_prompt of MMOral-Bench-EvalKit.
OFFICIAL_JUDGE_HEAD_SHA256 = "e5ba63e72f0e69149d8ed62e1b25fd0a04114298701daa953484a850ec1b60c4"


def b64_image(seed, size=(64, 32), fmt="JPEG"):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def write_tsvs(folder, closed_rows, open_rows):
    import pandas as pd
    pd.DataFrame(closed_rows).to_csv(folder / mm.CLOSED_FILE, sep="\t", index=False)
    pd.DataFrame(open_rows).to_csv(folder / mm.OPEN_FILE, sep="\t", index=False)


def synthetic_tsvs(folder):
    img0, img1 = b64_image(0), b64_image(1)
    closed = [dict(index=0, image_id=7, file_name="a.jpg", question="How many?", category="Teeth,Patho", image=img0, type="MCQ",
                   option1="10", option2="20", option3="30", option4="None", answer="B"),
              dict(index=1, image_id=8, file_name="b.jpg", question="Which tooth?", category="Jaw", image=img1, type="MCQ",
                   option1="#11", option2="#12", option3="#13", option4="#14", answer="D"),
              dict(index=2, image_id=7, file_name="a.jpg", question="Status?", category="HisT", image="0", type="MCQ",    # image by reference
                   option1="x", option2="y", option3="z", option4="w", answer="A")]
    opened = [dict(index=0, image_name="a.jpg", image=img0, question="Describe teeth", answer="30 teeth", category="Teeth"),
              dict(index=1, image_name="b.jpg", image=img1, question="Summary?", answer="No lesions", category="SumRec,Patho")]
    write_tsvs(folder, closed, opened)


# ------------------------------------------------------------------------------------------------------------- official prompts
def test_official_prompt_texts_are_exact():
    assert mm.open_prompt("Q?") == "Question: Q?\nPlease provide a detailed and accurate answer to the question."
    assert mm.closed_prompt("Q?", ["a", "b", "c", "d"]) == (
        "Question: Q?\nOptions:\nA. a\nB. b\nC. c\nD. d\n"
        "Please answer the above multiple-choice question by selecting the single correct option (A, B, C, or D). "
        "If the provided information is insufficient to determine a clear answer, please choose the most likely "
        "correct option based on the available data and your judgment.")


def test_judge_prompt_structure_and_markers():
    p = mm.judge_prompt("Q?", "gt <AND> two <OR> three", "pred")
    assert p.startswith("\nGiven the question, compare the ground truth and prediction from AI models")
    assert p.endswith("\nQ? | gt  <AND>  two  <OR>  three | pred | ")
    assert p.count("\n") > 15 and "Teeth position" in p and "\n{" in p      # real newlines inside the last example, as in the official file


def test_judge_head_hash_is_pinned():
    import hashlib
    assert hashlib.sha256(mm._JUDGE_HEAD.encode()).hexdigest() == mm.JUDGE_PROMPT_SHA256
    assert mm.JUDGE_PROMPT_SHA256 == OFFICIAL_JUDGE_HEAD_SHA256


# ----------------------------------------------------------------------------------------------------------- closed extraction
IDX = {"A": "one", "B": "text two", "C": "three", "D": "None"}


@pytest.mark.parametrize("text,expected", [("The answer is (B).", "B"), ("A", "A"), ("I think C is right", "C"), ("text two", "B"),
                                           ("(A) and (C)", "A"), ("  b ", None)])
def test_extract_choice_official_cases(text, expected):
    got = mm.extract_choice(text, mm.LETTERS, IDX, strict=True)
    assert got == expected


def test_unreadable_is_random_officially_and_none_strictly():
    assert mm.extract_choice("no idea", mm.LETTERS, IDX, strict=True) is None
    picks = {mm.extract_choice("no idea", mm.LETTERS, IDX, random.Random(i)) for i in range(40)}
    assert picks <= set(mm.LETTERS) and len(picks) > 1
    a = mm.extract_choice("no idea", mm.LETTERS, IDX, random.Random("0:5"))
    assert a == mm.extract_choice("no idea", mm.LETTERS, IDX, random.Random("0:5"))      # seeded per question: reproducible


@pytest.mark.parametrize("text", ["D, because", "I think C. is right"])
def test_official_quirk_letter_before_punctuation_is_unreadable(text):
    # ' D,' / ' C.' count as candidates, but the official position lookup only knows ' X ' and '(X)', so the official extraction falls
    # back to a random option. The port reproduces this; the strict score counts such answers as unreadable.
    assert mm.extract_choice(text, mm.LETTERS, IDX, strict=True) is None


def row(i, cat, ans, pred, **opts):
    return dict(index=i, category=cat, answer=ans, prediction=pred, option1="a", option2="b", option3="c", option4="d", image_id=i % 2, **opts)


def test_score_closed_categories_and_unreadable():
    rows = [row(0, "Teeth,Patho", "A", "A"), row(1, "Teeth", "B", "(B)"), row(2, "Jaw", "C", "zzz"), row(3, "HisT,Jaw", "D", "(D)")]
    res = mm.score_closed(rows, seed=0)
    clean = {r["category"]: r for r in res["strict"]["main"]}
    assert clean["Teeth"]["n"] == 2 and clean["Teeth"]["acc"] == 100.0           # multi-label rows count in every matching main category
    assert clean["Patho"]["n"] == 1 and clean["Jaw"]["n"] == 2 and clean["Jaw"]["acc"] == 50.0
    assert clean["Overall"]["acc"] == 75.0 and res["unreadable"] == 1 and res["unreadable_fraction"] == 0.25
    # the benchmark scripts' key weighting: a single-label question adds 3 times to its main category (closed-ended script)
    off = {r["category"]: r for r in res["strict"]["main_official_weighting"]}
    assert off["Jaw"]["n"] == 4 and off["Jaw"]["acc"] == 25.0 and off["Overall"]["n"] == 4 and off["Teeth"]["n"] == 4
    assert {r["category"] for r in res["strict"]["detailed"]} >= {"Teeth,Patho", "Teeth", "Jaw", "HisT,Jaw", "Overall"}
    assert res["official"]["main"][-1]["category"] == "Overall"


def test_strip_thinking():
    assert mm.strip_thinking("<think>A B C</think>(B)") == ("(B)", dict(thinking_stripped=True, unclosed_thinking=False))
    assert mm.strip_thinking("<think>never closes (A)") == ("", dict(thinking_stripped=True, unclosed_thinking=True))
    assert mm.strip_thinking("plain") == ("plain", dict(thinking_stripped=False, unclosed_thinking=False))


# ------------------------------------------------------------------------------------------------------------------------- judge
def test_judge_one_retry_rule():
    calls = []

    def complete(prompt, temperature):
        calls.append(temperature)
        return ["garbage", "1.7", " 0.6\n"][len(calls) - 1]

    res = mm.judge_one(complete, "Q", "gt", "pred")
    assert res["score"] == 0.6 and res["tries"] == 3 and calls == [0.0, 0.5, 1.0]
    res = mm.judge_one(lambda p, t: "nope", "Q", "gt", "pred")
    assert res["score"] == 0.0 and res["tries"] == 5 and "All 5 retries failed" in res["log"]


def test_score_open_clean_and_official_weighting():
    rows = [dict(category="Teeth", score=1.0), dict(category="Teeth,Report", score=0.5), dict(category="Jaw", score=0.0)]
    res = mm.score_open(rows)
    clean = {r["category"]: r for r in res["main"]}
    assert clean["Teeth"]["n"] == 2 and clean["Teeth"]["acc"] == 75.0 and clean["Report"]["acc"] == 50.0
    assert clean["Overall"]["acc"] == pytest.approx(50.0)
    off = {r["category"]: r for r in res["main_official_weighting"]}      # open-ended script: single-label rows count twice
    assert off["Teeth"]["n"] == 3 and off["Teeth"]["acc"] == pytest.approx(2.5 / 3 * 100) and off["Overall"]["acc"] == pytest.approx(50.0)


class FakeAPI:
    def __init__(self, replies):
        self.replies, self.kwargs, self.chat = list(replies), [], self
        self.completions = self

    def create(self, **kw):
        self.kwargs.append(kw)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": r})()})()]})()


class RateLimitError(Exception):
    status_code = 429


class BadRequestError(Exception):
    status_code = 400


def test_judge_client_retries_transient_and_raises_the_rest():
    sleeps = []
    api = FakeAPI([RateLimitError("slow down"), "0.5"])
    c = jg.JudgeClient("any-model", client=api, sleep=sleeps.append)
    assert c.complete("p", 0.0) == "0.5" and sleeps == [1.0]
    assert api.kwargs[0]["max_completion_tokens"] == 16384 and api.kwargs[0]["temperature"] == 0.0
    c2 = jg.JudgeClient("any-model", client=FakeAPI([BadRequestError("Unsupported value: temperature")]), sleep=sleeps.append)
    with pytest.raises(RuntimeError, match="JUDGE_SEND_TEMPERATURE"):
        c2.complete("p", 0.5)
    c3 = jg.JudgeClient("m", client=FakeAPI(["x"]), send_temperature=False)
    c3.complete("p", 0.0)
    assert "temperature" not in c3.client.kwargs[0]
    with pytest.raises(ValueError):
        jg.JudgeClient("", client=FakeAPI([]))


def test_judge_answers_resume_identity_and_transport_error(tmp_path):
    items = [dict(index=i, question="Q", ground_truth="g", prediction="p", category="Teeth") for i in range(4)]
    out = tmp_path / "j.jsonl"
    c = jg.JudgeClient("m1", client=FakeAPI(["0.5"] * 8))
    recs = jg.judge_answers(items[:2], out, c, workers=1)
    assert [r["score"] for r in recs] == [0.5, 0.5]
    recs = jg.judge_answers(items, out, jg.JudgeClient("m1", client=FakeAPI(["1.0"] * 8)), workers=1)       # resume: 2 old + 2 new
    assert sorted(r["score"] for r in recs) == [0.5, 0.5, 1.0, 1.0]
    with pytest.raises(SystemExit, match="different settings"):
        jg.judge_answers(items, out, jg.JudgeClient("other-judge", client=FakeAPI(["1.0"] * 8)), workers=1)
    with pytest.raises(SystemExit, match="already has"):
        jg.judge_answers(items, out, c, workers=1, existing="error")
    broken = jg.JudgeClient("m1", client=FakeAPI([BadRequestError("boom")]))
    with pytest.raises(BadRequestError):
        jg.judge_answers(items, tmp_path / "j2.jsonl", broken, workers=1)       # a transport/API error is never scored as 0.0
    assert jg.read_jsonl(tmp_path / "j2.jsonl") == []


# ------------------------------------------------------------------------------------------------------------------ preparation
def test_prepare_decodes_dedupes_and_validates(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    synthetic_tsvs(raw)
    with pytest.raises(ValueError, match="differs from the benchmark authors"):
        mm.prepare(raw, tmp_path / "prep")
    manifest = mm.prepare(raw, tmp_path / "prep", allow_md5_mismatch=True)
    assert manifest["n_closed"] == 3 and manifest["n_open"] == 2 and manifest["n_images"] == 2        # 2 distinct radiographs
    assert manifest["md5_matches_authors"] is False and manifest["closed_categories"]["Teeth,Patho"] == 1
    closed, opened, images = mm.load_prepared(tmp_path / "prep")
    assert list(images.columns) == ["id", "sha256", "local_path", "width", "height"] and sorted(images["id"]) == [0, 1]
    assert all(Path(p).is_file() for p in images["local_path"])
    # closed row 2 referenced row 0's image: same id; closed and open questions of one radiograph share it
    assert closed.loc[closed["index"] == 2, "image_id"].iloc[0] == closed.loc[closed["index"] == 0, "image_id"].iloc[0]
    assert opened.loc[opened["index"] == 0, "image_id"].iloc[0] == closed.loc[closed["index"] == 0, "image_id"].iloc[0]
    assert closed.loc[closed["index"] == 0, "option4"].iloc[0] == "None"
    again = mm.prepare(raw, tmp_path / "prep", allow_md5_mismatch=True)                                 # idempotent, same ids
    assert again["n_images"] == 2


def test_prepare_rejects_changed_schema(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    synthetic_tsvs(raw)
    import pandas as pd
    df = pd.read_csv(raw / mm.OPEN_FILE, sep="\t").drop(columns=["category"])
    df.to_csv(raw / mm.OPEN_FILE, sep="\t", index=False)
    with pytest.raises(ValueError, match="lacks column"):
        mm.prepare(raw, tmp_path / "prep", allow_md5_mismatch=True)


# ---------------------------------------------------------------------------------------------------------------- findings context
def test_findings_context_text_and_fdi_numbers():
    f = [dict(quadrant=3, tooth_position=6, diagnosis="Periapical Lesion", confidence=0.934),
         dict(quadrant=None, tooth_position=None, diagnosis="Caries", confidence=None)]
    text = mm.findings_context(f, True)
    assert "- #36: Periapical Lesion (confidence 0.93)" in text and "- an unlocated tooth: Caries" in text
    assert "does not assess any other finding" in text and text.endswith("\n\n")
    assert "none of the four conditions was found" in mm.findings_context([], True)
    assert "did not return a valid result" in mm.findings_context([], False)
    base = mm.open_prompt("Q?")
    assert mm.with_findings(base, f, True).endswith(base)             # the official prompt is untouched


# ------------------------------------------------------------------------------------------------------------------------ runner
def test_policy_phase_resume_and_settings(tmp_path):
    out = tmp_path / "p" / "findings.jsonl"
    imgs = [dict(id=i, local_path=f"/x/{i}.jpg") for i in range(3)]
    ran = []

    def run_one(iid, path):
        ran.append(iid)
        return dict(findings=[dict(quadrant=1, tooth_position=1, diagnosis="Caries", confidence=0.9)], answered=True, format_ok=True, turns=2, tool_calls=1)

    s = dict(a=1)
    rn.run_policy_phase(imgs, out, run_one, s, "resume", limit=2)
    rn.run_policy_phase(imgs, out, run_one, s, "resume")
    assert ran == [0, 1, 2]                                                # finished images are not run again
    with pytest.raises(SystemExit, match="different settings"):
        rn.run_policy_phase(imgs, out, run_one, dict(a=2), "resume")
    with pytest.raises(SystemExit, match="already has"):
        rn.run_policy_phase(imgs, out, run_one, s, "error")
    assert len(rn.load_findings(out, {0, 1, 2})) == 3
    with pytest.raises(SystemExit, match="no policy result"):
        rn.load_findings(out, {0, 1, 2, 9})


def test_answer_phase_plain_and_findings_condition(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    synthetic_tsvs(raw)
    mm.prepare(raw, tmp_path / "prep", allow_md5_mismatch=True)
    closed, opened, images = mm.load_prepared(tmp_path / "prep")
    qs = rn.build_questions(closed, opened, ("closed", "open"))
    assert len(qs) == 5 and qs[0]["prompt"].startswith("Question:")
    seen = []

    def answer(q, prompt):
        seen.append((q["kind"], prompt))
        return dict(raw="<think>x</think>(A)" if q["kind"] == "closed" else "some answer", new_tokens=5, hit_token_cap=False)

    res = rn.run_answer_phase(qs, tmp_path / "ans", answer, dict(system="s"), None, "error")
    assert res["closed"][0]["prediction"] == "(A)" and res["closed"][0]["thinking_stripped"] is True and "option1" in res["closed"][0]
    assert all(p.startswith("Question:") for _, p in seen)                        # plain condition: official prompt only
    seen.clear()
    findings = {int(i): dict(findings=[dict(quadrant=1, tooth_position=2, diagnosis="Caries", confidence=0.8)], answered=True) for i in images["id"]}
    rn.run_answer_phase(qs, tmp_path / "ans2", answer, dict(system="s2"), findings, "error")
    assert all(p.startswith("A dental radiograph analysis agent") and "#12: Caries" in p for _, p in seen)
    with pytest.raises(SystemExit, match="different settings"):
        rn.run_answer_phase(qs, tmp_path / "ans", answer, dict(system="other"), None, "resume")
    assert rn.safe_name("qwen3_5_9b_raw") and pytest.raises(ValueError, rn.safe_name, "a/b")


# ---------------------------------------------------------------------------------------------------------------------- leakage
def test_leakage_screen_finds_copies_but_not_other_images(tmp_path):
    def save(name, seed, size=(256, 128), fmt="PNG"):
        rng = np.random.default_rng(seed)
        base = rng.integers(0, 255, size=(32, 64), dtype=np.uint8)
        Image.fromarray(base).resize(size, Image.BICUBIC).convert("RGB").save(tmp_path / name, format=fmt)
        return str(tmp_path / name)

    ref = {"dentex::1": save("r1.png", 1), "dentex::2": save("r2.png", 2), "dentex::3": save("r3.png", 3)}
    bench = {"b_exact": save("b_exact.png", 1), "b_rescaled": save("b_rescaled.png", 2, size=(300, 150)), "b_other": save("b_other.png", 99)}
    rep = lk.screen(bench, ref, max_hamming=24, min_ncc=0.9)
    assert [e["bench"] for e in rep["exact"]] == ["b_exact"]
    assert {n["bench"] for n in rep["near"]} >= {"b_exact", "b_rescaled"} and "b_other" not in {n["bench"] for n in rep["near"]}
    assert rep["verdict"] == "OVERLAP FOUND" and rep["closest"][0]["hamming"] <= rep["closest"][-1]["hamming"]
    clean = lk.screen({"b_other": bench["b_other"]}, ref, max_hamming=24, min_ncc=0.9)
    assert clean["verdict"].startswith("no copy found") and clean["closest"][0]["ncc"] < 0.9


# ---------------------------------------------------------------------------------------------------------------------- tables
def test_cluster_bootstrap_and_paired_delta():
    a = {i: (i // 4, float(i % 2 == 0), "Teeth") for i in range(40)}           # 10 images x 4 questions, 50% correct
    b = {i: (i // 4, 0.0, "Teeth") for i in range(40)}
    ci = tb.cluster_mean_ci(a, None, n_boot=500, seed=1)
    assert ci["mean"] == 50.0 and ci["lo"] <= 50.0 <= ci["hi"] and ci["n"] == 40
    d = tb.paired_delta(a, b, None, n_boot=500, seed=1)
    assert d["mean"] == 50.0 and d["significant_95"] is True
    with pytest.raises(ValueError, match="different question sets"):
        tb.paired_delta(a, {k: v for k, v in b.items() if k < 10})
    assert tb.cluster_mean_ci(a, "Jaw")["n"] == 0


def test_render_markdown_keeps_published_rows_separate(tmp_path):
    measured = [dict(system="ours", judge="J", part="open", category=c, mean=61.0, lo=55.0, hi=66.0, n=10) for c in (*mm.OPEN_CATEGORIES, "Overall")]
    pub = [dict(name="Other", part="open", judge="gpt-x", source="their Table 2", Overall=59.1)]
    md = tb.render_markdown(measured, pub)
    assert "| ours | J |" in md and "*Other* (published; their Table 2) | gpt-x |" in md and "not comparable" in md
    p = tmp_path / "pub.json"
    p.write_text(json.dumps([dict(name="NoSource", part="open", Overall=1)]))
    with pytest.raises(ValueError, match="no 'source'"):
        tb.load_published(p)


# ------------------------------------------------------------------------------------------------------------------------ script
def test_script_parser_requires_explicit_budget_and_names(monkeypatch):
    try:
        import torch  # noqa: F401
    except ImportError:                      # the parser only needs the module to import; no tensor is touched
        import types
        monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    script = importlib.import_module("scripts.run_external_benchmark")
    p = script.build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["answer", "--system-name", "s"])                          # --max-new-tokens is mandatory
    with pytest.raises(SystemExit):
        p.parse_args(["answer", "--system-name", "s", "--max-new-tokens", "16384"])                 # --image-view is mandatory too
    a = p.parse_args(["answer", "--system-name", "s", "--max-new-tokens", "16384", "--image-view", "canonical", "--findings-from", "pol"])
    assert a.findings_from == "pol" and a.kinds == ["closed", "open"] and a.existing_results == "error" and a.image_view == "canonical"
    with pytest.raises(SystemExit):
        p.parse_args(["policy", "--policy-name", "p", "--adapter-path", "x", "--max-new-tokens", "16384"])      # harness mode is mandatory
    assert p.parse_args(["judge", "--system-name", "s", "--judge-model", "m"]).judge_send_temperature is True


# ------------------------------------------------------------------------------------------------------------------- policy source
def test_fetch_grpo_policy_exact_local_then_hub(tmp_path, monkeypatch):
    try:
        import dental_agent.training.experiment_profiles  # noqa: F401
    except ImportError:                      # no torch here: load the pure-Python naming module by path (it imports nothing heavy itself)
        import importlib.util
        spec = importlib.util.spec_from_file_location("dental_agent.training.experiment_profiles", REPO / "dental_agent/training/experiment_profiles.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setitem(sys.modules, "dental_agent.training.experiment_profiles", module)
    from dental_agent.benchmarks import policy_source as ps
    args = ("repo/x", "with_tools", 4, "dentex_alone", "bf16", "workstation48_lora_bf16", "rules_skills_index")
    loc = ps.grpo_policy_location(*args)
    assert loc["name"] == "qwen3_5_9b_grpo_with_tools_k4_dentex_alone_bf16__workstation48_lora_bf16--harness_rules_skills_index"
    assert loc["hub_prefix"] == f"grpo/{loc['name']}/grpo-with_tools-final"
    with pytest.raises(ValueError):
        ps.grpo_policy_location("repo/x", "no_tools", 4, "dentex_alone", "bf16", "t", "rules_skills_index")      # no_tools cannot use the harness

    def make(root):
        pol = Path(root) / "grpo-with_tools-final" / "grpo_policy"
        pol.mkdir(parents=True)
        (pol / "adapter_config.json").write_text("{}")
        (pol / "adapter_model.safetensors").write_bytes(b"w")
        (pol.parent / "checkpoint_meta.json").write_text(json.dumps({"step": 200, "extra": {"group_size": 4}}))

    make(tmp_path / "models" / loc["name"])
    got = ps.fetch_grpo_policy(*args, local_models=tmp_path / "models")
    assert got["source"] == "local" and got["path"].endswith("grpo_policy") and got["meta"]["step"] == 200

    hub_root = tmp_path / "hub"
    make(hub_root / "grpo" / loc["name"])
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: str(hub_root))
    got = ps.fetch_grpo_policy(*args, local_models=tmp_path / "empty")
    assert got["source"] == "hub:repo/x"
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: str(tmp_path / "nothing"))
    with pytest.raises(FileNotFoundError, match="Nothing else is substituted"):
        ps.fetch_grpo_policy(*args, local_models=tmp_path / "empty")
