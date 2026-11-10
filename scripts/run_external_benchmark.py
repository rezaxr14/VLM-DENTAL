#!/usr/bin/env python3
"""External benchmark runner (MMOral-OPG-Bench). Nothing here trains or tunes; every phase is resumable.

    prepare   download + verify + decode the benchmark                      (no GPU)
    leakage   screen the benchmark images against every DENTEX image        (no GPU)
    policy    the trained agent runs once per image (frozen protocol), findings cached
    answer    one model answers every question with the official prompt; --findings-from adds the cached findings (Option A)
    judge     LLM judge for the open-ended answers (needs the judge key)    (no GPU)
    score     tables with image-cluster bootstrap intervals                 (no GPU)
    compare   paired difference of two systems                              (no GPU)
    sync      copy the results folder to / from the results Hub repository

See docs/EXTERNAL_BENCHMARKS.md and notebooks/VLM_Dental_Colab_External_Benchmarks.ipynb.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dental_agent.benchmarks import judge as jg, leakage as lk, mmoral_opg as mm, runner as rn, tables as tb  # noqa: E402

DEFAULT_BENCH_DIR = "data/benchmarks/mmoral_opg"
DEFAULT_RESULTS_DIR = "data/evaluations/external/mmoral_opg"


# --------------------------------------------------------------------------------------------------------------------- helpers
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepared_dir(args) -> Path:
    return Path(args.bench_dir) / "prepared"


def load_all(args):
    closed, opened, images = mm.load_prepared(prepared_dir(args))
    manifest = json.loads((prepared_dir(args) / mm.MANIFEST).read_text(encoding="utf-8"))
    return closed, opened, images, manifest


def judge_file(results: Path, system: str, judge_model: str) -> Path:
    return results / "judge" / f"{rn.safe_name(system, 'system')}__{rn.safe_name(judge_model.replace('/', '-'), 'judge model')}.jsonl"


def add_common(p, *, results=True):
    p.add_argument("--bench-dir", default=DEFAULT_BENCH_DIR)
    if results:
        p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                       help="Use one folder per hardware profile tag, e.g. .../mmoral_opg/workstation48_lora_bf16")


def add_model_args(p):
    from dental_agent.model.local_runtime import add_runtime_args
    from dental_agent.utils.canonical import add_letterbox_arg
    p.add_argument("--model-id", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-9B"))
    p.add_argument("--model-revision", default="", help="Hub revision to pin for external models (recorded; empty = unpinned).")
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32", "qlora"])
    p.add_argument("--max-new-tokens", type=int, required=True)
    p.add_argument("--existing-results", default="error", choices=["error", "resume", "overwrite"])
    p.add_argument("--limit", type=int, default=None, help="First N items only (smoke test; the results are then not a full run).")
    add_runtime_args(p)
    add_letterbox_arg(p)


# ----------------------------------------------------------------------------------------------------------------------- prepare
def cmd_prepare(args):
    raw = Path(args.bench_dir) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in (mm.CLOSED_FILE, mm.OPEN_FILE):
        if not (raw / name).is_file():
            if not args.download:
                raise SystemExit(f"{raw / name} is missing. Re-run with --download (public dataset, no gate) or copy the file there.")
            from huggingface_hub import hf_hub_download
            print(f"[PREPARE] downloading {mm.DATASET_REPO}/{name} ...", flush=True)
            hf_hub_download(repo_id=mm.DATASET_REPO, filename=name, repo_type="dataset", local_dir=str(raw))
    manifest = mm.prepare(raw, prepared_dir(args), allow_md5_mismatch=args.allow_md5_mismatch)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if not manifest["md5_matches_authors"]:
        print("[PREPARE] WARNING: the files differ from the authors' MD5 and were accepted by --allow-md5-mismatch.", flush=True)


# ----------------------------------------------------------------------------------------------------------------------- leakage
def cmd_leakage(args):
    from dental_agent.data.dentex import load_dentex_dataset
    _, _, images, _ = load_all(args)
    bench = {int(r.id): r.local_path for r in images.itertuples()}
    reference = {}
    # train = the COMBINED enumeration folders (about 1,294 images, what the YOLO grounder was trained on; a superset of the VLM's train
    # images), plus the 50 validation images (the test split).
    for split, kw in (("train", dict(combine_enumeration_splits=True)), ("validation", {})):
        imgs, _, _ = load_dentex_dataset(data_dir=args.data_dir, split_name=split, **kw)
        for r in imgs.itertuples():
            path = str(r.local_path)
            if path and path != "None" and os.path.exists(path):
                reference[f"dentex_{split}::{int(r.id)}::{Path(path).name}"] = path
    print(f"[LEAKAGE] {len(bench)} benchmark images vs {len(reference)} DENTEX images", flush=True)
    report = lk.screen(bench, reference, max_hamming=args.max_hamming, min_ncc=args.min_ncc, top=args.top)
    out = Path(args.bench_dir) / "leakage_dentex.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("verdict", "n_bench", "n_reference", "exact", "near")}, indent=2))
    print("closest pairs:", json.dumps(report["closest"][:5], indent=2), f"\n[LEAKAGE] written to {out}")
    if report["exact"] or report["near"]:
        raise SystemExit("Benchmark images overlap with DENTEX. Do not report the affected images as held-out; see the file above.")


# -------------------------------------------------------------------------------------------------------------------------- policy
def _namespace_for_agent(args, harness_mode):
    ns = argparse.Namespace(**vars(args))
    ns.harness_mode, ns.dataset, ns.canonical_resize = harness_mode, "dentex", True
    ns.max_turns = args.max_turns
    return ns


def cmd_policy(args):
    if not any(a in args._argv for a in ("--canonical-letterbox", "--no-canonical-letterbox")):
        raise SystemExit("State the canonical view explicitly: --canonical-letterbox or --no-canonical-letterbox (it must match SFT, GRPO "
                         "and the DENTEX evaluation of this hardware profile).")
    from scripts import evaluate_models as ev
    from dental_agent.utils.canonical import set_letterbox
    ev.load_env()
    set_letterbox(args.canonical_letterbox)
    _, _, images, manifest = load_all(args)
    results = Path(args.results_dir)
    out = results / "policy" / rn.safe_name(args.policy_name, "policy name") / "findings.jsonl"
    adapter = args.adapter_path
    if not adapter or not Path(adapter).exists():
        raise SystemExit(f"--adapter-path {adapter!r} does not exist. The policy phase needs the exact trained adapter (grpo_policy).")
    ns = _namespace_for_agent(args, args.harness_mode)
    settings = {**ev.eval_settings(ns, "grpo_with_tools", adapter), "benchmark": mm.DATASET_REPO, "benchmark_md5": manifest["md5"],
                "policy_name": args.policy_name, "adapter_files_sha256": {p.name: sha256_of(p) for p in sorted(Path(adapter).glob("adapter_model*"))}}
    existing = args.existing_results
    if out.is_file() and existing != "overwrite":
        rn.resume_records(out, existing, settings, "image_id")      # settings are verified before the 9B model is loaded
    from dental_agent.evaluation.tool_ablation import AblationSpec, ToolProbe
    from dental_agent.model.local_runtime import runtime_from_args
    from dental_agent.tools.registry import ToolRegistry
    device, dtype, _ = ev.setup_eval_hardware(args.precision, getattr(args, "qlora_compute_dtype", "auto"))
    model, processor = ev.load_eval_model(args.model_id, adapter, device, dtype, "grpo_with_tools", quantize_4bit=args.precision == "qlora",
                                          runtime_options=runtime_from_args(args))
    registry = ToolRegistry.create_default()
    spec = AblationSpec()

    def run_one(image_id, _path):
        probe = ToolProbe(registry, spec, image_id)
        traj = ev.run_condition_trajectory(model, processor, images, image_id, probe.registry, ns)
        answered = isinstance(traj.final_answer, (list, dict))
        return dict(findings=ev.findings_from_trajectory(traj), answered=answered, format_ok=bool(traj.format_ok), turns=len(traj.turns),
                    tool_calls=int(traj.tool_calls), raw_output="\n---\n".join(str(t.get("raw_output", "")) for t in traj.turns),
                    trajectory=ev.transcript_for_record(traj))

    recs = rn.run_policy_phase(images.to_dict("records"), out, run_one, settings, existing, args.limit)
    n_ans = sum(bool(r.get("answered")) for r in recs)
    print(f"[POLICY] {len(recs)} images; {n_ans} with a valid final answer; {sum(bool(r['findings']) for r in recs)} with findings. -> {out}")


# -------------------------------------------------------------------------------------------------------------------------- answer
def load_external_vlm(model_id: str, revision: str, dtype, device):
    """A public non-repo VLM (e.g. an OralGPT release) with its own processor. No remote code is executed."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    kw = {"revision": revision} if revision else {}
    processor = AutoProcessor.from_pretrained(model_id, **kw)
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype, **kw).to(device)
    model.eval()
    return model, processor


def cmd_answer(args):
    closed, opened, images, manifest = load_all(args)
    results = Path(args.results_dir)
    kinds = tuple(args.kinds)
    questions = rn.build_questions(closed, opened, kinds)
    path_of = {int(r["id"]): r["local_path"] for r in images.to_dict("records")}
    findings, findings_meta = None, None
    if args.findings_from:
        src = results / "policy" / rn.safe_name(args.findings_from, "policy name") / "findings.jsonl"
        findings = rn.load_findings(src, {q["image_id"] for q in questions})
        pol = rn.read_jsonl(src)
        findings_meta = dict(findings_policy=args.findings_from, findings_file_sha256=sha256_of(src), findings_ctx_version=mm.FINDINGS_CONTEXT_VERSION,
                             policy_settings=pol[0]["settings"] if pol else None)
    settings = dict(system=args.system_name, benchmark=mm.DATASET_REPO, benchmark_md5=manifest["md5"], prompt_version=mm.PROMPT_VERSION,
                    loader=args.loader, model_id=args.model_id, model_revision=args.model_revision, adapter_path=args.adapter_path or "",
                    precision=args.precision, max_new_tokens=int(args.max_new_tokens), temperature=0.0, image_view="native",
                    thinking="closed <think> blocks removed before scoring", **(findings_meta or {"findings_policy": None}))
    out_dir = results / "answers" / rn.safe_name(args.system_name, "system name")
    # Verify resume settings before any model is loaded.
    for kind in kinds:
        existing_file = out_dir / f"{kind}.jsonl"
        if existing_file.is_file() and args.existing_results != "overwrite":
            rn.resume_records(existing_file, args.existing_results, {**settings, "kind": kind}, "index")
    from PIL import Image
    from scripts import evaluate_models as ev
    from dental_agent.model.inference import generate_agent_reply
    ev.load_env()
    device, dtype, _ = ev.setup_eval_hardware(args.precision, getattr(args, "qlora_compute_dtype", "auto"))
    if args.loader == "repo":
        from dental_agent.model.local_runtime import runtime_from_args
        model, processor = ev.load_eval_model(args.model_id, args.adapter_path, device, dtype, "base_no_tools" if not args.adapter_path else "grpo_with_tools",
                                              quantize_4bit=args.precision == "qlora", runtime_options=runtime_from_args(args))
    else:
        if args.adapter_path:
            raise SystemExit("--adapter-path is for the repo loader only.")
        model, processor = load_external_vlm(args.model_id, args.model_revision, dtype, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_info.json").write_text(json.dumps(dict(
        model_id=args.model_id, revision_requested=args.model_revision or None,
        commit_hash=getattr(getattr(model, "config", None), "_commit_hash", None), loader=args.loader), indent=2), encoding="utf-8")

    def answer_one(q, prompt):
        image = Image.open(path_of[q["image_id"]]).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        reply, _plen, ids = generate_agent_reply(model, processor, messages, max_new_tokens=int(args.max_new_tokens), temperature=0.0, return_ids=True)
        return dict(raw=reply, new_tokens=len(ids), hit_token_cap=len(ids) >= int(args.max_new_tokens))

    res = rn.run_answer_phase(questions, out_dir, answer_one, settings, findings, args.existing_results, args.limit)
    for kind, recs in res.items():
        capped = sum(r["hit_token_cap"] for r in recs)
        print(f"[ANSWER] {args.system_name} {kind}: {len(recs)} answers; {capped} hit the {args.max_new_tokens}-token cap; "
              f"{sum(r['unclosed_thinking'] for r in recs)} with unclosed thinking.")


# --------------------------------------------------------------------------------------------------------------------------- judge
def cmd_judge(args):
    results = Path(args.results_dir)
    answers = rn.read_jsonl(results / "answers" / args.system_name / "open.jsonl")
    if not answers:
        raise SystemExit(f"No open-ended answers for {args.system_name}; run the answer step first.")
    client = jg.JudgeClient(args.judge_model, base_url=args.judge_base_url or None, api_key_env=args.judge_key_env,
                            max_completion_tokens=args.judge_max_tokens, send_temperature=args.judge_send_temperature)
    items = [dict(index=a["index"], question=a["question"], ground_truth=a["answer"], prediction=a["prediction"], category=a["category"])
             for a in (answers[:args.limit] if args.limit else answers)]
    jg.judge_answers(items, judge_file(results, args.system_name, args.judge_model), client, workers=args.workers, existing=args.existing_results)


# --------------------------------------------------------------------------------------------------------------------------- score
def system_vectors(results: Path, system: str, judge_model: str | None, seed: int):
    closed_rows = rn.read_jsonl(results / "answers" / system / "closed.jsonl")
    open_rows = rn.read_jsonl(results / "answers" / system / "open.jsonl")
    v = {}
    if closed_rows:
        v["closed_official"] = tb.closed_vectors(closed_rows, seed, "official")
        v["closed_strict"] = tb.closed_vectors(closed_rows, seed, "strict")
    if open_rows:
        if not judge_model:
            raise SystemExit(f"{system} has open-ended answers: give --judge-model so they are scored.")
        v["open"] = tb.open_vectors(open_rows, rn.read_jsonl(judge_file(results, system, judge_model)))
    return v, closed_rows, open_rows


def cmd_score(args):
    results = Path(args.results_dir)
    measured, detail = [], {}

    def rows_for(vec, part):
        cats = mm.CLOSED_CATEGORIES if part == "closed" else mm.OPEN_CATEGORIES
        return [dict(part=part, category=g or "Overall", **tb.cluster_mean_ci(vec, g, args.n_boot, args.seed)) for g in (*cats, None)]

    for system in args.systems:
        vec, closed_rows, open_rows = system_vectors(results, system, args.judge_model, args.seed)
        for key, label, judge, part in (("closed_official", system, "none", "closed"), ("closed_strict", f"{system} [strict]", "none", "closed"),
                                        ("open", system, args.judge_model, "open")):
            if key in vec:
                measured += [dict(system=label, judge=judge, **row) for row in rows_for(vec[key], part)]
        detail[system] = dict(
            closed=mm.score_closed(closed_rows, args.seed) if closed_rows else None,
            open=mm.score_open([dict(category=r["category"], score=vec["open"][int(r["index"])][1]) for r in open_rows]) if open_rows else None)
    out = results / "tables"
    out.mkdir(parents=True, exist_ok=True)
    md = tb.render_markdown(measured, tb.load_published(args.published))
    (out / "mmoral_opg.md").write_text(md, encoding="utf-8")
    (out / "mmoral_opg_scores.json").write_text(json.dumps(dict(measured=measured, official_format=detail, judge_model=args.judge_model,
                                                               seed=args.seed, n_boot=args.n_boot), indent=2), encoding="utf-8")
    print(md)
    print(f"[SCORE] written to {out}")


def cmd_compare(args):
    results = Path(args.results_dir)
    a, *_ = system_vectors(results, args.a, args.judge_model, args.seed)
    b, *_ = system_vectors(results, args.b, args.judge_model, args.seed)
    report = {}
    for key in a:
        if key not in b:
            continue
        cats = mm.OPEN_CATEGORIES if key == "open" else mm.CLOSED_CATEGORIES
        report[key] = {(g or "Overall"): tb.paired_delta(a[key], b[key], g, args.n_boot, args.seed) for g in (*cats, None)}
    print(f"Paired difference {args.a} minus {args.b} (x100; interval resamples radiographs):")
    for key, rows in report.items():
        print(f"\n[{key}]")
        for cat, r in rows.items():
            if r["n"]:
                print(f"  {cat:<8} n={r['n']:<4} delta={r['mean']:+6.2f} [{r['lo']:+6.2f}, {r['hi']:+6.2f}]{'  *' if r['significant_95'] else ''}")
    out = results / "tables" / f"compare__{args.a}__minus__{args.b}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------------------------------------------------------- sync
def cmd_sync(args):
    from dental_agent.harness.notebook_eval import result_repo_type
    from huggingface_hub import HfApi, snapshot_download
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    repo_type = result_repo_type(api, args.repo, args.repo_type)
    results = Path(args.results_dir)
    prefix = f"external/mmoral_opg/{args.tag}"
    patterns = ["*.jsonl", "*.json", "*.md", "*.csv"]
    if args.direction == "up":
        if not results.is_dir():
            raise SystemExit(f"{results} does not exist")
        api.upload_folder(repo_id=args.repo, repo_type=repo_type, folder_path=str(results), path_in_repo=prefix, allow_patterns=patterns,
                          commit_message=f"MMOral-OPG-Bench results ({args.tag})")
        print(f"[SYNC] uploaded {results} -> {args.repo}/{prefix}")
    else:
        tmp = Path(args.bench_dir) / "_sync_tmp"
        snapshot_download(repo_id=args.repo, repo_type=repo_type, allow_patterns=[f"{prefix}/*"], local_dir=str(tmp), token=token)
        src, copied = tmp / prefix, 0
        for f in src.rglob("*") if src.is_dir() else []:
            if f.is_file():
                dst = results / f.relative_to(src)
                if not dst.exists():                         # never overwrite a local file
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dst)
                    copied += 1
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"[SYNC] restored {copied} file(s) into {results} (existing local files untouched)")


# ------------------------------------------------------------------------------------------------------------------------------- CLI
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare"); add_common(s, results=False)
    s.add_argument("--download", action="store_true"); s.add_argument("--allow-md5-mismatch", action="store_true"); s.set_defaults(fn=cmd_prepare)

    s = sub.add_parser("leakage"); add_common(s, results=False)
    s.add_argument("--data-dir", default=os.environ.get("DENTAL_AGENT_DATA_DIR", "data"))
    s.add_argument("--max-hamming", type=int, default=24); s.add_argument("--min-ncc", type=float, default=0.95); s.add_argument("--top", type=int, default=10)
    s.set_defaults(fn=cmd_leakage)

    s = sub.add_parser("policy"); add_common(s); add_model_args(s)
    s.add_argument("--policy-name", required=True); s.add_argument("--adapter-path", required=True)
    from dental_agent.harness.configuration import HARNESS_MODES
    s.add_argument("--harness-mode", choices=HARNESS_MODES, required=True)
    s.add_argument("--max-tool-calls", type=int, default=50); s.add_argument("--max-turns", type=int, default=None)
    s.add_argument("--malformed-retries", type=int, default=0); s.set_defaults(fn=cmd_policy)

    s = sub.add_parser("answer"); add_common(s); add_model_args(s)
    s.add_argument("--system-name", required=True); s.add_argument("--loader", choices=["repo", "external"], default="repo")
    s.add_argument("--adapter-path", default=None); s.add_argument("--findings-from", default=None, help="policy name whose cached findings are added (Option A)")
    s.add_argument("--kinds", nargs="+", choices=["closed", "open"], default=["closed", "open"]); s.set_defaults(fn=cmd_answer)

    s = sub.add_parser("judge"); add_common(s)
    s.add_argument("--system-name", required=True); s.add_argument("--judge-model", required=True)
    s.add_argument("--judge-base-url", default=""); s.add_argument("--judge-key-env", default="OPENAI_API_KEY")
    s.add_argument("--judge-max-tokens", type=int, default=16384)
    s.add_argument("--judge-send-temperature", action=argparse.BooleanOptionalAction, default=True)
    s.add_argument("--workers", type=int, default=4); s.add_argument("--limit", type=int, default=None)
    s.add_argument("--existing-results", default="resume", choices=["error", "resume", "overwrite"]); s.set_defaults(fn=cmd_judge)

    s = sub.add_parser("score"); add_common(s)
    s.add_argument("--systems", nargs="+", required=True); s.add_argument("--judge-model", default=None)
    s.add_argument("--published", default=None); s.add_argument("--seed", type=int, default=0); s.add_argument("--n-boot", type=int, default=2000)
    s.set_defaults(fn=cmd_score)

    s = sub.add_parser("compare"); add_common(s)
    s.add_argument("--a", required=True); s.add_argument("--b", required=True); s.add_argument("--judge-model", default=None)
    s.add_argument("--seed", type=int, default=0); s.add_argument("--n-boot", type=int, default=2000); s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("sync"); add_common(s)
    s.add_argument("--direction", choices=["up", "down"], required=True); s.add_argument("--tag", required=True)
    s.add_argument("--repo", default=os.environ.get("HF_EVAL_REPO", "Reza-Nadimi/vlm-dental-evaluations"))
    s.add_argument("--repo-type", default=None, choices=[None, "model", "dataset"]); s.set_defaults(fn=cmd_sync)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args._argv = list(sys.argv[1:] if argv is None else argv)
    args.fn(args)


if __name__ == "__main__":
    main()
