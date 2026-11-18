#!/usr/bin/env python3
"""Evaluate hosted (Kaggle Benchmarks) models with the project's own agent loop, tools, harness and clinical evaluator.

    estimate   worst-case / expected request, token and cost numbers for a roster (no network)
    preflight  grounder loads + SHA pinned, images present, prompts render, YOLO/test overlap (no model calls)
    roster     list models from the Kaggle CLI and/or probe vision (``--probe`` makes one tiny request per model)
    run        run one model x condition over the cohort LIVE (spends quota: needs --confirm-spend and a request budget). Pauses
               (never errors) on an expired key or a spend cap; run the same command later and it carries on without re-spending.
    hosted     the same experiment as hosted (non-live) Kaggle runs over the full model catalogue, in shards:
               push | status | collect | evaluate   (state lives in <root>/campaign.json; every step is idempotent)
    import     score transport files locally into evaluate_models-schema records
    table      build the paper comparison table from evaluated records

Nothing here runs implicitly; see docs/KAGGLE_EXTERNAL_MODELS.md for the gates (G1-G4) that must pass before a real run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dental_agent.external import cohort as cohort_mod
from dental_agent.external import conditions as C
from dental_agent.external import cost_guard, hosted_driver, identity, importer, pause, preflight, roster, tables


def _load_split(args):
    """Images / annotations / categories / cohort settings through the project's own loader (``--split official_test250`` is the 250-image cohort)."""
    images, annots, cats, cohort = cohort_mod.load_split(args.split, dataset=args.dataset, data_dir=args.data_dir, dev_split=args.dev_split)
    if args.ids_file:
        ids = {int(x) for x in Path(args.ids_file).read_text().split()}
        images = images[images["id"].isin(ids)].copy()
    return images, annots, cats, cohort


def _add_data(p):
    p.add_argument("--data-dir", default="data")
    p.add_argument("--dataset", default="dentex")
    p.add_argument("--split", default="test", help="test (official 50 images) | dev | official_test250 (the 250-image cohort: cohort prompts, scoring and identity)")
    p.add_argument("--dev-split", default=None)
    p.add_argument("--ids-file", default=None, help="whitespace-separated image ids to keep (a subset of the split)")


def cmd_estimate(a):
    price = cost_guard.Price(a.price_in, a.price_out) if a.price_in is not None and a.price_out is not None else None
    if a.model:
        from dental_agent.external import model_profiles
        price = model_profiles.get_profile(a.model).price
    out = {c: cost_guard.estimate(c, a.n_images, price=price, expected_turns=a.expected_turns if C.get_condition(c).with_tools else None)
           for c in (a.conditions or C.required_conditions())}
    print(json.dumps(out, indent=2))


def cmd_preflight(a):
    images, _, _, _ = _load_split(a)
    ids = [int(i) for i in images["id"]]
    res = {"images": preflight.check_images(images, ids), "prompts": preflight.check_prompts_render(a.split, a.dataset), "grounder": preflight.check_grounder()}
    if a.yolo_train_ids:
        res["yolo_overlap"] = preflight.check_yolo_overlap([int(x) for x in Path(a.yolo_train_ids).read_text().split()], ids)
    print(json.dumps(res, indent=2, default=str))
    preflight.require(res)


def cmd_roster(a):
    from dental_agent.external import model_profiles
    # the proxy serves far fewer models than `kaggle benchmarks tasks models` lists, so the default roster is the verified profiles
    entries = roster.parse_listing(Path(a.listing).read_text()) if a.listing else model_profiles.as_roster()
    if a.probe:
        from dental_agent.external.kaggle_provider import kaggle_cli_refresher
        for e in entries:
            e.vision = roster.probe_vision(model_profiles.backend_for(e.slug, refresher=kaggle_cli_refresher()))
            e.source += "+probe"
    roster.save(entries, a.out)
    print(f"{len(entries)} models, {len(roster.usable(entries))} with confirmed vision -> {a.out}")


def cmd_run(a):
    if not a.confirm_spend or a.max_requests is None:
        sys.exit("refusing to spend quota: pass --confirm-spend and --max-requests N")
    from dental_agent.external import coordinator
    from dental_agent.external import model_profiles
    from dental_agent.external.kaggle_provider import kaggle_cli_refresher
    profile = model_profiles.get_profile(a.model)
    images, _, _, cohort = _load_split(a)
    ids = [int(i) for i in images["id"]]
    if a.limit:
        ids = ids[: a.limit]
    g = preflight.check_grounder() if C.get_condition(a.condition).with_tools else {"ok": True, "sha256": None}
    preflight.require({"grounder": g, "images": preflight.check_images(images, ids)})
    settings = identity.run_settings(C.get_condition(a.condition), a.model, cohort=identity.cohort_sha(ids, cohort), split=a.split, dataset=a.dataset, grounder_sha256=g.get("sha256"), decoding=profile.decoding_identity(), cohort_settings=cohort)
    ledger = cost_guard.CostLedger(Path(a.out) / "ledger.jsonl", cost_guard.Budget(a.max_requests, a.max_cost_usd, a.max_input_tokens))
    backend = model_profiles.backend_for(a.model, path=a.endpoint_path, refresher=kaggle_cli_refresher())
    guard = None if a.no_quota_guard else pause.QuotaGuard(pause.kaggle_quota_runner, reserve_usd=a.reserve_usd)
    stats = coordinator.run_cell(backend=backend, condition=a.condition, images_df=images, image_ids=ids, out_dir=a.out, settings=settings,
                                 ledger=ledger, guard=guard, workers=a.workers)
    print(json.dumps(stats, indent=2))
    if stats["paused"]:
        print(f"PAUSED ({stats['paused']['kind']}): re-run this exact command later; answered turns are replayed for free.")


def _campaign(a):
    images, _, _, cohort = _load_split(a)
    return hosted_driver.build_campaign(
        Path(a.root), images, conditions=a.conditions or C.required_conditions(), models=a.models or None, split=a.split, dataset=a.dataset,
        shard_size=a.shard_size, workers=a.workers, env_dataset=None if a.no_env_dataset else "auto", hf_results_repo=a.hf_results_repo,
        plugins=a.plugin, extra_datasets=a.extra_dataset or None, cohort_settings=cohort)


def cmd_hosted(a):
    camp = _campaign(a)
    out = Path(a.root) / "results"
    if a.action == "push":
        print(json.dumps({"push": camp.push_missing(), "schedule": camp.schedule(max_inflight=a.max_inflight)}, indent=2))
    elif a.action == "status":
        camp.refresh()
        print(json.dumps(camp.report(out), indent=2, default=str))
    elif a.action == "collect":
        camp.refresh()
        tally = camp.collect(out)
        new = camp.plan_remainders(out)
        if new:
            print(f"{len(new)} remainder task(s) planned for images some models still lack; run `hosted push` to start them")
        print(json.dumps({"collected": tally, "report": camp.report(out)}, indent=2, default=str))
    else:  # evaluate
        _, annots, cats, _ = _load_split(a)
        res = camp.evaluate(out, annots, cats)
        print(json.dumps({k: {"evaluated": len(v["records"]), "pending": len(v["pending"]), "errors": len(v["errors"])} for k, v in res.items()}, indent=2))


def cmd_import(a):
    images, annots, cats, _ = _load_split(a)
    ids = {int(i) for i in images["id"]}
    paths = importer.cell_paths(a.out, a.model, a.condition)
    transport = Path(a.transport) if a.transport else paths["transport"]
    meta = json.loads(paths["meta"].read_text()) if paths["meta"].exists() else None
    settings = (meta or {}).get("settings")
    if settings is None:
        sys.exit("no meta.json next to the transport file: settings are needed to verify identity (copy it from the run)")
    res = importer.import_transport(transport, annots, cats, settings=settings, dataset=a.dataset, split=a.split, cohort_ids=ids, evaluated_path=paths["evaluated"])
    print(json.dumps({"evaluated": len(res["records"]), "pending": len(res["pending"]), "errors": len(res["errors"]), "path": str(paths["evaluated"])}, indent=2))


def cmd_table(a):
    from dental_agent.evaluation.matrix import load_records
    cells = {}
    for spec in a.cell:
        label, path = spec.split("=", 1)
        cells[label] = load_records(path)
    rows = tables.table_rows(cells, n_boot=a.n_boot)
    text = tables.markdown_table(rows)
    pairs = [tuple(x.split("|")) for x in a.pair] or [p for m in a.model for p in tables.standard_pairs(m)]
    text += "\n" + tables.markdown_effects(tables.harness_effect(cells, pairs, n_boot=a.n_boot))
    Path(a.out).write_text(text)
    print(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("estimate"); s.add_argument("--n-images", type=int, default=250); s.add_argument("--conditions", nargs="*")
    s.add_argument("--model", help="take the price from the verified profile"); s.add_argument("--expected-turns", type=float); s.add_argument("--price-in", type=float); s.add_argument("--price-out", type=float); s.set_defaults(fn=cmd_estimate)
    s = sub.add_parser("preflight"); _add_data(s); s.add_argument("--yolo-train-ids"); s.set_defaults(fn=cmd_preflight)
    s = sub.add_parser("roster"); s.add_argument("--listing"); s.add_argument("--probe", action="store_true"); s.add_argument("--out", default="results/kaggle/roster.json"); s.set_defaults(fn=cmd_roster)
    s = sub.add_parser("run"); _add_data(s); s.add_argument("--model", required=True); s.add_argument("--condition", required=True, choices=list(C.CONDITIONS))
    s.add_argument("--out", default="results/kaggle"); s.add_argument("--limit", type=int); s.add_argument("--endpoint-path", default="/openapi/chat/completions")
    s.add_argument("--confirm-spend", action="store_true"); s.add_argument("--max-requests", type=int); s.add_argument("--max-cost-usd", type=float); s.add_argument("--max-input-tokens", type=int)
    s.add_argument("--workers", type=int, default=1, help="images in parallel (the proxy accepts 16 parallel requests)")
    s.add_argument("--no-quota-guard", action="store_true", help="do not read `kaggle benchmarks quota` before each image"); s.add_argument("--reserve-usd", type=float, default=1.0)
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("hosted"); _add_data(s); s.add_argument("action", choices=["push", "status", "collect", "evaluate"])
    s.add_argument("--root", default="results/kaggle/hosted"); s.add_argument("--conditions", nargs="*"); s.add_argument("--models", nargs="*", help="default: every vision-capable hosted model")
    s.add_argument("--shard-size", type=int, default=25); s.add_argument("--workers", type=int, default=4); s.add_argument("--max-inflight", type=int, default=60)
    s.add_argument("--hf-results-repo", help="EXISTING private repo the workers push finished shards to (needs HF_TOKEN in the .env dataset)")
    s.add_argument("--no-env-dataset", action="store_true"); s.add_argument("--plugin", action="append", default=[], help="module that registers conditions / tool packs / evaluators")
    s.add_argument("--extra-dataset", action="append", default=[], help="extra Kaggle dataset to attach (e.g. the grounder weights)"); s.set_defaults(fn=cmd_hosted)
    s = sub.add_parser("import"); _add_data(s); s.add_argument("--model", required=True); s.add_argument("--condition", required=True, choices=list(C.CONDITIONS))
    s.add_argument("--out", default="results/kaggle"); s.add_argument("--transport"); s.set_defaults(fn=cmd_import)
    s = sub.add_parser("table"); s.add_argument("--cell", action="append", required=True, metavar="LABEL=evaluated.jsonl"); s.add_argument("--model", action="append", default=[])
    s.add_argument("--pair", action="append", default=[], metavar="LABEL|A|B"); s.add_argument("--n-boot", type=int, default=2000); s.add_argument("--out", default="results/kaggle/comparison.md"); s.set_defaults(fn=cmd_table)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
