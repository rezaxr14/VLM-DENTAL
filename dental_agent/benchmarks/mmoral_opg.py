"""MMOral-OPG-Bench (NeurIPS 2025): preparation, official prompts, official scoring.

Source of truth: the benchmark TSVs on the Hub (``OralGPT/MMOral-OPG-Bench``, MIT) and the scoring code of
``MMOral-Bench-EvalKit`` (``vlmeval/dataset/mmoral.py`` and ``utils/mmoral.py``, Apache-2.0, https://github.com/isjinghao/OralGPT).
The answer prompts, the closed-ended answer extraction, the open-ended judge prompt, its retry rule and the category aggregation are
ported verbatim; the only differences are the ones listed here, and each is visible in the results:

* closed-ended extraction falls back to a RANDOM option when no option can be read (official behaviour). Here that choice is seeded per
  question, and a STRICT score (unreadable = wrong) and the count of unreadable answers are reported next to the official one;
* a model's ``<think>...</think>`` block is removed before extraction and judging (``strip_thinking``), and counted;
* API/transport errors of the judge stop the run (it resumes); they are never scored as 0.0. A judge reply that cannot be parsed after the
  official 5 tries scores 0.0, as in the official code.

The Hub TSVs hold the images as base64 text. ``prepare`` decodes every distinct image once into ``images/`` and writes slim question files
(no base64), so later phases never re-read the 382 MB TSVs.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import random
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

DATASET_REPO = "OralGPT/MMOral-OPG-Bench"
CLOSED_FILE = "MMOral-OPG-Bench-Closed-Ended.tsv"
OPEN_FILE = "MMOral-OPG-Bench-Open-Ended.tsv"
# MD5 of the two TSVs as listed by the benchmark authors in VLMEvalKit (DATASET_MD5). A different file stops `prepare` unless the
# override is given explicitly; the actual MD5 is always printed and stored in the manifest.
EXPECTED_MD5 = {CLOSED_FILE: "d70060e427f7d68a7cad1edb513b3a5f", OPEN_FILE: "d328b1b527ef7467b328d8b35d5f8155"}
OPTION_COLUMNS = ("option1", "option2", "option3", "option4")
LETTERS = ("A", "B", "C", "D")
CLOSED_REQUIRED = ("index", "question", "category", "image", "answer", *OPTION_COLUMNS)
OPEN_REQUIRED = ("index", "question", "category", "image", "answer")
CLOSED_CATEGORIES = ("Teeth", "Patho", "HisT", "Jaw", "SumRec")
OPEN_CATEGORIES = ("Teeth", "Patho", "HisT", "Jaw", "SumRec", "Report")
MANIFEST = "manifest.json"
IMAGES_FILE = "images.csv"
CLOSED_SLIM = "closed.jsonl"
OPEN_SLIM = "open.jsonl"
PROMPT_VERSION = "official_v1"
FINDINGS_CONTEXT_VERSION = "findings_ctx_v1"
DENTEX_CLASSES = ("Caries", "Deep Caries", "Periapical Lesion", "Impacted Tooth")


# ----------------------------------------------------------------------------------------------------------------- preparation
def file_md5(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _resolve_image_refs(df):
    """VLMEvalKit convention: an ``image`` value of at most 64 characters is the ``index`` of another row that holds the base64."""
    by_index = {str(i): v for i, v in zip(df["index"], df["image"])}
    out = []
    for value in df["image"]:
        value = str(value)
        hops = 0
        while len(value) <= 64 and value in by_index and hops < 5:
            value, hops = str(by_index[value]), hops + 1
        if len(value) <= 64:
            raise ValueError(f"image field {value!r} is neither base64 nor the index of a row that holds it")
        out.append(value)
    return out


def read_tsv(path: str | Path, required: Sequence[str], what: str):
    import pandas as pd
    df = pd.read_csv(path, sep="\t")          # same reader as VLMEvalKit (default NA handling included)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{what}: the TSV {path} lacks column(s) {missing}; it has {list(df.columns)}. "
                         "The benchmark format changed: stop and inspect before running anything.")
    if df["index"].duplicated().any():
        raise ValueError(f"{what}: duplicate values in the 'index' column")
    return df


def decode_image(b64: str) -> tuple[bytes, str]:
    from PIL import Image
    raw = base64.b64decode(b64)
    with Image.open(io.BytesIO(raw)) as im:
        im.verify()
    with Image.open(io.BytesIO(raw)) as im:
        fmt = (im.format or "JPEG").lower()
    return raw, {"jpeg": "jpg"}.get(fmt, fmt)


def prepare(tsv_dir: str | Path, out_dir: str | Path, *, allow_md5_mismatch: bool = False) -> dict[str, Any]:
    """Decode the two TSVs into ``out_dir``: ``images/<sha16>.<ext>``, ``images.csv``, ``closed.jsonl``, ``open.jsonl``, ``manifest.json``.

    Image ids (0..N-1) follow the sorted SHA-256 of the image bytes, so they do not depend on row order and the closed and
    open-ended questions of one radiograph share one id. Returns the manifest.
    """
    import pandas as pd
    tsv_dir, out = Path(tsv_dir), Path(out_dir)
    md5 = {}
    for name in (CLOSED_FILE, OPEN_FILE):
        if not (tsv_dir / name).is_file():
            raise FileNotFoundError(f"{tsv_dir / name} not found (download {DATASET_REPO}).")
        md5[name] = file_md5(tsv_dir / name)
        if md5[name] != EXPECTED_MD5[name] and not allow_md5_mismatch:
            raise ValueError(f"{name}: MD5 {md5[name]} differs from the benchmark authors' {EXPECTED_MD5[name]}. The Hub file was "
                             "changed or damaged. Inspect it, then re-run with allow_md5_mismatch=True to record this exact file.")
    closed = read_tsv(tsv_dir / CLOSED_FILE, CLOSED_REQUIRED, "closed-ended")
    opened = read_tsv(tsv_dir / OPEN_FILE, OPEN_REQUIRED, "open-ended")
    for col in OPTION_COLUMNS:
        closed[col] = closed[col].fillna("None")          # VLMEvalKit post_build
    (out / "images").mkdir(parents=True, exist_ok=True)

    decoded: dict[str, tuple[bytes, str]] = {}
    sha_of_row: dict[tuple[str, Any], str] = {}
    for kind, df in (("closed", closed), ("open", opened)):
        for idx, b64 in zip(df["index"], _resolve_image_refs(df)):
            raw, ext = decode_image(b64)
            sha = hashlib.sha256(raw).hexdigest()
            decoded.setdefault(sha, (raw, ext))
            sha_of_row[(kind, idx)] = sha
    shas = sorted(decoded)
    ids = {sha: i for i, sha in enumerate(shas)}
    rows = []
    for sha in shas:
        raw, ext = decoded[sha]
        path = out / "images" / f"{sha[:16]}.{ext}"
        if not path.is_file() or path.read_bytes() != raw:
            path.write_bytes(raw)
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
        rows.append(dict(id=ids[sha], sha256=sha, local_path=str(path.resolve()), width=w, height=h))
    images = pd.DataFrame(rows)
    images.to_csv(out / IMAGES_FILE, index=False)

    def slim(df, kind, keep):
        d = df.drop(columns=["image"]).copy()
        d["image_id"] = [ids[sha_of_row[(kind, i)]] for i in df["index"]]
        d = d[[c for c in keep if c in d.columns]]
        d.to_json(out / (CLOSED_SLIM if kind == "closed" else OPEN_SLIM), orient="records", lines=True, force_ascii=False)
        return d

    c_keep = ["index", "image_id", "question", "category", "answer", *OPTION_COLUMNS]
    o_keep = ["index", "image_id", "question", "category", "answer"]
    c, o = slim(closed, "closed", c_keep), slim(opened, "open", o_keep)
    manifest = dict(
        dataset=DATASET_REPO, md5=md5, md5_expected=EXPECTED_MD5, md5_matches_authors=all(md5[k] == EXPECTED_MD5[k] for k in md5),
        n_closed=len(c), n_open=len(o), n_images=len(images),
        n_images_with_closed=int(c["image_id"].nunique()), n_images_with_open=int(o["image_id"].nunique()),
        closed_categories=c["category"].value_counts().to_dict(), open_categories=o["category"].value_counts().to_dict(),
        closed_answer_letters=c["answer"].astype(str).value_counts().to_dict(),
        nan_cells=dict(closed_question=int(c["question"].isna().sum()), open_question=int(o["question"].isna().sum()),
                       open_answer=int(o["answer"].isna().sum())),
        image_size_min=[int(images.width.min()), int(images.height.min())], image_size_max=[int(images.width.max()), int(images.height.max())],
    )
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def load_prepared(out_dir: str | Path):
    """(closed_df, open_df, images_df). ``images_df`` has the ``id`` and ``local_path`` columns ``run_agent`` needs."""
    import pandas as pd
    out = Path(out_dir)
    for name in (MANIFEST, IMAGES_FILE, CLOSED_SLIM, OPEN_SLIM):
        if not (out / name).is_file():
            raise FileNotFoundError(f"{out / name} missing: run the prepare step first.")
    closed = pd.read_json(out / CLOSED_SLIM, lines=True, dtype=False)
    opened = pd.read_json(out / OPEN_SLIM, lines=True, dtype=False)
    images = pd.read_csv(out / IMAGES_FILE)
    for col in ("question", "answer", "category", *OPTION_COLUMNS):
        if col in closed:
            closed[col] = closed[col].astype(str)
    for col in ("question", "answer", "category"):
        opened[col] = opened[col].astype(str)
    return closed, opened, images


# ----------------------------------------------------------------------------------------------------------------------- prompts
def closed_prompt(question: str, options: Sequence[Any]) -> str:
    """Official MMOral_OPG_CLOSED.build_prompt text (the image is attached separately)."""
    options_prompt = "Options:\n"
    for letter, value in zip(LETTERS, options):
        options_prompt += f"{letter}. {str(value)}\n"
    return (f"Question: {question}\n" + options_prompt
            + "Please answer the above multiple-choice question by selecting the single correct option (A, B, C, or D). "
            + "If the provided information is insufficient to determine a clear answer, please choose the most likely "
            + "correct option based on the available data and your judgment.")


def open_prompt(question: str) -> str:
    """Official MMOral_OPG_OPEN.build_prompt text."""
    return f"Question: {question}\nPlease provide a detailed and accurate answer to the question."


def findings_context(findings: Sequence[dict[str, Any]] | None, answered: bool) -> str:
    """The extra block of the agent-assisted condition (version ``findings_ctx_v1``). It is put before the official prompt, which
    stays byte-identical. The block says what the agent can and cannot see, so that 'no finding' is not read as 'healthy'."""
    from dental_agent.tools.fdi import fdi_encode
    head = ("A dental radiograph analysis agent has already examined this image with zoom, contrast and tooth-localisation tools. "
            f"It reports findings for only four conditions ({', '.join(DENTEX_CLASSES)}), using FDI tooth numbers. "
            "It does not assess any other finding, so the absence of a finding does not rule out anything.\n")
    if not answered:
        return head + "Agent result: the agent did not return a valid result for this image.\n\n"
    lines = []
    for f in findings or []:
        q, p = f.get("quadrant"), f.get("tooth_position")
        try:
            tooth = f"#{fdi_encode(int(q), int(p))}"
        except (TypeError, ValueError):
            tooth = "an unlocated tooth"
        conf = f.get("confidence")
        lines.append(f"- {tooth}: {f.get('diagnosis') or 'Unknown'}" + (f" (confidence {float(conf):.2f})" if conf is not None else ""))
    body = ("Agent result:\n" + "\n".join(lines)) if lines else "Agent result: none of the four conditions was found."
    return head + body + "\n\n"


def with_findings(prompt: str, findings: Sequence[dict[str, Any]] | None, answered: bool) -> str:
    return findings_context(findings, answered) + prompt


_THINK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def strip_thinking(text: str) -> tuple[str, dict[str, bool]]:
    """Remove closed ``<think>`` blocks (and an unclosed trailing one). The repo's agent parser does the same."""
    text = text or ""
    cleaned = _THINK.sub("", text)
    unclosed = bool(re.search(r"<think>", cleaned, re.IGNORECASE))
    if unclosed:
        cleaned = re.split(r"<think>", cleaned, maxsplit=1, flags=re.IGNORECASE)[0]
    return cleaned.strip(), dict(thinking_stripped=cleaned.strip() != text.strip(), unclosed_thinking=unclosed)


# ------------------------------------------------------------------------------------------------------------------ closed scoring
def extract_choice(response: str, all_choices: Sequence[str], index2ans: dict[str, Any], rng: random.Random | None = None,
                   *, strict: bool = False) -> str | None:
    """Port of ``get_single_choice_prediction``. With ``strict`` an unreadable answer returns None instead of a random option."""
    response = str(response)
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "
    candidates = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
    if len(candidates) == 0:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)
            elif f" {choice}." in response:
                candidates.append(choice)
            elif f" {choice}," in response:
                candidates.append(choice)
    if len(candidates) == 0:
        for index, ans in index2ans.items():
            if str(ans) in response:
                candidates.append(index)
    if len(candidates) > 0:
        positions = {}
        for c in candidates:
            pos = response.find(f" {c} ")
            if pos == -1:
                pos = response.find(f"({c})")
            if pos == -1:
                pos = response.find(str(index2ans[c]))
            if pos != -1:
                positions[c] = pos
        if positions:
            return min(positions.items(), key=lambda x: x[1])[0]
    if strict:
        return None
    return (rng or random).choice(list(all_choices))


def _table(tot, score, names, scale=100.0):
    return [dict(category=n, n=int(tot[n]) if float(tot[n]).is_integer() else float(tot[n]), acc=(score[n] / tot[n] * scale if tot[n] else 0.0))
            for n in names]


class _Tally:
    """Two tallies of one scoring pass.

    ``clean``: every question counts once in each main category it belongs to (a 'Teeth,Patho' question counts in both) and once in
    Overall. ``official``: the key list of the benchmark's own scripts. There a question whose category is a single label ('Jaw') adds to
    the main-category counter once for the main category and again for the exact category and its comma-free copy, which are the same key
    (3 times in the closed-ended script, 2 times in the open-ended one), so single-label questions weigh more in the five main-category
    columns. Overall and the exact-category tables are not affected. Both are returned; the paper table uses ``clean``.
    """

    def __init__(self, mains, official_extra):
        from collections import defaultdict
        self.mains, self.official_extra = mains, official_extra
        self.tc, self.sc = defaultdict(float), defaultdict(float)      # clean, main categories + Overall
        self.te, self.se = defaultdict(float), defaultdict(float)      # exact category strings (never share a key with a main category)
        self.to, self.so = defaultdict(float), defaultdict(float)      # official weighting
        self.cats: list[str] = []

    def add(self, cat: str, grade: float):
        if cat not in self.cats:
            self.cats.append(cat)
        hits = [m for m in self.mains if m in cat]
        for k in (*hits, "Overall"):
            self.tc[k] += 1
            self.sc[k] += grade
        self.te[cat] += 1
        self.se[cat] += grade
        for k in (*hits, *self.official_extra(cat), "Overall"):
            self.to[k] += 1
            self.so[k] += grade

    def tables(self):
        names = [*self.mains, "Overall"]
        return dict(main=_table(self.tc, self.sc, names), main_official_weighting=_table(self.to, self.so, names),
                    detailed=_table(self.te, self.se, sorted(self.cats)) + _table(self.tc, self.sc, ["Overall"]))


def score_closed(rows: Iterable[dict[str, Any]], seed: int = 0) -> dict[str, Any]:
    """rows: dicts with ``index``, ``category``, ``answer`` (a letter), ``prediction`` (text) and ``option1..option4``.

    A row with ``generation_refused`` (TPU static generation produced no answer) is wrong in both modes and is counted in ``refused``.
    Returns, for the official extraction (random fallback, seeded per question) and the strict one (unreadable = wrong): the main-category
    table counted once per question (``main``), the same table with the official scripts' key weighting (``main_official_weighting``) and
    the exact-category table (``detailed``); plus the number of unreadable answers.
    """
    rows = list(rows)
    out: dict[str, Any] = {}
    unreadable = 0
    refused = sum(1 for r in rows if r.get("generation_refused"))
    for mode in ("official", "strict"):
        tally = _Tally(CLOSED_CATEGORIES, lambda cat: (cat, cat.replace(",", "_")))
        for r in rows:
            if r.get("generation_refused"):                      # the static TPU generator produced no answer: wrong, never a random option
                tally.add(str(r["category"]), 0.0)
                continue
            index2ans = {L: r[f"option{i}"] for L, i in zip(LETTERS, (1, 2, 3, 4))}
            choice = extract_choice(r["prediction"], LETTERS, index2ans, random.Random(f"{seed}:{r['index']}"), strict=mode == "strict")
            if mode == "strict" and choice is None:
                unreadable += 1
            tally.add(str(r["category"]), float(choice == str(r["answer"])))
        out[mode] = tally.tables()
    out["n"] = len(rows)
    out["refused"] = refused
    out["unreadable"] = unreadable
    out["unreadable_fraction"] = unreadable / len(rows) if rows else 0.0
    return out


# -------------------------------------------------------------------------------------------------------------------- open scoring
_JUDGE_HEAD = """
Given the question, compare the ground truth and prediction from AI models, to generate a correctness score for the prediction.
The correctness score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right).
Just complete the last space of the correctness score.

Question | Ground truth | Prediction | Correctness
--- | --- | --- | ---
How many teeth are visualized in the radiograph? | 30 teeth are visualized with clear anatomical definition. | 30 | 1.0
How many teeth are visualized in the radiograph? | 30 teeth are visualized with clear anatomical definition. | 29 teeth are visualized with clear anatomical definition. | 0.0
What is the status of the wisdom teeth in the radiograph? | Three wisdom teeth are detected, all of which are impacted: #18, #28, and #48. | #18: impacted, #28: impacted, #48: erupted | 0.7
What is the condition of the teeth #26 and #14? | Teeth #26 and #14 show signs of periapical abscesses. | Teeth #26 and #23 show signs of periapical abscesses. | 0.5
What is the condition of the bone architecture and visible structures in the jaw? | No apparent bone loss is observed. Bilateral mandibular canals and maxillary sinuses are clearly visible. | Bilateral mandibular canals and maxillary sinuses are clearly visible. | 0.5
What is the clinical priority concern regarding the periapical lesions? | Periapical cysts at #11 and #12, and granuloma at #46 require endodontic evaluation. | Periapical lesions at #11, #12, and #46 require endodontic evaluation. | 0.8
What radiographic features are visible in tooth #31 on the panoramic X-ray? | [\n{\"Teeth position\": {\"point_2d\": [1242, 726]}},\n{\"Crown\": {\"box_2d\": [1220, 637, 1266, 741]}}\n] | Crown | 0.8
What radiographic features are visible in tooth #31 on the panoramic X-ray? | [\n{\"Teeth position\": {\"point_2d\": [1242, 726]}},\n{\"Crown\": {\"box_2d\": [1220, 637, 1266, 741]}}\n] | Crown at position: [1230, 627, 1276, 750] | 0.9
What radiographic features are visible in tooth #31 on the panoramic X-ray? | [\n{\"Teeth position\": {\"point_2d\": [1242, 726]}},\n{\"Crown\": {\"box_2d\": [1220, 637, 1266, 741]}}\n] | Teeth at position: {\"point_2d\": [1242, 726]}},\n{Crown at position: {\"box_2d\": [1230, 627, 1276, 750]}} | 1.0
"""
JUDGE_PROMPT_SHA256 = hashlib.sha256(_JUDGE_HEAD.encode("utf-8")).hexdigest()
JUDGE_TRIES = 5


def judge_prompt(question: str, ground_truth: str, prediction: str) -> str:
    """Official ``build_mmoral_opg_gpt4_prompt``."""
    return _JUDGE_HEAD + "\n" + " | ".join([str(question), str(ground_truth).replace("<AND>", " <AND> ").replace("<OR>", " <OR> "),
                                             str(prediction), ""])


def judge_one(complete: Callable[[str, float], str], question: str, ground_truth: str, prediction: str) -> dict[str, Any]:
    """Official ``MMOral_opg_auxeval``: up to 5 tries at temperature 0, 0.5, 1.0, ...; no valid score after 5 tries = 0.0.
    ``complete(prompt, temperature) -> text`` raises on transport errors (never swallowed here)."""
    prompt = judge_prompt(question, ground_truth, prediction)
    log = ""
    for i in range(JUDGE_TRIES):
        output = complete(prompt, i * 0.5)
        try:
            score = float(output)
        except (TypeError, ValueError):
            log += f"Try {i}: output is {output}, failed to parse.\n"
            continue
        if score < 0 or score > 1:
            log += f"Try {i}: output is {output}, invalid score: {score}.\n"
            continue
        return dict(score=score, log=log + "Succeed", tries=i + 1)
    return dict(score=0.0, log=log + f"All {JUDGE_TRIES} retries failed.\n", tries=JUDGE_TRIES)


def score_open(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """rows: dicts with ``category`` and ``score``. Mean score x 100 per main category (each question once: ``main``; with the official
    script's key weighting: ``main_official_weighting``) and per exact category."""
    tally = _Tally(OPEN_CATEGORIES, lambda cat: (cat.replace(",", "_"),))
    n = 0
    for r in rows:
        tally.add(str(r["category"]), float(r["score"]))
        n += 1
    return dict(n=n, **tally.tables())
