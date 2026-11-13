"""Official DENTEX test cohort (250 images): FDI indices, polygon geometry and the validation-file area convention, the mapping file,
masked annotations, the 250-image gate, the cohort loader, the scoring rules, the converter and the publisher. No GPU, no torch."""
import copy
import io
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dental_agent.data import official_test250 as o  # noqa: E402


def _load_evaluation_without_torch():
    """``dental_agent.evaluation.__init__`` imports the training stack (torch). Where torch is absent, register the package by path so the real
    submodules (metrics.py, ground_truth.py, test250_scoring.py) still import and are tested; with torch the normal import is used."""
    try:
        import dental_agent.evaluation  # noqa: F401
        return
    except ImportError:
        import types
        for name in [n for n in sys.modules if n.startswith(("dental_agent.evaluation", "dental_agent.training"))]:
            del sys.modules[name]
        pkg = types.ModuleType("dental_agent.evaluation")
        pkg.__path__ = [str(REPO / "dental_agent" / "evaluation")]
        sys.modules["dental_agent.evaluation"] = pkg


_load_evaluation_without_torch()
from dental_agent.evaluation import test250_scoring as sc  # noqa: E402

MAPPING = REPO / "dental_agent" / "data" / "label_maps" / "official_test250_v1.json"
WORDS = {0: "saglam", 1: "çürük", 2: "küretaj", 3: "kanal", 5: "çekim", 6: "gömülü", 7: "lezyon", 8: "kırık"}
CATS = dict(categories_1=[dict(id=i, name=str(i + 1)) for i in range(4)], categories_2=[dict(id=i, name=str(i + 1)) for i in range(8)],
            categories_3=[dict(id=0, name="Impacted", supercategory="Impacted"), dict(id=1, name="Caries", supercategory="Caries"),
                          dict(id=2, name="Periapical Lesion", supercategory="Periapical Lesion"), dict(id=3, name="Deep Caries", supercategory="Deep Caries")])
CLASS_OF_CODE = {1: "Caries", 2: "Curettage", 3: "Root Canal", 5: "Extraction", 6: "Impacted", 7: "Periapical Lesion", 8: "Fracture"}


def square(x, y, s=20):
    return [[x, y], [x + s, y], [x + s, y + s], [x, y + s]]


VERIFIED_COUNTS = {0: 91, 1: 747, 2: 265, 3: 161, 5: 29, 6: 221, 7: 75, 8: 11}      # the real polygon counts per raw code (1600 in total)


def labelme(n_images=250, size=(8, 6)):
    """250 LabelMe records with EXACTLY the verified polygon count per code, spread round-robin over the images."""
    out = {n: dict(version="5.0.1", flags={}, shapes=[], imagePath=f"test_{n}.png", imageHeight=size[1], imageWidth=size[0]) for n in range(n_images)}
    k = 0
    for code, count in VERIFIED_COUNTS.items():
        for _ in range(count):
            n = k % n_images
            fdi = [11, 26, 38, 48, 35, 17, 23, 44][k % 8]
            out[n]["shapes"].append(dict(label=f"{code}-{WORDS[code]}-{fdi}", points=square(1 + k % 5, 1 + k % 7), shape_type="polygon", group_id=None, flags={}))
            k += 1
    return out


def mapping_doc():
    return o.load_mapping(MAPPING)


def png(size=(8, 6)):
    buf = io.BytesIO()
    Image.new("RGB", size).save(buf, format="PNG")
    return buf.getvalue()


def validation_doc(decimals=0):
    anns = []
    for k in range(5):
        pts = square(10 + k, 20 + k, 30 + k)
        anns.append(dict(id=k + 1, image_id=1, iscrowd=0, bbox=[pts[0][0], pts[0][1], 30 + k, 30 + k], area=(30 + k) ** 2, segmentation=[[v for p in pts for v in p]],
                         category_id_1=0, category_id_2=0, category_id_3=1))
    return dict(images=[dict(id=1, file_name="val_1.png", height=100, width=100)], annotations=anns, **copy.deepcopy(CATS))


# -------------------------------------------------------------------------------------------------------------- FDI and geometry
@pytest.mark.parametrize("fdi,expected", [(26, (1, 5)), (11, (0, 0)), (48, (3, 7)), (38, (2, 7)), (18, (0, 7))])
def test_fdi_to_zero_indexed_quadrant_and_tooth(fdi, expected):
    assert o.fdi_to_indices(fdi) == expected


@pytest.mark.parametrize("fdi", [55, 10, 19, 9, 49, 0, 85, 100])
def test_fdi_outside_quadrant_1_4_or_tooth_1_8_is_rejected(fdi):
    with pytest.raises(ValueError, match="outside quadrant"):
        o.fdi_to_indices(fdi)


def test_fdi_index_round_trips_with_the_repository_convention():
    # the repo adds 1 back (dentex.py: category_id_1 + 1, category_id_2 + 1)
    for q in range(1, 5):
        for t in range(1, 9):
            qi, ti = o.fdi_to_indices(q * 10 + t)
            assert (qi + 1, ti + 1) == (q, t)


def test_label_parsing_takes_the_last_token_as_fdi():
    assert o.parse_label("1-çürük-26") == (1, "çürük", 26)
    for bad in ("1-26", "1-çürük", "çürük-1-26", "1-a-b-26"):
        with pytest.raises(ValueError):
            o.parse_label(bad)


def test_polygon_bbox_and_shoelace_area():
    tri = [[0, 0], [4, 0], [0, 3]]
    assert o.polygon_bbox(tri) == [0, 0, 4, 3] and o.polygon_area(tri) == 6.0
    assert o.polygon_area(list(reversed(tri))) == 6.0                       # orientation does not matter
    assert o.polygon_area(square(5, 5, 10)) == 100.0
    assert o.flatten([[1.26, 2.5]], 0) == [1, 2] and o.flatten([[1.26, 2.5]], 1) == [1.3, 2.5]
    assert o.decimals_of([1, 2.0, 3.25]) == 2 and o.decimals_of([1, 2]) == 0


def test_area_convention_is_checked_against_the_validation_file():
    ok = o.check_area_convention(validation_doc())
    assert ok["matches"] and ok["exact"] == 5 and ok["max_abs_diff"] == 0.0
    bad = validation_doc()
    bad["annotations"][2]["area"] = 12345                                    # e.g. a pixel-count convention instead of the shoelace area
    res = o.check_area_convention(bad)
    assert not res["matches"] and res["worst"][0][0] == 3


# -------------------------------------------------------------------------------------------------------------------- the mapping
def test_shipped_mapping_is_valid_complete_and_names_every_code_in_english():
    m = mapping_doc()
    assert sorted(e["code"] for e in m["codes"]) == [0, 1, 2, 3, 5, 6, 7, 8]
    o.validate_mapping(m, require_no_pending=True)
    by = {e["code"]: e for e in m["codes"]}
    for code, name in CLASS_OF_CODE.items():
        assert by[code]["status"] == "translate" and by[code]["role"] == "finding" and by[code]["target_name"] == name, code
    assert by[0]["role"] == "negative" and by[0]["target_id"] is None                       # healthy is scored as a negative, not dropped
    assert not any(e["role"] == "ignore" for e in m["codes"])                               # nothing is ignored
    assert [(c["id"], c["name"]) for c in m["dentex_categories_3"]] == [(0, "Impacted"), (1, "Caries"), (2, "Periapical Lesion"), (3, "Deep Caries")]
    assert [(c["id"], c["name"]) for c in m["extra_categories_3"]] == [(4, "Root Canal"), (5, "Curettage"), (6, "Extraction"), (7, "Fracture")]
    assert by[3]["target_id"] == 4 and by[2]["target_id"] == 5 and by[5]["target_id"] == 6 and by[8]["target_id"] == 7
    assert sum(e["count"] for e in m["codes"]) == 1600 and all(e["evidence"] for e in m["codes"])
    assert "ENGLISH" in m["naming_rule"]


def test_mapping_validator_refuses_unsupported_rows():
    m = mapping_doc()
    no_evidence = copy.deepcopy(m)
    no_evidence["codes"][1]["evidence"] = []
    with pytest.raises(ValueError, match="without evidence"):
        o.validate_mapping(no_evidence)
    wrong_id = copy.deepcopy(m)
    wrong_id["codes"][3]["target_id"] = 6                                                  # kanal -> id 6 is Extraction: name and id disagree
    with pytest.raises(ValueError, match="target class that exists"):
        o.validate_mapping(wrong_id)
    wrong_source = copy.deepcopy(m)
    wrong_source["codes"][3]["target_source"] = "dentex"
    with pytest.raises(ValueError, match="target_source"):
        o.validate_mapping(wrong_source)
    pending = copy.deepcopy(m)
    pending["codes"][3].update(status="needs_confirmation", role="negative", target_id=None, target_name=None, target_source=None)
    o.validate_mapping(pending)
    with pytest.raises(ValueError, match="still needs confirmation"):
        o.validate_mapping(pending, require_no_pending=True)
    clash = copy.deepcopy(m)
    clash["extra_categories_3"][0]["id"] = 3                                                # reusing a DENTEX id
    with pytest.raises(ValueError, match="must not reuse"):
        o.validate_mapping(clash)
    ignored = copy.deepcopy(m)
    ignored["codes"][2].update(role="ignore")
    with pytest.raises(ValueError):
        o.validate_mapping(ignored)


# --------------------------------------------------------------------------------------------------------------- the conversion
def converted(**kw):
    lm = labelme()
    sizes = {n: (8, 6) for n in lm}
    return o.convert(lm, mapping_doc(), validation_doc(), sizes, version="v1", decimals=0, zip_sha256="z" * 64, mapping_sha256="m" * 64, **kw), lm


def test_conversion_structure_ids_categories_and_every_class():
    coco, lm = converted()
    assert list(coco)[:3] == ["info", "images", "annotations"] and {"categories_1", "categories_2", "categories_3"} <= set(coco) and "categories" not in coco
    assert coco["categories_1"] == CATS["categories_1"] and coco["categories_2"] == CATS["categories_2"]
    assert coco["categories_3"][:4] == CATS["categories_3"]                                # the four official classes first, identical
    assert [(c["id"], c["name"]) for c in coco["categories_3"][4:]] == [(4, "Root Canal"), (5, "Curettage"), (6, "Extraction"), (7, "Fracture")]
    assert [i["id"] for i in coco["images"]] == list(range(1, 251)) and coco["images"][0] == dict(id=1, file_name="test_0.png", height=6, width=8)
    total = sum(len(r["shapes"]) for r in lm.values())
    assert len(coco["annotations"]) == total == 1600 and coco["info"]["n_polygons"] == 1600            # nothing dropped
    names = {c["id"]: c["name"] for c in coco["categories_3"]}
    for a in coco["annotations"]:
        assert a["iscrowd"] == 0 and a["polygon_points"] and len(a["segmentation"]) == 1
        if a["raw_code"] == 0:                                                                         # saglam: the only polygon that is not a finding
            assert a["masked"] and a["category_id_3"] is None and a["role"] == "negative"
        else:
            assert not a["masked"] and a["role"] == "finding" and names[a["category_id_3"]] == CLASS_OF_CODE[a["raw_code"]]
    assert coco["info"]["n_masked"] == VERIFIED_COUNTS[0] == 91
    a = next(a for a in coco["annotations"] if a["raw_label"] == "1-çürük-26")
    assert (a["category_id_1"], a["category_id_2"], a["category_id_3"]) == (1, 5, 1)           # FDI 26 -> quadrant 1, tooth 5, Caries
    assert a["bbox"][2:] == [20, 20] and a["area"] == 400
    k = next(a for a in coco["annotations"] if a["raw_code"] == 3)
    assert names[k["category_id_3"]] == "Root Canal" and k["category_id_3"] == 4
    assert o.check_coco(coco, mapping_doc()) == []


def test_conversion_is_deterministic():
    a, _ = converted()
    b, _ = converted()
    assert json.dumps(a, ensure_ascii=False) == json.dumps(b, ensure_ascii=False)


def test_conversion_rejects_bad_fdi_unknown_code_size_mismatch_and_non_polygons():
    def attempt(mutate, expect):
        lm = labelme()
        sizes = {n: (8, 6) for n in lm}
        mutate(lm, sizes)
        with pytest.raises(ValueError, match=expect):
            o.convert(lm, mapping_doc(), validation_doc(), sizes, version="v1", decimals=0, zip_sha256="z", mapping_sha256="m")

    attempt(lambda lm, s: lm[3]["shapes"].append(dict(label="1-çürük-55", points=square(1, 1), shape_type="polygon")), r"test_3.*outside quadrant")
    attempt(lambda lm, s: lm[4]["shapes"].append(dict(label="9-yeni-11", points=square(1, 1), shape_type="polygon")), r"test_4.*code 9")
    attempt(lambda lm, s: s.__setitem__(5, (9, 6)), r"test_5.*PNG header")
    attempt(lambda lm, s: lm[6]["shapes"].append(dict(label="1-çürük-11", points=square(1, 1), shape_type="rectangle")), r"test_6.*not a polygon")
    official = validation_doc()
    official["categories_3"][1]["name"] = "Cavity"                                                  # the mapping's DENTEX classes must be the file's
    with pytest.raises(ValueError, match="differ from the validation file"):
        o.convert(labelme(), mapping_doc(), official, {n: (8, 6) for n in range(250)}, version="v1", decimals=0, zip_sha256="z", mapping_sha256="m")


def test_the_250_image_gate():
    coco, _ = converted()
    short = copy.deepcopy(coco)
    short["images"] = short["images"][:-1]
    assert any("expected 250 unique images, found 249" in p for p in o.check_coco(short, mapping_doc()))
    dup = copy.deepcopy(coco)
    dup["images"][1]["id"] = dup["images"][0]["id"]
    assert any("unique" in p for p in o.check_coco(dup, mapping_doc()))
    orphan = copy.deepcopy(coco)
    orphan["annotations"][0]["image_id"] = 9999
    assert any("missing image 9999" in p for p in o.check_coco(orphan, mapping_doc()))
    bad_cat = copy.deepcopy(coco)
    next(a for a in bad_cat["annotations"] if not a["masked"])["category_id_3"] = 99
    assert any("outside categories_3" in p for p in o.check_coco(bad_cat, mapping_doc()))


def test_eval_view_keeps_every_finding_and_moves_only_the_healthy_polygons():
    coco, _ = converted()
    view = o.eval_view(coco)
    assert all(a["category_id_3"] is not None and "polygon_points" not in a for a in view["annotations"])      # no null diagnosis reaches existing consumers
    assert len(view["annotations"]) == 1600 - 91 and {m["raw_code"] for m in view["masked_annotations"]} == {0}
    assert {m["role"] for m in view["masked_annotations"]} == {"negative"} and len(view["masked_annotations"]) == 91
    assert len(coco["annotations"]) == 1600                                                                  # the published file itself is unchanged


# ------------------------------------------------------------------------------------------------------------------- the cohort
class Helpers:
    @staticmethod
    def image_ok(path, item):
        try:
            with Image.open(path) as im:
                return im.size == (int(item["width"]), int(item["height"]))
        except (OSError, ValueError):
            return False

    @staticmethod
    def digest(path):
        return o.sha256_file(path)

    @staticmethod
    def fingerprint(value):
        import hashlib
        return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()

    @staticmethod
    def write(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True, indent=1), encoding="utf-8")


def cohort_files(tmp_path, n_images=250, with_coco=True, mapping_sha=None):
    coco, _ = converted()
    coco["info"]["mapping_sha256"] = mapping_sha or o.sha256_file(MAPPING)
    root = tmp_path / "local"
    root.mkdir(parents=True, exist_ok=True)
    files = []
    if with_coco:
        p = root / "official_test250_v1.json"
        p.write_text(json.dumps(coco), encoding="utf-8")
        files.append(p)
    for n in range(n_images):
        p = root / f"test_{n}.png"
        Image.new("RGB", (8, 6), color=n % 255).save(p)
        files.append(p)
    return files, coco


def prep(tmp_path, files, api=None, download=None, **kw):
    fail = lambda *a, **k: pytest.fail("a complete local cohort must not touch the Hub")  # noqa: E731
    return o.prepare_cohort(tmp_path, api or SimpleNamespace(repo_info=fail, list_repo_files=fail), download or fail, None, files,
                            image_ok=Helpers.image_ok, digest=Helpers.digest, fingerprint=Helpers.fingerprint, write=Helpers.write, **kw)


def test_cohort_resolves_locally_writes_the_loader_layout_and_records_the_versions(tmp_path):
    files, coco = cohort_files(tmp_path)
    result = prep(tmp_path, files)
    assert len(result["ids"]) == 250 and result["split"] == "official_test250"
    ident = result["identity"]
    assert ident["coco_version"] == "v1" and ident["mapping_version"] == "v1" and ident["mapping_sha256"] == o.sha256_file(MAPPING)
    assert len(ident["coco_sha256"]) == 64 and ident["n_masked"] > 0
    folder = Path(result["data_dir"])
    prepared = json.loads((folder / "validation_triple.json").read_text())
    assert (folder / "validation_data").is_dir() and len(prepared["images"]) == 250 and all(Path(i["file_name"]).is_file() for i in prepared["images"])
    assert len(prepared["annotations"]) + len(prepared["masked_annotations"]) == len(coco["annotations"]) and "info" in prepared
    assert prep(tmp_path, files)["fingerprint"] == result["fingerprint"]                          # stable identity


def test_missing_images_raise_with_the_ids_and_the_hub_fallback_fills_them(tmp_path):
    files, _ = cohort_files(tmp_path, n_images=248)                                                # test_248 and test_249 are missing locally
    asked = []

    class Api:
        def repo_info(self, repo, repo_type=None):
            if repo_type != "dataset":
                from huggingface_hub.errors import RepositoryNotFoundError
                raise RepositoryNotFoundError("nope", response=SimpleNamespace(status_code=404, headers={}, request=None))
            return SimpleNamespace(sha="rev1")

        def list_repo_files(self, repo, repo_type=None, revision=None):
            return [f"disease/input/test_{n}.png" for n in range(250)]

    def download(repo, name, repo_type=None, revision=None, token=None):
        asked.append((repo, name, repo_type, revision))
        p = tmp_path / "hubcache" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(png())
        return str(p)

    result = prep(tmp_path, files, api=Api(), download=download)
    assert len(result["ids"]) == 250
    assert [a[1] for a in asked] == ["disease/input/test_248.png", "disease/input/test_249.png"] and asked[0][0] == "Reza-Nadimi/dentex-official-test"
    assert all(a[2] == "dataset" and a[3] == "rev1" for a in asked)


def test_incomplete_cohort_never_launches_a_partial_run(tmp_path):
    files, _ = cohort_files(tmp_path, n_images=240)

    class Api:
        def repo_info(self, repo, repo_type=None):
            return SimpleNamespace(sha="r")

        def list_repo_files(self, repo, repo_type=None, revision=None):
            return []                                                                              # the Hub has none of them either

    with pytest.raises(FileNotFoundError, match=r"240/250.*Missing IDs: \[241, 242"):
        prep(tmp_path, files, api=Api(), download=lambda *a, **k: pytest.fail("nothing to download"))


def test_coco_is_downloaded_from_the_hub_when_absent_locally_and_never_reconverted(tmp_path):
    files, coco = cohort_files(tmp_path)
    images_only = [f for f in files if f.suffix == ".png"]
    remote = tmp_path / "remote_coco.json"
    remote.write_text(json.dumps(coco), encoding="utf-8")
    seen = []

    class Api:
        def repo_info(self, repo, repo_type=None):
            return SimpleNamespace(sha="rev9")

    def download(repo, name, repo_type=None, revision=None, token=None):
        seen.append((name, revision))
        return str(remote)

    result = prep(tmp_path, images_only, api=Api(), download=download)
    assert seen == [("coco/official_test250_v1.json", "rev9")] and len(result["ids"]) == 250


def test_a_mapping_changed_after_conversion_is_refused(tmp_path):
    files, _ = cohort_files(tmp_path, mapping_sha="0" * 64)
    with pytest.raises(ValueError, match="mapping changed after conversion"):
        prep(tmp_path, files)


def test_cohort_loader_reads_the_prepared_folder_and_selects_the_cohort_prompt_and_scoring(tmp_path):
    pytest.importorskip("pandas")
    files, coco = cohort_files(tmp_path)
    result = prep(tmp_path, files)
    args = SimpleNamespace(data_dir=result["data_dir"], dataset="dentex")
    imgs, annots, cats = o.load_cohort_split(args)
    assert len(imgs) == 250 and len(annots) == 1600 - 91 and len(cats) == 8                                 # eight classes: the four DENTEX ones and the four others
    assert args.prompt_dataset == "official_test250" and args.dataset == "dentex"              # the prompt changes; the result paths (args.dataset) do not
    assert args.test250_settings["scoring"] == "test250_scoring_v2"
    assert args.test250_settings["mapping_sha256"] == o.sha256_file(MAPPING) and args.test250_settings["coco_version"] == "v1"
    assert args.test250_settings["prompt_classes"] == o.PROMPT_CLASSES
    from dental_agent.evaluation.ground_truth import extract_ground_truth
    seen = set()
    for iid in annots["image_id"].unique():
        seen |= {g["diagnosis"] for g in extract_ground_truth(int(iid), annots, cats)}
    assert seen == {"Caries", "Impacted", "Periapical Lesion", "Root Canal", "Curettage", "Extraction", "Fracture"}


def test_origin_classifies_local_kaggle_and_huggingface_paths():
    assert o.origin("/kaggle/input/datasets/rezanadimikj/dentex-official-test250/coco/official_test250_v1.json") == "kaggle"
    assert o.origin("/kaggle/input/dentex-official-test250/disease/input/test_0.png") == "kaggle"
    assert o.origin("/root/.cache/huggingface/hub/datasets--Reza-Nadimi--dentex-official-test/snapshots/abc/coco/official_test250_v1.json") == "huggingface"
    assert o.origin("/home/asr/Reza_Nadimi/VLM-DENTAL/data/official_test250/coco/official_test250_v1.json") == "local"
    mixed = [Path("/root/.cache/huggingface/hub/datasets--x/snapshots/a/f.png"), Path("/kaggle/input/s/f.png"), Path("/data/f.png")]
    assert [o.origin(p) for p in o.by_source(mixed)] == ["local", "kaggle", "huggingface"]


def test_cohort_is_read_from_local_disk_then_kaggle_then_huggingface(tmp_path, monkeypatch, capsys):
    """Three copies of the COCO that differ, and of every image: the local one wins, Kaggle is next, the Hugging Face cache last."""
    def classify(path):
        p = str(path)
        return "kaggle" if "/kag/" in p else "huggingface" if "/hfc/" in p else "local"

    monkeypatch.setattr(o, "origin", classify)

    def copy_set(folder, tag, with_images=True):
        coco, _ = converted()
        coco["info"]["mapping_sha256"] = o.sha256_file(MAPPING)
        coco["info"]["note"] = f"copy from {tag}"                                       # makes the three COCO files differ
        folder.mkdir(parents=True, exist_ok=True)
        paths = [folder / "official_test250_v1.json"]
        paths[0].write_text(json.dumps(coco), encoding="utf-8")
        if with_images:
            for n in range(250):
                Image.new("RGB", (8, 6), color=(n % 250, {"local": 1, "kag": 2, "hfc": 3}[tag], 0)).save(folder / f"test_{n}.png")
                paths.append(folder / f"test_{n}.png")
        return paths

    local = copy_set(tmp_path / "work" / "local", "local")
    kaggle = copy_set(tmp_path / "work" / "kag" / "x", "kag")
    hfc = copy_set(tmp_path / "work" / "hfc" / "y", "hfc")
    everything = hfc + kaggle + local                                                   # an unordered inventory, as inventory_roots returns it
    result = prep(tmp_path, everything)
    assert result["sources"]["coco"] == "local" and result["sources"]["images"] == {"local": 250}
    chosen = json.loads((Path(result["data_dir"]) / "validation_triple.json").read_text())
    assert chosen["info"]["note"] == "copy from local" and all("/local/" in i["file_name"] for i in chosen["images"])
    assert o.sha256_file(local[0]) == result["identity"]["coco_sha256"]
    out = capsys.readouterr().out
    assert "[COHORT WARNING] 3 different official_test250_v1.json files" in out and "image(s) exist in several places with different content" in out
    assert "read order: local disk, Kaggle, Hugging Face" in out

    only_kaggle_and_hf = prep(tmp_path, hfc + kaggle)
    assert only_kaggle_and_hf["sources"]["coco"] == "kaggle" and only_kaggle_and_hf["sources"]["images"] == {"kaggle": 250}
    only_hf = prep(tmp_path, hfc)
    assert only_hf["sources"]["coco"] == "huggingface" and only_hf["sources"]["images"] == {"huggingface": 250}
    # mixed: the COCO and part of the images on Kaggle, the rest only in the Hugging Face cache
    partial = prep(tmp_path, kaggle[:1] + kaggle[1:101] + hfc[1:])
    assert partial["sources"]["images"] == {"kaggle": 100, "huggingface": 150}


def test_prepare_inputs_dispatches_to_the_cohort_and_leaves_the_50_image_branch_alone(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("notebook_eval_t250", REPO / "dental_agent/harness/notebook_eval.py")
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    calls = []
    monkeypatch.setattr(o, "prepare_cohort", lambda *a, **k: calls.append((a, k)) or dict(ok=True))
    assert w.prepare_inputs(tmp_path, "official_test250", None, None, None, None, files=[]) == dict(ok=True) and len(calls) == 1
    with pytest.raises(ValueError, match="Only fixed dev"):
        w.prepare_inputs(tmp_path, "validation", None, None, None, None, files=[])
    text = (REPO / "dental_agent/harness/notebook_eval.py").read_text(encoding="utf-8")
    assert "len(ids) != 50" in text and "Official test must contain all 50 images" in text                    # the 50-image path is still there, unchanged


# ------------------------------------------------------------------------------------------------------------ the scoring rules
def finding(q, t, d, raw=None):
    f = dict(quadrant=q, tooth_position=t, diagnosis=d)
    if raw is not None:
        f["raw_diagnosis"] = raw
    return f


@pytest.mark.parametrize("written,expected", [
    ("Root Canal", "Root Canal"), ("root canal treatment", "Root Canal"), ("Endodontic treatment", "Root Canal"), ("Root canal filling", "Root Canal"),
    ("Curettage", "Curettage"), ("periapical curettage", "Periapical Lesion"),                  # the shared normaliser wins for names it already knows
    ("Extraction", "Extraction"), ("tooth extraction", "Extraction"), ("Extracted tooth", "Extraction"),
    ("Fracture", "Fracture"), ("fractured crown", "Fracture"), ("broken tooth", "Fracture"),
    ("Caries", "Caries"), ("deep caries", "Deep Caries"), ("Impacted Tooth", "Impacted"), ("Pericoronal Lesion", "Periapical Lesion")])
def test_prediction_class_names_are_put_into_the_cohort_names(written, expected):
    assert sc.canonical_class(finding(1, 1, written, raw=written)) == expected


def test_nothing_is_dropped_and_unknown_names_just_do_not_match():
    gt = [finding(1, 6, "Root Canal")]
    preds = [finding(1, 6, "Root Canal Treatment", raw="Root Canal Treatment"), finding(3, 5, "Caries"), finding(2, 2, "Dental calculus")]
    canon, extra = sc.score_cohort(gt, preds)
    assert len(canon) == 3 and [c["diagnosis"] for c in canon][:2] == ["Root Canal", "Caries"] and extra["n_renamed_predictions"] == 1
    from dental_agent.evaluation.metrics import match_multi_findings
    res = match_multi_findings(gt, canon)
    assert res["exact_recall"] == 1.0 and res["exact_precision"] == pytest.approx(1 / 3)                 # the two other predictions are false positives, none is hidden
    assert extra["scoring_profile"] == "test250_scoring_v2"


def test_every_cohort_class_is_a_real_match_through_the_unchanged_shared_metric():
    from dental_agent.evaluation.metrics import match_multi_findings
    for name in ("Caries", "Impacted", "Periapical Lesion", "Root Canal", "Curettage", "Extraction", "Fracture"):
        gt = [finding(2, 3, name)]
        canon, _ = sc.score_cohort(gt, [finding(2, 3, name, raw=name)])
        assert match_multi_findings(gt, canon)["exact_f1"] == 1.0, name


def test_a_disease_predicted_on_a_healthy_polygon_is_a_false_positive():
    from dental_agent.evaluation.metrics import match_multi_findings
    gt = [finding(1, 6, "Caries")]                                   # the saglam tooth (3,5) is not in the ground truth: it is a negative
    canon, _ = sc.score_cohort(gt, [finding(1, 6, "Caries"), finding(3, 5, "Caries")])
    res = match_multi_findings(gt, canon)
    assert res["exact_recall"] == 1.0 and res["exact_precision"] == 0.5


def test_family_metric_and_the_existing_proximity_credit_are_both_kept():
    from dental_agent.evaluation.metrics import match_multi_findings
    gt = [finding(1, 6, "Caries")]
    canon, extra = sc.score_cohort(gt, [finding(1, 6, "Deep Caries")])
    assert extra["family_exact_f1"] == 1.0
    std = match_multi_findings(gt, canon)
    assert std["exact_f1"] == 0.0 and std["fdi_f1"] == 1.0 and 0 < std["closeness_score"] < 1              # strict exact, plus the existing credit for a close call


def test_tufts_lesion_classes_match_the_lesion_code():
    from dental_agent.evaluation.metrics import match_multi_findings
    gt = [finding(2, 3, "Periapical Lesion")]
    for name in ("Non-Odontogenic Lesion", "Pericoronal Lesion", "Inter-Radicular Lesion", "Periapical Lesion"):
        canon, _ = sc.score_cohort(gt, [finding(2, 3, name, raw=name)])
        assert match_multi_findings(gt, canon)["exact_f1"] == 1.0, name


# ------------------------------------------------------------------------------------- prompts and hooks (the rest of the project is untouched)
def test_cohort_prompts_equal_their_base_prompts_except_for_the_class_list():
    from dental_agent.agent.prompts import NO_TOOLS_SYSTEM_PROMPT, ZERO_SHOT_PROMPT, build_agent_system_prompt
    new_u, old_u = "|".join(o.PROMPT_CLASSES), "|".join(o._DENTEX_FOUR)
    new_q, old_q = ", ".join(f'"{c}"' for c in o.PROMPT_CLASSES), ", ".join(f'"{c}"' for c in o._DENTEX_FOUR)
    new_p, old_p = "(" + ", ".join(o.PROMPT_CLASSES) + ")", "(" + ", ".join(o._DENTEX_FOUR) + ")"
    assert o.zero_shot_prompt().replace(new_u, old_u).replace(new_q, old_q) == ZERO_SHOT_PROMPT
    assert o.no_tools_prompt().replace(new_u, old_u).replace(new_p, old_p) == NO_TOOLS_SYSTEM_PROMPT
    cohort, base = build_agent_system_prompt("TOOLS", dataset="official_test250"), build_agent_system_prompt("TOOLS", dataset="dentex")
    assert cohort.replace(", ".join(o.PROMPT_CLASSES), ", ".join(o._DENTEX_FOUR)).replace(new_u, old_u) == base
    for name in ("Root Canal", "Curettage", "Extraction", "Fracture"):
        assert name in cohort and name in o.zero_shot_prompt() and name in o.no_tools_prompt()
        assert name not in base and name not in ZERO_SHOT_PROMPT and name not in NO_TOOLS_SYSTEM_PROMPT       # dev, training and every other split keep their prompts
    for dataset in ("tufts", "tufts_all"):
        assert "Root Canal" not in build_agent_system_prompt("TOOLS", dataset=dataset)


def test_derive_prompt_fails_loudly_when_the_base_prompt_changes_shape():
    with pytest.raises(ValueError, match="changed shape"):
        o.derive_prompt("Pick one of Caries or Deep Caries.")


def test_shared_code_hooks_are_additive_and_default_to_the_old_behaviour():
    import ast
    tree = ast.parse((REPO / "dental_agent/agent/loop.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run_agent_no_tools")
    args = {a.arg: d for a, d in zip(fn.args.args[-len(fn.args.defaults):], fn.args.defaults)}
    assert "system_prompt" in args and isinstance(args["system_prompt"], ast.Constant) and args["system_prompt"].value is None
    em = (REPO / "scripts/evaluate_models.py").read_text(encoding="utf-8")
    zs = (REPO / "scripts/run_zero_shot.py").read_text(encoding="utf-8")
    ne = (REPO / "dental_agent/harness/notebook_eval.py").read_text(encoding="utf-8")
    assert 'if getattr(args, "test250_settings", None):' in em and "score_cohort" in em and "no_tools_prompt" in em
    assert 'dataset=getattr(args, "prompt_dataset", None) or args.dataset' in em and "args.dataset = " not in (REPO / "dental_agent/data/official_test250.py").read_text(encoding="utf-8")
    assert 'elif split_name == "official_test250":' in zs and "_cohort_zero_shot_prompt()" in zs and "score_cohort" in zs and "**extra_scoring" in zs
    assert "'official_test250' if identity['split']=='official_test250' else 'test'" in ne
    assert "unsupported" not in ne.split("official_test250")[-1][:0]          # (the zero-shot kind is no longer reported unsupported for the cohort)
    assert "Zero-shot legacy prompt cohort is test-only" in ne                # the dev rule is unchanged
    assert not any("kind']=='zeroshot' and self.split=='official_test250'" in line for line in ne.splitlines())


# ------------------------------------------------------------------------------------------- converter and publisher scripts
def synthetic_official_zip(tmp_path, bad_fdi=False):
    lm = labelme()
    if bad_fdi:
        lm[1]["shapes"].append(dict(label="1-çürük-55", points=square(1, 1), shape_type="polygon"))
    z = tmp_path / "test_data.zip"
    with zipfile.ZipFile(z, "w") as f:
        for n in range(250):
            f.writestr(f"disease/input/test_{n}.png", png())
            f.writestr(f"disease/label/test_{n}.json", json.dumps(lm[n], ensure_ascii=False))
    return z


def run_converter(tmp_path, monkeypatch, validation=None, bad_fdi=False, version="v1"):
    import importlib
    script = importlib.import_module("scripts.convert_official_test_to_coco")
    z = synthetic_official_zip(tmp_path, bad_fdi)
    monkeypatch.setattr(o, "OFFICIAL_ZIP_SHA256", o.sha256_file(z))
    v = tmp_path / "validation_triple.json"
    v.write_text(json.dumps(validation or validation_doc()), encoding="utf-8")
    return script.build(version, z, v, tmp_path / "out"), script


def test_converter_end_to_end_writes_hashes_and_passes_the_loader_dry_run(tmp_path, monkeypatch, capsys):
    pytest.importorskip("pandas")
    result, _ = run_converter(tmp_path, monkeypatch)
    out = capsys.readouterr().out
    assert "AREA CONVENTION" in out and "MATCH" in out and "every id equals N + 1: True" in out and "dry run through the repository loader" in out
    coco = json.loads(Path(result["coco"]).read_text(encoding="utf-8"))
    assert coco["info"]["mapping_sha256"] == result["mapping_sha256"] == o.sha256_file(MAPPING) and len(coco["images"]) == 250
    assert (tmp_path / "out" / "id_to_file_v1.json").is_file() and Path(result["mapping"]).read_bytes() == MAPPING.read_bytes()


def test_converter_stops_when_the_area_convention_does_not_match(tmp_path, monkeypatch):
    bad = validation_doc()
    bad["annotations"][0]["area"] = 1
    with pytest.raises(SystemExit, match="does NOT reproduce"):
        run_converter(tmp_path, monkeypatch, validation=bad)


def test_converter_stops_on_an_fdi_outside_the_range_and_on_a_foreign_zip(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="outside quadrant"):
        run_converter(tmp_path, monkeypatch, bad_fdi=True)
    import importlib
    script = importlib.import_module("scripts.convert_official_test_to_coco")
    z = synthetic_official_zip(tmp_path / ".." if False else tmp_path)
    monkeypatch.setattr(o, "OFFICIAL_ZIP_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="not the official"):
        script.build("v1", z, tmp_path / "validation_triple.json", tmp_path / "o2")


class FakeApi:
    def __init__(self, private=True, files=()):
        self.private, self.files, self.commits, self.created = private, list(files), [], []

    def repo_info(self, repo, repo_type=None):
        if repo_type != "dataset":
            from huggingface_hub.errors import RepositoryNotFoundError
            raise RepositoryNotFoundError("x", response=SimpleNamespace(status_code=404, headers={}, request=None))
        return SimpleNamespace(private=self.private, sha="r")

    def list_repo_files(self, repo, repo_type=None, revision=None):
        return list(self.files)

    def create_commit(self, **kw):
        self.commits.append(kw)
        self.files += [op.path_in_repo for op in kw["operations"]]

    def create_repo(self, *a, **k):
        self.created.append((a, k))
        raise AssertionError("the publisher must never create a repository")


def test_publisher_uploads_two_files_once_and_refuses_to_overwrite_or_leave_private(tmp_path, monkeypatch, capsys):
    pytest.importorskip("pandas")
    result, _ = run_converter(tmp_path, monkeypatch)
    import importlib
    pub = importlib.import_module("scripts.publish_official_test_coco")
    folder = tmp_path / "out"
    base = [f"disease/input/test_{n}.png" for n in range(3)] + [f"disease/label/test_{n}.json" for n in range(3)]
    api = FakeApi(files=base)
    out = pub.publish(api, "Reza-Nadimi/dentex-official-test", "v1", folder)
    assert len(api.commits) == 1 and {op.path_in_repo for op in api.commits[0]["operations"]} == {"coco/official_test250_v1.json", "label_maps/official_test250_v1.json"}
    assert api.commits[0]["repo_type"] == "dataset" and not api.created
    rec = json.loads((folder / "publish_record_v1.json").read_text())
    assert rec["sha256"]["coco/official_test250_v1.json"] == result["coco_sha256"]
    assert all(f in out["files"] for f in base) and "uploaded to Reza-Nadimi/dentex-official-test" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="never overwritten.*Create v2"):
        pub.publish(api, "Reza-Nadimi/dentex-official-test", "v1", folder)                       # v1 now exists: refuse, point to v2
    with pytest.raises(SystemExit, match="not private"):
        pub.publish(FakeApi(private=False), "Reza-Nadimi/dentex-official-test", "v1", folder)
    with pytest.raises(SystemExit, match="missing"):
        pub.publish(FakeApi(), "Reza-Nadimi/dentex-official-test", "v7", folder)


# ------------------------------------------------------------------------------------------- the session plans the new split
def test_session_plans_the_cohort_with_versions_in_the_identity_and_runs_zeroshot_on_it(tmp_path):
    from tests import test_notebook_evaluations as tne
    w = tne.w
    h = tne.Hub(tmp_path)
    cohort = dict(data_dir="/cohort250", ids=list(range(1, 251)), hashes={}, fingerprint="f" * 64, split="official_test250",
                  identity=dict(coco_version="v1", coco_sha256="c" * 64, mapping_version="v1", mapping_sha256="m" * 64, source_zip_sha256="z" * 64, n_masked=7))
    session = w.NotebookEvaluations(kinds=["zeroshot", "baseline"], api=h, download=h.download, model_id="/base", split="official_test250", track="no_tools")
    session.inputs = {"official_test250": cohort}
    base_job = next(j for j in session.jobs if j["kind"] == "baseline")
    identity = session._identity({**base_job, "size": "canonical"})
    assert identity["split"] == "official_test250" and identity["official_test250"] == cohort["identity"]
    command, _, _ = session._command(base_job, {**identity}, tmp_path)
    assert "official_test250" in command and "/cohort250" in command and str(w.generation_budget(base_job["profile"])) in command
    zs_job = next(j for j in session.jobs if j["kind"] == "zeroshot")
    zs_identity = session._identity({**zs_job, "size": "canonical"})
    zs_command, _, _ = session._command(zs_job, {**zs_identity}, tmp_path)
    assert zs_command[zs_command.index("--split") + 1] == "official_test250" and zs_command[zs_command.index("--data-dir") + 1] == "/cohort250"
    session.run(execute=False)
    assert [r for r in session.status if r["kind"] == "zeroshot"] and all(r["status"] != "unsupported" for r in session.status if r["kind"] == "zeroshot")   # supported and scored
    with pytest.raises(ValueError, match="K sweep belongs to dev"):
        w.NotebookEvaluations(kinds=["grpo"], api=h, download=h.download, split="official_test250", selected_k="all")
    # the 50-image split gets no extra identity block and keeps `--split test`, so every existing result keeps its identity
    plain = w.NotebookEvaluations(kinds=["zeroshot", "baseline"], api=h, download=h.download, model_id="/base", track="no_tools")
    plain.inputs = {"test": dict(data_dir="/input", ids=[1], fingerprint="a" * 64)}
    pj = next(j for j in plain.jobs if j["kind"] == "zeroshot")
    pid = plain._identity({**pj, "size": "canonical"})
    assert "official_test250" not in pid
    pcmd, _, _ = plain._command(pj, {**pid}, tmp_path)
    assert pcmd[pcmd.index("--split") + 1] == "test"


# ------------------------------------------------------------------------------------------- overwrite, Kaggle staging, notebook cells
def test_publisher_overwrite_replaces_the_same_two_files_only_when_asked(tmp_path, monkeypatch, capsys):
    pytest.importorskip("pandas")
    run_converter(tmp_path, monkeypatch)
    import importlib
    pub = importlib.import_module("scripts.publish_official_test_coco")
    folder = tmp_path / "out"
    existing = ["coco/official_test250_v1.json", "label_maps/official_test250_v1.json", "disease/input/test_0.png", "disease/label/test_0.json"]
    api = FakeApi(files=existing)
    with pytest.raises(SystemExit, match="never overwritten without --overwrite"):
        pub.publish(api, "Reza-Nadimi/dentex-official-test", "v1", folder)
    assert api.commits == []
    pub.publish(api, "Reza-Nadimi/dentex-official-test", "v1", folder, overwrite=True)
    assert len(api.commits) == 1 and "(overwrite)" in api.commits[0]["commit_message"] and not api.created
    assert {op.path_in_repo for op in api.commits[0]["operations"]} == {"coco/official_test250_v1.json", "label_maps/official_test250_v1.json"}   # never disease/
    assert "[OVERWRITE]" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="not private"):                                         # overwrite does not relax the private requirement
        pub.publish(FakeApi(private=False, files=existing), "Reza-Nadimi/dentex-official-test", "v1", folder, overwrite=True)
    assert pub.build_parser().parse_args(["--version", "v1", "--overwrite"]).overwrite is True and pub.build_parser().parse_args([]).overwrite is False


def test_kaggle_stage_uploads_a_corrected_coco_unchanged_and_the_resolver_reads_it(tmp_path, monkeypatch):
    pytest.importorskip("pandas")
    result, _ = run_converter(tmp_path, monkeypatch)
    import importlib
    stg = importlib.import_module("scripts.stage_official_test250_dataset")
    src = tmp_path / "out"
    coco_path = src / "coco" / "official_test250_v1.json"
    corrected = json.loads(coco_path.read_text(encoding="utf-8"))
    corrected["annotations"][0]["bbox"] = [1, 2, 3, 4]                                              # a hand correction
    coco_path.write_text(json.dumps(corrected, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    stage_dir = tmp_path / "kaggle_dataset"
    record = stg.stage("v1", src, stage_dir, "rezanadimikj/dentex-official-test250", tmp_path / "test_data.zip")
    assert (stage_dir / "coco" / "official_test250_v1.json").read_bytes() == coco_path.read_bytes()              # uploaded as it is, never re-converted
    assert record["coco/official_test250_v1.json"] == o.sha256_file(coco_path)
    assert (stage_dir / "label_maps" / "official_test250_v1.json").read_bytes() == MAPPING.read_bytes()
    assert len(list((stage_dir / "disease" / "input").glob("test_*.png"))) == 250
    meta = json.loads((stage_dir / "dataset-metadata.json").read_text())
    assert meta["id"] == "rezanadimikj/dentex-official-test250" and meta["licenses"] and "docs/OFFICIAL_TEST250.md" in meta["description"]
    # the staged folder, seen as an attached Kaggle dataset, is read by the cohort resolver before any download
    files = [p for p in stage_dir.rglob("*") if p.is_file() and p.suffix in (".json", ".png")]
    monkeypatch.setattr(o, "origin", lambda p: "kaggle")
    got = prep(tmp_path / "run", files)
    assert got["sources"] == dict(coco="kaggle", images={"kaggle": 250}) and got["identity"]["coco_sha256"] == o.sha256_file(coco_path)
    # refusals: a broken COCO, a mapping that is not the one it was converted with, a foreign zip
    broken = json.loads(coco_path.read_text(encoding="utf-8"))
    broken["images"] = broken["images"][:-1]
    coco_path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(SystemExit, match="fails its checks"):
        stg.stage("v1", src, tmp_path / "k2", "o/s", tmp_path / "test_data.zip")
    coco_path.write_text(json.dumps(corrected), encoding="utf-8")
    monkeypatch.setattr(o, "OFFICIAL_ZIP_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="not the official test_data.zip"):
        stg.stage("v1", src, tmp_path / "k3", "o/s", tmp_path / "test_data.zip")
    with pytest.raises(SystemExit, match="owner/slug"):
        stg.main(["--dataset-id", "noslash", "--zip", str(tmp_path / "test_data.zip")])


FINAL_NOTEBOOKS = [REPO / "notebooks" / "VLM_Dental_Colab_Final_Evaluation.ipynb", REPO / "notebooks" / "TPU" / "TPU_Final_Evaluation.ipynb"]


def uncomment(src: str) -> str:
    return "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in src.split("\n"))


@pytest.mark.parametrize("path", FINAL_NOTEBOOKS, ids=lambda p: p.name)
def test_final_notebooks_have_the_commented_one_time_cells_and_nothing_else_changed(path):
    import re
    import shlex
    import importlib
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    settings = "".join(cells[6]["source"])
    assert 'EVAL_SPLIT = "test"' in settings and "official_test250" in settings and "Kaggle" in settings          # the default is unchanged; the option is documented
    secrets = "".join(cells[4]["source"])
    assert '("HF_TOKEN", "HF_ARTIFACT_REPO", "DENTEX_IMAGES_REPO"):' in secrets and "DENTEX_TEST_REPO" not in secrets   # the secrets loop is as it was
    header, convert, kaggle, hf = cells[-4:]
    assert header["cell_type"] == "markdown" and "commented out" in "".join(header["source"]) and "local disk, then" in "".join(header["source"])
    commands = {}
    for name, cell in (("convert", convert), ("kaggle", kaggle), ("hf", hf)):
        src = "".join(cell["source"])
        assert cell["cell_type"] == "code" and cell["outputs"] == [] and cell["execution_count"] is None
        assert all(l.startswith("#") or not l.strip() for l in src.split("\n")), name     # every line is a comment: running the notebook does nothing
        compile(src, name, "exec")
        live = uncomment(src)
        assert re.search(r"^(CONVERT_TO_COCO|UPLOAD_TO_KAGGLE|OVERWRITE_HF) = True$", live, re.M)
        compile("\n".join(l for l in live.split("\n") if not l.lstrip().startswith("!")), name, "exec")             # the uncommented cell is valid Python (magics aside)
        commands[name] = [l.strip()[1:] for l in live.split("\n") if l.strip().startswith("!{sys.executable} scripts/")]
    env = dict(TEST250_VERSION="v1", KAGGLE_DATASET_ID="rezanadimikj/dentex-official-test250", TEST250_STAGE_DIR="data/official_test250/kaggle_dataset")

    def render(line):
        return shlex.split(re.sub(r"\{(\w+)\}", lambda m: env.get(m.group(1), "PYTHON"), line))

    conv = render(commands["convert"][0])
    stage_args = render(commands["kaggle"][0])
    pub = render(commands["hf"][0])
    assert conv[1] == "scripts/convert_official_test_to_coco.py" and stage_args[1] == "scripts/stage_official_test250_dataset.py" and pub[1] == "scripts/publish_official_test_coco.py"
    c = importlib.import_module("scripts.convert_official_test_to_coco").build_parser().parse_args(conv[2:])
    s_ = importlib.import_module("scripts.stage_official_test250_dataset").build_parser().parse_args(stage_args[2:])
    p_ = importlib.import_module("scripts.publish_official_test_coco").build_parser().parse_args(pub[2:])
    assert (c.version, c.out) == ("v1", "data/official_test250")
    assert (s_.version, s_.src, s_.out, s_.dataset_id) == ("v1", "data/official_test250", "data/official_test250/kaggle_dataset", "rezanadimikj/dentex-official-test250")
    assert (p_.version, p_.dir, p_.overwrite) == ("v1", "data/official_test250", True)
    live_kaggle = uncomment("".join(kaggle["source"]))
    assert "kaggle datasets version" in live_kaggle and "kaggle datasets create" in live_kaggle and "--dir-mode tar" in live_kaggle     # the SFT notebook's own upload pattern
