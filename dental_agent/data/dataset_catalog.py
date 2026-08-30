"""
Catalog of publicly available panoramic dental imaging datasets, transcribed
from a peer-reviewed systematic review rather than assembled from scattered
web sources:

  Uribe SE, Issa J, Sohrabniya F, Denny A, Kim NN, Dayo AF, Chaurasia A,
  Sofi-Mahmudi A, Büttner M, Schwendicke F. "Publicly Available Dental Image
  Datasets for Artificial Intelligence." J Dent Res. 2024;103(13):1365-1374.
  DOI: 10.1177/00220345241272052. PMC11633071.

That review screened 131,028 records and found only 16 unique public dental
imaging datasets meeting a >=50-image inclusion bar -- and found that "the
methods used to establish labels were often unclear and inconsistent" across
them (their words, not ours). That finding is directly relevant to this
codebase: it's the same reason tufts.py's tooth-position/diagnosis mapping
functions raise NotImplementedError rather than guess at a mask convention
inferred from secondary sources, rather than a one-off caution specific to
Tufts.

This module holds only what's actually citable from that review -- name,
modality, size, license, annotation type/tool, and whether registration is
required. It does NOT include a loader for every entry; dentex.py, tufts.py,
and tunisia_panoramic.py are the only three with actual code (and of those,
only dentex.py currently returns usable annotations end-to-end -- the other
two each have one open, honestly-flagged verification question blocking
them, see their own module docstrings). Entries here are a prioritization
aid for "what to scaffold next," not a claim that all of them are ready to
use.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetEntry:
    name: str
    modality: str
    n_images: int
    country: str
    license: str  # as reported by source; "unspecified" means no license was found
    annotation_type: str  # Label / Pixel level / Box / unspecified
    registration_required: bool
    url: str
    has_diagnosis_labels: bool  # False = tooth identification/segmentation only, no pathology
    source_note: str = ""  # where this entry's facts came from, beyond the Uribe review
    notes: str = ""


# ---------------------------------------------------------------------------
# has_diagnosis_labels matters more than it might look like a minor filter
# field. DENTEX is still the only panoramic dataset found so far that pairs
# tooth position with an actual pathology label (caries/deep caries/
# periapical lesion/impaction) in the way this project's reward and trace-gen
# pipeline need. Every other dataset catalogued below that's usable at all is
# a tooth-IDENTIFICATION dataset (segmentation, instance masks, FDI
# numbering) with no diagnosis signal -- useful for a DIFFERENT purpose than
# DENTEX: generalizing locate_tooth's grounding accuracy across more images
# and imaging equipment, not generating more SFT/GRPO diagnosis traces.
#
# This isn't a guess -- there's a direct, real precedent for exactly this
# split-purpose combination: Merlin et al., BMC Oral Health 2024
# (10.1186/s12903-024-04129-5) trained a two-stage pipeline combining a
# tooth-instance-segmentation dataset (OdontoAI) with DENTEX specifically
# because DENTEX alone wasn't enough for grounding generalization and
# OdontoAI alone has no diagnosis labels -- the same gap this project is in.
# So "which dataset should we add next" has two different right answers
# depending on which pipeline stage you're trying to improve: locate_tooth's
# generalization (any dataset below, diagnosis-labeled or not) vs. trace
# generation for SFT/GRPO (currently: DENTEX only, until a second
# diagnosis-labeled dataset turns up).
# ---------------------------------------------------------------------------

# Panoramic-radiograph datasets only (this project's modality). The core 8
# come from the Uribe et al. 2024 systematic review (see module docstring);
# entries with a source_note came from targeted follow-up search on
# specific candidates, since that review's cutoff (screened through early
# 2024) predates several of these.
PANORAMIC_DATASETS: list[DatasetEntry] = [
    DatasetEntry(
        name="DENTEX",
        modality="panoramic",
        n_images=2332,
        country="Switzerland",
        license="unspecified (review); in-practice reuse via Zenodo/challenge terms",
        annotation_type="label (quadrant/enumeration/diagnosis; this project also uses its box coordinates directly)",
        registration_required=False,
        url="https://zenodo.org/records/7812323",
        has_diagnosis_labels=True,
        notes="Primary dataset already in use (dentex.py). The only panoramic dataset in this "
              "project pairing tooth position with all 4 pathology labels (caries/deep "
              "caries/periapical lesion/impaction) -- Tufts (below) pairs tooth position with "
              "real pathology too, but only Periapical Lesion has a Tufts analog.",
    ),
    DatasetEntry(
        name="Tufts Panoramic Dataset",
        modality="panoramic",
        n_images=1000,
        country="United States",
        license="unspecified",
        annotation_type="verified against the real files (Segmentation/teeth_bbox.json, "
                         "Expert/expert.json, Student/student.json, all 1,000 records each): "
                         "per-tooth bounding boxes (Universal Numbering, permanent 1-32 + "
                         "primary A-T) plus per-finding pathology polygons, free-text "
                         "description, and a 4-level clinical classification hierarchy. "
                         "Segmentation masks (teeth_polygon.json, teeth_mask/) exist in the "
                         "archive but are not yet inspected -- see tufts.py's "
                         "load_tufts_segmentation docstring.",
        registration_required=True,
        url="https://tdd.ece.tufts.edu/",
        has_diagnosis_labels=True,
        source_note="Real taxonomy (verified, supersedes the Uribe et al. 2024 estimate of "
                     "caries/oral-pathology/endodontics): expert.json's abnormality titles are "
                     "None (660), Periapical (217), Non-Odontogenic (105), Pericoronal (48), "
                     "Inter-Radicular (4) -- 340 abnormal / 660 normal images. Only Periapical "
                     "maps onto one of DENTEX's 4 classes (Periapical Lesion) -- Tufts has no "
                     "caries findings in this layer at all (6/1000 descriptions mention caries "
                     "at all, all incidental). Impacted has a real but unexploited path: 32/1000 "
                     "descriptions say 'impacted', clustering in the 48 Pericoronal findings -- "
                     "not yet extracted/implemented, would need per-finding text review, not a "
                     "blanket Pericoronal->Impacted remap (most Pericoronal findings aren't "
                     "impaction-related). See tufts.py's module docstring for the full mapping "
                     "rationale and the tooth-assignment recall caveat (periapical lesions often "
                     "sit just outside their tooth's own bbox by clinical definition).",
        notes="Loader implemented (tufts.py: load_tufts_dataset for diagnosis-bearing rows -- "
              "~202 images, Periapical Lesion only; load_tufts_tooth_boxes for the full "
              "~25,000-box grounding corpus across all 1,000 images, diagnosis-agnostic). "
              "2 annotators (Expert + Student, same schema, useful for inter-rater ablation), "
              "JPEG, 1615x840.",
    ),
    DatasetEntry(
        name="Dental Radiography",
        modality="panoramic",
        n_images=1272,
        country="Iran",
        license="CC BY-SA 4.0",
        annotation_type="unspecified in the review's annotation-type field, but the dataset "
                         "ships an _annotations.csv (image size + description per image)",
        registration_required=False,
        url="https://www.kaggle.com/datasets/imtkaggleteam/dental-radiography",
        has_diagnosis_labels=False,
        notes="Second-largest panoramic candidate after DENTEX, and a standard CSV rather than "
              "a mask convention -- but no confirmed diagnosis taxonomy; 'description per image' "
              "could be free text, not categorical. Verify before assuming this feeds diagnosis "
              "trace-gen rather than grounding only.",
    ),
    DatasetEntry(
        name="Panoramic Dental Xray Dataset",
        modality="panoramic",
        n_images=180,
        country="Tunisia",
        license="CC BY 4.0",
        annotation_type="pixel level, via VGG Image Annotator (VIA)",
        registration_required=False,
        url="https://data.mendeley.com/datasets/73n3kz2k4k/3",
        has_diagnosis_labels=False,
        source_note="Corrected after direct follow-up -- the Mendeley listing (v3) describes "
                     "three parts: 107 images with tooth-instance-segmentation annotations, 60 "
                     "images labeled by 8 tooth-TYPE classes (canine, incisor, molar, premolar "
                     "variants -- morphological type, not FDI quadrant+position), and 54 "
                     "unannotated high-resolution images. No pathology/diagnosis labels found "
                     "in any part.",
        notes="Most permissively licensed panoramic dataset found (plain CC BY). Still a strong "
              "candidate for expanding locate_tooth's training corpus specifically -- not for "
              "diagnosis trace-gen, contrary to this entry's earlier framing in this file. "
              "In progress (tunisia_panoramic.py): image discovery, VIA JSON parsing, and "
              "bbox-from-region geometry are implemented; region-to-FDI mapping is the one "
              "open question blocking category_id_1/category_id_2 construction -- see that "
              "module's _region_to_fdi docstring for exactly what to verify.",
    ),
    DatasetEntry(
        name="Panoramic-Caries-Segmentation",
        modality="panoramic",
        n_images=75,
        country="China",
        license="unspecified",
        annotation_type="pixel level (caries-specific)",
        registration_required=False,
        url="https://github.com/Zzz512/MLUA",
        has_diagnosis_labels=True,
        notes="Small, but directly caries-focused and ships a README -- worth a look for "
              "caries-specific augmentation even at this size, though likely binary "
              "(caries/not) rather than DENTEX's 4-class taxonomy; needs checking.",
    ),
    DatasetEntry(
        name="TK_Tooth_Number_Code",
        modality="panoramic",
        n_images=188,
        country="unspecified",
        license="unspecified",
        annotation_type="label (tooth numbering specifically)",
        registration_required=False,
        url="https://github.com/tanjidakabir/TK_Tooth_Number_Code",
        has_diagnosis_labels=False,
        notes="Ships a README, model description, mask file, and several xlsx files describing "
              "the data -- tabular ground truth, tractable to parse. Numbering-only, no "
              "diagnosis: a grounding-tool candidate, not a trace-gen one.",
    ),
    DatasetEntry(
        name="DNS (Detection, Numbering, and Segmentation)",
        modality="panoramic",
        n_images=543,
        country="unspecified",
        license="available upon request to the authors",
        annotation_type="pixel level + COCO-format boxes, with FDI tooth numbering",
        registration_required=True,
        url="https://github.com/IvisionLab/dns-panoramic-images-v2",
        has_diagnosis_labels=False,
        source_note="Found via targeted follow-up search, not in the Uribe et al. 2024 review. "
                     "Derived from the UFBA-UESC dental dataset; multiple follow-up papers "
                     "(TNDRS, orthodontic-appliance extensions) have extended it further. "
                     "Access is upon-request, similar constraint to Tufts, not a free download.",
        notes="Real FDI numbering in COCO format is exactly the shape locate_tooth/fdi_label "
              "need -- a strong grounding-tool candidate if access is granted, but no "
              "diagnosis/pathology labels, so it wouldn't feed trace generation.",
    ),
    DatasetEntry(
        name="TL-pano",
        modality="panoramic",
        n_images=197,
        country="Brazil",
        license="restricted -- non-commercial research use only",
        annotation_type="pixel level via VIA, with FDI tooth numbers AND quadrant sub-labels",
        registration_required=True,
        url="https://zenodo.org/records/15038971",
        has_diagnosis_labels=False,
        source_note="Found via targeted follow-up search, published Oct 2025, not in the Uribe "
                     "et al. 2024 review (predates it). Explicitly built as a *supporting* "
                     "dataset meant to complement caries/bone-loss-labeled datasets, not to "
                     "provide diagnosis labels itself.",
        notes="Directly relevant to fdi_label's quadrant+position output shape (VIA sub-labels "
              "for both), and hosted on Zenodo rather than gated behind a request form -- but "
              "the non-commercial license needs checking against how this project's outputs "
              "(a trained model, published traces) would actually be used/distributed before "
              "committing to it.",
    ),
    DatasetEntry(
        name="Panoramic Dental X-rays With Segmented Mandibles",
        modality="panoramic",
        n_images=232,
        country="Iran",
        license="CC BY-NC 3.0",
        annotation_type="pixel level (mandible only, not per-tooth)",
        registration_required=False,
        url="https://data.mendeley.com/datasets/hxt48yk462/2",
        has_diagnosis_labels=False,
        notes="Segments the mandible as a whole, not individual teeth -- not useful for "
              "locate_tooth/diagnosis training as-is, listed for completeness.",
    ),
    DatasetEntry(
        name="Panoramic-Paraguay",
        modality="panoramic",
        n_images=135,
        country="Paraguay",
        license="unspecified",
        annotation_type="unspecified",
        registration_required=False,
        url="https://zenodo.org/records/4457648",
        has_diagnosis_labels=False,
        notes="Thin documentation in the review (no annotation type or ground-truth method "
              "reported) -- lower priority until more is known.",
    ),
]


def summarize() -> str:
    """Quick human-readable summary for a notebook cell or REPL check."""
    lines = [f"{len(PANORAMIC_DATASETS)} panoramic datasets:"]
    for d in PANORAMIC_DATASETS:
        reg = "registration required" if d.registration_required else "no registration"
        diag = "has diagnosis labels" if d.has_diagnosis_labels else "grounding only, no diagnosis"
        lines.append(f"  - {d.name}: {d.n_images} images, {d.country}, {d.license}, {reg}, {diag}")
    return "\n".join(lines)
