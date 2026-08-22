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
required. It does NOT include a loader for every entry; dentex.py and
tufts.py are the only two with actual code. Entries here are a prioritization
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
    license: str  # as reported by the review; "unspecified" means the review found no license
    annotation_type: str  # Label / Pixel level / Box / unspecified -- the review's own categories
    registration_required: bool
    url: str
    notes: str = ""


# Panoramic-radiograph datasets only (this project's modality) from the
# review's Table 1. CBCT, cephalometric, 3D intraoral scan, and oral-pathology
# / non-tooth-detection datasets from the same table are omitted here since
# they're a different modality or task -- see the paper directly for those.
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
        notes="Primary dataset already in use (dentex.py). The review's 'label' "
              "categorization understates it somewhat -- DENTEX also ships COCO-style "
              "bounding boxes, which is what this project's grounding tool trains on.",
    ),
    DatasetEntry(
        name="Tufts Panoramic Dataset",
        modality="panoramic",
        n_images=1000,
        country="United States",
        license="unspecified",
        annotation_type="label (per this review; independent web sources describe additional "
                         "per-tooth/abnormality segmentation masks -- see tufts.py's module "
                         "docstring for why that discrepancy isn't resolved by guessing)",
        registration_required=True,
        url="https://tdd.ece.tufts.edu/",
        notes="In progress (tufts.py). 2 annotators, ground truth by expert decision, JPEG, "
              "840x1615.",
    ),
    DatasetEntry(
        name="Dental Radiography",
        modality="panoramic",
        n_images=1272,
        country="Iran",
        license="CC BY-SA 4.0",
        annotation_type="unspecified in the review's annotation-type field, but the dataset "
                         "ships an _annotations.csv (image size + description per image) -- a "
                         "concrete, standard format worth checking directly rather than a mask "
                         "convention to infer",
        registration_required=False,
        url="https://www.kaggle.com/datasets/imtkaggleteam/dental-radiography",
        notes="Second-largest panoramic candidate after DENTEX. Share-alike license -- fine to "
              "use, but downstream re-releases (e.g. a combined trace-gen dataset) would need "
              "to carry the same license forward.",
    ),
    DatasetEntry(
        name="Panoramic Dental Xray Dataset",
        modality="panoramic",
        n_images=180,
        country="Tunisia",
        license="CC BY 4.0",
        annotation_type="pixel level, via VGG Image Annotator (VIA) -- a well-documented, "
                         "standard tool with a known JSON export schema, not a bespoke format",
        registration_required=False,
        url="https://data.mendeley.com/datasets/73n3kz2k4k/2",
        notes="Most permissively licensed panoramic dataset in the review (plain CC BY, no "
              "share-alike or non-commercial restriction). VIA's export format is well-known "
              "enough to write a confident parser for, unlike Tufts' undocumented masks -- a "
              "genuinely strong candidate for the next loader after Tufts, license and format "
              "considered together.",
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
        notes="Small, but directly caries-focused and ships a README per the review's "
              "'additional information' field -- worth a look for caries-specific "
              "augmentation even at this size.",
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
        notes="Ships a README, model description, mask file, and several xlsx files describing "
              "the data (per the review) -- tabular ground truth is far more tractable to parse "
              "confidently than raw pixel masks, and tooth-numbering-specific data is directly "
              "relevant to generalizing locate_tooth/fdi_label across datasets.",
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
        notes="Thin documentation in the review (no annotation type or ground-truth method "
              "reported) -- lower priority until more is known.",
    ),
]


def summarize() -> str:
    """Quick human-readable summary for a notebook cell or REPL check."""
    lines = [f"{len(PANORAMIC_DATASETS)} panoramic datasets (Uribe et al. 2024):"]
    for d in PANORAMIC_DATASETS:
        reg = "registration required" if d.registration_required else "no registration"
        lines.append(f"  - {d.name}: {d.n_images} images, {d.country}, {d.license}, {reg}")
    return "\n".join(lines)
