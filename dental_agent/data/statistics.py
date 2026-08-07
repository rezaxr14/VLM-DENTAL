"""
Dataset statistics and visualization.

Produces the histograms and co-occurrence matrices from the proposal
(image dimensions, diagnosis distribution, bbox areas, quadrant × diagnosis).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def plot_image_dimensions(images_df: pd.DataFrame) -> None:
    """Plot width and height distributions of images."""
    if "width" in images_df.columns and "height" in images_df.columns:
        widths, heights = images_df["width"], images_df["height"]
    else:
        sizes = (
            images_df.dropna(subset=["local_path"])["local_path"]
            .apply(lambda p: Image.open(p).size)
        )
        widths, heights = zip(*sizes)
        widths, heights = pd.Series(widths), pd.Series(heights)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(widths, bins=30)
    axes[0].set_title("Image width distribution")
    axes[0].set_xlabel("pixels")
    axes[1].hist(heights, bins=30)
    axes[1].set_title("Image height distribution")
    axes[1].set_xlabel("pixels")
    plt.tight_layout()
    plt.show()

    print(f"Width:  min={widths.min()}  max={widths.max()}  mean={widths.mean():.0f}")
    print(f"Height: min={heights.min()}  max={heights.max()}  mean={heights.mean():.0f}")


def get_diagnosis_column(annots_df: pd.DataFrame) -> str | None:
    """Auto-detect the diagnosis category column."""
    for c in ("category_id_3", "category_id"):
        if c in annots_df.columns:
            return c
    return None


def plot_diagnosis_distribution(
    annots_df: pd.DataFrame,
    categories_df: pd.DataFrame,
    diag_col: str | None = None,
) -> pd.Series | None:
    """Plot the diagnosis class distribution."""
    diag_col = diag_col or get_diagnosis_column(annots_df)
    if not diag_col or not len(categories_df):
        print("Could not auto-detect the diagnosis category column.")
        return None

    cat_lookup = dict(zip(categories_df["id"], categories_df["name"]))
    diag_counts = annots_df[diag_col].map(cat_lookup).value_counts()

    plt.figure(figsize=(8, 4))
    diag_counts.plot(kind="bar")
    plt.title("Diagnosis class distribution (annotated abnormal teeth)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()
    return diag_counts


def plot_bbox_areas(
    annots_df: pd.DataFrame, images_df: pd.DataFrame
) -> None:
    """Plot bounding-box area distributions (absolute and relative)."""
    bbox_areas = annots_df["bbox"].apply(lambda b: b[2] * b[3])

    plt.figure(figsize=(8, 4))
    plt.hist(bbox_areas, bins=40)
    plt.title("Annotated tooth/lesion bounding-box area (pixels²)")
    plt.xlabel("area")
    plt.tight_layout()
    plt.show()

    if "width" in images_df.columns and "height" in images_df.columns:
        img_area = images_df.set_index("id")["width"] * images_df.set_index("id")["height"]
        rel_area = bbox_areas.values / annots_df["image_id"].map(img_area).values
        rel_area = pd.Series(rel_area).dropna()

        plt.figure(figsize=(8, 4))
        plt.hist(rel_area, bins=40)
        plt.title("Bounding-box area as a fraction of full image area")
        plt.xlabel("fraction")
        plt.tight_layout()
        plt.show()
        print(f"Median annotated box covers {rel_area.median() * 100:.2f}% of the full image.")


def plot_annotations_per_image(annots_df: pd.DataFrame) -> None:
    """Plot how many annotated abnormal teeth each image has."""
    per_image = annots_df.groupby("image_id").size()
    plt.figure(figsize=(8, 4))
    plt.hist(per_image, bins=range(1, per_image.max() + 2))
    plt.title("Number of annotated abnormal teeth per image")
    plt.xlabel("count")
    plt.tight_layout()
    plt.show()
    print(per_image.describe())


def plot_quadrant_diagnosis_matrix(
    annots_df: pd.DataFrame,
    categories_df: pd.DataFrame,
    quad_col: str = "category_id_1",
    diag_col: str | None = None,
) -> pd.DataFrame | None:
    """Plot the quadrant × diagnosis co-occurrence matrix."""
    diag_col = diag_col or get_diagnosis_column(annots_df)
    if not quad_col or quad_col not in annots_df.columns:
        return None
    if not diag_col or not len(categories_df):
        return None

    cat_lookup = dict(zip(categories_df["id"], categories_df["name"]))
    pivot = pd.crosstab(
        annots_df[quad_col].map(cat_lookup),
        annots_df[diag_col].map(cat_lookup),
    )

    plt.figure(figsize=(8, 5))
    plt.imshow(pivot.values, cmap="Blues", aspect="auto")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            plt.text(j, i, pivot.values[i, j], ha="center", va="center")
    plt.title("Quadrant × diagnosis co-occurrence")
    plt.colorbar(label="count")
    plt.tight_layout()
    plt.show()
    return pivot


plot_cooccurrence_matrix = plot_quadrant_diagnosis_matrix
