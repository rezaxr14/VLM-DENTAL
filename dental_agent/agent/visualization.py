"""
Qualitative visualization tools for dental radiographs and agent multi-turn trajectories.
"""

from __future__ import annotations

from typing import Any, Mapping
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def draw_annotations(
    image: Image.Image | str,
    annotations: list[dict[str, Any]],
    categories_lookup: dict[int, str] | None = None,
    outline_color: str = "red",
    line_width: int = 3,
) -> Image.Image:
    """Draw bounding boxes and class labels over a dental radiograph image."""
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    else:
        img = image.copy().convert("RGB")

    draw = ImageDraw.Draw(img)
    categories_lookup = categories_lookup or {}

    for ann in annotations:
        bbox = ann.get("bbox", [])
        if len(bbox) < 4:
            continue
        x, y, w, h = bbox
        draw.rectangle([x, y, x + w, y + h], outline=outline_color, width=line_width)

        label_parts = []
        if "fdi_label" in ann and ann["fdi_label"]:
            label_parts.append(f"#{ann['fdi_label']}")
        if "diagnosis" in ann and ann["diagnosis"]:
            label_parts.append(str(ann["diagnosis"]))
        elif "category_id_3" in ann and ann["category_id_3"] in categories_lookup:
            label_parts.append(categories_lookup[ann["category_id_3"]])

        if label_parts:
            text = " ".join(label_parts)
            draw.text((x, max(0, y - 12)), text, fill=outline_color)

    return img


def visualize_trajectory(trajectory: Any) -> None:
    """Plot panels of every visual state/crop viewed by the agent during a trajectory."""
    traj_dict = trajectory.to_dict() if hasattr(trajectory, "to_dict") else trajectory
    messages = traj_dict.get("messages", [])

    panels = []
    for msg in messages:
        if msg.get("role") == "system":
            continue
        content = msg.get("content", [])
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]

        img = next((c.get("image") for c in content if isinstance(c, dict) and c.get("type") == "image"), None)
        text = " ".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
        role = msg.get("role", "unknown")
        panels.append((f"[{role}] {text[:80]}", img))

    imgs_only = [(label, img) for label, img in panels if img is not None]
    if not imgs_only:
        print("No images found in trajectory message history.")
        return

    n = len(imgs_only)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (label, img) in zip(axes, imgs_only):
        ax.imshow(img)
        ax.set_title(label, fontsize=8, wrap=True)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
