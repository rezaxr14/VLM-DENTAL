"""Evaluation-time tool ablations and box-vs-ground-truth localization.

Nothing here touches training or the normal agent loop. ``run_agent`` takes a ``ToolRegistry``; an ablation is a
different registry, built per image by ``ToolProbe``:

* **Tool removal** (``remove_tools``): the tool is absent from the registry, so it is missing from the tool list in the
  system prompt and a call to it gets the loop's standard "not recognized" error. The prompt's rules still mention the
  tool (the checkpoint was trained with them), so how often the model asks for it anyway is recorded as
  ``removed_tool_attempts``. This measures how much a trained agent depends on a tool; it is not a model trained without
  it.
* **``locate_tooth`` perturbation** (``locate_shift_frac`` / ``locate_scale``): the detector's box is moved by a fraction
  of its own size in a pseudo-random direction and rescaled around its centre, then clamped to the image. The direction
  is fixed by (seed, image, tooth), so every condition and every run sees the same wrong box, and conditions can be paired.
  This is a stress test of the agent's self-correction (``nudge_crop``), not an evaluation of the detector, which stays
  the one fixed supporting tool.
* **Localization**: every ``locate_tooth`` / ``nudge_crop`` / ``zoom_crop`` call is logged. For each ground-truth finding
  whose tooth the agent located, the IoU of the working box with the annotated tooth box is reported right after
  ``locate_tooth`` (``iou_initial``) and after the last ``nudge_crop`` of that box (``iou_final``), plus how much of the
  annotated tooth the last ``zoom_crop`` of the box actually shows (``crop_coverage``).

Worked example: ground-truth tooth box [100, 100, 100, 100]; ``locate_tooth`` returns [150, 100, 100, 100] (shifted half a
box width). Intersection 50 x 100 = 5000, union 10000 + 10000 - 5000 = 15000, IoU 1/3. The agent calls ``nudge_crop`` with
dx_frac -0.5 and gets [100, 100, 100, 100]: IoU 1.0, so the finding is "recovered" at an IoU threshold of 0.5.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from dental_agent.tools.registry import ToolRegistry

MATCH_IOU = 0.9   # a nudge/zoom box refers to a located tooth when it overlaps that tooth's current box this much


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AblationSpec:
    remove_tools: tuple[str, ...] = ()
    locate_shift_frac: float = 0.0
    locate_scale: float = 1.0
    seed: int = 0

    @property
    def perturbs_locate(self) -> bool:
        return self.locate_shift_frac > 0 or self.locate_scale != 1.0

    @property
    def active(self) -> bool:
        return bool(self.remove_tools) or self.perturbs_locate

    def tag(self) -> str:
        """File-name tag of this ablation; empty for the unablated baseline."""
        if not self.active:
            return ""
        parts = [f"no-{t}" for t in sorted(self.remove_tools)]
        if self.perturbs_locate:
            parts.append(f"loc-shift{self.locate_shift_frac:g}-scale{self.locate_scale:g}-seed{self.seed}")
        return "__" + "+".join(parts)

    def settings(self) -> dict[str, Any]:
        return {"remove_tools": sorted(self.remove_tools), "locate_shift_frac": self.locate_shift_frac,
                "locate_scale": self.locate_scale, "seed": self.seed}


def spec_from_args(args: Any) -> AblationSpec:
    return AblationSpec(
        remove_tools=tuple(getattr(args, "remove_tools", None) or ()),
        locate_shift_frac=float(getattr(args, "locate_shift_frac", 0.0) or 0.0),
        locate_scale=float(getattr(args, "locate_scale", 1.0) or 1.0),
        seed=int(getattr(args, "ablation_seed", 0) or 0),
    )


# ---------------------------------------------------------------------------
# Box geometry ([x, y, w, h], pixels of the original image)
# ---------------------------------------------------------------------------

def box_iou(a: list[float], b: list[float]) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = min(ax2, bx2) - max(a[0], b[0]), min(ay2, by2) - max(a[1], b[1])
    inter = max(0.0, iw) * max(0.0, ih)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def zoom_rect(bbox: list[float], padding_frac: float, img_w: int, img_h: int) -> list[float]:
    """The rectangle ``tool_zoom_crop`` actually shows for ``bbox`` (same padding rule, clamped to the image), as [x, y, w, h]."""
    x, y, w, h = bbox
    pad_x, pad_y = max(w * padding_frac, 50.0), max(h * padding_frac, 50.0)
    left, top = max(0.0, x - pad_x), max(0.0, y - pad_y)
    right, bottom = min(float(img_w), x + w + pad_x), min(float(img_h), y + h + pad_y)
    return [left, top, right - left, bottom - top]


def coverage(view: list[float], target: list[float]) -> float:
    """Share of ``target``'s area that lies inside ``view``."""
    iw = min(view[0] + view[2], target[0] + target[2]) - max(view[0], target[0])
    ih = min(view[1] + view[3], target[1] + target[3]) - max(view[1], target[1])
    area = target[2] * target[3]
    return max(0.0, iw) * max(0.0, ih) / area if area > 0 else 0.0


def perturb_box(bbox: list[float], img_w: int, img_h: int, shift_frac: float, scale: float, angle: float) -> list[float]:
    """Move ``bbox`` by ``shift_frac`` of its own size in direction ``angle`` and rescale it about its centre, clamped to the image."""
    x, y, w, h = bbox
    cx = x + w / 2.0 + shift_frac * math.cos(angle) * w
    cy = y + h / 2.0 + shift_frac * math.sin(angle) * h
    nw, nh = min(w * scale, float(img_w)), min(h * scale, float(img_h))
    nx = max(0.0, min(cx - nw / 2.0, img_w - nw))
    ny = max(0.0, min(cy - nh / 2.0, img_h - nh))
    return [round(nx, 1), round(ny, 1), round(nw, 1), round(nh, 1)]


def perturbation_angle(seed: int, image_id: Any, tooth: Any) -> float:
    return random.Random(f"{seed}|{image_id}|{tooth}").uniform(0.0, 2.0 * math.pi)


# ---------------------------------------------------------------------------
# Per-image probe
# ---------------------------------------------------------------------------

def _as_box(v: Any) -> list[float] | None:
    try:
        b = [float(t) for t in v]
    except (TypeError, ValueError):
        return None
    return b if len(b) == 4 else None


def _fdi(q: Any, pos: Any) -> int | None:
    try:
        return int(q) * 10 + int(pos)
    except (TypeError, ValueError):
        return None


@dataclass
class ToolProbe:
    """Builds the registry one image is run with, and logs its localization-relevant tool calls."""

    base_registry: ToolRegistry
    spec: AblationSpec
    image_id: Any
    calls: list[dict[str, Any]] = field(default_factory=list)
    registry: ToolRegistry = field(init=False)

    def __post_init__(self) -> None:
        reg = ToolRegistry()
        for t in self.base_registry.list_tools():
            if t.name in self.spec.remove_tools:
                continue
            func = t.func
            if t.name == "locate_tooth":
                func = self._locate(func)
            elif t.name == "nudge_crop":
                func = self._logged(t.name, func)
            elif t.name == "zoom_crop":
                func = self._zoom(func)
            reg.register(t.name, func, t.description, t.schema)
        self.registry = reg

    # -- wrappers ----------------------------------------------------------
    def _locate(self, func):
        def wrapper(image: Image.Image, tooth: Any = None, **kw: Any) -> Any:
            out = func(image=image, tooth=tooth, **kw)
            entry: dict[str, Any] = {"tool": "locate_tooth", "tooth": _safe_int(tooth), "output_box": None}
            box = _as_box(out.get("bbox")) if isinstance(out, dict) else None
            if box is not None:
                entry["true_box"] = box
                if self.spec.perturbs_locate:
                    angle = perturbation_angle(self.spec.seed, self.image_id, _safe_int(tooth))
                    box = perturb_box(box, image.width, image.height, self.spec.locate_shift_frac, self.spec.locate_scale, angle)
                    out = {**out, "bbox": box}
                entry["output_box"] = box
            self.calls.append(entry)
            return out
        return wrapper

    def _logged(self, name: str, func):
        def wrapper(image: Image.Image, **kw: Any) -> Any:
            out = func(image=image, **kw)
            self.calls.append({"tool": name, "input_box": _as_box(kw.get("bbox")),
                               "output_box": _as_box(out.get("bbox")) if isinstance(out, dict) else None})
            return out
        return wrapper

    def _zoom(self, func):
        def wrapper(image: Image.Image, **kw: Any) -> Any:
            out = func(image=image, **kw)
            box = _as_box(kw.get("bbox"))
            pad = kw.get("context_padding")           # same rule as tool_zoom_crop: context_padding wins when given
            if pad is None:
                pad = kw.get("padding_frac")
            pad = 0.25 if pad is None else float(pad)
            if box is not None:
                self.calls.append({"tool": "zoom_crop", "input_box": box, "view": zoom_rect(box, pad, image.width, image.height)})
            return out
        return wrapper

    # -- results -----------------------------------------------------------
    def box_vs_gt(self, gt_findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """One entry per ground-truth finding with an annotated tooth box (see module docstring)."""
        initial: dict[int, list[float]] = {}
        latest: dict[int, list[float]] = {}
        nudges: dict[int, int] = {}
        viewed: dict[int, list[float]] = {}

        def owner(box: list[float] | None) -> int | None:
            if box is None:
                return None
            best, best_iou = None, MATCH_IOU
            for tooth, cur in latest.items():
                v = box_iou(box, cur)
                if v >= best_iou:
                    best, best_iou = tooth, v
            return best

        for c in self.calls:
            if c["tool"] == "locate_tooth" and c.get("output_box") and c.get("tooth") is not None:
                initial.setdefault(c["tooth"], c["output_box"])
                latest[c["tooth"]] = c["output_box"]
            elif c["tool"] == "nudge_crop" and c.get("output_box"):
                t = owner(c.get("input_box"))
                if t is not None:
                    latest[t] = c["output_box"]
                    nudges[t] = nudges.get(t, 0) + 1
            elif c["tool"] == "zoom_crop":
                t = owner(c.get("input_box"))
                if t is not None:
                    viewed[t] = c["view"]

        out = []
        for g in gt_findings:
            gt_box, tooth = _as_box(g.get("bbox")), _fdi(g.get("quadrant"), g.get("tooth_position"))
            if gt_box is None or tooth is None:
                continue
            if tooth not in initial:
                out.append({"tooth": tooth, "found": False})
                continue
            out.append({
                "tooth": tooth, "found": True,
                "iou_initial": box_iou(initial[tooth], gt_box),
                "iou_final": box_iou(latest[tooth], gt_box),
                "n_nudges": nudges.get(tooth, 0),
                "crop_coverage": coverage(viewed[tooth], gt_box) if tooth in viewed else None,
            })
        return out

    def summary(self, gt_findings: list[dict[str, Any]], turns: list[dict[str, Any]]) -> dict[str, Any]:
        """The ``ablation`` field stored in an evaluation record."""
        removed = set(self.spec.remove_tools)
        attempts = sum(
            1 for t in turns for c in t.get("tool_calls_this_turn", []) or []
            if c.get("tool_name") in removed
        )
        counts: dict[str, int] = {}
        for c in self.calls:
            counts[c["tool"]] = counts.get(c["tool"], 0) + 1
        return {
            "tag": self.spec.tag(),
            "spec": self.spec.settings(),
            "box_vs_gt": self.box_vs_gt(gt_findings),
            "logged_calls": counts,
            "removed_tool_attempts": attempts,
        }


def _safe_int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
