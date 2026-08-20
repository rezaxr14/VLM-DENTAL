"""
Contralateral comparison tool for symmetry assessment in dental radiographs.
"""
from PIL import Image
from dental_agent.tools.fdi import flip_quadrant

def tool_contralateral_compare(
    image: Image.Image,
    bbox: list[int],
    quadrant: int
) -> Image.Image:
    """
    Crops a region [x, y, w, h] in the specified quadrant and its anatomical mirror
    in the contralateral quadrant, returning a side-by-side composite for symmetry comparison.
    Asymmetry is a primary diagnostic signal for periapical radiolucency and bone loss.

    quadrant (1-4, FDI convention: 1-2 upper jaw, 3-4 lower jaw) constrains the
    mirror search to the same jaw half as the source crop. Previously accepted
    but not actually used -- a pure horizontal flip alone can pull the "mirror"
    crop from the wrong jaw when bbox sits close to the image's vertical midline
    (e.g. front incisors), silently comparing an upper tooth against a lower one.
    """
    if len(bbox) != 4:
        raise ValueError("bbox must be [x, y, w, h]")

    if quadrant not in (1, 2, 3, 4):
        raise ValueError("quadrant must be 1-4 (FDI convention).")

    x, y, w, h = [int(v) for v in bbox]
    img_w, img_h = image.size
    img_mid_y = img_h / 2.0

    # Target crop
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)

    if x2 <= x1 or y2 <= y1:
        # Invalid bbox fallback
        return image.copy()

    target_crop = image.crop((x1, y1, x2, y2))

    # Calculate mirrored bounding box across the vertical midline
    mirror_x2 = img_w - x
    mirror_x1 = mirror_x2 - w

    mx1, mx2 = max(0, int(mirror_x1)), min(img_w, int(mirror_x2))

    # Clamp the mirror crop's y-range to the same jaw half as `quadrant` (upper:
    # 1-2, lower: 3-4), so a box that straddles the midline can't pull context
    # from the opposite jaw into the comparison.
    if quadrant in (1, 2):
        my1, my2 = max(0, y), min(int(img_mid_y), y + h)
    else:
        my1, my2 = max(int(img_mid_y), y), min(img_h, y + h)

    if mx2 <= mx1 or my2 <= my1:
        return target_crop

    mirror_crop = image.crop((mx1, my1, mx2, my2))

    # Composite side-by-side
    target_w, target_h = target_crop.size
    mirror_w, mirror_h = mirror_crop.size

    composite_w = target_w + mirror_w + 10 # 10px divider
    composite_h = max(target_h, mirror_h)

    composite = Image.new("RGB", (composite_w, composite_h), color="black")

    # Left: Target, Right: Contralateral
    composite.paste(target_crop, (0, 0))
    composite.paste(mirror_crop, (target_w + 10, 0))

    return composite
