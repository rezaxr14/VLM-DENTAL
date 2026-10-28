---
{
  "name": "box-verification",
  "description": "Inspect a proposed tooth box, correct its alignment, then inspect again.",
  "version": "1.2.0",
  "sources": [
    "dental_agent/agent/prompts.py:build_agent_system_prompt",
    "dental_agent/tools/fdi.py",
    "dental_agent/tools/nudge.py"
  ]
}
---

1. Treat the locate_tooth box as a starting hypothesis, not a guarantee. Inspect its zoom_crop before diagnosis. [S1]

2. If off-center, too tight/loose, or showing a different tooth, use nudge_crop on the returned box and zoom_crop again. Corrections may be repeated. [S1]

3. dx_frac and dy_frac are fractions of the supplied box width and height; scale changes its size. Coordinates remain native-image coordinates. [S3]

4. Use patient FDI quadrants: 1 upper right, 2 upper left, 3 lower left, 4 lower right; positions 1 through 8. Do not substitute display left for patient left. [S2]
