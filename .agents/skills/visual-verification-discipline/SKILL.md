---
name: visual-verification-discipline
description: >-
  Strict protocol enforcing zero-hallucination image checking and mandatory visual verification using the view_file tool before presenting generated visual artifacts to the user in VLM-DENTAL.
---

# Visual Verification Discipline Protocol

This skill enforces a strict no-hallucination policy when generating, editing, or producing visual media (images, diagrams, bounding box drawings, UI outputs).

## Core Directives

1. **Mandatory Visual Inspection**: Before informing the user that an image has been successfully generated or modified (e.g., "I drew the bounding boxes!"), you MUST use the `view_file` tool to actually *look* at the resulting image yourself.
2. **Zero Code-Execution Trust**: Never assume an image is correct just because the underlying Python or plotting code ran without throwing an exception. Code can execute perfectly while drawing coordinates off-screen, using invisible colors, or creating blank canvases.
3. **Verify Expected Content**: When you `view_file` on an image, actively check for the expected visual elements (e.g., "Are the red boxes actually visible around the teeth?", "Is the plot line rendering?").
4. **Fix Before Presenting**: If the image is blank, malformed, or missing the expected elements, you must fix the code and re-generate the image *before* presenting it to the user.
5. **No Blind Artifacts**: When creating Markdown artifacts that embed images, you must have visually verified the embedded images first. Do not embed unverified images.

## Recommended Workflow

1. Write the script to generate or modify the image.
2. Run the script and wait for completion.
3. Call `view_file` on the resulting absolute path of the generated image (e.g., `view_file(AbsolutePath="/path/to/image.png")`).
4. Analyze the image visually to confirm success.
5. If successful, present it to the user. If unsuccessful, debug the coordinate scaling, colors, line widths, etc., and return to Step 1.
