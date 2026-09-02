# Visual Verification Discipline Invariant

## 1. Zero-Hallucination Visual Verification (CRITICAL)
- **Rule:** Before claiming to the user that an image contains specific visual elements (such as bounding boxes, charts, or rendered UI elements), you MUST visually verify the image yourself using the `view_file` tool.
- **Rule:** Never assume an image was generated correctly just because the code executed without errors or the coordinates in the logs looked plausible.
- **Rule:** If you generate an image or diagram to demonstrate a result, always `view_file` on the resulting image file to see it with your own "eyes" before proudly presenting it to the user.
- **Rule:** If the image is entirely black, the boxes are drawn off-screen, or the rendering is otherwise faulty, you must fix it *before* showing it to the user. Do not hallucinate success.
