---
{
  "name": "periapical-lesion",
  "description": "Investigate the Periapical Lesion hypothesis without assuming the proposed box is correct.",
  "version": "1.2.0",
  "sources": [
    "dental_agent/agent/prompts.py:build_agent_system_prompt",
    "dental_agent/evaluation/ground_truth.py:DENTEX_DEFAULT_DIAGNOSES",
    "https://www.aae.org/specialty/a-concise-guide-to-pathology-within-endodontics/",
    "https://www.aae.org/specialty/wp-content/uploads/sites/2/2017/07/aaeconsensusconferencerecommendeddiagnosticterminology.pdf"
  ]
}
---

1. Periapical Lesion is a permitted output label. [S2]

2. The existing example inspects the root apex and compares the mirrored side before committing. [S1]

3. Verify the requested tooth is visible in the crop; correct the box before trusting that view. [S1]

4. Apical periodontal inflammation may produce radiographic periapical radiolucency. Granuloma, cyst, abscess and scar cannot be reliably separated from each other by their radiographic appearance alone; retain the permitted Periapical Lesion label rather than inventing a histologic subtype. [S3]

5. Use locate_tooth, zoom_crop and an alignment check before assessing the apex. If the view is misplaced, nudge then re-crop. Use contralateral_compare to investigate asymmetry and window_level for unclear density. [S1]

6. Do not infer percussion tenderness, pain, sinus tracts or pulp vitality from a panoramic image. Clinical apical diagnoses incorporate findings beyond a radiolucency. [S4]
