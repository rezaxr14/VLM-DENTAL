---
{
  "name": "caries-vs-deep-caries",
  "description": "Keep the Caries and Deep Caries labels distinct and verify the visual evidence.",
  "version": "1.2.0",
  "sources": [
    "dental_agent/agent/prompts.py:build_agent_system_prompt",
    "dental_agent/evaluation/ground_truth.py:DENTEX_DEFAULT_DIAGNOSES",
    "https://onlinelibrary.wiley.com/doi/full/10.1111/iej.13080",
    "https://www.fda.gov/radiation-emitting-products/medical-x-ray-imaging/selection-patients-dental-radiographic-examinations",
    "https://www.nidcr.nih.gov/health-info/tooth-decay"
  ]
}
---

1. Caries and Deep Caries are distinct permitted output labels. [S2]

2. Inspect the returned tooth crop and improve contrast or windowing when density is unclear. [S1]

3. Radiographs contribute evidence about decay; clinical detection also uses examination. Do not infer symptoms or vitality from the image alone. [S5]

4. After verifying the tooth, inspect the crown and any suspected radiolucency, then use window_level or enhance_contrast if density is unclear; compare the suspected extent toward the pulp before choosing between the two permitted caries labels. [S1]

5. The ESE framework distinguishes deep from extremely deep dentinal lesions by their extent and remaining barrier to the pulp. This is clinical context, not an asserted numeric DENTEX annotation rule; do not add an Extremely Deep Caries output class. [S3]

6. Panoramic assessment is not proof of absence of proximal caries; the FDA/ADA selection guidance identifies bitewings as the efficient view for proximal lesions. Record the visible evidence rather than assuming a low-detail region is normal. [S4]
