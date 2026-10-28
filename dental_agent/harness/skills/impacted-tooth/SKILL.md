---
{
  "name": "impacted-tooth",
  "description": "Investigate the Impacted Tooth hypothesis with verified localization.",
  "version": "1.2.0",
  "sources": [
    "dental_agent/agent/prompts.py:build_agent_system_prompt",
    "dental_agent/evaluation/ground_truth.py:DENTEX_DEFAULT_DIAGNOSES",
    "https://aaoms.org/wp-content/uploads/2024/07/impacted_third_molars.pdf"
  ]
}
---

1. The dataset label Impacted is normalized for evaluation; the agent prompt permits Impacted Tooth. [S2]

2. Locate the requested tooth and inspect its crop before diagnosing. A detector error is not a clinical finding. [S1]

3. AAOMS defines impaction as inability to erupt into a normal oral position, potentially related to space, abnormal position or pathology. A tooth missing from a crop is not evidence of impaction. [S3]

4. Inspect the localized tooth and adjacent anatomy with the full panoramic context retained; AAOMS notes that panoramic imaging is commonly used but radiographs may not fully establish position or the surgical approach. Do not infer a treatment indication from this classification output. [S3]

5. If locate_tooth fails for the requested tooth, do not repeat the same failed detection; inspect a localized neighboring region or improve visibility with the existing contrast/window tools. [S1]
