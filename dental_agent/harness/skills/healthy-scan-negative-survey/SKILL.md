---
{
  "name": "healthy-scan-negative-survey",
  "description": "Survey all quadrants and return an empty findings list only when the examination supports it.",
  "version": "1.2.0",
  "sources": [
    "dental_agent/agent/prompts.py:build_agent_system_prompt",
    "dental_agent/training/trace_generation.py:verify_trace",
    "https://www.fda.gov/radiation-emitting-products/medical-x-ray-imaging/selection-patients-dental-radiographic-examinations"
  ]
}
---

1. A patient may have multiple findings; report all investigated findings. [S1]

2. For a clinically healthy dentition with no identifiable target pathologies, return final_answer: []. [S1]

3. The verifier rejects a nonempty candidate final_answer when the supplied ground truth is empty; ground truth is verifier-only and must never be exposed to the policy. [S2]

4. Screen all four quadrants, retain the full-image context, and investigate suspicious regions before returning an empty target-findings list. Do not stop the survey after confirming one normal crop. [S1]

5. Use window_level for unclear density, denoise for a noisy view and contrast enhancement for low contrast; verify any investigated crop still shows the intended tooth. [S1]

6. An empty target-findings list is not a claim that panoramic imaging excludes every dental disease. The FDA/ADA guidance uses clinical context and selected intraoral views, including bitewings for proximal caries, rather than treating a panoramic survey as a universal negative test. [S3]
