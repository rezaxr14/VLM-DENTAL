# Reusable paper rebuild / editing prompt

Read paper/README.md and the actual rules/skills/serving implementation. Work from 0048 with
0050 (evaluation) and 0051 (replacement paper), not rejected 0049.

Keep every original diagram in its original section/order. Keep the frozen sec:eval protocol
verbatim. Retain the detailed source-grounded mechanism figures and regenerate SVG/PDF twins
from build_mechanism_diagrams.py when editing. Show mechanisms, real observations and concrete
interfaces; do not substitute empty flowchart boxes or invent clinical/efficiency numbers.

Keep teacher corpus evidence, pending student SFT/GRPO outcomes, inference-time skills and
same-target serving speed separate. New comparisons remain exploratory and dev-only. Import
measured result JSON through the shared generator; keep absent outcomes/counters red pending.
Generate quantitative panels only from those inputs, with hashes/cohort sizes and failed gates
visible. Never tune prompts/checkpoints on the official 50-image final test set. Do not label
Tufts untouched external validation when it has contributed training data.

Verify new citations. Preserve old measured scores unless supplied replacement result evidence
is explicitly identified. The legacy bibliography TODOs and affiliations still need author review.
Do not change model/training scripts, dependency constraints, token headroom, notebooks or jobs
while rebuilding the manuscript. Do not run inference or training as a PDF build side effect.

Run paper tests, verify_paper.py, SVG text/arrow audits and build_paper.py. Use the isolated
staged TeX build; never compile against tracked stale .aux/.bbl products. Read individual pass
logs if citations/references fail, fix real missing keys, and render/inspect all changed pages.
Deliver the compiled PDF, a clean git-apply patch, exact commands and the evidence gaps. Use
Reza Nadimi <rezaxr14@gmail.com> as patch author; do not commit or push without instruction.
