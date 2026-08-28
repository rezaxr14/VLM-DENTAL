# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

                    Method / Model Format Ok (%) FDI Acc (%) FDI F1 Pathology Acc (%) Pathology F1 Exact Match (%) Exact Match 95% CI Closeness    ECE Avg Tool Calls
Zero-Shot (groq/qwen--qwen3.6-27b)        100.0%       66.7%  0.268             37.0%        0.370           37.0%     [0.185, 0.556]     0.485 0.4296           0.00

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllllllllll}
\toprule
Method / Model & Format Ok (%) & FDI Acc (%) & FDI F1 & Pathology Acc (%) & Pathology F1 & Exact Match (%) & Exact Match 95% CI & Closeness & ECE & Avg Tool Calls \\
\midrule
Zero-Shot (groq/qwen--qwen3.6-27b) & 100.0% & 66.7% & 0.268 & 37.0% & 0.370 & 37.0% & [0.185, 0.556] & 0.485 & 0.4296 & 0.00 \\
\bottomrule
\end{tabular}

```
