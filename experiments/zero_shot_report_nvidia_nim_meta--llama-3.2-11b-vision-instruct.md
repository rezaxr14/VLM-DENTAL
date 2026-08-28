# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

                                            Method / Model Format Ok (%) FDI Acc (%) FDI F1 Pathology Acc (%) Pathology F1 Exact Match (%) Exact Match 95% CI Closeness    ECE Avg Tool Calls
Zero-Shot (nvidia/nim_meta--llama-3.2-11b-vision-instruct)        100.0%       32.6%  0.110             13.0%        0.130           13.0%     [0.043, 0.217]     0.525 0.5609           0.00

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllllllllll}
\toprule
Method / Model & Format Ok (%) & FDI Acc (%) & FDI F1 & Pathology Acc (%) & Pathology F1 & Exact Match (%) & Exact Match 95% CI & Closeness & ECE & Avg Tool Calls \\
\midrule
Zero-Shot (nvidia/nim_meta--llama-3.2-11b-vision-instruct) & 100.0% & 32.6% & 0.110 & 13.0% & 0.130 & 13.0% & [0.043, 0.217] & 0.525 & 0.5609 & 0.00 \\
\bottomrule
\end{tabular}

```
