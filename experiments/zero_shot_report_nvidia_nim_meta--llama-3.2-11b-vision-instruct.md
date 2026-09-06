# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

| Method / Model                                            | Format Ok (%)   | FDI Acc (%)   |   FDI F1 | Pathology Acc (%)   |   Pathology F1 | Exact Match (%)   | Exact Match 95% CI   |   Closeness |   ECE |   Avg Tool Calls |
|-----------------------------------------------------------|-----------------|---------------|----------|---------------------|----------------|-------------------|----------------------|-------------|-------|------------------|
| Zero-Shot (nvidia_nim/meta/llama-3.2-11b-vision-instruct) | 100.0%          | 26.0%         |    0.077 | 4.0%                |           0.04 | 4.0%              | [0.000, 0.100]       |        0.48 | 0.678 |                0 |

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllrlrllrrr}
\hline
 Method / Model                                            & Format Ok (\%)   & FDI Acc (\%)   &   FDI F1 & Pathology Acc (\%)   &   Pathology F1 & Exact Match (\%)   & Exact Match 95\% CI   &   Closeness &   ECE &   Avg Tool Calls \\
\hline
 Zero-Shot (nvidia\_nim/meta/llama-3.2-11b-vision-instruct) & 100.0\%          & 26.0\%         &    0.077 & 4.0\%                &           0.04 & 4.0\%              & [0.000, 0.100]       &        0.48 & 0.678 &                0 \\
\hline
\end{tabular}
```
