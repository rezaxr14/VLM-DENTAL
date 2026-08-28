# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

| Method / Model                                            | Format Ok (%)   | FDI Acc (%)   |   FDI F1 | Pathology Acc (%)   |   Pathology F1 | Exact Match (%)   | Exact Match 95% CI   |   Closeness |    ECE |   Avg Tool Calls |
|-----------------------------------------------------------|-----------------|---------------|----------|---------------------|----------------|-------------------|----------------------|-------------|--------|------------------|
| Zero-Shot (nvidia_nim/meta/llama-3.2-90b-vision-instruct) | 100.0%          | 43.5%         |    0.115 | 23.9%               |          0.239 | 23.9%             | [0.130, 0.370]       |       0.532 | 0.5156 |                0 |

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllrlrllrrr}
\hline
 Method / Model                                            & Format Ok (\%)   & FDI Acc (\%)   &   FDI F1 & Pathology Acc (\%)   &   Pathology F1 & Exact Match (\%)   & Exact Match 95\% CI   &   Closeness &    ECE &   Avg Tool Calls \\
\hline
 Zero-Shot (nvidia\_nim/meta/llama-3.2-90b-vision-instruct) & 100.0\%          & 43.5\%         &    0.115 & 23.9\%               &          0.239 & 23.9\%             & [0.130, 0.370]       &       0.532 & 0.5156 &                0 \\
\hline
\end{tabular}
```
