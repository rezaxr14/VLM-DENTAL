# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

| Method / Model                      | Format Ok (%)   | FDI Acc (%)   |   FDI F1 | Pathology Acc (%)   |   Pathology F1 | Exact Match (%)   | Exact Match 95% CI   |   Closeness |   ECE |   Avg Tool Calls |
|-------------------------------------|-----------------|---------------|----------|---------------------|----------------|-------------------|----------------------|-------------|-------|------------------|
| Zero-Shot (gemini/gemini-3.5-flash) | 100.0%          | 72.4%         |    0.421 | 44.8%               |          0.448 | 44.8%             | [0.276, 0.622]       |       0.533 | 0.479 |                0 |

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllrlrllrrr}
\hline
 Method / Model                      & Format Ok (\%)   & FDI Acc (\%)   &   FDI F1 & Pathology Acc (\%)   &   Pathology F1 & Exact Match (\%)   & Exact Match 95\% CI   &   Closeness &   ECE &   Avg Tool Calls \\
\hline
 Zero-Shot (gemini/gemini-3.5-flash) & 100.0\%          & 72.4\%         &    0.421 & 44.8\%               &          0.448 & 44.8\%             & [0.276, 0.622]       &       0.533 & 0.479 &                0 \\
\hline
\end{tabular}
```
