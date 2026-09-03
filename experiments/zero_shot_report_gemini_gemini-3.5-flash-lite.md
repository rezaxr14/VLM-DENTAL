# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

| Method / Model                           | Format Ok (%)   | FDI Acc (%)   |   FDI F1 | Pathology Acc (%)   |   Pathology F1 | Exact Match (%)   | Exact Match 95% CI   |   Closeness |   ECE |   Avg Tool Calls |
|------------------------------------------|-----------------|---------------|----------|---------------------|----------------|-------------------|----------------------|-------------|-------|------------------|
| Zero-Shot (gemini/gemini-3.5-flash-lite) | 100.0%          | 58.7%         |     0.29 | 34.8%               |          0.348 | 34.8%             | [0.217, 0.478]       |       0.533 | 0.565 |                0 |

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllrlrllrrr}
\hline
 Method / Model                           & Format Ok (\%)   & FDI Acc (\%)   &   FDI F1 & Pathology Acc (\%)   &   Pathology F1 & Exact Match (\%)   & Exact Match 95\% CI   &   Closeness &   ECE &   Avg Tool Calls \\
\hline
 Zero-Shot (gemini/gemini-3.5-flash-lite) & 100.0\%          & 58.7\%         &     0.29 & 34.8\%               &          0.348 & 34.8\%             & [0.217, 0.478]       &       0.533 & 0.565 &                0 \\
\hline
\end{tabular}
```
