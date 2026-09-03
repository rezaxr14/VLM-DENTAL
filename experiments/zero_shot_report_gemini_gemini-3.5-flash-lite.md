# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

| Method / Model                           | Format Ok (%)   | FDI Acc (%)   |   FDI F1 | Pathology Acc (%)   |   Pathology F1 | Exact Match (%)   | Exact Match 95% CI   |   Closeness |    ECE |   Avg Tool Calls |
|------------------------------------------|-----------------|---------------|----------|---------------------|----------------|-------------------|----------------------|-------------|--------|------------------|
| Zero-Shot (gemini/gemini-3.5-flash-lite) | 100.0%          | 53.3%         |    0.263 | 28.9%               |          0.289 | 28.9%             | [0.156, 0.422]       |       0.493 | 0.6083 |                0 |

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllrlrllrrr}
\hline
 Method / Model                           & Format Ok (\%)   & FDI Acc (\%)   &   FDI F1 & Pathology Acc (\%)   &   Pathology F1 & Exact Match (\%)   & Exact Match 95\% CI   &   Closeness &    ECE &   Avg Tool Calls \\
\hline
 Zero-Shot (gemini/gemini-3.5-flash-lite) & 100.0\%          & 53.3\%         &    0.263 & 28.9\%               &          0.289 & 28.9\%             & [0.156, 0.422]       &       0.493 & 0.6083 &                0 \\
\hline
\end{tabular}
```
