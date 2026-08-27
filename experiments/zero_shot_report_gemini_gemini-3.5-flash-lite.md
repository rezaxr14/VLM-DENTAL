# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

| Method / Model                           | Format Ok (%)   | FDI Acc (%)   |   FDI F1 | Pathology Acc (%)   |   Pathology F1 | Exact Match (%)   | Exact Match 95% CI   |   Closeness |    ECE |   Avg Tool Calls |
|------------------------------------------|-----------------|---------------|----------|---------------------|----------------|-------------------|----------------------|-------------|--------|------------------|
| Zero-Shot (gemini/gemini-3.5-flash-lite) | 100.0%          | 54.3%         |    0.219 | 39.1%               |          0.391 | 39.1%             | [0.261, 0.522]       |       0.488 | 0.5302 |                0 |

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllrlrllrrr}
\hline
 Method / Model                           & Format Ok (\%)   & FDI Acc (\%)   &   FDI F1 & Pathology Acc (\%)   &   Pathology F1 & Exact Match (\%)   & Exact Match 95\% CI   &   Closeness &    ECE &   Avg Tool Calls \\
\hline
 Zero-Shot (gemini/gemini-3.5-flash-lite) & 100.0\%          & 54.3\%         &    0.219 & 39.1\%               &          0.391 & 39.1\%             & [0.261, 0.522]       &       0.488 & 0.5302 &                0 \\
\hline
\end{tabular}
```
