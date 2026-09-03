# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

| Method / Model                           | Format Ok (%)   | FDI Acc (%)   |   FDI F1 | Pathology Acc (%)   |   Pathology F1 | Exact Match (%)   | Exact Match 95% CI   |   Closeness |    ECE |   Avg Tool Calls |
|------------------------------------------|-----------------|---------------|----------|---------------------|----------------|-------------------|----------------------|-------------|--------|------------------|
| Zero-Shot (gemini/gemini-3.5-flash-lite) | 100.0%          | 54.3%         |     0.27 | 30.4%               |          0.304 | 30.4%             | [0.174, 0.435]       |       0.497 | 0.5914 |                0 |

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllrlrllrrr}
\hline
 Method / Model                           & Format Ok (\%)   & FDI Acc (\%)   &   FDI F1 & Pathology Acc (\%)   &   Pathology F1 & Exact Match (\%)   & Exact Match 95\% CI   &   Closeness &    ECE &   Avg Tool Calls \\
\hline
 Zero-Shot (gemini/gemini-3.5-flash-lite) & 100.0\%          & 54.3\%         &     0.27 & 30.4\%               &          0.304 & 30.4\%             & [0.174, 0.435]       &       0.497 & 0.5914 &                0 \\
\hline
\end{tabular}
```
