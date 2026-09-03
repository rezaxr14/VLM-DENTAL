# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

| Method / Model                      | Format Ok (%)   | FDI Acc (%)   |   FDI F1 | Pathology Acc (%)   |   Pathology F1 | Exact Match (%)   | Exact Match 95% CI   |   Closeness |   ECE |   Avg Tool Calls |
|-------------------------------------|-----------------|---------------|----------|---------------------|----------------|-------------------|----------------------|-------------|-------|------------------|
| Zero-Shot (gemini/gemini-3.5-flash) | 100.0%          | 70.0%         |    0.496 | 40.0%               |            0.4 | 40.0%             | [0.100, 0.700]       |       0.639 | 0.528 |                0 |

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllrlrllrrr}
\hline
 Method / Model                      & Format Ok (\%)   & FDI Acc (\%)   &   FDI F1 & Pathology Acc (\%)   &   Pathology F1 & Exact Match (\%)   & Exact Match 95\% CI   &   Closeness &   ECE &   Avg Tool Calls \\
\hline
 Zero-Shot (gemini/gemini-3.5-flash) & 100.0\%          & 70.0\%         &    0.496 & 40.0\%               &            0.4 & 40.0\%             & [0.100, 0.700]       &       0.639 & 0.528 &                0 \\
\hline
\end{tabular}
```
