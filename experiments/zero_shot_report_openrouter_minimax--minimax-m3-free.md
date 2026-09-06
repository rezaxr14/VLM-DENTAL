# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

| Method / Model                                 | Format Ok (%)   | FDI Acc (%)   |   FDI F1 | Pathology Acc (%)   |   Pathology F1 | Exact Match (%)   | Exact Match 95% CI   |   Closeness |    ECE |   Avg Tool Calls |
|------------------------------------------------|-----------------|---------------|----------|---------------------|----------------|-------------------|----------------------|-------------|--------|------------------|
| Zero-Shot (openrouter/minimax/minimax-m3:free) | 100.0%          | 62.0%         |    0.249 | 32.0%               |           0.32 | 32.0%             | [0.200, 0.440]       |       0.512 | 0.4932 |                0 |

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllrlrllrrr}
\hline
 Method / Model                                 & Format Ok (\%)   & FDI Acc (\%)   &   FDI F1 & Pathology Acc (\%)   &   Pathology F1 & Exact Match (\%)   & Exact Match 95\% CI   &   Closeness &    ECE &   Avg Tool Calls \\
\hline
 Zero-Shot (openrouter/minimax/minimax-m3:free) & 100.0\%          & 62.0\%         &    0.249 & 32.0\%               &           0.32 & 32.0\%             & [0.200, 0.440]       &       0.512 & 0.4932 &                0 \\
\hline
\end{tabular}
```
