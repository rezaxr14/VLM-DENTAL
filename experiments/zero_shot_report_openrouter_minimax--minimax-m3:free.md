# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

| Method / Model                                 | Format Ok (%)   | FDI Acc (%)   |   FDI F1 | Pathology Acc (%)   |   Pathology F1 | Exact Match (%)   | Exact Match 95% CI   |   Closeness |   ECE |   Avg Tool Calls |
|------------------------------------------------|-----------------|---------------|----------|---------------------|----------------|-------------------|----------------------|-------------|-------|------------------|
| Zero-Shot (openrouter/minimax/minimax-m3:free) | 95.7%           | 60.9%         |    0.247 | 41.3%               |          0.413 | 41.3%             | [0.261, 0.565]       |       0.543 | 0.394 |                0 |

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllrlrllrrr}
\hline
 Method / Model                                 & Format Ok (\%)   & FDI Acc (\%)   &   FDI F1 & Pathology Acc (\%)   &   Pathology F1 & Exact Match (\%)   & Exact Match 95\% CI   &   Closeness &   ECE &   Avg Tool Calls \\
\hline
 Zero-Shot (openrouter/minimax/minimax-m3:free) & 95.7\%           & 60.9\%         &    0.247 & 41.3\%               &          0.413 & 41.3\%             & [0.261, 0.565]       &       0.543 & 0.394 &                0 \\
\hline
\end{tabular}
```
