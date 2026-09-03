# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

| Method / Model                                 | Format Ok (%)   | FDI Acc (%)   |   FDI F1 | Pathology Acc (%)   |   Pathology F1 | Exact Match (%)   | Exact Match 95% CI   |   Closeness |    ECE |   Avg Tool Calls |
|------------------------------------------------|-----------------|---------------|----------|---------------------|----------------|-------------------|----------------------|-------------|--------|------------------|
| Zero-Shot (openrouter/minimax/minimax-m3:free) | 100.0%          | 65.2%         |    0.258 | 32.6%               |          0.326 | 32.6%             | [0.196, 0.457]       |       0.539 | 0.4813 |                0 |

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllrlrllrrr}
\hline
 Method / Model                                 & Format Ok (\%)   & FDI Acc (\%)   &   FDI F1 & Pathology Acc (\%)   &   Pathology F1 & Exact Match (\%)   & Exact Match 95\% CI   &   Closeness &    ECE &   Avg Tool Calls \\
\hline
 Zero-Shot (openrouter/minimax/minimax-m3:free) & 100.0\%          & 65.2\%         &    0.258 & 32.6\%               &          0.326 & 32.6\%             & [0.196, 0.457]       &       0.539 & 0.4813 &                0 \\
\hline
\end{tabular}
```
