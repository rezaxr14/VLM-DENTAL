# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

| Method / Model                                 | Format Ok (%)   | FDI Acc (%)   |   FDI F1 | Pathology Acc (%)   |   Pathology F1 | Exact Match (%)   | Exact Match 95% CI   |   Closeness |    ECE |   Avg Tool Calls |
|------------------------------------------------|-----------------|---------------|----------|---------------------|----------------|-------------------|----------------------|-------------|--------|------------------|
| Zero-Shot (openrouter/minimax/minimax-m3:free) | 100.0%          | 67.4%         |    0.271 | 34.8%               |          0.348 | 34.8%             | [0.217, 0.479]       |       0.556 | 0.4654 |                0 |

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllrlrllrrr}
\hline
 Method / Model                                 & Format Ok (\%)   & FDI Acc (\%)   &   FDI F1 & Pathology Acc (\%)   &   Pathology F1 & Exact Match (\%)   & Exact Match 95\% CI   &   Closeness &    ECE &   Avg Tool Calls \\
\hline
 Zero-Shot (openrouter/minimax/minimax-m3:free) & 100.0\%          & 67.4\%         &    0.271 & 34.8\%               &          0.348 & 34.8\%             & [0.217, 0.479]       &       0.556 & 0.4654 &                0 \\
\hline
\end{tabular}
```
