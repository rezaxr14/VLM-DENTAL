# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

| Method / Model                    | Format Ok (%)   | FDI Acc (%)   |   FDI F1 | Pathology Acc (%)   |   Pathology F1 | Exact Match (%)   | Exact Match 95% CI   |   Closeness |   ECE |   Avg Tool Calls |
|-----------------------------------|-----------------|---------------|----------|---------------------|----------------|-------------------|----------------------|-------------|-------|------------------|
| Zero-Shot (groq/qwen/qwen3.6-27b) | 97.7%           | 63.6%         |    0.267 | 40.9%               |          0.409 | 40.9%             | [0.273, 0.568]       |        0.49 |     0 |                0 |

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllrlrllrrr}
\hline
 Method / Model                    & Format Ok (\%)   & FDI Acc (\%)   &   FDI F1 & Pathology Acc (\%)   &   Pathology F1 & Exact Match (\%)   & Exact Match 95\% CI   &   Closeness &   ECE &   Avg Tool Calls \\
\hline
 Zero-Shot (groq/qwen/qwen3.6-27b) & 97.7\%           & 63.6\%         &    0.267 & 40.9\%               &          0.409 & 40.9\%             & [0.273, 0.568]       &        0.49 &     0 &                0 \\
\hline
\end{tabular}
```
