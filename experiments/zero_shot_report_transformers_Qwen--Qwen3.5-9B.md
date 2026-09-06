# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

| Method / Model                           | Format Ok (%)   | FDI Acc (%)   |   FDI F1 | Pathology Acc (%)   |   Pathology F1 | Exact Match (%)   | Exact Match 95% CI   |   Closeness |   ECE |   Avg Tool Calls |
|------------------------------------------|-----------------|---------------|----------|---------------------|----------------|-------------------|----------------------|-------------|-------|------------------|
| Zero-Shot (transformers/Qwen/Qwen3.5-9B) | 100.0%          | 53.1%         |    0.204 | 30.6%               |          0.306 | 30.6%             | [0.184, 0.429]       |       0.463 |   0.6 |                0 |

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllrlrllrrr}
\hline
 Method / Model                           & Format Ok (\%)   & FDI Acc (\%)   &   FDI F1 & Pathology Acc (\%)   &   Pathology F1 & Exact Match (\%)   & Exact Match 95\% CI   &   Closeness &   ECE &   Avg Tool Calls \\
\hline
 Zero-Shot (transformers/Qwen/Qwen3.5-9B) & 100.0\%          & 53.1\%         &    0.204 & 30.6\%               &          0.306 & 30.6\%             & [0.184, 0.429]       &       0.463 &   0.6 &                0 \\
\hline
\end{tabular}
```
