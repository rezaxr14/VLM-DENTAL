# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

| Method / Model                                                            | Format Ok (%)   | FDI Acc (%)   |   FDI F1 | Pathology Acc (%)   |   Pathology F1 | Exact Match (%)   | Exact Match 95% CI   |   Closeness |   ECE |   Avg Tool Calls |
|---------------------------------------------------------------------------|-----------------|---------------|----------|---------------------|----------------|-------------------|----------------------|-------------|-------|------------------|
| Zero-Shot (openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free) | 86.7%           | 46.7%         |    0.131 | 33.3%               |          0.333 | 26.7%             | [0.067, 0.533]       |       0.279 |     0 |                0 |

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllrlrllrrr}
\hline
 Method / Model                                                            & Format Ok (\%)   & FDI Acc (\%)   &   FDI F1 & Pathology Acc (\%)   &   Pathology F1 & Exact Match (\%)   & Exact Match 95\% CI   &   Closeness &   ECE &   Avg Tool Calls \\
\hline
 Zero-Shot (openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free) & 86.7\%           & 46.7\%         &    0.131 & 33.3\%               &          0.333 & 26.7\%             & [0.067, 0.533]       &       0.279 &     0 &                0 \\
\hline
\end{tabular}
```
