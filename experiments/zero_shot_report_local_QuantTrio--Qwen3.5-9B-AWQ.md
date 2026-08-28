# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

| Method / Model                             | Format Ok (%)   | FDI Acc (%)   |   FDI F1 | Pathology Acc (%)   |   Pathology F1 | Exact Match (%)   | Exact Match 95% CI   |   Closeness |   ECE |   Avg Tool Calls |
|--------------------------------------------|-----------------|---------------|----------|---------------------|----------------|-------------------|----------------------|-------------|-------|------------------|
| Zero-Shot (local/QuantTrio/Qwen3.5-9B-AWQ) | 13.3%           | 6.7%          |    0.019 | 6.7%                |          0.067 | 6.7%              | [0.000, 0.200]       |       0.058 |     0 |                0 |

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllrlrllrrr}
\hline
 Method / Model                             & Format Ok (\%)   & FDI Acc (\%)   &   FDI F1 & Pathology Acc (\%)   &   Pathology F1 & Exact Match (\%)   & Exact Match 95\% CI   &   Closeness &   ECE &   Avg Tool Calls \\
\hline
 Zero-Shot (local/QuantTrio/Qwen3.5-9B-AWQ) & 13.3\%           & 6.7\%          &    0.019 & 6.7\%                &          0.067 & 6.7\%              & [0.000, 0.200]       &       0.058 &     0 &                0 \\
\hline
\end{tabular}
```
