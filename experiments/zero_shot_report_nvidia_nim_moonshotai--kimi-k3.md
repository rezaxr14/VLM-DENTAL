# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

| Method / Model                            | Format Ok (%)   | FDI Acc (%)   |   FDI F1 | Pathology Acc (%)   |   Pathology F1 | Exact Match (%)   | Exact Match 95% CI   |   Closeness |   ECE |   Avg Tool Calls |
|-------------------------------------------|-----------------|---------------|----------|---------------------|----------------|-------------------|----------------------|-------------|-------|------------------|
| Zero-Shot (nvidia_nim/moonshotai/kimi-k3) | 100.0%          | 50.0%         |    0.327 | 50.0%               |            0.5 | 50.0%             | [0.125, 0.875]       |       0.566 |  0.19 |                0 |

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllrlrllrrr}
\hline
 Method / Model                            & Format Ok (\%)   & FDI Acc (\%)   &   FDI F1 & Pathology Acc (\%)   &   Pathology F1 & Exact Match (\%)   & Exact Match 95\% CI   &   Closeness &   ECE &   Avg Tool Calls \\
\hline
 Zero-Shot (nvidia\_nim/moonshotai/kimi-k3) & 100.0\%          & 50.0\%         &    0.327 & 50.0\%               &            0.5 & 50.0\%             & [0.125, 0.875]       &       0.566 &  0.19 &                0 \\
\hline
\end{tabular}
```
