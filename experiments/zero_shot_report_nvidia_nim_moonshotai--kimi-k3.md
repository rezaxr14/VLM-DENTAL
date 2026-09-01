# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

| Method / Model                            | Format Ok (%)   | FDI Acc (%)   |   FDI F1 | Pathology Acc (%)   |   Pathology F1 | Exact Match (%)   | Exact Match 95% CI   |   Closeness |    ECE |   Avg Tool Calls |
|-------------------------------------------|-----------------|---------------|----------|---------------------|----------------|-------------------|----------------------|-------------|--------|------------------|
| Zero-Shot (nvidia_nim/moonshotai/kimi-k3) | 100.0%          | 65.8%         |    0.283 | 31.6%               |          0.316 | 31.6%             | [0.158, 0.474]       |       0.523 | 0.3755 |                0 |

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllrlrllrrr}
\hline
 Method / Model                            & Format Ok (\%)   & FDI Acc (\%)   &   FDI F1 & Pathology Acc (\%)   &   Pathology F1 & Exact Match (\%)   & Exact Match 95\% CI   &   Closeness &    ECE &   Avg Tool Calls \\
\hline
 Zero-Shot (nvidia\_nim/moonshotai/kimi-k3) & 100.0\%          & 65.8\%         &    0.283 & 31.6\%               &          0.316 & 31.6\%             & [0.158, 0.474]       &       0.523 & 0.3755 &                0 \\
\hline
\end{tabular}
```
