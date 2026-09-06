# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

| Method / Model                            | Format Ok (%)   | FDI Acc (%)   |   FDI F1 | Pathology Acc (%)   |   Pathology F1 | Exact Match (%)   | Exact Match 95% CI   |   Closeness |    ECE |   Avg Tool Calls |
|-------------------------------------------|-----------------|---------------|----------|---------------------|----------------|-------------------|----------------------|-------------|--------|------------------|
| Zero-Shot (nvidia_nim/moonshotai/kimi-k3) | 100.0%          | 64.0%         |    0.321 | 46.0%               |           0.46 | 46.0%             | [0.320, 0.600]       |       0.534 | 0.2138 |                0 |

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllrlrllrrr}
\hline
 Method / Model                            & Format Ok (\%)   & FDI Acc (\%)   &   FDI F1 & Pathology Acc (\%)   &   Pathology F1 & Exact Match (\%)   & Exact Match 95\% CI   &   Closeness &    ECE &   Avg Tool Calls \\
\hline
 Zero-Shot (nvidia\_nim/moonshotai/kimi-k3) & 100.0\%          & 64.0\%         &    0.321 & 46.0\%               &           0.46 & 46.0\%             & [0.320, 0.600]       &       0.534 & 0.2138 &                0 \\
\hline
\end{tabular}
```
