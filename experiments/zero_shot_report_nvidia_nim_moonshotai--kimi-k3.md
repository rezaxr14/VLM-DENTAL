# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

| Method / Model                            | Format Ok (%)   | FDI Acc (%)   |   FDI F1 | Pathology Acc (%)   |   Pathology F1 | Exact Match (%)   | Exact Match 95% CI   |   Closeness |    ECE |   Avg Tool Calls |
|-------------------------------------------|-----------------|---------------|----------|---------------------|----------------|-------------------|----------------------|-------------|--------|------------------|
| Zero-Shot (nvidia_nim/moonshotai/kimi-k3) | 100.0%          | 61.5%         |    0.264 | 38.5%               |          0.385 | 38.5%             | [0.192, 0.577]       |       0.531 | 0.3165 |                0 |

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllrlrllrrr}
\hline
 Method / Model                            & Format Ok (\%)   & FDI Acc (\%)   &   FDI F1 & Pathology Acc (\%)   &   Pathology F1 & Exact Match (\%)   & Exact Match 95\% CI   &   Closeness &    ECE &   Avg Tool Calls \\
\hline
 Zero-Shot (nvidia\_nim/moonshotai/kimi-k3) & 100.0\%          & 61.5\%         &    0.264 & 38.5\%               &          0.385 & 38.5\%             & [0.192, 0.577]       &       0.531 & 0.3165 &                0 \\
\hline
\end{tabular}
```
