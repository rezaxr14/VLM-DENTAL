# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

| Method / Model                            | Format Ok (%)   | FDI Acc (%)   |   FDI F1 | Pathology Acc (%)   |   Pathology F1 | Exact Match (%)   | Exact Match 95% CI   |   Closeness |    ECE |   Avg Tool Calls |
|-------------------------------------------|-----------------|---------------|----------|---------------------|----------------|-------------------|----------------------|-------------|--------|------------------|
| Zero-Shot (nvidia_nim/moonshotai/kimi-k3) | 100.0%          | 65.2%         |    0.279 | 30.4%               |          0.304 | 30.4%             | [0.174, 0.457]       |       0.521 | 0.3885 |                0 |

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllrlrllrrr}
\hline
 Method / Model                            & Format Ok (\%)   & FDI Acc (\%)   &   FDI F1 & Pathology Acc (\%)   &   Pathology F1 & Exact Match (\%)   & Exact Match 95\% CI   &   Closeness &    ECE &   Avg Tool Calls \\
\hline
 Zero-Shot (nvidia\_nim/moonshotai/kimi-k3) & 100.0\%          & 65.2\%         &    0.279 & 30.4\%               &          0.304 & 30.4\%             & [0.174, 0.457]       &       0.521 & 0.3885 &                0 \\
\hline
\end{tabular}
```
