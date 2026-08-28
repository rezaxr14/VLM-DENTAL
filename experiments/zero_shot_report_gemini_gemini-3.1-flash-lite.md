# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

| Method / Model                           | Format Ok (%)   | FDI Acc (%)   |   FDI F1 | Pathology Acc (%)   |   Pathology F1 | Exact Match (%)   | Exact Match 95% CI   |   Closeness |    ECE |   Avg Tool Calls |
|------------------------------------------|-----------------|---------------|----------|---------------------|----------------|-------------------|----------------------|-------------|--------|------------------|
| Zero-Shot (gemini/gemini-3.1-flash-lite) | 100.0%          | 58.7%         |    0.264 | 32.6%               |          0.326 | 32.6%             | [0.196, 0.457]       |       0.482 | 0.6289 |                0 |

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllrlrllrrr}
\hline
 Method / Model                           & Format Ok (\%)   & FDI Acc (\%)   &   FDI F1 & Pathology Acc (\%)   &   Pathology F1 & Exact Match (\%)   & Exact Match 95\% CI   &   Closeness &    ECE &   Avg Tool Calls \\
\hline
 Zero-Shot (gemini/gemini-3.1-flash-lite) & 100.0\%          & 58.7\%         &    0.264 & 32.6\%               &          0.326 & 32.6\%             & [0.196, 0.457]       &       0.482 & 0.6289 &                0 \\
\hline
\end{tabular}
```
