# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

| Method / Model                      | Format Ok (%)   | FDI Acc (%)   |   FDI F1 | Pathology Acc (%)   |   Pathology F1 | Exact Match (%)   | Exact Match 95% CI   |   Closeness |    ECE |   Avg Tool Calls |
|-------------------------------------|-----------------|---------------|----------|---------------------|----------------|-------------------|----------------------|-------------|--------|------------------|
| Zero-Shot (gemini/gemini-3.5-flash) | 100.0%          | 62.9%         |    0.303 | 31.4%               |          0.314 | 31.4%             | [0.171, 0.486]       |        0.48 | 0.6029 |                0 |

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllrlrllrrr}
\hline
 Method / Model                      & Format Ok (\%)   & FDI Acc (\%)   &   FDI F1 & Pathology Acc (\%)   &   Pathology F1 & Exact Match (\%)   & Exact Match 95\% CI   &   Closeness &    ECE &   Avg Tool Calls \\
\hline
 Zero-Shot (gemini/gemini-3.5-flash) & 100.0\%          & 62.9\%         &    0.303 & 31.4\%               &          0.314 & 31.4\%             & [0.171, 0.486]       &        0.48 & 0.6029 &                0 \\
\hline
\end{tabular}
```
