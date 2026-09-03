# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

| Method / Model                      | Format Ok (%)   | FDI Acc (%)   |   FDI F1 | Pathology Acc (%)   |   Pathology F1 | Exact Match (%)   | Exact Match 95% CI   |   Closeness |   ECE |   Avg Tool Calls |
|-------------------------------------|-----------------|---------------|----------|---------------------|----------------|-------------------|----------------------|-------------|-------|------------------|
| Zero-Shot (gemini/gemini-3.5-flash) | 100.0%          | 66.7%         |    0.406 | 40.0%               |            0.4 | 40.0%             | [0.133, 0.667]       |       0.543 | 0.534 |                0 |

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllrlrllrrr}
\hline
 Method / Model                      & Format Ok (\%)   & FDI Acc (\%)   &   FDI F1 & Pathology Acc (\%)   &   Pathology F1 & Exact Match (\%)   & Exact Match 95\% CI   &   Closeness &   ECE &   Avg Tool Calls \\
\hline
 Zero-Shot (gemini/gemini-3.5-flash) & 100.0\%          & 66.7\%         &    0.406 & 40.0\%               &            0.4 & 40.0\%             & [0.133, 0.667]       &       0.543 & 0.534 &                0 \\
\hline
\end{tabular}
```
