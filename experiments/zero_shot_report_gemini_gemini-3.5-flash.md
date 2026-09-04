# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

| Method / Model                      | Format Ok (%)   | FDI Acc (%)   |   FDI F1 | Pathology Acc (%)   |   Pathology F1 | Exact Match (%)   | Exact Match 95% CI   |   Closeness |   ECE |   Avg Tool Calls |
|-------------------------------------|-----------------|---------------|----------|---------------------|----------------|-------------------|----------------------|-------------|-------|------------------|
| Zero-Shot (gemini/gemini-3.5-flash) | 100.0%          | 68.4%         |    0.399 | 42.1%               |          0.421 | 42.1%             | [0.211, 0.632]       |       0.541 |  0.51 |                0 |

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllrlrllrrr}
\hline
 Method / Model                      & Format Ok (\%)   & FDI Acc (\%)   &   FDI F1 & Pathology Acc (\%)   &   Pathology F1 & Exact Match (\%)   & Exact Match 95\% CI   &   Closeness &   ECE &   Avg Tool Calls \\
\hline
 Zero-Shot (gemini/gemini-3.5-flash) & 100.0\%          & 68.4\%         &    0.399 & 42.1\%               &          0.421 & 42.1\%             & [0.211, 0.632]       &       0.541 &  0.51 &                0 \\
\hline
\end{tabular}
```
