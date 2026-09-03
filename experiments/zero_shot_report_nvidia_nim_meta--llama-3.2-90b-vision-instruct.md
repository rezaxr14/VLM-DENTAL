# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

| Method / Model                                            | Format Ok (%)   | FDI Acc (%)   |   FDI F1 | Pathology Acc (%)   |   Pathology F1 | Exact Match (%)   | Exact Match 95% CI   |   Closeness |    ECE |   Avg Tool Calls |
|-----------------------------------------------------------|-----------------|---------------|----------|---------------------|----------------|-------------------|----------------------|-------------|--------|------------------|
| Zero-Shot (nvidia_nim/meta/llama-3.2-90b-vision-instruct) | 100.0%          | 45.7%         |    0.134 | 37.0%               |           0.37 | 37.0%             | [0.239, 0.522]       |       0.563 | 0.3948 |                0 |

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllrlrllrrr}
\hline
 Method / Model                                            & Format Ok (\%)   & FDI Acc (\%)   &   FDI F1 & Pathology Acc (\%)   &   Pathology F1 & Exact Match (\%)   & Exact Match 95\% CI   &   Closeness &    ECE &   Avg Tool Calls \\
\hline
 Zero-Shot (nvidia\_nim/meta/llama-3.2-90b-vision-instruct) & 100.0\%          & 45.7\%         &    0.134 & 37.0\%               &           0.37 & 37.0\%             & [0.239, 0.522]       &       0.563 & 0.3948 &                0 \\
\hline
\end{tabular}
```
