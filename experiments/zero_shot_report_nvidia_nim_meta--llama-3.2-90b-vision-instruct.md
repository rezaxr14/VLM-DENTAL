# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

| Method / Model                                            | Format Ok (%)   | FDI Acc (%)   |   FDI F1 | Pathology Acc (%)   |   Pathology F1 | Exact Match (%)   | Exact Match 95% CI   |   Closeness |    ECE |   Avg Tool Calls |
|-----------------------------------------------------------|-----------------|---------------|----------|---------------------|----------------|-------------------|----------------------|-------------|--------|------------------|
| Zero-Shot (nvidia_nim/meta/llama-3.2-90b-vision-instruct) | 100.0%          | 45.7%         |    0.119 | 26.1%               |          0.261 | 26.1%             | [0.152, 0.391]       |       0.546 | 0.5087 |                0 |

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllrlrllrrr}
\hline
 Method / Model                                            & Format Ok (\%)   & FDI Acc (\%)   &   FDI F1 & Pathology Acc (\%)   &   Pathology F1 & Exact Match (\%)   & Exact Match 95\% CI   &   Closeness &    ECE &   Avg Tool Calls \\
\hline
 Zero-Shot (nvidia\_nim/meta/llama-3.2-90b-vision-instruct) & 100.0\%          & 45.7\%         &    0.119 & 26.1\%               &          0.261 & 26.1\%             & [0.152, 0.391]       &       0.546 & 0.5087 &                0 \\
\hline
\end{tabular}
```
