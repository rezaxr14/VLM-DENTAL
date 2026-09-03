# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

| Method / Model                                            | Format Ok (%)   | FDI Acc (%)   |   FDI F1 | Pathology Acc (%)   |   Pathology F1 | Exact Match (%)   | Exact Match 95% CI   |   Closeness |    ECE |   Avg Tool Calls |
|-----------------------------------------------------------|-----------------|---------------|----------|---------------------|----------------|-------------------|----------------------|-------------|--------|------------------|
| Zero-Shot (nvidia_nim/meta/llama-3.2-11b-vision-instruct) | 100.0%          | 28.3%         |    0.084 | 4.3%                |          0.043 | 4.3%              | [0.000, 0.109]       |       0.522 | 0.6652 |                0 |

---

## LaTeX Source Table (for Publication)

```latex
\begin{tabular}{lllrlrllrrr}
\hline
 Method / Model                                            & Format Ok (\%)   & FDI Acc (\%)   &   FDI F1 & Pathology Acc (\%)   &   Pathology F1 & Exact Match (\%)   & Exact Match 95\% CI   &   Closeness &    ECE &   Avg Tool Calls \\
\hline
 Zero-Shot (nvidia\_nim/meta/llama-3.2-11b-vision-instruct) & 100.0\%          & 28.3\%         &    0.084 & 4.3\%                &          0.043 & 4.3\%              & [0.000, 0.109]       &       0.522 & 0.6652 &                0 \\
\hline
\end{tabular}
```
