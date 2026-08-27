"""
Paper-ready summary table generation and Markdown reporting (§26, §29).

Includes:
- Structured DataFrame summary from metrics dictionary (`metrics_to_dataframe`)
- Persistent JSON and Markdown results exporter (`save_results_report`)
- Publication table generator (`generate_summary_table`, `generate_markdown_report`)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import pandas as pd


def metrics_to_dataframe(named_metrics: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Convert a dictionary of named metrics into a standardized DataFrame."""
    rows = []
    for name, m in named_metrics.items():
        rows.append({
            "condition": name,
            "n": m.get("n_examples"),
            "fdi_accuracy": m.get("fdi_accuracy"),
            "balanced_accuracy": m.get("diagnosis_balanced_accuracy"),
            "format_compliance": m.get("format_compliance_rate"),
            "mean_reward": m.get("mean_reward"),
            "ECE": m.get("expected_calibration_error"),
        })
    return pd.DataFrame(rows).set_index("condition")


def save_results_report(
    named_metrics: dict[str, dict[str, Any]],
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Save evaluation metrics as both JSON and Markdown tables."""
    default_dir = os.environ.get("DENTAL_AGENT_DATA_DIR", "data")
    path_str = str(path or os.path.join(default_dir, "results_report"))
    base_path = path_str.replace(".json", "").replace(".md", "")
    os.makedirs(os.path.dirname(base_path) or ".", exist_ok=True)

    df = metrics_to_dataframe(named_metrics)
    df.to_json(base_path + ".json", orient="index", indent=2)
    with open(base_path + ".md", "w") as f:
        f.write("# Evaluation results\n\n")
        try:
            f.write(df.to_markdown())
        except Exception:
            f.write(df.to_string())
    print(f"Saved {base_path}.json and {base_path}.md")
    return df


def generate_summary_table(
    results: dict[str, dict[str, Any]],
    table_format: str = "github",
) -> str:
    """Format comparative evaluation metrics into a publication-quality table."""
    rows = []
    for model_name, metrics in results.items():
        ci = metrics.get("exact_match_ci_95", [0.0, 0.0])
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "N/A"
        row = {
            "Method / Model": model_name,
            "Format Ok (%)": f"{metrics.get('format_adherence', 0.0) * 100:.1f}%",
            "FDI Acc (%)": f"{metrics.get('fdi_localization_accuracy', 0.0) * 100:.1f}%",
            "FDI F1": f"{metrics.get('fdi_localization_f1', metrics.get('fdi_localization_accuracy', 0.0)):.3f}",
            "Pathology Acc (%)": f"{metrics.get('pathology_accuracy', 0.0) * 100:.1f}%",
            "Pathology F1": f"{metrics.get('pathology_macro_f1', 0.0):.3f}",
            "Exact Match (%)": f"{metrics.get('exact_match_accuracy', 0.0) * 100:.1f}%",
            "Exact Match 95% CI": ci_str,
            "Closeness": f"{metrics.get('closeness_score', 0.0):.3f}",
            "ECE": f"{metrics.get('ece', 0.0):.4f}",
            "Avg Tool Calls": f"{metrics.get('mean_tool_calls', 0.0):.2f}",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    try:
        from tabulate import tabulate
        return tabulate(df, headers="keys", tablefmt=table_format, showindex=False)
    except ImportError:
        if table_format in ("github", "pipe"):
            try:
                return df.to_markdown(index=False)
            except Exception:
                return df.to_string(index=False)
        elif table_format == "latex":
            try:
                return df.to_latex(index=False)
            except Exception:
                return df.to_string(index=False)
        return df.to_string(index=False)


def generate_markdown_report(
    results: dict[str, dict[str, Any]],
    output_path: str | Path = "experiments/evaluation_report.md",
) -> Path:
    """Generate and write a full Markdown experimental results report."""
    table_md = generate_summary_table(results, table_format="github")
    table_latex = generate_summary_table(results, table_format="latex")

    content = f"""# DENTEX Agentic VLM Experimental Evaluation Report

## Benchmark Performance Summary

{table_md}

---

## LaTeX Source Table (for Publication)

```latex
{table_latex}
```
"""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        f.write(content)

    print(f"Evaluation report written to: {out_p}")
    return out_p
