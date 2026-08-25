"""
Unified Click CLI for Dental-Agent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import click
import pandas as pd

from dental_agent.config import load_config, load_env
from dental_agent.utils.environment import get_system_summary

# Ensure .env is loaded before CLI commands run
load_env()
from dental_agent.data.dentex import load_dentex_dataset
from dental_agent.tools.registry import ToolRegistry
from dental_agent.tools.synthetic import make_synthetic_dental_image


@click.group()
@click.option(
    "--config", "-c", "config_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to YAML configuration file.",
)
@click.pass_context
def cli(ctx: click.Context, config_path: str | None) -> None:
    """Dental-Agent: Tool-Augmented Agentic VLM for Panoramic Dental Radiographs."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config_path)


@cli.command()
@click.pass_context
def info(ctx: click.Context) -> None:
    """Print system summary, GPU VRAM, and configuration details."""
    cfg = ctx.obj["config"]
    summary = get_system_summary()

    click.echo("==================================================")
    click.echo("       DENTAL-AGENT SYSTEM & ENVIRONMENT INFO      ")
    click.echo("==================================================")
    for k, v in summary.items():
        click.echo(f"  {k:<20}: {v}")
    click.echo("--------------------------------------------------")
    click.echo(f"  Base Model          : {cfg.model.name}")
    click.echo(f"  Quantization (4bit) : {cfg.model.load_in_4bit}")
    click.echo(f"  Persist Directory   : {cfg.persist_dir}")
    click.echo("==================================================")


@cli.command()
def test() -> None:
    """Execute fast offline self-tests across tools, parsing, and geometry."""
    click.echo("Running Dental-Agent offline self-tests...")
    import subprocess
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


@cli.command()
@click.option("--n-samples", default=20, help="Number of expert traces to generate.")
@click.option("--output", "-o", default="data/synthetic_traces.jsonl", help="Output JSONL path.")
@click.pass_context
def generate_traces(ctx: click.Context, n_samples: int, output: str) -> None:
    """Aim 1: Generate synthetic expert demonstration traces with cross-family verification."""
    cfg = ctx.obj["config"]
    images_df, annots_df, categories_df = load_dentex_dataset(cfg.persist_dir)

    from dental_agent.training.trace_generation import generate_trace_dataset
    generate_trace_dataset(
        images_df=images_df,
        annots_df=annots_df,
        categories_df=categories_df,
        output_jsonl=output,
        n_samples=n_samples,
    )


@cli.command()
@click.option("--data", "-d", default="data/synthetic_traces.jsonl", help="Path to trace JSONL dataset.")
@click.option("--epochs", default=3, help="Number of SFT training epochs.")
@click.option("--lr", default=2e-5, help="Learning rate.")
@click.pass_context
def train_sft_cmd(ctx: click.Context, data: str, epochs: int, lr: float) -> None:
    """Stage 1: Run Supervised Fine-Tuning (SFT) on expert traces."""
    cfg = ctx.obj["config"]
    from dental_agent.training.sft import train_sft
    train_sft(
        data_path=data,
        config=cfg,
        epochs=epochs,
        learning_rate=lr,
    )


@cli.command()
@click.option("--group-size", "-g", default=4, help="GRPO group size (rollouts per image).")
@click.option("--epochs-per-batch", default=2, help="Policy update epochs per sample batch.")
@click.pass_context
def train_grpo_cmd(ctx: click.Context, group_size: int, epochs_per_batch: int) -> None:
    """Stage 2: Run Group Relative Policy Optimization (GRPO) training."""
    cfg = ctx.obj["config"]
    images_df, annots_df, categories_df = load_dentex_dataset(cfg.persist_dir)

    from dental_agent.training.grpo import train_grpo
    train_grpo(
        images_df=images_df,
        annots_df=annots_df,
        categories_df=categories_df,
        config=cfg,
        group_size=group_size,
        epochs_per_batch=epochs_per_batch,
    )


@cli.command()
@click.option("--checkpoint-tag", default=None, help="Specific checkpoint tag to evaluate.")
@click.option("--limit", default=None, type=int, help="Sample limit for fast evaluation.")
@click.option("--output-report", default="experiments/evaluation_report.md", help="Path for report.")
@click.pass_context
def evaluate(ctx: click.Context, checkpoint_tag: str | None, limit: int | None, output_report: str) -> None:
    """Run full evaluation suite across test set and generate summary reports."""
    cfg = ctx.obj["config"]
    images_df, annots_df, _ = load_dentex_dataset(cfg.persist_dir)

    from dental_agent.evaluation.batch_runner import evaluate_dataset
    from dental_agent.evaluation.reporting import generate_markdown_report

    metrics = evaluate_dataset(
        images_df=images_df,
        annots_df=annots_df,
        checkpoint_tag=checkpoint_tag,
        sample_limit=limit,
    )

    tag_name = checkpoint_tag or "Dental-Agent (Default)"
    report_path = generate_markdown_report({tag_name: metrics}, output_path=output_report)
    click.echo(f"Evaluation complete. Report generated at: {report_path}")


@cli.command()
@click.option("--output-dir", default="experiments/sweep", help="Directory for sweep results.")
@click.pass_context
def sweep(ctx: click.Context, output_dir: str) -> None:
    """Run hyperparameter sweep over GRPO configurations."""
    cfg = ctx.obj["config"]
    images_df, annots_df, categories_df = load_dentex_dataset(cfg.persist_dir)

    from dental_agent.evaluation.sweep import run_hyperparameter_sweep
    run_hyperparameter_sweep(
        images_df=images_df,
        annots_df=annots_df,
        categories_df=categories_df,
        holdout_images_df=images_df,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    cli()
