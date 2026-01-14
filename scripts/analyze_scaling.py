#!/usr/bin/env python
"""Stage 2: Plot scaling laws from compute-optimal runs.

Usage:
    python scripts/analyze_scaling.py logs/scaling_optimal_tokenizer_*

This script:
1. Loads results.csv from compute-optimal experiments
2. Generates scaling law plots:
   - Loss vs total FLOPs (compute scaling)
   - Loss vs training time (wall-clock scaling)
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_runs(logs_dir: Path) -> pd.DataFrame:
    """Load results.csv from experiment directory."""
    csv_path = logs_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No results.csv in {logs_dir}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} runs from {csv_path}")
    return df


def plot_scaling(df: pd.DataFrame, out_dir: Path, metric: str = "final_loss"):
    """Generate loss vs FLOPs and loss vs time plots."""
    df = df.sort_values("params").copy()

    # Compute total FLOPs
    df["total_flops"] = df["flops_per_step"] * df["total_steps"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Metric vs FLOPs
    ax = axes[0]
    ax.scatter(df["total_flops"], df[metric], s=100, c="royalblue", edgecolors="black", zorder=5)

    # Add depth labels
    for _, row in df.iterrows():
        # Extract depth from run_name (e.g., "optimal_tokenizer_d8" -> 8)
        run_name = row["run_name"]
        if "_d" in run_name:
            depth = run_name.split("_d")[-1]
            label = f"d{depth}"
        else:
            label = f"{row['params']/1e6:.1f}M"
        ax.annotate(label, (row["total_flops"], row[metric]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)

    # Fit and extrapolate
    if len(df) >= 2:
        log_flops = np.log10(df["total_flops"].values)
        metric_vals = df[metric].values
        coeffs = np.polyfit(log_flops, metric_vals, 1)
        x_fit = np.logspace(log_flops.min() - 0.3, log_flops.max() + 1.5, 50)
        y_fit = coeffs[0] * np.log10(x_fit) + coeffs[1]
        ax.plot(x_fit, y_fit, "r--", linewidth=2, alpha=0.7, label="Linear fit (log scale)")
        ax.legend()

    ax.set_xscale("log")
    ax.set_xlabel("Total FLOPs")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} vs Compute")
    ax.grid(True, alpha=0.3)

    # Plot 2: Metric vs Time
    ax = axes[1]
    ax.scatter(df["hours"], df[metric], s=100, c="forestgreen", edgecolors="black", zorder=5)

    # Add depth labels
    for _, row in df.iterrows():
        run_name = row["run_name"]
        if "_d" in run_name:
            depth = run_name.split("_d")[-1]
            label = f"d{depth}"
        else:
            label = f"{row['params']/1e6:.1f}M"
        ax.annotate(label, (row["hours"], row[metric]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)

    # Fit and extrapolate
    if len(df) >= 2:
        log_time = np.log10(df["hours"].values)
        metric_vals = df[metric].values
        coeffs = np.polyfit(log_time, metric_vals, 1)
        x_fit = np.logspace(log_time.min() - 0.3, log_time.max() + 1.5, 50)
        y_fit = coeffs[0] * np.log10(x_fit) + coeffs[1]
        ax.plot(x_fit, y_fit, "r--", linewidth=2, alpha=0.7, label="Linear fit (log scale)")
        ax.legend()

    ax.set_xscale("log")
    ax.set_xlabel("Training Time (hours)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} vs Training Time")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "scaling_analysis.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze compute-optimal scaling experiments")
    parser.add_argument("logs_dir", type=Path, help="Path to experiment directory")
    parser.add_argument("--metric", default="final_psnr", help="Metric to plot")
    args = parser.parse_args()

    print("=" * 60)
    print("Compute-Optimal Scaling Analysis")
    print("=" * 60)

    # Load data
    print("\n1. Loading runs...")
    df = load_runs(args.logs_dir)

    if len(df) == 0:
        print("No runs found. Exiting.")
        return

    # Print summary table
    print("\n2. Summary:")
    print("-" * 80)
    summary_cols = ["run_name", "params", "total_steps", "hours", "final_loss", "final_psnr"]
    summary_cols = [c for c in summary_cols if c in df.columns]
    print(df[summary_cols].to_string(index=False))

    # Plot
    print("\n3. Generating plots...")
    plot_scaling(df, args.logs_dir, args.metric)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
