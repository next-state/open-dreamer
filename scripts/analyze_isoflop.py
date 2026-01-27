#!/usr/bin/env python
"""Stage 1: Analyze iso-FLOPs experiments to find optimal tokens_per_param.

Usage:
    python scripts/analyze_isoflop.py logs/scaling_isoflop_tokenizer_*

This script:
1. Loads results.csv from the experiment directory
2. Fits quadratic curves to each compute budget
3. Extracts optimal N(C) and D(C) relationships
4. Generates 3 Chinchilla-style plots
5. Outputs the recommended tokens_per_param ratio
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def load_runs(logs_dir: Path) -> pd.DataFrame:
    """Load results.csv from experiment directory."""
    csv_path = logs_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No results.csv in {logs_dir}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} runs from {csv_path}")
    return df


def is_higher_better(metric: str) -> bool:
    """Return True if higher metric values are better (e.g., PSNR)."""
    return "psnr" in metric.lower()


def fit_isoflop_curves(df: pd.DataFrame, metric: str = "final_loss") -> pd.DataFrame:
    """Fit quadratic curves to loss vs params for each FLOPs budget.

    For each compute budget C, fits: loss = a*(log N)^2 + b*(log N) + c
    and finds the optimal N that minimizes loss.
    """
    results = []

    for budget in sorted(df["flops_budget"].unique()):
        if budget == 0:
            continue
        group = df[df["flops_budget"] == budget].dropna(subset=["params", metric])
        if len(group) < 3:
            print(f"  Budget {budget:.2e}: skipping (only {len(group)} points)")
            continue

        # Sort by params for interpolation
        group = group.sort_values("params")
        log_n = np.log10(group["params"].values)
        metric_vals = group[metric].values
        metric_vals = -metric_vals if is_higher_better(metric) else metric_vals
        # Use total_tokens_trained for compute consistency (so N and D exponents sum to ~1)
        tokens = group["total_tokens_trained"].values

        # Quadratic fit: metric = a*x^2 + b*x + c
        try:
            popt, _ = curve_fit(lambda x, a, b, c: a * x**2 + b * x + c, log_n, metric_vals)
            a, b, c = popt

            if a <= 0:
                print(f"  Budget {budget:.2e}: skipping (not a U-curve, a={a:.4f})")
                continue

            # Optimal: minimum at x = -b/(2a)
            log_opt = -b / (2 * a)
            opt_params = 10**log_opt
            opt_loss = a * log_opt**2 + b * log_opt + c
            opt_tokens = np.interp(log_opt, log_n, tokens)  # Interpolate from data

            results.append({
                "flops_budget": budget,
                "opt_params": opt_params,
                "opt_tokens": opt_tokens,
                "opt_loss": opt_loss,
                "tokens_per_param": opt_tokens / opt_params,
                "a": a, "b": b, "c": c,
            })
            print(f"  Budget {budget:.2e}: N*={opt_params:.2e}, D/N={opt_tokens/opt_params:.1f}")

        except Exception as e:
            print(f"  Budget {budget:.2e}: fit failed ({e})")

    return pd.DataFrame(results)


def plot_chinchilla(df: pd.DataFrame, fits: pd.DataFrame, out_dir: Path, metric: str):
    """Generate 3 Chinchilla-style plots."""
    if len(fits) == 0:
        print("No successful fits to plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(fits)))

    # Plot 1: Iso-FLOPs parabolas
    ax = axes[0]
    for i, (_, row) in enumerate(fits.iterrows()):
        budget = row["flops_budget"]
        group = df[df["flops_budget"] == budget].sort_values("params")

        # Data points (negate if higher is better so parabola opens upward)
        metric_vals = -group[metric] if is_higher_better(metric) else group[metric]
        ax.scatter(group["params"], metric_vals, c=[colors[i]], s=60,
                   label=f"C={budget:.0e}", alpha=0.8)

        # Fitted curve
        x = np.logspace(np.log10(group["params"].min()) - 0.1,
                        np.log10(group["params"].max()) + 0.1, 50)
        y = row["a"] * np.log10(x)**2 + row["b"] * np.log10(x) + row["c"]
        ax.plot(x, y, c=colors[i], ls="--", alpha=0.7, linewidth=2)

        # Optimal point (star)
        ax.scatter([row["opt_params"]], [row["opt_loss"]], c=[colors[i]],
                   marker="*", s=200, zorder=10, edgecolors="black")

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (N)")
    ylabel = metric.replace("_", " ").title()
    if is_higher_better(metric):
        ylabel = f"-{ylabel} (lower is better)"
    ax.set_ylabel(ylabel)
    ax.legend(title="Compute Budget", fontsize=8)
    ax.set_title("Iso-FLOPs Curves")
    ax.grid(True, alpha=0.3)

    # Plot 2: Optimal N vs C
    ax = axes[1]
    ax.scatter(fits["flops_budget"], fits["opt_params"], s=80, c="royalblue", edgecolors="black")

    if len(fits) >= 2:
        # Power law fit: N = A * C^alpha
        log_c = np.log10(fits["flops_budget"].values)
        log_n = np.log10(fits["opt_params"].values)
        coeffs = np.polyfit(log_c, log_n, 1)
        alpha = coeffs[0]
        x_fit = np.logspace(log_c.min() - 0.3, log_c.max() + 0.3, 50)
        y_fit = 10**(coeffs[1]) * x_fit**alpha
        ax.plot(x_fit, y_fit, "r--", linewidth=2, label=f"N ~ C^{alpha:.2f}")
        ax.legend()

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute (FLOPs)")
    ax.set_ylabel("Optimal Parameters (N)")
    ax.set_title("Compute-Optimal Model Size")
    ax.grid(True, alpha=0.3)

    # Plot 3: Optimal D vs C
    ax = axes[2]
    ax.scatter(fits["flops_budget"], fits["opt_tokens"], s=80, c="forestgreen", edgecolors="black")

    if len(fits) >= 2:
        # Power law fit: D = A * C^alpha
        log_c = np.log10(fits["flops_budget"].values)
        log_d = np.log10(fits["opt_tokens"].values)
        coeffs = np.polyfit(log_c, log_d, 1)
        alpha = coeffs[0]
        x_fit = np.logspace(log_c.min() - 0.3, log_c.max() + 0.3, 50)
        y_fit = 10**(coeffs[1]) * x_fit**alpha
        ax.plot(x_fit, y_fit, "r--", linewidth=2, label=f"D ~ C^{alpha:.2f}")
        ax.legend()

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute (FLOPs)")
    ax.set_ylabel("Optimal Tokens (D)")
    ax.set_title("Compute-Optimal Training Tokens")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "isoflop_analysis.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze iso-FLOPs experiments")
    parser.add_argument("logs_dir", type=Path, help="Path to experiment directory")
    parser.add_argument("--metric", default="final_psnr", help="Metric to optimize")
    args = parser.parse_args()

    print("=" * 60)
    print("Iso-FLOPs Scaling Analysis")
    print("=" * 60)

    # Load data
    print("\n1. Loading runs...")
    df = load_runs(args.logs_dir)

    if len(df) == 0:
        print("No runs found. Exiting.")
        return

    # Fit curves
    print(f"\n2. Fitting quadratic curves (metric={args.metric})...")
    fits = fit_isoflop_curves(df, args.metric)

    if len(fits) == 0:
        print("No successful fits. Need at least 3 points per compute budget.")
        return

    # Save fit results
    fits_csv = args.logs_dir / "isoflop_fits.csv"
    fits.to_csv(fits_csv, index=False)
    print(f"\nSaved fits to: {fits_csv}")

    # Plot
    print("\n3. Generating plots...")
    plot_chinchilla(df, fits, args.logs_dir, args.metric)

    # Final recommendation
    ratio = fits["tokens_per_param"].mean()
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Optimal tokens_per_param: {ratio:.1f}")
    print(f"\nNext step:")
    print(f"  TOKENS_PER_PARAM={ratio:.0f} ./scripts/run_scaling.sh optimal tokenizer")


if __name__ == "__main__":
    main()
