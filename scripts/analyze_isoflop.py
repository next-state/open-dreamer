#!/usr/bin/env python
"""Analyze iso-FLOPs experiments to discover optimal tokens_per_param ratio.

Usage:
    python scripts/analyze_isoflop.py --entity YOUR_ENTITY --project tiny_dreamer_4

This script:
1. Loads iso-FLOPs runs from W&B
2. Groups runs by compute budget
3. Fits quadratic curves to find compute-optimal model size for each budget
4. Extracts the optimal N(C) relationship to determine tokens_per_param
5. Generates publication-quality plots like Karpathy's nanochat analysis
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from omegaconf import OmegaConf


def parse_metrics_file(run_dir: Path, run_name: str) -> tuple:                                                                                               
    """Parse metrics from structured metrics file.                                                                                               
                                                                                                                                        
    Args:
        run_dir: Path to run directory (for metrics.json lookup)
        run_name: Name of the run (for metrics.json filename)
    
    Returns:                                                                                                                               
        (total_params, flops_per_step, final_metrics_list, total_steps, train_elapsed_hours, tokens_trained)
        where final_metrics_list is the list of final metrics (last 5 steps, last one is the final step)
    """                                                                                                                                    
    metrics_file = run_dir / f"{run_name}_metrics.json"
    if not metrics_file.exists():
        return None, None, None, None, None, None
    
    try:
        with open(metrics_file, "r") as f:
            metrics_data = json.load(f)
        # Get the entire final_metrics list (last 5 steps)
        final_metrics_list = metrics_data.get("final_metrics", [])
        return (
            metrics_data.get("total_params"),
            metrics_data.get("flops_per_step"),
            final_metrics_list,
            metrics_data.get("total_steps"),
            metrics_data.get("train_elapsed_hours"),
            metrics_data.get("tokens_trained"),
        )
    except (json.JSONDecodeError, KeyError):
        return None, None, None, None, None, None


def load_local_runs(logs_dir: str) -> pd.DataFrame:                                                                                        
    """Load iso-FLOPs runs from local log directory.                                                                                       
                                                                                                                                        
    Args:                                                                                                                                  
        logs_dir: Path to experiment directory (e.g., logs/isoflop_tokenizer_20260112_170415)                                              
                                                                                                                                        
    Returns:                                                                                                                               
        DataFrame with run data                                                                                                            
    """                                                                                                                                    
    logs_path = Path(logs_dir)                                                                                                             
    records = []                                                                                                                           
                                                                                                                                        
    # Find all run subdirectories (have .hydra/config.yaml)                                                                                
    for run_dir in sorted(logs_path.iterdir()):                                                                                            
        if not run_dir.is_dir():                                                                                                           
            continue                                                                                                                       
                                                                                                                                        
        config_path = run_dir / ".hydra" / "config.yaml"                                                                                   
        if not config_path.exists():                                                                                                       
            continue                                                                                                                       
                                                                                                                                        
        # Parse Hydra config
        config = OmegaConf.load(config_path)

        # Parse metrics from structured file
        total_params, flops_per_step, final_metrics_list, total_steps, train_elapsed_hours, tokens_trained = parse_metrics_file(run_dir, run_dir.name)

        record = {
            "run_name": run_dir.name,
            "depth": config.encoder.depth,
            "d_model": config.encoder.d_model,
            "total_params": total_params,
            "flops_budget": config.scaling_flops_budget,
            "flops_per_step": flops_per_step,
            "total_flops": flops_per_step * total_steps if flops_per_step and total_steps else None,
            "total_steps": total_steps,
            "tokens_trained": tokens_trained,
            "train_elapsed_hours": train_elapsed_hours,
            "final_metrics": final_metrics_list,
        }
        records.append(record)

    return pd.DataFrame(records)


def quadratic_fit(log_params: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Quadratic function for fitting: loss = a*x^2 + b*x + c where x = log10(params)."""
    return a * log_params**2 + b * log_params + c


def find_optimal_params(log_params: np.ndarray, a: float, b: float, c: float) -> float:
    """Find the log10(params) that minimizes the quadratic.

    For ax^2 + bx + c, minimum is at x = -b/(2a)
    """
    if a <= 0:
        # Not a valid U-curve, return NaN
        return np.nan
    return -b / (2 * a)


def fit_isoflop_curves(df: pd.DataFrame, metric: str = "final_loss", smooth_n: int = 1) -> pd.DataFrame:
    """Fit quadratic curves to each iso-FLOPs group.

    Args:
        df: DataFrame with run data
        metric: Metric name to extract from final_metrics dict (e.g., "psnr", "loss_total", "loss_mse")
        smooth_n: Number of last metrics to average over (1 = no smoothing, use last metric only)

    Returns:
        DataFrame with fit results for each compute budget
    """
    # Extract the specified metric from final_metrics list, optionally smoothed
    df = df.copy()
    metric_values = []
    for final_metrics_list in df["final_metrics"]:
        if final_metrics_list is not None and isinstance(final_metrics_list, list) and len(final_metrics_list) > 0:
            # Take the last n metrics and average the specified metric
            last_n_metrics = final_metrics_list[-smooth_n:]
            metric_vals = [m.get(metric) for m in last_n_metrics if isinstance(m, dict) and metric in m]
            if metric_vals:
                metric_values.append(np.mean(metric_vals))
            else:
                metric_values.append(None)
        else:
            metric_values.append(None)
    df["_metric_value"] = metric_values
    
    # Determine if metric is "higher is better" (like PSNR) or "lower is better" (like loss)
    # For metrics containing "loss" or "mse", lower is better; for others like "psnr", higher is better
    if "loss" in metric.lower() or "mse" in metric.lower():
        # Lower is better, use as-is
        loss_col = "_metric_value"
    else:
        # Higher is better (e.g., PSNR), negate so lower is better for fitting
        df["_loss"] = -df["_metric_value"]
        loss_col = "_loss"

    results = []
    budgets = sorted(df["flops_budget"].dropna().unique())

    for budget in budgets:
        group = df[df["flops_budget"] == budget].dropna(subset=["total_params", loss_col])

        if len(group) < 3:
            print(f"  Budget {budget:.2e}: Not enough points ({len(group)}) for quadratic fit")
            continue

        log_params = np.log10(group["total_params"].values)
        loss_values = group[loss_col].values

        try:
            # Fit quadratic
            popt, pcov = curve_fit(quadratic_fit, log_params, loss_values, maxfev=5000)
            a, b, c = popt

            # Find optimal log(params)
            log_optimal = find_optimal_params(log_params, a, b, c)
            optimal_params = 10 ** log_optimal if not np.isnan(log_optimal) else np.nan

            # Compute optimal loss
            optimal_loss = quadratic_fit(log_optimal, a, b, c) if not np.isnan(log_optimal) else np.nan

            # Compute tokens at optimal point using interpolation from actual data (Karpathy's approach)
            if not np.isnan(optimal_params):
                # Interpolate from actual experimental data points
                tokens_values = group["tokens_trained"].values
                # Interpolate in log space (matching Karpathy's approach)
                optimal_tokens = np.interp(log_optimal, log_params, tokens_values)
                optimal_ratio = optimal_tokens / optimal_params
            else:
                optimal_tokens = np.nan
                optimal_ratio = np.nan

            # R-squared
            residuals = loss_values - quadratic_fit(log_params, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((loss_values - np.mean(loss_values))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            results.append({
                "flops_budget": budget,
                "n_points": len(group),
                "a": a,
                "b": b,
                "c": c,
                "r_squared": r_squared,
                "log_optimal_params": log_optimal,
                "optimal_params": optimal_params,
                "optimal_tokens": optimal_tokens,
                "optimal_loss": optimal_loss,
                "tokens_per_param": optimal_ratio,
            })

            print(f"  Budget {budget:.2e}: optimal N={optimal_params:.2e}, D/N={optimal_ratio:.1f}, R²={r_squared:.3f}")

        except Exception as e:
            print(f"  Budget {budget:.2e}: Fit failed - {e}")

    return pd.DataFrame(results)


def _plot_isoflop_curves_subplot(
    ax, df: pd.DataFrame, fit_results: pd.DataFrame, budgets: list, colors: np.ndarray,
    loss_col: str, ylabel: str, is_subplot: bool = False
):
    """Plot iso-FLOPs curves on given axes.
    
    Args:
        ax: Matplotlib axes to plot on
        df: DataFrame with run data
        fit_results: DataFrame with quadratic fit results
        budgets: List of compute budgets
        colors: Color array for each budget
        loss_col: Column name for loss values
        ylabel: Y-axis label
        is_subplot: If True, use smaller font sizes and markers
    """
    marker_size = 60 if is_subplot else 80
    star_size = 150 if is_subplot else 200
    font_size = 11 if is_subplot else 12
    title_size = 12 if is_subplot else 14
    legend_size = 8 if is_subplot else 9
    
    for i, budget in enumerate(budgets):
        group = df[df["flops_budget"] == budget].dropna(subset=["total_params", loss_col])
        if len(group) == 0:
            continue

        log_params = np.log10(group["total_params"].values)
        loss_values = group[loss_col].values

        # Plot data points
        ax.scatter(
            group["total_params"], loss_values,
            color=colors[i], s=marker_size, alpha=0.8, edgecolors="white", linewidth=0.5,
            label=f"C={budget:.0e}"
        )

        # Plot quadratic fit if available
        fit_row = fit_results[fit_results["flops_budget"] == budget]
        if len(fit_row) > 0:
            fit_row = fit_row.iloc[0]
            x_fit = np.linspace(log_params.min() - 0.2, log_params.max() + 0.2, 100)
            y_fit = quadratic_fit(x_fit, fit_row["a"], fit_row["b"], fit_row["c"])
            ax.plot(10**x_fit, y_fit, color=colors[i], linestyle="--", alpha=0.7, linewidth=2)

            # Mark optimal point
            if not np.isnan(fit_row["optimal_params"]):
                ax.scatter(
                    [fit_row["optimal_params"]], [fit_row["optimal_loss"]],
                    color=colors[i], s=star_size, marker="*", edgecolors="black", linewidth=1,
                    zorder=10
                )

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (N)", fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    title = "Iso-FLOPs Curves" if is_subplot else "Iso-FLOPs Curves: Loss vs Model Size at Fixed Compute"
    ax.set_title(title, fontsize=title_size)
    ax.legend(title="Compute Budget", loc="upper right", fontsize=legend_size, ncol=1)
    ax.grid(True, alpha=0.3)


def _plot_optimal_n_vs_compute_subplot(
    ax, fit_results: pd.DataFrame, is_subplot: bool = False
):
    """Plot optimal N vs compute on given axes.
    
    Args:
        ax: Matplotlib axes to plot on
        fit_results: DataFrame with fit results
        is_subplot: If True, use smaller font sizes and markers
    """
    marker_size = 80 if is_subplot else 100
    font_size = 11 if is_subplot else 12
    title_size = 12 if is_subplot else 14
    legend_size = 10 if is_subplot else 11
    
    valid_fits = fit_results.dropna(subset=["optimal_params", "flops_budget"])
    if len(valid_fits) >= 2:
        ax.scatter(
            valid_fits["flops_budget"], valid_fits["optimal_params"],
            s=marker_size, color="royalblue", edgecolors="black", linewidth=1
        )

        # Fit power law: N_opt = A * C^alpha
        try:
            log_c = np.log10(valid_fits["flops_budget"].values)
            log_n = np.log10(valid_fits["optimal_params"].values)
            coeffs = np.polyfit(log_c, log_n, 1)
            alpha = coeffs[0]
            A = 10 ** coeffs[1]

            x_fit = np.logspace(np.log10(valid_fits["flops_budget"].min()) - 0.5,
                               np.log10(valid_fits["flops_budget"].max()) + 0.5, 100)
            y_fit = A * x_fit ** alpha
            ax.plot(x_fit, y_fit, "r--", linewidth=2, label=f"N ∝ C^{alpha:.2f}")
            ax.legend(fontsize=legend_size)
        except Exception as e:
            print(f"Power law fit failed: {e}")

        ax.set_xscale("log")
        ax.set_yscale("log")
    
    ax.set_xlabel("Compute Budget (FLOPs)", fontsize=font_size)
    ax.set_ylabel("Optimal Parameters (N)", fontsize=font_size)
    ax.set_title("Compute-Optimal Model Size", fontsize=title_size)
    ax.grid(True, alpha=0.3)


def _plot_optimal_d_vs_compute_subplot(
    ax, fit_results: pd.DataFrame, is_subplot: bool = False
):
    """Plot optimal D vs compute on given axes.
    
    Args:
        ax: Matplotlib axes to plot on
        fit_results: DataFrame with fit results
        is_subplot: If True, use smaller font sizes and markers
    """
    marker_size = 80 if is_subplot else 100
    font_size = 11 if is_subplot else 12
    title_size = 12 if is_subplot else 14
    legend_size = 10 if is_subplot else 11
    
    valid_tokens = fit_results.dropna(subset=["optimal_tokens", "flops_budget"])
    if len(valid_tokens) >= 2:
        ax.scatter(
            valid_tokens["flops_budget"], valid_tokens["optimal_tokens"],
            s=marker_size, color="forestgreen", edgecolors="black", linewidth=1
        )

        # Fit power law: D_opt = A * C^alpha
        try:
            log_c = np.log10(valid_tokens["flops_budget"].values)
            log_d = np.log10(valid_tokens["optimal_tokens"].values)
            coeffs = np.polyfit(log_c, log_d, 1)
            alpha = coeffs[0]
            A = 10 ** coeffs[1]

            x_fit = np.logspace(np.log10(valid_tokens["flops_budget"].min()) - 0.5,
                               np.log10(valid_tokens["flops_budget"].max()) + 0.5, 100)
            y_fit = A * x_fit ** alpha
            ax.plot(x_fit, y_fit, "r--", linewidth=2, label=f"D ∝ C^{alpha:.2f}")
            ax.legend(fontsize=legend_size)
        except Exception as e:
            print(f"Power law fit failed: {e}")

        ax.set_xscale("log")
        ax.set_yscale("log")
    
    ax.set_xlabel("Compute Budget (FLOPs)", fontsize=font_size)
    ax.set_ylabel("Optimal Training Tokens (D)", fontsize=font_size)
    ax.set_title("Compute-Optimal Training Tokens", fontsize=title_size)
    ax.grid(True, alpha=0.3)


def _find_optimal_runs(df: pd.DataFrame, fit_results: pd.DataFrame) -> pd.DataFrame:
    """Find the optimal run for each compute budget to get depth and training time.
    
    Args:
        df: DataFrame with run data
        fit_results: DataFrame with fit results containing optimal_params
        
    Returns:
        DataFrame with one row per budget containing optimal metric, depth, FLOPs, and training time
    """
    optimal_runs = []
    
    for _, fit_row in fit_results.iterrows():
        budget = fit_row["flops_budget"]
        optimal_params = fit_row["optimal_params"]
        
        if np.isnan(optimal_params):
            continue
        
        # Find runs with this budget
        budget_runs = df[df["flops_budget"] == budget].dropna(subset=["total_params"])
        
        if len(budget_runs) == 0:
            continue
        
        # Find the run with params closest to optimal
        budget_runs = budget_runs.copy()
        budget_runs["_param_diff"] = np.abs(budget_runs["total_params"] - optimal_params)
        closest_run = budget_runs.loc[budget_runs["_param_diff"].idxmin()]
        
        # Use actual total_flops if available, otherwise use budget
        total_flops = closest_run.get("total_flops")
        if total_flops is None or np.isnan(total_flops):
            total_flops = budget
        
        optimal_runs.append({
            "flops_budget": budget,
            "optimal_params": optimal_params,
            "optimal_metric": fit_row["optimal_loss"],
            "depth": closest_run["depth"],
            "total_flops": total_flops,
            "train_elapsed_hours": closest_run.get("train_elapsed_hours"),
        })
    
    return pd.DataFrame(optimal_runs)


def _plot_metric_vs_flops_subplot(
    ax, optimal_runs: pd.DataFrame, metric: str, ylabel: str, is_subplot: bool = False
):
    """Plot metric vs FLOPs with depth labels and extrapolation fit.
    
    Args:
        ax: Matplotlib axes to plot on
        optimal_runs: DataFrame with optimal runs (one per budget)
        metric: Metric name (for determining if higher/lower is better)
        ylabel: Y-axis label
        is_subplot: If True, use smaller font sizes and markers
    """
    marker_size = 80 if is_subplot else 100
    font_size = 11 if is_subplot else 12
    title_size = 12 if is_subplot else 14
    label_size = 9 if is_subplot else 10
    
    valid_runs = optimal_runs.dropna(subset=["optimal_metric", "total_flops"])
    if len(valid_runs) < 2:
        return
    
    # Convert optimal_metric back if it was negated (for "higher is better" metrics)
    if "loss" not in metric.lower() and "mse" not in metric.lower():
        # Higher is better, so optimal_loss was negated
        metric_values = -valid_runs["optimal_metric"].values
    else:
        # Lower is better, use as-is
        metric_values = valid_runs["optimal_metric"].values
    
    flops_values = valid_runs["total_flops"].values
    depths = valid_runs["depth"].values
    
    # Plot points
    ax.scatter(
        flops_values, metric_values,
        s=marker_size, color="royalblue", edgecolors="black", linewidth=1, zorder=5
    )
    
    # Add depth labels
    for flops, metric_val, depth in zip(flops_values, metric_values, depths):
        if not np.isnan(depth):
            ax.annotate(
                f"d{int(depth)}",
                (flops, metric_val),
                xytext=(5, 5), textcoords="offset points",
                fontsize=label_size, alpha=0.8
            )
    
    # Fit linear relationship on log scale: metric = a * log10(FLOPs) + b
    # This will appear as a line on the log-scale plot
    try:
        log_flops = np.log10(flops_values)
        coeffs = np.polyfit(log_flops, metric_values, 1)
        a, b = coeffs
        
        # Generate fit line extending two orders of magnitude to the right for extrapolation
        x_fit = np.logspace(np.log10(flops_values.min()) - 0.5,
                           np.log10(flops_values.max()) + 2.0, 100)
        y_fit = a * np.log10(x_fit) + b
        ax.plot(x_fit, y_fit, "r--", linewidth=2, alpha=0.7, label="Linear fit (log scale)")
        ax.legend(fontsize=font_size - 1)
    except Exception as e:
        print(f"Linear fit failed for metric vs FLOPs: {e}")
    
    ax.set_xscale("log")
    # Extend x-axis two orders of magnitude to the right for visual extrapolation
    ax.set_xlim(left=flops_values.min() * 0.1, right=flops_values.max() * 100)
    ax.set_xlabel("Total FLOPs", fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.set_title(f"{ylabel} vs Compute (FLOPs)", fontsize=title_size)
    ax.grid(True, alpha=0.3)


def _plot_metric_vs_time_subplot(
    ax, optimal_runs: pd.DataFrame, metric: str, ylabel: str, is_subplot: bool = False
):
    """Plot metric vs training time with depth labels and extrapolation fit.
    
    Args:
        ax: Matplotlib axes to plot on
        optimal_runs: DataFrame with optimal runs (one per budget)
        metric: Metric name (for determining if higher/lower is better)
        ylabel: Y-axis label
        is_subplot: If True, use smaller font sizes and markers
    """
    marker_size = 80 if is_subplot else 100
    font_size = 11 if is_subplot else 12
    title_size = 12 if is_subplot else 14
    label_size = 9 if is_subplot else 10
    
    valid_runs = optimal_runs.dropna(subset=["optimal_metric", "train_elapsed_hours"])
    if len(valid_runs) < 2:
        return
    
    # Convert optimal_metric back if it was negated (for "higher is better" metrics)
    if "loss" not in metric.lower() and "mse" not in metric.lower():
        # Higher is better, so optimal_loss was negated
        metric_values = -valid_runs["optimal_metric"].values
    else:
        # Lower is better, use as-is
        metric_values = valid_runs["optimal_metric"].values
    
    time_values = valid_runs["train_elapsed_hours"].values
    depths = valid_runs["depth"].values
    
    # Plot points
    ax.scatter(
        time_values, metric_values,
        s=marker_size, color="forestgreen", edgecolors="black", linewidth=1, zorder=5
    )
    
    # Add depth labels
    for time, metric_val, depth in zip(time_values, metric_values, depths):
        if not np.isnan(depth):
            ax.annotate(
                f"d{int(depth)}",
                (time, metric_val),
                xytext=(5, 5), textcoords="offset points",
                fontsize=label_size, alpha=0.8
            )
    
    # Fit linear relationship on log scale: metric = a * log10(time) + b
    # This will appear as a line on the log-scale plot
    try:
        log_time = np.log10(time_values)
        coeffs = np.polyfit(log_time, metric_values, 1)
        a, b = coeffs
        
        # Generate fit line extending two orders of magnitude to the right for extrapolation
        x_fit = np.logspace(np.log10(time_values.min()) - 0.5,
                           np.log10(time_values.max()) + 2.0, 100)
        y_fit = a * np.log10(x_fit) + b
        ax.plot(x_fit, y_fit, "r--", linewidth=2, alpha=0.7, label="Linear fit (log scale)")
        ax.legend(fontsize=font_size - 1)
    except Exception as e:
        print(f"Linear fit failed for metric vs time: {e}")
    
    ax.set_xscale("log")
    # Extend x-axis two orders of magnitude to the right for visual extrapolation
    ax.set_xlim(left=time_values.min() * 0.1, right=time_values.max() * 100)
    ax.set_xlabel("Training Time (hours)", fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.set_title(f"{ylabel} vs Training Time", fontsize=title_size)
    ax.grid(True, alpha=0.3)


def plot_isoflop_curves(
    df: pd.DataFrame,
    fit_results: pd.DataFrame,
    metric: str = "final_loss",
    smooth_n: int = 1,
    output_dir: str = "plots",
):
    """Generate Karpathy-style iso-FLOPs plots.

    Args:
        df: DataFrame with run data
        fit_results: DataFrame with quadratic fit results
        metric: Metric name to extract from final_metrics dict
        smooth_n: Number of last metrics to average over (1 = no smoothing, use last metric only)
        output_dir: Directory to save plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract the specified metric from final_metrics list, optionally smoothed
    df = df.copy()
    metric_values = []
    for final_metrics_list in df["final_metrics"]:
        if final_metrics_list is not None and isinstance(final_metrics_list, list) and len(final_metrics_list) > 0:
            # Take the last n metrics and average the specified metric
            last_n_metrics = final_metrics_list[-smooth_n:]
            metric_vals = [m.get(metric) for m in last_n_metrics if isinstance(m, dict) and metric in m]
            if metric_vals:
                metric_values.append(np.mean(metric_vals))
            else:
                metric_values.append(None)
        else:
            metric_values.append(None)
    df["_metric_value"] = metric_values
    
    # Determine if metric is "higher is better" or "lower is better"
    if "loss" in metric.lower() or "mse" in metric.lower():
        # Lower is better
        loss_col = "_metric_value"
        ylabel = metric.replace("_", " ").title()
        metric_ylabel = ylabel
    else:
        # Higher is better (e.g., PSNR), negate for plotting
        df["_loss"] = -df["_metric_value"]
        loss_col = "_loss"
        ylabel = f"Negative {metric.replace('_', ' ').title()} (lower is better)"
        metric_ylabel = metric.replace("_", " ").title()

    budgets = sorted(df["flops_budget"].dropna().unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(budgets)))

    # =========================================================================
    # Plot 1: Iso-FLOPs curves with quadratic fits
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 7))
    _plot_isoflop_curves_subplot(ax, df, fit_results, budgets, colors, loss_col, ylabel, is_subplot=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/isoflop_curves.png", dpi=150)
    # plt.savefig(f"{output_dir}/isoflop_curves.pdf")
    plt.close()
    print(f"Saved: {output_dir}/isoflop_curves.png")

    # =========================================================================
    # Plot 2: Optimal N vs Compute (compute-optimal frontier)
    # =========================================================================
    valid_fits = fit_results.dropna(subset=["optimal_params", "flops_budget"])
    if len(valid_fits) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        _plot_optimal_n_vs_compute_subplot(ax, fit_results, is_subplot=False)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/optimal_n_vs_compute.png", dpi=150)
        # plt.savefig(f"{output_dir}/optimal_n_vs_compute.pdf")
        plt.close()
        print(f"Saved: {output_dir}/optimal_n_vs_compute.png")

    # =========================================================================
    # Plot 3: Optimal D vs Compute (optimal training tokens)
    # =========================================================================
    valid_tokens = fit_results.dropna(subset=["optimal_tokens", "flops_budget"])
    if len(valid_tokens) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        _plot_optimal_d_vs_compute_subplot(ax, fit_results, is_subplot=False)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/tokens_per_param.png", dpi=150)
        # plt.savefig(f"{output_dir}/tokens_per_param.pdf")
        plt.close()
        print(f"Saved: {output_dir}/tokens_per_param.png")

    # =========================================================================
    # Plot 4: Metric vs FLOPs (compute-optimal points)
    # =========================================================================
    optimal_runs = _find_optimal_runs(df, fit_results)
    if len(optimal_runs) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        _plot_metric_vs_flops_subplot(ax, optimal_runs, metric, metric_ylabel, is_subplot=False)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/metric_vs_flops.png", dpi=150)
        # plt.savefig(f"{output_dir}/metric_vs_flops.pdf")
        plt.close()
        print(f"Saved: {output_dir}/metric_vs_flops.png")

    # =========================================================================
    # Plot 5: Metric vs Training Time (compute-optimal points)
    # =========================================================================
    if len(optimal_runs) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        _plot_metric_vs_time_subplot(ax, optimal_runs, metric, metric_ylabel, is_subplot=False)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/metric_vs_time.png", dpi=150)
        # plt.savefig(f"{output_dir}/metric_vs_time.pdf")
        plt.close()
        print(f"Saved: {output_dir}/metric_vs_time.png")

    # =========================================================================
    # Plot 6: Combined summary plot (2x2 grid)
    # =========================================================================
    # Define variables needed for summary plot
    valid_ratios = fit_results.dropna(subset=["tokens_per_param"])
    mean_ratio = valid_ratios["tokens_per_param"].mean() if len(valid_ratios) >= 1 else None
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 4a: Iso-FLOPs curves
    _plot_isoflop_curves_subplot(axes[0, 0], df, fit_results, budgets, colors, loss_col, ylabel, is_subplot=True)

    # 4b: Optimal N vs C
    _plot_optimal_n_vs_compute_subplot(axes[0, 1], fit_results, is_subplot=True)

    # 4c: Optimal D vs C
    _plot_optimal_d_vs_compute_subplot(axes[1, 0], fit_results, is_subplot=True)

    # 4d: Summary statistics
    ax = axes[1, 1]
    ax.axis("off")
    summary_text = "Summary Statistics\n" + "=" * 30 + "\n\n"
    summary_text += f"Number of compute budgets: {len(fit_results)}\n"
    summary_text += f"Total runs analyzed: {len(df)}\n\n"
    if len(valid_ratios) >= 1:
        summary_text += f"Optimal tokens_per_param:\n"
        summary_text += f"  Mean: {mean_ratio:.2f}\n"
        summary_text += f"  Std:  {valid_ratios['tokens_per_param'].std():.2f}\n"
        summary_text += f"  Min:  {valid_ratios['tokens_per_param'].min():.2f}\n"
        summary_text += f"  Max:  {valid_ratios['tokens_per_param'].max():.2f}\n\n"
        summary_text += f"Recommendation:\n"
        summary_text += f"  Use scaling_tokens_per_param={mean_ratio:.1f}\n"
        summary_text += f"  in run_scaling.sh"
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/isoflop_summary.png", dpi=150)
    # plt.savefig(f"{output_dir}/isoflop_summary.pdf")
    plt.close()
    print(f"Saved: {output_dir}/isoflop_summary.png")


def main():
    parser = argparse.ArgumentParser(description="Analyze iso-FLOPs experiments")
    parser.add_argument("--logs-dir", required=True,
                        help="Local logs directory (e.g., logs/isoflop_tokenizer_20260112_170415)")
    parser.add_argument("--metric", default="psnr",
                        help="Metric name to extract from final_metrics (e.g., 'psnr', 'loss_total', 'loss_mse', 'loss_lpips')")
    parser.add_argument("--smooth-n", type=int, default=1,
                        help="Number of last metrics to average over for smoothing (1 = no smoothing, use last metric only)")
    parser.add_argument("--output-dir", default="plots/isoflop", help="Output directory")
    parser.add_argument("--output-csv", default="isoflop_results.csv", help="Output CSV path")
    args = parser.parse_args()

    print("=" * 60)
    print("Iso-FLOPs Scaling Law Analysis")
    print("=" * 60)

    # Load data
    print("\n1. Loading local runs...")
    df = load_local_runs(args.logs_dir)
    print(df.head())

    if len(df) == 0:
        print("No completed runs found. Exiting.")
        return

    # Save raw data
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved raw data to: {args.output_csv}")

    # Fit quadratic curves
    print(f"\n2. Fitting quadratic curves to each compute budget (metric={args.metric}, smooth_n={args.smooth_n})...")
    fit_results = fit_isoflop_curves(df, metric=args.metric, smooth_n=args.smooth_n)

    if len(fit_results) == 0:
        print("No successful fits. Need more data points per compute budget.")
        return

    # Save fit results
    fit_csv = args.output_csv.replace(".csv", "_fits.csv")
    fit_results.to_csv(fit_csv, index=False)
    print(f"\nSaved fit results to: {fit_csv}")

    # Generate plots
    print("\n3. Generating plots...")
    plot_isoflop_curves(df, fit_results, metric=args.metric, smooth_n=args.smooth_n, output_dir=args.output_dir)

    # Print final recommendation
    valid_ratios = fit_results.dropna(subset=["tokens_per_param"])
    if len(valid_ratios) >= 1:
        mean_ratio = valid_ratios["tokens_per_param"].mean()
        print("\n" + "=" * 60)
        print("RECOMMENDATION")
        print("=" * 60)
        print(f"\nOptimal tokens_per_param ratio: {mean_ratio:.1f}")
        print(f"\nTo train compute-optimal models, update run_scaling.sh:")
        print(f"  TOKENS_PER_PARAM={mean_ratio:.1f}")
        print("\nOr run directly:")
        print(f"  python scripts/train_tokenizer.py scaling_tokens_per_param={mean_ratio:.1f} ...")


if __name__ == "__main__":
    main()
