#!/usr/bin/env python
"""Export scaling experiment results from W&B to CSV for analysis.

Usage:
    python scripts/export_scaling_results.py --entity YOUR_ENTITY --project tiny_dreamer_4

This script queries W&B for runs matching the scaling_ prefix and exports
metrics to a CSV file for further analysis and plotting.
"""
import argparse
import numpy as np
import pandas as pd


def export_scaling_results(
    entity: str,
    project: str,
    run_prefix: str = "scaling_",
    output_path: str = "scaling_results.csv",
) -> pd.DataFrame:
    """Export scaling runs from W&B to CSV.

    Args:
        entity: W&B entity (username or team name)
        project: W&B project name
        run_prefix: Prefix to filter runs (default: "scaling_")
        output_path: Path to save CSV file

    Returns:
        DataFrame with exported results
    """
    import wandb

    api = wandb.Api()

    # Query runs matching prefix
    runs = api.runs(
        f"{entity}/{project}",
        filters={"display_name": {"$regex": f"^{run_prefix}"}}
    )

    print(f"Found {len(runs)} runs matching prefix '{run_prefix}'")

    records = []
    for run in runs:
        config = run.config
        summary = run.summary

        # Extract scaling metadata (check both config and summary)
        depth = (
            config.get("scaling/depth")
            or config.get("encoder", {}).get("depth")
            or config.get("dynamics", {}).get("depth")
        )
        d_model = (
            config.get("scaling/d_model")
            or config.get("encoder", {}).get("d_model")
            or config.get("dynamics", {}).get("d_model")
        )

        record = {
            # Identifiers
            "run_name": run.name,
            "run_id": run.id,
            "state": run.state,

            # Scaling configuration
            "depth": depth,
            "d_model": d_model,
            "n_heads": config.get("encoder", {}).get("n_heads") or config.get("dynamics", {}).get("n_heads"),

            # Compute
            "total_params": config.get("scaling/total_params") or summary.get("scaling/total_params"),
            "tokens_per_param": config.get("scaling/tokens_per_param"),
            "flops_per_step": config.get("scaling/flops_per_step"),
            "total_flops": summary.get("scaling/total_flops"),
            "total_steps": summary.get("scaling/total_steps"),

            # Performance
            "final_psnr": summary.get("scaling/final_psnr") or summary.get("psnr"),
            "final_loss": summary.get("loss") or summary.get("flow_mse"),

            # Efficiency
            "wall_time_hours": summary.get("scaling/wall_time_hours"),
            "throughput": summary.get("scaling/throughput_steps_per_sec"),
        }
        records.append(record)

    df = pd.DataFrame(records)

    if len(df) == 0:
        print("No runs found. Check your entity, project, and run_prefix.")
        return df

    # Compute derived metrics (only for non-null values)
    if df["total_params"].notna().any():
        df["log10_params"] = np.log10(df["total_params"].astype(float))
    if df["total_flops"].notna().any():
        df["log10_flops"] = np.log10(df["total_flops"].astype(float))

    # Sort by depth for plotting
    df = df.sort_values("depth")

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} runs to {output_path}")

    # Print summary table
    print("\nSummary:")
    print("-" * 80)
    cols = ["depth", "d_model", "total_params", "total_steps", "final_psnr", "wall_time_hours"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))

    return df


def plot_scaling_curves(df: pd.DataFrame, output_dir: str = "plots"):
    """Generate standard scaling law plots.

    Args:
        df: DataFrame with scaling results
        output_dir: Directory to save plots
    """
    import os
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Filter to completed runs with valid data
    df = df[df["state"] == "finished"].copy()
    df = df.dropna(subset=["total_params", "final_psnr"])

    if len(df) < 2:
        print("Not enough completed runs with valid data for plotting.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. PSNR vs Parameters
    ax = axes[0, 0]
    ax.scatter(df["total_params"], df["final_psnr"], s=100, alpha=0.7)
    for _, row in df.iterrows():
        ax.annotate(f'd{int(row["depth"])}', (row["total_params"], row["final_psnr"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Validation PSNR (dB)")
    ax.set_xscale("log")
    ax.set_title("Performance vs Model Size")
    ax.grid(True, alpha=0.3)

    # 2. PSNR vs FLOPs
    ax = axes[0, 1]
    if df["total_flops"].notna().any():
        ax.scatter(df["total_flops"], df["final_psnr"], s=100, alpha=0.7)
        ax.set_xlabel("Total FLOPs")
        ax.set_ylabel("Validation PSNR (dB)")
        ax.set_xscale("log")
        ax.set_title("Performance vs Compute")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No FLOPs data", ha="center", va="center", transform=ax.transAxes)

    # 3. FLOPs vs Parameters
    ax = axes[1, 0]
    if df["total_flops"].notna().any():
        ax.scatter(df["total_params"], df["total_flops"], s=100, alpha=0.7)
        ax.set_xlabel("Parameters")
        ax.set_ylabel("Total FLOPs")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("Compute vs Model Size")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No FLOPs data", ha="center", va="center", transform=ax.transAxes)

    # 4. Training time vs Parameters
    ax = axes[1, 1]
    if df["wall_time_hours"].notna().any():
        ax.scatter(df["total_params"], df["wall_time_hours"], s=100, alpha=0.7)
        ax.set_xlabel("Parameters")
        ax.set_ylabel("Wall Time (hours)")
        ax.set_xscale("log")
        ax.set_title("Training Time vs Model Size")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No timing data", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    plot_path = f"{output_dir}/scaling_curves.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export scaling experiment results from W&B")
    parser.add_argument("--entity", required=True, help="W&B entity (username or team)")
    parser.add_argument("--project", default="tiny_dreamer_4", help="W&B project name")
    parser.add_argument("--prefix", default="scaling_", help="Run name prefix to filter")
    parser.add_argument("--output", default="scaling_results.csv", help="Output CSV path")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--plot-dir", default="plots", help="Directory for plots")
    args = parser.parse_args()

    df = export_scaling_results(
        entity=args.entity,
        project=args.project,
        run_prefix=args.prefix,
        output_path=args.output,
    )

    if args.plot and len(df) > 0:
        plot_scaling_curves(df, output_dir=args.plot_dir)
