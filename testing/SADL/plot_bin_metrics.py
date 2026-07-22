import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _configure_matplotlib_for_papers() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.linewidth": 0.8,
        }
    )


def _read_metrics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected = {"bin", "images", "map50", "f1"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in {csv_path}: {sorted(missing)}")

    df = df.sort_values("bin").reset_index(drop=True)
    for col in ["map50", "f1", "images"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _to_5_bins_from_10(df10: pd.DataFrame) -> pd.DataFrame:
    """
    Derive a 5-bin F1 curve by merging adjacent 10-bin rows and taking
    image-count weighted mean F1.
    """
    if len(df10) != 10:
        raise ValueError("Expected exactly 10 bins in input CSV to derive 5-bin plot.")

    rows = []
    for i in range(5):
        left = df10.iloc[2 * i]
        right = df10.iloc[2 * i + 1]

        images = float(left["images"]) + float(right["images"])
        if images > 0:
            f1 = (float(left["f1"]) * float(left["images"]) + float(right["f1"]) * float(right["images"])) / images
        else:
            f1 = float("nan")

        rows.append({"bin": i, "images": images, "f1": f1})

    return pd.DataFrame(rows)


def _fit_line_and_corr(df: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    """Return fitted y values, Spearman rho, and slope."""
    x = df["bin"].to_numpy(dtype=float)
    y = df["f1"].to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        fitted = np.full_like(df["bin"].to_numpy(dtype=float), np.nan, dtype=float)
        return fitted, float("nan"), float("nan")

    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * df["bin"].to_numpy(dtype=float) + intercept
    spearman_rho = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    return fitted, spearman_rho, float(slope)


def _plot_f1_line(df: pd.DataFrame, out_dir: Path, stem: str, title: str) -> None:
    x = df["bin"].to_numpy()
    fitted, spearman_rho, slope = _fit_line_and_corr(df)

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.scatter(x, df["f1"], s=42, color="#d62728", zorder=3, label="F1")
    ax.plot(x, fitted, linestyle="--", linewidth=2.0, color="#1f77b4", label="Best fit")

    ax.set_xlabel("SADL Bin (low \u2192 high surprise)")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)

    stats_text = (
        f"Spearman ρ = {spearman_rho:.3f}\n"
        f"Slope = {slope:.3f}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#999999", alpha=0.92),
    )

    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def _plot_f1_comparison(df10: pd.DataFrame, df5: pd.DataFrame, out_dir: Path, stem: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True)

    fitted5, spearman5, slope5 = _fit_line_and_corr(df5)
    fitted10, spearman10, slope10 = _fit_line_and_corr(df10)

    axes[0].scatter(df5["bin"], df5["f1"], s=42, color="#f28e2b", zorder=3)
    axes[0].plot(df5["bin"], fitted5, linestyle="--", linewidth=2.0, color="#1f77b4")
    axes[0].set_title("Bin size = 5")
    axes[0].set_xlabel("Bin")
    axes[0].set_ylabel("F1 Score")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.25)
    axes[0].text(
        0.03,
        0.97,
        f"Spearman ρ = {spearman5:.3f}\nSlope = {slope5:.3f}",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#999999", alpha=0.92),
    )

    axes[1].scatter(df10["bin"], df10["f1"], s=42, color="#d62728", zorder=3)
    axes[1].plot(df10["bin"], fitted10, linestyle="--", linewidth=2.0, color="#1f77b4")
    axes[1].set_title("Bin size = 10")
    axes[1].set_xlabel("Bin")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.25)
    axes[1].text(
        0.03,
        0.97,
        f"Spearman ρ = {spearman10:.3f}\nSlope = {slope10:.3f}",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#999999", alpha=0.92),
    )

    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot F1-only graphs for SADL bin sizes 5 and 10.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("binned_results/lsa_bin_metrics.csv"),
        help="Path to 10-bin metrics CSV (default: binned_results/lsa_bin_metrics.csv)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("binned_results/figures"),
        help="Output directory for figures (default: binned_results/figures)",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default="lsa_f1",
        help="Filename stem for generated plots.",
    )
    args = parser.parse_args()

    _configure_matplotlib_for_papers()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df10 = _read_metrics(args.csv)
    df5 = _to_5_bins_from_10(df10)

    _plot_f1_line(df5, args.out_dir, f"{args.stem}_bin5", "F1 by SADL bin (bin size = 5)")
    _plot_f1_line(df10, args.out_dir, f"{args.stem}_bin10", "F1 by SADL bin (bin size = 10)")
    _plot_f1_comparison(df10, df5, args.out_dir, f"{args.stem}_comparison")

    df5.to_csv(args.out_dir / f"{args.stem}_bin5_values.csv", index=False)
    df10[["bin", "images", "f1"]].to_csv(args.out_dir / f"{args.stem}_bin10_values.csv", index=False)

    print(f"Saved F1-only figures to: {args.out_dir.resolve()}")
    print(f"Input 10-bin CSV: {args.csv.resolve()}")


if __name__ == "__main__":
    main()
