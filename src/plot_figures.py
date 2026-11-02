"""SPHARE-BD Final Version: Publication-Quality IEEE Figure Generator.

Generates:
1. figures/fig1_coverage_vs_far.{pdf,png}
2. figures/fig2_stratum_acceptance.{pdf,png}
3. figures/fig3_gate_ablation.{pdf,png}
4. figures/fig4_baseline_comparison_90pct.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = PROJECT_ROOT / "results" / "csv"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Publication styling
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "Times"]
mpl.rcParams["font.size"] = 8
mpl.rcParams["axes.labelsize"] = 8
mpl.rcParams["axes.titlesize"] = 8.5
mpl.rcParams["xtick.labelsize"] = 7.5
mpl.rcParams["ytick.labelsize"] = 7.5
mpl.rcParams["legend.fontsize"] = 7.5
mpl.rcParams["figure.titlesize"] = 9
mpl.rcParams["lines.linewidth"] = 1.2
mpl.rcParams["axes.linewidth"] = 0.8
mpl.rcParams["grid.linewidth"] = 0.5
mpl.rcParams["grid.alpha"] = 0.4


def plot_fig1_coverage_vs_far():
    """Figure 1: Supported Coverage vs SDUST Unsupported FAR."""
    base_path = RESULTS_CSV / "baseline_comparison_coverage_matched_90pct.csv"
    if not base_path.exists():
        print(f"File not found: {base_path}")
        return
    df = pd.read_csv(base_path)
    
    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=300)
    ax.grid(True, linestyle="--", alpha=0.5, zorder=0)
    
    # Baseline points
    markers = {"energy_ood": "o", "knn_ood": "s", "deep_mahalanobis": "^", "react_energy": "D", "cnn1d_confidence": "v"}
    colors = {"energy_ood": "#1f77b4", "knn_ood": "#ff7f0e", "deep_mahalanobis": "#9467bd", "react_energy": "#8c564b", "cnn1d_confidence": "#7f7f7f"}
    labels = {
        "energy_ood": "Energy-OOD (70.7% FAR)",
        "knn_ood": "KNN-OOD (54.0% FAR)",
        "deep_mahalanobis": "Deep Mahalanobis (54.7% FAR)",
        "react_energy": "ReAct-Energy (74.0% FAR)",
        "cnn1d_confidence": "CNN-1D Confidence (58.0% FAR)",
    }
    
    for _, row in df.iterrows():
        k = row["model_key"]
        if k in markers:
            ax.scatter(
                row["unsupported_sdust_fa_mean"] * 100,
                row["supported_coverage_mean"] * 100,
                marker=markers[k],
                color=colors[k],
                s=60,
                label=labels[k],
                zorder=4,
            )
            
    # SPHARE-BD Final Version (Stages 1+2)
    ax.scatter(
        0.0, 90.0,
        marker="*",
        s=160,
        color="#2ca02c",
        edgecolor="black",
        linewidth=0.8,
        label="SPHARE-BD Final (90.0% Cov, 0% FAR)",
        zorder=5,
    )
    
    # Full SPHARE-BD (Stages 1-3)
    ax.scatter(
        0.0, 66.67,
        marker="P",
        s=90,
        color="#17becf",
        edgecolor="black",
        linewidth=0.8,
        label="Full SPHARE-BD (66.7% Cov, 0% FAR)",
        zorder=5,
    )
    
    ax.set_xlabel("SDUST Unsupported False Acceptance Rate (%)")
    ax.set_ylabel("Supported SDUST Coverage (%)")
    ax.set_xlim(-3, 85)
    ax.set_ylim(55, 102)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="#cccccc")
    
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_coverage_vs_far.pdf")
    fig.savefig(FIG_DIR / "fig1_coverage_vs_far.png", dpi=300)
    plt.close(fig)
    print("Saved fig1_coverage_vs_far")


def plot_fig2_stratum_acceptance():
    """Figure 2: Stratum-Wise Acceptance Rate for SPHARE-BD Final Version."""
    strata_path = RESULTS_CSV / "stratum_coverage_rejection.csv"
    if not strata_path.exists():
        return
    df = pd.read_csv(strata_path)
    means = df.groupby("stratum")["acceptance_rate"].mean() * 100
    
    order = ["IN", "EDGE-1500", "EDGE-2500", "OUT", "NEW_AXIS", "COMPOUND", "CWRU"]
    labels = ["IN\n(2000 RPM)", "EDGE-1500\n(1500 RPM)", "EDGE-2500\n(2500 RPM)", "OUT\n(Speed)", "NEW_AXIS\n(Load)", "COMPOUND\n(Both)", "CWRU\n(Foreign)"]
    vals = [means[s] for s in order]
    
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=300)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    
    x = np.arange(len(order))
    bars = []
    for i, v in enumerate(vals):
        if v > 0:
            b = ax.bar(x[i], v, color="#2ca02c", edgecolor="#1b611b", width=0.55, zorder=3)
            ax.text(x[i], v + 2.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        else:
            ax.scatter(x[i], 0.0, marker="x", color="#d62728", s=65, linewidth=2.0, zorder=4)
            ax.text(x[i], 4.0, "0.0%\n(Reject)", ha="center", va="bottom", fontsize=7.0, color="#d62728", fontweight="bold")
            
    # Add stratum role shading
    ax.axvspan(-0.5, 2.5, color="#2ca02c", alpha=0.08, zorder=0, label="Supported SDUST Pool")
    ax.axvspan(2.5, 5.5, color="#ff7f0e", alpha=0.08, zorder=0, label="Unsupported SDUST Pool")
    ax.axvspan(5.5, 6.5, color="#d62728", alpha=0.08, zorder=0, label="Foreign Machine (CWRU)")
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Record Acceptance Rate (%)")
    ax.set_ylim(-2, 118)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#cccccc")
    
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_stratum_acceptance.pdf")
    fig.savefig(FIG_DIR / "fig2_stratum_acceptance.png", dpi=300)
    plt.close(fig)
    print("Saved fig2_stratum_acceptance")


def plot_fig3_gate_ablation():
    """Figure 3: Gate Ablation Comparison (Coverage, SDUST FAR, CWRU FAR)."""
    ab_path = RESULTS_CSV / "gate_ablation_comparison.csv"
    if not ab_path.exists():
        return
    df = pd.read_csv(ab_path)
    
    # Filter 4 primary configurations
    configs = ["Classifier only", "Stage 1 only", "Stages 1+2 (SPHARE-BD Final Version)", "Full SPHARE-BD (Stages 1-3)"]
    labels = ["Classifier\nOnly", "Stage 1\nOnly", "Stages 1+2\n(Final)", "Full SPHARE-BD\n(Stages 1-3)"]
    sub = df[df["configuration"].isin(configs)].set_index("configuration").loc[configs]
    
    cov = sub["supported_coverage_mean"] * 100
    sdust_fa = sub["unsupported_sdust_fa_mean"] * 100
    cwru_fa = sub["cwru_fa_mean"] * 100
    
    x = np.arange(len(configs))
    w = 0.25
    
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=300)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    
    r1 = ax.bar(x - w, cov, width=w, label="Supported Coverage (%)", color="#1f77b4", edgecolor="#114f79", zorder=3)
    r2 = ax.bar(x, sdust_fa, width=w, label="SDUST Unsupported FAR (%)", color="#d62728", edgecolor="#8e1718", zorder=3)
    r3 = ax.bar(x + w, cwru_fa, width=w, label="CWRU Foreign FAR (%)", color="#ff7f0e", edgecolor="#b35504", zorder=3)
    
    # Value annotations
    for rects in [r1, r2, r3]:
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax.text(rect.get_x() + rect.get_width() / 2.0, h + 2.0, f"{h:.0f}%", ha="center", va="bottom", fontsize=6.5)
            else:
                ax.text(rect.get_x() + rect.get_width() / 2.0, 2.0, "0%", ha="center", va="bottom", fontsize=6.5, color="gray")
                
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 118)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#cccccc")
    
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_gate_ablation.pdf")
    fig.savefig(FIG_DIR / "fig3_gate_ablation.png", dpi=300)
    plt.close(fig)
    print("Saved fig3_gate_ablation")


def plot_fig4_baseline_comparison():
    """Figure 4: Recalibrated Baseline Comparison at ~90% Coverage."""
    base_path = RESULTS_CSV / "baseline_comparison_coverage_matched_90pct.csv"
    if not base_path.exists():
        return
    df = pd.read_csv(base_path)
    
    models = [
        "SPHARE-BD Final\n(Stages 1+2)",
        "KNN-OOD\n(2022)",
        "Deep Mahal.\n(2021)",
        "ReAct-Energy\n(2021)",
        "Energy-OOD\n(2020)",
        "CNN-1D\nConfidence",
    ]
    keys = ["sphare_bd_final", "knn_ood", "deep_mahalanobis", "react_energy", "energy_ood", "cnn1d_confidence"]
    sub = df.set_index("model_key").loc[keys]
    
    cov = sub["supported_coverage_mean"] * 100
    sdust_fa = sub["unsupported_sdust_fa_mean"] * 100
    cwru_fa = sub["cwru_fa_mean"] * 100
    
    x = np.arange(len(models))
    w = 0.26
    
    fig, ax = plt.subplots(figsize=(5.6, 3.4), dpi=300)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    
    r1 = ax.bar(x - w, cov, width=w, label="Supported Coverage (%)", color="#2ca02c", edgecolor="#1a631a", zorder=3)
    r2 = ax.bar(x, sdust_fa, width=w, label="SDUST Unsupported FAR (%)", color="#d62728", edgecolor="#8e1718", zorder=3)
    r3 = ax.bar(x + w, cwru_fa, width=w, label="CWRU Foreign FAR (%)", color="#ff7f0e", edgecolor="#b35504", zorder=3)
    
    for rects in [r1, r2, r3]:
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax.text(rect.get_x() + rect.get_width() / 2.0, h + 2.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=6.5)
            else:
                ax.text(rect.get_x() + rect.get_width() / 2.0, 2.0, "0%", ha="center", va="bottom", fontsize=6.5, color="gray")
                
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 118)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#cccccc")
    
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_baseline_comparison_90pct.pdf")
    fig.savefig(FIG_DIR / "fig4_baseline_comparison_90pct.png", dpi=300)
    plt.close(fig)
    print("Saved fig4_baseline_comparison_90pct")


def generate_all_figures():
    """Generates all four publication figures."""
    plot_fig1_coverage_vs_far()
    plot_fig2_stratum_acceptance()
    plot_fig3_gate_ablation()
    plot_fig4_baseline_comparison()
    print("All publication figures generated successfully.")


if __name__ == "__main__":
    generate_all_figures()

