"""Generates publication-quality IEEE Signal Processing Letters figures.

Figure 1: SPHARE-BD Framework Vector Architecture Diagram (fig1_framework.pdf)
Figure 2: Comprehensive 3-Panel Experimental Summary (fig2_three_panel.pdf)
"""
from __future__ import annotations

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = REPO_ROOT / "results" / "csv"
DATA_BASELINES = REPO_ROOT / "data" / "baselines" / "risk_coverage_sweep.csv"
FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def generate_fig1_framework():
    """Generates Figure 1 vector architecture diagram."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(8.0, 3.4), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)

    def draw_box(x, y, w, h, title, subtitle="", color="#f0f4f8", border="#2b5c8f", lw=1.2, radius=0.15):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad={radius}",
            facecolor=color,
            edgecolor=border,
            linewidth=lw,
            zorder=2
        )
        ax.add_patch(rect)
        if subtitle:
            ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#1a2530", zorder=3)
            ax.text(x + w / 2, y + h * 0.32, subtitle, ha="center", va="center", fontsize=6.8, color="#34495e", zorder=3)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#1a2530", zorder=3)

    # Inputs
    draw_box(0.3, 2.6, 1.8, 1.0, "Operational\nTelemetry", "c = [s, L]^T", color="#e8f4f8", border="#1f77b4")
    draw_box(0.3, 0.6, 1.8, 1.0, "Vibration Signal", "x[n] (fs = 25.6 kHz)", color="#e8f4f8", border="#1f77b4")

    # Stage 1
    draw_box(2.8, 2.4, 2.3, 1.4, "Stage 1: Support Gate", "g1(c) in {0, 1}\nCollinear Bounding Box\n[1400, 2600] RPM, [-0.15, 0.15] kN", color="#eef9f0", border="#2ca02c")

    # Conditional Preprocessing
    draw_box(2.8, 0.6, 2.3, 1.0, "Conditional Preprocessing", "Lw = 2048, Hop = 100%\nZero-mean & Scale (s = 3.899)", color="#fdfefe", border="#7f8c8d")

    # Stage 2
    draw_box(5.8, 1.5, 2.3, 1.4, "Stage 2: Spectral Gate", "g2(x, c) in {0, 1}\nOrder-Domain Envelopes\n11 Order Energy Descriptors", color="#eef9f0", border="#2ca02c")

    # Classifier
    draw_box(8.5, 1.5, 1.3, 1.4, "Frozen 1D-CNN", "f_theta(x)\nMajority Vote\ny_hat in {0,1,2,3}", color="#fef9e7", border="#f39c12")

    # Abstention
    draw_box(5.8, 3.4, 1.6, 0.6, "Abstain (BOT)", "Early Exit", color="#fadbd8", border="#e74c3c")

    # Arrows
    def draw_arrow(x1, y1, x2, y2, label="", color="#2c3e50", lw=1.2, ls="-"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=ls)
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.15
            ax.text(mx, my, label, ha="center", va="center", fontsize=7.2, color=color, fontweight="bold")

    # Connections
    draw_arrow(2.1, 3.1, 2.8, 3.1)
    draw_arrow(2.1, 1.1, 2.8, 1.1)
    draw_arrow(3.95, 2.4, 3.95, 1.6, label="Pass (g1=1)", color="#2ca02c")
    draw_arrow(5.1, 1.1, 5.8, 1.8)
    draw_arrow(5.1, 3.1, 5.8, 3.65, label="Fail (g1=0)", color="#c0392b", ls="--")
    draw_arrow(6.95, 2.9, 6.95, 3.4, label="Fail (g2=0)", color="#c0392b", ls="--")
    draw_arrow(8.1, 2.2, 8.5, 2.2, label="Pass", color="#2ca02c")

    out_pdf = FIG_DIR / "fig1_framework.pdf"
    out_png = FIG_DIR / "fig1_framework.png"
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Figure] Saved Figure 1 to {out_pdf} and {out_png}")


def generate_fig2_three_panel():
    """Generates Figure 2 three-panel summary."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.0,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 5.8,
        "lines.linewidth": 1.1,
        "lines.markersize": 4.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.78), dpi=300)
    plt.subplots_adjust(wspace=0.30, left=0.10, right=0.99, top=0.87, bottom=0.21)

    # Panel (a): Stratum Acceptance
    ax_a = axes[0]
    categories = ["CWRU", "COMPOUND", "NEW_AXIS", "OUT", "IN", "EDGE-2500", "EDGE-1500"]
    values = [0.0, 0.0, 0.0, 0.0, 70.0, 100.0, 100.0]
    is_supp = [False, False, False, False, True, True, True]
    y_pos = np.arange(len(categories))

    for y, val, supp in zip(y_pos, values, is_supp):
        if supp:
            ax_a.barh(y, val, color="#2b5c8f", height=0.56, edgecolor="black", linewidth=0.5, zorder=3)
            ax_a.text(val + 3.0, y, f"{val:.0f}%", va="center", ha="left", fontsize=6.8, color="black")
        else:
            ax_a.scatter([0.0], [y], marker="x", color="#d32f2f", s=40, linewidths=1.4, zorder=4)
            ax_a.text(4.0, y, "0%", va="center", ha="left", fontsize=6.8, color="#a93226")

    ax_a.axhline(3.5, color="dimgrey", linestyle=":", linewidth=0.7)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(categories, fontsize=7.0)
    ax_a.set_xlabel("Acceptance Rate (%)", fontsize=7.5)
    ax_a.set_xlim(-3, 118)
    ax_a.set_xticks([0, 50, 100])
    ax_a.set_title("(a) Record Acceptance Rate", fontsize=8.0)
    ax_a.grid(True, axis="x", alpha=0.3, linestyle="--")

    # Panel (b): Coverage vs SDUST FAR
    ax_b = axes[1]
    if DATA_BASELINES.exists():
        df_sweep = pd.read_csv(DATA_BASELINES)
        mean_sweep = df_sweep.groupby(["model", "val_quantile"]).agg({
            "supported_coverage": "mean",
            "unsupported_fa": "mean",
        }).reset_index()

        model_styles = {
            "energy_ood": {"label": "Energy", "color": "#d95f02", "ls": "-", "marker": "o"},
            "react_energy": {"label": "ReAct", "color": "#7570b3", "ls": "--", "marker": "s"},
            "knn_ood": {"label": "KNN", "color": "#1b9e77", "ls": "-.", "marker": "^"},
            "deep_mahalanobis": {"label": "Mahalanobis", "color": "#e7298a", "ls": ":", "marker": "d"},
        }
        for m_name, st in model_styles.items():
            sub = mean_sweep[mean_sweep["model"] == m_name].sort_values("supported_coverage")
            ax_b.plot(sub["unsupported_fa"] * 100, sub["supported_coverage"] * 100, label=st["label"], color=st["color"], linestyle=st["ls"], alpha=0.85)

        # Matched points
        ax_b.scatter([70.7], [91.1], marker="o", color="#d95f02", s=25, edgecolor="black", linewidth=0.5, zorder=5)
        ax_b.scatter([74.0], [90.0], marker="s", color="#7570b3", s=25, edgecolor="black", linewidth=0.5, zorder=5)
        ax_b.scatter([54.0], [90.0], marker="^", color="#1b9e77", s=25, edgecolor="black", linewidth=0.5, zorder=5)
        ax_b.scatter([54.7], [91.1], marker="d", color="#e7298a", s=25, edgecolor="black", linewidth=0.5, zorder=5)

    ax_b.scatter([58.0], [90.0], marker="v", color="#4d4d4d", s=30, edgecolor="black", linewidth=0.5, label="MSP", zorder=5)
    ax_b.scatter([0.0], [90.0], marker="*", color="#2ca02c", s=95, edgecolor="black", linewidth=0.6, label="SPHARE-BD", zorder=6)
    ax_b.set_xlabel("SDUST Unsupported FAR (%)", fontsize=7.5)
    ax_b.set_ylabel("Supported Coverage (%)", fontsize=7.5)
    ax_b.set_xlim(-3, 85)
    ax_b.set_ylim(60, 102)
    ax_b.set_title("(b) Coverage vs. Unsupported FAR", fontsize=8.0)
    ax_b.grid(True, alpha=0.3, linestyle="--")
    ax_b.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="none", labelspacing=0.2)

    # Panel (c): Synthetic 2D Support Geometry
    ax_c = axes[2]
    # Source hull vertices
    hull_pts = np.array([
        [1000, 600], [1000, 1000], [1400, 1400], [2000, 1400], [2200, 1000], [2000, 600]
    ])
    ch = ConvexHull(hull_pts)
    poly = patches.Polygon(hull_pts[ch.vertices], closed=True, facecolor="#e8f4f8", edgecolor="#2b5c8f", linewidth=1.1, label="2D Hull")
    ax_c.add_patch(poly)

    # 1D box
    box = patches.Rectangle((950, 550), 1300, 900, fill=False, edgecolor="#7f8c8d", linestyle="--", linewidth=0.9, label="1D Box")
    ax_c.add_patch(box)

    # Interior supported points
    in_pts = np.array([[1200, 800], [1600, 1000], [1800, 1200], [1400, 900]])
    ax_c.scatter(in_pts[:, 0], in_pts[:, 1], color="#2ca02c", s=25, label="Supported (IN)", zorder=5)

    # Joint-out points
    joint_pts = np.array([[1050, 1350], [2150, 1350], [1050, 620], [2150, 620]])
    ax_c.scatter(joint_pts[:, 0], joint_pts[:, 1], marker="x", color="#d32f2f", s=30, linewidths=1.2, label="Joint-Out", zorder=5)

    ax_c.set_xlabel("Shaft Speed (RPM)", fontsize=7.5)
    ax_c.set_ylabel("Radial Load (N)", fontsize=7.5)
    ax_c.set_xlim(800, 2400)
    ax_c.set_ylim(400, 1600)
    ax_c.set_title("(c) Synthetic 2D Geometry", fontsize=8.0)
    ax_c.grid(True, alpha=0.3, linestyle="--")
    ax_c.legend(loc="lower left", frameon=True, framealpha=0.9, edgecolor="none", labelspacing=0.2)

    out_pdf = FIG_DIR / "fig2_three_panel.pdf"
    out_png = FIG_DIR / "fig2_three_panel.png"
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Figure] Saved Figure 2 to {out_pdf} and {out_png}")


if __name__ == "__main__":
    generate_fig1_framework()
    generate_fig2_three_panel()
