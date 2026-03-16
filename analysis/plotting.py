"""
Plotting functions for the factored representations experiment.

Produces all figures specified in Part 4 of the experiment spec.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Optional
import os


PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

COMPONENT_COLORS = ["#e41a1c", "#377eb8", "#4daf4a"]  # red, blue, green
LAYER_COLORS = ["#984ea3", "#ff7f00", "#a65628", "#f781bf"]


def plot_training_loss(steps, losses, save_path=None):
    """Plot training loss curve."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, losses, linewidth=0.5, alpha=0.3, color="gray")
    # Smoothed version
    window = min(100, max(1, len(losses) // 10))
    if window > 1:
        kernel = np.ones(window) / window
        smoothed = np.convolve(losses, kernel, mode="valid")
        ax.plot(steps[window-1:], smoothed, linewidth=1.5, color="C0", label="smoothed")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = save_path or os.path.join(PLOT_DIR, "training_loss.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cev_global(cev_by_layer, layer_names, K=3, save_path=None):
    """
    Plot CEV curves at each layer (global, all positions pooled).
    Mark 2K and 3K-1 dimensionality thresholds.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (cev, name) in enumerate(zip(cev_by_layer, layer_names)):
        n_comp = len(cev)
        ax.plot(range(1, n_comp + 1), cev, marker="o", markersize=3,
                label=name, color=LAYER_COLORS[i % len(LAYER_COLORS)])

    ax.axvline(x=2*K, color="red", linestyle="--", alpha=0.7, label=f"2K={2*K}")
    ax.axvline(x=3*K-1, color="blue", linestyle="--", alpha=0.7, label=f"3K-1={3*K-1}")
    ax.axhline(y=0.95, color="gray", linestyle=":", alpha=0.5, label="95% threshold")
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("CEV — Global (all positions)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20)
    fig.tight_layout()
    path = save_path or os.path.join(PLOT_DIR, "cev_global.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cev_per_position(cev_per_pos_by_layer, layer_names, save_path=None):
    """
    For each layer, plot several CEV curves (one per selected position).
    """
    n_layers = len(cev_per_pos_by_layer)
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5), squeeze=False)
    selected_positions = [0, 2, 5, 9, 14]  # BOS, early, mid, late, final

    for li, (cev_data, name) in enumerate(zip(cev_per_pos_by_layer, layer_names)):
        ax = axes[0, li]
        for pos in selected_positions:
            if pos < len(cev_data):
                cev = cev_data[pos]
                ax.plot(range(1, len(cev) + 1), cev, marker=".", markersize=2,
                        label=f"pos {pos}", alpha=0.8)
        ax.axhline(y=0.95, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("# PCA components")
        ax.set_ylabel("CEV")
        ax.set_title(f"CEV per position — {name}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 20)

    fig.tight_layout()
    path = save_path or os.path.join(PLOT_DIR, "cev_per_position.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_eff_dim_vs_position(eff_dims_by_layer, layer_names, K=3, save_path=None):
    """
    Effective dimensionality (95% CEV) vs context position, for each layer.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (dims, name) in enumerate(zip(eff_dims_by_layer, layer_names)):
        ax.plot(range(len(dims)), dims, marker="o", markersize=4,
                label=name, color=LAYER_COLORS[i % len(LAYER_COLORS)])

    ax.axhline(y=2*K, color="red", linestyle="--", alpha=0.5, label=f"2K={2*K}")
    ax.axhline(y=3*K-1, color="blue", linestyle="--", alpha=0.5, label=f"3K-1={3*K-1}")
    ax.set_xlabel("Context position (0=BOS)")
    ax.set_ylabel("Effective dimensionality (95% CEV)")
    ax.set_title("Effective Dimensionality vs Context Position")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = save_path or os.path.join(PLOT_DIR, "eff_dim_vs_position.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_regression_rmse(
    regression_results_by_layer,
    layer_names,
    save_path=None,
):
    """
    RMSE vs context position for each regression target, at each layer.
    """
    targets = ["concat_rmse", "true_rmse", "posterior_rmse"]
    target_labels = [
        "Concatenated beliefs (R^9)",
        "True component belief (R^3)",
        "Posterior P(k|x) (R^3)",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ti, (target_key, label) in enumerate(zip(targets, target_labels)):
        ax = axes[ti]
        for li, (result, name) in enumerate(zip(regression_results_by_layer, layer_names)):
            positions = result["positions"]
            rmse = result[target_key]
            ax.plot(positions, rmse, marker="o", markersize=3,
                    label=name, color=LAYER_COLORS[li % len(LAYER_COLORS)])
        ax.set_xlabel("Context position")
        ax.set_ylabel("RMSE")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = save_path or os.path.join(PLOT_DIR, "regression_rmse.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_subspace_overlaps(overlaps_by_position, positions, K=3, save_path=None):
    """
    Plot subspace overlap matrices at selected positions.
    """
    n = len(positions)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    if n == 1:
        axes = [axes]

    for i, (overlap, pos) in enumerate(zip(overlaps_by_position, positions)):
        ax = axes[i]
        im = ax.imshow(overlap, vmin=0, vmax=1, cmap="RdYlBu_r")
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels([f"Comp {k}" for k in range(K)], fontsize=8)
        ax.set_yticklabels([f"Comp {k}" for k in range(K)], fontsize=8)
        ax.set_title(f"Position {pos}")
        # Annotate values
        for ii in range(K):
            for jj in range(K):
                ax.text(jj, ii, f"{overlap[ii,jj]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Subspace Overlap (normalised)", fontsize=12)
    fig.tight_layout()
    path = save_path or os.path.join(PLOT_DIR, "subspace_overlaps.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_pca_scatter(
    activations: np.ndarray,
    components: np.ndarray,
    true_beliefs: np.ndarray,
    positions: List[int],
    title_prefix: str = "",
    save_path: Optional[str] = None,
):
    """
    2D PCA scatter plots coloured by component (top row) and by belief
    confidence (bottom row).
    """
    from sklearn.decomposition import PCA

    n_pos = len(positions)
    fig, axes = plt.subplots(2, n_pos, figsize=(5 * n_pos, 9))
    if n_pos == 1:
        axes = axes.reshape(2, 1)

    for pi, pos in enumerate(positions):
        X = activations[:, pos, :]
        pca = PCA(n_components=2)
        Z = pca.fit_transform(X)

        # Row 1: coloured by component
        ax = axes[0, pi]
        for k in range(3):
            mask = components == k
            ax.scatter(Z[mask, 0], Z[mask, 1], s=1, alpha=0.3,
                       color=COMPONENT_COLORS[k], label=f"Comp {k}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"{title_prefix}pos={pos} (by component)")
        ax.legend(markerscale=5, fontsize=7)

        # Row 2: coloured by belief confidence (max of belief vector)
        ax = axes[1, pi]
        belief_confidence = true_beliefs[:, pos, :].max(axis=1)
        sc = ax.scatter(Z[:, 0], Z[:, 1], s=1, alpha=0.3,
                        c=belief_confidence, cmap="viridis", vmin=0.33, vmax=1.0)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"{title_prefix}pos={pos} (belief confidence)")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    path = save_path or os.path.join(PLOT_DIR, "pca_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_additivity(additivity_by_position, positions, save_path=None):
    """Bar chart comparing sum of per-component eff dims vs union eff dim."""
    sums = [a["sum_per_component"] for a in additivity_by_position]
    unions = [a["union_eff_dim"] for a in additivity_by_position]

    x = np.arange(len(positions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width/2, sums, width, label="Sum of per-component", color="C0")
    ax.bar(x + width/2, unions, width, label="Union", color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"pos {p}" for p in positions])
    ax.set_ylabel("Effective dimensionality")
    ax.set_title("Additivity Test: Sum vs Union Effective Dimensionality")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = save_path or os.path.join(PLOT_DIR, "additivity_test.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
