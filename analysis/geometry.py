"""
Analysis of residual stream geometry.

Implements:
  3a: Cumulative Explained Variance (global & per-position), effective dimensionality
  3b: Linear regression to ground-truth belief states
  3c: Subspace identification and orthogonality (overlap metric, additivity test)
  3d: PCA scatter visualisations (data prep; plotting in plotting.py)
"""

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from typing import List, Optional, Tuple

from model.transformer import HookedTransformer


# ============================================================================
# Activation extraction
# ============================================================================
def extract_activations(
    model: HookedTransformer,
    tokens: np.ndarray,
    layer: int,
    device: str = "cpu",
    batch_size: int = 512,
) -> np.ndarray:
    """
    Extract residual stream activations at a given layer for all sequences.

    Args:
        model:  trained HookedTransformer
        tokens: int array (N, L_total) including BOS
        layer:  -1 for input embeddings, 0..n_layers-1 for block outputs
        device: "cpu" or "cuda"
        batch_size: forward pass batch size

    Returns:
        activations: float array (N, L_total-1, d_model)
            — we feed tokens[:, :-1] as input, so positions correspond to
              predicting tokens[:, 1:].  Position 0 = BOS input position,
              position ell = input position for the (ell+1)-th token.
    """
    model.eval()
    N = len(tokens)
    all_acts = []

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = torch.tensor(tokens[start:end, :-1], dtype=torch.long, device=device)
            acts = model.get_residual_stream(batch, layer)  # (B, T-1, d_model)
            all_acts.append(acts.cpu().numpy())

    return np.concatenate(all_acts, axis=0)


# ============================================================================
# 3a: Cumulative Explained Variance
# ============================================================================
def compute_cev(
    activations: np.ndarray,
    positions: Optional[List[int]] = None,
) -> dict:
    """
    Compute PCA cumulative explained variance.

    Args:
        activations: (N, T, d_model) array
        positions:   if None, use all positions (global); else list of position indices

    Returns dict with:
        explained_variance_ratio: array of per-component variance ratios
        cumulative:               cumulative explained variance
        eff_dim_95:               effective dimensionality at 95% threshold
    """
    N, T, D = activations.shape

    if positions is None:
        X = activations.reshape(-1, D)
    else:
        X = activations[:, positions, :].reshape(-1, D)

    pca = PCA(n_components=min(D, X.shape[0]))
    pca.fit(X)

    cev = np.cumsum(pca.explained_variance_ratio_)
    eff_dim = int(np.searchsorted(cev, 0.95) + 1)

    return {
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative": cev,
        "eff_dim_95": eff_dim,
        "pca": pca,
    }


def cev_per_position(activations: np.ndarray) -> dict:
    """
    Compute CEV and effective dimensionality for each context position separately.

    Returns dict with:
        eff_dims:  list of effective dimensionalities, one per position
        cev_all:   list of CEV arrays, one per position
    """
    N, T, D = activations.shape
    eff_dims = []
    cev_all = []

    for pos in range(T):
        result = compute_cev(activations, positions=[pos])
        eff_dims.append(result["eff_dim_95"])
        cev_all.append(result["cumulative"])

    return {"eff_dims": eff_dims, "cev_all": cev_all}


# ============================================================================
# 3b: Linear regression to belief states
# ============================================================================
def regression_analysis(
    activations: np.ndarray,
    per_component_beliefs: np.ndarray,
    posteriors: np.ndarray,
    true_beliefs: np.ndarray,
    positions: Optional[List[int]] = None,
) -> dict:
    """
    Linear regression from activations to ground-truth belief targets.

    Alignment between activations and beliefs:
      - activations have T = L_total - 1 positions (model input = tokens[:, :-1])
      - Position p in activations: model has seen tokens[0..p] = [BOS, t1, ..., t_p]
      - Corresponding belief index = p:
          p=0 -> prior (before any content token)
          p=1 -> belief after t1
          ...
          p=14 -> belief after t14

    Args:
        activations:           (N, T, d_model)   T = L_total - 1 = 15
        per_component_beliefs: (N, L_total, K, 3) L_total = 16
        posteriors:            (N, L_total, K)
        true_beliefs:          (N, L_total, 3)
        positions:             list of activation positions to analyse (if None, all)

    Returns dict with:
        concat_rmse:    RMSE for concatenated per-component beliefs, per position
        true_rmse:      RMSE for true component belief, per position
        posterior_rmse: RMSE for posterior over components, per position
    """
    N, T, D = activations.shape
    _, L_total, K, _ = per_component_beliefs.shape

    if positions is None:
        positions = list(range(T))

    concat_rmse = []
    true_rmse = []
    posterior_rmse = []

    for pos in positions:
        X = activations[:, pos, :]  # (N, D)

        # Target 1: Concatenated per-component beliefs -> R^{3K}
        y_concat = per_component_beliefs[:, pos, :, :].reshape(N, -1)  # (N, 3K)

        # Target 2: True component belief -> R^3
        y_true = true_beliefs[:, pos, :]  # (N, 3)

        # Target 3: Posterior over components -> R^K
        y_post = posteriors[:, pos, :]  # (N, K)

        # Fit ridge regressions (small alpha for regularisation)
        for target, rmse_list in [(y_concat, concat_rmse),
                                   (y_true, true_rmse),
                                   (y_post, posterior_rmse)]:
            reg = Ridge(alpha=1e-4)
            reg.fit(X, target)
            pred = reg.predict(X)
            rmse = np.sqrt(np.mean((pred - target) ** 2))
            rmse_list.append(rmse)

    return {
        "concat_rmse": concat_rmse,
        "true_rmse": true_rmse,
        "posterior_rmse": posterior_rmse,
        "positions": positions,
    }


# ============================================================================
# 3c: Subspace identification and orthogonality
# ============================================================================
def subspace_overlap(Q_A: np.ndarray, Q_B: np.ndarray) -> float:
    """
    Normalised overlap between two subspaces.

    overlap(A, B) = (1/d_min) * ||Q_A^T @ Q_B||_F^2

    where Q_A (D, d_A) and Q_B (D, d_B) are orthonormal bases.
    """
    d_min = min(Q_A.shape[1], Q_B.shape[1])
    cross = Q_A.T @ Q_B  # (d_A, d_B)
    return float(np.sum(cross ** 2) / d_min)


def per_component_subspaces(
    activations: np.ndarray,
    components: np.ndarray,
    position: int,
    n_dims: int = 2,
) -> dict:
    """
    For each component, compute the top `n_dims` PCA subspace of activations
    at the given position.

    Returns dict with:
        bases:      list of (d_model, n_dims) orthonormal basis arrays, one per component
        eff_dims:   list of effective dimensionalities (at 95%) per component
        overlaps:   K x K matrix of pairwise overlaps
    """
    K = int(components.max()) + 1
    N, T, D = activations.shape

    bases = []
    eff_dims = []

    for k in range(K):
        mask = components == k
        X = activations[mask, position, :]  # (n_k, D)
        X = X - X.mean(axis=0)

        pca = PCA(n_components=min(D, X.shape[0]))
        pca.fit(X)

        Q = pca.components_[:n_dims].T  # (D, n_dims) — columns are PCs
        bases.append(Q)

        cev = np.cumsum(pca.explained_variance_ratio_)
        eff_dim = int(np.searchsorted(cev, 0.95) + 1)
        eff_dims.append(eff_dim)

    # Pairwise overlaps
    overlaps = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            overlaps[i, j] = subspace_overlap(bases[i], bases[j])

    return {"bases": bases, "eff_dims": eff_dims, "overlaps": overlaps}


def additivity_test(
    activations: np.ndarray,
    components: np.ndarray,
    position: int,
) -> dict:
    """
    Compare sum of per-component effective dims vs. union effective dim.

    If representations are factored, sum ~ union (subspaces are orthogonal).
    """
    K = int(components.max()) + 1
    N, T, D = activations.shape

    # Per-component effective dims
    per_comp_eff = []
    for k in range(K):
        mask = components == k
        X = activations[mask, position, :]
        X = X - X.mean(axis=0)
        pca = PCA(n_components=min(D, X.shape[0]))
        pca.fit(X)
        cev = np.cumsum(pca.explained_variance_ratio_)
        per_comp_eff.append(int(np.searchsorted(cev, 0.95) + 1))

    # Union effective dim
    X_all = activations[:, position, :]
    X_all = X_all - X_all.mean(axis=0)
    pca = PCA(n_components=min(D, X_all.shape[0]))
    pca.fit(X_all)
    cev = np.cumsum(pca.explained_variance_ratio_)
    union_eff = int(np.searchsorted(cev, 0.95) + 1)

    return {
        "per_component_eff_dims": per_comp_eff,
        "sum_per_component": sum(per_comp_eff),
        "union_eff_dim": union_eff,
    }
