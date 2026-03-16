"""
Mess3 Hidden Markov Model data generation — VECTORISED.

Implements:
  - Mess3 transition matrix construction
  - Batch sequence generation (all sequences for a component in parallel)
  - Batch belief computation
  - Non-ergodic mixture dataset creation (K=3 components)

All heavy loops are over the 15 time-steps only; the sequence dimension
is handled by NumPy broadcasting / einsum, giving ~50-100x speedup over
the naive per-sequence Python loop.
"""

import numpy as np
from typing import List, Tuple

BOS_TOKEN = 3
VOCAB_SIZE = 4          # {0, 1, 2, BOS}
NUM_HIDDEN_STATES = 3


# ---------------------------------------------------------------------------
# Mess3 transition matrices
# ---------------------------------------------------------------------------
def build_mess3_matrices(alpha: float, x: float) -> np.ndarray:
    """
    Build the 3 token-labelled transition matrices for a Mess3 process.

    Returns:
        T: (3, 3, 3) array where T[token, from_state, to_state].
    """
    b = (1 - alpha) / 2
    y = 1 - 2 * x
    T = np.zeros((3, 3, 3), dtype=np.float64)

    T[0] = [[alpha*y, b*x,     b*x],
            [alpha*x, b*y,     b*x],
            [alpha*x, b*x,     b*y]]

    T[1] = [[b*y,     alpha*x, b*x],
            [b*x,     alpha*y, b*x],
            [b*x,     alpha*x, b*y]]

    T[2] = [[b*y,     b*x,     alpha*x],
            [b*x,     b*y,     alpha*x],
            [b*x,     b*x,     alpha*y]]

    assert np.allclose(T.sum(axis=0).sum(axis=1), 1.0), "T_net rows must sum to 1"
    return T


# Default process parameters from the experiment spec
PROCESS_PARAMS = [(0.5, 0.1), (0.8, 0.3), (0.6, 0.05)]

# T_ALL[k] = (3, 3, 3) transition matrices for process k
T_ALL = [build_mess3_matrices(a, x) for a, x in PROCESS_PARAMS]

# Precompute CDF tables for fast vectorised sampling:
# JOINT_CDF[k] = (3, 9) cumulative distribution over (token, next_state) for each state
JOINT_CDF = []
for _k in range(3):
    cdf_k = np.zeros((3, 9), dtype=np.float64)
    for _s in range(3):
        joint = np.array([T_ALL[_k][tok][_s, :] for tok in range(3)]).ravel()
        joint /= joint.sum()
        cdf_k[_s] = np.cumsum(joint)
    JOINT_CDF.append(cdf_k)


# ---------------------------------------------------------------------------
# Vectorised dataset generation
# ---------------------------------------------------------------------------
def generate_dataset(
    n_sequences: int,
    seq_length: int = 15,
    seed: int = 42,
) -> dict:
    """
    Generate a non-ergodic mixture dataset (vectorised).

    Groups sequences by component and generates all sequences for each
    component simultaneously.  The inner loop is only `seq_length` steps;
    the sequence dimension is fully vectorised.

    Args:
        n_sequences:  number of sequences
        seq_length:   number of *content* tokens (total length = seq_length + 1 with BOS)
        seed:         random seed

    Returns dict with:
        tokens:      int array (n_sequences, seq_length+1)  — BOS at position 0
        components:  int array (n_sequences,)
        beliefs:     float array (n_sequences, seq_length+1, 3) — belief under true component
    """
    K = 3
    rng = np.random.default_rng(seed)
    total_len = seq_length + 1

    all_components = rng.integers(0, K, size=n_sequences)
    all_tokens = np.empty((n_sequences, total_len), dtype=np.int64)
    all_tokens[:, 0] = BOS_TOKEN
    all_beliefs = np.empty((n_sequences, total_len, 3), dtype=np.float64)
    all_beliefs[:, 0, :] = 1.0 / 3.0  # prior

    for k in range(K):
        mask = all_components == k
        nk = mask.sum()
        if nk == 0:
            continue

        T_k = T_ALL[k]           # (3, 3, 3)
        cdf_k = JOINT_CDF[k]     # (3, 9)

        # Initial hidden states: uniform over {0,1,2}
        states = rng.integers(0, 3, size=nk)
        tokens_k = np.empty((nk, seq_length), dtype=np.int64)

        # Belief tracking
        eta = np.full((nk, 3), 1.0 / 3.0, dtype=np.float64)
        beliefs_k = np.empty((nk, seq_length + 1, 3), dtype=np.float64)
        beliefs_k[:, 0, :] = eta

        for t in range(seq_length):
            # Vectorised CDF-inversion sampling
            u = rng.random(nk)
            cdf_rows = cdf_k[states]                        # (nk, 9)
            idx = np.clip((cdf_rows < u[:, None]).sum(axis=1), 0, 8)
            tok = idx // 3
            next_state = idx % 3

            tokens_k[:, t] = tok
            states = next_state

            # Batch belief update: eta = eta @ T_k[tok]
            T_sel = T_k[tok]                                # (nk, 3, 3)
            eta = np.einsum('ni,nij->nj', eta, T_sel)
            eta /= eta.sum(axis=1, keepdims=True)
            beliefs_k[:, t + 1, :] = eta

        all_tokens[mask, 1:] = tokens_k
        all_beliefs[mask] = beliefs_k

    return {
        "tokens": all_tokens,
        "components": all_components,
        "beliefs": all_beliefs,
    }


# ---------------------------------------------------------------------------
# Vectorised belief & posterior computation for analysis
# ---------------------------------------------------------------------------
def compute_all_beliefs_and_posteriors(
    tokens: np.ndarray,
    components: np.ndarray,
) -> dict:
    """
    For the analysis set, compute (vectorised over all N sequences):
      - Per-component belief vectors for ALL K=3 components
      - Posterior P(k | x_{1:ell}) over components at each position
      - True beliefs (under the generating component)

    Args:
        tokens:      (N, L_total) with BOS at position 0
        components:  (N,) ground-truth component labels

    Returns dict with:
        per_component_beliefs: (N, L_total, K, 3)
        posteriors:            (N, L_total, K)
        true_beliefs:          (N, L_total, 3)
    """
    K = 3
    N, L_total = tokens.shape
    seq_length = L_total - 1

    per_comp_beliefs = np.zeros((N, L_total, K, 3), dtype=np.float64)
    cum_lls = np.zeros((N, L_total, K), dtype=np.float64)

    content_tokens = tokens[:, 1:]  # (N, seq_length)

    for k in range(K):
        T_k = T_ALL[k]  # (3, 3, 3)
        eta = np.full((N, 3), 1.0 / 3.0, dtype=np.float64)
        per_comp_beliefs[:, 0, k, :] = eta

        for t in range(seq_length):
            tok_t = content_tokens[:, t]                    # (N,)
            T_sel = T_k[tok_t]                              # (N, 3, 3)
            unnorm = np.einsum('ni,nij->nj', eta, T_sel)    # (N, 3)
            p_tok = unnorm.sum(axis=1)                      # (N,)
            cum_lls[:, t + 1, k] = cum_lls[:, t, k] + np.log(p_tok + 1e-300)
            eta = unnorm / p_tok[:, None]
            per_comp_beliefs[:, t + 1, k, :] = eta

    # Posterior P(k | x_{1:ell}) via Bayes rule with uniform prior
    log_probs = cum_lls - cum_lls.max(axis=2, keepdims=True)
    probs = np.exp(log_probs)
    posteriors = probs / probs.sum(axis=2, keepdims=True)

    # True beliefs: select the generating component per sequence
    true_beliefs = per_comp_beliefs[np.arange(N), :, components, :]  # (N, L_total, 3)

    return {
        "per_component_beliefs": per_comp_beliefs,
        "posteriors": posteriors,
        "true_beliefs": true_beliefs,
    }
