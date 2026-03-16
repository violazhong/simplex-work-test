import os
import sys
import json
import time
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.mess3 import (
    generate_dataset,
    compute_all_beliefs_and_posteriors,
)
from model.transformer import HookedTransformer, TransformerConfig
from model.train import train
from analysis.geometry import (
    extract_activations,
    compute_cev,
    cev_per_position,
    regression_analysis,
    per_component_subspaces,
    additivity_test,
)
from analysis.plotting import (
    plot_training_loss,
    plot_cev_global,
    plot_cev_per_position,
    plot_eff_dim_vs_position,
    plot_regression_rmse,
    plot_subspace_overlaps,
    plot_pca_scatter,
    plot_additivity,
)


# ============================================================================
# Configuration
# ============================================================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data
N_TRAIN = 500_000
N_ANALYSIS = 50_000
SEQ_LENGTH = 15       # content tokens (total = 16 with BOS)

# Model
MODEL_CFG = TransformerConfig(
    n_layers=2,
    d_model=32,
    d_mlp=128,
    n_heads=2,
    d_head=16,
    n_ctx=16,
    d_vocab=4,
    act_fn="gelu",
)

# Training
N_STEPS = 5000
BATCH_SIZE = 1024
LR = 5e-4
CHECKPOINT_STEPS = [0, 100, 500, 1000, 2000, 5000]

# Directories
CHECKPOINT_DIR = "checkpoints"
PLOT_DIR = "plots"
RESULTS_DIR = "results"

for d in [CHECKPOINT_DIR, PLOT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ============================================================================
# Part 1: Data Generation
# ============================================================================
def run_data_generation():
    print("=" * 70)
    print("PART 1: Data Generation")
    print("=" * 70)

    t0 = time.time()

    print(f"\nGenerating training set ({N_TRAIN} sequences)...")
    train_data = generate_dataset(N_TRAIN, SEQ_LENGTH, seed=SEED)
    print(f"  tokens shape: {train_data['tokens'].shape}")
    print(f"  component distribution: {np.bincount(train_data['components'])}")

    print(f"\nGenerating analysis set ({N_ANALYSIS} sequences)...")
    analysis_data = generate_dataset(N_ANALYSIS, SEQ_LENGTH, seed=SEED + 1)
    print(f"  tokens shape: {analysis_data['tokens'].shape}")

    print(f"\nComputing all beliefs and posteriors for analysis set...")
    belief_data = compute_all_beliefs_and_posteriors(
        analysis_data["tokens"],
        analysis_data["components"],
    )
    print(f"  per_component_beliefs shape: {belief_data['per_component_beliefs'].shape}")
    print(f"  posteriors shape: {belief_data['posteriors'].shape}")

    dt = time.time() - t0
    print(f"\nData generation complete ({dt:.1f}s)")

    return train_data, analysis_data, belief_data


# ============================================================================
# Part 2: Model Training
# ============================================================================
def run_training(train_data):
    print("\n" + "=" * 70)
    print("PART 2: Model Training")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = HookedTransformer(MODEL_CFG).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {MODEL_CFG.n_layers} layers, {n_params:,} parameters")
    print(f"Device: {DEVICE}")

    t0 = time.time()
    train_result = train(
        model,
        train_data["tokens"],
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        lr=LR,
        checkpoint_steps=CHECKPOINT_STEPS,
        checkpoint_dir=CHECKPOINT_DIR,
        device=DEVICE,
        seed=SEED,
    )
    dt = time.time() - t0
    print(f"\nTraining complete ({dt:.1f}s)")
    print(f"  Final loss: {train_result['losses'][-1]:.4f}")

    # Plot training loss
    plot_training_loss(
        train_result["steps"],
        train_result["losses"],
        save_path=os.path.join(PLOT_DIR, "training_loss.png"),
    )

    return model, train_result


# ============================================================================
# Part 3: Analysis
# ============================================================================
def run_analysis(model, analysis_data, belief_data):
    print("\n" + "=" * 70)
    print("PART 3: Analysis — Residual Stream Geometry")
    print("=" * 70)

    tokens = analysis_data["tokens"]
    components = analysis_data["components"]
    per_comp_beliefs = belief_data["per_component_beliefs"]
    posteriors = belief_data["posteriors"]
    true_beliefs = belief_data["true_beliefs"]
    K = 3

    # Use a subsample for efficiency in some analyses
    N_sub = min(10_000, len(tokens))
    sub_idx = np.random.default_rng(SEED).choice(len(tokens), N_sub, replace=False)
    tokens_sub = tokens[sub_idx]
    components_sub = components[sub_idx]
    per_comp_sub = per_comp_beliefs[sub_idx]
    posteriors_sub = posteriors[sub_idx]
    true_beliefs_sub = true_beliefs[sub_idx]

    n_layers = MODEL_CFG.n_layers
    layer_indices = list(range(-1, n_layers))   # -1 = input, 0, 1, ...
    layer_names = ["Input (embed)"] + [f"Layer {i}" for i in range(n_layers)]

    # Extract activations at each layer
    print("\nExtracting activations...")
    acts_by_layer = {}
    for layer in layer_indices:
        print(f"  layer={layer} ...")
        acts_by_layer[layer] = extract_activations(
            model, tokens_sub, layer, device=DEVICE, batch_size=512
        )
    print(f"  Activation shape per layer: {acts_by_layer[0].shape}")

    results_summary = {}

    # ------------------------------------------------------------------
    # 3a: CEV analysis
    # ------------------------------------------------------------------
    print("\n--- Analysis 3a: Cumulative Explained Variance ---")

    cev_global_curves = []
    cev_per_pos_data = []
    eff_dims_by_layer = []

    for layer, name in zip(layer_indices, layer_names):
        acts = acts_by_layer[layer]

        # Global
        cev_result = compute_cev(acts)
        cev_global_curves.append(cev_result["cumulative"])
        print(f"  {name}: global eff_dim(95%)={cev_result['eff_dim_95']}")

        # Per position
        per_pos = cev_per_position(acts)
        cev_per_pos_data.append(per_pos["cev_all"])
        eff_dims_by_layer.append(per_pos["eff_dims"])

    plot_cev_global(cev_global_curves, layer_names, K=K,
                    save_path=os.path.join(PLOT_DIR, "cev_global.png"))
    plot_cev_per_position(cev_per_pos_data, layer_names,
                          save_path=os.path.join(PLOT_DIR, "cev_per_position.png"))
    plot_eff_dim_vs_position(eff_dims_by_layer, layer_names, K=K,
                             save_path=os.path.join(PLOT_DIR, "eff_dim_vs_position.png"))

    results_summary["cev"] = {
        name: {"eff_dims_per_position": dims}
        for name, dims in zip(layer_names, eff_dims_by_layer)
    }

    # ------------------------------------------------------------------
    # 3b: Linear regression to belief states
    # ------------------------------------------------------------------
    print("\n--- Analysis 3b: Linear Regression to Beliefs ---")

    regression_by_layer = []
    positions = list(range(acts_by_layer[0].shape[1]))

    for layer, name in zip(layer_indices, layer_names):
        acts = acts_by_layer[layer]
        print(f"  {name}: running regressions...")
        reg = regression_analysis(
            acts, per_comp_sub, posteriors_sub, true_beliefs_sub,
            positions=positions,
        )
        regression_by_layer.append(reg)
        # Print summary at a few positions
        for p in [0, 5, 14]:
            if p < len(reg["concat_rmse"]):
                print(f"    pos {p}: concat={reg['concat_rmse'][p]:.4f} "
                      f"true={reg['true_rmse'][p]:.4f} "
                      f"post={reg['posterior_rmse'][p]:.4f}")

    plot_regression_rmse(regression_by_layer, layer_names,
                         save_path=os.path.join(PLOT_DIR, "regression_rmse.png"))

    results_summary["regression"] = {
        name: {
            "concat_rmse": [float(x) for x in reg["concat_rmse"]],
            "true_rmse": [float(x) for x in reg["true_rmse"]],
            "posterior_rmse": [float(x) for x in reg["posterior_rmse"]],
        }
        for name, reg in zip(layer_names, regression_by_layer)
    }

    # ------------------------------------------------------------------
    # 3c: Subspace overlap and additivity
    # ------------------------------------------------------------------
    print("\n--- Analysis 3c: Subspace Overlap & Additivity ---")

    final_layer = n_layers - 1
    final_acts = acts_by_layer[final_layer]

    selected_positions = [1, 5, 14]
    overlaps_list = []
    additivity_list = []

    for pos in selected_positions:
        sub = per_component_subspaces(final_acts, components_sub, pos, n_dims=2)
        overlaps_list.append(sub["overlaps"])
        print(f"  Position {pos} overlaps:\n{sub['overlaps']}")
        print(f"  Position {pos} per-component eff dims: {sub['eff_dims']}")

        add = additivity_test(final_acts, components_sub, pos)
        additivity_list.append(add)
        print(f"  Position {pos} additivity: sum={add['sum_per_component']}, "
              f"union={add['union_eff_dim']}")

    plot_subspace_overlaps(overlaps_list, selected_positions, K=K,
                           save_path=os.path.join(PLOT_DIR, "subspace_overlaps.png"))
    plot_additivity(additivity_list, selected_positions,
                    save_path=os.path.join(PLOT_DIR, "additivity_test.png"))

    results_summary["subspace"] = {
        f"pos_{pos}": {
            "overlaps": ovl.tolist(),
            "additivity": {
                "sum": add["sum_per_component"],
                "union": add["union_eff_dim"],
            },
        }
        for pos, ovl, add in zip(selected_positions, overlaps_list, additivity_list)
    }

    # ------------------------------------------------------------------
    # 3d: PCA scatter visualisation
    # ------------------------------------------------------------------
    print("\n--- Analysis 3d: PCA Scatter Visualisation ---")

    N_scatter = min(5000, N_sub)
    scatter_idx = np.arange(N_scatter)

    for layer, name in zip(layer_indices, layer_names):
        acts = acts_by_layer[layer][scatter_idx]
        comps = components_sub[scatter_idx]
        tb = true_beliefs_sub[scatter_idx]

        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        plot_pca_scatter(
            acts, comps, tb,
            positions=[1, 7, 14],
            title_prefix=f"{name} — ",
            save_path=os.path.join(PLOT_DIR, f"pca_scatter_{safe_name}.png"),
        )

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------
    summary_path = os.path.join(RESULTS_DIR, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nResults summary saved to {summary_path}")

    return results_summary


# ============================================================================
# Main
# ============================================================================
def main():
    print(f"Experiment: Factored Representations in Transformers")
    print(f"Seed: {SEED} | Device: {DEVICE}")
    print(f"Model: {MODEL_CFG.n_layers}L, d={MODEL_CFG.d_model}, "
          f"h={MODEL_CFG.n_heads}, ctx={MODEL_CFG.n_ctx}")
    print()

    # Part 1
    train_data, analysis_data, belief_data = run_data_generation()

    # Part 2
    model, train_result = run_training(train_data)

    # Part 3 & 4
    results = run_analysis(model, analysis_data, belief_data)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Plots saved in: {PLOT_DIR}/")
    print(f"Checkpoints in: {CHECKPOINT_DIR}/")
    print(f"Results in:     {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
