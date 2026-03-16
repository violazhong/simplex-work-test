"""
Training loop for the HookedTransformer on the Mess3 mixture dataset.

- Next-token cross-entropy loss
- Adam optimizer, lr=5e-4
- Checkpoint saving at specified steps
- Loss logging
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

from model.transformer import HookedTransformer, TransformerConfig


def make_batches(tokens: np.ndarray, batch_size: int, rng: np.random.Generator):
    """Yield shuffled batches of token sequences as torch tensors."""
    n = len(tokens)
    indices = rng.permutation(n)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        yield torch.tensor(tokens[batch_idx], dtype=torch.long)


def train(
    model: HookedTransformer,
    train_tokens: np.ndarray,
    n_steps: int = 5000,
    batch_size: int = 1024,
    lr: float = 5e-4,
    checkpoint_steps: Optional[List[int]] = None,
    checkpoint_dir: str = "checkpoints",
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """
    Train the model with next-token prediction loss.

    Args:
        model:            HookedTransformer instance (already on device)
        train_tokens:     int array (N, L_total) including BOS
        n_steps:          total training steps
        batch_size:       sequences per batch
        lr:               learning rate
        checkpoint_steps: list of step numbers at which to save
        checkpoint_dir:   directory for checkpoint files
        device:           "cpu" or "cuda"
        seed:             random seed for batch shuffling

    Returns:
        dict with "losses" (list of floats) and "steps" (list of ints)
    """
    if checkpoint_steps is None:
        checkpoint_steps = [0, 100, 500, 1000, 2000, 5000]
    os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    losses = []
    steps = []

    # Save initial checkpoint (step 0)
    if 0 in checkpoint_steps:
        path = os.path.join(checkpoint_dir, "step_0000.pt")
        torch.save(model.state_dict(), path)
        print(f"  [checkpoint] step 0 saved")

    step = 0
    epoch = 0
    while step < n_steps:
        epoch += 1
        for batch in make_batches(train_tokens, batch_size, rng):
            if step >= n_steps:
                break

            batch = batch.to(device)
            # Input: all tokens except last; Target: all tokens except first
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]

            logits = model(input_ids)  # (B, T-1, V)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            current_loss = loss.item()
            losses.append(current_loss)
            steps.append(step)

            if step % 200 == 0 or step == 1:
                print(f"  step {step:5d} | loss {current_loss:.4f}")

            if step in checkpoint_steps:
                path = os.path.join(checkpoint_dir, f"step_{step:04d}.pt")
                torch.save(model.state_dict(), path)
                print(f"  [checkpoint] step {step} saved")

    # Save final checkpoint
    path = os.path.join(checkpoint_dir, f"step_final_{step}.pt")
    torch.save(model.state_dict(), path)
    print(f"  [checkpoint] final (step {step}) saved")

    return {"losses": losses, "steps": steps}
