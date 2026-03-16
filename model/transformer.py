"""
Minimal GPT-2 style decoder-only transformer with residual stream hooks.

Designed for the factored representations experiment:
  - Small architecture (2-3 layers, d_model=32)
  - Hook dict populated during forward pass for activation extraction
  - Next-token prediction (causal LM)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class TransformerConfig:
    n_layers: int = 2
    d_model: int = 32
    d_mlp: int = 128
    n_heads: int = 2
    d_head: int = 16
    n_ctx: int = 16
    d_vocab: int = 4
    act_fn: str = "gelu"


class Attention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.W_K = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.W_V = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.W_O = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.n_ctx, cfg.n_ctx)).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        nh, dh = self.cfg.n_heads, self.cfg.d_head

        Q = self.W_Q(x).view(B, T, nh, dh).transpose(1, 2)  # (B, nh, T, dh)
        K = self.W_K(x).view(B, T, nh, dh).transpose(1, 2)
        V = self.W_V(x).view(B, T, nh, dh).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(dh)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, nh * dh)
        return self.W_O(out)


class MLP(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.fc2 = nn.Linear(cfg.d_mlp, cfg.d_model)
        self.act_fn = F.gelu if cfg.act_fn == "gelu" else F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_fn(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class HookedTransformer(nn.Module):
    """
    GPT-2 style transformer with hooks for residual stream extraction.

    After each forward pass, `self.hook_cache` contains:
        "hook_embed"              : token embedding output          (B, T, d_model)
        "hook_pos_embed"          : positional embedding output     (B, T, d_model)
        "blocks.{i}.hook_resid_post" : residual stream after block i (B, T, d_model)
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.d_vocab, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.n_ctx, cfg.d_model)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.d_vocab, bias=False)

        self.hook_cache: Dict[str, torch.Tensor] = {}

        self._init_weights()

    def _init_weights(self):
        """Small init for stable training of tiny models."""
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        store_hooks: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) int tensor of token indices
            store_hooks: if True, populate self.hook_cache

        Returns:
            logits: (B, T, d_vocab)
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)

        tok_emb = self.embed(input_ids)        # (B, T, d_model)
        pos_emb = self.pos_embed(positions)    # (1, T, d_model)

        if store_hooks:
            self.hook_cache = {}
            self.hook_cache["hook_embed"] = tok_emb.detach()
            self.hook_cache["hook_pos_embed"] = pos_emb.expand(B, -1, -1).detach()

        x = tok_emb + pos_emb

        for i, block in enumerate(self.blocks):
            x = block(x)
            if store_hooks:
                self.hook_cache[f"blocks.{i}.hook_resid_post"] = x.detach()

        x = self.ln_final(x)
        logits = self.unembed(x)
        return logits

    def get_residual_stream(
        self,
        input_ids: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        """
        Convenience: run forward with hooks and return residual stream at `layer`.
        layer=-1 means the input (embed + pos_embed), layer=0..n_layers-1 are block outputs.
        """
        self.forward(input_ids, store_hooks=True)
        if layer == -1:
            return self.hook_cache["hook_embed"] + self.hook_cache["hook_pos_embed"]
        return self.hook_cache[f"blocks.{layer}.hook_resid_post"]
