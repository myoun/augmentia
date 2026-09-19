"""Opt-in V4 CNN + typed-token attention encoder.

The V3 frontend remains the single implementation of the public observation
contract. Attention refines its state, board, and entity outputs without seeing
legal actions, visits, RNG state, or any private engine data.
"""

from __future__ import annotations

import torch
from torch import nn

from v3.model import AugmentNetV3, V3ModelConfig, V3StateEncoding
from v3.tensorize import V3Batch


class HybridAttentionBlock(nn.Module):
    """Pre-norm attention using exportable matmul, with masked keys AND queries.

    The state summary and 64 board squares are always valid. Thus every row has
    at least one unmasked key, including states without any piece/card records.
    Entity sequence positions are deliberately not embedded: slots identify
    instances for downstream lookup, not their arbitrary storage order.
    """

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.attention_norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.attention_output = nn.Linear(dim, dim)
        self.feed_forward_norm = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = torch.where(mask.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        batch, length, dim = tokens.shape
        qkv = self.qkv(self.attention_norm(tokens)).reshape(
            batch, length, 3, self.heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attended = torch.matmul(torch.softmax(scores, dim=-1), value)
        attended = attended.transpose(1, 2).reshape(batch, length, dim)
        tokens = tokens + self.attention_output(attended)
        tokens = tokens + self.feed_forward(self.feed_forward_norm(tokens))
        return torch.where(mask.unsqueeze(-1), tokens, torch.zeros_like(tokens))


class HybridStateEncoderV4(AugmentNetV3):
    """Public V3 frontend with bidirectional square/entity/state attention.

    Inheritance keeps the baseline frontend parameter names and the existing
    encoder/scorer export interface. This adds new trainable parameters; a V3
    checkpoint is not a trained Hybrid checkpoint. The inherited flat V3 action
    head is unused, just as it is in the baseline V4 encoder.
    """

    encoder_kind = "hybrid"

    def __init__(self, config: V3ModelConfig, *, layers: int = 2, heads: int = 4) -> None:
        if layers < 1 or heads < 1 or config.model_dim % heads:
            raise ValueError("Hybrid layers/heads must be positive and heads must divide model_dim")
        super().__init__(config)
        self.hybrid_layers = layers
        self.hybrid_heads = heads
        self.hybrid_type_embedding = nn.Embedding(4, config.model_dim)
        # Existing V3 square vocabulary is one-based; zero is padding.
        self.register_buffer("hybrid_square_ids", torch.arange(1, 65))
        self.hybrid_blocks = nn.ModuleList(
            HybridAttentionBlock(config.model_dim, heads) for _ in range(layers)
        )
        self.hybrid_output_norm = nn.LayerNorm(config.model_dim)

    def encode_state(self, batch: V3Batch) -> V3StateEncoding:
        base = super().encode_state(batch)
        types = self.hybrid_type_embedding.weight
        tokens = torch.cat((
            base.state.unsqueeze(1) + types[0],
            base.board_tokens + self.square_embedding(self.hybrid_square_ids).unsqueeze(0) + types[1],
            base.piece_entities + types[2],
            base.card_entities + types[3],
        ), dim=1)
        fixed_mask = torch.ones_like(tokens[:, :65, 0], dtype=torch.bool)
        mask = torch.cat((fixed_mask, batch.piece_mask, batch.card_mask), dim=1)
        for block in self.hybrid_blocks:
            tokens = block(tokens, mask)
        tokens = self.hybrid_output_norm(tokens)
        tokens = torch.where(mask.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        piece_end = 65 + base.piece_entities.shape[1]
        return V3StateEncoding(
            state=tokens[:, 0],
            board_tokens=tokens[:, 1:65],
            piece_entities=tokens[:, 65:piece_end],
            card_entities=tokens[:, piece_end:],
        )
