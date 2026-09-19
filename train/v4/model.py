"""Factorized V4 prefix encoder and conditional edge scorer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from v3.model import AugmentNetV3, V3ModelConfig, V3StateEncoding
from v3.schema import PIECE_KIND_VOCABULARY_SIZE
from v3.tensorize import V3Batch

from .batch import (
    FACTOR_CATEGORY_COUNT,
    FACTOR_FIELD_COUNT,
    FACTOR_FIELD_IDS,
    FactorEdgeBatch,
    PrefixBatchV4,
    V4Batch,
)


@dataclass(frozen=True)
class V4ModelConfig:
    """V4 scorer and explicit encoder choice; old configs retain the baseline."""

    text_vocabulary_size: int
    piece_kind_vocabulary_size: int = PIECE_KIND_VOCABULARY_SIZE
    channels: int = 128
    residual_blocks: int = 6
    model_dim: int = 128
    runtime_message_passes: int = 2
    encoder_kind: str = "baseline"
    hybrid_layers: int = 2
    hybrid_heads: int = 4

    def __post_init__(self) -> None:
        self.v3_config()  # Keep the authoritative dimension/vocabulary checks.
        if self.encoder_kind not in ("baseline", "hybrid"):
            raise ValueError("encoder_kind must be baseline or hybrid")
        if self.hybrid_layers < 1 or self.hybrid_heads < 1:
            raise ValueError("Hybrid layers and heads must be positive")
        if self.encoder_kind == "hybrid" and self.model_dim % self.hybrid_heads:
            raise ValueError("hybrid_heads must divide model_dim")

    def v3_config(self) -> V3ModelConfig:
        return V3ModelConfig(
            text_vocabulary_size=self.text_vocabulary_size,
            piece_kind_vocabulary_size=self.piece_kind_vocabulary_size,
            channels=self.channels,
            residual_blocks=self.residual_blocks,
            model_dim=self.model_dim,
            runtime_message_passes=self.runtime_message_passes,
        )


@dataclass
class V4ModelOutput:
    prefix_logits: torch.Tensor
    value: torch.Tensor
    forced_win_distance: torch.Tensor | None = None


class AugmentNetV4(nn.Module):
    """Baseline or opt-in Hybrid encoder with a conditional factor-edge scorer.

    There is no flat complete-action output in this model. ``score_prefixes`` emits
    one logit for each legal outgoing edge in every packed prefix row.
    """

    def __init__(
        self, config: V4ModelConfig, state_encoder: AugmentNetV3 | None = None
    ) -> None:
        super().__init__()
        self.config = config
        dim = config.model_dim
        # Both choices use the authoritative V3 public-observation frontend.
        # Baseline construction/order and parameter names remain unchanged.
        if state_encoder is None:
            if config.encoder_kind == "hybrid":
                from .hybrid import HybridStateEncoderV4

                state_encoder = HybridStateEncoderV4(
                    config.v3_config(), layers=config.hybrid_layers, heads=config.hybrid_heads
                )
            else:
                state_encoder = AugmentNetV3(config.v3_config())
        if getattr(state_encoder, "encoder_kind", "baseline") != config.encoder_kind:
            raise ValueError("injected state encoder must match encoder_kind")
        if state_encoder.config != config.v3_config():
            raise ValueError("state encoder config must match the V4 public frontend config")
        if config.encoder_kind == "hybrid" and (
            state_encoder.hybrid_layers != config.hybrid_layers
            or state_encoder.hybrid_heads != config.hybrid_heads
        ):
            raise ValueError("injected Hybrid layers/heads must match V4 config")
        self.state_encoder = state_encoder
        self.factor_field_embedding = nn.Embedding(
            FACTOR_FIELD_COUNT + 1, dim, padding_idx=0
        )
        self.factor_category_embedding = nn.Embedding(
            FACTOR_CATEGORY_COUNT + 1, dim, padding_idx=0
        )
        self.factor_numeric = nn.Sequential(
            nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.factor_edge_norm = nn.LayerNorm(dim)
        self.prefix_state_projection = nn.Linear(dim, dim)
        self.prefix_norm = nn.LayerNorm(dim)
        self.prefix_scorer = nn.Sequential(
            nn.Linear(dim * 6, dim * 3),
            nn.SiLU(),
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
        )

    def encode_state(self, states: V3Batch) -> V3StateEncoding:
        """Encode public observation with the explicitly selected architecture."""
        return self.state_encoder.encode_state(states)

    def encode_factor_edges(
        self,
        edges: FactorEdgeBatch,
        prefixes: PrefixBatchV4,
        states: V3Batch,
        encoding: V3StateEncoding,
    ) -> torch.Tensor:
        """Embed typed edges and attach their state/entity/square context."""
        state_index = prefixes.state_index
        raw_index = edges.index
        numeric_index = torch.where(
            raw_index >= 0,
            torch.log1p(raw_index.to(torch.float32)),
            torch.zeros_like(raw_index, dtype=torch.float32),
        )
        numeric_ordinal = torch.log1p(edges.ordinal.to(torch.float32))
        edge = (
            self.factor_field_embedding(edges.field)
            + self.factor_category_embedding(edges.category)
            + self.factor_numeric(torch.stack((numeric_index, numeric_ordinal), dim=-1))
        )

        piece_context = _lookup_entity_slots(
            encoding.piece_entities, states.piece_slot, states.piece_mask, state_index, raw_index
        )
        card_context = _lookup_entity_slots(
            encoding.card_entities, states.card_slot, states.card_mask, state_index, raw_index
        )
        square_context = _lookup_board_squares(encoding.board_tokens, state_index, raw_index)

        field = edges.field
        edge = edge + piece_context * (field == FACTOR_FIELD_IDS["piece"]).unsqueeze(-1)
        edge = edge + card_context * (field == FACTOR_FIELD_IDS["card"]).unsqueeze(-1)
        is_square = (
            (field == FACTOR_FIELD_IDS["source_square"])
            | (field == FACTOR_FIELD_IDS["destination"])
            | (field == FACTOR_FIELD_IDS["target_square"])
            | (field == FACTOR_FIELD_IDS["target_sequence"])
        )
        edge = edge + square_context * is_square.unsqueeze(-1)

        is_promotion = (field == FACTOR_FIELD_IDS["promotion"]) & (raw_index >= 0)
        promotion_indexes = torch.where(is_promotion, raw_index, torch.zeros_like(raw_index))
        if promotion_indexes.numel() and int(promotion_indexes.max()) >= self.config.piece_kind_vocabulary_size:
            raise ValueError("promotion factor exceeds the V3 piece-kind vocabulary")
        promotion = self.state_encoder.piece_kind_embedding(promotion_indexes)
        edge = edge + promotion * is_promotion.unsqueeze(-1)
        return self.factor_edge_norm(edge) * edges.mask.unsqueeze(-1)

    def encode_prefixes(
        self,
        prefixes: PrefixBatchV4,
        states: V3Batch,
        encoding: V3StateEncoding,
    ) -> torch.Tensor:
        """Encode prefix edges as a masked mean; an empty root keeps the state value.

        Canonical trie paths have a fixed field order, and the ``ordinal`` edge
        channel carries the position of ordered ``target_N`` squares. A masked
        multiset embedding is therefore a lossless prefix representation, and it
        keeps the graph free of data-dependent loops so ONNX shapes stay dynamic.
        """
        row_state = encoding.state.index_select(0, prefixes.state_index)
        tokens = self.encode_factor_edges(prefixes.prefix, prefixes, states, encoding)
        mask = prefixes.prefix.mask.unsqueeze(-1)
        pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        hidden = self.prefix_state_projection(row_state) + pooled
        return self.prefix_norm(hidden)

    def score_prefixes(
        self,
        prefixes: PrefixBatchV4,
        states: V3Batch,
        encoding: V3StateEncoding,
    ) -> torch.Tensor:
        """Return conditional logits ``[prefix rows, legal children]``."""
        state = encoding.state.index_select(0, prefixes.state_index)
        prefix = self.encode_prefixes(prefixes, states, encoding)
        child = self.encode_factor_edges(prefixes.children, prefixes, states, encoding)
        width = child.shape[1]
        state = state.unsqueeze(1).expand(-1, width, -1)
        prefix = prefix.unsqueeze(1).expand(-1, width, -1)
        features = torch.cat(
            (state, prefix, child, state * child, prefix * child, state * prefix), dim=-1
        )
        logits = self.prefix_scorer(features).squeeze(-1)
        return logits.masked_fill(
            ~prefixes.legal_mask, torch.finfo(logits.dtype).min
        )

    def forward(self, batch: V4Batch) -> V4ModelOutput:
        encoding = self.encode_state(batch.states)
        return V4ModelOutput(
            prefix_logits=self.score_prefixes(batch.prefixes, batch.states, encoding),
            value=self.state_encoder.value_head(encoding.state).squeeze(-1),
            forced_win_distance=self.state_encoder.forced_win_distance_head(encoding.state).squeeze(-1),
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _lookup_entity_slots(
    entities: torch.Tensor,
    slots: torch.Tensor,
    entity_mask: torch.Tensor,
    state_index: torch.Tensor,
    references: torch.Tensor,
) -> torch.Tensor:
    row_entities = entities.index_select(0, state_index)
    row_slots = slots.index_select(0, state_index)
    row_mask = entity_mask.index_select(0, state_index)
    matches = (
        (row_slots.unsqueeze(1) == references.unsqueeze(-1))
        & row_mask.unsqueeze(1)
        & (references >= 0).unsqueeze(-1)
    )
    return torch.sum(row_entities.unsqueeze(1) * matches.unsqueeze(-1), dim=2)


def _lookup_board_squares(
    board_tokens: torch.Tensor, state_index: torch.Tensor, references: torch.Tensor
) -> torch.Tensor:
    rows = board_tokens.index_select(0, state_index)
    safe = references.clamp(min=0, max=63)
    gathered = torch.gather(
        rows,
        1,
        safe.unsqueeze(-1).expand(-1, -1, rows.shape[-1]),
    )
    return gathered * (references >= 0).unsqueeze(-1)
