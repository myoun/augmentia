"""First hybrid neural network for ObservationV3 and dynamic ActionV3 policies."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import nn
from torch.nn import functional as F

from .schema import GLOBAL_SHAPE, RUNTIME_FIELD_COUNT, SPATIAL_SHAPE
from .tensorize import V3Batch


@dataclass(frozen=True)
class V3ModelConfig:
    text_vocabulary_size: int
    channels: int = 128
    residual_blocks: int = 6
    model_dim: int = 128
    runtime_message_passes: int = 2

    def __post_init__(self) -> None:
        if self.text_vocabulary_size < 2:
            raise ValueError(
                "text_vocabulary_size must include <none> and at least one token"
            )
        if self.channels <= 0 or self.model_dim <= 0 or self.residual_blocks <= 0:
            raise ValueError(
                "model dimensions and residual block count must be positive"
            )
        if self.channels % 8 != 0:
            raise ValueError("channels must be divisible by 8 for GroupNorm")
        if self.runtime_message_passes < 1:
            raise ValueError("runtime_message_passes must be positive")


@dataclass
class V3ModelOutput:
    policy_logits: torch.Tensor
    value: torch.Tensor


@dataclass
class V3Loss:
    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    policy_samples: int
    value_samples: int


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.norm1(self.conv1(inputs)))
        return F.silu(inputs + self.norm2(self.conv2(hidden)))


class AugmentNetV3(nn.Module):
    """CNN state encoder plus exact-vocabulary entity and legal-action scoring."""

    def __init__(self, config: V3ModelConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.model_dim

        self.board_stem = nn.Sequential(
            nn.Conv2d(
                SPATIAL_SHAPE[0], config.channels, kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(8, config.channels),
            nn.SiLU(),
        )
        self.board_blocks = nn.Sequential(
            *(ResidualBlock(config.channels) for _ in range(config.residual_blocks))
        )
        self.board_projection = nn.Conv2d(config.channels, dim, kernel_size=1)
        self.global_encoder = _mlp(GLOBAL_SHAPE[0], dim, dim)
        self.entity_count_encoder = _mlp(2, dim, dim)

        self.text_embedding = nn.Embedding(
            config.text_vocabulary_size, dim, padding_idx=0
        )
        self.owner_embedding = nn.Embedding(3, dim, padding_idx=0)
        self.square_embedding = nn.Embedding(65, dim, padding_idx=0)
        self.piece_kind_embedding = nn.Embedding(52, dim, padding_idx=0)

        self.piece_numeric = _mlp(2, dim, dim)
        self.piece_norm = nn.LayerNorm(dim)
        self.card_numeric = _mlp(5, dim, dim)
        self.card_norm = nn.LayerNorm(dim)
        self.state_text_position = _mlp(1, dim, dim)
        self.global_norm = nn.LayerNorm(dim)

        self.phase_kind_embedding = nn.Embedding(5, dim, padding_idx=0)
        self.phase_numeric = _mlp(2, dim, dim)
        self.phase_text_position = _mlp(1, dim, dim)
        self.phase_norm = nn.LayerNorm(dim)

        self.runtime_field_embedding = nn.Embedding(
            RUNTIME_FIELD_COUNT + 1, dim, padding_idx=0
        )
        self.runtime_type_embedding = nn.Embedding(13, dim, padding_idx=0)
        self.runtime_numeric = _mlp(7, dim, dim)
        self.runtime_base_norm = nn.LayerNorm(dim)
        self.runtime_parent_update = _mlp(dim * 2, dim * 2, dim)
        self.runtime_attention = nn.Linear(dim, 1)

        self.action_kind_embedding = nn.Embedding(6, dim, padding_idx=0)
        self.ability_embedding = nn.Embedding(7, dim, padding_idx=0)
        self.target_kind_embedding = nn.Embedding(5, dim, padding_idx=0)
        self.action_numeric = _mlp(6, dim, dim)
        self.action_from_projection = nn.Linear(dim * 2, dim)
        self.action_to_projection = nn.Linear(dim * 2, dim)
        self.target_position = _mlp(1, dim, dim)
        self.action_norm = nn.LayerNorm(dim)

        self.state_fusion = nn.Sequential(
            nn.Linear(dim * 6, dim * 3),
            nn.SiLU(),
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.SiLU(),
            nn.LayerNorm(dim),
        )
        self.policy_scorer = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
            nn.Tanh(),
        )

    def forward(self, batch: V3Batch) -> V3ModelOutput:
        self._validate_batch(batch)
        board_hidden = self.board_blocks(self.board_stem(batch.spatial))
        board_map = self.board_projection(board_hidden)
        board_tokens = board_map.flatten(2).transpose(1, 2)
        board_pool = board_tokens.mean(dim=1)
        state_text_tokens = self.text_embedding(
            batch.state_text
        ) + self.state_text_position(batch.state_text_position)
        global_state = self.global_norm(
            self.global_encoder(batch.global_features)
            + self.entity_count_encoder(batch.entity_counts)
            + _masked_mean(state_text_tokens, batch.state_text_mask, dim=1)
        )

        piece_entities = self.piece_norm(
            self.piece_kind_embedding(batch.piece_kind)
            + self.square_embedding(batch.piece_square)
            + self.owner_embedding(batch.piece_owner)
            + self.piece_numeric(batch.piece_numeric)
        )
        piece_entities = piece_entities * batch.piece_mask.unsqueeze(-1)
        piece_pool = _masked_mean(piece_entities, batch.piece_mask, dim=1)

        card_entities = self.card_norm(
            self.text_embedding(batch.card_definition)
            + self.owner_embedding(batch.card_owner)
            + self.card_numeric(batch.card_numeric)
        )
        card_entities = card_entities * batch.card_mask.unsqueeze(-1)
        card_pool = _masked_mean(card_entities, batch.card_mask, dim=1)

        phase = self._encode_phase(batch, piece_entities, card_entities)
        runtime = self._encode_runtime(batch, piece_entities, card_entities)
        state = self.state_fusion(
            torch.cat(
                [board_pool, global_state, piece_pool, card_pool, phase, runtime],
                dim=-1,
            )
        )
        actions = self._encode_actions(
            batch, board_tokens, piece_entities, card_entities
        )
        expanded_state = state.unsqueeze(1).expand_as(actions)
        logits = self.policy_scorer(
            torch.cat([expanded_state, actions, expanded_state * actions], dim=-1)
        ).squeeze(-1)
        logits = logits.masked_fill(~batch.action_mask, torch.finfo(logits.dtype).min)
        return V3ModelOutput(
            policy_logits=logits, value=self.value_head(state).squeeze(-1)
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _encode_phase(
        self,
        batch: V3Batch,
        piece_entities: torch.Tensor,
        card_entities: torch.Tensor,
    ) -> torch.Tensor:
        text_tokens = self.text_embedding(batch.phase_text) + self.phase_text_position(
            batch.phase_text_position
        )
        text_pool = _masked_mean(text_tokens, batch.phase_text_mask, dim=1)
        piece_ref = _lookup_entities(
            piece_entities,
            batch.piece_slot,
            batch.piece_mask,
            batch.phase_piece_ref.unsqueeze(1),
        ).squeeze(1)
        card_ref = _lookup_entities(
            card_entities,
            batch.card_slot,
            batch.card_mask,
            batch.phase_card_ref.unsqueeze(1),
        ).squeeze(1)
        return self.phase_norm(
            self.phase_kind_embedding(batch.phase_kind)
            + self.owner_embedding(batch.phase_owner)
            + self.phase_numeric(batch.phase_numeric)
            + text_pool
            + piece_ref
            + card_ref
        )

    def _encode_runtime(
        self,
        batch: V3Batch,
        piece_entities: torch.Tensor,
        card_entities: torch.Tensor,
    ) -> torch.Tensor:
        piece_refs = _lookup_entities(
            piece_entities,
            batch.piece_slot,
            batch.piece_mask,
            batch.runtime_piece_ref,
        )
        card_refs = _lookup_entities(
            card_entities,
            batch.card_slot,
            batch.card_mask,
            batch.runtime_card_ref,
        )
        base = self.runtime_base_norm(
            self.runtime_field_embedding(batch.runtime_field)
            + self.runtime_type_embedding(batch.runtime_type)
            + self.text_embedding(batch.runtime_text)
            + self.owner_embedding(batch.runtime_player)
            + self.piece_kind_embedding(batch.runtime_piece_kind)
            + self.square_embedding(batch.runtime_square)
            + self.runtime_numeric(batch.runtime_numeric)
            + piece_refs
            + card_refs
        )
        hidden = base
        for _ in range(self.config.runtime_message_passes):
            parent = _lookup_parent(hidden, batch.runtime_parent, batch.runtime_mask)
            hidden = self.runtime_parent_update(torch.cat([base, parent], dim=-1))
            hidden = hidden * batch.runtime_mask.unsqueeze(-1)
        scores = self.runtime_attention(hidden).squeeze(-1)
        weights = _masked_softmax(scores, batch.runtime_mask, dim=1)
        return torch.sum(hidden * weights.unsqueeze(-1), dim=1)

    def _encode_actions(
        self,
        batch: V3Batch,
        board_tokens: torch.Tensor,
        piece_entities: torch.Tensor,
        card_entities: torch.Tensor,
    ) -> torch.Tensor:
        piece_refs = _lookup_entities(
            piece_entities, batch.piece_slot, batch.piece_mask, batch.action_piece_ref
        )
        card_refs = _lookup_entities(
            card_entities, batch.card_slot, batch.card_mask, batch.action_card_ref
        )
        from_board = _lookup_squares(board_tokens, batch.action_from)
        to_board = _lookup_squares(board_tokens, batch.action_to)
        from_context = self.action_from_projection(
            torch.cat([self.square_embedding(batch.action_from), from_board], dim=-1)
        )
        to_context = self.action_to_projection(
            torch.cat([self.square_embedding(batch.action_to), to_board], dim=-1)
        )
        target_tokens = self.square_embedding(
            batch.action_target_sequence
        ) + self.target_position(batch.action_target_sequence_position)
        target_context = _masked_mean(
            target_tokens,
            batch.action_target_sequence_mask,
            dim=2,
        )
        action = self.action_norm(
            self.action_kind_embedding(batch.action_kind)
            + self.piece_kind_embedding(batch.action_piece_kind)
            + from_context
            + to_context
            + self.piece_kind_embedding(batch.action_promotion)
            + self.ability_embedding(batch.action_ability)
            + piece_refs
            + card_refs
            + self.text_embedding(batch.action_definition)
            + self.text_embedding(batch.action_handler)
            + self.text_embedding(batch.action_option)
            + self.target_kind_embedding(batch.action_target_kind)
            + target_context
            + self.action_numeric(batch.action_numeric)
        )
        return action * batch.action_mask.unsqueeze(-1)

    @staticmethod
    def _validate_batch(batch: V3Batch) -> None:
        if batch.spatial.ndim != 4 or tuple(batch.spatial.shape[1:]) != tuple(
            SPATIAL_SHAPE
        ):
            raise ValueError(f"spatial shape must be [batch, {SPATIAL_SHAPE}]")
        if (
            batch.global_features.ndim != 2
            or batch.global_features.shape[1] != GLOBAL_SHAPE[0]
        ):
            raise ValueError(f"global feature width must be {GLOBAL_SHAPE[0]}")
        if (
            batch.action_mask.ndim != 2
            or batch.action_mask.shape[0] != batch.batch_size
        ):
            raise ValueError("action mask must be [batch, legal_action_padding]")
        for field in fields(batch):
            tensor = getattr(batch, field.name)
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f"model input {field.name} contains NaN or infinity")


def compute_v3_loss(
    output: V3ModelOutput,
    action_mask: torch.Tensor,
    policy_targets: torch.Tensor,
    policy_supervision: torch.Tensor,
    value_targets: torch.Tensor,
    value_supervision: torch.Tensor,
    value_weight: float = 1.0,
) -> V3Loss:
    if (
        output.policy_logits.shape != action_mask.shape
        or policy_targets.shape != action_mask.shape
    ):
        raise ValueError(
            "policy logits, targets and action mask must have identical shapes"
        )
    if (
        output.value.shape != value_targets.shape
        or output.value.shape != value_supervision.shape
    ):
        raise ValueError(
            "value output, targets and supervision mask must have identical shapes"
        )
    if torch.any(policy_targets < 0) or not torch.isfinite(policy_targets).all():
        raise ValueError("policy targets must be finite and non-negative")
    if torch.any(policy_targets.masked_select(~action_mask) != 0):
        raise ValueError("policy target assigns mass to a padded or illegal action")
    supervised_policy_sums = policy_targets.sum(dim=1).masked_select(policy_supervision)
    if supervised_policy_sums.numel() and not torch.allclose(
        supervised_policy_sums,
        torch.ones_like(supervised_policy_sums),
        atol=1e-5,
    ):
        raise ValueError("each supervised policy target must sum to one")

    log_probabilities = F.log_softmax(output.policy_logits, dim=1)
    policy_per_sample = -(policy_targets * log_probabilities).sum(dim=1)
    policy_loss = _masked_loss_mean(policy_per_sample, policy_supervision)

    safe_value_targets = torch.where(
        value_supervision, value_targets, output.value.detach()
    )
    if not torch.isfinite(safe_value_targets.masked_select(value_supervision)).all():
        raise ValueError("supervised value targets must be finite")
    value_loss = _masked_loss_mean(
        (output.value - safe_value_targets).square(), value_supervision
    )
    return V3Loss(
        total=policy_loss + value_weight * value_loss,
        policy=policy_loss,
        value=value_loss,
        policy_samples=int(policy_supervision.sum().item()),
        value_samples=int(value_supervision.sum().item()),
    )


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    numerator = torch.sum(values * weights, dim=dim)
    denominator = torch.sum(weights, dim=dim).clamp_min(1.0)
    return numerator / denominator


def _masked_softmax(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    masked = values.masked_fill(~mask, torch.finfo(values.dtype).min)
    probabilities = F.softmax(masked, dim=dim) * mask.to(values.dtype)
    return probabilities / probabilities.sum(dim=dim, keepdim=True).clamp_min(
        torch.finfo(values.dtype).eps
    )


def _lookup_entities(
    entities: torch.Tensor,
    entity_slots: torch.Tensor,
    entity_mask: torch.Tensor,
    references: torch.Tensor,
) -> torch.Tensor:
    matches = references.unsqueeze(-1).eq(entity_slots.unsqueeze(1))
    matches = matches & entity_mask.unsqueeze(1) & references.unsqueeze(-1).ge(0)
    return torch.matmul(matches.to(entities.dtype), entities)


def _lookup_parent(
    runtime: torch.Tensor,
    parent_indices: torch.Tensor,
    runtime_mask: torch.Tensor,
) -> torch.Tensor:
    safe_indices = (
        parent_indices.clamp_min(0).unsqueeze(-1).expand(-1, -1, runtime.shape[-1])
    )
    parent = torch.gather(runtime, dim=1, index=safe_indices)
    parent_mask = parent_indices.ge(0) & runtime_mask
    return parent * parent_mask.unsqueeze(-1)


def _lookup_squares(
    board_tokens: torch.Tensor, square_indices: torch.Tensor
) -> torch.Tensor:
    safe_indices = (square_indices - 1).clamp_min(0)
    gathered = torch.gather(
        board_tokens,
        dim=1,
        index=safe_indices.unsqueeze(-1).expand(-1, -1, board_tokens.shape[-1]),
    )
    return gathered * square_indices.gt(0).unsqueeze(-1)


def _masked_loss_mean(losses: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = losses.masked_select(mask)
    if selected.numel() == 0:
        return losses.sum() * 0.0
    return selected.mean()
