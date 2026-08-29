"""Deterministic, provenance-bound source mixtures for V3 training."""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .replay import DeterministicSamplerV3, ReplayDatasetV3, TrainingExampleV3
from .pretensor import _merge_training_batches
from .schema import SchemaError
from .tensorize import V3Vocabulary

MIXTURE_SCHEMA_ID = "augmentia.training-mixture.v3"
CONCAT_SCHEMA_ID = "augmentia.training-concat.v3"
SOURCE_NAMES = (
    "lichess",
    "solver",
    "weak_selfplay",
    "champion_selfplay",
)
MIXTURE_PROFILES: dict[str, dict[str, float]] = {
    "lichess_selfplay_bootstrap": {
        "lichess": 0.80,
        "solver": 0.0,
        "weak_selfplay": 0.20,
        "champion_selfplay": 0.0,
    },
    "lichess_solver_warmup": {
        "lichess": 0.77,
        "solver": 0.04,
        "weak_selfplay": 0.19,
        "champion_selfplay": 0.0,
    },
    "bootstrap": {
        "lichess": 0.50,
        "solver": 0.30,
        "weak_selfplay": 0.20,
        "champion_selfplay": 0.0,
    },
    "first_champion": {
        "lichess": 0.20,
        "solver": 0.20,
        "weak_selfplay": 0.10,
        "champion_selfplay": 0.50,
    },
    "mature": {
        "lichess": 0.10,
        "solver": 0.15,
        "weak_selfplay": 0.05,
        "champion_selfplay": 0.70,
    },
}


@dataclass(frozen=True)
class MixtureSourceIdentityV3:
    name: str
    weight: float
    quota: int
    dataset: dict[str, Any]


@dataclass(frozen=True)
class MixtureIdentityV3:
    dataset_id: str
    manifest_path: str
    manifest_sha256: str
    shard_sha256: tuple[tuple[str, str], ...]
    ruleset_id: str
    ruleset_manifest_sha256: str
    observation_schema_id: str
    action_schema_id: str
    profile: str
    epoch_size: int
    seed: int
    sources: tuple[MixtureSourceIdentityV3, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "identity_kind": "mixture_v3",
            "dataset_id": self.dataset_id,
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "shard_sha256": [list(item) for item in self.shard_sha256],
            "ruleset_id": self.ruleset_id,
            "ruleset_manifest_sha256": self.ruleset_manifest_sha256,
            "observation_schema_id": self.observation_schema_id,
            "action_schema_id": self.action_schema_id,
            "profile": self.profile,
            "epoch_size": self.epoch_size,
            "seed": self.seed,
            "sources": [
                {
                    "name": source.name,
                    "weight": source.weight,
                    "quota": source.quota,
                    "dataset": source.dataset,
                }
                for source in self.sources
            ],
        }


@dataclass(frozen=True)
class MixtureEntryV3:
    source_name: str
    dataset_index: int


@dataclass(frozen=True)
class ConcatIdentityV3:
    dataset_id: str
    manifest_path: str
    manifest_sha256: str
    shard_sha256: tuple[tuple[str, str], ...]
    ruleset_id: str
    ruleset_manifest_sha256: str
    observation_schema_id: str
    action_schema_id: str
    datasets: tuple[dict[str, Any], ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "identity_kind": "concat_v3",
            "dataset_id": self.dataset_id,
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "shard_sha256": [list(item) for item in self.shard_sha256],
            "ruleset_id": self.ruleset_id,
            "ruleset_manifest_sha256": self.ruleset_manifest_sha256,
            "observation_schema_id": self.observation_schema_id,
            "action_schema_id": self.action_schema_id,
            "datasets": list(self.datasets),
        }


class ReplayConcatV3(Sequence[TrainingExampleV3]):
    """A deterministic concatenation that preserves every source identity."""

    def __init__(
        self,
        datasets: Sequence[Any],
        *,
        config_path: str,
        config_sha256: str,
    ) -> None:
        if not datasets or any(len(dataset) == 0 for dataset in datasets):
            raise ValueError("concat datasets must be non-empty")
        rulesets = {
            (dataset.identity.ruleset_id, dataset.identity.ruleset_manifest_sha256)
            for dataset in datasets
        }
        schemas = {
            (dataset.identity.observation_schema_id, dataset.identity.action_schema_id)
            for dataset in datasets
        }
        if len(rulesets) != 1 or len(schemas) != 1:
            raise SchemaError("concat datasets must use one ruleset and schema pair")
        identities = tuple(dataset.identity.to_json() for dataset in datasets)
        canonical = json.dumps(
            identities, sort_keys=True, separators=(",", ":")
        ).encode()
        identity_hash = hashlib.sha256(canonical).hexdigest()
        shard_hashes = tuple(
            (f"dataset-{source_index}:{path}", digest)
            for source_index, dataset in enumerate(datasets)
            for path, digest in (
                getattr(dataset.identity, "shard_sha256", ())
                or getattr(dataset, "shard_sha256", ())
            )
        )
        ruleset_id, ruleset_hash = next(iter(rulesets))
        observation_schema_id, action_schema_id = next(iter(schemas))
        ends = []
        total = 0
        sampler_groups = []
        for dataset in datasets:
            total += len(dataset)
            ends.append(total)
            sampler_groups.extend(dataset.sampler_group_sizes)
        self.datasets = tuple(datasets)
        splits = {getattr(dataset, "split", None) for dataset in datasets}
        self.split = next(iter(splits)) if len(splits) == 1 else None
        self._ends = tuple(ends)
        group_offsets = []
        group_total = 0
        for dataset in datasets:
            group_offsets.append(group_total)
            group_total += len(dataset.sampler_group_sizes)
        self._sampler_group_offsets = tuple(group_offsets)
        self._sampler_group_sizes = tuple(sampler_groups)
        self.identity = ConcatIdentityV3(
            dataset_id=f"concat-v3:{identity_hash}",
            manifest_path=config_path,
            manifest_sha256=config_sha256,
            shard_sha256=shard_hashes,
            ruleset_id=ruleset_id,
            ruleset_manifest_sha256=ruleset_hash,
            observation_schema_id=observation_schema_id,
            action_schema_id=action_schema_id,
            datasets=identities,
        )

    def __len__(self) -> int:
        return self._ends[-1]

    def _source_index(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        source_index = bisect_right(self._ends, index)
        offset = 0 if source_index == 0 else self._ends[source_index - 1]
        return source_index, index - offset

    def __getitem__(self, index: int) -> TrainingExampleV3:
        source_index, local_index = self._source_index(index)
        example = self.datasets[source_index][local_index]
        if hasattr(example, "_example"):
            batch, row = example._example()
            return TrainingExampleV3(
                decision_id=f"dataset-{source_index}:{batch.decision_ids[row]}",
                observation=None,
                legal_actions=None,
                policy=None,
                value=None,
                forced_win_distance=None,
            )
        return TrainingExampleV3(
            decision_id=f"dataset-{source_index}:{example.decision_id}",
            observation=example.observation,
            legal_actions=example.legal_actions,
            policy=example.policy,
            value=example.value,
            forced_win_distance=example.forced_win_distance,
        )

    def shard_index(self, index: int) -> int:
        source_index, local_index = self._source_index(index)
        return self._sampler_group_offsets[source_index] + self.datasets[
            source_index
        ].shard_index(local_index)

    def collate_indices(self, indices: Sequence[int]) -> Any:
        if not indices:
            raise ValueError("cannot collate an empty batch")
        grouped: dict[int, list[tuple[int, int]]] = {}
        for output_index, index in enumerate(indices):
            source_index, local_index = self._source_index(index)
            grouped.setdefault(source_index, []).append((output_index, local_index))
        pieces = []
        for source_index, rows in grouped.items():
            dataset = self.datasets[source_index]
            local_indices = [row[1] for row in rows]
            if hasattr(dataset, "collate_indices"):
                batch = dataset.collate_indices(local_indices)
                batch.decision_ids = tuple(
                    f"dataset-{source_index}:{identifier}"
                    for identifier in batch.decision_ids
                )
            else:
                from .replay import collate_examples

                examples = []
                for index in local_indices:
                    example = dataset[index]
                    examples.append(
                        TrainingExampleV3(
                            decision_id=f"dataset-{source_index}:{example.decision_id}",
                            observation=example.observation,
                            legal_actions=example.legal_actions,
                            policy=example.policy,
                            value=example.value,
                            forced_win_distance=example.forced_win_distance,
                        )
                    )
                batch = collate_examples(examples, self.build_vocabulary())
            pieces.append((rows, batch))
        ordered = sorted(pieces, key=lambda piece: piece[0][0][0])
        merged = _merge_training_batches([piece[1] for piece in ordered])
        current_order = [position for piece in ordered for position, _ in piece[0]]
        restore = [0] * len(current_order)
        for slot, position in enumerate(current_order):
            restore[position] = slot
        from .pretensor import _select_training_batch

        import torch

        return _select_training_batch(merged, torch.tensor(restore, dtype=torch.long))

    def action_count(self, index: int) -> int:
        source_index, local_index = self._source_index(index)
        dataset = self.datasets[source_index]
        if hasattr(dataset, "action_count"):
            return dataset.action_count(local_index)
        return len(dataset[local_index].legal_actions["actions"])

    @property
    def sampler_group_sizes(self) -> tuple[int, ...]:
        return self._sampler_group_sizes

    @property
    def vocabulary_tokens(self) -> frozenset[str]:
        return frozenset(
            token
            for dataset in self.datasets
            for token in dataset.vocabulary_tokens
        )

    def build_vocabulary(self) -> V3Vocabulary:
        return V3Vocabulary.from_tokens(self.vocabulary_tokens)

    def clear_cache(self) -> None:
        for dataset in self.datasets:
            dataset.clear_cache()

    @classmethod
    def from_config(cls, path: str | Path) -> "ReplayConcatV3":
        config_path = Path(path).resolve()
        raw = config_path.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, dict) or set(payload) != {
            "concat_schema_id",
            "datasets",
        }:
            raise SchemaError("concat config fields do not match the V3 contract")
        if payload["concat_schema_id"] != CONCAT_SCHEMA_ID:
            raise SchemaError("unsupported concat config schema")
        paths = payload["datasets"]
        if not isinstance(paths, list) or not paths:
            raise SchemaError("concat config must list datasets")
        datasets = [
            load_training_dataset(_resolve_config_path(config_path, item))
            for item in paths
        ]
        return cls(
            datasets,
            config_path=str(config_path),
            config_sha256=hashlib.sha256(raw).hexdigest(),
        )

    @classmethod
    def from_identity(cls, payload: Mapping[str, Any]) -> "ReplayConcatV3":
        if payload.get("identity_kind") != "concat_v3":
            raise SchemaError("checkpoint does not contain a concat identity")
        datasets = [load_dataset_from_identity(item) for item in payload["datasets"]]
        concat = cls(
            datasets,
            config_path=str(payload["manifest_path"]),
            config_sha256=str(payload["manifest_sha256"]),
        )
        if concat.identity.to_json() != dict(payload):
            raise RuntimeError("checkpoint concat provenance no longer matches sources")
        return concat


class ReplayMixtureV3(Sequence[TrainingExampleV3]):
    """One deterministic virtual epoch backed by source/index references."""

    def __init__(
        self,
        datasets: Mapping[str, ReplayDatasetV3],
        weights: Mapping[str, float],
        *,
        epoch_size: int,
        seed: int,
        profile: str,
        config_path: str = "<embedded>",
        config_sha256: str | None = None,
    ) -> None:
        _validate_weights(datasets, weights)
        active = {name: weight for name, weight in weights.items() if weight > 0}
        if epoch_size < len(active):
            raise ValueError("mixture epoch_size must cover every active source")
        rulesets = {
            (item.identity.ruleset_id, item.identity.ruleset_manifest_sha256)
            for item in datasets.values()
        }
        schemas = {
            (item.identity.observation_schema_id, item.identity.action_schema_id)
            for item in datasets.values()
        }
        if len(rulesets) != 1 or len(schemas) != 1:
            raise SchemaError("mixture sources must use one ruleset and schema pair")
        quotas = _allocate_quotas(active, epoch_size)
        source_identities = []
        entries: list[MixtureEntryV3] = []
        entry_groups: list[tuple[str, int]] = []
        shard_hashes = []
        for source_index, name in enumerate(sorted(active)):
            dataset = datasets[name]
            quota = quotas[name]
            sampler = DeterministicSamplerV3(
                len(dataset),
                _source_seed(seed, name, source_index),
                dataset.sampler_group_sizes,
            )
            indices: list[int] = []
            while len(indices) < quota:
                indices.extend(sampler.take(quota - len(indices)))
                sampler.advance_epoch_if_complete()
            for index in indices:
                entries.append(MixtureEntryV3(name, index))
                entry_groups.append((name, dataset.shard_index(index)))
            source_json = dataset.identity.to_json()
            source_identities.append(
                MixtureSourceIdentityV3(name, active[name], quota, source_json)
            )
            identity = dataset.identity
            shard_hashes.extend(
                (f"{name}:{path}", digest)
                for path, digest in getattr(identity, "shard_sha256", ())
                or getattr(dataset, "shard_sha256", ())
            )
        canonical = {
            "profile": profile,
            "epoch_size": epoch_size,
            "seed": seed,
            "sources": [
                {
                    "name": source.name,
                    "weight": source.weight,
                    "quota": source.quota,
                    "dataset": source.dataset,
                }
                for source in source_identities
            ],
        }
        identity_hash = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        ruleset_id, ruleset_hash = next(iter(rulesets))
        observation_schema_id, action_schema_id = next(iter(schemas))
        self.datasets = dict(datasets)
        self.entries = tuple(entries)
        self._sampler_group_sizes = _run_lengths(entry_groups)
        self.quotas = quotas
        self.weights = dict(active)
        self.identity = MixtureIdentityV3(
            dataset_id=f"mixture-v3:{identity_hash}",
            manifest_path=config_path,
            manifest_sha256=config_sha256 or identity_hash,
            shard_sha256=tuple(shard_hashes),
            ruleset_id=ruleset_id,
            ruleset_manifest_sha256=ruleset_hash,
            observation_schema_id=observation_schema_id,
            action_schema_id=action_schema_id,
            profile=profile,
            epoch_size=epoch_size,
            seed=seed,
            sources=tuple(source_identities),
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> TrainingExampleV3:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        entry = self.entries[index]
        dataset = self.datasets[entry.source_name]
        example = dataset[entry.dataset_index]
        if hasattr(example, "_example"):
            batch, local_index = example._example()
            return TrainingExampleV3(
                decision_id=f"{entry.source_name}:{batch.decision_ids[local_index]}",
                observation=None,
                legal_actions=None,
                policy=None,
                value=None,
                forced_win_distance=None,
            )
        if not hasattr(example, "decision_id"):
            dataset = self.datasets[entry.source_name]
            raise TypeError(
                f"source {entry.source_name} does not expose individual examples"
            )
        return TrainingExampleV3(
            decision_id=f"{entry.source_name}:{example.decision_id}",
            observation=example.observation,
            legal_actions=example.legal_actions,
            policy=example.policy,
            value=example.value,
            forced_win_distance=example.forced_win_distance,
        )

    def collate_indices(self, indices) -> Any:
        """Collate across sources without materializing raw examples."""
        if not indices:
            raise ValueError("cannot collate an empty batch")
        grouped: dict[str, list[tuple[int, int]]] = {}
        for output_index, index in enumerate(indices):
            if index < 0:
                index += len(self)
            entry = self.entries[index]
            grouped.setdefault(entry.source_name, []).append(
                (output_index, entry.dataset_index)
            )
        pieces = []
        for source_name, rows in grouped.items():
            dataset = self.datasets[source_name]
            local_indices = [row[1] for row in rows]
            if hasattr(dataset, "collate_indices"):
                batch = dataset.collate_indices(local_indices)
                batch.decision_ids = tuple(
                    f"{source_name}:{identifier}"
                    for identifier in batch.decision_ids
                )
            else:
                from .replay import collate_examples

                examples = []
                for local_index in local_indices:
                    example = dataset[local_index]
                    examples.append(
                        TrainingExampleV3(
                            decision_id=f"{source_name}:{example.decision_id}",
                            observation=example.observation,
                            legal_actions=example.legal_actions,
                            policy=example.policy,
                            value=example.value,
                            forced_win_distance=example.forced_win_distance,
                        )
                    )
                batch = collate_examples(examples, self.build_vocabulary())
            pieces.append((rows, batch))
        ordered = sorted(pieces, key=lambda piece: piece[0][0][0])
        merged = _merge_training_batches([piece[1] for piece in ordered])
        current_order = [
            position for piece in ordered for position, _ in piece[0]
        ]
        restore = [0] * len(current_order)
        for slot, position in enumerate(current_order):
            restore[position] = slot
        from .pretensor import _select_training_batch

        import torch

        return _select_training_batch(
            merged, torch.tensor(restore, dtype=torch.long)
        )

    @property
    def sampler_group_sizes(self) -> tuple[int, ...]:
        return self._sampler_group_sizes

    @property
    def vocabulary_tokens(self) -> frozenset[str]:
        return frozenset(
            token
            for dataset in self.datasets.values()
            for token in dataset.vocabulary_tokens
        )

    def build_vocabulary(self) -> V3Vocabulary:
        return V3Vocabulary.from_tokens(self.vocabulary_tokens)

    def clear_cache(self) -> None:
        for dataset in self.datasets.values():
            dataset.clear_cache()

    def action_count(self, index: int) -> int:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        entry = self.entries[index]
        dataset = self.datasets[entry.source_name]
        if hasattr(dataset, "action_count"):
            return dataset.action_count(entry.dataset_index)
        example = dataset[entry.dataset_index]
        return len(example.legal_actions["actions"])

    @classmethod
    def from_config(cls, path: str | Path) -> ReplayMixtureV3:
        config_path = Path(path).resolve()
        raw = config_path.read_bytes()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise SchemaError(f"invalid mixture JSON: {error}") from error
        required = {
            "mixture_schema_id",
            "profile",
            "seed",
            "epoch_size",
            "sources",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise SchemaError("mixture config fields do not match the V3 contract")
        if payload["mixture_schema_id"] != MIXTURE_SCHEMA_ID:
            raise SchemaError("unsupported mixture config schema")
        profile = payload["profile"]
        if profile not in MIXTURE_PROFILES:
            raise SchemaError(f"unknown mixture profile {profile!r}")
        source_paths = payload["sources"]
        if not isinstance(source_paths, dict) or any(
            name not in SOURCE_NAMES for name in source_paths
        ):
            raise SchemaError("mixture sources contain an unknown source name")
        weights = MIXTURE_PROFILES[profile]
        missing = [
            name
            for name, weight in weights.items()
            if weight > 0 and name not in source_paths
        ]
        if missing:
            raise SchemaError(
                f"mixture profile {profile} is missing sources: {missing}"
            )
        datasets = {
            name: load_training_dataset(_resolve_config_path(config_path, value))
            for name, value in source_paths.items()
            if weights.get(name, 0) > 0
        }
        return cls(
            datasets,
            weights,
            epoch_size=_positive_int(payload["epoch_size"], "epoch_size"),
            seed=_integer(payload["seed"], "seed"),
            profile=profile,
            config_path=str(config_path),
            config_sha256=hashlib.sha256(raw).hexdigest(),
        )

    @classmethod
    def from_identity(cls, payload: Mapping[str, Any]) -> ReplayMixtureV3:
        if payload.get("identity_kind") != "mixture_v3":
            raise SchemaError("checkpoint does not contain a V3 mixture identity")
        sources = payload.get("sources")
        if not isinstance(sources, list) or not sources:
            raise SchemaError("checkpoint mixture has no sources")
        datasets: dict[str, ReplayDatasetV3] = {}
        weights: dict[str, float] = {}
        for source in sources:
            dataset_identity = source["dataset"]
            dataset = load_dataset_from_identity(dataset_identity)
            datasets[source["name"]] = dataset
            weights[source["name"]] = float(source["weight"])
        mixture = cls(
            datasets,
            weights,
            epoch_size=_positive_int(payload["epoch_size"], "epoch_size"),
            seed=_integer(payload["seed"], "seed"),
            profile=str(payload["profile"]),
            config_path=str(payload["manifest_path"]),
            config_sha256=str(payload["manifest_sha256"]),
        )
        if mixture.identity.to_json() != dict(payload):
            raise RuntimeError(
                "checkpoint mixture provenance no longer matches its sources"
            )
        return mixture


def load_training_dataset(
    path: str | Path,
) -> ReplayDatasetV3 | ReplayMixtureV3 | ReplayConcatV3:
    candidate = Path(path).resolve()
    try:
        payload = json.loads(candidate.read_bytes())
    except json.JSONDecodeError as error:
        raise SchemaError(f"invalid dataset JSON: {error}") from error
    if (
        isinstance(payload, dict)
        and payload.get("mixture_schema_id") == MIXTURE_SCHEMA_ID
    ):
        return ReplayMixtureV3.from_config(candidate)
    if (
        isinstance(payload, dict)
        and payload.get("concat_schema_id") == CONCAT_SCHEMA_ID
    ):
        return ReplayConcatV3.from_config(candidate)
    if (
        isinstance(payload, dict)
        and payload.get("artifact_schema_id") == "augmentia.pretensor-manifest.v3.1"
    ):
        from .pretensor import PretensorizedReplayDatasetV3

        return PretensorizedReplayDatasetV3(candidate)
    return ReplayDatasetV3(candidate)


def load_dataset_from_identity(
    payload: Mapping[str, Any],
) -> ReplayDatasetV3 | ReplayMixtureV3 | ReplayConcatV3:
    if payload.get("identity_kind") == "pretensorized_v3":
        from .pretensor import PretensorizedReplayDatasetV3

        dataset = PretensorizedReplayDatasetV3(payload["artifact_manifest_path"])
        if dataset.identity.to_json() != dict(payload):
            raise RuntimeError(
                "checkpoint pre-tensor provenance no longer matches its artifact"
            )
        return dataset
    if payload.get("identity_kind") == "mixture_v3":
        return ReplayMixtureV3.from_identity(payload)
    if payload.get("identity_kind") == "concat_v3":
        return ReplayConcatV3.from_identity(payload)
    dataset = ReplayDatasetV3(payload["manifest_path"])
    if dataset.identity.to_json() != dict(payload):
        raise RuntimeError(
            "checkpoint dataset provenance no longer matches the manifest"
        )
    return dataset


def _validate_weights(
    datasets: Mapping[str, ReplayDatasetV3], weights: Mapping[str, float]
) -> None:
    unknown = set(weights) - set(SOURCE_NAMES)
    if unknown:
        raise ValueError(f"unknown mixture sources: {sorted(unknown)}")
    active = {name: value for name, value in weights.items() if value > 0}
    if not active or set(active) != set(datasets):
        raise ValueError("datasets must exactly match positive mixture weights")
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
        for value in weights.values()
    ):
        raise ValueError("mixture weights must be finite and non-negative")
    if abs(sum(active.values()) - 1.0) > 1e-9:
        raise ValueError("positive mixture weights must sum to one")


def _allocate_quotas(weights: Mapping[str, float], epoch_size: int) -> dict[str, int]:
    names = tuple(sorted(weights))
    ideal = {name: epoch_size * weights[name] for name in names}
    quotas = {name: max(1, math.floor(ideal[name])) for name in names}
    if sum(quotas.values()) > epoch_size:
        raise ValueError(
            "mixture epoch_size cannot provide one example for every active source"
        )
    unallocated = epoch_size - sum(quotas.values())
    order = sorted(names, key=lambda name: (-(ideal[name] % 1), name))
    for index in range(unallocated):
        quotas[order[index % len(order)]] += 1
    return quotas


def _source_seed(seed: int, name: str, index: int) -> int:
    digest = hashlib.sha256(f"{seed}\0{name}\0{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def _run_lengths(values: Sequence[tuple[str, int]]) -> tuple[int, ...]:
    if not values:
        return ()
    lengths = []
    previous = values[0]
    length = 0
    for value in values:
        if value != previous:
            lengths.append(length)
            previous = value
            length = 0
        length += 1
    lengths.append(length)
    return tuple(lengths)


def _resolve_config_path(config: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise SchemaError("mixture source paths must be non-empty strings")
    candidate = Path(value)
    return candidate if candidate.is_absolute() else config.parent / candidate


def _integer(value: Any, name: str) -> int:
    if not isinstance(value, int):
        raise SchemaError(f"mixture {name} must be an integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    result = _integer(value, name)
    if result <= 0:
        raise SchemaError(f"mixture {name} must be positive")
    return result
