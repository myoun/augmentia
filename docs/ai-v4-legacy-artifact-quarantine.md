# V4 legacy target artifact quarantine

Phase 1 preserves the pre-Phase-0 V4 artifacts without moving, deleting, copying, or
rewriting their bytes. The machine-readable inventory is
[`ai-v4-legacy-quarantine-registry.json`](ai-v4-legacy-quarantine-registry.json).

## Classification

Every registered artifact is classified as:

```text
legacy/non-promotable/contaminated-target-semantics
```

The legacy lineage is `augmentia.replay.game.v4` plus
`augmentia.factorized-mcts-visits.v2`, which has no `opened_action_mask` or prefix
`opened_masks`. These records remain valid historical evidence, but the old
all-open trainer interpretation must not be used as opened-only evidence.

The registry is sealed with inventory digest:

```text
43a968058371d2cc634598eadfe81abe3c1676993ecd958c7f1eb925a07feefe
```

The sibling artifact root also contains the idempotent marker
`../augmentia-v4-artifacts/.augmentia-v4-legacy-quarantine.json`, which points back
to the registry and its digest.

## Inventory and preservation

The scan covers 645 registered file identities: 8 legacy corpus manifests and
304 native games, 1 dataset index, 27 training runs with 74 checkpoint/optimizer
pairs plus their run/metrics/history metadata, and 7 exported bundles. Original
artifact bytes remain in place. The registry records paths, sizes,
declared/content SHA-256 values, Merkle roots, lineage, and `promotable=false`.

Legacy artifacts may be used only for historical analysis, provenance auditing, or
the explicitly isolated all-open control arm. They must not be used for promotion,
web-default selection, self-play producer selection, opened-only training, or
checkpoint resume. New training data must be regenerated with the v5/v3 producer;
opening metadata must never be guessed from legacy records.

## Verification

From `train/`:

```powershell
uv run python -m v4.quarantine_legacy build `
  --artifact-root ../../augmentia-v4-artifacts `
  --registry ../docs/ai-v4-legacy-quarantine-registry.json `
  --root-marker --hash-games --hash-binaries

uv run python -m v4.quarantine_legacy verify `
  ../docs/ai-v4-legacy-quarantine-registry.json `
  --artifact-root ../../augmentia-v4-artifacts `
  --hash-games --hash-binaries

uv run python -m unittest v4.test_quarantine_legacy -v
```

The build is idempotent and refuses to overwrite a different registry or marker.
Verification rehashes declared binary hashes and all registered game content, and
fails closed on path traversal, unknown lineage, missing files, or drift.
