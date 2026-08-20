# V2 AI baseline before the V3 observation migration

Recorded on 2026-08-21 before removing stale V2 experiments and generated artifacts.
This document is the reproducible comparison point for the ObservationV3 and
ActionDescriptorV3 migration. It is not a promotion claim.

The ordered migration plan and its completion gates are recorded in
[`ai-v3-roadmap.md`](ai-v3-roadmap.md).

## Source baseline

- Engine commit at the start of the audit: `6d030bd04f2ad53c259696907a03f34c71e9c0c8`
- The working tree was dirty and contained ongoing AI, web, and training changes.
- Ruleset: `augment_chess_20260807`
- Observation schema: `canonical_v2`
- Active model ID: `augment_net_v2_canonical_baseline`
- Preserved local artifact: `models/baseline/augment_net_v2_canonical_baseline.onnx`
- ONNX SHA-256: `53d3c81cb94f6ae277164d219f8bdae944f2a963fdf60cb09409e9ccf5a4933c`
- ONNX size: 277,100,854 bytes
- ONNX opset: 17
- The model is self-contained. No `.onnx.data` file is referenced by its initializers.

The root `augment_net.onnx` was byte-identical to the preserved baseline and was
removed as a duplicate. Rejected candidate models and PyTorch checkpoints were
also removed.

## Runtime configuration

The web bot used:

- `c_puct = 2.0`
- 150 simulations from the browser request
- root noise off
- maximum-visit action selection; the configured `temperature = 0.5` did not
  affect normal MCTS selection
- a new tree for every bot action
- batch-one ONNX leaf evaluation
- depth-four quiescence blended into every neural value

The retained V2 runtime source is in `crates/augment-ai`. The tracked
`crates/augment-cli/src/bin/self_play_gen2.rs` is retained only as a reference
for explicit root noise, temperature sampling, telemetry, and terminal metadata.

## Measured baseline

Local release measurements on the audited working tree:

- `GET /api/state`: 551 ms, 563 ms, and 557 ms
- early 150-simulation bot actions: approximately 0.97 to 2.13 seconds
- a later measured bot action: 4.26 seconds

The state endpoint recreated an ONNX session for each view. This behavior is
recorded for comparison and remains a later migration item; the cleanup only
consolidates the model path.

Quality evidence:

- T6 queen-sacrifice correct prior: approximately `4.2e-11`
- with 400 simulations and root noise off, only 1 of 38 root actions was visited
- sealed smoke calibration over 516 positions from 8 terminal games:
  - RMSE: `1.0859174728`
  - MAE: `0.8092734218`

The sealed smoke used quiescence depth one while the restored configuration used
depth four, so it is diagnostic evidence only and is not promotion-eligible.
The official depth-four holdout had zero completed games at cleanup time.

## Known structural defects

- 48 piece kinds are compressed into 16 per-side buckets; approximately 25
  distinct kinds share the catch-all bucket.
- Card identity uses a 256-bucket scalar hash. Of 192 cards, 106 share a bucket
  with another card.
- The neural observation references only about 13 of 147 public
  `CardRuntimeState` fields.
- The 32,768-entry fixed action space clamps overflowing multi-square actions to
  index 32,767. One sealed position contained 164 duplicate legal-action
  indices.
- Python and Rust used different promotion-index mappings.
- The canonical dataset used standard-chess legality and outcomes rather than
  Augmentia-native rules.
- Gen1 self-play labelled truncated games as draws. In the retained audit,
  2,080 of 2,950 rows came from games whose values were all zero.
- `dataset_selfplay_gen2_combined.jsonl` contained the same 2,950 parsed Gen1
  rows plus only 65 additional rows.

## Retention policy through the V3 migration

Keep until V3 passes the promotion gate:

- this document;
- the single baseline ONNX named above;
- the V2 AI runtime implementation in `crates/augment-ai`;
- the minimal V2 training architecture in `train/model.py`, `train/dataset.py`,
  `train/train.py`, and `train/export_onnx.py`;
- the tracked Gen2 self-play reference implementation.

Do not retain generated V2 datasets, rejected candidates, duplicate model
copies, checkpoints, virtual environments, target directories, invalid
holdouts, one-off diagnostic binaries, or local logs. They are not valid V3
training inputs and are not required to reproduce the comparison contract.

V2 source and the baseline model may be removed only after the V3 champion has
passed schema validation, tactical gates, sealed calibration, equal-simulation
arena, and equal-wall-clock arena.

## Git object cleanup

The `main` commit history did not contain the large model and dataset blobs.
They were retained by four internal Codex turn-diff checkpoint refs, alongside
failed loose-object temporary files. On 2026-08-21 those four artifact-bearing
checkpoint refs were removed, reflogs were expired, and unreachable objects
were pruned and packed.

- `main` remained at `6d030bd04f2ad53c259696907a03f34c71e9c0c8`.
- No commit IDs were rewritten.
- Remaining reachable blobs larger than 10 MiB: zero.
- `.git` size after cleanup: approximately 0.35 MiB.
- `git fsck --full` completed without errors.
