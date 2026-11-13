# Arena stall: budget bypass in root tactical screening

## Scope and evidence

The material-versus-positional arena stopped producing finished-game records after three games of the seed-8 batch. The process had consumed more than ten CPU-hours. That run did not record per-action positions or call boundaries, and its process is now gone. Its exact stopped instruction and board cannot be recovered from the surviving three summaries.

The fourth game uses seed `1000012`: base seed `8` + pairing index `1` + black-color offset `1000003`. Material plays Black, positional plays White.

A new bounded run of that seed, using the old search code and the original limits (1,000 ms, 8,000 nodes, depth 4 turns, four chance samples), reached action index 50 and exceeded a five-second no-progress watchdog inside `search`. The preceding **actions**, not another run of the time-limited search, were saved in `crates/augment-search/fixtures/arena-stall-seed1000012.json`.

This is a captured reproduction of the failure mechanism, not a claim that action 50 was the exact location of the original ten-hour run. Time-limited search can select different actions across machines and runs. Replaying the saved actions pins the position.

## Measured cause

In the captured position White has 22 root candidates. After the first candidate Black has **211,930** replies, predominantly Hypocrisy's four-square combinations. On this machine in the release build:

- Root generation: 22 actions, 0.029 ms.
- First root transition: 0.116 ms.
- Reply generation: 211,930 actions, 13.634 ms.
- First 100 reply transitions: 2.577 ms. This is a sample, not a runtime estimate for all replies.

The old safety pass applied root candidates and opponent replies without checking time, cancellation or node limits. It also omitted those transitions from `nodes`, `apply_calls` and `apply_ms`. The main search only checked its budget **after** that pass finished.

A second, independent complexity defect regenerated the entire root legal-action batch inside `is_two_ply_blunder` for every candidate. Combination cards could therefore multiply a large root enumeration by its own candidate count.

The zero-time and pre-cancelled regression tests expose the bypass without a long game: each performed **31** legal enumerations where only the one needed for a legal fallback should occur. Both tests failed in 0.03 seconds before the fix.

The current positional evaluator does not enumerate legal moves. Blaming its evaluation cost for the stall was unsupported. Engine transition cost and leaf evaluation cost are different things.

## Repair

- All applied safety and tree transitions share the same time, cancellation and node budget. `nodes == apply_calls` includes the safety pass. Safety cost is separately visible as `safety_nodes` and `safety_ms`.
- The immutable root batch is reused. Direct root outcomes are inspected first, then opponent replies. A large reply space no longer prevents checking a later direct win while budget remains.
- Interrupted screening sets `safety_completed=false`. Unexamined or sampled outcomes remain unknown, not safe proofs. On exhaustion the result keeps a verified non-losing fallback if available, otherwise an unrefuted legal move. It does not claim a completed iteration.
- A same-player continuation is not an opponent reply. Only an actual deterministic win by the player replying proves that candidate lost; an opponent action that loses to the root player is not an opponent win.
- A sampled transition is never promoted to an immediate-win proof or used to prune a root candidate as certainly lost.
- A direct terminal win uses `stop_reason=immediate_win`, score `100000`, `completed_depth=0`, and `scored_root_actions=0`. It is an exact terminal result, not a completed full-root iteration.
- When every candidate is proven lost, a legal move is still returned, rather than an `EmptyOngoingState` error.

The deadline remains **cooperative**. One synchronous engine enumeration or transition cannot be preempted. Checks around those calls prevent launching an unbounded succession of additional calls after the budget expires, but do not promise a hard wall-clock cutoff inside a single engine call. The replay example uses a separate five-second process watchdog for this reason.

## Fixed-position measurements after repair

Command:

```powershell
cargo run --release -p augment-search --example replay-search
cargo run --release -p augment-search --example replay-search -- crates/augment-search/fixtures/arena-stall-seed1000012.json material
```

Same captured position, 1,000 ms / 8,000 nodes / depth 4 turns:

- Positional: **137.510 ms**, exactly **8,000** applied transitions, two legal enumerations, `node_limit`.
- Material: **141.493 ms**, exactly **8,000** applied transitions, two legal enumerations, `node_limit`.
- Both report `safety_completed=false`, `completed_depth=0`, `scored_root_actions=0`.
- The saved 50-action prefix replays in 6.353 / 5.960 ms respectively.

The safety pass consumed the budget at this very wide position. These results prove bounded return and honest accounting, **not** good playing strength or a completed depth-4 search. No claim that positional is faster or stronger follows from these two timings.

The original analyzer blunder is independently pinned by `fixtures/web-selfmate-seed20260807.json`: replay 31 actions without search, then check that depth-one screening rejects the three moves permitting White's immediate king capture. Testing only whether a new self-play game lasts longer would not establish that the original defect was fixed.

## Persistent arena diagnostics

Use different, new paths for every run:

```powershell
cargo run --release -p augment-search -- arena --evaluation material --against positional --games 1 --max-actions 4 --time-ms 100 --nodes 1000 --depth 4 --seed 1000012 --trace artifacts/arena-diagnostic.trace.jsonl --out artifacts/arena-diagnostic.games.jsonl
```

`--trace` records versioned run metadata, each actual game seed and color, and `new_game`, `search`, and `apply` start/end boundaries. It includes readable board/state snapshots and decision timing/stats. A complete start record without its end identifies the active call after interruption. Structured actions plus the actual seed permit replaying the completed prefix without re-searching it; the full Rust Debug snapshot itself is diagnostic, not deserializable.

`--out` preserves the legacy one-GameRecord-per-line output and also works for self-play and policy comparisons. `AUGMENTIA_ARENA_OUT` remains the fallback if `--out` is absent. Both outputs refuse to overwrite existing files. Write and flush errors stop the run rather than silently losing data. Successful events are flushed before the next engine call. This protects against process interruption, not power loss (`fsync` is not promised); a forcibly interrupted final partial line must be ignored.

Traces include private engine/RNG state for local diagnosis. Do not mistake this for public-state teacher input or publish it as such.

## Corrections to earlier interpretation

`Winner::First` means **material** in these runs, and `Winner::Second` means **positional**, regardless of color. The first batch contains material 2 wins, positional 1 win, and 2 truncated games. Earlier narration reversed those wins.

Changing the base seed from 7 to 8 did **not** create an independent batch. The scheduler increments the base for each pairing:

- First batch's five completed game seeds: `7`, `1000010`, `8`, `1000011`, `9`.
- Second batch's three completed game seeds: `8`, `1000011`, `9`.

All three second-batch records overlap the first batch. They must not be counted as three new independent observations. Color exchange currently also uses different seeds, not a matched-seed pair. Trace metadata now records these limitations explicitly. No material-versus-positional strength conclusion is supported by these records.

Equal-time matches are valid for practical strength at a fixed time budget. Equal-node matches answer a different question and do not remove all confounding effects. Neither comparison was reliable while the safety pass ignored and undercounted its budget.

## Verification

The complete `scripts/verify-search.ps1` ran successfully, **without** `-SkipBenchmarks`. This includes:

- Workspace formatting and strict Clippy (`--workspace --all-targets -- -D warnings`).
- Workspace tests: 25 app, 641 core, one card-contract, four card-definition, 58 search/unit, seven real-CLI, four arena-observer, and one compile-fail doctest. All passed.
- All 14 offline web/source checks, tactical fixtures and the source-based cannon oracle.
- Four serial benchmark smoke runs: 250 ms for seeds 1–3, and 1,000 ms for seed 1. Each output contains eight measured scenario reports; the script passes `--decisions 24` for preparing the scenario positions, not 24 measured reports.
- Three serial short-play smoke runs: 250 ms × 32 decisions for seeds 1–3, plus a final-state summary each. All three end at the requested cap and are explicitly truncated, not counted as decided games.
- Across these outputs: no recorded errors; largest reported single-decision overrun 1.0392 ms. These timings are single-machine smoke observations, not a hard deadline guarantee.

Artifacts are in `artifacts/stall-fix-verification/`. These are runtime/regression smoke checks, not a strength arena. The contract fixture passed without being changed.

The trace tests also verify write/flush error propagation, refusing existing paths, all three arena variants, visibility before the next engine call, and exact seed/action replay fingerprints. A trace cut after `apply_start` is explicitly treated as completed-prefix-plus-pending-action, not as an already applied move.
