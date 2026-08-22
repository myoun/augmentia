# V3 Rust/Python contract

Rust owns game semantics and emits the checked-in fixture at
`fixtures/observation_action_v3.json`. Python validates and consumes that record; it does not
reimplement card rules, legality, perspective transforms, or action indexing.

Regenerate the fixture from the repository root:

```powershell
cargo run -p augment-cli --bin export_v3_fixtures
```

Run Python parity tests from `train`:

```powershell
uv run python -m unittest discover -s v3 -p "test_*.py"
```

The Rust workspace test compares the checked-in bytes with a freshly generated suite. Any change
to ObservationV3, ActionDescriptorV3, representative states, or serialization therefore requires
an explicit fixture regeneration and Python validation.

Full-game and independent-decision training data use the Rust-authored `GameRecordV3` and
`DecisionRecordV3` contracts documented in `docs/data-v3-format.md`. Compressed replay shards are
external artifacts and are not checked into Git.

The checked-in `bootstrap_standard_v1.pgn` and `bootstrap_tactics_v1.json` inputs exercise the
stage-7 board-pretraining importer and exact Augmentia forced-win solver. They are regression
fixtures, not a training corpus. See `docs/bootstrap-v3-data.md` before importing larger sources.

`model.py` contains the stage-8 hybrid V3 network. `tensorize.py` converts validated Rust-authored
records into padded tensors while preserving exact vocabularies, runtime parent relations, local
piece/card references, and ordered multi-square targets. Run its tests and benchmark from `train`:

```powershell
uv run python -m unittest v3.test_model -v
uv run python -m v3.benchmark_model --case runtime_rich --device cpu --threads 1
```

`replay.py`, `trainer.py`, and `onnx_export.py` implement the stage-9 verified loader, exact-step
resume, auxiliary supervision, and provenance-bound ONNX artifacts. `onnx_runtime.py` and Rust
`augment_ai::runtime_v3` implement the stage-10 reusable batched runtime. See
`docs/ai-v3-training.md` and `docs/ai-v3-runtime.md`. `augment_ai::mcts_v3` implements the
stage-11 local-action batch MCTS, documented in `docs/ai-v3-search.md`. Include the Rust search
benchmark with:

```powershell
uv run python -m v3.benchmark_runtime --iterations 20 --soak 1000 --rust `
  --search-simulations 100 --search-iterations 5
```

V2 weights and fallback are never loaded.

Large JSON corpora use a content-addressed verification sidecar. Cold semantic verification uses
all logical CPUs for large manifests; warm startup verifies only compressed byte counts and
SHA-256. Generate the separate training-time binary artifact with:

```powershell
uv run python -m v3.pretensor path\to\manifest.json ..\replay-buffers\v3-tensorized `
  --chunk-size 64 --workers 8
```

Pass the generated pre-tensor `manifest.json` to `v3.trainer`. It preserves source, vocabulary,
tensorizer-version and checksum provenance while removing JSON parsing and tensorization from the
training hot path.

Measure the artifact independently with:

```powershell
uv run python -m v3.benchmark_pretensor path\to\tensorized\manifest.json `
  --batch-size 64 --batches 20
```

On the measured RTX 3060 host, use batch 1024 with one pinned-memory background prefetch. Batch
1536 did not improve throughput and left too little VRAM headroom. `--steps` remains the complete
run target; `--max-steps-this-run` pauses one invocation without changing checkpoint identity:

```powershell
uv run python -m v3.trainer path\to\tensorized\manifest.json ..\runs\v3-pretrain `
  --batch-size 1024 --steps 100000 --max-steps-this-run 10000 `
  --checkpoint-interval 500 --prefetch-batches 1 --log-interval 50

uv run python -m v3.trainer path\to\tensorized\manifest.json ..\runs\v3-pretrain `
  --batch-size 1024 --steps 100000 --max-steps-this-run 10000 `
  --checkpoint-interval 500 --prefetch-batches 1 --log-interval 50 --resume
```

Training metrics are streamed instead of retained for the entire run. A normal limited invocation,
SIGINT, or SIGTERM writes an atomic exact-step checkpoint; prefetched-but-unconsumed indices are not
committed to sampler state.

`self_play_v3.rs`, `mixture.py`, and `iteration.py` implement the stage-12 reproducible loop.
An iteration generates engine-replayable MCTS games, applies a fixed source-mixture profile,
resumes exact-step training, exports a candidate, records complete lineage, and enforces bounded
artifact retention. See `docs/ai-v3-self-play.md`; the iteration command is:

```powershell
uv run python -m v3.iteration path\to\iteration.json ..\runs\v3-iterations
```

`promotion.py` and Rust `arena_v3` implement the stage-13 fail-closed evaluation gate. They check
sealed tactics, value calibration, complete implemented-ruleset coverage, paired colors/seeds,
equal simulations, equal wall-clock time, and the preserved V2 baseline without modifying a
champion registry. See `docs/ai-v3-promotion.md` and run:

```powershell
uv run python -m v3.promotion path\to\promotion.json path\to\report.json
```
