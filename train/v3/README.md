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
`docs/ai-v3-training.md` and `docs/ai-v3-runtime.md`. V2 weights and fallback are never loaded.
