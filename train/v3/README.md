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
