# V4 factorized policy contract

Status: experimental implementation. Grammar, replay target, Python model/export,
self-play, Rust factorized MCTS, and the Rust split-ONNX evaluator exist in the current
worktree. The first sealed fixed-budget baseline is recorded in
[ai-v4-baseline.md](ai-v4-baseline.md); arena promotion remains incomplete.
See [implementation status and verification](ai-v4-implementation-status.md) for
completed fixes, test results, reproduction commands, and the remaining roadmap.

V4 replaces V3's flat score for every complete `ActionDescriptorV3` with a legal,
conditional action trie. The engine remains the authority for legal complete actions.
The trie is an inference and search representation only.

## Contract

A V4 policy decision has these ordered levels:

1. `family`: `move`, `piece_ability`, `card`, `draft_pick`, or `resolve_choice`.
2. `source`: the piece/card/offer/choice instance selected by the family.
3. `detail`: family-specific fixed fields, such as ability, move destination, promotion,
   handler, or choice option.
4. `target`: one or more canonical target selections, or `stop` where the complete
   action is legal at that prefix.

The grammar is generated from the exact `LegalActionSetV3`, not reconstructed from
heuristics. Every complete legal descriptor must have exactly one path. A prefix
contains exactly the actions that share it. The engine decodes and applies only the
complete descriptor at a terminal trie leaf.

A trie edge has a stable, typed identifier. IDs must not depend on UI labels or raw
RNG state. Piece and card *instances* use their existing observation slots; card
*definitions*, handler IDs, ability IDs, squares, offer indexes and options use the
existing V3 stable encodings. A change to this grammar requires a new
`factorized_policy_schema_id`.

## Policy target

`augmentia.factorized-mcts-visits.v2` stores a complete-action visit vector plus the
generated trie version. V2 also records the number of public-information chance outcomes
sampled per stochastic action; the earlier experimental V1 target must not be mixed silently. The loader derives every prefix target by summing visits below each
legal child. It applies cross entropy at every visited prefix with its full engine
legal child mask. A child is zero only when it was legal and opened at that prefix;
it is never created merely because its complete-action visit is zero.

The target record includes:

- `factorized_policy_schema_id`;
- exact legal complete actions in V3 local order;
- complete-action visits and completed simulations;
- canonical trie checksum and all prefix legal-child masks;
- `search_semantics_id`, chance-sampling and root-selection provenance;
- `chance_outcome_samples` in `1..=64`.

V3 flat targets and V4 factorized targets must not share a model checkpoint or be
silently mixed in a loss.

## Search

At the root, use fixed-candidate Gumbel sequential halving, without Dirichlet noise.
Its noise seed is derived from a dedicated Gumbel seed domain plus the root public-information
key; it is never reused as the chance-lane seed. At internal wide prefixes, use PUCT with
progressive widening. Draft offers visible
to the actor are all opened. Future Draft offers and other stochastic transitions use
multiple search-only samples derived from `ObservationV3` plus the exact legal action set;
the live `GameState` RNG and full replay fingerprint are forbidden seed inputs. Samples
for the same action are selected evenly so their values aggregate at the complete-action
leaf. Virtual prefix nodes do not apply the engine, change the actor, or flip backup sign.
Only a terminal trie leaf maps to one V3 descriptor and calls `Engine::apply_for_search`;
actual play still uses `Engine::apply`.

## Artifact compatibility

V4 uses an independent model metadata schema and a split bundle:
`state_encoder.onnx`, `factorized_scorer.onnx`, and metadata that hashes both files,
the grammar, ruleset, vocabulary, and model configuration. It does not replace or
reinterpret a V3 artifact. The runtime rejects a mixed bundle.

## Rust split runtime

`V4OnnxRuntime::from_bundle` takes the metadata path plus the expected ruleset ID
and manifest SHA-256. It validates schema IDs, full runtime vocabulary, graph hashes,
and declared and actual ONNX I/O dtype/rank contracts before evaluation. Runtime
options reuse `V3RuntimeOptions`; the bundle is not loaded through `V3OnnxRuntime`.

The runtime implements `FactorizedEvaluatorV4` for `MctsV4Tree`. It encodes the
public observation once and scores the legal trie prefix rows in a second graph.
Complete-action logits are sums of conditional log probabilities along each path.
The encoder cache uses serialized `ObservationV3`, never the full state or live RNG,
and retains at most 64 entries. This first runtime still materializes and scores
all legal prefix rows. Lazy prefix scoring and the NN-only beam fast path are later work.

Regenerate the shared fixture from `train/` with
`uv run python -m v4.export_fixture_bundle`. The vocabulary must include the full
runtime contract before model construction; editing metadata alone cannot extend
an existing embedding. Rust ONNX tests execute when `ORT_DYLIB_PATH` names the
installed ONNX Runtime library and skip only when the variable is absent.

## Adoption gates

- path set equals `LegalActionSetV3` exactly, with one canonical path per action;
- prefix masks, aggregate visits, terminal action indexes and replay round trips are
  deterministic;
- no illegal action or prefix/leaf visit-conservation violation;
- V4 candidate recall, solver quality, p95 latency and same-wall-clock paired arena
  beat the V3 full-policy baseline before default promotion.
