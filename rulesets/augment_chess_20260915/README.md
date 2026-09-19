# augment_chess_20260915

Snapshot of the 2026-09-15 augmentchess.org update.

## Provenance

- Live bundle: https://augmentchess.org/assets/main-6qeAcxbX.js captured immutable at artifacts/ruleset-0915-baseline/main-6qeAcxbX.js (sha256 in sources/source-manifest.json).
- Announcement text: https://augmentchess.org/updates/ captured at artifacts/ruleset-0915-baseline/updates.html.
- Source card evidence: artifacts/ruleset-0915-card-definitions.json (source_definition / announced / effective_live).

## Scope

- Implement the 2026-09-15 announced ruleset. Live bundle contains additional unannounced overrides and separate gameplay changes; those are audit data only.
- 31 new cards and 1 RULE from 2026-09-08 and 2026-09-15 announcements plus existing 208 visible cards.
- The 0818 ruleset stays byte-immutable. Old constructors (bundled_20260807/bundled_20260818) remain and are not the engine default.

## Files

- manifest.json: ruleset identity and coverage counters.
- cards/catalog.json: normalized visible catalog (0818 frozen cards plus announced 0915 delta).
- cards/implemented.json: implemented behavioral-slice contract. It started at 209 entries (208 existing cards plus Highlander) and now contains all 240 catalog entries. This does not imply full site parity; coverage metadata must stay aligned with the actual registered slices.
- notation.json: carried over from 0818.
- sources/source-manifest.json: bundle and page hashes.

## Snapshot policy

The announced catalog is the 0818 catalog plus explicit announcement deltas plus new card definitions.
Live-only unpublished values (for example Highlander 5 stars) are excluded from announced data.
The raw source files are also retained under `sources/raw/` with exact byte length and SHA256.
The initial 09.15 definitions come from the current source. No archived release bundle was available.
The announcement proves the phase changes: Princess and Vanguard are OPENING, despite PIECE/MIDDLE base definitions.
The latest constructor loads announced metadata. Legacy constructors do not switch the current engine back to historical behavior.
No cross-card site parity, AI model compatibility, or chaos-mode support is claimed.
