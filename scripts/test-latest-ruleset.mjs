// Offline provenance, canonical extraction replay, and fail-closed profile checks.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, mkdtempSync, readFileSync, rmdirSync, unlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const root = new URL('../', import.meta.url);
const snapshot = new URL('rulesets/augment_chess_20261102/', root);
const json = url => JSON.parse(readFileSync(url, 'utf8'));
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const manifest = json(new URL('manifest.json', snapshot));
const provenance = json(new URL('sources/source-manifest.json', snapshot));
const catalog = json(new URL('cards/catalog.json', snapshot));
const profile = json(new URL('cards/implemented.json', snapshot));
const delta = json(new URL('sources/0915-delta.json', snapshot));
const ports = json(new URL('cards/ported.json', snapshot));
for (const file of provenance.files) {
  const bytes = readFileSync(new URL(file.path, snapshot));
  assert.equal(bytes.length, file.byte_length, file.path);
  assert.equal(digest(bytes), file.sha256, file.path);
}
assert.equal(provenance.historical_release_bytes_verified, false);
assert.equal(catalog.cards.length, 281);
assert.equal(catalog.cards.filter(card => card.enabled).length, 272);
assert.deepEqual(catalog.hidden_by_default_ids, [
  'beacon', 'blue-jeans', 'chess-n-pow-30', 'democracy', 'martyrdom',
  'monochrome-chess', 'qxe1', 'royal-shield', 'underground-bunker',
]);
for (const card of catalog.cards) {
  if (card.phase === 'RULE') { assert.equal(card.rarity, null); continue; }
  assert.equal(card.draft_weight_units, { common: 85, rare: 70, epic: 58, legendary: 44 }[card.rarity], card.id);
  if (card.enabled && card.phase === 'OPENING') assert.equal(card.activation, 'PASSIVE', card.id);
}
const oldCatalog = new Map(json(new URL('rulesets/augment_chess_20260915/cards/catalog.json', root)).cards.map(card => [card.id, card]));
const oldHandlers = new Map(json(new URL('rulesets/augment_chess_20260915/cards/implemented.json', root)).cards.map(card => [card.id, card]));
// A card carried from the 0915 handlers unchanged: same contract fields and the
// same activation/turn policy it had there. A card with a port record is excluded
// because it was re-verified against a newer source rather than inherited.
const declaredPorts = new Set(ports.cards.map(card => card.id));
const inherited = catalog.cards.filter(card => {
  if (!card.enabled || declaredPorts.has(card.id)) return false;
  const old = oldCatalog.get(card.id), handler = oldHandlers.get(card.id);
  // Inheriting means the executable contract is unchanged: same effect, phase and
  // activation/turn policy. The displayed description is allowed to change, since
  // that does not alter behavior.
  return old && handler && ['effect', 'phase'].every(key => old[key] === card[key])
    && handler.activation === (card.phase === 'RULE' ? 'rule' : card.activation.toLowerCase())
    && handler.ends_turn === (card.turn_policy === 'END_TURN');
}).map(card => card.id).sort();
const portIds = ports.cards.map(card => card.id).sort();
// Port records now carry cards that have no 0915 counterpart at all (they were
// introduced after that snapshot), so the profile is not simply inherited + ports.
// What must hold is: every profile card is either a declared port or an inherited
// 0915 contract, the two sets are disjoint, and profile + pending partitions the
// catalog exactly.
assert.equal(new Set(portIds).size, ports.cards.length);
assert.equal(inherited.filter(id => portIds.includes(id)).length, 0);
const profileIds = profile.cards.map(card => card.id).sort();
const coveredBySource = new Set([...inherited, ...portIds]);
const uncovered = profileIds.filter(id => !coveredBySource.has(id));
assert.deepEqual(uncovered, [], 'every profile card must have a source record');
assert.deepEqual(
  profileIds,
  [...coveredBySource].filter(id => profileIds.includes(id)).sort(),
  'profile must equal the subset of source records it declares',
);
assert.equal(profile.cards.length + profile.pending.length, catalog.cards.length);
assert.equal(new Set([...profileIds, ...profile.pending.map(card => card.id)]).size, 281);
assert.deepEqual(profile.pending, delta.unsupported);

// Release gate condition on card coverage: every publicly active card and rule is
// executable. This is separate from site behavior parity, which the manifest reports
// through site_parity_verified_card_count and which stays unverified.
const activeCards = catalog.cards.filter(card => card.enabled).length;
assert.equal(manifest.coverage.publicly_active_card_count, activeCards, 'active count must match the catalog');
assert.equal(manifest.coverage.implemented_card_count, profile.cards.length, 'implemented count must match the profile');
assert.equal(
  manifest.coverage.status,
  profile.cards.length >= activeCards ? 'publicly_active' : 'partial',
  'status must follow from implemented vs active',
);
assert.ok(
  profile.pending.every(entry => entry.reason === 'disabled_upstream'),
  'nothing executable may remain pending',
);

const directory = mkdtempSync(join(tmpdir(), 'augmentia-catalog-'));
const output = join(directory, 'catalog.json'), evidence = join(directory, 'evidence.json');
const invalid = join(directory, 'unreviewed.js');
const run = bundle => spawnSync(process.execPath, [fileURLToPath(new URL('scripts/extract-upstream-catalog.mjs', root)),
  '--bundle', bundle, '--output', output, '--evidence', evidence, '--ruleset-id', manifest.id], { encoding: 'utf8', timeout: 60000 });
// The bundle filename changes between snapshots, so locate it through the
// snapshot's own source manifest instead of hardcoding it.
const bundlePath = provenance.files.find(file => file.path.includes('main')).path;
try {
  const result = run(fileURLToPath(new URL(bundlePath, snapshot)));
  assert.equal(result.status, 0, result.stderr);
  assert.equal(readFileSync(output, 'utf8'), readFileSync(new URL('cards/catalog.json', snapshot), 'utf8'));
  assert.equal(readFileSync(evidence, 'utf8'), readFileSync(new URL('sources/catalog-extraction.json', snapshot), 'utf8'));
  unlinkSync(output); unlinkSync(evidence);
  writeFileSync(invalid, 'throw new Error("must not execute unreviewed input");');
  const rejected = run(invalid);
  assert.notEqual(rejected.status, 0);
  assert.match(rejected.stderr, /Unreviewed bundle/);
  assert.equal(existsSync(output), false);
  assert.equal(existsSync(evidence), false);
} finally {
  for (const file of [output, evidence, invalid]) if (existsSync(file)) unlinkSync(file);
  rmdirSync(directory);
}
console.log(`Latest ruleset: raw hashes, canonical replay, ${inherited.length} inherited + ${portIds.length} scoped ports / ${profile.pending.length} blocked, unknown source rejection passed.`);
