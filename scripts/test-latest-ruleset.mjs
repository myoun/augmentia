// Offline provenance, canonical extraction replay, and fail-closed profile checks.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, mkdtempSync, readFileSync, rmdirSync, unlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const root = new URL('../', import.meta.url);
const snapshot = new URL('rulesets/augment_chess_20261017/', root);
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
assert.equal(catalog.cards.length, 275);
assert.equal(catalog.cards.filter(card => card.enabled).length, 271);
assert.deepEqual(catalog.hidden_by_default_ids, ['democracy', 'qxe1', 'summon-colossus', 'underground-bunker']);
for (const card of catalog.cards) {
  if (card.phase === 'RULE') { assert.equal(card.rarity, null); continue; }
  assert.equal(card.draft_weight_units, { common: 85, rare: 70, epic: 58, legendary: 44 }[card.rarity], card.id);
  if (card.enabled && card.phase === 'OPENING') assert.equal(card.activation, 'PASSIVE', card.id);
}
const oldCatalog = new Map(json(new URL('rulesets/augment_chess_20260915/cards/catalog.json', root)).cards.map(card => [card.id, card]));
const oldHandlers = new Map(json(new URL('rulesets/augment_chess_20260915/cards/implemented.json', root)).cards.map(card => [card.id, card]));
const inherited = catalog.cards.filter(card => {
  const old = oldCatalog.get(card.id), handler = oldHandlers.get(card.id);
  return card.enabled && old && handler && ['effect', 'text_ko', 'phase'].every(key => old[key] === card[key])
    && handler.activation === (card.phase === 'RULE' ? 'rule' : card.activation.toLowerCase())
    && handler.ends_turn === (card.turn_policy === 'END_TURN');
}).map(card => card.id).sort();
const portIds = ports.cards.map(card => card.id).sort();
assert.equal(new Set(portIds).size, 30);
assert.equal(inherited.filter(id => portIds.includes(id)).length, 0);
assert.deepEqual(profile.cards.map(card => card.id).sort(), [...inherited, ...portIds].sort());
assert.equal(inherited.length + portIds.length, manifest.coverage.implemented_card_count);
assert.equal(profile.pending.length, 157);
assert.equal(new Set([...inherited, ...portIds, ...profile.pending.map(card => card.id)]).size, 275);
assert.deepEqual(profile.pending, delta.unsupported);

const directory = mkdtempSync(join(tmpdir(), 'augmentia-catalog-'));
const output = join(directory, 'catalog.json'), evidence = join(directory, 'evidence.json');
const invalid = join(directory, 'unreviewed.js');
const run = bundle => spawnSync(process.execPath, [fileURLToPath(new URL('scripts/extract-upstream-catalog.mjs', root)),
  '--bundle', bundle, '--output', output, '--evidence', evidence, '--ruleset-id', manifest.id], { encoding: 'utf8', timeout: 60000 });
try {
  const result = run(fileURLToPath(new URL('sources/raw/main-RjrZ5fFP.js', snapshot)));
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
console.log('Latest ruleset: raw hashes, canonical replay, 88 inherited + 30 scoped ports / 157 blocked, unknown source rejection passed.');
