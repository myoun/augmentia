#!/usr/bin/env node
// Offline extraction/regression test. Never imports or executes the application bundle.
// Run: node scripts/test-reference-help.mjs
// Initial creation only: node scripts/test-reference-help.mjs --write
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const root = new URL('../', import.meta.url);
const bundlePath = 'rulesets/augment_chess_20260915/sources/raw/main-6qeAcxbX.js';
const outputPath = 'crates/augment-app/src/web/static/reference-help.json';
const sourceSha256 = '60d5816e78896d4b5371c0d5cc37ef0a302aa0cc814c3b027be3936124878dfc';
const read = (path) => readFileSync(new URL(path, root), 'utf8');

// Explicit core snake_case -> official table/key mapping. Do not guess from names.
const pieceAliases = {
  pawn: ['RANDOM_ROULETTE_PIECE_HELP', 'pawn'],
  knight: ['RANDOM_ROULETTE_PIECE_HELP', 'knight'],
  bishop: ['RANDOM_ROULETTE_PIECE_HELP', 'bishop'],
  rook: ['RANDOM_ROULETTE_PIECE_HELP', 'rook'],
  queen: ['RANDOM_ROULETTE_PIECE_HELP', 'queen'],
  king: ['RANDOM_ROULETTE_PIECE_HELP', 'king'],
  assassin: ['RANDOM_ROULETTE_PIECE_HELP', 'assassin'],
  grasshopper: ['RANDOM_ROULETTE_PIECE_HELP', 'grasshopper'],
  amazon: ['RANDOM_ROULETTE_PIECE_HELP', 'amazon'],
  prime_minister: ['RANDOM_ROULETTE_PIECE_HELP', 'primeMinister'],
  dragon: ['RANDOM_ROULETTE_PIECE_HELP', 'dragon'],
  cannon: ['RANDOM_ROULETTE_PIECE_HELP', 'cannon'],
  herald: ['RANDOM_ROULETTE_PIECE_HELP', 'herald'],
  knightmaster: ['RANDOM_ROULETTE_PIECE_HELP', 'knightmaster'],
  unicorn: ['RANDOM_ROULETTE_PIECE_HELP', 'pegasus'],
  hook: ['RANDOM_ROULETTE_PIECE_HELP', 'hook'],
  idol: ['RANDOM_ROULETTE_PIECE_HELP', 'idol'],
  jester: ['RANDOM_ROULETTE_PIECE_HELP', 'jester'],
  recruiter: ['RANDOM_ROULETTE_PIECE_HELP', 'recruiter'],
  reaper: ['RANDOM_ROULETTE_PIECE_HELP', 'reaper'],
  wizard: ['RANDOM_ROULETTE_PIECE_HELP', 'wizard'],
  alibaba: ['RANDOM_ROULETTE_PIECE_HELP', 'eagle'],
  log: ['RANDOM_ROULETTE_PIECE_HELP', 'log'],
  missionary: ['RANDOM_ROULETTE_PIECE_HELP', 'missionary'],
  cardinal: ['RANDOM_ROULETTE_PIECE_HELP', 'cardinal'],
  protestant: ['RANDOM_ROULETTE_PIECE_HELP', 'protestant'],
  standard_bearer: ['RANDOM_ROULETTE_PIECE_HELP', 'standardBearer'],
  windmill: ['RANDOM_ROULETTE_PIECE_HELP', 'windmill'],
  squire: ['RANDOM_ROULETTE_PIECE_HELP', 'squire'],
  big_rook: ['RANDOM_ROULETTE_PIECE_HELP', 'bigRook'],
  checker: ['RANDOM_ROULETTE_PIECE_HELP', 'checker'],
  checker_king: ['RANDOM_ROULETTE_PIECE_HELP', 'checkerKing'],
  alfil: ['RANDOM_ROULETTE_PIECE_HELP', 'alfil'],
  guard: ['RANDOM_ROULETTE_PIECE_HELP', 'guard'],
  fanatic: ['RANDOM_ROULETTE_PIECE_HELP', 'fanatic'],
  colossus: ['RANDOM_ROULETTE_PIECE_HELP', 'colossus'],
  royal_knight: ['RANDOM_ROULETTE_PIECE_HELP', 'royalKnight'],
  man: ['RANDOM_ROULETTE_PIECE_HELP', 'man'],
  merchant: ['CARD_HELP', 'merchant-guild'],
  dark_wizard: ['RANDOM_ROULETTE_PIECE_HELP', 'darkWizard'],
  monster: ['CARD_HELP', 'monster'],
  scarecrow: ['RANDOM_ROULETTE_PIECE_HELP', 'scarecrow'],
  camel: ['RANDOM_ROULETTE_PIECE_HELP', 'camel'],
  ferz: ['RANDOM_ROULETTE_PIECE_HELP', 'ferz'],
  bat: ['RANDOM_ROULETTE_PIECE_HELP', 'bat'],
  baby_bear: ['RANDOM_ROULETTE_PIECE_HELP', 'babyBear'],
  lobster: ['RANDOM_ROULETTE_PIECE_HELP', 'lobster'],
  bear: ['RANDOM_ROULETTE_PIECE_HELP', 'bear'],
  vip: ['RANDOM_ROULETTE_PIECE_HELP', 'vip'],
  crown: ['CARD_HELP', 'crown'],
  football: ['CARD_HELP', 'football'],
  siege_ram: ['RANDOM_ROULETTE_PIECE_HELP', 'siegeRam'],
  magic_girl: ['RANDOM_ROULETTE_PIECE_HELP', 'magicGirl'],
  berserker: ['RANDOM_ROULETTE_PIECE_HELP', 'berserker'],
  slime: ['RANDOM_ROULETTE_PIECE_HELP', 'slime'],
  siren: ['RANDOM_ROULETTE_PIECE_HELP', 'siren'],
  trickster: ['RANDOM_ROULETTE_PIECE_HELP', 'trickster'],
  undead: ['RANDOM_ROULETTE_PIECE_HELP', 'undead'],
  clockwork: ['RANDOM_ROULETTE_PIECE_HELP', 'clockwork'],
  parrot: ['RANDOM_ROULETTE_PIECE_HELP', 'parrot'],
  paladin: ['RANDOM_ROULETTE_PIECE_HELP', 'paladin'],
  princess: ['RANDOM_ROULETTE_PIECE_HELP', 'princess'],
  octopus: ['RANDOM_ROULETTE_PIECE_HELP', 'octopus'],
  hedgehog: ['RANDOM_ROULETTE_PIECE_HELP', 'hedgehog'],
  campfire: ['RANDOM_ROULETTE_PIECE_HELP', 'campfire'],
  thief: ['RANDOM_ROULETTE_PIECE_HELP', 'thief'],
  brutus: ['RANDOM_ROULETTE_PIECE_HELP', 'brutus'],
  big_bishop: ['CARD_HELP', 'big-bishop'],
};

function extractTables(source) {
  // These reviewed declarations contain only literals, table references, spreads,
  // Object.freeze, and string interpolation. SHA pin prevents silently evaluating
  // changed code. Anchored closing lines exclude surrounding runtime/application code.
  const names = [
    'SEPTEMBER_HELP', 'INTERNAL_EIGHT_HELP', 'INTERNAL_THREE_HELP',
    'INTERNAL_FIVE_HELP', 'CARD_HELP', 'PIECE_HELP',
    'TURN_CONSUMING_CARD_HELP', 'TURN_EXCLUSIVE_CARD_HELP', 'CARD_HELP_ITEMS',
    'RANDOM_ROULETTE_PIECE_HELP',
  ];
  const blocks = names.map((name) => {
    const matches = [...source.matchAll(new RegExp(`^const ${name} = (?:Object\\.freeze\\()?\\{[\\s\\S]*?^\\}\\)?;`, 'gm'))];
    assert.equal(matches.length, 1, `Expected exactly one pure ${name} declaration`);
    assert.ok(matches[0][0].length < 50000, `${name} extraction exceeds bound`);
    return matches[0][0];
  });
  // Preserve the bundle's explicit squire help correction before roulette is built.
  const patches = [...source.matchAll(/^CARD_HELP\["apprentice-knights"\] = "[^"\r\n]*";/gm)];
  assert.equal(patches.length, 1, 'Expected the official squire help correction');
  blocks.splice(blocks.length - 1, 0, patches[0][0],
    'CARD_HELP_ITEMS["apprentice-knights"][0].text = CARD_HELP["apprentice-knights"];');
  const context = vm.createContext(Object.create(null), {
    codeGeneration: { strings: false, wasm: false },
  });
  const script = new vm.Script(`${blocks.join('\n')}\nJSON.stringify({${names.join(',')}});`,
    { filename: 'reference-help-pure-constants.js' });
  // No require/process/browser globals, no full bundle execution, bounded timeout.
  return JSON.parse(script.runInContext(context, { timeout: 250 }));
}

const bytes = readFileSync(new URL(bundlePath, root));
const sha = createHash('sha256').update(bytes).digest('hex');
assert.equal(sha, sourceSha256, 'Trusted bundle SHA changed; review constants before extraction');
const manifest = JSON.parse(read('rulesets/augment_chess_20260915/sources/source-manifest.json'));
assert.equal(manifest.bundle.sha256, sha);
const tables = extractTables(bytes.toString('utf8'));
const catalog = JSON.parse(read('rulesets/augment_chess_20260915/cards/catalog.json'));
assert.equal(catalog.source_bundle_sha256, sha);
const cardIds = new Set(catalog.cards.map((card) => card.id));
// The catalog intentionally hides shotgun-king, but it remains an official card.
for (const id of catalog.hidden_by_default_ids ?? []) cardIds.add(id);
const setupList = read('crates/augment-app/src/web/setup.rs').match(/pub const PIECE_KINDS:[\s\S]*?= &\[([\s\S]*?)\];/);
assert.ok(setupList, 'Core piece list not found');
const coreKinds = [...setupList[1].matchAll(/PieceKind::([A-Za-z]+)/g)]
  .map(([, kind]) => kind.replace(/([a-z0-9])([A-Z])/g, '$1_$2').toLowerCase());
assert.equal(coreKinds.length, 70);
assert.equal(new Set(coreKinds).size, 70);
// October compound help is pinned separately in test-compound-pieces.mjs.
const historicalKinds = coreKinds.filter(kind => !['archbishop', 'chancellor'].includes(kind));
assert.deepEqual(Object.keys(pieceAliases), historicalKinds, 'Alias mapping must cover historical core kinds in order');

const pieces = {};
for (const kind of historicalKinds) {
  const [table, key] = pieceAliases[kind];
  const text = tables[table][key];
  // Never synthesize missing descriptions.
  if (typeof text === 'string' && text.length > 0) pieces[kind] = text;
}
const cards = {};
for (const id of [...cardIds].sort()) {
  // Matches official CARD_HELP_ITEMS preference, with CARD_HELP fallback.
  // Baseline catalog text is deliberately NOT duplicated or substituted here.
  const items = tables.CARD_HELP_ITEMS[id];
  const text = items ? items.map((item) => item.text).join(' ') : tables.CARD_HELP[id];
  if (typeof text === 'string' && text.length > 0) cards[id] = text;
}
const expected = {
  provenance: {
    sourceBundle: bundlePath,
    sourceBundleSha256: sha,
    sourceConstants: [
      'SEPTEMBER_HELP', 'INTERNAL_*_HELP', 'CARD_HELP', 'PIECE_HELP',
      'CARD_HELP_ITEMS', 'RANDOM_ROULETTE_PIECE_HELP',
    ],
    notes: [
      'Official Korean help from the local bundle; baseline general rules/card text remains in the parent source.',
      'Card help joins the explicit CARD_HELP_ITEMS, or falls back to CARD_HELP for known card IDs. Runtime phase-based help additions are not evaluated.',
      'The source manifest notes that this bundle contains unannounced live overrides; these help strings are not a claim of announced 2026-09-15 behavioral parity.',
    ],
  },
  pieces,
  cards,
};
const unmappedCoreKinds = historicalKinds.filter((kind) => !Object.hasOwn(pieces, kind));
const usedOfficialTypes = new Set(Object.values(pieceAliases)
  .filter(([table]) => table === 'RANDOM_ROULETTE_PIECE_HELP').map(([, key]) => key));
const nonCoreOfficialTypes = Object.keys(tables.RANDOM_ROULETTE_PIECE_HELP)
  .filter((key) => !usedOfficialTypes.has(key)).sort();

assert.ok(process.argv.slice(2).every((arg) => arg === '--write'), 'Only --write is supported');
if (process.argv.includes('--write')) {
  // Exclusive create refuses to overwrite any existing file, even valid output.
  writeFileSync(new URL(outputPath, root), `${JSON.stringify(expected, null, 2)}\n`, { flag: 'wx' });
}
const actual = JSON.parse(read(outputPath));
assert.deepEqual(Object.keys(actual).sort(), ['cards', 'pieces', 'provenance']);
for (const [section, keyPattern] of [['pieces', /^[a-z][a-z0-9_]*$/], ['cards', /^[a-z][a-z0-9-]*$/]]) {
  assert.ok(actual[section] && !Array.isArray(actual[section]) && typeof actual[section] === 'object');
  for (const [key, text] of Object.entries(actual[section])) {
    assert.match(key, keyPattern);
    assert.ok(!['__proto__', 'constructor', 'prototype'].includes(key));
    assert.ok(typeof text === 'string' && text.trim().length > 0 && text.length < 10000, `${section}.${key} must be bounded text`);
    assert.match(text, /[가-힣]/, `${section}.${key} must contain Korean help`);
    assert.doesNotMatch(text, /[<>\u0000-\u001f\u007f]/, `${section}.${key} is plain text, not markup/control characters`);
    if (section === 'pieces') assert.ok(coreKinds.includes(key), `Unknown core kind ${key}`);
    else assert.ok(cardIds.has(key), `Unknown card ID ${key}`);
  }
}
assert.deepEqual(actual, expected, 'Every description must exactly match extracted official help');
assert.equal(unmappedCoreKinds.length, 0, `Unmapped core kinds: ${unmappedCoreKinds.join(', ')}`);
// Regression checks for easily-confused aliases and corrected official descriptions.
assert.equal(actual.pieces.royal_knight, '로얄 나이트: 나이트처럼 움직이는 왕입니다. 모든 승패 판정에서 킹으로 취급됩니다.');
assert.equal(actual.pieces.baby_bear, '아기곰: 킹처럼 행마합니다. 기본적으로 자신의 턴이 시작할때 무작위 위치로 이동합니다. 기물을 잡을 수 없습니다.');
assert.equal(actual.pieces.unicorn, tables.CARD_HELP.pegasus);
assert.equal(actual.pieces.alibaba, tables.CARD_HELP.eagle);
assert.ok(actual.pieces.squire.endsWith('프로모션이 가능합니다.'));
assert.equal(actual.cards.knightmate, `${actual.pieces.royal_knight} ${actual.pieces.man}`);
assert.equal(actual.cards['baby-bear'], `${actual.pieces.baby_bear} ${actual.pieces.bear}`);
console.log(`PASS: ${Object.keys(pieces).length}/${historicalKinds.length} historical pieces; ${Object.keys(cards).length}/${cardIds.size} cards with optional detailed help. October compound help is checked separately.`);
console.log(`Unmapped core kinds: ${unmappedCoreKinds.length ? unmappedCoreKinds.join(', ') : '(none)'}`);
console.log(`Official roulette types without a core kind: ${nonCoreOfficialTypes.join(', ')}`);
console.log(`Source bundle SHA-256: ${sha}`);
console.log(`Verified ${fileURLToPath(new URL(outputPath, root))}`);
