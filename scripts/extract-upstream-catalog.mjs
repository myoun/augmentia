// Pinned catalog-only oracle. Never executes the application entry, UI or auth.
// The VM timeout is a resource bound, not a security sandbox.
import { readFileSync, writeFileSync } from 'node:fs';
import vm from 'node:vm';
import { createHash } from 'node:crypto';

const args = process.argv.slice(2);
const allowed = new Set(['--bundle', '--output', '--evidence', '--ruleset-id']);
if (args.length !== 8 || args.some((arg, index) => index % 2 === 0 && !allowed.has(arg))) {
  throw new Error('Expected exactly four known option/value pairs');
}
const options = Object.fromEntries(Array.from({ length: args.length / 2 }, (_, i) =>
  [args[i * 2], args[i * 2 + 1]]));
if (!options['--bundle'] || !options['--output'] || !options['--evidence'] || !options['--ruleset-id']) {
  throw new Error('Expected --bundle FILE --output FILE --evidence FILE --ruleset-id ID');
}
const bytes = readFileSync(options['--bundle']);
const sha256 = value => createHash('sha256').update(value).digest('hex');
const bundleHash = sha256(bytes);
// Reviewed bundle hashes. Each entry was checked by resolving the catalog
// declarations under the capability blocklist below before being added, and the
// evidence lives in that snapshot's sources/catalog-extraction.json.
const REVIEWED_BUNDLES = new Set([
  // 2026-10-17 canonical v19.
  '2e932da7b9391effbeab3593ea5fdf855041fe5a74f611dfdfde678eee46beb8',
  // 2026-11-02, release 2026-11-01-independent-draft-expansion-v1.
  'af362a64df02321edc00f225126c750b0ca41156c158d85930f273bbfad3f8b5',
]);
if (!REVIEWED_BUNDLES.has(bundleHash)) {
  throw new Error('Unreviewed bundle: catalog oracle must be reviewed for the new source hash');
}
const source = bytes.toString('utf8');
const definitions = new Map();
const locations = new Map();
const escaped = value => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
function declaration(name) {
  const match = new RegExp(`(?:^|\\n)(const ${escaped(name)} = |function ${escaped(name)}\\()`).exec(source);
  if (!match) throw new Error(`Missing catalog declaration ${name}`);
  const start = match.index + (source[match.index] === '\n' ? 1 : 0);
  locations.set(name, start);
  const isFunction = source.startsWith('function ', start);
  let end = start;
  for (let count = 0; count < 20000; count++) {
    end = source.indexOf(isFunction ? '\n' : ';', end + 1);
    if (end < 0 || end - start > 2000000) break;
    const code = source.slice(start, end + (isFunction ? 0 : 1));
    try { new vm.Script(code); } catch { continue; }
    if (/\b(?:eval|Function|fetch|XMLHttpRequest|document|window|process|require|import)\b|\.constructor\b/.test(code)) {
      throw new Error(`Non-catalog capability in ${name}`);
    }
    return code;
  }
  throw new Error(`Unterminated catalog declaration ${name}`);
}
for (const name of ['BETA_PUBLIC_CARD_CATALOG_V19', 'REFORM_CATALOG_HASH']) definitions.set(name, declaration(name));
let resolved;
for (let attempt = 0; attempt < 1600; attempt++) {
  const order = [...definitions.keys()].sort((a, b) => locations.get(a) - locations.get(b));
  const code = order.map(name => definitions.get(name)).join('\n') +
    '\nJSON.stringify({ catalog: BETA_PUBLIC_CARD_CATALOG_V19, catalogHash: REFORM_CATALOG_HASH });';
  try {
    const context = vm.createContext({}, { codeGeneration: { strings: false, wasm: false } });
    resolved = JSON.parse(new vm.Script(code).runInContext(context, { timeout: 5000 }));
    break;
  } catch (error) {
    const missing = /^(\S+) is not defined$/.exec(error.message)?.[1] ??
      /^Cannot access '([^']+)' before initialization$/.exec(error.message)?.[1];
    if (!missing || definitions.has(missing)) throw error;
    definitions.set(missing, declaration(missing));
  }
}
if (!resolved) throw new Error('Catalog dependency budget exceeded');
const cards = resolved.catalog.cards;
// The catalog grows between snapshots, so require a plausible non-empty catalog
// with unique ids rather than a pinned count. The snapshot's own
// source-manifest.json records the expected totals and is checked against this
// output by test-latest-ruleset.mjs.
if (!Array.isArray(cards) || cards.length === 0) throw new Error('Empty catalog');
if (new Set(cards.map(card => card.id)).size !== cards.length) {
  throw new Error('Catalog coverage mismatch: duplicate card ids');
}
const normalized = cards.map(card => ({
  id: card.id, name_ko: card.name, phase: card.draftCategory, half_stars: card.ratingHalfStars,
  effect: card.effect, text_ko: card.description, art: card.art, enabled: card.enabled !== false,
  rarity: card.draftCategory === 'RULE' ? null : card.rarity ?? (['joker','prophecy','hook','ultimatum','vanish'].includes(card.id) || card.ratingHalfStars === 10 ? 'legendary' : card.ratingHalfStars <= 6 ? 'common' : card.ratingHalfStars === 7 ? 'rare' : 'epic'),
  activation: card.activation, play_timing: card.playTiming, turn_policy: card.turnPolicy,
  draft_weight_units: card.draftWeightUnits, draw_requirement: card.drawRequirement, selection: card.selection,
})).sort((a, b) => a.id.localeCompare(b.id, 'en'));
const output = {
  ruleset_id: options['--ruleset-id'], schema_version: 2, source_kind: 'canonical-v19',
  source_bundle_sha256: bundleHash, source_bundle_byte_length: bytes.length,
  source_catalog_hash: resolved.catalogHash, source_schema_version: resolved.catalog.schemaVersion,
  raw_card_definitions: cards.length, catalog_entry_count: cards.length,
  visible_card_count: normalized.filter(card => card.enabled).length,
  hidden_by_default_ids: normalized.filter(card => !card.enabled).map(card => card.id), cards: normalized,
};
const evidence = {
  bundle_sha256: bundleHash, catalog_hash: resolved.catalogHash,
  catalog_metadata: Object.fromEntries(Object.entries(resolved.catalog).filter(([key]) => key !== 'cards')),
  method: 'pinned catalog declarations only; application entry not executed; VM is not a security sandbox',
  declarations: [...definitions].map(([name, code]) => ({ name, offset: locations.get(name), sha256: sha256(code) })),
  normalized_catalog_sha256: sha256(JSON.stringify(output, null, 2) + '\n'),
};
writeFileSync(options['--output'], JSON.stringify(output, null, 2) + '\n', { flag: 'wx' });
writeFileSync(options['--evidence'], JSON.stringify(evidence, null, 2) + '\n', { flag: 'wx' });
console.log(JSON.stringify({ cards: cards.length, active: output.visible_card_count, disabled: output.hidden_by_default_ids, catalogHash: resolved.catalogHash }));
