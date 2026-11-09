// Narrow checks of pinned source contracts. This is not full application parity.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import vm from 'node:vm';

const root = new URL('../rulesets/augment_chess_20261017/', import.meta.url);
const bytes = readFileSync(new URL('sources/raw/main-RjrZ5fFP.js', root));
const hash = createHash('sha256').update(bytes).digest('hex');
assert.equal(hash, '2e932da7b9391effbeab3593ea5fdf855041fe5a74f611dfdfde678eee46beb8');
const source = bytes.toString('utf8');
const declarations = [
  'BASE_MINOR_PIECE_TYPES', 'RECLASSIFIED_MINOR_PIECE_TYPES', 'EXPANDED_MINOR_PIECE_TYPES', 'NATIVE_MINOR_PIECE_TYPES',
  'minorPieceTypesForCatalog', 'isMinorPieceTypeForCatalog', 'canonicalMinorPieceTypesForCatalog',
  'OVERWHELM_KING_TYPES', 'isOverwhelmKingPiece', 'isOverwhelmRoyalTarget', 'isOverwhelmCaptureBlocked',
  'highlanderEligible', 'initiativeDuration', 'assemblyPromotion', 'V7_SELECTED_TRANSFORMS',
];
function declaration(name) {
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const match = new RegExp(`(?:^|\\n)(const ${escaped} = |function ${escaped}\\()`).exec(source);
  assert.ok(match, name);
  const start = match.index + (source[match.index] === '\n' ? 1 : 0);
  const isFunction = source.startsWith('function ', start);
  let end = start;
  for (let attempt = 0; attempt < 1000; attempt++) {
    end = source.indexOf(isFunction ? '\n' : ';', end + 1);
    if (end < 0 || end - start > 25000) break;
    const code = source.slice(start, end + (isFunction ? 0 : 1));
    try { new vm.Script(code); } catch { continue; }
    assert.doesNotMatch(code, /\b(?:eval|Function|fetch|document|window|process|require|import)\b|\.constructor\b/);
    return code;
  }
  throw new Error(`Unrecognized declaration ${name}`);
}
// Select current catalog policies explicitly; the source supports old hashes too.
// These adapters select the captured profile, not gameplay implementations.
const adapters = `
const usesOctober13Reform = () => true;
const usesOctober15CardBalance = () => true;
const usesOctober7Balance = () => true;
const usesNativePieceClassification = () => true;
const usesSeptember26Rebalance = () => true;
const opponent$2 = color => color === 'white' ? 'black' : 'white';
`;
const checks = `
const transforms = ['assassin', 'dragon', 'knightmaster', 'log', 'standard-bearer', 'local-conscription']
  .map(id => ({ id, ...V7_SELECTED_TRANSFORMS[id] }));
const highlander = [
  ['king', 'knight', 'bishop'], ['king', 'knight', 'bishop', 'rook'],
  ['king', 'knight', 'knight', 'rook'], ['royalKnight', 'knight', 'bishop', 'rook']
].map(types => highlanderEligible(types.map(type => ({color:'white', type})), 'white'));
const overwhelm = ['queen','king','rook'].map(type => isOverwhelmCaptureBlocked({white:true}, {color:'black',type:'king'}, {color:'white',type}));
const assembly = [['rook','bishop'],['bishop','rook'],['rook','pawn'],['queen','bishop']]
  .map(([a,b]) => assemblyPromotion({color:'white',type:a},{color:'black',type:b},true));
JSON.stringify({ transforms, highlander, overwhelm, assembly, initiative:initiativeDuration(),
  minors: canonicalMinorPieceTypesForCatalog(null) });
`;
const context = vm.createContext({}, { codeGeneration: { strings: false, wasm: false } });
const result = JSON.parse(new vm.Script(adapters + declarations.map(declaration).join('\n') + checks)
  .runInContext(context, { timeout: 1000 }));
assert.deepEqual(result.highlander, [false, true, false, false]);
assert.deepEqual(result.overwhelm, [true, false, false]);
assert.deepEqual(result.assembly, [true, true, false, false]);
assert.equal(result.initiative, 8);
assert.deepEqual(result.transforms.map(c => [c.id, c.from, c.to]), [
  ['assassin',['knight'],'assassin'], ['dragon',['rook'],'dragon'],
  ['knightmaster',['knight'],'knightmaster'], ['log',['pawn'],'log'],
  ['standard-bearer',['pawn'],'standard-bearer'], ['local-conscription',['queen'],'recruiter'],
]);
assert.equal(result.transforms.find(c => c.id === 'local-conscription').attributes.capturesMade, 0);
assert.ok(result.minors.includes('wizard'));
assert.ok(result.minors.includes('monster'));
assert.ok(!result.minors.includes('queen'));
assert.ok(!result.minors.includes('man'));
const ports = JSON.parse(readFileSync(new URL('cards/ported.json', root), 'utf8'));
assert.equal(ports.source_bundle_sha256, hash);
const lines = source.split(/\r?\n/);
for (const card of ports.cards) for (const range of card.source_ranges) {
  assert.ok(Number.isInteger(range[0]) && range[0] > 0 && range[1] >= range[0] && range[1] <= lines.length);
}
console.log('Effect runtime ports: pinned transform contracts, native minors, initiative/overwhelm/highlander/assembly source helpers passed; full interaction parity unverified.');
