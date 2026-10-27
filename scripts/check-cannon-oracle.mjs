// Narrow original-source differential smoke; NOT whole-card/site parity.
// Rust fixtures are supplied separately. No subprocess, network, DOM or whole bundle execution.
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import vm from 'node:vm';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const fixturesPath = process.argv[2];
if (!fixturesPath) throw new Error('Usage: node scripts/check-cannon-oracle.mjs <cannon-fixtures.json>');
const bundlePath = path.join(root, 'rulesets/augment_chess_20260915/sources/raw/main-6qeAcxbX.js');
const bytes = readFileSync(bundlePath);
const hash = createHash('sha256').update(bytes).digest('hex');
const expected = '60d5816e78896d4b5371c0d5cc37ef0a302aa0cc814c3b027be3936124878dfc';
if (hash !== expected) throw new Error('Bundle hash changed; re-audit source ranges before running');
const lines = bytes.toString('utf8').split(/\r?\n/);
const ranges = [[1409, 1409], [2462, 2464], [2553, 2580], [2601, 2684]];
const source = ranges.map(([first, last]) => lines.slice(first - 1, last).join('\n')).join('\n');
// VM timeout is not a security sandbox. Execute only pinned, inspected declarations.
const fixtures = JSON.parse(readFileSync(fixturesPath, 'utf8').replace(/^\uFEFF/, ''));
if (fixtures.scope !== 'cannon-path/no-rules/no-passives' || fixtures.cases.length !== 3) throw new Error('Unexpected fixture scope');
const context = vm.createContext({ cases: fixtures.cases });
const body = [
  'cases.map(test => {',
  '  const position = name => ({ col: name.charCodeAt(0) - 97, row: 8 - Number(name.slice(1)) });',
  '  const board = test.board.map(p => ({ ...p, ...position(p.square) }));',
  '  const origin = position(test.origin);',
  '  const moves = cannonPathMoves({ ...origin,',
  '    getOccupant: (r, c) => board.find(p => p.row === r && p.col === c) || null,',
  "    canCapture: target => target.color === 'black' });",
  "  const destinations = moves.map(p => String.fromCharCode(97 + p.col) + (8 - p.row)).sort();",
  '  return { name: test.name, original_destinations: destinations, rust_destinations: test.rust_destinations,',
  '    match: JSON.stringify(destinations) === JSON.stringify(test.rust_destinations) };',
  '})',
].join('\n');
const results = new vm.Script(source + '\n' + body).runInContext(context, { timeout: 1000 });
const report = { scope: fixtures.scope, ruleset: fixtures.ruleset, bundle_sha256: hash,
  extracted_sha256: createHash('sha256').update(source).digest('hex'), source_ranges: ranges,
  historical_0915_parity: false, whole_card_parity: false, site_parity_claim: false,
  passed: results.every(r => r.match), results };
console.log(JSON.stringify(report, null, 2));
if (!report.passed) process.exitCode = 1;
