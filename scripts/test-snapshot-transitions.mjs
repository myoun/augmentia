// Pinned 2026-11-02 source contracts for the four P0-2 areas.
//
// The older oracles execute the 2026-10-17 bundle, so nothing compared this snapshot's transition
// rules against upstream. This script runs the current bundle directly and checks the four areas
// the backlog names: card locking, follow-up phase, probability and termination.
//
// Only self-contained predicates are pinned, so the adapter stays small and every assertion is
// decided by real bundle code rather than by a reimplementation.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import vm from 'node:vm';

const root = new URL('../', import.meta.url);
const bytes = readFileSync(
  new URL('rulesets/augment_chess_20261102/sources/raw/main-COPLmLfz.js', root),
);
assert.equal(
  createHash('sha256').update(bytes).digest('hex'),
  'af362a64df02321edc00f225126c750b0ca41156c158d85930f273bbfad3f8b5',
);
const source = bytes.toString('utf8');

function declaration(name) {
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const match = new RegExp(`(?:^|\\n)(const ${escaped} = |function ${escaped}\\()`).exec(source);
  assert.ok(match, `missing declaration ${name}`);
  const start = match.index + (source[match.index] === '\n' ? 1 : 0);
  const isFunction = source.startsWith('function ', start);
  let end = start;
  for (let attempt = 0; attempt < 2000; attempt += 1) {
    end = source.indexOf(isFunction ? '\n}\n' : ';', end + 1);
    if (end < 0 || end - start > 40000) break;
    const code = source.slice(start, isFunction ? end + 2 : end + 1);
    try {
      new vm.Script(code);
    } catch {
      continue;
    }
    assert.doesNotMatch(
      code,
      /\b(?:eval|Function|fetch|document|window|process|require|import)\b|\.constructor\b/,
    );
    return code;
  }
  throw new Error(`Unrecognized declaration ${name}`);
}

// Hash constants the policy predicates compare against, taken verbatim from the bundle.
const HASH_NAMES = [
  'REFORM_CATALOG_HASH',
  'PRE_NOVEMBER01_EXPANSION_HASH',
  'PRE_NOVEMBER01_HASH',
  'PRE_TROJAN_TURN_HASH',
  'PRE_OCTOBER27_BALANCE_HASH',
];
const hashPins = HASH_NAMES.map(name => declaration(name));

// The ruleset-ordinal table backs usesRulesAtLeast, which every `usesV<n>Rules` gate delegates to.
// It is built from version constants, so pin those and rebuild the map the same way.
const ordinalAt = source.indexOf('const RULESET_ORDINAL');
assert.ok(ordinalAt > 0, 'RULESET_ORDINAL must exist in this bundle');
let ordinalDepth = 0;
let ordinalEnd = source.indexOf('[', ordinalAt);
for (; ordinalEnd < source.length; ordinalEnd += 1) {
  if (source[ordinalEnd] === '[') ordinalDepth += 1;
  else if (source[ordinalEnd] === ']') {
    ordinalDepth -= 1;
    if (ordinalDepth === 0) break;
  }
}
// The map literal references AUTHORITATIVE_CARD_RULESET_V<n>_VERSION constants; inline them.
const ordinalLiteral = source.slice(source.indexOf('[', ordinalAt), ordinalEnd + 1).replace(
  /AUTHORITATIVE_CARD_RULESET_V(\d+)_VERSION\$2/g,
  (_m, n) => `'v${n}'`,
);
const ordinalPin = `const RULESET_ORDINAL = new Map(${ordinalLiteral});`;

const names = [
  // card locking
  'usesTurnEndingTrojan',
  'isWorkerCardPendingNextTurn',
  'normalizeCardsUsedThisTurn',
  'authoritativeCardsUsedThisTurn',
  // follow-up phase
  'ownerStartDuePly',
  'ownerStartsRemaining',
  'retainV6Turn',
  // probability
  'workerStableIndex',
  'stableHash',
  'stableIndex',
  'stableShuffle',
  // termination / policy gates
  'usesOctober27Balance',
  'usesNovemberExpansion',
  'usesRulesAtLeast',
  'usesV13Rules',
];

const contracts = names.map(declaration).join('\n');
const context = vm.createContext({ structuredClone });
new vm.Script(
  `${hashPins.join('\n')}\n${ordinalPin}\n${contracts}\nglobalThis.contracts={${names.join(',')}};`,
).runInContext(context, { timeout: 2000 });
const c = context.contracts;

// The snapshot's own catalog hash decides every hash-gated branch.
const catalog = JSON.parse(
  readFileSync(new URL('rulesets/augment_chess_20261102/cards/catalog.json', root), 'utf8'),
);
const HASH = catalog.source_catalog_hash;
const withHash = extra => ({ profile: { catalogHash: HASH }, ...extra });

// --- card locking -----------------------------------------------------------
// trojan-horse gained END_TURN in this snapshot; the gate must say so for our hash.
assert.equal(c.usesTurnEndingTrojan(withHash({})), true, 'turn-ending trojan applies here');

const pendingCard = { nextTurnPending: true };
const notPending = { nextTurnPending: false };
assert.equal(c.isWorkerCardPendingNextTurn(pendingCard), true);
assert.equal(c.isWorkerCardPendingNextTurn(notPending), false);
assert.equal(c.isWorkerCardPendingNextTurn(pendingCard, true), true);
// dev cards are exempt from the next-turn lock.
assert.equal(c.isWorkerCardPendingNextTurn({ nextTurnPending: true, devCard: true }), false);

// The per-turn budget is read from the card inventories, counted by the ply each card was used.
const livePosition = { turn: 'white', fullmoveNumber: 2 };
const liveCards = {
  schemaVersion: 2,
  inventories: {
    white: [{ attributes: { usedAtPly: 2 } }, { attributes: { usedAtPly: 9 } }],
    black: [{ attributes: { usedAtPly: 2 } }, { attributes: { usedAtPly: 2 } }],
  },
};
const counted = c.authoritativeCardsUsedThisTurn(liveCards, livePosition);
assert.equal(counted.white, 1, 'only cards used on the current ply count');
assert.equal(counted.black, 2);
// Without v2 inventories there is nothing authoritative to read.
const none = c.authoritativeCardsUsedThisTurn({}, livePosition);
assert.equal(none.white, 0);
assert.equal(none.black, 0);

// --- follow-up phase --------------------------------------------------------
// A declaration made now becomes due after `starts` owner starts, measured in half-plies.
const whiteMove1 = { turn: 'white', fullmoveNumber: 1 };
assert.equal(c.ownerStartDuePly(whiteMove1, 0), 0);
assert.equal(c.ownerStartDuePly(whiteMove1, 1), 2);
assert.equal(c.ownerStartDuePly(whiteMove1, 3), 6);
assert.equal(c.ownerStartDuePly(whiteMove1, 7), 14, 'ascension now waits seven owner starts');
// Remaining starts counts down to zero and never goes negative.
assert.equal(c.ownerStartsRemaining(whiteMove1, 'white', 6), 3);
assert.equal(c.ownerStartsRemaining(whiteMove1, 'white', 0), 0);

// Retaining a turn restores the actor and clears the en-passant window. A black actor that had
// already advanced the fullmove counter is rolled back so the ply does not double-count.
const retained = { turn: 'black', fullmoveNumber: 5, enPassantTarget: 'e3' };
c.retainV6Turn({ position: { fullmoveNumber: 4 } }, 'black', retained);
assert.equal(retained.turn, 'black', 'the actor keeps the turn');
assert.equal(retained.enPassantTarget, null, 'the en passant target is cleared');
assert.equal(retained.fullmoveNumber, 4, 'a black actor rolls the counter back one');

// An END_TURN card is queued rather than resolved inline; that helper needs the card-state
// plumbing, which is outside this scoped oracle, so it is not asserted here.

// --- probability ------------------------------------------------------------
// Stable index is a pure hash of the seed, so the same seed always picks the same entry.
assert.equal(c.workerStableIndex('omega:white:7:42', 2), c.workerStableIndex('omega:white:7:42', 2));
assert.equal(c.workerStableIndex('any', 1), 0, 'a single candidate needs no choice');
for (let length = 2; length <= 8; length += 1) {
  const index = c.workerStableIndex('seed:value', length);
  assert.ok(Number.isInteger(index) && index >= 0 && index < length, `index within ${length}`);
}

// The FNV-style hash and the index built on it are stable and bounded.
assert.equal(typeof c.stableHash('abc'), 'number');
assert.equal(c.stableHash('abc'), c.stableHash('abc'), 'the hash is deterministic');
assert.ok(c.stableHash('abc') >= 0, 'the hash is unsigned');
assert.equal(c.stableIndex('label', 0), 0, 'a zero bound yields index zero');
for (let bound = 1; bound <= 6; bound += 1) {
  const index = c.stableIndex('label', bound);
  assert.ok(index >= 0 && index < bound, `stable index within ${bound}`);
}
// A shuffle is a permutation: same multiset, deterministic for the same label.
const shuffled = c.stableShuffle([1, 2, 3, 4, 5], 'spot');
assert.deepEqual([...shuffled].sort((a, b) => a - b), [1, 2, 3, 4, 5], 'a shuffle permutes');
assert.deepEqual(
  [...c.stableShuffle([1, 2, 3, 4, 5], 'spot')],
  [...shuffled],
  'the same label shuffles the same way',
);

// --- termination ------------------------------------------------------------
// Both policy gates resolve from the snapshot hash, and this snapshot is the reform hash.
assert.equal(c.usesOctober27Balance(withHash({})), true, 'october 27 balance applies');
assert.equal(c.usesNovemberExpansion(withHash({})), true, 'november expansion applies');
// A snapshot without the hash field falls back to the explicit policy flag.
assert.equal(c.usesNovemberExpansion({ novemberExpansionPolicy: 1 }), true);
assert.equal(c.usesNovemberExpansion({}), false);

// The normalized per-turn card budget clamps junk input instead of trusting it.
// The values come from the vm realm, so compare plain field reads.
const clamped = c.normalizeCardsUsedThisTurn({ white: '2', black: -5 });
assert.equal(clamped.white, 2);
assert.equal(clamped.black, 0);
const empty = c.normalizeCardsUsedThisTurn(undefined);
assert.equal(empty.white, 0);
assert.equal(empty.black, 0);

console.log(
  'Snapshot transition contracts: turn-ending trojan, next-turn card lock, owner-start timers, ' +
    'stable probability index, balance/expansion gates and the per-turn card budget match the ' +
    '2026-11-02 bundle. Full interaction parity is not claimed.',
);
