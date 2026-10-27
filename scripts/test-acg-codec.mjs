#!/usr/bin/env node
// Node 22+: node scripts/test-acg-codec.mjs [path/to/real.acg]
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { decodeAcgRecord, encodeAcgRecord, reconstructReplayFrame } from '../crates/augment-app/src/web/static/acg-codec.js';

const fixture = process.argv[2] || String.raw`C:\Users\myoun\.dsh\attachments\v1\files\af\afdd96689bde2fef1e127c59813cebf36d1b51de4b687b6678729eeef9595768\augmentchess_260918_0411.acg`;
const text = await readFile(fixture, 'utf8');
const document = await decodeAcgRecord(text);
const { record, hash } = document;
assert.equal(hash, '0589ca402e7e6c6cddd6992ff4780eef87f957f638a79afd48aac246dcf3f706');
assert.equal(record.version, 2);
assert.equal(record.replay.events.length, 62);
assert.equal(record.replay.notationTimeline.length, 61);
assert.equal(record.metadata.white, 'myoun');
assert.equal(record.metadata.black, '아라안다고');
assert.equal(record.metadata.result, '1-0');
const marker = '\n\n%%ACG-DATA 2\n';
const hashMarker = '\n%%ACG-SHA256\n';
const payload = text.slice(text.indexOf(marker) + marker.length, text.indexOf(hashMarker)).trim();
assert.equal(createHash('sha256').update(payload, 'utf8').digest('hex'), hash);
assert.ok(Array.isArray(JSON.parse(payload)), 'ACG v2 is a compact tuple, not a JSON wrapper');

// Independently extract the unmodified official dependencies as a compatibility oracle.
// This imports only trusted repository code; the reference bundle is never written.
const bundle = await readFile(new URL('../rulesets/augment_chess_20260915/sources/raw/main-6qeAcxbX.js', import.meta.url), 'utf8');
function slice(start, end) {
  const from = bundle.indexOf(start);
  assert.ok(from >= 0, `reference start missing: ${start}`);
  const to = bundle.indexOf(end, from);
  assert.ok(to > from, `reference end missing: ${end}`);
  return bundle.slice(from, to);
}
const oracleSource = slice('const SUSPICIOUS_POTION_EFFECTS =', 'function suspiciousPotionEffectsForPiece')
  + slice('function cloneObject(value, seen)', 'function retainRecentReplaySnapshots')
  + slice('const ACG_DATA_MARKER =', 'function pad2(value)')
  + '\nexport { decodeAcgRecord, encodeAcgRecord, reconstructReplayFrame };';
const official = await import(`data:text/javascript;base64,${Buffer.from(oracleSource).toString('base64')}`);
assert.deepEqual(document, await official.decodeAcgRecord(text));
const encoded = await encodeAcgRecord(record);
assert.deepEqual(encoded, await official.encodeAcgRecord(record));
assert.deepEqual(await decodeAcgRecord(encoded.text), await official.decodeAcgRecord(encoded.text));
assert.deepEqual(encoded.record, record);
assert.deepEqual((await encodeAcgRecord(encoded.record)).text, encoded.text);

const sample = reconstructReplayFrame(record.replay.baseFrame, record.replay.events, 3);
assert.equal(sample.turn, 'black');
assert.equal(sample.moveCount, 1);
assert.equal(sample.board[6][4], null);
assert.deepEqual(sample.board[4][4], { id: 'pk', color: 'white', type: 'pawn', moved: true, origin: 'e2', coolGuyCapturedLast: false });
for (let i = 0; i <= record.replay.events.length; i++) {
  assert.deepEqual(reconstructReplayFrame(record.replay.baseFrame, record.replay.events, i), official.reconstructReplayFrame(record.replay.baseFrame, record.replay.events, i));
}
assert.equal(reconstructReplayFrame(record.replay.baseFrame, record.replay.events).winner, 'white');
sample.board[4][4].type = 'mutated';
assert.equal(record.replay.events[2].delta.board.cells[0].after.type, 'pawn');
assert.deepEqual(reconstructReplayFrame(record.replay.baseFrame, record.replay.events, -10), record.replay.baseFrame);
assert.deepEqual((await decodeAcgRecord('\uFEFF' + text.replace(/\n/g, '\r\n'))).record, record);

await assert.rejects(decodeAcgRecord(text.replace('myoun', 'tampered')), /display notation/);
await assert.rejects(decodeAcgRecord(text.replace(payload, payload + ' ' + '\n' + '[]')), /SHA-256 mismatch/);
// Whitespace outside the payload is ignored by the official decoder; internal raw
// JSON whitespace must be hashed exactly, never hash JSON.stringify(JSON.parse()).
const pretty = JSON.stringify(JSON.parse(payload), null, 1);
const prettyText = text.replace(payload, pretty).replace(hash, createHash('sha256').update(pretty).digest('hex'));
assert.deepEqual((await decodeAcgRecord(prettyText)).record, record);
await assert.rejects(decodeAcgRecord(text.replace(payload, pretty)), /SHA-256 mismatch/);
await assert.rejects(decodeAcgRecord(JSON.stringify(record)), /markers/);
await assert.rejects(decodeAcgRecord(text.replace(hash, 'A'.repeat(64))), /SHA-256/);
await assert.rejects(decodeAcgRecord('x'.repeat(32 * 1024 * 1024 + 1)), /too large/);
await assert.rejects(decodeAcgRecord(text, { allowedPieceTypes: ['pawn'] }), /unknown piece type/);

function repack(mutator) {
  const compact = JSON.parse(payload);
  mutator(compact);
  const changed = JSON.stringify(compact);
  return text.replace(payload, changed).replace(hash, createHash('sha256').update(changed).digest('hex'));
}
await assert.rejects(decodeAcgRecord(repack(c => c[0].push('__proto__'))), /dictionary key/);
await assert.rejects(decodeAcgRecord(repack(c => c[0].push('constructor'))), /dictionary key/);
await assert.rejects(decodeAcgRecord(repack(c => c[3][3][3].push('__proto__', [0]))), /frame key/);
await assert.rejects(decodeAcgRecord(repack(c => c[3][3][3].push('custom', [0, 'prototype', null]))), /object key/);
await assert.rejects(decodeAcgRecord(repack(c => c[3][0] = '@zzzzzz')), /reference is out of range/);
await assert.rejects(decodeAcgRecord(repack(c => c[3][4][0][2] = 3)), /event color/);
await assert.rejects(decodeAcgRecord(repack(c => c[3][3][2].push(c[3][3][2][0], null))), /duplicate compact board position/);
await assert.rejects(decodeAcgRecord(repack(c => c[1].push('https://evil.invalid'))), /external URL/);
await assert.rejects(decodeAcgRecord(repack(c => { c[2][0] = '^0'; c[3][3][3].push('custom', '^0'); })), /nested compact structure reference/);
await assert.rejects(decodeAcgRecord(repack(c => {
  const index = c[2].length;
  c[2].push([1, ...Array(12000).fill(0)]);
  c[3][3][3].push('custom', [1, ...Array(64).fill('^' + index.toString(36))]);
})), /too many values|too many expanded/);
await assert.rejects(decodeAcgRecord(repack(c => {
  const event = c[3][4][0];
  while (event.length < 7) event.push(0);
  event[6] = ['custom', [4, 80001]];
})), /array patch length/);
const bad = structuredClone(record);
bad.replay.events[0].delta.fields.push({ key: '__proto__', after: {}, beforeExists: false });
await assert.rejects(encodeAcgRecord(bad), /replay field/);
assert.throws(() => reconstructReplayFrame(bad.replay.baseFrame, bad.replay.events, 1), /replay field/);
assert.equal({}.polluted, undefined);

// Exercise genuine compact piece, object/array patches, field deletion, board
// resizing, literal @/^ strings, initial timeline entries, and synthesized visuals.
const tiny = {
  format: 'augmentchess-acg', version: 2, recordId: '@literal', createdAt: '2026-09-17T00:00:00Z',
  metadata: { white: '^literal', result: '*', board: '2x2' },
  replay: { baseFrame: { board: [[{ id: 'original', type: 'pawn', color: 'white', moved: false }, null], [null, null]], custom: { keep: Array(30).fill(0), deleteMe: 1 }, gone: true }, events: [], notationTimeline: [] }
};
const before = tiny.replay.baseFrame.custom;
const after = { keep: Array(30).fill(0), added: '@literal' };
after.keep[4] = 9;
tiny.replay.events.push({ id: 'ignored', label: 'test', moveNumber: 1, color: 'white', notations: [], visuals: [{ type: 'card', animation: { id: 'test-card', name: 'Card', color: 'white' } }], delta: { board: { before: { rows: 2, cols: 2 }, after: { rows: 1, cols: 1 }, cells: [] }, fields: [{ key: 'custom', beforeExists: true, afterExists: true, before, after }, { key: 'gone', beforeExists: true, afterExists: false, before: true, after: null }] } });
const tinyEncoded = await encodeAcgRecord(tiny);
assert.deepEqual(tinyEncoded, await official.encodeAcgRecord(tiny));
const tinyFrame = reconstructReplayFrame(tinyEncoded.record.replay.baseFrame, tinyEncoded.record.replay.events);
assert.equal(tinyFrame.board.length, 1);
assert.equal(tinyFrame.board[0].length, 1);
assert.deepEqual(tinyFrame.custom, after);
assert.ok(!Object.hasOwn(tinyFrame, 'gone'));
assert.equal(tinyEncoded.record.replay.baseFrame.board[0][0].id, 'p0');
assert.equal(tinyEncoded.record.replay.events[0].visuals[0].animation.nonce, 'acg-1-1');
console.log('PASS: real fixture, exact raw SHA-256/human text, official cross-compatibility, all 63 frames, patches/resizing, malformed/security bounds');
console.log(JSON.stringify({ hash, events: record.replay.events.length, sampleFrameIndex: 3, turn: 'black', moveCount: 1, e4: record.replay.events[2].delta.board.cells[0].after, e2: null }, null, 2));
