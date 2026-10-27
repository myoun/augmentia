import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
// Cheap wiring checks: no DOM, browser, or HTTP server.
const ui = readFileSync(new URL('../crates/augment-app/src/web/static/app.js', import.meta.url), 'utf8');
const html = readFileSync(new URL('../crates/augment-app/src/web/static/index.html', import.meta.url), 'utf8');
const elementNames = new Set([...ui.match(/const elements = \{([\s\S]*?)\n\};/)[1].matchAll(/^\s*(\w+):/gm)].map(m=>m[1]));
for (const [,name] of ui.matchAll(/elements\.(\w+)/g)) assert.ok(elementNames.has(name), `undefined elements.${name}`);
const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]));
for (const [,id] of ui.matchAll(/(?:getElementById\(["']|querySelector\(["']#)([\w-]+)["']\)/g)) assert.ok(ids.has(id), `missing HTML id ${id}`);
import { encodeAnalysisPath } from '../crates/augment-app/src/web/static/acg-export.js';
import { decodeAcgRecord, reconstructReplayFrame } from '../crates/augment-app/src/web/static/acg-codec.js';
import { acgBoardCells, acgAnalysisReadiness } from '../crates/augment-app/src/web/static/acg-view.js';
const node = (square) => ({id:0,label:'분석 수',kind:'move',actor:'white',moveNumber:1,
  board:[{square,piece:{id:1,kind:'unicorn',owner:'white',anchor:true,moved:true,origin:'b1'}}],
  turn:'black',ply:1,turnsTaken:{white:1,black:0},actionsRemaining:1,
  cards:{white:[],black:[]},rules:[],winner:null});
const a=node('b1'), b=node('c3'); b.id=1; b.cards.white=[{id:'amazon',instanceId:'c1',color:'white',used:false,usesRemaining:1,passiveApplied:false,nextTurnPending:false}]; b.rules=['acceleration'];
const before=JSON.stringify([a,b]);
const exported=await encodeAnalysisPath({nodes:[a,b]},'2026-09-18T00:00:00Z');
const decoded=await decodeAcgRecord(exported.text);
assert.equal(decoded.record.metadata.history,'Partial');
assert.equal(decoded.record.replay.events.length,1);
const final=reconstructReplayFrame(decoded.record.replay.baseFrame, decoded.record.replay.events,1);
assert.equal(final.board[5][2].type,'pegasus');
assert.equal(final.board[7][1],null);
assert.equal(final.cardState.white[0].id,'amazon');
assert.equal(final.appliedRuleCard.id,'acceleration');
assert.equal(JSON.stringify([a,b]),before);
const adapted=acgBoardCells(final,[{kind:'unicorn',name:'유니콘',description:'공식 설명',whiteSymbol:'U',blackSymbol:'U'}]);
assert.equal(adapted.find(c=>c.square==='c3').piece.kind,'unicorn');
assert.equal(adapted.find(c=>c.square==='c3').piece.description,'공식 설명');
const readiness = acgAnalysisReadiness(final,[{kind:'unicorn'}]);
assert.equal(readiness.boardCopyAllowed,true);
assert.equal(readiness.cardCount,1);
assert.equal(readiness.ruleCount,1);
assert.equal(acgAnalysisReadiness(final,[]).boardCopyAllowed,false);
const neutral = structuredClone(final); neutral.board[5][2].color='neutral';
assert.equal(acgAnalysisReadiness(neutral,[{kind:'unicorn'}]).boardCopyAllowed,false);
const wide = structuredClone(final); wide.board[0].push(null);
assert.equal(acgAnalysisReadiness(wide,[{kind:'unicorn'}]).boardCopyAllowed,false);
console.log('ACG export, adapter, and safe board-copy diagnostics passed (offline).');
