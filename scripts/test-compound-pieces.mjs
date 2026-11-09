// Execute bounded helpers from the pinned source, never the application entry.
// Adapters cover state plumbing and disable unrelated movement effects.
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import vm from 'node:vm';
const root = new URL('../', import.meta.url);
const bytes = readFileSync(new URL('rulesets/augment_chess_20261017/sources/raw/main-RjrZ5fFP.js', root));
const sha = '2e932da7b9391effbeab3593ea5fdf855041fe5a74f611dfdfde678eee46beb8';
assert.equal(createHash('sha256').update(bytes).digest('hex'), sha);
const source = bytes.toString('utf8');
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
const names = ['COMPOUND_CARD_SPECS', 'COMPOUND_CARD_HELP', 'amazonSelectionContract', 'compoundCardTargets', 'applyAmazonV7', 'parrotBaseMoves'];
const adapters = `
const COLORS$n = ['white','black'];
const squareCoordinates$3 = s => /^[a-h][1-8]$/.test(s) ? {name:s,row:8-Number(s[1]),col:s.charCodeAt(0)-97} : null;
const squareName$7 = (r,c) => String.fromCharCode(97+c)+(8-r);
const selectedSquares = (a,n) => a.payload.selection.squares.length===n ? a.payload.selection.squares.map(squareCoordinates$3) : null;
const v5PieceAt = (s,cell) => { const p=s.entries.find(e=>e.square===cell); return p ? {...squareCoordinates$3(cell),instanceId:p.id,metadata:p} : null; };
const effectiveType$1 = p => p.typeOverride??p.type, cardIdentityType = effectiveType$1;
const v15KingAugmentRole = (s,c,p) => p.royal===true;
const october13Entries = (s,c) => s.entries.filter(p=>p.color===c).map(p=>({...squareCoordinates$3(p.square),metadata:p,identity:{instanceId:p.id}}));
const deepFreeze$i = x => x, rejection$1 = reason => ({ok:false,reason}), usesV13Rules = () => true;
const removeCells = (p,i,cells) => {const position=structuredClone(p),pieceIdentity=structuredClone(i); for(const c of cells){position.board[c.row][c.col]=null;pieceIdentity.board[c.row][c.col]=null;}return {position,pieceIdentity};};
const consumeCard = c => {const copy=structuredClone(c);copy.consumed=true;return copy;};
const syncEntityBoard = (c,p,i) => {const alive=new Set(i.board.flat().filter(Boolean).map(x=>x.instanceId)); for(const id of Object.keys(c.pieces))if(!alive.has(id))delete c.pieces[id]; return c;};
const grantAuthorityVigilance = () => {}, assertPieceIdentityState = i => i;
const finishV4CardAction = (s,a,o,action,patch,cause) => ({ok:true,...patch,cause});
const ORTH=[[-1,0],[1,0],[0,-1],[0,1]],DIAG=[[-1,-1],[-1,1],[1,-1],[1,1]],KING=[...ORTH,...DIAG];
const KNIGHT=[[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]];
const movementTypeName = t => t, injuryBlocksKnightLeap = () => false;
const usesSeptember22Rules = () => true, usesVipKingAlfilMovement = () => true;
`;
const context = vm.createContext({ structuredClone });
new vm.Script(adapters + names.map(declaration).join('\n') + '\nglobalThis.contracts={COMPOUND_CARD_SPECS,COMPOUND_CARD_HELP,amazonSelectionContract,compoundCardTargets,applyAmazonV7,parrotBaseMoves};').runInContext(context,{timeout:1000});
const c = context.contracts;
const action = squares => ({payload:{selection:{squares}}});
const entries = [
  {id:'n',square:'b1',color:'white',type:'knight'},
  {id:'b',square:'d4',color:'white',type:'bishop',attributes:{shield:true,captureCount:3}},
  {id:'r',square:'f4',color:'white',type:'rook',attributes:{shield:true,captureCount:3}},
  {id:'enemy',square:'b8',color:'black',type:'knight'},
];
for(const id of ['archbishop','chancellor']) {
  const base = id==='archbishop'?'d4':'f4', state={entries,cardState:{}};
  assert.deepEqual(Array.from(c.compoundCardTargets(state,'white',id,[]).squares),['b1']);
  assert.deepEqual(Array.from(c.compoundCardTargets(state,'white',id,['b1']).squares),[base]);
  assert.equal(c.compoundCardTargets(state,'white',id,[base]).ok,false);
  assert.equal(c.amazonSelectionContract(state,'white',action(['b1',base]),c.COMPOUND_CARD_SPECS[id].from).ok,true);
  for(const pair of [['b1','b1'],['b8',base],['b1','e4']]) assert.equal(c.amazonSelectionContract(state,'white',action(pair),c.COMPOUND_CARD_SPECS[id].from).ok,false);
  for(const square of ['b1',base]) {
    const royal={...state,entries:entries.map(p=>({...p,royal:p.square===square}))};
    assert.equal(c.amazonSelectionContract(royal,'white',action(['b1',base]),c.COMPOUND_CARD_SPECS[id].from).ok,false);
  }
  const full={entries,position:{board:Array.from({length:8},()=>Array(8).fill(null))},pieceIdentity:{board:Array.from({length:8},()=>Array(8).fill(null))},cardState:{gameId:'test',pieces:{}}};
  for(const p of entries) {const row=8-Number(p.square[1]),col=p.square.charCodeAt(0)-97;full.position.board[row][col]=p.type;full.pieceIdentity.board[row][col]={instanceId:p.id,moved:false};full.cardState.pieces[p.id]=structuredClone({...p,attributes:p.attributes??{}});}
  const out=c.applyAmazonV7(full,'white',{cardId:id,cardInstanceId:'c1'},action(['b1',base]),c.COMPOUND_CARD_SPECS[id].from,id);
  assert.equal(out.ok,true);assert.equal(out.pieceIdentity.board[7][1],null);
  const original=entries.find(p=>p.square===base),p=out.cardState.pieces[original.id];
  assert.equal(p.typeOverride,id);assert.equal(p.attributes.moved,true);assert.equal(p.attributes.shield,true);assert.equal(p.attributes.captureCount,3);
  assert.equal(p.attributes.freshNoCaptureUntil,undefined);
  assert.equal(out.pieceIdentity.board[8-Number(base[1])][base.charCodeAt(0)-97].instanceId,original.id);
}
const fixtures=[];
for(const kind of ['archbishop','chancellor']) for(const owner of ['white','black']) for(const from of ['a1','d4','h8']) for(const blocked of from==='d4'?[false,true]:[false]) {
  const opponent=owner==='white'?'black':'white';
  const blockers=blocked?[{square:'e4',owner},{square:'e5',owner},{square:'b4',owner:opponent},{square:'b6',owner:opponent},{square:'f5',owner:opponent}]:[];
  const pieces=new Map(blockers.map(p=>[p.square,{color:p.owner}]));
  const moves=c.parrotBaseMoves({memory:{type:kind},row:8-Number(from[1]),col:from.charCodeAt(0)-97,color:owner,at:(r,col)=>pieces.get(String.fromCharCode(97+col)+(8-r))??null,canCapture:p=>p.color!==owner,state:{}});
  const expected=Array.from(moves,m=>String.fromCharCode(97+m.col)+(8-m.row)).sort();
  fixtures.push({kind,owner,from,blockers,expected});
}
const expected={source_bundle_sha256:sha,scope:'Native compound rays and leaps without unrelated effects; generated from pinned parrotBaseMoves.',fixtures};
const fixturePath=new URL('crates/augment-core/tests/fixtures/effects-compound.json',root);
assert.ok(process.argv.slice(2).every(x=>x==='--write'));
if(process.argv.includes('--write')) {mkdirSync(new URL('.',fixturePath),{recursive:true});writeFileSync(fixturePath,JSON.stringify(expected,null,2)+'\n',{flag:'wx'});}
assert.deepEqual(JSON.parse(readFileSync(fixturePath,'utf8')),expected);
const view=readFileSync(new URL('crates/augment-app/src/web/view.rs',root),'utf8');
for(const id of ['archbishop','chancellor']) assert.ok(view.includes(JSON.stringify(c.COMPOUND_CARD_HELP[id])),id+' exact source help');
console.log('Compound pieces: pinned selection, noncapture transformation, identity/status preservation, help and 16 movement fixtures passed; full interaction parity unverified.');
