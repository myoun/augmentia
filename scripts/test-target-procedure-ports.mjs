// Pinned target/procedure source checks. Adapters replace state plumbing only;
// these tests do not execute the app entry or establish full interaction parity.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import vm from 'node:vm';
const root = new URL('../rulesets/augment_chess_20261017/', import.meta.url);
const bytes = readFileSync(new URL('sources/raw/main-RjrZ5fFP.js', root));
assert.equal(createHash('sha256').update(bytes).digest('hex'), '2e932da7b9391effbeab3593ea5fdf855041fe5a74f611dfdfde678eee46beb8');
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
const names = [
  'BASE_MINOR_PIECE_TYPES', 'RECLASSIFIED_MINOR_PIECE_TYPES', 'EXPANDED_MINOR_PIECE_TYPES', 'NATIVE_MINOR_PIECE_TYPES',
  'minorPieceTypesForCatalog', 'isMinorPieceTypeForCatalog', 'canonicalMinorPieceTypesForCatalog',
  'V7_SELECTED_TRANSFORMS', 'selectedTransformSpec', 'selectedTransformContractV7', 'reversalSelectionContract',
  'checkerSummonCells', 'poisonedPawnCandidates', 'applyPoisonedPawn', 'applyPoisonCapture', 'tickPoisonStuns',
  'applyV19CardAction', 'V5_TURN_COLOR_EFFECTS',
];
const adapters = `
const usesV19Rules = () => true, usesOctober13Reform = () => true, usesSeptember26Rebalance = () => true,
  usesSeptember18Balance = () => true, usesNativePieceClassification = () => true;
const usesOctober13Cards = () => false, usesTemporaryHorseRiding = () => true;
const SEPTEMBER26_TRANSFORMS = {};
const legacyReportOct22Depth = 0, legacyOctober1RulesDepth = 0, legacySourceBindingsDepth = 0;
const COLORS$n = ['white','black'];
const effectiveType$1 = p => p.typeOverride ?? p.type;
const cardIdentityType = effectiveType$1;
const statusHas$1 = (p,s) => p?.statuses?.includes(s) === true;
const withStatus = (p,s) => ({...p,statuses:[...new Set([...(p.statuses??[]),s])]});
const withoutStatus = (p,s) => ({...p,statuses:(p.statuses??[]).filter(x=>x!==s)});
const v15KingAugmentRole = (world,cards,p) => p?.royal===true || p?.type==='king';
const opposite$2 = c => c==='white'?'black':'white';
const sourceCardInstance = () => 'c1';
const rejection$1 = code => ({ok:false,code});
const selectedSquares = (a,n) => a.squares.length===n ? a.squares.map(name=>({name,col:name.charCodeAt(0)-97,row:8-Number(name.slice(1))})) : null;
const v5PieceAt = (w,s) => w.bySquare?.[typeof s==='string'?s:s.name] ?? null;
const colorEffect$2 = cards => cards.reversalActive===true;
const v8Pieces = (w,predicate) => w.entries.filter(e=>predicate(e.metadata));
const v4Pieces = (w,predicate) => Object.entries(w.bySquare).map(([name,e])=>({...e,identity:{instanceId:e.instanceId},...selectedSquares({squares:[name]},1)[0]}))
  .filter(e=>predicate(e.metadata,e.row,e.col,e.identity));
const v5ColorEffectAction = (w,a,o,action,owner,kind,attributes) => ({kind,attributes});
const consumeCard = cards => structuredClone(cards);
const finishCardAction = (w,a,o,action,cards) => ({ok:true,cardState:cards});
`;
const checks = `
const world = {cardState:{pieces:{}},pieceIdentity:{board:Array.from({length:8},()=>Array(8).fill(null))},bySquare:{}};
function put(square,id,type,color='white',royal=false) {
 const cell=selectedSquares({squares:[square]},1)[0], metadata={type,color,royal,attributes:{},statuses:[]};
 world.cardState.pieces[id]=metadata; world.pieceIdentity.board[cell.row][cell.col]={instanceId:id};
 world.bySquare[square]={metadata,instanceId:id};
}
put('b1','n','knight'); put('c1','b','bishop'); put('a1','r','rook'); put('d1','q','queen');
put('e1','k','king'); put('f1','w','wizard'); put('g1','royal','wizard','white',true); put('h1','enemy','bishop','black');
const reversal = ['b1','f1','a1','d1','e1','g1','h1'].map(square=>reversalSelectionContract(world,'white',{squares:[square]}).ok);
const ordination=selectedTransformSpec(world,'ordination'), pegasus=selectedTransformSpec(world,'pegasus');
const transforms = [
 selectedTransformContractV7(world,'white','ordination',{squares:['c1']},ordination).ok,
 selectedTransformContractV7(world,'white','pegasus',{squares:['a1']},pegasus).ok,
 selectedTransformContractV7(world,'white','pegasus',{squares:['b1']},pegasus).ok
];
const horse=applyV19CardAction(world,'white',{cardId:'horse-riding'},{squares:[]},[],{});
const entries = Array.from({length:6},(_,i)=>({identity:{instanceId:'p'+i},metadata:{color:'white',type:'pawn',attributes:{},statuses:i===0?['poisoned-pawn']:[]}}));
const poisonWorld={entries,cardState:{pieces:Object.fromEntries(entries.map(e=>[e.identity.instanceId,e.metadata]))}};
const poisoned=applyPoisonedPawn(poisonWorld,'white',{cardInstanceId:'c1'},{squares:[]},[2,0,1]);
const selected=Object.entries(poisoned.cardState.pieces).filter(([id,p])=>id!=='p0'&&statusHas$1(p,'poisoned-pawn')).map(([id])=>id).sort();
const singleton={entries:entries.slice(1,2),cardState:poisonWorld.cardState};
const missingDraw=applyPoisonedPawn(singleton,'white',{cardInstanceId:'c1'},{squares:[]},[]).ok;
const pieces={attacker:{type:'rook',color:'black',statuses:[],attributes:{}},victim:{type:'pawn',color:'white',statuses:['poisoned-pawn'],attributes:{}}};
let stunned=applyPoisonCapture({pieces},'black','attacker','victim',{cardState:{pieces}});
const durations=[stunned.pieces.attacker.attributes.poisonStunTurns];
for(let i=0;i<3;i++){stunned=tickPoisonStuns(stunned,'black');durations.push(stunned.pieces.attacker.attributes.poisonStunTurns??0);}
const friendly=structuredClone(pieces);friendly.victim.color='black';
const friendlyStun=applyPoisonCapture({pieces:friendly},'black','attacker','victim',{cardState:{pieces:friendly}}).pieces.attacker.statuses;
JSON.stringify({reversal,transforms,ordination,pegasus,horse,horseTimed:V5_TURN_COLOR_EFFECTS.has('horse-riding'),selected,missingDraw,durations,friendlyStun,
 white:checkerSummonCells(Array.from({length:8},()=>Array(8).fill(null)),'white'),
 black:checkerSummonCells(Array.from({length:8},()=>Array(8).fill(null)),'black')});
`;
const context = vm.createContext({structuredClone}, {codeGeneration:{strings:false,wasm:false}});
const result = JSON.parse(new vm.Script(adapters + names.map(declaration).join('\n') + checks).runInContext(context,{timeout:1000}));
assert.deepEqual(result.reversal,[true,true,false,false,false,false,false]);
assert.deepEqual(result.transforms,[true,true,false]);
assert.deepEqual(result.horse,{kind:'horse-riding',attributes:{remaining:1}}); assert.equal(result.horseTimed,true);
assert.deepEqual(result.ordination.from,['bishop']); assert.deepEqual(result.pegasus.from,['rook']);
assert.deepEqual(result.selected,['p1','p3','p4']); assert.equal(result.missingDraw,false);
assert.deepEqual(result.durations,[3,2,1,0]); assert.deepEqual(result.friendlyStun,[]);
assert.deepEqual(result.white,[{row:5,col:0},{row:5,col:1},{row:5,col:6},{row:5,col:7}]);
assert.deepEqual(result.black,[{row:2,col:0},{row:2,col:1},{row:2,col:6},{row:2,col:7}]);
console.log('Target/procedure ports: pinned native-minor sacrifice, one-bishop/rook transforms, checker cells, poison sampling and stun source contracts passed; full interaction parity unverified.');
