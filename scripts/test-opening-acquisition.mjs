// Pinned original acquisition control flow and layout helpers, not full game parity.
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
  'OPENING_CONVERTED_PASSIVE_IDS', 'OPENING_RANDOM_TARGET_IDS', 'OPENING_PASSIVE_DESCRIPTIONS',
  'openingPassiveDefinition', 'openingPassiveSkipsFreshCapture', 'openingPassiveSettlementPriority',
  'septemberBigBishopOpening', 'londonLayout', 'falseStartMoves',
  'V15_ROULETTE_MAJOR_TYPES', 'AUTHORITATIVE_RANDOM_ROULETTE_MAJOR_TYPES_V19', 'V19_ROULETTE_MAJOR_TYPES', 'KILLER_MAJOR_TYPES',
  'OPENING_PASSIVE_APPLICATION_STATES', 'OPENING_PASSIVE_APPLICATION_CARDS', 'openingPassiveAction', 'applyOpeningPassiveAcquisition',
];
// Only the source's candidate choice/RNG orchestration is tested here. The game
// action is an adapter; Rust tests independently cover removal and delayed return.
const adapters = `
const AUTHORITATIVE_CARD_RULE_STATE_V2_SCHEMA_VERSION = 2;
let targetCount = 0, calls = 0, applied = 0;
const v8Pieces = () => Array.from({length:targetCount},(_,i)=>({name:'p'+i}));
const v12TakeAcquisitionRandom = (context,purpose,maxExclusive) => { calls++; return 0; };
const applyOtherworldV11 = (world,actor,owned,action,values) => { applied++; return {ok:true,state:world}; };
`;
const checks = `
const results = [0,1,2].map(count => {
  targetCount=count; calls=0; applied=0;
  applyOpeningPassiveAcquisition({position:{},pieceIdentity:{},cardState:{profile:{rulesetVersion:'v19'}}},
    'white','otherworld',{cardInstanceId:'c1'},{},null,true);
  return {count,calls,applied};
});
const entries = [
  {row:6,col:5,item:{color:'white'},cells:[{row:6,col:5},{row:7,col:5},{row:6,col:6},{row:7,col:6}]},
  {row:5,col:5,item:{color:'white'}}, {row:5,col:6,item:{color:'white'}},
  {row:5,col:7,item:{color:'white'}}, {row:3,col:6,item:{color:'black'}},
  {row:6,col:1,item:{color:'white'}}
];
const plan = falseStartMoves(entries,'white',8,8,(row,col)=>row===5&&col===1)
  .map(e=>[e.row,e.col,e.to.row,e.to.col]);
JSON.stringify({results,plan,white:londonLayout('white'),black:londonLayout('black'),
  bishop:septemberBigBishopOpening('white'),hordeBishop:septemberBigBishopOpening('black','pawn'),
  skips:['london-system','big-bishop','otherworld'].map(openingPassiveSkipsFreshCapture),
  priorities:['london-system','big-bishop','false-start'].map(openingPassiveSettlementPriority),
  definition:openingPassiveDefinition({id:'false-start',enabled:true,draftCategory:'OPENING',activation:'ACTIVE'}),
  majors:[...KILLER_MAJOR_TYPES]});
`;
const context = vm.createContext({}, { codeGeneration: { strings: false, wasm: false } });
const result = JSON.parse(new vm.Script(adapters + names.map(declaration).join('\n') + checks).runInContext(context, { timeout: 1000 }));
assert.deepEqual(result.results, [{count:0,calls:0,applied:0},{count:1,calls:0,applied:1},{count:2,calls:1,applied:1}]);
assert.deepEqual(result.plan, [[6,5,5,5],[5,5,4,5],[5,6,4,6],[5,7,3,7]]);
assert.equal(result.white.length,16); assert.equal(result.black.length,16);
assert.deepEqual(result.white.find(([square])=>square==='d2'),['d2','N']);
assert.deepEqual(result.black.find(([square])=>square==='d7'),['d7','N']);
assert.equal(result.bishop.anchor,'f2'); assert.equal(result.hordeBishop.anchor,'f8');
assert.deepEqual(result.hordeBishop.placements.at(-1),['h6','pawn',true]);
assert.deepEqual(result.skips,[true,true,false]); assert.deepEqual(result.priorities,[0,2,4]);
assert.equal(result.definition.activation,'PASSIVE'); assert.equal(result.definition.turnPolicy,'NOT_APPLICABLE');
for (const kind of ['octopus','man','big-bishop','queen','rook']) assert.ok(result.majors.includes(kind));
for (const kind of ['wizard','recruiter','trickster','crown','king']) assert.ok(!result.majors.includes(kind));

// The current d4 passive applies both to the opponent and to its owner's royal.
const d4helpers = ['expansionNamedSquare', 'expansionDestinationAllowed'].map(declaration).join('\n');
const d4result = JSON.parse(new vm.Script(`
 const inside$1=(board,r,c)=>r>=0&&r<8&&c>=0&&c<8;
 const opponent$2=c=>c==='white'?'black':'white';
 ${d4helpers}
 const board=Array.from({length:8},()=>Array(8).fill(null));
 JSON.stringify(['white','black'].flatMap(color=>[false,true].map(ownKing=>expansionDestinationAllowed({board,color,cells:[{row:4,col:3}],d4:{white:true,black:false},ownKing}))))
`).runInNewContext({}, {timeout:1000,contextCodeGeneration:{strings:false,wasm:false}}));
assert.deepEqual(d4result,[true,false,false,false]);

// The source retains earlier protection and leaves later timed protection active
// when a non-promotion transformation removes the gambit status.
const protectionResult = JSON.parse(new vm.Script(`
 ${declaration('clearQueensGambitProtectionAfterTransformation')}
 const prior={protected:true,queensGambitProtection:true,queensGambitPreviousProtected:true};
 const timed={protected:true,queensGambitProtection:true,lastResistance:{previousProtected:true}};
 const plain={protected:true,queensGambitProtection:true};
 [prior,timed,plain].forEach(clearQueensGambitProtectionAfterTransformation);
 JSON.stringify({prior,timed,plain});
`).runInNewContext({}, {timeout:1000,contextCodeGeneration:{strings:false,wasm:false}}));
assert.deepEqual(protectionResult,{prior:{protected:true},timed:{protected:true,lastResistance:{previousProtected:false}},plain:{}});

const thiefResult = JSON.parse(new vm.Script(`
 const usesThiefRemake=()=>true;
 ${['thiefMoveDirection','thiefMoveAllowed','noteThiefMove'].map(declaration).join('\n')}
 const item={type:'thief'}, from={row:4,col:3}, to={row:2,col:3};
 noteThiefMove(item,from,to,{});
 JSON.stringify({back:thiefMoveAllowed(item,{row:5,col:3},{},to),
 visited:thiefMoveAllowed(item,from,{},to),
 sideways:thiefMoveAllowed(item,{row:2,col:2},{},to)});
`).runInNewContext({}, {timeout:1000,contextCodeGeneration:{strings:false,wasm:false}}));
assert.deepEqual(thiefResult,{back:false,visited:false,sideways:true});

const fractureResult = JSON.parse(new vm.Script(`
 ${['TYPES$1','isFracturableLarge','isFracturedLarge','largeBodyCells','damageLargeBodyBoardCells'].map(declaration).join('\n')}
 const result=[];
 for(let hit=0;hit<4;hit++) {
   const board=Array.from({length:8},()=>Array(8).fill(null));
   const item={id:'r',type:'bigRook',largeMissingCells:[],anchorRow:2,anchorCol:3};
   for(let i=0;i<4;i++) board[2+(i>>1)][3+(i&1)]=item;
   const first=damageLargeBodyBoardCells(board,item,[{row:2+(hit>>1),col:3+(hit&1)}]);
   const next=(hit+1)%4;
   const second=damageLargeBodyBoardCells(board,item,[{row:2+(next>>1),col:3+(next&1)}]);
   result.push({first:first.defeated,missing:item.largeMissingCells,second:second.defeated});
 }
 JSON.stringify(result);
`).runInNewContext({}, {timeout:1000,contextCodeGeneration:{strings:false,wasm:false}}));
assert.deepEqual(fractureResult,[0,1,2,3].map(hit=>({first:false,missing:[hit],second:true})));

const castles = JSON.parse(new vm.Script(`
 ${declaration('v10BigRookCastleAnchor')}
 JSON.stringify(['white','black'].flatMap(color=>[2,6].map(col=>v10BigRookCastleAnchor(color,col))));
`).runInNewContext({}, {timeout:1000,contextCodeGeneration:{strings:false,wasm:false}}));
assert.deepEqual(castles,[{row:6,col:3},{row:6,col:4},{row:0,col:3},{row:0,col:4}]);
console.log('Opening source checks: acquisition RNG, layouts, d4, protection, thief path, large-body fracture and castling passed.');
