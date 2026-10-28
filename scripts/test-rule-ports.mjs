// Pinned pure rule helpers; never load the application entry or UI.
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import vm from 'node:vm';
const root = new URL('../', import.meta.url);
const bytes = readFileSync(new URL('rulesets/augment_chess_20261017/sources/raw/main-RjrZ5fFP.js',root));
const sha = createHash('sha256').update(bytes).digest('hex');
assert.equal(sha,'2e932da7b9391effbeab3593ea5fdf855041fe5a74f611dfdfde678eee46beb8');
const source = bytes.toString('utf8');
const definitions = new Map();
function declaration(name) {
  const escaped=name.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
  const match=new RegExp(`(?:^|\\n)(function ${escaped}\\(|class ${escaped} |(?:const|let) ${escaped} = )`).exec(source);
  assert.ok(match,`missing pure dependency ${name}`);
  const start=match.index+(source[match.index]==='\n'?1:0);
  const fn=source.startsWith('function ',start)||source.startsWith('class ',start);
  let end=start;
  for(let i=0;i<10000;i++) {
    end=source.indexOf(fn?'\n':';',end+1);
    if(end<0||end-start>100000)break;
    const code=source.slice(start,end+(fn?0:1));
    try{new vm.Script(code);}catch{continue;}
    assert.doesNotMatch(code,/\b(?:eval|Function|fetch|XMLHttpRequest|document|window|process|require|import)\b|\.constructor\b/);
    return {start,code};
  }
  throw new Error(`Unrecognized pure dependency ${name}`);
}
const expression=`JSON.stringify({
 source_sha256:${JSON.stringify(sha)},
 chess960:Array.from({length:960},(_,i)=>chess960BackRank(i)),
 diagonal:V9_DIAGONAL_LAYOUT,
 queen_directions:QUEEN_DIRECTIONS,
 chaos_values:CHESS_N_POW_30_PIECE_VALUES,
 don_routes:[{from:{row:7,col:0},targets:[{row:0,col:7}]},{from:{row:4,col:4},targets:[{row:0,col:0},{row:7,col:7}]},{from:{row:3,col:3},targets:[{row:0,col:7},{row:7,col:0},{row:6,col:6}]}].map(input=>({...input,path:shortestDonQuixoteRampage(input).path})),
 trickster_types:tricksterMovementTypes({cardState:{profile:{catalogHash:REFORM_CATALOG_HASH}}}),
 chaos_home_weights:Object.fromEntries(Object.keys(CHESS_N_POW_30_PIECE_VALUES).map(type=>[type,chessNPow30HomeRankWeight(type)])),
 combat_values:ENCYCLOPEDIA_PIECE_VALUES,
 minor_types:Object.keys(CHESS_N_POW_30_PIECE_VALUES).filter(t=>isMinorPieceTypeForCatalog(t,{cardState:{profile:{catalogHash:REFORM_CATALOG_HASH}}})),
 major_types:Object.keys(CHESS_N_POW_30_PIECE_VALUES).filter(t=>isMajorPieceTypeForCatalog(t,{cardState:{profile:{catalogHash:REFORM_CATALOG_HASH}}})),
 mistake_immobile:[...MISTAKE_IMMOBILE_TYPES],
 chaos:Array.from({length:12},(_,seed)=>({seed,...generateChessNPow30Setup(v19SeededRandom(seed))})),
 acceleration:[0,2,7].flatMap(black=>['white','black'].map(turn=>({black,turn,due:accelerationStartAfterBlackTurns(black,turn)*2+1}))),
 collapse:Array.from({length:5},(_,depth)=>collapseRingCells(8,8,depth)),
 crown:Array.from({length:12},(_,turn)=>advanceCrownHoldEntry({holdingColor:'white',heldMoves:{white:0,black:0},lastCountedMove:0},'white',turn))
})`;
for(const name of ['chess960BackRank','generateChessNPow30Setup','v19SeededRandom','accelerationStartAfterBlackTurns','collapseRingCells','advanceCrownHoldEntry'])definitions.set(name,declaration(name));
let result;
for(let attempt=0;attempt<250;attempt++) {
  const program=[...definitions.values()].sort((a,b)=>a.start-b.start).map(x=>x.code).join('\n')+'\n'+expression;
  try{result=JSON.parse(vm.runInNewContext(program,{}, {timeout:3000}));break;}
  catch(error){
    const missing=/^(\w[\w$]*) is not defined$/.exec(error.message);
    if(error.name!=='ReferenceError'||!missing)throw error;
    assert.ok(!definitions.has(missing[1]),missing[1]);
    definitions.set(missing[1],declaration(missing[1]));
  }
}
assert.ok(result);
const fixture=new URL('crates/augment-core/tests/fixtures/current-rules.json',root);
if(process.argv.includes('--write'))writeFileSync(fixture,JSON.stringify(result,null,2)+'\n');
assert.deepEqual(result,JSON.parse(readFileSync(fixture,'utf8')));
console.log(`Rule helpers: 960 layouts, ${result.chaos.length} seeded armies, ${Object.keys(result.chaos_values).length} chaos types, timers and collapse/crown fixtures passed.`);
