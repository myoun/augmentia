// Pinned current piece-card contracts; only isolated source helpers are evaluated.
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import vm from 'node:vm';
const root=new URL('../',import.meta.url);
const bytes=readFileSync(new URL('rulesets/augment_chess_20261017/sources/raw/main-RjrZ5fFP.js',root));
const sha=createHash('sha256').update(bytes).digest('hex');
assert.equal(sha,'2e932da7b9391effbeab3593ea5fdf855041fe5a74f611dfdfde678eee46beb8');
const source=bytes.toString('utf8');

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

function run(expression,adapters='') {
 const definitions=new Map();
 for(let attempt=0;attempt<400;attempt++) {
  const program=[...definitions.values()].sort((a,b)=>a.start-b.start).map(x=>x.code).join('\n')+'\n'+adapters+'\nJSON.stringify('+expression+');';
  try{return JSON.parse(vm.runInNewContext(program,{structuredClone},{timeout:3000,contextCodeGeneration:{strings:false,wasm:false}}));}
  catch(error){const missing=/^([\w$]+) is not defined$/.exec(error.message);if(error.name!=='ReferenceError'||!missing)throw error;assert.ok(!definitions.has(missing[1]),missing[1]);definitions.set(missing[1],declaration(missing[1]));}
 }
 throw new Error('Unbounded source dependency graph');
}
const worldAdapter=`
function world(entries) {
 const state={rulesetVersion:AUTHORITATIVE_CARD_RULESET_V19_VERSION$2,position:{board:Array.from({length:8},()=>Array(8).fill(null)),turn:'white',fullmoveNumber:1,status:'active'},pieceIdentity:{board:Array.from({length:8},()=>Array(8).fill(null))},cardState:{board:{installations:Array.from({length:8},()=>Array.from({length:8},()=>[]))},profile:{catalogHash:REFORM_CATALOG_HASH,rulesetVersion:AUTHORITATIVE_CARD_RULESET_V19_VERSION$2},pieces:{},effects:{global:[],colors:{white:[],black:[]}}}};
 for(const [i,p] of entries.entries()) {const row=8-Number(p.square[1]),col=p.square.charCodeAt(0)-97,id='piece'+i;state.position.board[row][col]=(p.color==='white'?'w':'b')+'Q';state.pieceIdentity.board[row][col]={instanceId:id,moved:false};state.cardState.pieces[id]={color:p.color,baseType:'queen',typeOverride:p.type,statuses:[],attributes:{...p.royal?{editorRoyal:true}:{}}};}
 return state;
}
const action=squares=>({payload:{selection:{kind:squares.length?'squares':'none',squares,choice:null}}});
`;
const selected=run(`(()=>{
 const cards=['barrier','beacon','campfire','constitutional-monarchy','don-quixote','eastern-policy','idol','jester','killer','medium','reaper','revolving-door','transmission-tower','wizard'];
 const types=['queen','rook','bishop','knight','camel','clockwork','parrot','pawn','king','grappler','revolving-door','don-quixote','medium','killer','beacon','transmission-tower','barrier'];
 return cards.flatMap(card=>types.flatMap(type=>['white','black'].flatMap(color=>[false,true].map(royal=>{
  const s=world([{square:'d4',type,color,royal}]);
  const spec=selectedTransformSpec(s,card);let allowed;
  if(['barrier','beacon','killer','transmission-tower'].includes(card))allowed=october13TargetAllowed(s,'white',card,'d4');
  else if(['wizard','idol'].includes(card))allowed=transformV10SelectionContract(s,'white',action(['d4']),['queen']).ok===true;
  else allowed=selectedTransformContractV7(s,'white',card,action(['d4']),spec).ok===true;
  return {card,type,color,royal,allowed};
 }))));
})()`,worldAdapter);
const facts=run(`(()=>{const s={cardState:{profile:{catalogHash:REFORM_CATALOG_HASH}}};return {source_sha256:${JSON.stringify(sha)},minor_types:canonicalMinorPieceTypesForCatalog(s),spells:V10_WIZARD_SPELLS,reaper_target:reaperCaptureTarget(s),reaper_allied_loss:usesReaperAlliedLossRule(s),reaper_counts:['white','black'].flatMap(reaper=>['white','black'].flatMap(victim=>['white','black',null].map(captor=>({reaper,victim,captor,counts:reaperSoulCountsCapture(reaper,victim,captor,s)})))),meteor:Array.from({length:64},(_,i)=>{const row=Math.floor(i/8),col=i%8;return {at:squareName$7(row,col),cells:v10LargeFootprint({row:Math.min(row,6),col:Math.min(col,6)}).map(c=>squareName$7(c.row,c.col))}})};})()`);
const pairs=run(`(()=>{
 const entries=[{square:'d4',type:'queen',color:'white'},{square:'b2',type:'bishop',color:'white'},{square:'c2',type:'rook',color:'white'},{square:'d2',type:'knight',color:'black'}];
 const cases=[['d4','b2'],['b2','d4'],['d4','d4'],['d4','d2'],['d4','c2'],['b2','c2'],['c2','b2']];
 return cases.flatMap(squares=>[false,true].map(royal=>{const s=world(entries.map(p=>({...p,royal})));return {squares,royal,grappler:grapplerSelectionContract(s,'white',action(squares)).ok===true,windmill:pairTransformSelectionContract(s,'white',action(squares),['bishop','rook']).ok===true}}));
})()`,worldAdapter);
const native=run(`(()=>{
 const cases=[];
 for(const [kind,capability] of [['prime-minister','v7-prime-minister'],['reaper','v7-reaper'],['campfire','v19-campfire'],['idol','v8-idol'],['jester','v7-jester'],['wizard','v8-wizard'],['cannon','v7-cannon'],['windmill','v7-windmill']]) for(const color of ['white','black']) for(const blocked of [false,true]) {
  const opposite=color==='white'?'black':'white';
  const blockers=[{square:'h1',type:'king',color:'white'},{square:'h8',type:'king',color:'black'},...blocked?[{square:'d5',type:'pawn',color},{square:'e5',type:'pawn',color:opposite},{square:'a4',type:'rook',color:opposite},{square:'g7',type:'king',color:opposite}]:[]];
  const state=world([{square:'d4',type:kind,color},...blockers]);state.cardState.pieces.piece0.attributes.windmillMode='bishop';
  const from=squareCoordinates$3('d4'),expected=[];
  for(let r=0;r<8;r++)for(let c=0;c<8;c++) if(r!==from.row||c!==from.col) {
   if(manualCapabilityGeometry(state,color,capability,from,{row:r,col:c},{skipInjury:true}))expected.push(squareName$7(r,c));
  }
  cases.push({kind,color,blockers,expected:expected.sort()});
 }
 return cases;
})()`,worldAdapter);
const eagle=run(`[
 {origin:'d4',open:['e4','c1'],attacked:['e4'],reserved:[]},
 {origin:'d4',open:['e4','d5','c1'],attacked:[],reserved:[]},
 {origin:'d4',open:['e4','d5'],attacked:[],reserved:['d5']},
 {origin:'d4',open:['e4'],attacked:['e4'],reserved:[]}
].map(input=>{const cell=squareCoordinates$3(input.origin),state={position:{},cardState:{open:input.open,attacked:input.attacked}};const result=v8EaglePlacementCandidates(state,{cell,metadata:{color:'white'}},new Set(input.reserved));return {...input,expected:result.map(c=>c.name)}})`, `
const v19PiecePlacementSquareOpen=(s,r,c)=>s.cardState.open.includes(squareName$7(r,c));
const v9HasBlockingInstallation=()=>false;
const v8CurrentPieces=(p,i,c)=>[{metadata:{color:'black'},attacked:c.attacked}];
const v6Attacks=(s,a,attacker,target)=>attacker.attacked.includes(target.name);
`);
const borrowed=run(`(()=>{
 const cases=[];
 for(const type of ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king', 'assassin', 'grasshopper', 'amazon', 'prime-minister', 'dragon', 'cannon', 'herald', 'knightmaster', 'unicorn', 'hook', 'idol', 'jester', 'recruiter', 'reaper', 'wizard', 'alibaba', 'log', 'missionary', 'cardinal', 'protestant', 'standard-bearer', 'windmill', 'squire', 'big-rook', 'checker', 'checker-king', 'alfil', 'guard', 'fanatic', 'colossus', 'royal-knight', 'man', 'merchant', 'dark-wizard', 'monster', 'scarecrow', 'camel', 'ferz', 'bat', 'baby-bear', 'lobster', 'bear', 'vip', 'crown', 'football', 'siege-ram', 'magic-girl', 'berserker', 'slime', 'siren', 'trickster', 'undead', 'clockwork', 'parrot', 'paladin', 'princess', 'octopus', 'hedgehog', 'campfire', 'thief', 'brutus', 'big-bishop', 'archbishop', 'chancellor', 'grappler', 'revolving-door', 'don-quixote', 'medium', 'berolina-pawn', 'killer', 'beacon', 'transmission-tower', 'barrier'])for(const mode of type==='windmill'?['bishop','rook']:[null])for(const color of ['white','black'])for(const blocked of [false,true]) {
  const other=color==='white'?'black':'white';
  const blockers=[{square:'h1',type:'king',color:'white'},{square:'h8',type:'king',color:'black'},...blocked?[{square:'d5',type:'pawn',color},{square:'e5',type:'pawn',color:other},{square:'a4',type:'rook',color:other}]:[]];
  const board=new Map(blockers.map(p=>[p.square,p]));
  const state=world([]);const result=parrotBaseMoves({memory:{type,...mode?{windmillMode:mode}:{}},row:4,col:3,color,moved:true,at:(r,c)=>board.get(squareName$7(r,c))??null,canCapture:p=>p.color!==color,isRoyal:p=>p.type==='king',state});
  cases.push({type,mode,color,blockers,expected:[...new Set(result.map(m=>squareName$7(m.row,m.col)))].sort()});
 }
 return cases;
})()`,worldAdapter+'\n'+declaration('ORTH').code);
const fixture={...facts,selected,pairs,native,eagle,borrowed};
const path=new URL('crates/augment-core/tests/fixtures/current-piece-cards.json',root);
assert.ok(process.argv.slice(2).every(x=>x==='--write'));
if(process.argv.includes('--write'))writeFileSync(path,JSON.stringify(fixture,null,2)+'\n',{flag:'wx'});
assert.deepEqual(fixture,JSON.parse(readFileSync(path,'utf8')));
console.log(`Piece cards: ${selected.length} source selection cases, royal selection contracts, ${native.length} authority and ${borrowed.length} borrowed movement cases, eagle placement, reaper captures, spell costs and 64 meteor footprints passed.`);
