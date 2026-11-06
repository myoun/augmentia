import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import vm from 'node:vm';
const bytes=readFileSync('rulesets/augment_chess_20261017/sources/raw/main-RjrZ5fFP.js');
assert.equal(createHash('sha256').update(bytes).digest('hex'),'2e932da7b9391effbeab3593ea5fdf855041fe5a74f611dfdfde678eee46beb8');
const source=bytes.toString('utf8');
function declaration(name) {
  const match=new RegExp('(?:^|\\n)function '+name.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+'\\(').exec(source);
  assert.ok(match,name);
  const start=match.index+(source[match.index]==='\n'?1:0);
  let end=start;
  while (end>=0 && end-start<25000) {
    end=source.indexOf('\n',end+1); if(end<0)break;
    const code=source.slice(start,end);
    try {new vm.Script(code);} catch {continue;}
    assert.doesNotMatch(code,/\b(?:eval|Function|fetch|document|window|process|require|import)\b|\.constructor\b/);
    return code;
  }
  throw Error('unrecognized declaration '+name);
}
const names=['balancedTrolleyScores','v12TrolleyEligible','v12TrolleyRandom','v12TrolleyShuffle','v12TrolleyScoreMap','v12TrolleyDisjoint','v12TrolleyPairs','v12TrolleyBundles'];
const setup=`
const V12_TROLLEY_MAX_SCORE=10,V12_TROLLEY_LOW_SCORES$1=[2,3,4,5,6,7,8],V12_TROLLEY_MAX_BUNDLE_PIECES=4,V12_TROLLEY_MAX_BUNDLES_PER_SCORE=240,V12_TROLLEY_MAX_ELIGIBLE_PIECES=32,V12_TROLLEY_MAX_SCORE_GAP=2;
const distanceFromRange=(v,min,max)=>Math.max(min-v,0,v-max);
const usesBalancedTrolley=()=>true,usesOctober7Balance=()=>true,usesV15Rules=()=>true;
const v15KingAugmentRole=()=>false, effectiveType$1=p=>p.type, opposite$2=c=>c==='white'?'black':'white';
const v12PieceValue=p=>p.value;
const v4Pieces=(s,predicate)=>s.entries.filter(e=>predicate(e.metadata,0,0,e.identity));
`;
const context=vm.createContext({});
new vm.Script(setup+names.map(declaration).join('\n')+'\nglobalThis.sample=v12TrolleyBundles;').runInContext(context,{timeout:1000});
const entries=[];
for(const [color,list] of [['white',[[10,'rook',5],[11,'pawn',1],[12,'bishop',3],[13,'knight',3],[14,'pawn',1],[15,'pawn',1]]],['black',[[20,'rook',5],[21,'knight',3],[22,'pawn',1],[23,'bishop',3],[24,'pawn',1],[25,'rook',5],[26,'queen',9],[27,'bishop',3]]]]) {
  for(const [id,type,value] of list) entries.push({metadata:{color,type,value},identity:{instanceId:String(id)}});
}
const state={entries,cardState:{}};
const fixtures = [
  [0, [["22","23"],["20","21","27","11","10","14"]], [4,18]],
  [1, [["20","27","24","10","15"],["21"]], [15,3]],
  [42, [["23","24"],["20","25","11","10"]], [4,16]],
  [0xdeadbeef, [["26","22","14","12","13"],["23"]], [17,3]],
];
for (const [seed, bundles, values] of fixtures) {
  const actual = JSON.parse(JSON.stringify(context.sample(state, 'black', seed)));
  assert.deepEqual(actual, {bundles, values}, `seed ${seed}`);
}
const sparse = { entries: [
  { metadata: {color:'black', type:'bishop', value:3}, identity: {instanceId:'21'} },
  { metadata: {color:'black', type:'queen', value:9}, identity: {instanceId:'26'} },
  { metadata: {color:'white', type:'bishop', value:3}, identity: {instanceId:'12'} },
], cardState: {} };
assert.deepEqual(JSON.parse(JSON.stringify(context.sample(sparse, 'black', 0))), {
  bundles: [['21'], ['26', '12']], values: [3, 12],
});
console.log('Trolley: five pinned-source balanced-selection fixtures passed.');
