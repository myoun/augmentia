// Bounded pinned-source contracts. State/identity plumbing is adapted; unrelated
// effects and large-body fracture are excluded. This is not whole-app parity.
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import vm from 'node:vm';
const root=new URL('../',import.meta.url);
const bytes=readFileSync(new URL('rulesets/augment_chess_20261017/sources/raw/main-RjrZ5fFP.js',root));
assert.equal(createHash('sha256').update(bytes).digest('hex'),'2e932da7b9391effbeab3593ea5fdf855041fe5a74f611dfdfde678eee46beb8');
const source=bytes.toString('utf8');
function declaration(name) {
  const escaped=name.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
  const match=new RegExp(`(?:^|\\n)(const ${escaped} = |function ${escaped}\\()`).exec(source);
  assert.ok(match,name);
  const start=match.index+(source[match.index]==='\n'?1:0),fn=source.startsWith('function ',start);
  let end=start;
  for(let i=0;i<1000;i++) {
    end=source.indexOf(fn?'\n':';',end+1);
    if(end<0||end-start>25000)break;
    const code=source.slice(start,end+(fn?0:1));
    try{new vm.Script(code);}catch{continue;}
    assert.doesNotMatch(code,/\b(?:eval|Function|fetch|document|window|process|require|import)\b|\.constructor\b/);
    return code;
  }
  throw new Error(`Unrecognized declaration ${name}`);
}
const names=['grudgeMask','name','grudgeTargetAllowed','grudgeMovementMask','applyGrudgeCapture','grudgeMoveAllowed','compare','distance','links','spatialPawnChainLinks','hordePawnChainLinks','cancelCheckedProphecies','awaitsIcbmTurnStart','settleRoyalCheckObjectives','v11CancelProphecies','v11AdvanceTurnEffects'];
const adapters=`
const COLORS$n=['white','black'],legacyTurnStartIcbmDepth=0,automaticMovementDepth=0,legacyPentaCheckCounterDepth=0;
const usesOctober18ReportFixes=()=>true,usesOctober13Cards=()=>true,usesV11Rules=()=>true,usesV19Rules=()=>true;
const opposite$2=c=>c==='white'?'black':'white',currentPly=p=>(p.fullmoveNumber-1)*2+(p.turn==='black'?1:0);
const colorEffect$2=(cards,color,kind)=>cards.effects.colors[color].find(e=>e.kind===kind)??null;
const v6OpponentKingAttacked=(s,color)=>Boolean(s.checked?.[color]);
const v11RemovedPieceIds=(a,b)=>(a?.ids??[]).filter(id=>!(b?.ids??[]).includes(id));
const v12AdvanceTurnEffects=(s,a,position,pieceIdentity,cardState)=>({position,pieceIdentity,cardState});
const v14ResolveEndOfTurn=(s,a,x)=>x,markRuleExecutionError=e=>e;
`;
const context=vm.createContext({structuredClone});
const start=source.indexOf('  if (position.turn !== actor) for (const color2 of COLORS$n) {');
const candidateBlock=source.slice(start,source.indexOf('  const highlander =',start));
assert.ok(start>0&&candidateBlock.length<1500&&candidateBlock.includes('remainingHalfTurns'));
new vm.Script(adapters+names.map(declaration).join('\n')+'\nfunction prophecyCandidates(before,position,cardState,actor){const candidates=[];'+candidateBlock+'return candidates;}\nglobalThis.contracts={'+names.join(',')+',prophecyCandidates};').runInContext(context,{timeout:1000});
const c=context.contracts;
const model=readFileSync(new URL('crates/augment-core/src/model.rs',root),'utf8');
const kinds=[.../pub enum PieceKind \{([\s\S]*?)\n\}/.exec(model)[1].matchAll(/^\s+([A-Za-z]+),$/gm)].map(m=>m[1]);
const typeFor=kind=>({Unicorn:'pegasus',Alibaba:'eagle'}[kind]??kind.replace(/([a-z])([A-Z])/g,'$1-$2').toLowerCase());
const masks=[];
for(const kind of kinds) {
  const piece={type:typeFor(kind),color:'white',attributes:{}};
  if(kind==='Monster'||!c.grudgeTargetAllowed(piece,false))continue;
  const ownKing={type:'king',color:'white',id:'king'},board=[[ownKing,piece]];
  masks.push({kind,mask:c.grudgeMovementMask(piece,{board})});
}
for(const [kind,flags,attributes] of [['Rook',{reversal:true},{}],['Bishop',{reversal:true},{}],['Princess',{queen:true},{}],['MagicGirl',{surge:true},{}],['Berserker',{army:true},{}],['Trickster',{}, {tricksterMoveType:'knight'}],['Bishop',{}, {grudgeLoss:2}]]) {
  const piece={type:typeFor(kind),color:'white',attributes};const state={board:[[{type:'king',color:'white'},piece]],reversal:{white:flags.reversal===true},magicGirlSurge:{white:flags.surge===true}};
  if(flags.queen)state.board[0].push({type:'queen',color:'white'});
  if(flags.army)for(let i=0;i<6;i++)state.board[0].push({id:i,type:'pawn',color:'white'});
  masks.push({kind,flags,loss:attributes.grudgeLoss??0,learned:attributes.tricksterMoveType??null,mask:c.grudgeMovementMask(piece,state)});
}
const restrictions=[];
for(const kind of ['Queen','Cardinal','Protestant','Hook','Brutus'])for(const loss of [1,2,4,8,15]) {
  const expected=[];
  for(let rank=0;rank<8;rank++)for(let file=0;file<8;file++)if(c.grudgeMoveAllowed({type:typeFor(kind),attributes:{grudgeLoss:loss}},{row:4,col:3},{row:7-rank,col:file}))expected.push(String.fromCharCode(97+file)+(rank+1));
  restrictions.push({kind,loss,from:'d4',expected:expected.sort()});
}
const graphInputs=[
  {squares:['a7','b7','c7','d7','e7','f7','g7','h7']},
  {squares:['a7','b7','c6','d5','e6','f7','g7','h7']},
  {squares:['a1','a8','h1','h8']},
  {squares:['b2','b4','c3','d2','d4','e3']},
  {squares:['c3']},{squares:[]},
  ...['white','black'].map(horde=>({horde,squares:['a2','b2','c2','d2','e2','f2','g2','h2','a3','b3','c3','d3','e3','f3','g3','h3']})),
];
const graphs=graphInputs.map(({squares,horde=null})=>{const entries=squares.map((s,i)=>({id:String(i+1).padStart(4,'0'),row:8-Number(s[1]),col:s.charCodeAt(0)-97}));const output=horde?c.hordePawnChainLinks(entries,horde):c.spatialPawnChainLinks(entries);return {squares,horde,expected:Array.from(output,pair=>Array.from(pair,Number))};});
const fixture={source_bundle_sha256:createHash('sha256').update(bytes).digest('hex'),scope:'Native grudge masks, geometric loss restrictions and deterministic chain layouts. Unrelated state plumbing excluded.',masks,restrictions,graphs};
const path=new URL('crates/augment-core/tests/fixtures/october-objectives.json',root);
assert.ok(process.argv.slice(2).every(a=>a==='--write'));
if(process.argv.includes('--write'))writeFileSync(path,JSON.stringify(fixture,null,2)+'\n',{flag:'wx'});
assert.deepEqual(JSON.parse(readFileSync(path,'utf8')),fixture);
let attacker={type:'knight',color:'white',attributes:{}},victim={type:'bishop',color:'black',attributes:{grudge:true}};
assert.equal(c.applyGrudgeCapture(attacker,victim),true);assert.equal(attacker.attributes.grudgeLoss,2,'current report policy loses victim component even without native overlap');
assert.equal(c.grudgeMoveAllowed(attacker,{row:4,col:3},{row:3,col:4}),false);
assert.equal(c.grudgeMoveAllowed(attacker,{row:4,col:3},{row:3,col:4,dragonSwap:true}),true);
function world(){return {position:{status:'active',turn:'white',fullmoveNumber:1},pieceIdentity:{ids:['a','b']},checked:{white:false,black:false},cardState:{effects:{colors:{white:[],black:[]},global:[]}}};}
const effect=(kind,attributes)=>({kind,attributes});
let before=world(),next=structuredClone(before);next.cardState.effects.colors.white.push(effect('penta-check',{count:0,lastPly:-1}));
before.checked.white=true;next.checked.white=true;c.settleRoyalCheckObjectives(before,next);assert.equal(next.cardState.effects.colors.white[0].attributes.count,0);
for(let i=0;i<6;i++){before=structuredClone(next);before.checked.white=false;next.checked.white=true;c.settleRoyalCheckObjectives(before,next);assert.equal(next.cardState.effects.colors.white[0].attributes.count,Math.min(i+1,5));before=structuredClone(next);c.settleRoyalCheckObjectives(before,next);assert.equal(next.cardState.effects.colors.white[0].attributes.count,Math.min(i+1,5));}
let s=world();for(const color of ['white','black'])s.cardState.effects.colors[color].push(effect('prophecy',{remainingHalfTurns:6,checkFree:true}));s.checked.black=true;c.cancelCheckedProphecies(s);assert.equal(s.cardState.effects.colors.white.length,0);assert.equal(s.cardState.effects.colors.black.length,0);
s=world();s.cardState.effects.colors.white.push(effect('prophecy',{remainingHalfTurns:6,checkFree:true}));
const held=c.v11AdvanceTurnEffects(s,'white',[],{...s.position},s.cardState);assert.equal(held.cardState.effects.colors.white[0].attributes.remainingHalfTurns,6,'extra action does not tick');
for(let i=0;i<6;i++){const actor=s.position.turn,pos={...s.position,turn:actor==='white'?'black':'white',fullmoveNumber:1+Math.floor((i+1)/2)};const out=c.v11AdvanceTurnEffects(s,actor,[],pos,s.cardState);assert.equal(out.cardState.effects.colors.white[0].attributes.remainingHalfTurns,5-i);assert.equal(c.prophecyCandidates(s,pos,out.cardState,actor).length,i===5?1:0);s={...s,...out};}
const removed=world();removed.cardState.effects.colors.white.push(effect('prophecy',{remainingHalfTurns:1,checkFree:true}));const out=c.v11AdvanceTurnEffects(removed,'white',[],{...removed.position,turn:'black'},removed.cardState,{ids:['a']});assert.equal(out.cardState.effects.colors.white.length,0,'noncapture removal cancels before final tick');
console.log(`Check objectives: ${masks.length} pinned grudge masks, ${restrictions.length} loss matrices, ${graphs.length} chain layouts, check edges and six-turn prophecy contracts passed; full interaction parity unverified.`);
