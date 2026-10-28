// Bounded pinned-source contracts. State/identity plumbing is adapted; unrelated
// effects and large-body fracture are excluded. This is not whole-app parity.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
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
const names=['ownerStartDuePly','ownerStartsRemaining','greekGiftVictims','greekGiftSelectionContract','october13TargetAllowed','october13SelectionContract','applyOctober13Card','october13Start','september26SelectionContract','applySeptember26SelectionCard','resolveSeptember26Taboo','v11ResolveOtherworld'];
const adapters=`
const COLORS$n=['white','black'],COMPOUND_CARD_SPECS={},OCTOBER13_TRANSFORMS={};
const OCTOBER13_CARDS=[{id:'seal',target:'enemy'},{id:'ascension',target:'own-queen-empty'}];
const october13Selection=d=>({maxSquares:d.id==='ascension'?2:1});
const squareCoordinates$3=s=>typeof s==='string'&&/^[a-h][1-8]$/.test(s)?{name:s,row:8-Number(s[1]),col:s.charCodeAt(0)-97}:null;
const squareName$7=(r,c)=>String.fromCharCode(97+c)+(8-r);
const selectedSquares=(a,n)=>{const s=a.payload?.selection?.squares;if(!Array.isArray(s)||s.length!==n||new Set(s).size!==s.length)return null;const cells=s.map(squareCoordinates$3);return cells.every(Boolean)?cells:null;};
const effectiveType$1=p=>p.typeOverride??p.type??p.baseType;
const v15KingAugmentRole=(s,c,p)=>p.royal===true||effectiveType$1(p)==='king';
const opposite$2=c=>c==='white'?'black':'white',opponent$2=opposite$2;
const rejection$1=reason=>({ok:false,reason});
const currentPly=p=>(p.fullmoveNumber-1)*2+(p.turn==='black'?1:0);
const squareForPieceId=(identity,id)=>{for(let row=0;row<8;row++)for(let col=0;col<8;col++)if(identity.board[row][col]?.instanceId===id)return {row,col};return null;};
const v5PieceAt=(s,cell)=>{const c=typeof cell==='string'?squareCoordinates$3(cell):cell;if(!c)return null;const id=s.pieceIdentity.board[c.row][c.col]?.instanceId;return id?{...c,instanceId:id,metadata:s.cardState.pieces[id]}:null;};
const v10LargeOccupant=(s,r,c)=>{const id=s.pieceIdentity.board[r][c]?.instanceId;return id?{instanceId:id}:null;};
const v19PiecePlacementSquareOpen=(s,r,c)=>s.position.board[r]?.[c]===null&&!s.cardState.board.installations[r][c].length;
const consumeCard=c=>({...structuredClone(c),consumed:true});
const nextUnusedTriggerId=()=> 'trigger',nextUnusedEffectId=()=> 'effect';
const v8CardFinish=(s,a,o,action,patch)=>({ok:true,state:{...s,...patch}});
const v8RemoveCurrentPieces=(position,identity,cards,ids,options={})=>{
  removals.push({ids:[...ids],options});const p=structuredClone(position),i=structuredClone(identity),c=structuredClone(cards);
  for(const id of ids){delete c.pieces[id];for(let r=0;r<8;r++)for(let col=0;col<8;col++)if(i.board[r][col]?.instanceId===id){i.board[r][col]=null;p.board[r][col]=null;}}
  return {position:p,pieceIdentity:i,cardState:c};
};
const syncEntityBoard=(cards,p,i)=>{const c=structuredClone(cards),alive=new Set(i.board.flat().filter(Boolean).map(x=>x.instanceId));for(const id of Object.keys(c.pieces))if(!alive.has(id))delete c.pieces[id];for(let r=0;r<8;r++)for(let col=0;col<8;col++){const id=i.board[r][col]?.instanceId;if(id&&!c.pieces[id])c.pieces[id]={color:p.board[r][col][0]==='w'?'white':'black',attributes:{}};}return c;};
const usesV11Rules=()=>true,usesOctober13Reform=()=>false,legacyPentaCheckCounterDepth=0;
const colorEffect$2=()=>null;
const september26DestinationRestriction=()=>null,statusHas$1=(p,k)=>p.statuses?.includes(k)===true;
const legacyLandingSummonDepth=0,isFracturedLarge=()=>false;
const v9CollapsedCell=(s,r,c)=>s.collapsed===squareName$7(r,c);
const v9RemoveInstallationsAt=(cards,cells,kinds)=>{const c=structuredClone(cards);for(const cell of cells)c.board.installations[cell.row][cell.col]=c.board.installations[cell.row][cell.col].filter(x=>!kinds.includes(x.kind));return c;};
const V9_CONVEYOR_MOVABLE_INSTALLATION_KINDS=new Set(['wall','football','monster']);
const v9Installation=(cards,cell,kind,owner,source,index,attributes)=>{cards.board.installations[cell.row][cell.col].push({kind,owner,attributes});};
const expansionBoardEntries=board=>board.flatMap((row,r)=>row.flatMap((item,c)=>item?[{item,row:r,col:c,cells:[{row:r,col:c}]}]:[]));
`;
const context=vm.createContext({structuredClone,removals:[]});
const releaseStart=source.indexOf('  if (sourceRelocated && source) {');
const release=source.slice(releaseStart,source.indexOf('  if (directMove && exhaustionSource',releaseStart));
assert.ok(releaseStart>0&&release.length<1000);
assert.ok(release.includes('if (sourceRelocated && source)')&&release.includes('delete piece2.attributes.grapplerBound'));
new vm.Script(adapters+names.map(declaration).join('\n')+'\nfunction releaseBounds(next,source,sourcePieceId,sourceRelocated){'+release+'}\nglobalThis.contracts={'+names.join(',')+',releaseBounds};').runInContext(context,{timeout:1000});
const c=context.contracts;
const action=squares=>({payload:{selection:{squares}}});
function world() {return {position:{board:Array.from({length:8},()=>Array(8).fill(null)),turn:'white',fullmoveNumber:1,status:'active'},pieceIdentity:{board:Array.from({length:8},()=>Array(8).fill(null))},cardState:{pieces:{},pending:{triggers:[]},effects:{colors:{white:[],black:[]},global:[]},board:{installations:Array.from({length:8},()=>Array.from({length:8},()=>[]))}}};}
function put(s,id,square,type,color='white',attributes={}) {const r=8-Number(square[1]),col=square.charCodeAt(0)-97;s.position.board[r][col]=(color==='white'?'w':'b')+(type==='queen'?'Q':'P');s.pieceIdentity.board[r][col]={instanceId:id,moved:false};s.cardState.pieces[id]={type,color,attributes:structuredClone(attributes)};}
const owned=id=>({cardId:id,cardInstanceId:'c1'});
for(const color of ['white','black']) {const p={turn:color,fullmoveNumber:4};assert.equal(c.ownerStartDuePly(p,1),c.ownerStartDuePly(p,0)+2);assert.equal(c.ownerStartsRemaining(p,color,c.ownerStartDuePly(p)),3);}
let s=world();put(s,'enemy','d5','rook','black');
s=c.applyOctober13Card(s,'white',owned('seal'),action(['d5']),[]).state;
assert.equal(s.cardState.pending.triggers[0].payload.duePly,2);
c.october13Start(s,'white',true);assert.equal(s.cardState.pieces.enemy.attributes.grapplerBound,undefined);
s.position.fullmoveNumber=2;c.october13Start(s,'white',true);assert.equal(s.cardState.pieces.enemy.attributes.grapplerBound.untilColor,'black');assert.equal(s.cardState.pending.triggers.length,0);
const next={pieces:{moving:{color:'black',attributes:{}},bound:{color:'black',attributes:{grapplerBound:{}}},other:{color:'white',attributes:{grapplerBound:{}}}}};
c.releaseBounds(next,next.pieces.moving,'moving',false);assert.ok(next.pieces.bound.attributes.grapplerBound);
c.releaseBounds(next,next.pieces.moving,'moving',true);assert.equal(next.pieces.bound.attributes.grapplerBound,undefined);assert.ok(next.pieces.other.attributes.grapplerBound);
s=world();put(s,'q','d1','queen','white',{shield:true,capturesMade:4,frozenRemaining:2});
let out=c.applyOctober13Card(s,'white',owned('ascension'),action(['d1','e5']),[]).state;
assert.equal(out.cardState.pieces.q,undefined);assert.equal(context.removals.at(-1).options.rememberCapture,false);
for(const [turn,fullmoveNumber] of [['white',2],['black',3],['white',3]]) {out.position={...out.position,turn,fullmoveNumber};out={...out,...c.v11ResolveOtherworld(out,out.position,out.pieceIdentity,out.cardState)};assert.equal(out.cardState.pieces.q,undefined);}
put(out,'blocker','e5','pawn','black');out.position.turn='white';out.position.fullmoveNumber=4;
out={...out,...c.v11ResolveOtherworld(out,out.position,out.pieceIdentity,out.cardState)};
assert.equal(out.pieceIdentity.board[3][4].instanceId,'q');assert.equal(out.pieceIdentity.board[3][4].moved,false);assert.equal(out.cardState.pieces.q.attributes.frozenRemaining,2);assert.equal(out.cardState.pieces.q.attributes.capturesMade,4);assert.equal(out.cardState.pieces.blocker,undefined);assert.equal(out.cardState.pending.triggers.length,0);
s=world();put(s,'q','d1','queen');
out=c.applySeptember26SelectionCard(s,'white',owned('taboo'),action(['d1','e5'])).state;assert.ok(out.cardState.pieces.q);
out.position.board[7][3]=null;out.pieceIdentity.board[7][3]=null;put(out,'q','e5','rook');
out=c.resolveSeptember26Taboo(out,'white');assert.equal(out.cardState.pieces.q,undefined);assert.equal(out.cardState.board.installations[3][4].length,1);assert.equal(out.cardState.board.installations[3][4][0].owner,null);assert.equal(out.cardState.board.installations[3][4][0].kind,'monster');
s=world();put(s,'b','c1','bishop');put(s,'enemy','c8','bishop','black');
assert.equal(c.greekGiftSelectionContract(s,'white',action(['c1'])).ok,true);assert.equal(c.greekGiftSelectionContract(s,'white',action(['c8'])).ok,false);
const board=Array.from({length:8},()=>Array(8).fill(null));board[0][7]={type:'king',color:'black',royal:true};board[1][6]={type:'pawn',color:'black'};board[1][7]={type:'pawn',color:'black',royal:true};board[3][6]={type:'pawn',color:'black'};
assert.equal(c.greekGiftVictims(board,'white',p=>p.royal===true).length,1);board[0][7]=null;board[1][7]=null;board[4][7]={type:'king',color:'black',royal:true};assert.equal(c.greekGiftVictims(board,'white',p=>p.royal===true)[0].row,3);
console.log('Delayed ports: pinned owner-start timers, seal/bound release, queen suspension/landing, taboo neutral installation and Greek-gift targets passed; unrelated effects and large fracture parity unverified.');
