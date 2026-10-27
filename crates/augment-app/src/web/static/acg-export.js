// ACG is a public presentation record, not a deterministic core-state save.
import { encodeAcgRecord } from './acg-codec.js';
const typeName = kind => ({unicorn:'pegasus', alibaba:'eagle'}[kind] || kind.replace(/_([a-z])/g, (_, c) => c.toUpperCase()));
function frameFromNode(node) {
  const board = Array.from({length:8}, () => Array(8).fill(null));
  for (const cell of node.board) {
    const piece = cell.piece;
    if (!piece?.anchor) continue;
    const col = cell.square.charCodeAt(0) - 97;
    const row = 8 - Number(cell.square.slice(1));
    board[row][col] = {id:`p${piece.id}`, type:typeName(piece.kind), color:piece.owner, moved:piece.moved, origin:piece.origin};
  }
  return {board, mode:'local', gameStyle:'normal', turn:node.turn, fullMove:Math.floor(node.ply/2)+1,
    moveCount:node.ply, actionsRemaining:node.actionsRemaining, turnsTaken:node.turnsTaken,
    cardState:Object.fromEntries(['white','black'].map(owner => [owner,node.cards[owner].map((card,slot) => ({...card,slot,acquiredOrder:Number(card.instanceId.slice(1)) || slot}))])), appliedRuleCard:node.rules[0] ? {id:node.rules[0]} : null,
    additionalRuleCards:node.rules.slice(1).map(id=>({id})), winner:node.winner};
}
function delta(before, after) {
  const cells=[];
  for(let row=0;row<8;row++) for(let col=0;col<8;col++) {
    if (JSON.stringify(before.board[row][col]) !== JSON.stringify(after.board[row][col])) cells.push({row,col,before:before.board[row][col],after:after.board[row][col]});
  }
  const fields=Object.keys(after).filter(key=>key!=='board' && JSON.stringify(before[key])!==JSON.stringify(after[key]))
    .map(key=>({key,beforeExists:true,afterExists:true,before:before[key],after:after[key]}));
  return {board:{before:{rows:8,cols:8},after:{rows:8,cols:8},cells},fields};
}
export async function encodeAnalysisPath(path, now=new Date().toISOString()) {
  if (!Array.isArray(path.nodes) || !path.nodes.length) throw new Error('분석 경로가 비어 있습니다.');
  if (path.nodes.some(node => node.cards.white.length>32 || node.cards.black.length>32 || node.rules.length>32)) throw new Error('ACG 카드/룰 개수 제한(각 32개)을 초과했습니다.');
  const frames=path.nodes.map(frameFromNode);
  const notationTimeline=[];
  const events=path.nodes.slice(1).map((node,index)=>{
    const notation={id:`analysis-n${index+1}`,kind:node.kind,color:node.actor,text:node.label.slice(0,96),description:'Augmentia 분석 분기 · 보드/카드/룰 스냅샷 (비공개 상태·RNG·일부 상태효과 미포함)',moveNumber:node.moveNumber};
    notationTimeline.push({moveNumber:node.moveNumber,color:node.actor,notation,replayIndex:index+1});
    return {id:`analysis-e${index+1}`,label:node.label.slice(0,80),moveNumber:node.moveNumber,color:node.actor,notations:[notation],visuals:[],delta:delta(frames[index],frames[index+1])};
  });
  const record={format:'augmentchess-acg',version:2,recordId:crypto.randomUUID(),createdAt:now,
    metadata:{startedAt:'',endedAt:'',mode:'Augmentia analysis snapshots',campaign:'',rules:path.nodes.at(-1).rules,
      white:'WHITE',black:'BLACK',result:({white:'1-0',black:'0-1',draw:'1/2-1/2'}[path.nodes.at(-1).winner] || '*'),termination:'분석 경로 스냅샷: 비공개 엔진 상태·RNG·일부 상태효과는 포함하지 않음',board:'8x8',history:'Partial',cardHistory:'Partial',appVersion:'augment-web'},
    replay:{baseFrame:frames[0],events,notationTimeline}};
  return encodeAcgRecord(record);
}
