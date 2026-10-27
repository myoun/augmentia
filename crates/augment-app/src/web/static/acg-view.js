// Presentation only: never infer a full engine state from a public replay frame.
export function acgAnalysisReadiness(frame, catalog = []) {
  const cells = acgBoardCells(frame, catalog).filter(cell => cell.piece);
  const blockers = [];
  if (frame.board.length !== 8 || frame.board.some(row => row.length !== 8)) blockers.push('현재 엔진은 8×8 보드만 편집할 수 있습니다.');
  const unknown = [...new Set(cells.filter(cell => !catalog.some(kind => kind.kind === cell.piece.kind)).map(cell => cell.piece.name))];
  if (unknown.length) blockers.push(`기물 미지원: ${unknown.join(', ')}`);
  if (cells.some(cell => !['white', 'black'].includes(cell.piece.owner))) blockers.push('중립 기물의 소유권을 현재 엔진으로 복원할 수 없습니다.');
  const cardCount = ['white', 'black'].reduce((sum, owner) => sum + (frame.cardState?.[owner] || []).filter(Boolean).length, 0);
  const ruleCount = (frame.appliedRuleCard ? 1 : 0) + (frame.additionalRuleCards || []).filter(Boolean).length;
  return { boardCopyAllowed: blockers.length === 0, blockers, cardCount, ruleCount,
    resetSummary: `보드만 복사하면 카드 ${cardCount}장·룰 ${ruleCount}개 및 상태효과·반복 이력·확률 상태를 초기화합니다. 차례만 유지하며 캐슬링과 이후 드래프트는 기본 해제됩니다.` };
}

export function acgBoardCells(frame, catalog = []) {
  const cells = [];
  for (let row = 0; row < frame.board.length; row++) {
    for (let col = 0; col < frame.board[row].length; col++) {
      const source = frame.board[row][col];
      const square = `${String.fromCharCode(97 + col)}${frame.board.length - row}`;
      const sourceKey = source?.type?.replace(/([a-z])([A-Z])/g, '$1_$2').replaceAll('-', '_').toLowerCase();
      const key = ({pegasus:'unicorn', eagle:'alibaba'})[sourceKey] || sourceKey;
      const kind = catalog.find(k => k.kind === key);
      const piece = source ? { ...kind, kind: key, name: kind?.name || source.type,
        id: source.id, owner: source.color, ownerLabel: source.color === 'white' ? '백' : source.color === 'black' ? '흑' : '중립',
        symbol: kind ? (source.color === 'white' ? kind.whiteSymbol : kind.blackSymbol) : source.type,
        isUnicode: kind?.isUnicode || false, anchor: true, anchorSquare: square, captureCount: source.captureCount || 0,
        statuses: Object.entries(source).filter(([k, v]) => !['id','type','color','origin'].includes(k) && v !== false && v !== null)
          .map(([k,v]) => `${k}: ${typeof v === 'object' ? JSON.stringify(v) : v}`),
      } : null;
      cells.push({square, piece, source});
    }
  }
  return cells;
}
