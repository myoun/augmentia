import { decodeAcgRecord, reconstructReplayFrame } from "/acg-codec.js";
import { acgBoardCells, acgAnalysisReadiness } from "/acg-view.js";
import { encodeAnalysisPath } from "/acg-export.js";

const elements = {
  board: document.querySelector("#board"),
  seedInput: document.querySelector("#seedInput"),
  newGameButton: document.querySelector("#newGameButton"),
  connectionState: document.querySelector("#connectionState"),
  turnPill: document.querySelector("#turnPill"),
  phaseLabel: document.querySelector("#phaseLabel"),
  phaseDetail: document.querySelector("#phaseDetail"),
  plyValue: document.querySelector("#plyValue"),
  actionValue: document.querySelector("#actionValue"),
  turnValue: document.querySelector("#turnValue"),
  whiteStars: document.querySelector("#whiteStars"),
  blackStars: document.querySelector("#blackStars"),
  whiteCards: document.querySelector("#whiteCards"),
  blackCards: document.querySelector("#blackCards"),
  decisionPanel: document.querySelector("#decisionPanel"),
  selectedHint: document.querySelector("#selectedHint"),
  cancelSelection: document.querySelector("#cancelSelection"),
  statusBadge: document.querySelector("#statusBadge"),
  rulesetValue: document.querySelector("#rulesetValue"),
  fingerprintValue: document.querySelector("#fingerprintValue"),
  deathmatchValue: document.querySelector("#deathmatchValue"),
  platformValue: document.querySelector("#platformValue"),
  rulesList: document.querySelector("#rulesList"),
  abilityList: document.querySelector("#abilityList"),
  eventLog: document.querySelector("#eventLog"),
  actionDialog: document.querySelector("#actionDialog"),
  dialogTitle: document.querySelector("#dialogTitle"),
  dialogOptions: document.querySelector("#dialogOptions"),
  dialogClose: document.querySelector("#dialogClose"),
  toast: document.querySelector("#toast"),
  evalBarFill: document.querySelector("#evalBarFill"),
  evalScoreWhite: document.querySelector("#evalScoreWhite"),
  evalScoreBlack: document.querySelector("#evalScoreBlack"),
  evalBadge: document.querySelector("#evalBadge"),
};

let game = null;
let selectedSquare = null;
let targeting = null;
let busy = false;
let toastTimer = null;
let suggestions = null;
let suggestionRequest = 0;
let setupData = null;
let setupPending = null;
let setupRequest = 0;
let acgDocument = null;
let acgFrame = null;
let acgIndex = 0;
let acgOriginal = "";
let acgName = "record.acg";
let editor = null;
let editorAcgIndex = null;
let pieceCatalog = [];
let cardCatalog = new Map();
let analysisEnabled = true;
let analysisController = null;
let boardFlipped = false;
const branchChoices = new Map();
let popoverAnchor = null;
let popoverRect = null;
let popoverShowTimer = null;
let popoverHideTimer = null;
let dismissedInfoAnchor = null;

function registerInfo(anchor, title, description) {
  anchor.dataset.infoTitle = title;
  anchor.dataset.infoDescription = description;
  anchor.removeAttribute("title");
  anchor.setAttribute("aria-haspopup", "dialog");
}
function hideInfoPopover() {
  clearTimeout(popoverShowTimer); clearTimeout(popoverHideTimer);
  document.getElementById("infoPopover").classList.add("hidden");
  popoverAnchor?.removeAttribute("aria-describedby");
  popoverAnchor = null; popoverRect = null;
}
function positionInfoPopover(rect) {
  const popup = document.getElementById("infoPopover");
  const bounds = popup.getBoundingClientRect();
  const gap = 10, edge = 12;
  let left = rect.right + gap;
  if (left + bounds.width > window.innerWidth - edge) left = rect.left - bounds.width - gap;
  if (left < edge) left = Math.max(edge, Math.min(rect.left, window.innerWidth - bounds.width - edge));
  const top = Math.max(edge, Math.min(rect.top, window.innerHeight - bounds.height - edge));
  popup.style.left = `${left}px`; popup.style.top = `${top}px`;
}
function openInfoPopover(anchor) {
  if (!anchor?.isConnected || elements.actionDialog.open) return;
  hideInfoPopover(); popoverAnchor = anchor; popoverRect = anchor.getBoundingClientRect();
  document.getElementById("selectionTitle").textContent = anchor.dataset.infoTitle;
  document.getElementById("selectionDescription").textContent = anchor.dataset.infoDescription;
  anchor.setAttribute("aria-describedby", "selectionDescription");
  document.getElementById("infoPopover").classList.remove("hidden");
  positionInfoPopover(popoverRect);
}
function scheduleInfoHide() {
  clearTimeout(popoverShowTimer); clearTimeout(popoverHideTimer);
  popoverHideTimer = setTimeout(() => {
    const popup = document.getElementById("infoPopover");
    if (!popup.contains(document.activeElement) && document.activeElement !== popoverAnchor) hideInfoPopover();
  }, 250);
}
function rememberBranch(nodes, cursor) {
  const byId = new Map(nodes.map(node => [node.id, node]));
  let node = byId.get(cursor);
  for (let i = 0; node?.parent != null && i < nodes.length; i++) {
    branchChoices.set(node.parent, node.id); node = byId.get(node.parent);
  }
}
function navigationTargets() {
  if (acgFrame) return { first: 0, previous: Math.max(0, acgIndex - 1), next: Math.min(acgDocument.record.replay.events.length, acgIndex + 1), last: acgDocument.record.replay.events.length, cursor: acgIndex };
  if (!game) return null;
  const nodes = game.notation.nodes, cursor = game.notation.cursor;
  const byId = new Map(nodes.map(node => [node.id, node]));
  const current = byId.get(cursor);
  const nextChild = id => {
    const children = nodes.filter(node => node.parent === id);
    return children.find(node => node.id === branchChoices.get(id)) || children[0];
  };
  let root = current, last = current;
  for (let i = 0; root?.parent != null && i < nodes.length; i++) root = byId.get(root.parent);
  for (let i = 0; last && nextChild(last.id) && i < nodes.length; i++) last = nextChild(last.id);
  return { first: root?.id ?? cursor, previous: current?.parent ?? cursor, next: nextChild(cursor)?.id ?? cursor, last: last?.id ?? cursor, cursor };
}
function renderNavigation() {
  const targets = navigationTargets();
  for (const [id, key] of [["firstNode", "first"], ["previousNode", "previous"], ["nextNode", "next"], ["lastNode", "last"]]) {
    document.getElementById(id).disabled = busy || !!editor || !targets || targets[key] === targets.cursor;
  }
}
function navigateDirection(direction) {
  if (busy || editor) return;
  const targets = navigationTargets(); if (!targets || targets[direction] === targets.cursor) return;
  if (acgFrame) setAcgIndex(targets[direction]); else navigateNode(targets[direction]);
}


async function request(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `요청 실패 (${response.status})`);
  }
  return payload;
}

async function loadState() {
  setConnection("loading", "연결 중");
  try {
    game = await request("/api/state");
    elements.seedInput.value = game.seed;
    clearInteraction();
    render();
    setConnection("ready", "연결됨");
    refreshSuggestions();
  } catch (error) {
    setConnection("error", "연결 끊김");
    showToast(error.message);
  }
}

async function applyAction(index, { preserveAuto = false } = {}) {
  if (busy || !game || editor || acgFrame) return;
  setBusy(true);
  try {
    game = await request("/api/action", {
      method: "POST",
      body: JSON.stringify({ index, fingerprint: game.fingerprint }),
    });
    clearInteraction();
    render();
  } catch (error) {
    showToast(error.message);
    await loadState();
    return;
  } finally {
    setBusy(false);
  }
  refreshSuggestions();
}

async function newGame() {
  if (busy) return;
  const seed = Number(elements.seedInput.value.trim());
  if (!Number.isSafeInteger(seed) || seed < 0) {
    showToast("seed는 0 이상의 정수여야 합니다.");
    return;
  }
  if (game && game.ply > 0 && !window.confirm("현재 게임을 버리고 새 게임을 시작할까요?")) {
    return;
  }
  setBusy(true);
  try {
    game = await request("/api/new", {
      method: "POST",
      body: JSON.stringify({ seed }),
    });
    acgFrame = null; branchChoices.clear(); suggestions = null; setupData = null;
    clearInteraction();
    render();
  } catch (error) {
    showToast(error.message);
    return;
  } finally {
    setBusy(false);
  }
  refreshSuggestions();
}

function render() {
  if (!game) return;
  renderSetupPicker();
  if (acgFrame && !editor) { renderAcg(); return; }
  renderHeader();
  renderBoard();
  renderCards("white", elements.whiteCards);
  renderCards("black", elements.blackCards);
  renderDecisionPanel();
  renderAbilities();
  renderStatus();
  renderSuggestions();
  renderLog();
  renderNotation();
  renderInteractionHint();
}

function renderHeader() {
  elements.turnPill.textContent = `${game.turnLabel} 차례`;
  elements.turnPill.className = `turn-pill ${game.turn}`;
  elements.phaseLabel.textContent = game.phase.label;
  elements.phaseDetail.textContent = phaseDetail();
  elements.plyValue.textContent = game.ply;
  elements.actionValue.textContent = game.actionsRemaining;
  elements.turnValue.textContent = `${game.turns.white} / ${game.turns.black}`;
  elements.whiteStars.textContent = `${game.stars.white} ★`;
  elements.blackStars.textContent = `${game.stars.black} ★`;
}

function phaseDetail() {
  if (game.status.kind !== "ongoing") {
    return game.status.reason || "새 게임을 시작할 수 있습니다.";
  }
  if (game.phase.kind === "draft") {
    return `${game.phase.draftPhaseLabel} 카드 한 장을 선택하세요.`;
  }
  if (game.phase.kind === "choice") {
    return "카드 효과를 계속하려면 한 가지를 선택하세요.";
  }
  if (game.phase.kind === "forced_move") {
    return `기물 #${game.phase.pieceId}의 이동을 완료해야 합니다.`;
  }
  return game.actionsRemaining > 1
    ? `이번 차례에 ${game.actionsRemaining}번 행동할 수 있습니다.`
    : "기물을 움직이거나 사용 가능한 카드를 선택하세요.";
}

function renderBoard() {
  const cellMap = new Map((editor ? editorBoard() : acgFrame ? acgBoardCells(acgFrame, pieceCatalog) : game.board).map((cell) => [cell.square, cell]));
  const legalMoves = selectedSquare
    ? game.actions.filter((action) => action.kind === "move" && action.from === selectedSquare)
    : [];
  const legalDestinations = new Set(legalMoves.map((action) => action.to));
  const targetSquares = new Set(
    targeting ? targeting.actions.flatMap((action) => action.targetSquares || []) : [],
  );
  const fragment = document.createDocumentFragment();

  const rows = acgFrame && !editor ? acgFrame.board.length : 8;
  const cols = acgFrame && !editor ? Math.max(...acgFrame.board.map(row => row.length)) : 8;
  elements.board.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
  elements.board.style.gridTemplateRows = `repeat(${rows}, 1fr)`;
  elements.board.style.aspectRatio = `${cols} / ${rows}`;
  elements.board.style.setProperty("--piece-scale", `${72 / cols}cqw`);
  elements.board.style.setProperty("--code-scale", `${32 / cols}cqw`);
  document.querySelector(".board-stage").style.setProperty("--board-ratio", cols / rows);
  document.querySelector(".board-column").classList.toggle("flipped", boardFlipped);
  elements.evalScoreBlack.textContent = boardFlipped ? "백" : "흑";
  elements.evalScoreWhite.textContent = boardFlipped ? "흑" : "백";
  for (let displayRank = 0; displayRank < rows; displayRank += 1) {
    const rank = boardFlipped ? displayRank : rows - 1 - displayRank;
    for (let displayFile = 0; displayFile < cols; displayFile += 1) {
      const file = boardFlipped ? cols - 1 - displayFile : displayFile;
      const squareName = `${String.fromCharCode(97 + file)}${rank + 1}`;
      const cell = cellMap.get(squareName);
      const square = document.createElement("button");
      square.type = "button";
      square.className = `square ${(file + rank) % 2 === 0 ? "dark" : "light"}`;
      square.dataset.square = squareName;
      square.setAttribute("role", "gridcell");
      square.setAttribute("aria-label", squareAriaLabel(cell));
      if (cell?.piece) square.classList.add("has-piece");
      if (cell?.platform) square.classList.add("platform-cell");
      if (selectedSquare === squareName) square.classList.add("selected");
      if (legalDestinations.has(squareName)) square.classList.add("legal");
      if (targetSquares.has(squareName)) square.classList.add("target");
      square.addEventListener("click", () => handleSquareClick(squareName));

      if (displayFile === 0) {
        const coordinate = document.createElement("span");
        coordinate.className = "coord rank";
        coordinate.textContent = rank + 1;
        square.append(coordinate);
      }
      if (displayRank === rows - 1) {
        const coordinate = document.createElement("span");
        coordinate.className = "coord file";
        coordinate.textContent = String.fromCharCode(97 + file);
        square.append(coordinate);
      }
      if (cell?.piece) {
        square.append(renderPiece(cell.piece));
        registerInfo(square, `${squareName} · ${cell.piece.name}`, `${cell.piece.description || "기물 설명 준비 중"}\n${pieceTitle(cell.piece)}${acgFrame && !editor ? "\n원본 ACG · 읽기 전용" : ""}`);
      }
      fragment.append(square);
    }
  }
  elements.board.replaceChildren(fragment);
}

function renderPiece(piece) {
  const element = document.createElement("span");
  element.className = `piece ${piece.isUnicode ? "unicode" : "code"} ${piece.owner}`;
  if (!piece.anchor) element.classList.add("large-body");
  element.textContent = piece.symbol;
  // The square owns the accessible hover/focus description.
  if (piece.statuses.length > 0 && piece.anchor) {
    const dot = document.createElement("span");
    dot.className = "piece-status-dot";
    element.append(dot);
  }
  return element;
}

function squareAriaLabel(cell) {
  if (!cell?.piece) return cell?.square || "빈 칸";
  return `${cell.square} ${cell.piece.ownerLabel} ${cell.piece.name}`;
}

function pieceTitle(piece) {
  const details = [`${piece.ownerLabel} ${piece.name}`, `ID #${piece.id}`];
  if (piece.captureCount > 0) details.push(`포획 ${piece.captureCount}`);
  details.push(...piece.statuses);
  return details.join(" · ");
}

function handleSquareClick(square) {
  if (busy) return;
  if (editor) {
    const kind = document.getElementById("editorKind").value;
    if (kind === "erase") editor.delete(square);
    else editor.set(square, {square, kind, owner: document.getElementById("editorOwner").value});
    renderBoard();
    return;
  }
  if (acgFrame) {
    const cell = acgBoardCells(acgFrame, pieceCatalog).find(c => c.square === square);
    if (cell?.piece) showSelectionInfo(`${square} ${cell.piece.name}`, `${cell.piece.description || "공식 기물 설명 준비 중"}\n원본 상태: ${cell.piece.statuses.join(" · ") || "없음"}`);
    return;
  }
  if (busy || !game) return;
  const inspected = game.board.find(c => c.square === square)?.piece;
  if (inspected) showSelectionInfo(`${square} ${inspected.name}`, `${inspected.description || "이 기물의 합법 이동은 코어 엔진에 따라 표시됩니다."}\n${pieceTitle(inspected)}`);
  if (game.status.kind !== "ongoing") return;

  if (targeting) {
    const matching = targeting.actions.filter((action) =>
      (action.targetSquares || []).includes(square),
    );
    if (matching.length === 1) {
      applyAction(matching[0].index);
    } else if (matching.length > 1) {
      showActionDialog(targeting.label, matching);
    }
    return;
  }

  if (selectedSquare) {
    const matching = game.actions.filter(
      (action) =>
        action.kind === "move" && action.from === selectedSquare && action.to === square,
    );
    if (matching.length === 1) {
      applyAction(matching[0].index);
      return;
    }
    if (matching.length > 1) {
      showActionDialog("프로모션 선택", matching);
      return;
    }
  }

  const cell = game.board.find((candidate) => candidate.square === square);
  const sourceSquare = cell?.piece?.anchorSquare || square;
  const moves = game.actions.filter(
    (action) => action.kind === "move" && action.from === sourceSquare,
  );
  if (moves.length > 0) {
    selectedSquare = sourceSquare;
    targeting = null;
  } else {
    selectedSquare = null;
  }
  renderBoard();
  renderInteractionHint();
}

function renderCards(owner, container) {
  const cards = game.cards[owner];
  if (cards.length === 0) {
    container.className = "card-list empty-message";
    container.textContent = "아직 카드가 없습니다.";
    return;
  }
  container.className = "card-list";
  const fragment = document.createDocumentFragment();
  cards.forEach((card) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `card-button ${card.available ? "active" : ""}`;
    button.disabled = busy || !!editor;
    button.dataset.instanceId = card.instanceId;
    const stateLabel = card.openingAutoPending
      ? "첫 이동 후 자동 발동"
      : card.locked
        ? `${card.availableAfterTurn}턴까지 잠김`
        : card.activation === "passive"
          ? "지속 효과"
          : card.charges === 0 ? "사용 완료" : `충전 ${card.charges}`;
    button.innerHTML = `
      <span class="card-name"></span>
      <span class="card-stars"></span>
      <span class="card-meta"></span>
    `;
    button.querySelector(".card-name").textContent = card.name;
    button.querySelector(".card-stars").textContent = card.openingAutoPending || card.locked ? "대기" : card.activation === "passive" ? "∞" : card.charges === 0 ? "✓" : String(card.charges);
    button.querySelector(".card-meta").textContent = stateLabel;
    button.classList.toggle("spent", card.charges === 0 && card.activation !== "passive" && !card.openingAutoPending && !card.locked);
    button.setAttribute("aria-label", `${card.name} · ${stateLabel}`);
    registerInfo(button, card.name, `${card.description}${card.help ? `\n${card.help}` : ""}\n${stateLabel} · ${card.stars}★`);
    button.addEventListener("click", () => {
      if (busy || editor || acgFrame) return;
      showSelectionInfo(card.name, `${card.description}${card.help ? `\n${card.help}` : ""}\n${stateLabel}`);
      if (card.available) activateActionGroup(actionsByIndices(card.actionIndices), card.name);
    });
    fragment.append(button);
  });
  container.replaceChildren(fragment);
}

function renderDecisionPanel() {
  if (editor) {
    elements.decisionPanel.classList.add("hidden");
    elements.decisionPanel.replaceChildren();
    return;
  }
  const phaseActions = game.actions.filter((action) =>
    game.phase.kind === "draft" ? action.kind === "draft" : action.kind === "choice",
  );
  if (
    game.status.kind === "ongoing" &&
    game.phase.kind !== "draft" &&
    game.phase.kind !== "choice"
  ) {
    elements.decisionPanel.classList.add("hidden");
    elements.decisionPanel.replaceChildren();
    return;
  }

  elements.decisionPanel.classList.remove("hidden");
  const heading = document.createElement("div");
  heading.className = "decision-title";
  const title = document.createElement("div");
  const eyebrow = document.createElement("span");
  eyebrow.className = "eyebrow";
  const name = document.createElement("h2");
  const explanation = document.createElement("p");
  if (game.status.kind !== "ongoing") {
    eyebrow.textContent = "MATCH COMPLETE";
    name.textContent = game.status.label;
    explanation.textContent = game.status.reason || "왼쪽 아래의 새 분석 버튼으로 다시 시작할 수 있습니다.";
  } else if (game.phase.kind === "draft") {
    eyebrow.textContent = `${game.phase.draftPhaseLabel.toUpperCase()} DRAFT`;
    name.textContent = `${game.turnLabel} 카드 선택`;
    explanation.textContent = "한 장을 선택하면 상대 드래프트로 이어집니다.";
  } else {
    eyebrow.textContent = "EFFECT CHOICE";
    name.textContent = "카드 효과 선택";
    explanation.textContent = "효과를 계속할 옵션을 고르세요.";
  }
  title.append(eyebrow, name);
  heading.append(title, explanation);

  const grid = document.createElement("div");
  grid.className = "decision-grid";
  phaseActions.forEach((action, position) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "decision-card";
    const card = action.card;
    button.innerHTML = `
      <span class="number">${String(position + 1).padStart(2, "0")}</span>
      <strong></strong>
      <small></small>
      <span class="stars"></span>
    `;
    button.querySelector("strong").textContent = action.label;
    button.querySelector("small").textContent = card
      ? `${activationLabel(card.activation)} · ${card.phase.toUpperCase()}`
      : "후속 효과";
    button.querySelector(".stars").textContent = card ? `${card.stars} ★` : "";
    if (card) registerInfo(button, card.name, `${card.description}${card.help ? `\n${card.help}` : ""}`);
    button.disabled = busy;
    button.addEventListener("click", () => applyAction(action.index));
    grid.append(button);
  });

  elements.decisionPanel.replaceChildren(heading);
  if (phaseActions.length > 0) elements.decisionPanel.append(grid);
}

function renderAbilities() {
  const abilities = game.actions.filter((action) => action.kind === "ability");
  if (abilities.length === 0) {
    elements.abilityList.className = "ability-list empty-message";
    elements.abilityList.textContent = "사용 가능한 특수 행동이 없습니다.";
    return;
  }
  const groups = new Map();
  abilities.forEach((action) => {
    const key = `${action.pieceId}:${action.ability}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(action);
  });
  elements.abilityList.className = "ability-list";
  const fragment = document.createDocumentFragment();
  groups.forEach((actions) => {
    const first = actions[0];
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ability-button";
    button.innerHTML = "<strong></strong><span></span>";
    button.querySelector("strong").textContent = first.abilityLabel;
    button.querySelector("span").textContent = `${first.sourceSquare || `#${first.pieceId}`} · ${actions.length}개 대상`;
    button.addEventListener("click", () =>
      activateActionGroup(actions, `${first.abilityLabel} 대상 선택`),
    );
    fragment.append(button);
  });
  elements.abilityList.replaceChildren(fragment);
}

function renderStatus() {
  if (game.eval) {
    elements.evalBarFill.style.height = `${game.eval.whiteRatio ?? 50}%`;
    elements.evalScoreWhite.textContent = boardFlipped ? "흑" : "백";
    elements.evalScoreBlack.textContent = boardFlipped ? "백" : "흑";
    elements.evalBadge.textContent = game.eval.score.toFixed(2);
  }
  elements.statusBadge.textContent = game.status.label;
  elements.statusBadge.className = `status-badge ${game.status.kind === "ongoing" ? "" : "finished"}`;
  elements.rulesetValue.textContent = `${game.ruleset} · ${game.coverage?.label ?? ""}`;
  elements.fingerprintValue.textContent = game.fingerprint;
  elements.fingerprintValue.title = game.fingerprint;
  elements.deathmatchValue.textContent = game.automaticDraw
    ? `${game.automaticDraw.quietTurns} / 100턴${game.automaticDraw.progressThisTurn ? " · 진행 있음" : ""}`
    : game.deathmatch
    ? `${game.deathmatch.quietFullTurns} / ${game.deathmatch.interval}`
    : "대기";
  elements.platformValue.textContent = game.platform || "없음";
  const tags = game.ruleCards.map((rule) => {
    const tag = document.createElement("button");
    tag.className = "tag";
    tag.textContent = `RULE · ${rule.name}`;
    registerInfo(tag, rule.name, `${rule.description}${rule.help ? `\n${rule.help}` : ""}`);
    tag.addEventListener("click", () => showSelectionInfo(rule.name, `${rule.description}${rule.help ? `\n${rule.help}` : ""}`));
    return tag;
  });
  elements.rulesList.replaceChildren(...tags);
}

function renderLog() {
  const fragment = document.createDocumentFragment();
  [...game.events].reverse().forEach((event) => {
    const item = document.createElement("li");
    item.textContent = event;
    fragment.append(item);
  });
  elements.eventLog.replaceChildren(fragment);
}

function activateActionGroup(actions, label) {
  if (actions.length === 0 || busy || editor || acgFrame) return;
  const card = actions[0].card;
  if (card) showSelectionInfo(card.name, `${card.description}${card.help ? `\n${card.help}` : ""}`);
  if (actions.length === 1 && (actions[0].targetSquares || []).length === 0) {
    showActionDialog(label, actions);
    return;
  }
  if (actions.every((action) => (action.targetSquares || []).length > 0)) {
    selectedSquare = null;
    targeting = { actions, label };
    renderBoard();
    renderInteractionHint();
    return;
  }
  showActionDialog(label, actions);
}

function showActionDialog(title, actions) {
  elements.dialogTitle.textContent = title;
  const fragment = document.createDocumentFragment();
  actions.forEach((action) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "dialog-option";
    button.textContent = action.label;
    button.addEventListener("click", () => {
      elements.actionDialog.close();
      applyAction(action.index);
    });
    fragment.append(button);
  });
  elements.dialogOptions.replaceChildren(fragment);
  elements.actionDialog.showModal();
}

function renderInteractionHint() {
  if (targeting) {
    const count = new Set(targeting.actions.flatMap((action) => action.targetSquares || [])).size;
    elements.selectedHint.textContent = `${targeting.label} · 강조된 ${count}개 칸 중 하나를 선택하세요.`;
    elements.cancelSelection.classList.remove("hidden");
  } else if (selectedSquare) {
    const count = game.actions.filter(
      (action) => action.kind === "move" && action.from === selectedSquare,
    ).length;
    elements.selectedHint.textContent = `${selectedSquare} 선택 · ${count}개 이동 가능`;
    elements.cancelSelection.classList.remove("hidden");
  } else if (game?.phase.kind === "draft") {
    elements.selectedHint.textContent = "위의 드래프트 카드 중 한 장을 선택하세요.";
    elements.cancelSelection.classList.add("hidden");
  } else if (game?.phase.kind === "choice") {
    elements.selectedHint.textContent = "위의 카드 효과 옵션을 선택하세요.";
    elements.cancelSelection.classList.add("hidden");
  } else if (game?.status.kind !== "ongoing") {
    elements.selectedHint.textContent = `${game.status.label} · 새 게임을 시작할 수 있습니다.`;
    elements.cancelSelection.classList.add("hidden");
  } else {
    elements.selectedHint.textContent = "기물 또는 보드 위아래의 카드를 선택하세요.";
    elements.cancelSelection.classList.add("hidden");
  }
}

function showSelectionInfo(title, description) {
  document.getElementById("selectionTitle").textContent = title;
  document.getElementById("selectionDescription").textContent = description;
  if (popoverRect && !document.getElementById("infoPopover").classList.contains("hidden")) positionInfoPopover(popoverRect);
}
function clearInteraction() {
  hideInfoPopover();
  selectedSquare = null;
  targeting = null;
  if (elements.actionDialog.open) elements.actionDialog.close();
}

function actionsByIndices(indices) {
  const wanted = new Set(indices);
  return game.actions.filter((action) => wanted.has(action.index));
}

function activationLabel(activation) {
  return {
    active: "액티브",
    passive: "패시브",
    rule: "RULE",
  }[activation] || activation;
}

function setBusy(value) {
  busy = value;
  document.body.classList.toggle("busy", value);
  elements.newGameButton.disabled = value;
  document.getElementById("editorCommit").disabled = value;
  // render() runs before finally clears busy: synchronize newly rendered controls too.
  for (const button of document.querySelectorAll(".decision-card, .card-button, .ability-button")) {
    button.disabled = value || !!editor;
  }
  renderSetupPicker();
  renderSuggestions();
  renderNavigation();
}

function setConnection(state, label) {
  elements.connectionState.dataset.state = state;
  elements.connectionState.lastElementChild.textContent = label;
}

function showToast(message) {
  window.clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.add("visible");
  toastTimer = window.setTimeout(() => elements.toast.classList.remove("visible"), 4200);
}

async function refreshSuggestions() {
  const fingerprint = game?.fingerprint;
  const node = game?.notation.cursor;
  const requestId = ++suggestionRequest;
  analysisController?.abort();
  analysisController = new AbortController();
  suggestions = null;
  renderSuggestions();
  if (!analysisEnabled || !game || editor || acgFrame || game.status.kind !== "ongoing") return;
  const getNumber = (id) => {
    const input = document.getElementById(id);
    if (!input.checkValidity() || !Number.isSafeInteger(Number(input.value))) throw new Error("엔진 설정 범위를 확인하세요.");
    return Number(input.value);
  };
  try {
    const result = await request("/api/analyze", {
      method: "POST", signal: analysisController.signal,
      body: JSON.stringify({ fingerprint, depth: getNumber("depthInput"),
        depthUnit: document.getElementById("depthUnitInput").value,
        timeMs: getNumber("timeInput"), nodes: getNumber("nodesInput"),
        chanceSamples: getNumber("chanceInput"), searchSeed: getNumber("searchSeedInput") }),
    });
    if (requestId !== suggestionRequest || game?.fingerprint !== result.fingerprint || game?.notation.cursor !== node) return;
    suggestions = result;
    renderSuggestions();
  } catch (error) {
    if (requestId !== suggestionRequest || error.name === "AbortError") return;
    document.getElementById("analysisMeta").textContent = error.message;
    showToast(error.message);
  }
}

function renderSuggestions() {
  const score = document.getElementById("analysisScore");
  const move = document.getElementById("analysisMove");
  const meta = document.getElementById("analysisMeta");
  const choose = document.getElementById("playRecommendation");
  document.getElementById("analysisToggle").textContent = analysisEnabled ? "중지" : "재시작";
  choose.disabled = busy || !!acgFrame || !!editor || !suggestions?.recommendation || suggestions.fingerprint !== game?.fingerprint;
  if (!suggestions) {
    score.textContent = "—";
    move.textContent = acgFrame ? "ACG 재생 · 엔진 상태 미복원" : editor ? "보드 편집 중" : analysisEnabled && game?.status.kind === "ongoing" ? "분석 중…" : "분석 중지";
    meta.textContent = "완료된 깊이의 결과만 표시합니다.";
    return;
  }
  const report = suggestions.report;
  const value = suggestions.whiteScore;
  const text = Math.abs(value) >= 100000 ? (value > 0 ? "백 승리 평가" : "흑 승리 평가") : `${value > 0 ? "+" : ""}${value.toFixed(2)}`;
  score.textContent = `${text} · 백 기준`;
  move.textContent = suggestions.recommendation?.label || "종료 국면";
  meta.textContent = `깊이 ${report.completed_depth} ${report.depth_unit === "turns" ? "턴" : "액션"} · ${report.stats.nodes.toLocaleString()} 노드 · ${report.elapsed_ms.toFixed(0)}ms · ${report.stop_reason}`;
  document.getElementById("analysisWarning").textContent = `${suggestions.warning}${report.completed_depth === 0 ? " 깊이 1 미완료: 표시 점수는 시작 국면 평가이며 추천은 미평가 대체 행동입니다." : ""}`;
  elements.evalBadge.textContent = text;
  elements.evalBarFill.style.height = `${50 + Math.max(-20, Math.min(20, value)) * 2}%`;
}

async function navigateNode(node) {
  if (busy || !game || editor || acgFrame) return;
  ++suggestionRequest;
  analysisController?.abort();
  setBusy(true);
  try {
    game = await request("/api/navigate", { method: "POST", body: JSON.stringify({ node }) });
    clearInteraction(); render();
  } catch (error) { showToast(error.message); }
  finally { setBusy(false); }
  refreshSuggestions();
}

function renderNotation() {
  const tree = document.getElementById("notationTree");
  const nodes = game.notation.nodes;
  rememberBranch(nodes, game.notation.cursor);
  const byId = new Map(nodes.map(node => [node.id, node]));
  const childCounts = new Map();
  for (const node of nodes) childCounts.set(node.parent, (childCounts.get(node.parent) || 0) + 1);
  const fragment = document.createDocumentFragment();
  // Explicit parent labels avoid presenting branches as one chronological game.
  for (const node of nodes) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `notation-node ${node.id === game.notation.cursor ? "selected" : ""}`;
    button.textContent = node.parent === null ? "시작 국면" : `${node.id}. ${node.label}`;
    button.title = node.parent === null ? "시작" : `이전 국면 #${node.parent}`;
    button.setAttribute("aria-current", String(node.id === game.notation.cursor));
    let depth = 0, ancestor = node;
    for (let i = 0; ancestor?.parent != null && i < nodes.length; i++) {
      if (childCounts.get(ancestor.parent) > 1) depth++;
      ancestor = byId.get(ancestor.parent);
    }
    button.style.setProperty("--branch-depth", Math.min(depth, 6));
    if (node.parent !== null && childCounts.get(node.parent) > 1) button.textContent += ` (분기 · #${node.parent} 이후)`;
    button.addEventListener("click", () => navigateNode(node.id));
    fragment.append(button);
  }
  tree.replaceChildren(fragment);
  tree.querySelector('[aria-current="true"]')?.scrollIntoView({block: "nearest", inline: "nearest"});
  renderNavigation();
}

document.getElementById("analysisToggle").addEventListener("click", () => {
  analysisEnabled = !analysisEnabled;
  ++suggestionRequest;
  analysisController?.abort();
  if (analysisEnabled) refreshSuggestions();
  else {
    request("/api/analysis-stop", { method: "POST", body: "{}" }).catch(error => showToast(error.message));
    renderSuggestions();
  }
});
document.getElementById("playRecommendation").addEventListener("click", () => {
  if (suggestions?.recommendation) applyAction(suggestions.recommendation.index);
});
document.querySelectorAll(".engine-settings input, .engine-settings select").forEach(input => input.addEventListener("change", refreshSuggestions));
document.getElementById("firstNode").addEventListener("click", () => navigateDirection("first"));
document.getElementById("previousNode").addEventListener("click", () => navigateDirection("previous"));
document.getElementById("nextNode").addEventListener("click", () => navigateDirection("next"));
document.getElementById("lastNode").addEventListener("click", () => navigateDirection("last"));
document.getElementById("flipBoardButton").addEventListener("click", () => {
  if (!game) return;
  hideInfoPopover(); boardFlipped = !boardFlipped;
  document.getElementById("flipBoardButton").setAttribute("aria-pressed", String(boardFlipped));
  renderBoard();
});
function editorBoard() {
  return [...editor.values()].map(cell => {
    const kind = pieceCatalog.find(k => k.kind === cell.kind);
    return { square: cell.square, piece: { ...kind, id: 0, owner: cell.owner,
      ownerLabel: cell.owner === "white" ? "백" : "흑", symbol: cell.owner === "white" ? kind.whiteSymbol : kind.blackSymbol,
      anchor: true, anchorSquare: cell.square, statuses: [], captureCount: 0 } };
  });
}
function closeEditor() {
  const replayIndex = editorAcgIndex;
  editor = null; editorAcgIndex = null;
  document.getElementById("boardEditor").classList.add("hidden");
  if (replayIndex !== null) { setAcgIndex(replayIndex); return; }
  clearInteraction(); render(); refreshSuggestions();
}
document.getElementById("editBoardButton").addEventListener("click", async () => {
  if (acgFrame) { copyAcgBoard(); return; }
  if (busy || editor) return;
  try {
    pieceCatalog = await request("/api/pieces");
    const select = document.getElementById("editorKind");
    select.replaceChildren(new Option("지우개", "erase"), ...pieceCatalog.map(k => new Option(`${k.name} · ${k.code}`, k.kind)));
    editor = new Map(game.board.filter(c => c.piece?.anchor).map(c => [c.square, {square: c.square, kind: c.piece.kind, owner: c.piece.owner}]));
    document.getElementById("editorTurn").value = game.turn;
    ++suggestionRequest; analysisController?.abort(); suggestions = null;
    await request("/api/analysis-stop", { method: "POST", body: "{}" });
    clearInteraction();
    document.getElementById("boardEditor").classList.remove("hidden");
    renderBoard(); renderSuggestions();
  } catch (error) { showToast(error.message); }
});
document.getElementById("editorKind").addEventListener("change", () => {
  const kind = pieceCatalog.find(k => k.kind === document.getElementById("editorKind").value);
  if (kind) showSelectionInfo(kind.name, `${kind.description}\n보드를 클릭해 배치합니다. 적용하면 카드/룰/상태효과는 초기화됩니다.`);
});
document.getElementById("editorCancel").addEventListener("click", () => { if (!busy) closeEditor(); });
document.getElementById("editorClear").addEventListener("click", () => { editor.clear(); renderBoard(); });
document.getElementById("editorStandard").addEventListener("click", () => {
  editor.clear();
  const back = ["rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook"];
  for (let file = 0; file < 8; file++) for (const owner of ["white", "black"]) {
    const square = `${String.fromCharCode(97 + file)}${owner === "white" ? 1 : 8}`;
    const pawnSquare = `${square[0]}${owner === "white" ? 2 : 7}`;
    editor.set(square, {square, kind: back[file], owner});
    editor.set(pawnSquare, {square: pawnSquare, kind: "pawn", owner});
  }
  renderBoard();
});
document.getElementById("editorCommit").addEventListener("click", async () => {
  if (!editor || busy) return;
  setBusy(true);
  try {
    game = await request("/api/edit-board", {method: "POST", body: JSON.stringify({
      fingerprint: game.fingerprint, pieces: [...editor.values()], turn: document.getElementById("editorTurn").value,
      allowCastling: document.getElementById("editorCastling").checked, futureDrafts: document.getElementById("editorDrafts").checked,
    })});
    editor = null; editorAcgIndex = null;
    document.getElementById("boardEditor").classList.add("hidden");
    clearInteraction(); render();
  } catch (error) { showToast(error.message); }
  finally { setBusy(false); }
  refreshSuggestions();
});
elements.newGameButton.addEventListener("click", () => { if (!editor) newGame(); });

function renderSetupPicker() {
  const panel = document.getElementById("manualSetupPanel");
  if (panel.classList.contains("hidden")) return;
  const list = document.getElementById("setupCardList");
  if (editor || acgFrame) { list.textContent = "엔진 국면에서 보드 편집을 완료한 뒤 설정할 수 있습니다."; return; }
  const owner = document.getElementById("setupOwner").value;
  const key = `${game.fingerprint}/${game.notation.cursor}/${owner}`;
  if (setupData?.key !== key) {
    list.textContent = "획득 조건 검사 중…";
    if (setupPending !== key) {
      setupPending = key; const generation = ++setupRequest;
      request("/api/setup-options", {method:"POST", body:JSON.stringify({owner})}).then(data => {
        if (generation !== setupRequest) return;
        setupPending = null; setupData = {...data, key}; renderSetupPicker();
      }).catch(error => { if (generation === setupRequest) { setupPending = null; list.textContent = error.message; } });
    }
    return;
  }
  const filter = document.getElementById("setupFilter").value.trim().toLowerCase();
  const cards = setupData.cards.filter(card => `${card.name} ${card.id} ${card.phase} ${card.activation}`.toLowerCase().includes(filter));
  list.replaceChildren(...cards.map(card => {
    const wrapper = document.createElement("div"); wrapper.className = "setup-card";
    wrapper.tabIndex = 0;
    registerInfo(wrapper, card.name, `${card.description}${card.help ? `\n${card.help}` : ""}${card.reason ? `\n${card.reason}` : ""}`);
    const button = document.createElement("button"); button.className = "button secondary";
    button.textContent = `${card.name} · ${card.stars}★ · ${card.phase}`;
    button.disabled = busy || !card.eligible; button.title = card.reason || card.description;
    button.addEventListener("click", () => acquireSetupCard(card.id, owner, setupData.fingerprint));
    const description = document.createElement("p"); description.textContent = card.description;
    wrapper.append(button, description);
    if (!card.eligible) { const reason = document.createElement("p"); reason.className = "analysis-warning"; reason.textContent = card.reason; wrapper.append(reason); }
    return wrapper;
  }));
}
async function acquireSetupCard(cardId, owner, fingerprint) {
  if (busy || editor || acgFrame || game.fingerprint !== fingerprint) return;
  setBusy(true); clearInteraction(); ++suggestionRequest; analysisController?.abort();
  try {
    game = await request("/api/setup-card", {method:"POST", body:JSON.stringify({cardId, owner, fingerprint})});
    suggestions = null; setupData = null; render(); refreshSuggestions();
  } catch (error) { showToast(error.message); } finally { setBusy(false); }
}
function openSetup(owner) {
  if (!game || busy || editor) return;
  hideInfoPopover();
  if (owner) document.getElementById("setupOwner").value = owner;
  document.getElementById("manualSetupPanel").classList.remove("hidden");
  renderSetupPicker(); document.getElementById("setupFilter").focus();
}
document.getElementById("manualSetupButton").addEventListener("click", () => openSetup());
document.querySelectorAll("[data-setup-owner]").forEach(button => button.addEventListener("click", () => openSetup(button.dataset.setupOwner)));
document.getElementById("setupClose").addEventListener("click", () => {
  hideInfoPopover(); document.getElementById("manualSetupPanel").classList.add("hidden");
  document.getElementById("manualSetupButton").focus();
});
document.getElementById("setupOwner").addEventListener("change", renderSetupPicker);
document.getElementById("setupFilter").addEventListener("input", renderSetupPicker);

function setAcgIndex(index) {
  if (!acgDocument || editor) return;
  const replay = acgDocument.record.replay;
  acgIndex = Math.max(0, Math.min(replay.events.length, index));
  acgFrame = reconstructReplayFrame(replay.baseFrame, replay.events, acgIndex);
  clearInteraction(); suggestions = null;
  render();
}
function renderAcg() {
  const record = acgDocument.record;
  const events = record.replay.events;
  const event = events[acgIndex - 1];
  const readiness = acgAnalysisReadiness(acgFrame, pieceCatalog);
  document.getElementById("acgRestoreNotice").textContent = readiness.boardCopyAllowed ? readiness.resetSummary : readiness.blockers.join("\n");
  document.getElementById("acgCopyBoard").disabled = !readiness.boardCopyAllowed || busy;
  document.getElementById("acgSeek").value = acgIndex;
  document.getElementById("acgPosition").textContent = `${acgIndex} / ${events.length} · ${event?.notations.map(n => n.text).join(" · ") || "시작 국면"}`;
  elements.turnPill.textContent = `${acgFrame.turn === "white" ? "백" : "흑"} 차례`;
  elements.turnPill.className = `turn-pill ${acgFrame.turn}`;
  elements.phaseLabel.textContent = "ACG 재생";
  elements.phaseDetail.textContent = "원본 기록 · 읽기 전용";
  elements.plyValue.textContent = acgFrame.moveCount ?? "—";
  elements.actionValue.textContent = acgFrame.actionsRemaining ?? "—";
  elements.turnValue.textContent = `${acgFrame.turnsTaken?.white ?? "—"} / ${acgFrame.turnsTaken?.black ?? "—"}`;
  elements.statusBadge.textContent = acgFrame.winner ? `종료 · ${record.metadata.result}` : "기록 국면";
  elements.fingerprintValue.textContent = acgDocument.hash.slice(0, 16);
  elements.rulesetValue.textContent = `ACG v${record.version}`;
  elements.deathmatchValue.textContent = "원본 재생";
  elements.platformValue.textContent = acgFrame.platformRule ? "기록에 있음" : "없음";
  const recordedRules = [acgFrame.appliedRuleCard, ...(acgFrame.additionalRuleCards || [])].filter(Boolean);
  const rules = (recordedRules.length ? recordedRules : (record.metadata.rules || []).map(id => ({id}))).map(rule => {
    const meta = cardCatalog.get(rule.id); const tag = document.createElement("button"); tag.className = "tag"; tag.textContent = `RULE · ${meta?.name || rule.id}`;
    registerInfo(tag, meta?.name || rule.id, `${meta?.description || "기록에 저장된 룰입니다."}${meta?.help ? `\n${meta.help}` : ""}\n읽기 전용 ACG 국면`);
    tag.addEventListener("click", () => showSelectionInfo(tag.dataset.infoTitle, tag.dataset.infoDescription)); return tag;
  });
  elements.rulesList.replaceChildren(...rules);
  for (const owner of ["white", "black"]) {
    const target = owner === "white" ? elements.whiteCards : elements.blackCards;
    const cards = (acgFrame.cardState?.[owner] || []).filter(Boolean).map(card => {
      const meta = cardCatalog.get(card.id); const item = document.createElement("button"); item.className = `acg-card ${card.used && !card.passiveApplied ? "spent" : ""}`;
      const name = meta?.name || card.id;
      const state = `${card.passiveApplied ? "지속 효과" : card.used ? "사용됨" : "보유"} · 남은 횟수 ${card.usesRemaining ?? "—"}${card.nextTurnPending ? " · 다음 턴 대기" : ""}`;
      const label = document.createElement("span"); label.className = "card-name"; label.textContent = name;
      const uses = document.createElement("span"); uses.className = "card-stars"; uses.textContent = card.passiveApplied ? "∞" : card.usesRemaining ?? "—";
      item.append(label, uses); item.setAttribute("aria-label", `${name} · ${state}`);
      registerInfo(item, name, `${meta?.description || "현재 카탈로그에 설명이 없는 원본 카드입니다."}${meta?.help ? `\n${meta.help}` : ""}\n${state}\n원본 카드 상태:\n${JSON.stringify(card, null, 2)}`);
      item.addEventListener("click", () => showSelectionInfo(item.dataset.infoTitle, item.dataset.infoDescription)); return item;
    });
    target.className = cards.length ? "card-list" : "card-list empty-message";
    target.replaceChildren(...cards); if (!cards.length) target.textContent = "카드 없음";
  }
  elements.whiteStars.textContent = "ACG"; elements.blackStars.textContent = "ACG";
  elements.decisionPanel.classList.add("hidden"); elements.abilityList.replaceChildren();
  elements.eventLog.replaceChildren(...(event?.notations || []).map(n => { const li = document.createElement("li"); li.textContent = `${n.text} · ${n.description}`; return li; }));
  elements.selectedHint.textContent = "원본 ACG 재생입니다. 엔진 분석을 위해서는 정확한 상태 변환이 필요합니다.";
  elements.evalBadge.textContent = "—"; elements.evalBarFill.style.height = "50%";
  const tree = document.getElementById("notationTree");
  const nodes = [{id:0, label:"시작 국면"}, ...events.map((e,i) => ({id:i+1, label:e.notations.map(n=>`${n.moveNumber}.${n.color === "white" ? "W" : "B"} ${n.text}`).join(" · ") || e.label}))];
  tree.replaceChildren(...nodes.map(n=> { const b = document.createElement("button"); b.className = `notation-node ${n.id === acgIndex ? "selected" : ""}`; b.setAttribute("aria-current", String(n.id === acgIndex)); b.textContent = n.label; b.addEventListener("click",()=>setAcgIndex(n.id)); return b; }));
  tree.querySelector('[aria-current="true"]')?.scrollIntoView({block: "nearest", inline: "nearest"});
  renderNavigation();
  document.getElementById("acgPrevious").disabled = acgIndex === 0;
  document.getElementById("acgNext").disabled = acgIndex === events.length;
  renderBoard(); renderSuggestions();
  document.getElementById("analysisMeta").textContent = "SHA-256·기보 검증 완료. 기록된 카드/상태/RNG의 엔진 복원은 미지원입니다.";
}
function copyAcgBoard() {
  if (!acgFrame || busy || editor) return;
  const readiness = acgAnalysisReadiness(acgFrame, pieceCatalog);
  if (!readiness.boardCopyAllowed) { showToast(readiness.blockers.join("\n")); return; }
  if (!window.confirm(`${readiness.resetSummary}\n원본 ACG는 보존됩니다. 새 분석용 보드 편집을 시작할까요?`)) return;
  const cells = acgBoardCells(acgFrame, pieceCatalog).filter(c => c.piece);
  const turn = acgFrame.turn;
  editorAcgIndex = acgIndex;
  editor = new Map(cells.map(c => [c.square, {square:c.square, kind:c.piece.kind, owner:c.piece.owner}]));
  acgFrame = null;
  const select = document.getElementById("editorKind");
  select.replaceChildren(new Option("지우개", "erase"), ...pieceCatalog.map(k => new Option(`${k.name} · ${k.code}`, k.kind)));
  document.getElementById("editorTurn").value = turn;
  document.getElementById("editorCastling").checked = false;
  document.getElementById("editorDrafts").checked = false;
  document.getElementById("boardEditor").classList.remove("hidden");
  clearInteraction(); render();
}
document.getElementById("acgFile").addEventListener("change", async event => {
  const file = event.target.files[0]; event.target.value = "";
  if (!file || busy || editor) return;
  if (file.size > 4_000_000) { showToast("ACG 파일 크기 제한(4MB)을 초과했습니다."); return; }
  try {
    const text = await file.text(); const document = await decodeAcgRecord(text);
    const [pieces, cards] = await Promise.all([request("/api/pieces"), request("/api/card-catalog")]);
    pieceCatalog = pieces; cardCatalog = new Map(cards.map(card => [card.id, card]));
    ++suggestionRequest; analysisController?.abort();
    await request("/api/analysis-stop", { method: "POST", body: "{}" });
    acgDocument = document; acgOriginal = text; acgName = file.name;
    const panel = window.document.getElementById("acgPanel"); panel.classList.remove("hidden");
    window.document.getElementById("acgMetadata").textContent = `${document.record.metadata.white} – ${document.record.metadata.black} · ${document.record.metadata.result} · ${document.record.metadata.termination}`;
    window.document.getElementById("acgSeek").max = document.record.replay.events.length;
    setAcgIndex(0);
  } catch (error) { showToast(`ACG 가져오기 실패: ${error.message}`); }
});
document.getElementById("acgSeek").addEventListener("input", event => setAcgIndex(Number(event.target.value)));
document.getElementById("acgPrevious").addEventListener("click", () => setAcgIndex(acgIndex - 1));
document.getElementById("acgNext").addEventListener("click", () => setAcgIndex(acgIndex + 1));
document.getElementById("acgCopyBoard").addEventListener("click", copyAcgBoard);
document.getElementById("acgReturn").addEventListener("click", () => {
  if (editor) return;
  acgFrame = null; clearInteraction(); render(); refreshSuggestions();
});
document.getElementById("exportAnalysisButton").addEventListener("click", async () => {
  if (busy || editor || acgFrame) { showToast("엔진 분석 국면으로 돌아온 뒤 저장하세요. 원본 ACG는 왼쪽에서 저장할 수 있습니다."); return; }
  try {
    const path = await request("/api/export-path");
    const encoded = await encodeAnalysisPath(path);
    const url = URL.createObjectURL(new Blob([encoded.text], {type:"text/plain;charset=utf-8"}));
    const link = document.createElement("a"); link.href = url; link.download = `augmentia-analysis-${game.notation.cursor}.acg`; link.click(); URL.revokeObjectURL(url);
    showToast("현재 경로의 보드/카드/룰·기보를 저장했습니다. RNG·일부 상태효과는 포함하지 않습니다.");
  } catch(error) { showToast(error.message); }
});
document.getElementById("acgExport").addEventListener("click", () => {
  if (!acgOriginal) return;
  const url = URL.createObjectURL(new Blob([acgOriginal], {type:"text/plain;charset=utf-8"}));
  const a = document.createElement("a"); a.href = url; a.download = acgName; a.click(); URL.revokeObjectURL(url);
});
elements.cancelSelection.addEventListener("click", () => {
  clearInteraction();
  renderBoard();
  renderInteractionHint();
});
elements.dialogClose.addEventListener("click", () => elements.actionDialog.close());
elements.actionDialog.addEventListener("click", (event) => {
  if (event.target === elements.actionDialog) elements.actionDialog.close();
});
function dismissInfoPopover() {
  const anchor = popoverAnchor;
  hideInfoPopover(); dismissedInfoAnchor = anchor;
  if (anchor?.isConnected) anchor.focus({preventScroll: true});
}
const infoPopup = document.getElementById("infoPopover");
document.getElementById("popoverClose").addEventListener("click", dismissInfoPopover);
document.addEventListener("pointerover", event => {
  const anchor = event.target.closest?.("[data-info-title]");
  if (!anchor || anchor.contains(event.relatedTarget)) return;
  clearTimeout(popoverShowTimer); clearTimeout(popoverHideTimer);
  dismissedInfoAnchor = null;
  if (popoverAnchor !== anchor) hideInfoPopover();
  popoverShowTimer = setTimeout(() => openInfoPopover(anchor), 300);
});
document.addEventListener("pointerout", event => {
  const anchor = event.target.closest?.("[data-info-title]");
  if (anchor && !anchor.contains(event.relatedTarget) && !infoPopup.contains(event.relatedTarget)) scheduleInfoHide();
});
infoPopup.addEventListener("pointerenter", () => { clearTimeout(popoverHideTimer); clearTimeout(popoverShowTimer); });
infoPopup.addEventListener("pointerleave", scheduleInfoHide);
document.addEventListener("focusin", event => {
  const anchor = event.target.closest?.("[data-info-title]");
  if (anchor && anchor !== dismissedInfoAnchor) openInfoPopover(anchor);
  else if (!anchor && !infoPopup.contains(event.target)) hideInfoPopover();
});
document.addEventListener("focusout", event => {
  if (!infoPopup.contains(event.relatedTarget) && !event.target.contains?.(event.relatedTarget)) scheduleInfoHide();
});
document.addEventListener("pointerdown", event => {
  if (!infoPopup.contains(event.target) && !event.target.closest?.("[data-info-title]")) hideInfoPopover();
  const tools = document.querySelector(".more-tools");
  if (!tools.contains(event.target)) tools.open = false;
});
document.addEventListener("scroll", event => {
  if (!infoPopup.contains(event.target)) hideInfoPopover();
}, true);
window.addEventListener("resize", hideInfoPopover);
document.querySelector(".acg-upload").addEventListener("keydown", event => {
  if (event.key === "Enter" || event.key === " ") { event.preventDefault(); document.getElementById("acgFile").click(); }
});
document.addEventListener("keydown", (event) => {
  const anchor = event.target.closest?.("[data-info-title]");
  if (event.key === "F2" && anchor && !elements.actionDialog.open) {
    event.preventDefault(); openInfoPopover(anchor); document.getElementById("selectionDescription").focus(); return;
  }
  if (event.key !== "Escape" || elements.actionDialog.open) return;
  if (!infoPopup.classList.contains("hidden")) { dismissInfoPopover(); return; }
  if (!document.getElementById("manualSetupPanel").classList.contains("hidden")) { document.getElementById("setupClose").click(); return; }
  if (editor && !busy) { closeEditor(); return; }
  document.querySelector(".more-tools").open = false;
  clearInteraction();
  if (game) { renderBoard(); renderInteractionHint(); }
});
for (const rail of [elements.whiteCards, elements.blackCards]) {
  rail.addEventListener("wheel", event => {
    if (rail.scrollWidth <= rail.clientWidth || Math.abs(event.deltaX) >= Math.abs(event.deltaY)) return;
    event.preventDefault(); rail.scrollLeft += event.deltaY;
  }, {passive: false});
}
renderNavigation();
loadState();
