const elements = {
  board: document.querySelector("#board"),
  seedInput: document.querySelector("#seedInput"),
  newGameButton: document.querySelector("#newGameButton"),
  model1Card: document.querySelector("#model1Card"),
  model2Card: document.querySelector("#model2Card"),
  model1Move: document.querySelector("#model1Move"),
  model2Move: document.querySelector("#model2Move"),
  model1Meta: document.querySelector("#model1Meta"),
  model2Meta: document.querySelector("#model2Meta"),
  chooseModel1: document.querySelector("#chooseModel1"),
  chooseModel2: document.querySelector("#chooseModel2"),
  autoModel1: document.querySelector("#autoModel1"),
  autoModel2: document.querySelector("#autoModel2"),
  stopAutoButton: document.querySelector("#stopAutoButton"),
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
let autoModel = null;
let autoPlayTimer = null;

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
  if (busy || !game) return;
  if (!preserveAuto) setAutoModel(null);
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
  setAutoModel(null);
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
  renderHeader();
  renderSuggestions();
  renderBoard();
  renderCards("white", elements.whiteCards);
  renderCards("black", elements.blackCards);
  renderDecisionPanel();
  renderAbilities();
  renderStatus();
  renderLog();
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
  const cellMap = new Map(game.board.map((cell) => [cell.square, cell]));
  const legalMoves = selectedSquare
    ? game.actions.filter((action) => action.kind === "move" && action.from === selectedSquare)
    : [];
  const legalDestinations = new Set(legalMoves.map((action) => action.to));
  const targetSquares = new Set(
    targeting ? targeting.actions.flatMap((action) => action.targetSquares || []) : [],
  );
  const fragment = document.createDocumentFragment();

  for (let rank = 7; rank >= 0; rank -= 1) {
    for (let file = 0; file < 8; file += 1) {
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

      if (file === 0) {
        const coordinate = document.createElement("span");
        coordinate.className = "coord rank";
        coordinate.textContent = rank + 1;
        square.append(coordinate);
      }
      if (rank === 0) {
        const coordinate = document.createElement("span");
        coordinate.className = "coord file";
        coordinate.textContent = String.fromCharCode(97 + file);
        square.append(coordinate);
      }
      if (cell?.piece) square.append(renderPiece(cell.piece));
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
  element.title = pieceTitle(piece);
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
  if (busy || !game || game.status.kind !== "ongoing") return;

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
    button.disabled = !card.available;
    button.dataset.instanceId = card.instanceId;
    const stateLabel = card.openingAutoPending
      ? "첫 이동 후 자동 발동"
      : card.locked
      ? `${card.availableAfterTurn}턴까지 잠김`
      : card.charges === 0
        ? "사용 완료"
        : card.activation === "passive"
          ? "지속 효과"
          : `충전 ${card.charges}`;
    button.innerHTML = `
      <span class="card-name"></span>
      <span class="card-stars"></span>
      <span class="card-meta"></span>
    `;
    button.querySelector(".card-name").textContent = card.name;
    button.querySelector(".card-stars").textContent = `${card.stars} ★`;
    button.querySelector(".card-meta").textContent = stateLabel;
    if (card.openingAutoPending) {
      button.title = "자신의 첫 이동 직후 자동으로 발동합니다.";
    } else if (card.available) {
      button.title = "사용할 카드를 선택합니다.";
      button.addEventListener("click", () => {
        const actions = actionsByIndices(card.actionIndices);
        activateActionGroup(actions, card.name);
      });
    }
    fragment.append(button);
  });
  container.replaceChildren(fragment);
}

function renderDecisionPanel() {
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
    explanation.textContent = game.status.reason || "상단의 새 게임 버튼으로 다시 시작할 수 있습니다.";
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
    if (elements.evalBarFill) {
      elements.evalBarFill.style.height = (game.eval.whiteRatio || 50) + "%";
    }
    if (elements.evalScoreWhite) {
      elements.evalScoreWhite.textContent = game.eval.whitePercent;
    }
    if (elements.evalScoreBlack) {
      elements.evalScoreBlack.textContent = game.eval.blackPercent;
    }
    if (elements.evalBadge) {
      elements.evalBadge.textContent = game.eval.score;
    }
  }
  if (game.eval) {
    if (elements.evalBarFill) {
      elements.evalBarFill.style.height = (game.eval.whiteRatio || 50) + "%";
    }
    if (elements.evalScoreWhite) {
      elements.evalScoreWhite.textContent = game.eval.whitePercent;
    }
    if (elements.evalScoreBlack) {
      elements.evalScoreBlack.textContent = game.eval.blackPercent;
    }
    if (elements.evalBadge) {
      elements.evalBadge.textContent = game.eval.score;
    }
  }
  elements.statusBadge.textContent = game.status.label;
  elements.statusBadge.className = `status-badge ${game.status.kind === "ongoing" ? "" : "finished"}`;
  elements.rulesetValue.textContent = game.ruleset;
  elements.fingerprintValue.textContent = game.fingerprint;
  elements.fingerprintValue.title = game.fingerprint;
  elements.deathmatchValue.textContent = game.deathmatch
    ? `${game.deathmatch.quietFullTurns} / ${game.deathmatch.interval}`
    : "대기";
  elements.platformValue.textContent = game.platform || "없음";
  const tags = game.rules.map((rule) => {
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = `RULE · ${rule}`;
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
  if (actions.length === 0) return;
  if (actions.length === 1 && (actions[0].targetSquares || []).length === 0) {
    applyAction(actions[0].index);
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
    elements.selectedHint.textContent = "기물을 선택하거나 왼쪽의 사용 가능한 카드를 누르세요.";
    elements.cancelSelection.classList.add("hidden");
  }
}

function clearInteraction() {
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
  renderSuggestions();
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
  const requestId = ++suggestionRequest;
  suggestions = null;
  renderSuggestions();
  if (!game || game.status.kind !== "ongoing") {
    setAutoModel(null);
    return;
  }
  try {
    const result = await request("/api/suggestions", {
      method: "POST",
      body: JSON.stringify({ sims: 150, fingerprint }),
    });
    if (requestId !== suggestionRequest || game?.fingerprint !== result.fingerprint) return;
    suggestions = result;
    renderSuggestions();
    scheduleAutoMove();
  } catch (error) {
    if (requestId !== suggestionRequest) return;
    showToast(error.message);
    if (autoModel) setAutoModel(null);
  }
}

function renderSuggestions() {
  if (!elements.model1Card) return;
  renderSuggestion("model-1", {
    card: elements.model1Card,
    move: elements.model1Move,
    meta: elements.model1Meta,
    choose: elements.chooseModel1,
    auto: elements.autoModel1,
  });
  renderSuggestion("model-2", {
    card: elements.model2Card,
    move: elements.model2Move,
    meta: elements.model2Meta,
    choose: elements.chooseModel2,
    auto: elements.autoModel2,
  });
  elements.stopAutoButton.classList.toggle("hidden", !autoModel);
}

function renderSuggestion(modelId, view) {
  const suggestion = currentSuggestion(modelId);
  const ongoing = game?.status.kind === "ongoing";
  const ready = suggestion?.status === "ready";
  if (!ongoing) {
    view.card.dataset.state = "idle";
    view.move.textContent = game ? game.status.label : "게임 대기";
    view.meta.textContent = "진행 중인 게임이 없습니다.";
  } else if (!suggestion) {
    view.card.dataset.state = "loading";
    view.move.textContent = "계산 중…";
    view.meta.textContent = "두 모델이 같은 상태에서 동시에 탐색 중입니다.";
  } else if (ready) {
    view.card.dataset.state = "ready";
    view.move.textContent = suggestion.action.label;
    const details = [
      `방문 ${suggestion.visits}`,
      `${suggestion.elapsedMs.toFixed(0)} ms`,
    ];
    if (suggestion.warning) details.push(suggestion.warning);
    view.meta.textContent = details.join(" · ");
  } else {
    view.card.dataset.state = "error";
    view.move.textContent = "제안 불가";
    view.meta.textContent = suggestion.error || "모델 탐색에 실패했습니다.";
  }
  view.choose.disabled = busy || !ready;
  view.auto.disabled = busy || !ready;
  view.auto.classList.toggle("active", autoModel === modelId);
}

function currentSuggestion(modelId) {
  if (!suggestions || suggestions.fingerprint !== game?.fingerprint) return null;
  return suggestions.suggestions.find((item) => item.modelId === modelId) || null;
}

async function chooseSuggestion(modelId, preserveAuto = false) {
  const suggestion = currentSuggestion(modelId);
  if (busy || suggestion?.status !== "ready") return;
  await applyAction(suggestion.actionIndex, { preserveAuto });
}

function setAutoModel(modelId) {
  if (autoPlayTimer) {
    clearTimeout(autoPlayTimer);
    autoPlayTimer = null;
  }
  autoModel = modelId;
  renderSuggestions();
  if (modelId) scheduleAutoMove();
}

function startAutoModel(modelId) {
  if (!game || game.status.kind !== "ongoing") {
    showToast("진행 중인 게임이 없습니다.");
    return;
  }
  setAutoModel(modelId);
}

function scheduleAutoMove() {
  if (autoPlayTimer) clearTimeout(autoPlayTimer);
  autoPlayTimer = null;
  if (!autoModel || busy || !game || game.status.kind !== "ongoing") return;
  const suggestion = currentSuggestion(autoModel);
  if (!suggestion) return;
  if (suggestion.status !== "ready") {
    showToast(`${suggestion.name} 제안을 만들 수 없어 자동 진행을 중지했습니다.`);
    setAutoModel(null);
    return;
  }
  autoPlayTimer = setTimeout(() => {
    autoPlayTimer = null;
    chooseSuggestion(autoModel, true);
  }, 650);
}

elements.chooseModel1.addEventListener("click", () => chooseSuggestion("model-1"));
elements.chooseModel2.addEventListener("click", () => chooseSuggestion("model-2"));
elements.autoModel1.addEventListener("click", () => startAutoModel("model-1"));
elements.autoModel2.addEventListener("click", () => startAutoModel("model-2"));
elements.stopAutoButton.addEventListener("click", () => setAutoModel(null));
elements.newGameButton.addEventListener("click", newGame);
elements.cancelSelection.addEventListener("click", () => {
  clearInteraction();
  renderBoard();
  renderInteractionHint();
});
elements.dialogClose.addEventListener("click", () => elements.actionDialog.close());
elements.actionDialog.addEventListener("click", (event) => {
  if (event.target === elements.actionDialog) elements.actionDialog.close();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !elements.actionDialog.open) {
    clearInteraction();
    if (game) {
      renderBoard();
      renderInteractionHint();
    }
  }
});

loadState();
