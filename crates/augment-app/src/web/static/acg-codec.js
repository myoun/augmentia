/**
 * Standalone ACG codec, extracted from the official 2026-09-15 ruleset bundle
 * (codec lines 34964–36488, replay lines 33260–33388). No game/UI dependencies.
 * decodeAcgRecord(text, options?) and encodeAcgRecord(record, options?) are async
 * and return { record, text, hash }. options may restrict frame keys, piece types,
 * card IDs and blood-effect IDs using the official allowed* sets/iterables.
 * reconstructReplayFrame(baseFrame, events, targetIndex?) returns a detached frame;
 * index 0 is the base, index N applies the first N events (clamped).
 * Requires globalThis.crypto.subtle (secure browser context or Node 22+).
 * Encoding is semantic, not byte-preserving: official canonical piece IDs,
 * synthesized notation descriptions/IDs and visual defaults are intentional.
 * Legacy v1 decode is retained; encoding only emits genuine compact ACG v2.
 * Safety additions reject prototype keys in delta field names and bound expansion
 * to 750,000 compact values and replay-copy work to 5,000,000 frame values / 64 MiB
 * cumulative strings. Extremely large otherwise-official files can exceed these
 * stricter resource ceilings. SHA-256 detects corruption, not authenticity.
 */
function cloneObject(value, seen) {
  if (seen.has(value)) return seen.get(value);
  const output = {};
  seen.set(value, output);
  Object.entries(value).forEach(([key, entry]) => {
    if (BLOCKED_OBJECT_KEYS.has(key)) throw invalidAcg(`blocked object key: ${key}`);
    if (["undefined", "function", "symbol"].includes(typeof entry)) return;
    output[key] = cloneReplayValue(entry, seen);
  });
  return output;
}
function cloneReplayValue(value, seen = /* @__PURE__ */ new WeakMap()) {
  if (value === null || typeof value !== "object") return value;
  if (value instanceof Set) return [...value].map((entry) => cloneReplayValue(entry, seen));
  if (value instanceof Map) {
    return [...value.entries()].map(([key, entry]) => [
      cloneReplayValue(key, seen),
      cloneReplayValue(entry, seen)
    ]);
  }
  if (Array.isArray(value)) {
    if (seen.has(value)) return seen.get(value);
    const output = [];
    seen.set(value, output);
    value.forEach((entry) => output.push(cloneReplayValue(entry, seen)));
    return output;
  }
  return cloneObject(value, seen);
}
function valuesEqual(left, right) {
  if (Object.is(left, right)) return true;
  return JSON.stringify(left) === JSON.stringify(right);
}
function boardShape(board) {
  const rows = Array.isArray(board) ? board.length : 0;
  const cols = rows ? Math.max(0, ...board.map((row) => Array.isArray(row) ? row.length : 0)) : 0;
  return { rows, cols };
}
function boardCell(board, row, col) {
  return Array.isArray(board?.[row]) && col < board[row].length ? board[row][col] ?? null : null;
}
function resizedBoard(board, rows, cols) {
  const source = Array.isArray(board) ? board : [];
  return Array.from({ length: rows }, (_, row) => Array.from({ length: cols }, (_2, col) => cloneReplayValue(boardCell(source, row, col))));
}
function createReplayDelta(beforeFrame = {}, afterFrame = {}) {
  const before = cloneReplayValue(beforeFrame || {});
  const after = cloneReplayValue(afterFrame || {});
  const beforeShape = boardShape(before.board);
  const afterShape = boardShape(after.board);
  const maxRows = Math.max(beforeShape.rows, afterShape.rows);
  const maxCols = Math.max(beforeShape.cols, afterShape.cols);
  const cells = [];
  for (let row = 0; row < maxRows; row += 1) {
    for (let col = 0; col < maxCols; col += 1) {
      const previous = cloneReplayValue(boardCell(before.board, row, col));
      const next = cloneReplayValue(boardCell(after.board, row, col));
      if (!valuesEqual(previous, next)) cells.push({ row, col, before: previous, after: next });
    }
  }
  const fields = [];
  const keys = /* @__PURE__ */ new Set([...Object.keys(before), ...Object.keys(after)]);
  keys.delete("board");
  keys.forEach((key) => {
    const beforeExists = Object.prototype.hasOwnProperty.call(before, key);
    const afterExists = Object.prototype.hasOwnProperty.call(after, key);
    const previous = beforeExists ? before[key] : null;
    const next = afterExists ? after[key] : null;
    if (beforeExists === afterExists && valuesEqual(previous, next)) return;
    fields.push({
      key,
      beforeExists,
      afterExists,
      before: cloneReplayValue(previous),
      after: cloneReplayValue(next)
    });
  });
  return {
    board: {
      before: beforeShape,
      after: afterShape,
      cells
    },
    fields
  };
}
function replayDeltaHasChanges(delta) {
  if (!delta || typeof delta !== "object") return false;
  const board = delta.board || {};
  const before = board.before || {};
  const after = board.after || {};
  return Boolean(
    Number(before.rows) !== Number(after.rows) || Number(before.cols) !== Number(after.cols) || Array.isArray(board.cells) && board.cells.length || Array.isArray(delta.fields) && delta.fields.length
  );
}
function applyReplayDelta(frame = {}, delta = {}, direction = "forward") {
  const reverse = direction === "reverse" || direction === -1;
  const side = reverse ? "before" : "after";
  const existsSide = reverse ? "beforeExists" : "afterExists";
  const output = cloneReplayValue(frame || {});
  const boardDelta = delta?.board || {};
  const shape = boardDelta[side] || boardShape(output.board);
  const rows = Math.max(0, Number(shape.rows) || 0);
  const cols = Math.max(0, Number(shape.cols) || 0);
  output.board = resizedBoard(output.board, rows, cols);
  (Array.isArray(boardDelta.cells) ? boardDelta.cells : []).forEach((cell) => {
    const row = Number(cell?.row);
    const col = Number(cell?.col);
    if (!Number.isInteger(row) || !Number.isInteger(col) || row < 0 || col < 0 || row >= rows || col >= cols) return;
    output.board[row][col] = cloneReplayValue(cell[side] ?? null);
  });
  (Array.isArray(delta?.fields) ? delta.fields : []).forEach((field) => {
    const key = String(field?.key || "");
    if (!key || key === "board") return;
    if (field[existsSide] === false) {
      delete output[key];
      return;
    }
    output[key] = cloneReplayValue(field[side]);
  });
  return output;
}
function reconstructReplayFrame(baseFrame = {}, events = [], targetIndex = events.length) {
  sanitizeStructuredValue(baseFrame);
  sanitizeStructuredValue(events);
  if (events.length > ACG_MAX_EVENTS) throw invalidAcg("too many replay events");
  const safeEvents = Array.isArray(events) ? events : [];
  const end = Math.max(0, Math.min(safeEvents.length, Math.floor(Number(targetIndex) || 0)));
  let frame = cloneReplayValue(baseFrame || {});
  const budget = {};
  const limits = { ...ACG_STRUCTURE_LIMITS, maxNodes: 5e6, maxTotalStringBytes: 64 * 1024 * 1024 };
  for (let index = 0; index < end; index += 1) {
    validateDelta(safeEvents[index]?.delta, {});
    sanitizeStructuredValue(frame, limits, budget);
    frame = applyReplayDelta(frame, safeEvents[index]?.delta, "forward");
  }
  return frame;
}
const ACG_DATA_MARKER = "%%ACG-DATA 2";
const ACG_LEGACY_DATA_MARKER = "%%ACG-DATA 1";
const ACG_HASH_MARKER = "%%ACG-SHA256";
const BLOCKED_OBJECT_KEYS = /* @__PURE__ */ new Set(["__proto__", "prototype", "constructor"]);
const FORBIDDEN_DATA_KEYS = /* @__PURE__ */ new Set(["html", "url", "imageUrl", "avatarUrl"]);
const ACG_ROOT_KEYS = /* @__PURE__ */ new Set(["format", "version", "recordId", "createdAt", "metadata", "replay"]);
const ACG_REPLAY_KEYS = /* @__PURE__ */ new Set(["baseFrame", "events", "notationTimeline"]);
const ACG_METADATA_KEYS = /* @__PURE__ */ new Set([
  "startedAt",
  "endedAt",
  "mode",
  "campaign",
  "rules",
  "white",
  "black",
  "result",
  "termination",
  "board",
  "history",
  "cardHistory",
  "appVersion"
]);
const ACG_NOTATION_KINDS = /* @__PURE__ */ new Set(["move", "card", "gain", "effect", "special"]);
const ACG_COLORS = /* @__PURE__ */ new Set(["white", "black"]);
const ACG_GAME_STYLES = /* @__PURE__ */ new Set(["normal", "chaos", "grand"]);
const ACG_RESULTS = /* @__PURE__ */ new Set(["1-0", "0-1", "1/2-1/2", "*"]);
const ACG_COMPLETENESS = /* @__PURE__ */ new Set(["Full", "Partial"]);
const ACG_CARD_STATE_KEYS = /* @__PURE__ */ new Set(["white", "black"]);
const ACG_SUSPICIOUS_POTION_EFFECT_IDS = new Set(["sacrificeProtection", "lastResistance", "coronationProtection", "shield", "evasion", "parry", "basicTraining", "stealth", "loyalist", "ghost", "chameleon", "submerge", "freeze", "chimera", "witchTrial", "stake", "callingCard", "emptyLunchbox", "poisonStun", "disarm", "mannerNoCapture", "saturationNoCapture", "explosive", "poisonedPawn", "queensGambitProtection", "trojanHorse", "severance", "inertia", "outpostProtection", "nullification", "recurrence"]);
const ACG_CARD_SNAPSHOT_KEYS = /* @__PURE__ */ new Set([
  "id",
  "instanceId",
  "color",
  "slot",
  "acquiredOrder",
  "used",
  "passiveApplied",
  "recovering",
  "nextTurnPending",
  "nextTurnPendingSinceTurn",
  "deckCard",
  "campaignCard",
  "firstTurnCard",
  "devCard",
  "disabled",
  "usesRemaining",
  "bloodEffectId",
  "bloodRevealed",
  "boxRevealedCardId",
  "randomRouletteResultType",
  "suspiciousPotionResultId"
]);
const ACG_CARD_BOOLEAN_FIELDS = [
  "used",
  "passiveApplied",
  "recovering",
  "nextTurnPending",
  "deckCard",
  "campaignCard",
  "firstTurnCard",
  "devCard",
  "disabled",
  "bloodRevealed"
];
const ACG_REPLAY_VISUAL_TYPES = /* @__PURE__ */ new Set([
  "move",
  "card",
  "trolley",
  "random-roulette",
  "time-is-mine",
  "mistake",
  "parry",
  "evasion",
  "shield-break",
  "merchant-buy",
  "collapse",
  "collapse-warning",
  "monster-move",
  "board-change",
  "rule-bomb-sweep",
  "reaper-execution",
  "magic",
  "effect"
]);
const ACG_FORMAT = "augmentchess-acg";
const ACG_LEGACY_VERSION = 1;
const ACG_NOTATION_KIND_LIST = ["move", "card", "gain", "effect", "special"];
const ACG_COMPACT_ARRAY = 1;
const ACG_COMPACT_OBJECT = 0;
const ACG_COMPACT_PIECE = 2;
const ACG_PATCH_OBJECT = 3;
const ACG_PATCH_ARRAY = 4;
const ACG_PATCH_DELETE = 5;
const COMPACT_DELETE_RESULT = /* @__PURE__ */ Symbol("compact-delete");
const ACG_COMPACT_MAX_KEYS = 4096;
const ACG_COMPACT_MAX_STRINGS = 4096;
const ACG_COMPACT_MAX_STRUCTURES = 256;
const ACG_FRAME_KEY_LIST = "mode onlineGameStarted campaign turn board captures winner replayStartedAt replayEndedAt replayEndReason fullMove moveCount appliedRuleCard additionalRuleCards pendingRuleTickets effects palaces regency retreat kingDead skipTurn enPassant enPassantFrenzy taunt hallucination pendingPawnStorm pendingPanic pendingFreeMoves pendingIcbm pendingTrojanHorse pendingTrolley pendingScarecrows activeTrolley lastMove drawOffer turnsTaken cardsUsedThisTurn actionsRemaining pendingPromotion coronation earlyPromotion fastGrowth pawnSprint pawnLeap pawnConversion racingKing radicalCharge conscription conscriptionUsed encouragement ironMonarch imperialStudies religiousVictory fianchetto freeCastling switcheroo substitution cornerKick chainBonds moving castlingCanceled castled bishopSnipe backwardKnight trojanHorse madHorse knightInjury fileSurge underpromotion finalWeapon afterimageQueen recycling breakthroughPawns quantumPending highGround legacyRemovedRuleSlot2 highway blackHole winterKingdom initiative machoChess feudalContracts pendingFeudalStrike coolGuy capturedTypes turnCaptures lastTurnCaptures kingKnight royalKnightKing knightmate monochromeChess acceleration accelerationPendingFor accelerationPendingTurns socialism ultimatum prophecy collapsePending collapsed delayedHazards ruleOpeningEnabled ruleSelectionEnabled selectedRuleCardId shotgunDlc shotgunOpeningColor firstMoveCardsForced diceLocks starWinLimit deathmatchEnabled deathmatchLimitTurns deathmatch middleDraftDone endDraftDone endPhaseStartMove temporaryQueens necromancy accelerationTrail cardState clonePassive clonedPassiveCards vanishing collapseDepth periodicCollapse ruleBombs portalRule mistakeRule legacyRemovedRuleSlot transcendenceRule binaMate camouflageRule overwhelm saturationRule royalCommand gameStyle crownRule conveyorRule armistice pendingPortals exhaustion legacyRemovedCardSlot3 democracy mistakeCard freeMoveCaptureLock selectedRuleCardIds queensGambitFiles judgmentExiles frontlineResponse relay fieldPromotion magicGirlSurge moveReplay gomoku gomokuVictoryCells sirenExposure undeadResurrections pendingLobsters collapsedCells majesty infiltration killerKing resolve vanguard overtake pendingRecurrences resolveReady resolveSpentTurn resolveMoveCredit reversal captureTheFlag accelerationStartsAfterBlackTurns".split(" ");
const ACG_REMOVED_FRAME_KEYS = /* @__PURE__ */ new Set(["legacyRemovedRuleSlot", "legacyRemovedRuleSlot2", "legacyRemovedCardSlot3"]);
const ACG_FRAME_KEY_IDS = new Map(ACG_FRAME_KEY_LIST.map((key, index) => [key, index]));
const ACG_VERSION = 2;
const ACG_MAX_BYTES = 32 * 1024 * 1024;
const ACG_MAX_EVENTS = 2e4;
const ACG_STRUCTURE_LIMITS = Object.freeze({
  maxDepth: 24,
  maxNodes: 75e4,
  maxArrayLength: ACG_MAX_EVENTS * 4,
  maxObjectKeys: 256,
  maxStringLength: 8192,
  maxTotalStringBytes: 16 * 1024 * 1024
});
function isRecord(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}
function utf8Bytes(value) {
  return new TextEncoder().encode(String(value || ""));
}
function invalidAcg(reason) {
  return new Error(`Invalid ACG file: ${reason}`);
}
function requireRecord(value, label) {
  if (!isRecord(value)) throw invalidAcg(`${label} must be an object`);
  return value;
}
function requireOnlyKeys(value, allowed, label) {
  Object.keys(value).forEach((key) => {
    if (!allowed.has(key)) throw invalidAcg(`${label} contains an unknown field: ${key}`);
  });
}
function cleanText(value, limit, fallback = "") {
  const text = String(value ?? fallback).replace(/[\r\n\u0000-\u001f\u007f]+/g, " ").trim();
  return text.slice(0, limit);
}
function validateTimestamp(value, label, optional = false) {
  const text = cleanText(value, 40);
  if (!text && optional) return "";
  if (!text || !Number.isFinite(Date.parse(text))) throw invalidAcg(`${label} is not a valid timestamp`);
  return new Date(text).toISOString();
}
function sanitizeStructuredValue(root, limits = ACG_STRUCTURE_LIMITS, budget = null) {
  let nodes = budget?.nodes || 0;
  let totalStringBytes = budget?.totalStringBytes || 0;
  const stack = [{ value: root, depth: 0, key: "" }];
  while (stack.length) {
    const { value, depth, key } = stack.pop();
    nodes += 1;
    if (nodes > limits.maxNodes) throw invalidAcg("too many values");
    if (depth > limits.maxDepth) throw invalidAcg("nesting is too deep");
    if (FORBIDDEN_DATA_KEYS.has(key)) throw invalidAcg(`forbidden data field: ${key}`);
    if (typeof value === "string") {
      if (value.length > limits.maxStringLength) throw invalidAcg("string is too long");
      if (/<\/?[a-z!][^>]*>/i.test(value) || /(?:javascript|data|https?):\s*/i.test(value)) {
        throw invalidAcg("executable markup or external URL is not allowed");
      }
      totalStringBytes += utf8Bytes(value).byteLength;
      if (totalStringBytes > limits.maxTotalStringBytes) throw invalidAcg("string data is too large");
      continue;
    }
    if (value === null || typeof value === "boolean") continue;
    if (typeof value === "number") {
      if (!Number.isFinite(value)) throw invalidAcg("number is not finite");
      continue;
    }
    if (Array.isArray(value)) {
      if (value.length > limits.maxArrayLength) throw invalidAcg("array is too long");
      for (let index = value.length - 1; index >= 0; index -= 1) {
        stack.push({ value: value[index], depth: depth + 1, key: String(index) });
      }
      continue;
    }
    if (!isRecord(value)) throw invalidAcg("unsupported value type");
    const keys = Object.keys(value);
    if (keys.length > limits.maxObjectKeys) throw invalidAcg("object has too many fields");
    keys.forEach((childKey) => {
      if (BLOCKED_OBJECT_KEYS.has(childKey)) throw invalidAcg(`blocked object key: ${childKey}`);
      stack.push({ value: value[childKey], depth: depth + 1, key: childKey });
    });
  }
  if (budget) Object.assign(budget, { nodes, totalStringBytes });
}
function normalizeNotation(value) {
  const source = requireRecord(value, "notation");
  const text = cleanText(source.text, 96);
  if (!text) throw invalidAcg("notation text is empty");
  const kind = ACG_NOTATION_KINDS.has(source.kind) ? source.kind : "special";
  const color2 = ACG_COLORS.has(source.color) ? source.color : "";
  const notation = {
    id: cleanText(source.id, 80),
    kind,
    color: color2,
    text,
    description: cleanText(source.description || text, 180, text)
  };
  if (Number.isFinite(Number(source.moveNumber))) {
    notation.moveNumber = Math.max(0, Math.floor(Number(source.moveNumber)));
  }
  return notation;
}
function normalizeTimelineEntry(value, maxReplayIndex) {
  const source = requireRecord(value, "notation timeline entry");
  const replayIndex = Math.floor(Number(source.replayIndex));
  if (!Number.isInteger(replayIndex) || replayIndex < 0 || replayIndex > maxReplayIndex) {
    throw invalidAcg("notation replay index is out of range");
  }
  const notation = normalizeNotation(source.notation);
  return {
    moveNumber: Math.max(0, Math.floor(Number(source.moveNumber) || 0)),
    color: ACG_COLORS.has(source.color) ? source.color : notation.color || "white",
    notation,
    replayIndex
  };
}
function normalizeBoardShape(value, label) {
  const source = requireRecord(value, label);
  const rows = Math.floor(Number(source.rows));
  const cols = Math.floor(Number(source.cols));
  if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows < 1 || cols < 1 || rows > 12 || cols > 12) {
    throw invalidAcg(`${label} is outside the supported board size`);
  }
  return { rows, cols };
}
function validatePiece(value, allowedPieceTypes) {
  if (value === null) return null;
  const piece2 = requireRecord(value, "board piece");
  const type = cleanText(piece2.type, 40);
  if (!type || allowedPieceTypes && !allowedPieceTypes.has(type)) {
    throw invalidAcg(`unknown piece type: ${type || "empty"}`);
  }
  if (piece2.color != null && !["white", "black", "neutral"].includes(piece2.color)) {
    throw invalidAcg("invalid piece color");
  }
  return piece2;
}
function validateBoard(board, allowedPieceTypes) {
  if (!Array.isArray(board) || board.length < 1 || board.length > 12) throw invalidAcg("invalid base board height");
  const cols = Array.isArray(board[0]) ? board[0].length : 0;
  if (cols < 1 || cols > 12) throw invalidAcg("invalid base board width");
  board.forEach((row) => {
    if (!Array.isArray(row) || row.length !== cols) throw invalidAcg("invalid base board row");
    row.forEach((piece2) => validatePiece(piece2, allowedPieceTypes));
  });
}
function validateCardState(value, allowedCardIds, allowedBloodEffectIds, allowedPieceTypes) {
  if (value == null) return;
  const source = requireRecord(value, "card state");
  requireOnlyKeys(source, ACG_CARD_STATE_KEYS, "card state");
  ["white", "black"].forEach((color2) => {
    const cards2 = source[color2];
    if (!Array.isArray(cards2) || cards2.length > 32) throw invalidAcg(`invalid ${color2} card state`);
    cards2.forEach((card2, index) => {
      if (card2 === null) return;
      const snapshot = requireRecord(card2, "card snapshot");
      requireOnlyKeys(snapshot, ACG_CARD_SNAPSHOT_KEYS, "card snapshot");
      const id = cleanText(snapshot.id, 80);
      if (!id || allowedCardIds && !allowedCardIds.has(id)) throw invalidAcg(`unknown card id: ${id || "empty"}`);
      if (snapshot.color !== color2) throw invalidAcg("card owner does not match its card state");
      const slot = Number(snapshot.slot);
      if (!Number.isInteger(slot) || slot !== index) throw invalidAcg("card slot is out of range");
      if (!cleanText(snapshot.instanceId, 120)) throw invalidAcg("card instance id is empty");
      const acquiredOrder = Number(snapshot.acquiredOrder);
      if (!Number.isInteger(acquiredOrder) || acquiredOrder < 0 || acquiredOrder > 1e9) {
        throw invalidAcg("card acquisition order is invalid");
      }
      ACG_CARD_BOOLEAN_FIELDS.forEach((field) => {
        if (snapshot[field] != null && typeof snapshot[field] !== "boolean") {
          throw invalidAcg(`card ${field} state is invalid`);
        }
      });
      if (snapshot.usesRemaining != null) {
        const usesRemaining = Number(snapshot.usesRemaining);
        if (!Number.isFinite(usesRemaining) || usesRemaining < 0 || usesRemaining > 1e6) {
          throw invalidAcg("card remaining uses are invalid");
        }
      }
      if (snapshot.nextTurnPendingSinceTurn != null) {
        const pendingSinceTurn = Number(snapshot.nextTurnPendingSinceTurn);
        if (!Number.isInteger(pendingSinceTurn) || pendingSinceTurn < 0 || pendingSinceTurn > 1e9) {
          throw invalidAcg("card next-turn pending turn is invalid");
        }
      }
      if (snapshot.boxRevealedCardId) {
        const revealedId = cleanText(snapshot.boxRevealedCardId, 80);
        if (!revealedId || allowedCardIds && !allowedCardIds.has(revealedId)) {
          throw invalidAcg(`unknown card id: ${revealedId || "empty"}`);
        }
      }
      if (snapshot.bloodEffectId) {
        const effectId = cleanText(snapshot.bloodEffectId, 80);
        if (!effectId || allowedBloodEffectIds && !allowedBloodEffectIds.has(effectId)) {
          throw invalidAcg(`unknown blood effect id: ${effectId || "empty"}`);
        }
      }
      if (snapshot.randomRouletteResultType != null) {
        const resultType = cleanText(snapshot.randomRouletteResultType, 40);
        if (!resultType || allowedPieceTypes && !allowedPieceTypes.has(resultType)) {
          throw invalidAcg(`unknown random roulette result type: ${resultType || "empty"}`);
        }
      }
      if (snapshot.suspiciousPotionResultId != null) {
        const effectId = cleanText(snapshot.suspiciousPotionResultId, 40);
        if (!effectId || !ACG_SUSPICIOUS_POTION_EFFECT_IDS.has(effectId)) {
          throw invalidAcg(`unknown suspicious potion result id: ${effectId || "empty"}`);
        }
      }
    });
  });
}
function validateKnownCardId(value, allowedCardIds, label) {
  const id = cleanText(value, 80);
  if (!id || allowedCardIds && !allowedCardIds.has(id)) {
    throw invalidAcg(`unknown card id in ${label}: ${id || "empty"}`);
  }
}
function validateCardReferenceField(key, value, allowedCardIds) {
  if (key === "gameStyle") {
    if (value != null && !ACG_GAME_STYLES.has(value)) throw invalidAcg("unsupported game style");
    return;
  }
  if (key === "appliedRuleCard") {
    if (value == null) return;
    validateKnownCardId(requireRecord(value, key).id, allowedCardIds, key);
    return;
  }
  if (key === "additionalRuleCards") {
    if (!Array.isArray(value) || value.length > 32) throw invalidAcg("invalid additional rule cards");
    value.forEach((card2) => validateKnownCardId(requireRecord(card2, key).id, allowedCardIds, key));
    return;
  }
  if (key === "selectedRuleCardId") {
    if (value) validateKnownCardId(value, allowedCardIds, key);
    return;
  }
  if (key === "selectedRuleCardIds") {
    if (!Array.isArray(value) || value.length > 64 || new Set(value).size !== value.length) {
      throw invalidAcg("invalid selected rule card ids");
    }
    value.forEach((cardId) => {
      if (cardId !== "__none__") validateKnownCardId(cardId, allowedCardIds, key);
    });
    return;
  }
  if (key === "pendingRuleTickets") {
    if (!Array.isArray(value) || value.length > 32) throw invalidAcg("invalid pending rule tickets");
    value.forEach((ticket) => validateKnownCardId(requireRecord(ticket, key).ruleId, allowedCardIds, key));
  }
}
function validateReplayFrame(frame, options) {
  const source = requireRecord(frame, "replay base frame");
  if (options.allowedFrameKeys) {
    Object.keys(source).forEach((key) => {
      if (key !== "cardState" && !options.allowedFrameKeys.has(key)) throw invalidAcg(`unknown replay frame field: ${key}`);
    });
  }
  validateBoard(source.board, options.allowedPieceTypes);
  validateCardState(source.cardState, options.allowedCardIds, options.allowedBloodEffectIds, options.allowedPieceTypes);
  Object.entries(source).forEach(([key, value]) => validateCardReferenceField(key, value, options.allowedCardIds));
  return source;
}
function validateDelta(delta, options) {
  const source = requireRecord(delta, "replay delta");
  requireOnlyKeys(source, /* @__PURE__ */ new Set(["board", "fields"]), "replay delta");
  const board = requireRecord(source.board, "replay board delta");
  requireOnlyKeys(board, /* @__PURE__ */ new Set(["before", "after", "cells"]), "replay board delta");
  const before = normalizeBoardShape(board.before, "replay board before shape");
  const after = normalizeBoardShape(board.after, "replay board after shape");
  if (!Array.isArray(board.cells) || board.cells.length > 144) throw invalidAcg("too many changed board cells");
  board.cells.forEach((cell) => {
    const entry = requireRecord(cell, "replay board cell");
    const row = Number(entry.row);
    const col = Number(entry.col);
    const maxRows = Math.max(before.rows, after.rows);
    const maxCols = Math.max(before.cols, after.cols);
    if (!Number.isInteger(row) || !Number.isInteger(col) || row < 0 || col < 0 || row >= maxRows || col >= maxCols) {
      throw invalidAcg("changed board cell is out of range");
    }
    validatePiece(entry.before ?? null, options.allowedPieceTypes);
    validatePiece(entry.after ?? null, options.allowedPieceTypes);
  });
  if (!Array.isArray(source.fields) || source.fields.length > 256) throw invalidAcg("too many changed replay fields");
  const seenFields = /* @__PURE__ */ new Set();
  source.fields.forEach((field) => {
    const entry = requireRecord(field, "replay field delta");
    const key = cleanText(entry.key, 80);
    if (!key || key !== entry.key || !validCompactKeyName(key) || key === "board" || seenFields.has(key)) throw invalidAcg("invalid or duplicate replay field");
    if (options.allowedFrameKeys && key !== "cardState" && !options.allowedFrameKeys.has(key)) {
      throw invalidAcg(`unknown replay field: ${key}`);
    }
    seenFields.add(key);
    if (key === "cardState") {
      if (entry.beforeExists !== false) validateCardState(entry.before, options.allowedCardIds, options.allowedBloodEffectIds, options.allowedPieceTypes);
      if (entry.afterExists !== false) validateCardState(entry.after, options.allowedCardIds, options.allowedBloodEffectIds, options.allowedPieceTypes);
    }
    if (key !== "cardState") {
      if (entry.beforeExists !== false) validateCardReferenceField(key, entry.before, options.allowedCardIds);
      if (entry.afterExists !== false) validateCardReferenceField(key, entry.after, options.allowedCardIds);
    }
  });
  return source;
}
function validateReplayVisual(value, options) {
  const visual = requireRecord(value, "replay visual");
  if (!ACG_REPLAY_VISUAL_TYPES.has(visual.type)) {
    throw invalidAcg(`unknown replay visual type: ${cleanText(visual.type, 40)}`);
  }
  if (visual.type === "card") {
    const animation = requireRecord(visual.animation, "card replay visual");
    const id = cleanText(animation.id, 80);
    if (!id || options.allowedCardIds && !options.allowedCardIds.has(id)) throw invalidAcg(`unknown card id: ${id || "empty"}`);
  }
  if (visual.type === "evasion") {
    const validPosition = (position) => {
      if (!position || typeof position !== "object" || Array.isArray(position)) return false;
      return Number.isInteger(position.row) && Number.isInteger(position.col) && position.row >= 0 && position.row <= 11 && position.col >= 0 && position.col <= 11;
    };
    if (!ACG_COLORS.has(visual.color) || !validPosition(visual.from) || !validPosition(visual.to) || visual.from.row === visual.to.row && visual.from.col === visual.to.col) {
      throw invalidAcg("invalid evasion replay visual");
    }
  }
  return visual;
}
function normalizeReplayEvent(value, options) {
  const source = requireRecord(value, "replay event");
  const notations2 = Array.isArray(source.notations) ? source.notations.map(normalizeNotation) : [];
  if (notations2.length > 32) throw invalidAcg("too many notations in one replay event");
  const visuals = Array.isArray(source.visuals) ? source.visuals.map((visual) => validateReplayVisual(visual, options)) : [];
  if (visuals.length > 64) throw invalidAcg("too many visuals in one replay event");
  return {
    id: cleanText(source.id, 80),
    label: cleanText(source.label, 80, "event"),
    moveNumber: Math.max(0, Math.floor(Number(source.moveNumber) || 0)),
    color: ACG_COLORS.has(source.color) ? source.color : "white",
    notations: notations2,
    visuals,
    delta: validateDelta(source.delta, options)
  };
}
function normalizedNotationSignature(notation) {
  return JSON.stringify([
    notation.id,
    notation.kind,
    notation.color,
    notation.text,
    notation.description,
    Object.prototype.hasOwnProperty.call(notation, "moveNumber") ? notation.moveNumber : null
  ]);
}
function validateNotationTimeline(events, timeline) {
  timeline.forEach((entry) => {
    if (entry.notation.color && entry.notation.color !== entry.color) {
      throw invalidAcg("notation color does not match its timeline entry");
    }
    if (Object.prototype.hasOwnProperty.call(entry.notation, "moveNumber") && entry.notation.moveNumber !== entry.moveNumber) {
      throw invalidAcg("notation move number does not match its timeline entry");
    }
  });
  events.forEach((event, index) => {
    const replayIndex = index + 1;
    const eventNotations = event.notations.map(normalizedNotationSignature).sort();
    const timelineNotations = timeline.filter((entry) => entry.replayIndex === replayIndex).map((entry) => normalizedNotationSignature(entry.notation)).sort();
    if (eventNotations.length !== timelineNotations.length || eventNotations.some((signature, notationIndex) => signature !== timelineNotations[notationIndex])) {
      throw invalidAcg(`replay notation timeline does not match event ${replayIndex}`);
    }
  });
}
function normalizeMetadata(value) {
  const source = requireRecord(value, "metadata");
  requireOnlyKeys(source, ACG_METADATA_KEYS, "metadata");
  const result = cleanText(source.result, 12, "*");
  const history = cleanText(source.history, 12, "Full");
  const cardHistory = cleanText(source.cardHistory, 12, history);
  if (!ACG_RESULTS.has(result)) throw invalidAcg("unsupported result");
  if (!ACG_COMPLETENESS.has(history) || !ACG_COMPLETENESS.has(cardHistory)) throw invalidAcg("invalid history completeness");
  const rules = Array.isArray(source.rules) ? source.rules.map((rule) => cleanText(rule, 120)).filter(Boolean) : [];
  if (rules.length > 32) throw invalidAcg("too many rules");
  return {
    startedAt: validateTimestamp(source.startedAt, "startedAt", true),
    endedAt: validateTimestamp(source.endedAt, "endedAt", true),
    mode: cleanText(source.mode, 80, "Augment Chess"),
    campaign: cleanText(source.campaign, 120),
    rules,
    white: cleanText(source.white, 80, "WHITE") || "WHITE",
    black: cleanText(source.black, 80, "BLACK") || "BLACK",
    result,
    termination: cleanText(source.termination, 500),
    board: cleanText(source.board, 20, "8x8"),
    history,
    cardHistory,
    appVersion: cleanText(source.appVersion, 40)
  };
}
function normalizeAcgRecordForVersions(value, options = {}, supportedVersions = /* @__PURE__ */ new Set([ACG_VERSION])) {
  const source = requireRecord(value, "root");
  requireOnlyKeys(source, ACG_ROOT_KEYS, "root");
  if (source.format !== ACG_FORMAT) throw invalidAcg("unsupported format");
  const version = Number(source.version);
  if (!supportedVersions.has(version)) throw invalidAcg("unsupported format version");
  const replay = requireRecord(source.replay, "replay");
  requireOnlyKeys(replay, ACG_REPLAY_KEYS, "replay");
  if (!Array.isArray(replay.events) || replay.events.length > ACG_MAX_EVENTS) throw invalidAcg("too many replay events");
  sanitizeStructuredValue(value);
  const normalizedOptions = {
    allowedFrameKeys: options.allowedFrameKeys ? new Set(options.allowedFrameKeys) : null,
    allowedPieceTypes: options.allowedPieceTypes ? new Set(options.allowedPieceTypes) : null,
    allowedCardIds: options.allowedCardIds ? new Set(options.allowedCardIds) : null,
    allowedBloodEffectIds: options.allowedBloodEffectIds ? new Set(options.allowedBloodEffectIds) : null
  };
  const events = replay.events.map((event) => normalizeReplayEvent(event, normalizedOptions));
  if (!Array.isArray(replay.notationTimeline) || replay.notationTimeline.length > ACG_MAX_EVENTS * 8) {
    throw invalidAcg("too many notation entries");
  }
  const notationTimeline = replay.notationTimeline.map((entry) => normalizeTimelineEntry(entry, events.length));
  validateNotationTimeline(events, notationTimeline);
  return {
    format: ACG_FORMAT,
    version,
    recordId: cleanText(source.recordId, 80),
    createdAt: validateTimestamp(source.createdAt, "createdAt"),
    metadata: normalizeMetadata(source.metadata),
    replay: {
      baseFrame: validateReplayFrame(replay.baseFrame, normalizedOptions),
      events,
      notationTimeline
    }
  };
}
function normalizeAcgRecord(value, options = {}) {
  return normalizeAcgRecordForVersions(value, options, /* @__PURE__ */ new Set([ACG_VERSION]));
}
function escapeHeaderValue(value) {
  return cleanText(value, 500).replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}
function headerLine(name, value) {
  return `[${name} "${escapeHeaderValue(value)}"]`;
}
function formatAcgMovetext(timeline = []) {
  const grouped = [];
  const groupByKey = /* @__PURE__ */ new Map();
  timeline.forEach((entry, order) => {
    const notation = entry?.notation;
    if (!notation?.text) return;
    const replayIndex = Math.max(0, Math.floor(Number(entry.replayIndex) || 0));
    const moveNumber = Math.max(0, Math.floor(Number(entry.moveNumber) || Number(notation.moveNumber) || 0));
    const color2 = entry.color === "black" || notation.color === "black" ? "black" : "white";
    const key = `${replayIndex}:${moveNumber}:${color2}`;
    let group = groupByKey.get(key);
    if (!group) {
      group = { replayIndex, moveNumber, color: color2, order, texts: [] };
      groupByKey.set(key, group);
      grouped.push(group);
    }
    group.texts.push(cleanText(notation.text, 96));
  });
  return grouped.sort((left, right) => left.replayIndex - right.replayIndex || left.order - right.order).map((entry) => `${entry.moveNumber}.${entry.color === "black" ? "B" : "W"} ${entry.texts.join(" ")}`).join("\n");
}
function formatAcgHumanSection(record2) {
  const metadata = record2.metadata;
  const headers = [
    headerLine("ACG", record2.version),
    headerLine("RecordId", record2.recordId),
    headerLine("CreatedAt", record2.createdAt),
    headerLine("StartedAt", metadata.startedAt || "-"),
    headerLine("EndedAt", metadata.endedAt || "-"),
    headerLine("Mode", metadata.mode),
    headerLine("Campaign", metadata.campaign || "-"),
    headerLine("White", metadata.white),
    headerLine("Black", metadata.black),
    headerLine("Result", metadata.result),
    headerLine("Termination", metadata.termination || "-"),
    headerLine("Board", metadata.board),
    headerLine("History", metadata.history),
    headerLine("CardHistory", metadata.cardHistory),
    headerLine("AppVersion", metadata.appVersion || "-")
  ];
  metadata.rules.forEach((rule) => headers.push(headerLine("Rule", rule)));
  const movetext = formatAcgMovetext(record2.replay.notationTimeline) || "-";
  return `${headers.join("\n")}

${movetext}

${metadata.result}`;
}
async function sha256Hex(value) {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) throw new Error("SHA-256 is unavailable in this browser");
  const digest = await subtle.digest("SHA-256", utf8Bytes(value));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}
function addFrequency(map, value) {
  map.set(value, (map.get(value) || 0) + 1);
}
function hasNestedString(value) {
  if (typeof value === "string") return true;
  if (!value || typeof value !== "object") return false;
  if (Array.isArray(value)) return value.some(hasNestedString);
  return Object.values(value).some(hasNestedString);
}
function collectCompactStrings(value, stats) {
  if (typeof value === "string") {
    addFrequency(stats.strings, value);
    return;
  }
  if (!value || typeof value !== "object") return;
  if (Array.isArray(value)) {
    value.forEach((entry) => collectCompactStrings(entry, stats));
    return;
  }
  Object.values(value).forEach((entry) => collectCompactStrings(entry, stats));
}
function isCompactPiece(value) {
  return isRecord(value) && typeof value.id === "string" && typeof value.type === "string" && ["white", "black", "neutral"].includes(value.color);
}
function compactPieceExtras(piece2) {
  const extras = {};
  Object.entries(piece2).forEach(([key, value]) => {
    if (["id", "type", "color"].includes(key)) return;
    if (["moved", "shielded"].includes(key) && value === false) return;
    extras[key] = value;
  });
  return extras;
}
function collectReplayPieceIds(value, ids) {
  if (!value || typeof value !== "object") return;
  if (isCompactPiece(value) && !ids.has(value.id)) ids.set(value.id, ids.size);
  if (Array.isArray(value)) {
    value.forEach((entry) => collectReplayPieceIds(entry, ids));
    return;
  }
  Object.values(value).forEach((entry) => collectReplayPieceIds(entry, ids));
}
function replaceReplayPieceIds(value, ids) {
  if (typeof value === "string") {
    const index = ids.get(value);
    return index == null ? value : `p${index.toString(36)}`;
  }
  if (!value || typeof value !== "object") return value;
  if (Array.isArray(value)) return value.map((entry) => replaceReplayPieceIds(entry, ids));
  const output = {};
  Object.entries(value).forEach(([key, entry]) => {
    if (isCompactPiece(value) && ["moved", "shielded"].includes(key) && entry === false) return;
    output[key] = replaceReplayPieceIds(entry, ids);
  });
  return output;
}
function canonicalizeCompactRecord(record2) {
  const output = cloneReplayValue(record2);
  const pieceIds = /* @__PURE__ */ new Map();
  collectReplayPieceIds(output.replay, pieceIds);
  output.replay = replaceReplayPieceIds(output.replay, pieceIds);
  return output;
}
function collectCompactValue(value, stats) {
  if (typeof value === "string") {
    addFrequency(stats.strings, value);
    return;
  }
  if (!value || typeof value !== "object") return;
  if (isCompactPiece(value)) {
    addFrequency(stats.strings, value.type);
    const extras = compactPieceExtras(value);
    Object.entries(extras).forEach(([key, entry]) => {
      addFrequency(stats.keys, key);
      collectCompactValue(entry, stats);
    });
    return;
  }
  const serialized = JSON.stringify(value);
  const size = utf8Bytes(serialized).byteLength;
  if (!hasNestedString(value) && size >= 8 && size <= 512) {
    const current = stats.structures.get(serialized);
    stats.structures.set(serialized, {
      count: (current?.count || 0) + 1,
      size,
      value: current?.value || value
    });
  }
  if (Array.isArray(value)) {
    value.forEach((entry) => collectCompactValue(entry, stats));
    return;
  }
  Object.entries(value).forEach(([key, entry]) => {
    addFrequency(stats.keys, key);
    collectCompactValue(entry, stats);
  });
}
function collectNotationStrings(notation, stats) {
  addFrequency(stats.strings, notation.text);
}
function collectCompactVisual(visual, stats) {
  if (visual?.type === "move" && visual.move) {
    [visual.move.soundName, visual.move.pieceId, visual.move.pieceType, visual.move.hiddenFrom].filter((value) => typeof value === "string" && value).forEach((value) => addFrequency(stats.strings, value));
    return;
  }
  if (visual?.type === "card" && visual.animation) {
    [visual.animation.id, visual.animation.name].filter((value) => typeof value === "string" && value).forEach((value) => addFrequency(stats.strings, value));
    return;
  }
  collectCompactValue(visual, stats);
}
function compactReferenceLength(prefix, index) {
  return utf8Bytes(JSON.stringify(`${prefix}${index.toString(36)}`)).byteLength;
}
function createCompactContext(record2) {
  const stats = {
    keys: /* @__PURE__ */ new Map(),
    strings: /* @__PURE__ */ new Map(),
    structures: /* @__PURE__ */ new Map()
  };
  addFrequency(stats.strings, record2.recordId);
  addFrequency(stats.strings, record2.createdAt);
  collectCompactStrings([
    record2.metadata.startedAt,
    record2.metadata.endedAt,
    record2.metadata.mode,
    record2.metadata.campaign,
    record2.metadata.rules,
    record2.metadata.white,
    record2.metadata.black,
    record2.metadata.result,
    record2.metadata.termination,
    record2.metadata.board,
    record2.metadata.appVersion
  ], stats);
  record2.replay.baseFrame.board.flat().filter(Boolean).forEach((piece2) => collectCompactValue(piece2, stats));
  Object.entries(record2.replay.baseFrame).forEach(([key, value]) => {
    if (key === "board") return;
    collectCompactValue(value, stats);
  });
  record2.replay.events.forEach((event) => {
    addFrequency(stats.strings, event.label);
    event.notations.forEach((notation) => collectNotationStrings(notation, stats));
    event.visuals.forEach((visual) => collectCompactVisual(visual, stats));
    event.delta.board.cells.forEach((cell) => collectCompactValue(cell.after ?? null, stats));
    event.delta.fields.forEach((field) => {
      if (field.afterExists !== false) collectCompactValue(field.after, stats);
    });
  });
  record2.replay.notationTimeline.filter((entry) => entry.replayIndex === 0).forEach((entry) => collectNotationStrings(entry.notation, stats));
  const keyCandidates = [...stats.keys.entries()].map(([key, count]) => {
    const raw = utf8Bytes(JSON.stringify(key)).byteLength;
    const estimatedReference = utf8Bytes("0").byteLength;
    return { key, count, raw, score: raw * count - raw - estimatedReference * count };
  }).filter((entry) => entry.count >= 2 && entry.score > 1).sort((left, right) => right.score - left.score || left.key.localeCompare(right.key));
  const keys = [];
  keyCandidates.forEach((entry) => {
    if (keys.length >= ACG_COMPACT_MAX_KEYS) return;
    const reference = utf8Bytes(String(keys.length)).byteLength;
    if (entry.raw * entry.count > entry.raw + 1 + reference * entry.count) keys.push(entry.key);
  });
  if (keys.length > ACG_COMPACT_MAX_KEYS) throw invalidAcg("too many compact object keys");
  const stringCandidates = [...stats.strings.entries()].map(([text, count]) => {
    const raw = utf8Bytes(JSON.stringify(text)).byteLength;
    const estimatedReference = compactReferenceLength("@", 0);
    return { text, count, raw, score: raw * count - raw - estimatedReference * count };
  }).filter((entry) => entry.count >= 2 && entry.score > 2).sort((left, right) => right.score - left.score || left.text.localeCompare(right.text));
  const strings = [];
  stringCandidates.forEach((entry) => {
    if (strings.length >= ACG_COMPACT_MAX_STRINGS) return;
    const reference = compactReferenceLength("@", strings.length);
    if (entry.raw * entry.count > entry.raw + 1 + reference * entry.count) strings.push(entry.text);
  });
  const structureCandidates = [...stats.structures.entries()].map(([serialized, entry]) => {
    const estimatedReference = compactReferenceLength("^", 0);
    return {
      serialized,
      ...entry,
      score: entry.size * entry.count - entry.size - estimatedReference * entry.count
    };
  }).filter((entry) => entry.count >= 2 && entry.score > 4).sort((left, right) => right.score - left.score || left.serialized.localeCompare(right.serialized));
  const structures = [];
  structureCandidates.forEach((entry) => {
    if (structures.length >= ACG_COMPACT_MAX_STRUCTURES) return;
    const reference = compactReferenceLength("^", structures.length);
    if (entry.size * entry.count > entry.size + 1 + reference * entry.count) structures.push(entry.value);
  });
  return {
    keys,
    keyIds: new Map(keys.map((key, index) => [key, index])),
    strings,
    stringIds: new Map(strings.map((text, index) => [text, index])),
    structures,
    structureIds: new Map(structures.map((value, index) => [JSON.stringify(value), index]))
  };
}
function packCompactString(value, context) {
  const text = String(value ?? "");
  if (context.stringIds.has(text)) return `@${context.stringIds.get(text).toString(36)}`;
  if (text.startsWith("@")) return `@@${text.slice(1)}`;
  if (text.startsWith("^")) return `^^${text.slice(1)}`;
  return text;
}
function validCompactKeyName(value) {
  return typeof value === "string" && value.length > 0 && value.length <= 80 && !BLOCKED_OBJECT_KEYS.has(value) && !FORBIDDEN_DATA_KEYS.has(value);
}
function packCompactObjectKey(key, context) {
  const keyId = context.keyIds.get(key);
  return keyId == null ? key : keyId;
}
function unpackCompactObjectKey(value, context) {
  if (typeof value === "string") {
    if (!validCompactKeyName(value)) throw invalidAcg("invalid inline compact object key");
    return value;
  }
  const keyId = Number(value);
  if (!Number.isInteger(keyId) || keyId < 0 || keyId >= context.keys.length) {
    throw invalidAcg("compact object key is out of range");
  }
  return context.keys[keyId];
}
function packCompactFrameKey(key) {
  const keyId = ACG_FRAME_KEY_IDS.get(key);
  return keyId == null ? key : keyId;
}
function unpackCompactFrameKey(value) {
  if (typeof value === "string") {
    if (!validCompactKeyName(value)) throw invalidAcg("invalid inline compact frame key");
    return value;
  }
  const keyId = Number(value);
  if (!Number.isInteger(keyId) || keyId < 0 || keyId >= ACG_FRAME_KEY_LIST.length) {
    throw invalidAcg("compact frame key is out of range");
  }
  return ACG_FRAME_KEY_LIST[keyId];
}
function compactPieceColorCode(color2) {
  if (color2 === "white") return 0;
  if (color2 === "black") return 1;
  if (color2 === "neutral") return 2;
  throw invalidAcg("invalid compact piece color");
}
function compactPieceColorFromCode(value) {
  if (value === 0) return "white";
  if (value === 1) return "black";
  if (value === 2) return "neutral";
  throw invalidAcg("invalid compact piece color");
}
function packCompactPiece(piece2, context, allowStructureReferences) {
  const idMatch = /^p([0-9a-z]+)$/.exec(piece2.id);
  if (!idMatch) throw invalidAcg("piece id was not canonicalized");
  const id = Number.parseInt(idMatch[1], 36);
  if (!Number.isInteger(id) || id < 0 || id > 1e6) throw invalidAcg("compact piece id is out of range");
  const extras = compactPieceExtras(piece2);
  const packed = [
    ACG_COMPACT_PIECE,
    id,
    compactPieceColorCode(piece2.color),
    packCompactString(piece2.type, context)
  ];
  if (Object.keys(extras).length) packed.push(packCompactValue(extras, context, allowStructureReferences));
  return packed;
}
function unpackCompactPiece(value, context, depth, allowStructureReferences) {
  if (value.length < 4 || value.length > 5) throw invalidAcg("invalid compact piece");
  const id = Number(value[1]);
  if (!Number.isInteger(id) || id < 0 || id > 1e6) throw invalidAcg("compact piece id is out of range");
  const piece2 = {
    id: `p${id.toString(36)}`,
    color: compactPieceColorFromCode(Number(value[2])),
    type: readCompactString(value[3], context, "piece type")
  };
  if (value.length === 5) {
    const extras = unpackCompactValue(value[4], context, depth + 1, allowStructureReferences);
    if (!isRecord(extras)) throw invalidAcg("invalid compact piece state");
    ["id", "color", "type"].forEach((key) => {
      if (Object.prototype.hasOwnProperty.call(extras, key)) throw invalidAcg("compact piece state overrides identity");
    });
    Object.assign(piece2, extras);
  }
  return piece2;
}
function packCompactValue(value, context, allowStructureReferences = true) {
  if (value === null || typeof value === "boolean" || typeof value === "number") return value;
  if (typeof value === "string") return packCompactString(value, context);
  if (!value || typeof value !== "object") throw invalidAcg("unsupported compact value");
  if (isCompactPiece(value)) return packCompactPiece(value, context, allowStructureReferences);
  if (allowStructureReferences) {
    const reference = context.structureIds.get(JSON.stringify(value));
    if (reference != null) return `^${reference.toString(36)}`;
  }
  if (Array.isArray(value)) {
    return [ACG_COMPACT_ARRAY, ...value.map((entry) => packCompactValue(entry, context, allowStructureReferences))];
  }
  const packed = [ACG_COMPACT_OBJECT];
  Object.entries(value).forEach(([key, entry]) => {
    packed.push(packCompactObjectKey(key, context), packCompactValue(entry, context, allowStructureReferences));
  });
  return packed;
}
function parseCompactReference(value, prefix, size, label) {
  const token2 = value.slice(1);
  if (!/^[0-9a-z]+$/.test(token2)) throw invalidAcg(`invalid compact ${label} reference`);
  const index = Number.parseInt(token2, 36);
  if (!Number.isInteger(index) || index < 0 || index >= size) throw invalidAcg(`compact ${label} reference is out of range`);
  return index;
}
function unpackCompactString(value, context, depth, allowStructureReferences) {
  if (value.startsWith("@@")) return `@${value.slice(2)}`;
  if (value.startsWith("^^")) return `^${value.slice(2)}`;
  if (value.startsWith("@")) {
    return context.strings[parseCompactReference(value, "@", context.strings.length, "string")];
  }
  if (value.startsWith("^")) {
    if (!allowStructureReferences) throw invalidAcg("nested compact structure reference is not allowed");
    const index = parseCompactReference(value, "^", context.structureDefs.length, "structure");
    if (!context.structureCache.has(index)) {
      context.structureCache.set(index, unpackCompactValue(context.structureDefs[index], context, depth + 1, false));
    }
    const structure = context.structureCache.get(index);
    // Bound expansion before cloning dictionary references (not just packed input).
    sanitizeStructuredValue(structure, ACG_STRUCTURE_LIMITS, context.expansionBudget);
    return cloneReplayValue(structure);
  }
  return value;
}
function unpackCompactValue(value, context, depth = 0, allowStructureReferences = true) {
  context.expansionBudget.nodes = (context.expansionBudget.nodes || 0) + 1;
  if (context.expansionBudget.nodes > ACG_STRUCTURE_LIMITS.maxNodes) throw invalidAcg("too many expanded compact values");
  if (depth > 32) throw invalidAcg("compact value nesting is too deep");
  if (value === null || typeof value === "boolean") return value;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw invalidAcg("compact number is not finite");
    return value;
  }
  if (typeof value === "string") return unpackCompactString(value, context, depth, allowStructureReferences);
  if (!Array.isArray(value) || !value.length) throw invalidAcg("invalid compact value");
  if (value[0] === ACG_COMPACT_PIECE) {
    return unpackCompactPiece(value, context, depth, allowStructureReferences);
  }
  if (value[0] === ACG_COMPACT_ARRAY) {
    return value.slice(1).map((entry) => unpackCompactValue(entry, context, depth + 1, allowStructureReferences));
  }
  if (value[0] !== ACG_COMPACT_OBJECT || value.length % 2 !== 1) throw invalidAcg("invalid compact object");
  const output = {};
  for (let index = 1; index < value.length; index += 2) {
    const key = unpackCompactObjectKey(value[index], context);
    if (Object.prototype.hasOwnProperty.call(output, key)) throw invalidAcg("duplicate compact object key");
    output[key] = unpackCompactValue(value[index + 1], context, depth + 1, allowStructureReferences);
  }
  return output;
}
function compactValuesEqual(left, right) {
  if (Object.is(left, right)) return true;
  return JSON.stringify(left) === JSON.stringify(right);
}
function compactPackedSize(value) {
  return utf8Bytes(JSON.stringify(value)).byteLength;
}
function packCompactPatch(before, after, context, depth = 0) {
  const replacement = packCompactValue(after, context);
  if (depth >= 16) return replacement;
  if (Array.isArray(before) && Array.isArray(after)) {
    const patch = [ACG_PATCH_ARRAY, after.length];
    for (let index = 0; index < after.length; index += 1) {
      if (index < before.length && compactValuesEqual(before[index], after[index])) continue;
      patch.push(index, packCompactPatch(before[index], after[index], context, depth + 1));
    }
    if (compactPackedSize(patch) < compactPackedSize(replacement)) return patch;
  }
  if (isRecord(before) && isRecord(after) && !isCompactPiece(before) && !isCompactPiece(after)) {
    const patch = [ACG_PATCH_OBJECT];
    const keys = /* @__PURE__ */ new Set([...Object.keys(before), ...Object.keys(after)]);
    keys.forEach((key) => {
      if (!Object.prototype.hasOwnProperty.call(after, key)) {
        patch.push(packCompactObjectKey(key, context), [ACG_PATCH_DELETE]);
        return;
      }
      if (Object.prototype.hasOwnProperty.call(before, key) && compactValuesEqual(before[key], after[key])) return;
      patch.push(packCompactObjectKey(key, context), packCompactPatch(before[key], after[key], context, depth + 1));
    });
    if (compactPackedSize(patch) < compactPackedSize(replacement)) return patch;
  }
  return replacement;
}
function applyCompactPatch(before, patch, context, depth = 0) {
  if (depth > 24) throw invalidAcg("compact patch nesting is too deep");
  if (!Array.isArray(patch) || ![ACG_PATCH_OBJECT, ACG_PATCH_ARRAY, ACG_PATCH_DELETE].includes(patch[0])) {
    return unpackCompactValue(patch, context);
  }
  if (patch[0] === ACG_PATCH_DELETE) {
    if (patch.length !== 1) throw invalidAcg("invalid compact delete patch");
    return COMPACT_DELETE_RESULT;
  }
  if (patch[0] === ACG_PATCH_ARRAY) {
    if (patch.length < 2 || patch.length % 2 !== 0) throw invalidAcg("invalid compact array patch");
    const length = Number(patch[1]);
    if (!Number.isInteger(length) || length < 0 || length > ACG_STRUCTURE_LIMITS.maxArrayLength) {
      throw invalidAcg("compact array patch length is invalid");
    }
    // Array patches can expand far beyond their compact byte count.
    context.expansionBudget.nodes = (context.expansionBudget.nodes || 0) + length;
    if (context.expansionBudget.nodes > ACG_STRUCTURE_LIMITS.maxNodes) throw invalidAcg("too many expanded compact values");
    const output2 = Array.isArray(before) ? cloneReplayValue(before.slice(0, length)) : [];
    output2.length = length;
    const seen2 = /* @__PURE__ */ new Set();
    for (let index = 2; index < patch.length; index += 2) {
      const itemIndex = Number(patch[index]);
      if (!Number.isInteger(itemIndex) || itemIndex < 0 || itemIndex >= length || seen2.has(itemIndex)) {
        throw invalidAcg("invalid compact array patch index");
      }
      seen2.add(itemIndex);
      const next = applyCompactPatch(output2[itemIndex], patch[index + 1], context, depth + 1);
      if (next === COMPACT_DELETE_RESULT) throw invalidAcg("array entries cannot use a delete patch");
      output2[itemIndex] = next;
    }
    return output2;
  }
  if (patch.length % 2 !== 1) throw invalidAcg("invalid compact object patch");
  const output = isRecord(before) ? cloneReplayValue(before) : {};
  const seen = /* @__PURE__ */ new Set();
  for (let index = 1; index < patch.length; index += 2) {
    const key = unpackCompactObjectKey(patch[index], context);
    if (seen.has(key)) throw invalidAcg("duplicate compact object patch key");
    seen.add(key);
    const next = applyCompactPatch(output[key], patch[index + 1], context, depth + 1);
    if (next === COMPACT_DELETE_RESULT) delete output[key];
    else output[key] = next;
  }
  return output;
}
function compactColorCode(color2) {
  return color2 === "black" ? 1 : 0;
}
function compactColorFromCode(value, label = "color") {
  if (value === 0) return "white";
  if (value === 1) return "black";
  throw invalidAcg(`invalid compact ${label}`);
}
function compactNotationColorCode(color2) {
  if (color2 === "white") return 1;
  if (color2 === "black") return 2;
  return 0;
}
function compactNotationColorFromCode(value) {
  if (value === 0) return "";
  if (value === 1) return "white";
  if (value === 2) return "black";
  throw invalidAcg("invalid compact notation color");
}
function readCompactString(value, context, label) {
  if (typeof value !== "string") throw invalidAcg(`compact ${label} must be text`);
  return unpackCompactString(value, context, 0, true);
}
function packCompactNotation(notation, context) {
  const kind = ACG_NOTATION_KIND_LIST.indexOf(notation.kind);
  const hasMoveNumber = Object.prototype.hasOwnProperty.call(notation, "moveNumber");
  const packed = [
    -((kind >= 0 ? kind : ACG_NOTATION_KIND_LIST.length - 1) + 1),
    packCompactString(notation.text, context)
  ];
  const color2 = compactNotationColorCode(notation.color);
  if (color2 || hasMoveNumber) packed.push(color2);
  if (hasMoveNumber) packed.push(notation.moveNumber);
  return packed;
}
function compactNotationDescription(kind, text) {
  const value = String(text || "");
  const subject = value.replace(/^[+@!]/, "").replace(/#$/, "");
  if (kind === "gain") return `Gain card: ${subject}`;
  if (kind === "card") return `Use card: ${subject}`;
  if (kind === "effect") return `Automatic effect: ${subject}`;
  if (kind === "move") {
    const action = /^O-O/.test(value) ? "Castle" : value.includes("x") ? "Capture" : "Move";
    return `${action}: ${value}${value.endsWith("#") ? " (game end)" : ""}`;
  }
  return `Special action: ${value}`;
}
function unpackCompactNotation(value, context, id) {
  if (!Array.isArray(value) || value.length < 2 || value.length > 5) throw invalidAcg("invalid compact notation");
  const kindToken = Number(value[0]);
  const compactLayout = Number.isInteger(kindToken) && kindToken < 0;
  const kindIndex = compactLayout ? -kindToken - 1 : kindToken;
  if (!Number.isInteger(kindIndex) || kindIndex < 0 || kindIndex >= ACG_NOTATION_KIND_LIST.length) {
    throw invalidAcg("invalid compact notation kind");
  }
  const text = readCompactString(value[1], context, "notation");
  const kind = ACG_NOTATION_KIND_LIST[kindIndex];
  const description = compactLayout ? compactNotationDescription(kind, text) : value.length > 2 && value[2] !== 0 ? readCompactString(value[2], context, "notation description") : text;
  const colorIndex = compactLayout ? 2 : 3;
  const moveIndex = compactLayout ? 3 : 4;
  const color2 = value.length > colorIndex ? compactNotationColorFromCode(Number(value[colorIndex])) : "";
  const notation = {
    id,
    kind,
    color: color2,
    text,
    description
  };
  if (value.length > moveIndex) {
    const moveNumber = Number(value[moveIndex]);
    if (!Number.isInteger(moveNumber) || moveNumber < 0) throw invalidAcg("invalid compact notation move number");
    notation.moveNumber = moveNumber;
  }
  return notation;
}
function packCompactMetadata(metadata, context) {
  return [
    packCompactString(metadata.startedAt, context),
    packCompactString(metadata.endedAt, context),
    packCompactString(metadata.mode, context),
    packCompactString(metadata.campaign, context),
    metadata.rules.map((rule) => packCompactString(rule, context)),
    packCompactString(metadata.white, context),
    packCompactString(metadata.black, context),
    packCompactString(metadata.result, context),
    packCompactString(metadata.termination, context),
    packCompactString(metadata.board, context),
    metadata.history === "Full" ? 1 : 0,
    metadata.cardHistory === "Full" ? 1 : 0,
    packCompactString(metadata.appVersion, context)
  ];
}
function unpackCompactMetadata(value, context) {
  if (!Array.isArray(value) || value.length !== 13) throw invalidAcg("invalid compact metadata");
  if (!Array.isArray(value[4]) || value[4].length > 32) throw invalidAcg("invalid compact rules");
  if (![0, 1].includes(value[10]) || ![0, 1].includes(value[11])) throw invalidAcg("invalid compact history flag");
  return {
    startedAt: readCompactString(value[0], context, "startedAt"),
    endedAt: readCompactString(value[1], context, "endedAt"),
    mode: readCompactString(value[2], context, "mode"),
    campaign: readCompactString(value[3], context, "campaign"),
    rules: value[4].map((rule) => readCompactString(rule, context, "rule")),
    white: readCompactString(value[5], context, "white name"),
    black: readCompactString(value[6], context, "black name"),
    result: readCompactString(value[7], context, "result"),
    termination: readCompactString(value[8], context, "termination"),
    board: readCompactString(value[9], context, "board size"),
    history: value[10] ? "Full" : "Partial",
    cardHistory: value[11] ? "Full" : "Partial",
    appVersion: readCompactString(value[12], context, "app version")
  };
}
function compactBoardShape(board) {
  const rows = Array.isArray(board) ? board.length : 0;
  const cols = rows && Array.isArray(board[0]) ? board[0].length : 0;
  return { rows, cols };
}
function validateCompactBoardShape(rowsValue, colsValue, label) {
  const rows = Number(rowsValue);
  const cols = Number(colsValue);
  if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows < 1 || cols < 1 || rows > 12 || cols > 12) {
    throw invalidAcg(`compact ${label} is outside the supported board size`);
  }
  return { rows, cols };
}
function emptyCompactBoard(rows, cols) {
  return Array.from({ length: rows }, () => Array(cols).fill(null));
}
function resizeCompactBoard(board, rows, cols) {
  return Array.from({ length: rows }, (_, row) => Array.from({ length: cols }, (_2, col) => cloneReplayValue(board?.[row]?.[col] ?? null)));
}
function compactPosition(row, col) {
  return row * 12 + col;
}
function unpackCompactPosition(value, rows, cols, seen) {
  const position = Number(value);
  const row = Math.floor(position / 12);
  const col = position % 12;
  if (!Number.isInteger(position) || position < 0 || row >= rows || col >= cols || seen.has(position)) {
    throw invalidAcg("invalid or duplicate compact board position");
  }
  seen.add(position);
  return { row, col };
}
function packCompactBaseFrame(frame, context) {
  const { rows, cols } = compactBoardShape(frame.board);
  const cells = [];
  frame.board.forEach((row, rowIndex) => row.forEach((piece2, colIndex) => {
    if (!piece2) return;
    cells.push(compactPosition(rowIndex, colIndex), packCompactValue(piece2, context));
  }));
  const fields = [];
  Object.entries(frame).forEach(([key, value]) => {
    if (key === "board") return;
    fields.push(packCompactFrameKey(key), packCompactValue(value, context));
  });
  return [rows, cols, cells.length ? cells : 0, fields.length ? fields : 0];
}
function unpackCompactBaseFrame(value, context) {
  if (!Array.isArray(value) || value.length !== 4) throw invalidAcg("invalid compact base frame");
  const { rows, cols } = validateCompactBoardShape(value[0], value[1], "base board");
  const board = emptyCompactBoard(rows, cols);
  const cells = value[2] === 0 ? [] : value[2];
  if (!Array.isArray(cells) || cells.length % 2 !== 0 || cells.length > 288) throw invalidAcg("invalid compact base board cells");
  const seenCells = /* @__PURE__ */ new Set();
  for (let index = 0; index < cells.length; index += 2) {
    const position = unpackCompactPosition(cells[index], rows, cols, seenCells);
    board[position.row][position.col] = unpackCompactValue(cells[index + 1], context);
  }
  const frame = { board };
  const fields = value[3] === 0 ? [] : value[3];
  if (!Array.isArray(fields) || fields.length % 2 !== 0 || fields.length > 512) throw invalidAcg("invalid compact base frame fields");
  const seenFields = /* @__PURE__ */ new Set();
  for (let index = 0; index < fields.length; index += 2) {
    const key = unpackCompactFrameKey(fields[index]);
    if (key === "board" || seenFields.has(key)) throw invalidAcg("invalid or duplicate compact frame key");
    seenFields.add(key);
    const fieldValue = unpackCompactValue(fields[index + 1], context);
    if (!ACG_REMOVED_FRAME_KEYS.has(key)) frame[key] = fieldValue;
  }
  return frame;
}
function compactVisualPosition(value) {
  const row = Number(value?.row);
  const col = Number(value?.col);
  if (!Number.isInteger(row) || !Number.isInteger(col) || row < 0 || col < 0 || row > 11 || col > 11) return null;
  return compactPosition(row, col);
}
function compactVisualCell(value) {
  const position = Number(value);
  if (!Number.isInteger(position) || position < 0 || position > 143) throw invalidAcg("invalid compact visual position");
  return { row: Math.floor(position / 12), col: position % 12 };
}
function packCompactVisual(visual, context) {
  if (visual?.type === "move") {
    const from = compactVisualPosition(visual.move?.from);
    const to = compactVisualPosition(visual.move?.to);
    if (from != null && to != null) {
      const packed = [
        0,
        from,
        to,
        packCompactString(visual.move?.soundName || "", context),
        compactNotationColorCode(visual.move?.soundColor),
        packCompactString(visual.move?.pieceId || "", context),
        packCompactString(visual.move?.pieceType || "", context),
        packCompactString(visual.move?.hiddenFrom || "", context)
      ];
      while (packed.length > 3 && ["", 0].includes(packed[packed.length - 1])) packed.pop();
      return packed;
    }
  }
  if (visual?.type === "card" && visual.animation?.id) {
    return [
      1,
      packCompactString(visual.animation.id, context),
      packCompactString(visual.animation.name || visual.animation.id, context),
      compactNotationColorCode(visual.animation.color)
    ];
  }
  return [2, packCompactValue(visual, context)];
}
function unpackCompactVisual(value, context, eventIndex, visualIndex) {
  if (!Array.isArray(value) || !value.length) throw invalidAcg("invalid compact replay visual");
  if (value[0] === 0) {
    if (value.length < 3 || value.length > 8) throw invalidAcg("invalid compact move visual");
    const move = {
      from: compactVisualCell(value[1]),
      to: compactVisualCell(value[2]),
      pieceId: value.length > 5 ? readCompactString(value[5], context, "move piece id") : "",
      pieceType: value.length > 6 ? readCompactString(value[6], context, "move piece type") : "",
      soundName: value.length > 3 ? readCompactString(value[3], context, "move sound") : "",
      soundColor: value.length > 4 ? compactNotationColorFromCode(Number(value[4])) : "",
      hiddenFrom: value.length > 7 ? readCompactString(value[7], context, "hidden move origin") : ""
    };
    return { type: "move", move };
  }
  if (value[0] === 1) {
    if (value.length !== 4) throw invalidAcg("invalid compact card visual");
    const id = readCompactString(value[1], context, "card visual id");
    return {
      type: "card",
      animation: {
        id,
        instanceId: `${id}-acg-${eventIndex + 1}-${visualIndex + 1}`,
        name: readCompactString(value[2], context, "card visual name"),
        color: compactNotationColorFromCode(Number(value[3])),
        nonce: `acg-${eventIndex + 1}-${visualIndex + 1}`,
        rect: { left: 0, top: 0, width: 160, height: 220 }
      }
    };
  }
  if (value[0] === 2 && value.length === 2) return unpackCompactValue(value[1], context);
  throw invalidAcg("unknown compact replay visual");
}
function packCompactEvent(event, context) {
  const board = event.delta.board;
  const shapeChanged = board.before.rows !== board.after.rows || board.before.cols !== board.after.cols;
  const cells = [];
  board.cells.forEach((cell) => {
    if (cell.row >= board.after.rows || cell.col >= board.after.cols) return;
    cells.push(compactPosition(cell.row, cell.col), packCompactValue(cell.after ?? null, context));
  });
  const fields = [];
  event.delta.fields.forEach((field) => {
    const key = packCompactFrameKey(field.key);
    if (field.afterExists === false) {
      fields.push(key, []);
    } else {
      fields.push(key, packCompactPatch(field.beforeExists === false ? void 0 : field.before, field.after, context));
    }
  });
  const packed = [
    packCompactString(event.label, context),
    event.moveNumber,
    compactColorCode(event.color),
    event.notations.length ? event.notations.map((notation) => packCompactNotation(notation, context)) : 0,
    event.visuals.length ? event.visuals.map((visual) => packCompactVisual(visual, context)) : 0,
    cells.length ? cells : 0,
    fields.length ? fields : 0,
    shapeChanged ? [board.after.rows, board.after.cols] : 0
  ];
  while (packed.length > 3 && packed[packed.length - 1] === 0) packed.pop();
  return packed;
}
function unpackCompactEvent(value, context, eventIndex, currentFrame) {
  if (!Array.isArray(value) || value.length < 3 || value.length > 8) throw invalidAcg("invalid compact replay event");
  const moveNumber = Number(value[1]);
  if (!Number.isInteger(moveNumber) || moveNumber < 0) throw invalidAcg("invalid compact event move number");
  const color2 = compactColorFromCode(Number(value[2]), "event color");
  const packedNotations = value[3] == null || value[3] === 0 ? [] : value[3];
  if (!Array.isArray(packedNotations) || packedNotations.length > 32) throw invalidAcg("too many compact event notations");
  const notations2 = packedNotations.map((notation, notationIndex) => unpackCompactNotation(notation, context, `acg-${eventIndex + 1}-n${notationIndex + 1}`));
  const packedVisuals = value[4] == null || value[4] === 0 ? [] : value[4];
  if (!Array.isArray(packedVisuals) || packedVisuals.length > 64) throw invalidAcg("too many compact replay visuals");
  const visuals = packedVisuals.map((visual, visualIndex) => unpackCompactVisual(visual, context, eventIndex, visualIndex));
  const beforeShape = compactBoardShape(currentFrame.board);
  const packedShape = value[7] == null || value[7] === 0 ? null : value[7];
  const afterShape = packedShape == null ? beforeShape : Array.isArray(packedShape) && packedShape.length === 2 ? validateCompactBoardShape(packedShape[0], packedShape[1], "event board") : (() => {
    throw invalidAcg("invalid compact event board shape");
  })();
  const nextFrame = cloneReplayValue(currentFrame);
  nextFrame.board = resizeCompactBoard(currentFrame.board, afterShape.rows, afterShape.cols);
  const cells = value[5] == null || value[5] === 0 ? [] : value[5];
  if (!Array.isArray(cells) || cells.length % 2 !== 0 || cells.length > 288) throw invalidAcg("invalid compact event board cells");
  const seenCells = /* @__PURE__ */ new Set();
  for (let index = 0; index < cells.length; index += 2) {
    const position = unpackCompactPosition(cells[index], afterShape.rows, afterShape.cols, seenCells);
    nextFrame.board[position.row][position.col] = unpackCompactValue(cells[index + 1], context);
  }
  const fields = value[6] == null || value[6] === 0 ? [] : value[6];
  if (!Array.isArray(fields) || fields.length % 2 !== 0 || fields.length > 512) throw invalidAcg("invalid compact event fields");
  const seenFields = /* @__PURE__ */ new Set();
  for (let index = 0; index < fields.length; index += 2) {
    const key = unpackCompactFrameKey(fields[index]);
    if (key === "board" || seenFields.has(key)) throw invalidAcg("invalid or duplicate compact event field");
    seenFields.add(key);
    const packedFieldValue = fields[index + 1];
    if (ACG_REMOVED_FRAME_KEYS.has(key)) continue;
    if (Array.isArray(packedFieldValue) && packedFieldValue.length === 0) {
      delete nextFrame[key];
      continue;
    }
    const previous = Object.prototype.hasOwnProperty.call(nextFrame, key) ? nextFrame[key] : void 0;
    const next = applyCompactPatch(previous, packedFieldValue, context);
    if (next === COMPACT_DELETE_RESULT) throw invalidAcg("top-level field patch cannot delete implicitly");
    nextFrame[key] = next;
  }
  return {
    event: {
      id: `acg-event-${eventIndex + 1}`,
      label: readCompactString(value[0], context, "event label"),
      moveNumber,
      color: color2,
      notations: notations2,
      visuals,
      delta: createReplayDelta(currentFrame, nextFrame)
    },
    nextFrame
  };
}
function packCompactTimeline(record2, context) {
  const notationQueues = record2.replay.events.map((event) => {
    const queues = /* @__PURE__ */ new Map();
    event.notations.forEach((notation, notationIndex) => {
      const signature = normalizedNotationSignature(notation);
      if (!queues.has(signature)) queues.set(signature, []);
      queues.get(signature).push(notationIndex);
    });
    return queues;
  });
  return record2.replay.notationTimeline.map((entry) => {
    if (entry.replayIndex === 0) {
      return [0, packCompactNotation(entry.notation, context), entry.moveNumber, compactColorCode(entry.color)];
    }
    const queues = notationQueues[entry.replayIndex - 1];
    const notationIndex = queues?.get(normalizedNotationSignature(entry.notation))?.shift();
    if (!Number.isInteger(notationIndex)) throw invalidAcg("unable to compact notation timeline");
    return [entry.replayIndex, notationIndex, entry.moveNumber, compactColorCode(entry.color)];
  });
}
function unpackCompactTimeline(value, context, events) {
  if (!Array.isArray(value) || value.length > ACG_MAX_EVENTS * 8) throw invalidAcg("too many compact notation entries");
  return value.map((entry, timelineIndex) => {
    if (!Array.isArray(entry) || entry.length !== 4) throw invalidAcg("invalid compact notation timeline entry");
    const replayIndex = Number(entry[0]);
    const moveNumber = Number(entry[2]);
    if (!Number.isInteger(replayIndex) || replayIndex < 0 || replayIndex > events.length) {
      throw invalidAcg("compact notation replay index is out of range");
    }
    if (!Number.isInteger(moveNumber) || moveNumber < 0) throw invalidAcg("invalid compact timeline move number");
    const color2 = compactColorFromCode(Number(entry[3]), "timeline color");
    let notation;
    if (replayIndex === 0) {
      notation = unpackCompactNotation(entry[1], context, `acg-initial-n${timelineIndex + 1}`);
    } else {
      const notationIndex = Number(entry[1]);
      if (!Number.isInteger(notationIndex) || notationIndex < 0 || notationIndex >= events[replayIndex - 1].notations.length) {
        throw invalidAcg("compact notation reference is out of range");
      }
      notation = cloneReplayValue(events[replayIndex - 1].notations[notationIndex]);
    }
    return { moveNumber, color: color2, notation, replayIndex };
  });
}
function packCompactPayload(record2) {
  const context = createCompactContext(record2);
  const events = record2.replay.events.map((event) => packCompactEvent(event, context));
  const data = [
    packCompactString(record2.recordId, context),
    packCompactString(record2.createdAt, context),
    packCompactMetadata(record2.metadata, context),
    packCompactBaseFrame(record2.replay.baseFrame, context),
    events,
    packCompactTimeline(record2, context)
  ];
  const structures = context.structures.map((value) => packCompactValue(value, context, false));
  return [context.keys, context.strings, structures, data];
}
function unpackCompactPayload(value) {
  sanitizeStructuredValue(value);
  if (!Array.isArray(value) || value.length !== 4) throw invalidAcg("invalid compact data root");
  const [keys, strings, structureDefs, data] = value;
  if (!Array.isArray(keys) || keys.length > ACG_COMPACT_MAX_KEYS) throw invalidAcg("invalid compact key dictionary");
  if (!Array.isArray(strings) || strings.length > ACG_COMPACT_MAX_STRINGS) throw invalidAcg("invalid compact string dictionary");
  if (!Array.isArray(structureDefs) || structureDefs.length > ACG_COMPACT_MAX_STRUCTURES) {
    throw invalidAcg("invalid compact structure dictionary");
  }
  const seenKeys = /* @__PURE__ */ new Set();
  keys.forEach((key) => {
    if (typeof key !== "string" || !key || key.length > 80 || seenKeys.has(key) || BLOCKED_OBJECT_KEYS.has(key) || FORBIDDEN_DATA_KEYS.has(key)) {
      throw invalidAcg("invalid or duplicate compact dictionary key");
    }
    seenKeys.add(key);
  });
  const seenStrings = /* @__PURE__ */ new Set();
  strings.forEach((text) => {
    if (typeof text !== "string" || text.length > ACG_STRUCTURE_LIMITS.maxStringLength || seenStrings.has(text)) {
      throw invalidAcg("invalid or duplicate compact dictionary string");
    }
    seenStrings.add(text);
  });
  const context = { keys, strings, structureDefs, structureCache: new Map(), expansionBudget: {} };
  if (!Array.isArray(data) || data.length !== 6) throw invalidAcg("invalid compact record data");
  const baseFrame = unpackCompactBaseFrame(data[3], context);
  if (!Array.isArray(data[4]) || data[4].length > ACG_MAX_EVENTS) throw invalidAcg("too many compact replay events");
  const events = [];
  let currentFrame = cloneReplayValue(baseFrame);
  // Limit cumulative frame-copy work, including sparse array patch expansion.
  const frameBudget = {};
  const frameLimits = { ...ACG_STRUCTURE_LIMITS, maxNodes: 5e6, maxTotalStringBytes: 64 * 1024 * 1024 };
  data[4].forEach((packedEvent, eventIndex) => {
    sanitizeStructuredValue(currentFrame, frameLimits, frameBudget);
    const decoded = unpackCompactEvent(packedEvent, context, eventIndex, currentFrame);
    sanitizeStructuredValue(decoded.nextFrame);
    events.push(decoded.event);
    currentFrame = decoded.nextFrame;
  });
  return {
    format: ACG_FORMAT,
    version: ACG_VERSION,
    recordId: readCompactString(data[0], context, "record id"),
    createdAt: readCompactString(data[1], context, "createdAt"),
    metadata: unpackCompactMetadata(data[2], context),
    replay: {
      baseFrame,
      events,
      notationTimeline: unpackCompactTimeline(data[5], context, events)
    }
  };
}
async function encodeAcgRecord(value, options = {}) {
  const sourceRecord = normalizeAcgRecordForVersions(
    canonicalizeCompactRecord(normalizeAcgRecord(value, options)),
    options,
    /* @__PURE__ */ new Set([ACG_VERSION])
  );
  const compact = packCompactPayload(sourceRecord);
  const record2 = normalizeAcgRecordForVersions(
    unpackCompactPayload(compact),
    options,
    /* @__PURE__ */ new Set([ACG_VERSION])
  );
  const payload = JSON.stringify(compact);
  if (utf8Bytes(payload).byteLength > ACG_MAX_BYTES) throw invalidAcg("record data is too large");
  const hash = await sha256Hex(payload);
  const human = formatAcgHumanSection(record2);
  const text = `${human}

${ACG_DATA_MARKER}
${payload}
${ACG_HASH_MARKER}
${hash}
`;
  if (utf8Bytes(text).byteLength > ACG_MAX_BYTES) throw invalidAcg("file is too large");
  return { record: record2, text, hash };
}
async function ensureEncodedAcgDocument(value, options = {}) {
  if (typeof value?.text === "string" && value.text.length > 0) return value;
  if (!value?.record) throw invalidAcg("record document is missing");
  return encodeAcgRecord(value.record, options);
}
async function decodeAcgRecord(input, options = {}) {
  const normalizedText2 = String(input || "").replace(/^\uFEFF/, "").replace(/\r\n?/g, "\n");
  if (!normalizedText2 || utf8Bytes(normalizedText2).byteLength > ACG_MAX_BYTES) throw invalidAcg("file is empty or too large");
  const boundaries = [
    { version: ACG_VERSION, text: `

${ACG_DATA_MARKER}
` },
    { version: ACG_LEGACY_VERSION, text: `

${ACG_LEGACY_DATA_MARKER}
` }
  ].map((entry) => ({ ...entry, index: normalizedText2.indexOf(entry.text) })).filter((entry) => entry.index >= 0).sort((left, right) => left.index - right.index);
  const boundary = boundaries[0];
  if (!boundary) throw invalidAcg("required data markers are missing");
  const hashBoundary = `
${ACG_HASH_MARKER}
`;
  const payloadStart = boundary.index + boundary.text.length;
  const hashIndex = normalizedText2.indexOf(hashBoundary, payloadStart);
  if (hashIndex < 0) throw invalidAcg("required data markers are missing");
  const human = normalizedText2.slice(0, boundary.index).trimEnd();
  const payload = normalizedText2.slice(payloadStart, hashIndex).trim();
  const hash = normalizedText2.slice(hashIndex + hashBoundary.length).trim();
  if (!/^[a-f0-9]{64}$/.test(hash)) throw invalidAcg("invalid SHA-256 value");
  if (await sha256Hex(payload) !== hash) throw invalidAcg("SHA-256 mismatch");
  let parsed;
  try {
    parsed = JSON.parse(payload);
  } catch {
    throw invalidAcg("data block is not valid JSON");
  }
  const record2 = boundary.version === ACG_LEGACY_VERSION ? normalizeAcgRecordForVersions(parsed, options, /* @__PURE__ */ new Set([ACG_LEGACY_VERSION])) : normalizeAcgRecordForVersions(unpackCompactPayload(parsed), options, /* @__PURE__ */ new Set([ACG_VERSION]));
  if (formatAcgHumanSection(record2) !== human) throw invalidAcg("display notation does not match replay data");
  return { record: record2, text: normalizedText2.endsWith("\n") ? normalizedText2 : `${normalizedText2}
`, hash };
}

export { decodeAcgRecord, encodeAcgRecord, reconstructReplayFrame };
