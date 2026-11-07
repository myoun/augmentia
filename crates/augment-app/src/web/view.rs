use augment_core::{
    Action, ActionTarget, CardActivation, Event, GamePhase, GameState, GameStatus, MoveAction,
    Piece, PieceAbility, PieceKind, Player, Square,
};
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub(crate) fn state_view(
    state: &GameState,
    seed: u64,
    history: &[String],
    actions: &[Action],
) -> Value {
    let catalog = augment_core::latest_catalog();
    let supported = catalog.iter().filter(|card| card.supported()).count();
    let action_views = actions
        .iter()
        .enumerate()
        .map(|(index, action)| action_view(state, index, action))
        .collect::<Vec<_>>();
    let mut action_indices_by_card = BTreeMap::<u32, Vec<usize>>::new();
    for (index, action) in actions.iter().enumerate() {
        if let Action::UseCard { instance_id, .. } = action {
            action_indices_by_card
                .entry(*instance_id)
                .or_default()
                .push(index);
        }
    }

    json!({
        "ruleset": state.ruleset_id,
        "coverage": { "supported": supported, "catalog": catalog.len(), "label": format!("카드 {supported}/{}장 지원 · Standard", catalog.len()) },
        "automaticDraw": state.automatic_draw.as_ref().map(|draw| json!({ "quietTurns": draw.quiet_turns, "progressThisTurn": draw.progress_this_turn })),
        "seed": seed,
        "fingerprint": format!("{:016x}", state.fingerprint()),
        "turn": player_key(state.turn),
        "turnLabel": player_name(state.turn),
        "ply": state.ply,
        "actionsRemaining": state.actions_remaining,
        "turns": {
            "white": state.turns_taken[Player::White.index()],
            "black": state.turns_taken[Player::Black.index()],
        },
        "stars": {
            "white": format_half_stars(state.half_star_totals[Player::White.index()]),
            "black": format_half_stars(state.half_star_totals[Player::Black.index()]),
        },
        "phase": phase_view(state),
        "status": status_view(&state.status),
        "board": board_view(state),
        "actions": action_views,
        "cards": {
            "white": cards_view(state, Player::White, &action_indices_by_card),
            "black": cards_view(state, Player::Black, &action_indices_by_card),
        },
        "rules": state.applied_rules.iter().map(|id| card_name(id)).collect::<Vec<_>>(),
        "ruleCards": state.applied_rules.iter().map(|id| card_meta(id)).collect::<Vec<_>>(),
        "events": history,
        "eval": null,
        "deathmatch": state.deathmatch.map(|deathmatch| json!({
            "startedAt": deathmatch.started_at_shared_turn,
            "quietFullTurns": deathmatch.quiet_full_turns,
            "interval": deathmatch.interval_full_turns,
        })),
        "platform": state.card_runtime.platform.as_ref().and_then(|platform| platform.cell).map(|square| square.to_string()),
    })
}

fn phase_view(state: &GameState) -> Value {
    match &state.phase {
        GamePhase::Playing => json!({ "kind": "playing", "label": "일반 행동" }),
        GamePhase::ForcedMove {
            piece_id, source, ..
        } => json!({
            "kind": "forced_move",
            "label": if matches!(source, augment_core::ForcedMoveSource::Effect(id) if id == augment_core::IRON_MONARCH || id == augment_core::THIEF) { "선택 추가 이동" } else { "강제 이동" },
            "optional": matches!(source, augment_core::ForcedMoveSource::Effect(id) if id == augment_core::IRON_MONARCH || id == augment_core::THIEF),
            "pieceId": piece_id,
            "detail": format!("{source:?}"),
        }),
        GamePhase::Draft(draft) => json!({
            "kind": "draft",
            "label": "드래프트",
            "draftPhase": format!("{:?}", draft.phase).to_ascii_lowercase(),
            "draftPhaseLabel": match draft.phase {
                augment_core::DraftPhase::Opening => "오프닝",
                augment_core::DraftPhase::Middle => "중반",
                augment_core::DraftPhase::End => "후반",
            },
            "offers": draft.offers.iter().map(|id| card_meta(id)).collect::<Vec<_>>(),
        }),
        GamePhase::AwaitingChoice(choice) => json!({
            "kind": "choice",
            "label": "추가 선택",
            "choiceId": choice.id,
            "handler": choice.handler_id,
            "options": choice.options,
        }),
    }
}

fn status_view(status: &GameStatus) -> Value {
    match status {
        GameStatus::Ongoing => json!({ "kind": "ongoing", "label": "진행 중" }),
        GameStatus::Won { winner, reason } => json!({
            "kind": "won",
            "label": format!("{} 승리", player_name(*winner)),
            "winner": player_key(*winner),
            "reason": format!("{reason:?}"),
        }),
        GameStatus::Draw { reason } => json!({
            "kind": "draw",
            "label": "무승부",
            "reason": format!("{reason:?}"),
        }),
    }
}

pub(crate) fn board_view(state: &GameState) -> Vec<Value> {
    let mut cells = BTreeMap::<Square, (Piece, bool, Square)>::new();
    if let Some(square) = state.card_runtime.football_pos {
        cells.insert(
            square,
            (Piece::new(Player::White, PieceKind::Football), true, square),
        );
    }
    for (square, piece) in state.board.iter() {
        cells.insert(square, (piece, true, square));
    }
    for (piece_id, status) in &state.card_runtime.piece_status {
        let Some(anchor) = status.large.map(|body| body.anchor) else {
            continue;
        };
        let Some((storage, piece)) = state.board.find_by_id(*piece_id) else {
            continue;
        };
        for (index, square) in large_piece_cells(anchor).into_iter().enumerate() {
            if state.card_runtime.large_missing_cells.get(piece_id) == Some(&(index as u8)) {
                continue;
            }
            cells.insert(square, (piece, square == storage, storage));
        }
    }
    for (piece_id, position) in &state.card_runtime.quantum_positions {
        if let Some((_, piece)) = state.board.find_by_id(*piece_id) {
            cells.entry(*position).or_insert((piece, true, *position));
        }
    }

    (0..64u8)
        .map(|index| {
            let square = Square::from_index(index).expect("board index");
            let piece = cells
                .get(&square)
                .map(|(piece, anchor, anchor_square)| {
                    piece_view(state, *piece, *anchor, *anchor_square)
                });
            json!({
                "square": square.to_string(),
                "file": square.file(),
                "rank": square.rank(),
                "piece": piece,
                "platform": state.card_runtime.platform.as_ref().and_then(|platform| platform.cell) == Some(square),
                "terrain": {
                    "blackHole":state.card_runtime.black_holes.contains(&square),
                    "highGround":state.card_runtime.high_ground_squares.contains(&square),
                    "highway":state.card_runtime.highways.contains(&square.file())
                        || state.card_runtime.highway_cells.contains(&square),
                    "wall":state.card_runtime.barricades.contains(&square),
                    "fileConveyorUp":state.card_runtime.file_conveyors.iter().any(|(file, owner, _)|
                        *file == square.file() && *owner == Player::White),
                    "fileConveyorDown":state.card_runtime.file_conveyors.iter().any(|(file, owner, _)|
                        *file == square.file() && *owner == Player::Black),
                    "fileConveyorPending":state.card_runtime.file_conveyors.iter().any(|(file, owner, due)| {
                        let next_owner_start = state.turns_taken[owner.index()].saturating_add(1);
                        *file == square.file() && (next_owner_start < *due
                            || (next_owner_start == *due && state.turn != *owner))
                    }),
                    "bomb":state.card_runtime.rule_bombs.contains(&square),
                    "portal":state.card_runtime.rule_portals.is_some_and(|pair|pair.contains(&square)),
                    "crown":state.card_runtime.crown_pos==Some(square),
                    "collapsed":square.file().min(7-square.file()).min(square.rank()).min(7-square.rank())<state.rule_runtime.collapse_depth,
                },
            })
        })
        .collect()
}

fn piece_view(state: &GameState, piece: Piece, anchor: bool, anchor_square: Square) -> Value {
    let mut statuses = Vec::new();
    let october = &state.card_runtime.october;
    if october.grudged_piece_ids.contains(&piece.id) {
        statuses.push("원한".into());
    }
    if let Some(mask) = october.grudge_losses.get(&piece.id) {
        let lost = [(1, "직선"), (2, "대각선"), (4, "나이트"), (8, "낙타")]
            .into_iter()
            .filter(|(bit, _)| mask & bit != 0)
            .map(|(_, label)| label)
            .collect::<Vec<_>>()
            .join("·");
        statuses.push(format!("행마 상실: {lost}"));
    }
    let partners = october
        .pawn_chain_links
        .iter()
        .filter(|pair| pair.contains(&piece.id))
        .filter_map(|pair| {
            state
                .board
                .find_by_id(if pair[0] == piece.id {
                    pair[1]
                } else {
                    pair[0]
                })
                .map(|(square, _)| square.to_string())
        })
        .collect::<Vec<_>>();
    if !partners.is_empty() {
        statuses.push(format!("사슬 · {}", partners.join("·")));
    }
    if state
        .card_runtime
        .october
        .bound_piece_ids
        .contains(&piece.id)
    {
        statuses.push("속박 · 다른 아군 이동 시 해제".into());
    }
    if state
        .card_runtime
        .october
        .pending_seals
        .iter()
        .any(|p| p.piece_id == piece.id)
    {
        statuses.push("봉인 대기".into());
    }
    if state
        .card_runtime
        .october
        .pending_taboo
        .iter()
        .any(|p| p.piece_id == piece.id)
    {
        statuses.push("뒤틀린 소환 · 턴 종료 시 희생".into());
    }
    if state.card_runtime.shielded_piece_ids.contains(&piece.id) {
        statuses.push("보호막".to_owned());
    }
    if state.card_runtime.protected_piece_ids.contains(&piece.id) {
        statuses.push("영구 보호".to_owned());
    }
    if let Some(status) = state.card_runtime.frozen.get(&piece.id) {
        statuses.push(format!("빙결 {}턴", status.remaining_turns));
    }
    if let Some(remaining) = state.card_runtime.poison_stun_turns.get(&piece.id) {
        statuses.push(format!("독 마비 {remaining}턴"));
    }
    if state.card_runtime.horse_riding[piece.owner.index()]
        && (matches!(
            piece.kind,
            PieceKind::King | PieceKind::RoyalKnight | PieceKind::Merchant | PieceKind::DarkWizard
        ) || state.card_runtime.royal_piece_ids.contains(&piece.id))
    {
        statuses.push("승마 · 이번 턴".to_owned());
    }
    if let Some(turn) = state.card_runtime.disarmed.get(&piece.id) {
        statuses.push(format!("무장해제 ~{turn}턴"));
    }
    for (label, status) in [
        ("빙판", state.card_runtime.ice_sheet.get(&piece.id)),
        ("절단", state.card_runtime.severed.get(&piece.id)),
        (
            "희생 보호",
            state.card_runtime.sacrifice_protection.get(&piece.id),
        ),
        ("마녀재판", state.card_runtime.witch_trials.get(&piece.id)),
    ] {
        if let Some(status) = status {
            statuses.push(format!("{label} {}턴", status.remaining_turns));
        }
    }
    for (label, active) in [
        (
            "회피",
            state.card_runtime.evasion_piece_ids.contains(&piece.id),
        ),
        (
            "고스트",
            state.card_runtime.ghost_piece_ids.contains(&piece.id),
        ),
        (
            "관성",
            state.card_runtime.inertia_piece_ids.contains(&piece.id),
        ),
        (
            "은신",
            state.card_runtime.stealth_piece_ids.contains(&piece.id),
        ),
        (
            "잠복",
            state.card_runtime.submerged_piece_ids.contains(&piece.id),
        ),
        (
            "독",
            state.card_runtime.poisoned_piece_ids.contains(&piece.id),
        ),
        (
            "자폭",
            state.card_runtime.explosive_piece_ids.contains(&piece.id),
        ),
        (
            "트로이",
            state.card_runtime.trojan_piece_ids.contains(&piece.id),
        ),
        (
            "키메라",
            state.card_runtime.chimera_piece_ids.contains(&piece.id),
        ),
        (
            "카멜레온",
            state.card_runtime.chameleon_piece_ids.contains(&piece.id),
        ),
    ] {
        if active {
            statuses.push(label.to_owned());
        }
    }
    if let Some(square) = state.card_runtime.quantum_positions.get(&piece.id) {
        statuses.push(format!("양자 위치 {square}"));
    }
    if let Some(health) = state.card_runtime.health(piece.id) {
        statuses.push(format!("HP {}/{}", health.hp, health.max_hp));
    }
    if let Some(mana) = state.card_runtime.wizard_mana.get(&piece.id) {
        statuses.push(format!("마나 {mana}"));
    }
    if let Some(souls) = state.card_runtime.reaper_souls.get(&piece.id) {
        statuses.push(if state.ruleset_id == augment_core::LATEST_RULESET_ID {
            format!("영혼 {souls}/2")
        } else {
            format!("영혼 {souls}")
        });
    }
    let halfmove = state.turns_taken[1] * 2 + u32::from(state.turn == Player::Black);
    if state
        .rule_runtime
        .piece_fresh_until
        .get(&piece.id)
        .is_some_and(|due| *due > halfmove)
    {
        statuses.push("다음 자기 턴까지 포획 불가".into());
    }
    if let Some(gold) = state.card_runtime.merchant_gold.get(&piece.id) {
        statuses.push(format!("골드 {gold}"));
    }
    if let Some(ready) = state.card_runtime.baby_bear_grow_turns.get(&piece.id) {
        statuses.push(format!(
            "성장까지 {}수",
            ready.saturating_sub(state.shared_turn_count())
        ));
    }
    if let Some(remaining) = state.card_runtime.bear_retaliations.get(&piece.id) {
        statuses.push(format!("역습 {remaining}회"));
    }
    if state.card_runtime.loyalist_piece_ids.contains(&piece.id) {
        statuses.push("충신".to_owned());
    }
    if state.card_runtime.royal_piece_ids.contains(&piece.id) {
        statuses.push("왕족".to_owned());
    }
    if let Some(turn) = state.card_runtime.holdout_promotions.get(&piece.id) {
        statuses.push(format!("존버 ~{turn}턴"));
    }
    if state.card_runtime.calling_cards.contains_key(&piece.id) {
        statuses.push("예고장".to_owned());
    }

    if state.rule_runtime.winter_frozen.contains(&piece.id) {
        statuses.push("겨울 동결".into());
    }
    if state.rule_runtime.winter_preview.contains(&piece.id) {
        statuses.push("겨울 동결 예고".into());
    }
    if state.rule_runtime.crown_id == Some(piece.id) {
        statuses.push(format!("왕관 {}/10수", state.rule_runtime.crown_count));
    }
    if let Some(until) = state.rule_runtime.grappler_bound.get(&piece.id) {
        statuses.push(format!("잡기 속박 · {} 턴 종료까지", player_name(*until)));
    }
    if let Some((_, due)) = state.rule_runtime.beacon_due.get(&piece.id) {
        let now = state.turns_taken[1] * 2 + u32::from(state.turn == Player::Black);
        statuses.push(format!("신호기 점화 · {}턴 뒤", due.saturating_sub(now)));
    }
    if piece.kind == PieceKind::Medium
        && let Some(memory) = state.rule_runtime.medium_memory
    {
        statuses.push(format!(
            "기억한 행마 · {}",
            piece_meta(memory.effective, piece.owner).1
        ));
    }
    let (kind, name, code, unicode) = piece_meta(piece.kind, piece.owner);
    json!({
        "id": piece.id,
        "owner": if state.ruleset_id == augment_core::LATEST_RULESET_ID && matches!(piece.kind,PieceKind::Monster|PieceKind::Football) { "neutral" } else { player_key(piece.owner) },
        "ownerLabel": if state.ruleset_id == augment_core::LATEST_RULESET_ID && matches!(piece.kind,PieceKind::Monster|PieceKind::Football) { "중립" } else { player_name(piece.owner) },
        "kind": kind,
        "name": name,
        "description": piece_description(kind),
        "code": code,
        "symbol": unicode.unwrap_or(code),
        "isUnicode": unicode.is_some(),
        "anchor": anchor,
        "anchorSquare": anchor_square.to_string(),
        "captureCount": piece.capture_count,
        "statuses": statuses,
    })
}

fn pending_card_label(state: &GameState, id: &str, owner: Player) -> String {
    let rt = &state.card_runtime.october;
    match id {
        augment_core::PENTA_CHECK => rt.penta_check[owner.index()]
            .map_or_else(String::new, |p| format!("체크 {}/5", p.count)),
        augment_core::PAWN_CHAIN if !rt.pawn_chain_links.is_empty() => {
            format!("사슬 {}개", rt.pawn_chain_links.len())
        }
        augment_core::PROPHECY => rt.prophecy_remaining[owner.index()]
            .map_or_else(String::new, |remaining| format!("양측 {remaining}턴 남음")),
        augment_core::ASCENSION => rt
            .pending_ascensions
            .iter()
            .filter(|p| p.piece.owner == owner)
            .map(|p| {
                format!(
                    "{} · 내 턴 시작 {}회 후 착지",
                    p.square,
                    p.due_turn.saturating_sub(state.turns_taken[owner.index()])
                )
            })
            .collect::<Vec<_>>()
            .join(" / "),
        augment_core::SEAL if rt.pending_seals.iter().any(|p| p.by == owner) => {
            "다음 내 턴에 속박".into()
        }
        augment_core::GREEK_GIFT if rt.greek_gift_pending[owner.index()] => {
            "턴 종료에 폰 제거".into()
        }
        augment_core::TABOO if rt.pending_taboo.iter().any(|p| p.owner == owner) => {
            "턴 종료에 소환".into()
        }
        _ => String::new(),
    }
}

fn cards_view(
    state: &GameState,
    owner: Player,
    action_indices: &BTreeMap<u32, Vec<usize>>,
) -> Vec<Value> {
    state
        .cards
        .iter()
        .filter(|card| card.owner == owner)
        .map(|card| {
            let definition = augment_core::ruleset_card(&state.ruleset_id, &card.definition_id);
            let indices = action_indices
                .get(&card.instance_id)
                .cloned()
                .unwrap_or_default();
            let locked = card.available_after_turn.is_some_and(|turn| {
                state.turns_taken[owner.index()] < turn
            });
            json!({
                "instanceId": card.instance_id,
                "id": card.definition_id,
                "description": card_description(&card.definition_id),
                "name": definition.map_or(card.definition_id.as_str(), |definition| definition.name_ko),
                "stars": format_half_stars(card.half_stars),
                "charges": card.charges,
                "activation": definition.map_or("unknown", |definition| activation_key(definition.activation)),
                "available": !indices.is_empty(),
                "actionIndices": indices,
                "locked": locked,
                "pending": pending_card_label(state, &card.definition_id, owner),
                "openingAutoPending": state.card_runtime.opening_auto_card_ids.contains(&card.instance_id),
                "availableAfterTurn": card.available_after_turn,
            })
        })
        .collect()
}

pub(crate) fn action_view(state: &GameState, index: usize, action: &Action) -> Value {
    match action {
        Action::SkipOptionalMove => json!({
            "index": index,
            "kind": "skip_optional_move",
            "label": "추가 이동 건너뛰기",
            "targetSquares": Vec::<String>::new(),
        }),
        Action::Move(movement) => json!({
            "index": index,
            "kind": "move",
            "label": movement_label(state, movement),
            "from": movement.from.to_string(),
            "to": movement.to.to_string(),
            "targetSquares": [movement.to.to_string()],
            "promotion": movement.promotion.map(kind_key),
            "promotionLabel": movement.promotion.map(kind_name),
        }),
        Action::UseCard {
            instance_id,
            target,
        } => {
            let definition_id = state
                .cards
                .iter()
                .find(|card| card.instance_id == *instance_id)
                .map_or("알 수 없는 카드", |card| card.definition_id.as_str());
            json!({
                "index": index,
                "kind": "card",
                "label": format!("{} · {}", card_name(definition_id), target_label(target)),
                "instanceId": instance_id,
                "cardId": definition_id,
                "selectionSteps": match definition_id {
                    augment_core::ARCHBISHOP => vec!["희생할 아군 나이트", "변환할 아군 비숍"],
                    augment_core::CHANCELLOR => vec!["희생할 아군 나이트", "변환할 아군 룩"],
                    augment_core::GRAPPLER => vec!["변환할 아군 퀸", "희생할 아군 마이너"],
                    augment_core::WINDMILL => vec!["변환할 아군 비숍 또는 룩", "함께 변환할 다른 종류의 기물"],
                    augment_core::ASCENSION => vec!["하늘로 올릴 아군 퀸", "착지할 빈 칸"],
                    augment_core::TABOO => vec!["희생을 예약할 아군 퀸", "괴물을 소환할 빈 칸"],
                    augment_core::FEUDAL_CONTRACT => vec!["첫 번째 계약 기물", "두 번째 계약 기물"],
                    augment_core::TWINS => vec!["첫 번째 아군 기물", "두 번째 아군 기물"],
                    augment_core::PANIC => vec!["먼저 이동할 상대 기물", "다음에 이동할 상대 기물"],
                    augment_core::SACRIFICE => vec!["희생할 아군 기물", "보호할 아군 기물"],
                    augment_core::FILE_CONVEYOR => vec!["컨베이어를 설치할 파일"],
                    _ => Vec::new(),
                },
                "targetKind": target_kind(target),
                "targetSquares": target_squares(target),
            })
        }
        Action::UsePieceAbility {
            piece_id,
            ability,
            target,
        } => json!({
            "index": index,
            "kind": "ability",
            "label": format!("{} · {}", ability_name(*ability), target_label(target)),
            "pieceId": piece_id,
            "sourceSquare": state.board.find_by_id(*piece_id).map(|(square, _)| square.to_string()),
            "ability": ability_key(*ability),
            "abilityLabel": ability_name(*ability),
            "targetKind": target_kind(target),
            "targetSquares": target_squares(target),
        }),
        Action::DraftPick { offer_index } => {
            let card_id = match &state.phase {
                GamePhase::Draft(draft) => draft.offers.get(usize::from(*offer_index)),
                _ => None,
            };
            json!({
                "index": index,
                "kind": "draft",
                "label": card_id.map_or_else(|| format!("드래프트 #{}", offer_index + 1), |id| card_name(id).to_owned()),
                "offerIndex": offer_index,
                "card": card_id.map(|id| card_meta(id)),
                "targetSquares": Vec::<String>::new(),
            })
        }
        Action::ResolveChoice {
            choice_id,
            option_index,
        } => {
            let option = match &state.phase {
                GamePhase::AwaitingChoice(choice) if choice.id == *choice_id => {
                    choice.options.get(usize::from(*option_index))
                }
                _ => None,
            };
            json!({
                "index": index,
                "kind": "choice",
                "label": option.map_or_else(|| format!("선택 #{}", option_index + 1), |option| choice_option_label(state, option)),
                "choiceId": choice_id,
                "optionIndex": option_index,
                "targetSquares": Vec::<String>::new(),
            })
        }
    }
}

fn reference_help() -> &'static Value {
    static HELP: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    HELP.get_or_init(|| {
        serde_json::from_str(include_str!("static/reference-help.json"))
            .expect("bundled reference help")
    })
}

pub(crate) fn piece_description(kind: &str) -> &'static str {
    // Captured October COMPOUND_CARD_HELP; historical help remains source-pinned.
    match kind {
        "archbishop" => return "아치비숍: 나이트와 비숍의 이동을 모두 사용할 수 있습니다.",
        "chancellor" => return "챈슬러: 나이트와 룩의 이동을 모두 사용할 수 있습니다.",
        "campfire" => {
            return "캠프파이어: 룩처럼 이동하며 잡을 수 없습니다. 상하좌우 인접 아군의 비왕실 기물을 보호합니다.";
        }
        "reaper" => {
            return "사신: 킹과 알필처럼 이동합니다. 인접한 아군이 적에게 두 번 잡히면 상대 왕실 기물을 처형하고 승리합니다.";
        }
        "protestant" => {
            return "몬시뇰: 대각선으로 최대 세 칸 이동하며 중간 기물을 뛰어넘을 수 있습니다.";
        }
        "prime_minister" => {
            return "국무총리: 킹처럼 한 칸 또는 두 번 이어서 이동합니다. 중간 칸이 비어 있어야 두 번째 칸으로 갈 수 있습니다.";
        }
        "idol" => {
            return "아이돌: 잡을 수 없는 퀸처럼 이동합니다. 인접 아군의 첫 이동에 턴당 한 번 앙코르를 주며, 먼저 움직인 기물은 이번 턴에 쉽니다.";
        }
        "wizard" => {
            return "마법사: 주변 빈칸으로 한 칸 이동합니다. 아군이 잡힐 때 마나를 얻습니다. 번개·보호막·2×2 운석·시간 정지의 비용은 1·2·3·5이며 마나는 최대 5입니다.";
        }
        "grappler" => {
            return "그래플러: 퀸처럼 이동합니다. 같은 직선의 먼 상대를 인접 칸으로 당기며 두 기물은 상대 턴 종료까지 속박됩니다.";
        }
        "revolving_door" => {
            return "회전문: 룩처럼 이동합니다. 자기 턴 종료에 주변 여덟 칸을 시계 방향으로 회전시킵니다.";
        }
        "don_quixote" => {
            return "돈 키호테: 나이트처럼 이동합니다. 자기 턴 시작에 자동 이동하며, 풍차가 있으면 풍차를 모두 지나는 경로로 돌진합니다.";
        }
        "medium" => {
            return "무덤: 마지막으로 잡힌 기물의 행마를 기억합니다. 기억한 행마가 없으면 움직일 수 없습니다.";
        }
        "berolina_pawn" => {
            return "베롤리나 폰: 대각선으로 전진하고 바로 앞의 적을 잡습니다. 첫 두 칸 이동, 앙파상과 승진을 지원합니다.";
        }
        "killer" => {
            return "킬러: 대각선으로 최대 두 칸 이동합니다. 자기 턴 종료에 공격할 수 있는 상대 왕실 기물을 자동으로 잡습니다.";
        }
        "beacon" => {
            return "신호기: 주변 빈칸으로 한 칸 이동합니다. 자기 턴 종료에 주변 여섯 칸 이상이 차 있으면 비왕실 이웃을 제거하고 세 번째 자기 턴 시작에 승리합니다.";
        }
        "transmission_tower" => {
            return "전송탑: 스스로 움직이지 않습니다. 다른 아군 기물이 이 전송탑 주변의 빈칸으로 이동할 수 있습니다.";
        }
        "barrier" => {
            return "배리어: 스스로 움직이지 않습니다. 살아 있는 동안 아군 왕실 기물의 일반 포획을 막습니다.";
        }
        _ => {}
    }
    reference_help()["pieces"][kind]
        .as_str()
        .unwrap_or("합법 이동은 현재 국면의 코어 엔진 판정에 따릅니다.")
}

pub(crate) fn card_description(id: &str) -> &'static str {
    augment_core::latest_catalog()
        .iter()
        .find(|card| card.id == id)
        .map_or("설명 없음", |card| card.text_ko.as_str())
}

pub(crate) fn card_meta(id: &str) -> Value {
    augment_core::latest_catalog().iter().find(|card| card.id == id).map_or_else(
        || json!({ "id": id, "name": id, "stars": "?", "activation": "unknown" }),
        |definition| {
            json!({
                "id": definition.id,
                "name": definition.name_ko,
                "description": definition.text_ko,
                "stars": format_half_stars(definition.half_stars),
                "activation": if definition.phase == "RULE" { "rule".into() } else { definition.activation.to_ascii_lowercase() },
                "phase": definition.phase.to_ascii_lowercase(),
                "rarity": definition.rarity,
                "enabled": definition.enabled,
                "supported": definition.supported(),
            })
        },
    )
}

fn choice_option_label(state: &GameState, option: &str) -> String {
    option
        .parse::<u32>()
        .ok()
        .and_then(|piece_id| state.board.find_by_id(piece_id))
        .map_or_else(
            || option.to_owned(),
            |(square, piece)| {
                format!(
                    "{square} {} {}",
                    player_name(piece.owner),
                    kind_name(piece.kind)
                )
            },
        )
}

pub(crate) fn event_label(event: &Event) -> Option<String> {
    match event {
        Event::OpeningRuleChecked {
            triggered,
            definition_id,
        } => Some(if *triggered {
            format!(
                "RULE CHECK 적중 · {}",
                definition_id
                    .as_deref()
                    .map_or("실행 가능한 RULE 없음", card_name)
            )
        } else {
            "RULE CHECK 미발동".to_owned()
        }),
        Event::RuleApplied { definition_id } => {
            Some(format!("전역 RULE 적용 · {}", card_name(definition_id)))
        }
        Event::DraftStarted { phase, player, .. } => Some(format!(
            "{} · {} 드래프트 시작",
            player_name(*player),
            format!("{phase:?}").to_ascii_uppercase()
        )),
        Event::DraftPicked {
            player,
            definition_id,
            ..
        } => Some(format!(
            "{} · {} 선택",
            player_name(*player),
            card_name(definition_id)
        )),
        Event::DraftSkipped { phase, player } => Some(format!(
            "{} · {phase:?} 드래프트 후보 없음",
            player_name(*player)
        )),
        Event::CardUsed {
            owner,
            definition_id,
            ..
        } => Some(format!(
            "{} · {} 사용",
            player_name(*owner),
            card_name(definition_id)
        )),
        Event::ActionApplied {
            actor,
            action: Action::Move(movement),
        } => Some(format!(
            "{} · {} → {}{}",
            player_name(*actor),
            movement.from,
            movement.to,
            movement
                .promotion
                .map_or_else(String::new, |kind| format!(" = {}", kind_name(kind)))
        )),
        Event::PieceCaptured { actor, piece, at } => Some(format!(
            "{} · {at}에서 {} {} 포획",
            player_name(*actor),
            player_name(piece.owner),
            kind_name(piece.kind)
        )),
        Event::PieceRemoved {
            piece, at, cause, ..
        } => Some(format!(
            "{at} · {} {} 제거 ({cause})",
            player_name(piece.owner),
            kind_name(piece.kind)
        )),
        Event::ForcedMoveStarted {
            player,
            piece_id,
            source,
        } => Some(format!(
            "{} · 기물 #{piece_id} 강제 이동 ({source})",
            player_name(*player)
        )),
        Event::ChoiceRequested {
            owner, handler_id, ..
        } => Some(format!(
            "{} · 추가 선택 필요 ({handler_id})",
            player_name(*owner)
        )),
        Event::PlatformSpawned { square, .. } => Some(square.map_or_else(
            || "발판을 놓을 빈 칸 없음".to_owned(),
            |square| format!("발판 생성 · {square}"),
        )),
        Event::DeathmatchStarted {
            interval_full_turns,
            ..
        } => Some(format!(
            "오버타임 시작 · {interval_full_turns}회 연속 무진전 시 별 판정"
        )),
        Event::GameEnded { status } => Some(match status {
            GameStatus::Won { winner, reason } => {
                format!("게임 종료 · {} 승리 ({reason:?})", player_name(*winner))
            }
            GameStatus::Draw { reason } => format!("게임 종료 · 무승부 ({reason:?})"),
            GameStatus::Ongoing => return None,
        }),
        _ => None,
    }
}

fn target_kind(target: &ActionTarget) -> &'static str {
    match target {
        ActionTarget::None => "none",
        ActionTarget::Choice(_) => "choice",
        ActionTarget::Square(_) => "square",
        ActionTarget::PieceAt(_) => "piece",
        ActionTarget::Squares(_) => "squares",
    }
}

fn movement_label(state: &GameState, movement: &MoveAction) -> String {
    if state
        .board
        .get(movement.from)
        .is_some_and(|piece| piece.kind == PieceKind::King)
        && let Some(notation) = castling_notation(movement)
    {
        return format!("{notation} · {} → {}", movement.from, movement.to);
    }
    format!(
        "{} → {}{}",
        movement.from,
        movement.to,
        movement
            .promotion
            .map_or_else(String::new, |kind| format!(" = {}", kind_name(kind)))
    )
}

fn castling_notation(movement: &MoveAction) -> Option<&'static str> {
    (movement.from.file() == 4
        && matches!(movement.from.rank(), 0 | 7)
        && movement.from.rank() == movement.to.rank())
    .then(|| match movement.to.file() {
        6 => Some("O-O"),
        2 => Some("O-O-O"),
        _ => None,
    })
    .flatten()
}

fn target_squares(target: &ActionTarget) -> Vec<String> {
    match target {
        ActionTarget::None | ActionTarget::Choice(_) => Vec::new(),
        ActionTarget::Square(square) | ActionTarget::PieceAt(square) => {
            vec![square.to_string()]
        }
        ActionTarget::Squares(squares) => squares.iter().map(ToString::to_string).collect(),
    }
}

fn target_label(target: &ActionTarget) -> String {
    if let ActionTarget::Choice(id) = target {
        return card_name(id).to_owned();
    }
    let squares = target_squares(target);
    if squares.is_empty() {
        "즉시 발동".to_owned()
    } else {
        squares.join(" · ")
    }
}

fn card_name(id: &str) -> &str {
    augment_core::latest_catalog()
        .iter()
        .find(|card| card.id == id)
        .map_or(id, |card| &card.name_ko)
}

fn activation_key(activation: CardActivation) -> &'static str {
    match activation {
        CardActivation::Passive => "passive",
        CardActivation::Active => "active",
        CardActivation::Rule => "rule",
    }
}

fn player_key(player: Player) -> &'static str {
    match player {
        Player::White => "white",
        Player::Black => "black",
    }
}

fn player_name(player: Player) -> &'static str {
    match player {
        Player::White => "백",
        Player::Black => "흑",
    }
}

fn format_half_stars(half_stars: u16) -> String {
    if half_stars % 2 == 0 {
        (half_stars / 2).to_string()
    } else {
        format!("{}.5", half_stars / 2)
    }
}

fn ability_key(ability: PieceAbility) -> &'static str {
    match ability {
        PieceAbility::WizardLightning => "wizard_lightning",
        PieceAbility::WizardShield => "wizard_shield",
        PieceAbility::WizardMeteor => "wizard_meteor",
        PieceAbility::WizardTimeStop => "wizard_time_stop",
        PieceAbility::ColossusSector => "colossus_sector",
        PieceAbility::MerchantBuy => "merchant_buy",
    }
}

fn ability_name(ability: PieceAbility) -> &'static str {
    match ability {
        PieceAbility::WizardLightning => "번개",
        PieceAbility::WizardShield => "보호막",
        PieceAbility::WizardMeteor => "운석",
        PieceAbility::WizardTimeStop => "시간 정지",
        PieceAbility::ColossusSector => "구역 공격",
        PieceAbility::MerchantBuy => "매입",
    }
}

pub(crate) fn piece_meta(
    kind: PieceKind,
    owner: Player,
) -> (
    &'static str,
    &'static str,
    &'static str,
    Option<&'static str>,
) {
    let (key, name, code, white, black) = match kind {
        PieceKind::Pawn => ("pawn", "폰", "P", "♙", "♟"),
        PieceKind::Knight => ("knight", "나이트", "N", "♘", "♞"),
        PieceKind::Bishop => ("bishop", "비숍", "B", "♗", "♝"),
        PieceKind::Rook => ("rook", "룩", "R", "♖", "♜"),
        PieceKind::Queen => ("queen", "퀸", "Q", "♕", "♛"),
        PieceKind::King => ("king", "킹", "K", "♔", "♚"),
        other => {
            let (key, name, code) = special_piece_meta(other);
            return (key, name, code, None);
        }
    };
    (
        key,
        name,
        code,
        Some(if owner == Player::White { white } else { black }),
    )
}

fn special_piece_meta(kind: PieceKind) -> (&'static str, &'static str, &'static str) {
    match kind {
        PieceKind::Grappler => ("grappler", "그래플러", "GP"),
        PieceKind::RevolvingDoor => ("revolving_door", "회전문", "RD"),
        PieceKind::DonQuixote => ("don_quixote", "돈 키호테", "DQ"),
        PieceKind::Medium => ("medium", "무덤", "MD"),
        PieceKind::BerolinaPawn => ("berolina_pawn", "베롤리나 폰", "BP"),
        PieceKind::Killer => ("killer", "킬러", "KI"),
        PieceKind::Beacon => ("beacon", "신호기", "BC"),
        PieceKind::TransmissionTower => ("transmission_tower", "전송탑", "TT"),
        PieceKind::Barrier => ("barrier", "배리어", "BA"),
        // November expansion pieces. The bundle defines no notation code for these,
        // so these are local display codes, like the other compound pieces.
        PieceKind::Offering => ("offering", "제물", "OF"),
        PieceKind::Wanderer => ("wanderer", "방랑자", "WD"),
        PieceKind::OmegaChesstron => ("omega_chesstron", "오메가체스트론", "OX"),
        PieceKind::Assassin => ("assassin", "암살자", "AS"),
        PieceKind::Grasshopper => ("grasshopper", "그래스호퍼", "G"),
        PieceKind::Amazon => ("amazon", "아마존", "Z"),
        PieceKind::PrimeMinister => ("prime_minister", "국무총리", "PM"),
        PieceKind::Dragon => ("dragon", "드래곤", "D"),
        PieceKind::Cannon => ("cannon", "포", "CN"),
        PieceKind::Herald => ("herald", "전령", "H"),
        PieceKind::Knightmaster => ("knightmaster", "기사단장", "KM"),
        PieceKind::Unicorn => ("unicorn", "유니콘", "U"),
        PieceKind::Hook => ("hook", "구행", "HK"),
        PieceKind::Idol => ("idol", "아이돌", "ID"),
        PieceKind::Jester => ("jester", "광대", "J"),
        PieceKind::Recruiter => ("recruiter", "징집관", "RC"),
        PieceKind::Reaper => ("reaper", "사신", "RP"),
        PieceKind::Wizard => ("wizard", "마법사", "WZ"),
        PieceKind::Alibaba => ("alibaba", "알리바바", "AB"),
        PieceKind::Log => ("log", "통나무", "L"),
        PieceKind::Missionary => ("missionary", "선교사", "MI"),
        PieceKind::Cardinal => ("cardinal", "추기경", "CA"),
        PieceKind::Protestant => ("protestant", "몬시뇰", "P"),
        PieceKind::StandardBearer => ("standard_bearer", "기수", "SB"),
        PieceKind::Windmill => ("windmill", "풍차", "W"),
        PieceKind::Squire => ("squire", "종자", "SQ"),
        PieceKind::BigRook => ("big_rook", "빅룩", "BR"),
        PieceKind::Checker => ("checker", "체커", "CH"),
        PieceKind::CheckerKing => ("checker_king", "체커 킹", "CH"),
        PieceKind::Alfil => ("alfil", "알필", "A"),
        PieceKind::Guard => ("guard", "근위병", "GD"),
        PieceKind::Fanatic => ("fanatic", "광신도", "F"),
        PieceKind::Colossus => ("colossus", "거신병", "X"),
        PieceKind::RoyalKnight => ("royal_knight", "로얄 나이트", "RN"),
        PieceKind::Man => ("man", "만", "MN"),
        PieceKind::Merchant => ("merchant", "상인", "M"),
        PieceKind::DarkWizard => ("dark_wizard", "흑마법사", "DW"),
        PieceKind::Monster => ("monster", "괴물", "MO"),
        PieceKind::Scarecrow => ("scarecrow", "허수아비", "S"),
        PieceKind::Camel => ("camel", "낙타", "C"),
        PieceKind::Ferz => ("ferz", "페르즈", "FZ"),
        PieceKind::Bat => ("bat", "박쥐", "BT"),
        PieceKind::BabyBear => ("baby_bear", "아기곰", "BB"),
        PieceKind::Lobster => ("lobster", "랍스터", "LB"),
        PieceKind::Bear => ("bear", "곰", "BE"),
        PieceKind::Vip => ("vip", "귀빈", "VP"),
        PieceKind::Crown => ("crown", "왕관", "CR"),
        PieceKind::Football => ("football", "축구공", "FB"),
        PieceKind::SiegeRam => ("siege_ram", "공성추", "SR"),
        PieceKind::MagicGirl => ("magic_girl", "마법소녀", "MG"),
        PieceKind::Berserker => ("berserker", "버서커", "BK"),
        PieceKind::Slime => ("slime", "슬라임", "SL"),
        PieceKind::Siren => ("siren", "세이렌", "SI"),
        PieceKind::Trickster => ("trickster", "트릭스터", "TR"),
        PieceKind::Undead => ("undead", "언데드", "UD"),
        PieceKind::Clockwork => ("clockwork", "태엽인형", "CW"),
        PieceKind::Parrot => ("parrot", "앵무새", "PR"),
        PieceKind::Paladin => ("paladin", "팔라딘", "PL"),
        PieceKind::Princess => ("princess", "프린세스", "PS"),
        PieceKind::Octopus => ("octopus", "문어", "OC"),
        PieceKind::Hedgehog => ("hedgehog", "고슴도치", "HG"),
        PieceKind::Campfire => ("campfire", "캠프파이어", "CP"),
        PieceKind::Thief => ("thief", "도적", "TH"),
        PieceKind::Brutus => ("brutus", "브루투스", "BT"),
        PieceKind::BigBishop => ("big_bishop", "빅비숍", "BB"),
        PieceKind::Archbishop => ("archbishop", "아치비숍", "AC"),
        PieceKind::Chancellor => ("chancellor", "챈슬러", "CC"),
        PieceKind::Pawn
        | PieceKind::Knight
        | PieceKind::Bishop
        | PieceKind::Rook
        | PieceKind::Queen
        | PieceKind::King => unreachable!(),
    }
}

fn kind_key(kind: PieceKind) -> &'static str {
    piece_meta(kind, Player::White).0
}

fn kind_name(kind: PieceKind) -> &'static str {
    piece_meta(kind, Player::White).1
}

fn large_piece_cells(anchor: Square) -> Vec<Square> {
    [(0, 0), (1, 0), (0, -1), (1, -1)]
        .into_iter()
        .filter_map(|(file_delta, rank_delta)| anchor.offset(file_delta, rank_delta))
        .collect()
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    #[test]
    fn fractured_large_body_does_not_hide_the_piece_on_its_missing_cell() {
        let mut state = augment_core::GameState::standard(
            augment_core::LATEST_RULESET_ID,
            augment_core::Mode::Standard,
            1,
        );
        state.board = augment_core::Board::empty();
        let big = augment_core::Piece::new(
            augment_core::Player::Black,
            augment_core::PieceKind::BigRook,
        )
        .with_id(20);
        let attacker =
            augment_core::Piece::new(augment_core::Player::White, augment_core::PieceKind::Knight)
                .with_id(21);
        state.board.set("h8".parse().unwrap(), Some(big));
        state.board.set("g8".parse().unwrap(), Some(attacker));
        state.card_runtime.insert_large_body(
            big.id,
            "g8".parse().unwrap(),
            augment_core::PieceHealth::damaged(1, 2),
        );
        state.card_runtime.large_missing_cells.insert(big.id, 0);
        let cells = super::board_view(&state);
        let cell = |square| cells.iter().find(|cell| cell["square"] == square).unwrap();
        assert_eq!(cell("g8")["piece"]["kind"], "knight");
        assert_eq!(cell("h8")["piece"]["kind"], "big_rook");
        assert_eq!(cell("g7")["piece"]["kind"], "big_rook");
    }
    use super::super::analyzer::{DEFAULT_SEED, GameSession};
    use super::*;

    #[test]
    fn session_view_contains_board_and_opening_draft_actions() {
        let session = GameSession::new(DEFAULT_SEED).unwrap();
        let view = session.view().unwrap();
        assert_eq!(view["board"].as_array().unwrap().len(), 64);
        assert_eq!(view["phase"]["kind"], "draft");
        assert!(
            view["actions"]
                .as_array()
                .unwrap()
                .iter()
                .all(|action| action["kind"] == "draft")
        );
    }

    #[test]
    fn action_index_advances_the_same_engine_session() {
        let mut session = GameSession::new(DEFAULT_SEED).unwrap();
        let fingerprint = format!("{:016x}", session.state().fingerprint());
        let view = session.apply_index(0, Some(&fingerprint)).unwrap();
        assert_eq!(view["turn"], "black");
        assert!(
            view["events"]
                .as_array()
                .unwrap()
                .iter()
                .any(|event| event.as_str().unwrap().contains("선택"))
        );
    }

    #[test]
    fn stale_action_is_rejected() {
        let mut session = GameSession::new(DEFAULT_SEED).unwrap();
        assert!(session.apply_index(0, Some("0000000000000000")).is_err());
    }

    #[test]
    fn castling_action_uses_chess_notation_in_the_ui() {
        let state = GameState::standard("test", augment_core::Mode::Standard, 0);
        let action = Action::Move(MoveAction::new(
            "e1".parse().unwrap(),
            "g1".parse().unwrap(),
            None,
        ));
        let view = action_view(&state, 0, &action);
        assert_eq!(view["label"], "O-O · e1 → g1");
    }

    #[test]
    fn latest_opening_cards_are_passive_without_legacy_auto_use() {
        let (mut session, offer_index) = (0..1_000)
            .find_map(|seed| {
                let session = GameSession::new(seed).ok()?;
                let GamePhase::Draft(draft) = &session.state().phase else {
                    return None;
                };
                let index = draft.offers.iter().position(|id| {
                    augment_core::ruleset_card(augment_core::LATEST_RULESET_ID, id)
                        .is_some_and(|card| card.phase == augment_core::CardPhase::Opening)
                })?;
                Some((session, index))
            })
            .expect("an eligible passive OPENING offer");
        let view = session.apply_index(offer_index, None).unwrap();
        assert_eq!(view["ruleset"], augment_core::LATEST_RULESET_ID);
        let card = &view["cards"]["white"][0];
        assert_eq!(card["activation"], "passive");
        assert_eq!(card["openingAutoPending"], false);
        assert!(card["actionIndices"].as_array().unwrap().is_empty());
    }

    #[test]
    fn rule_board_view_exposes_neutral_ball_terrain_and_runtime_status() {
        let engine = augment_core::Engine::bundled_latest().unwrap();
        let state = GameState::standard(
            augment_core::LATEST_RULESET_ID,
            augment_core::Mode::Standard,
            31,
        );
        let state = engine
            .apply_rule(&state, augment_core::FOOTBALL)
            .unwrap()
            .state;
        let mut state = engine
            .apply_rule(&state, augment_core::HIGHWAY)
            .unwrap()
            .state;
        let ball = state.card_runtime.football_pos.unwrap();
        let pawn = state.board.get("b2".parse().unwrap()).unwrap();
        state.rule_runtime.winter_preview.push(pawn.id);
        let cells = board_view(&state);
        let ball_cell = cells
            .iter()
            .find(|c| c["square"] == ball.to_string())
            .unwrap();
        assert_eq!(ball_cell["piece"]["kind"], "football");
        assert_eq!(ball_cell["piece"]["owner"], "neutral");
        let pawn_cell = cells.iter().find(|c| c["square"] == "b2").unwrap();
        assert_eq!(pawn_cell["terrain"]["highway"], true);
        assert!(
            pawn_cell["piece"]["statuses"]
                .as_array()
                .unwrap()
                .iter()
                .any(|s| s == "겨울 동결 예고")
        );
    }

    #[test]
    fn chaos_dependency_pieces_have_distinct_metadata_and_help() {
        let mut codes = std::collections::BTreeSet::new();
        for kind in [
            PieceKind::Grappler,
            PieceKind::RevolvingDoor,
            PieceKind::DonQuixote,
            PieceKind::Medium,
            PieceKind::BerolinaPawn,
            PieceKind::Killer,
            PieceKind::Beacon,
            PieceKind::TransmissionTower,
            PieceKind::Barrier,
        ] {
            let (key, _, code, _) = piece_meta(kind, Player::White);
            assert!(codes.insert(code));
            assert!(!piece_description(key).starts_with("합법 이동은"));
        }
    }

    #[test]
    fn grappler_and_windmill_show_their_actual_selection_steps() {
        let engine = augment_core::Engine::bundled_latest().unwrap();
        for id in [augment_core::GRAPPLER, augment_core::WINDMILL] {
            let state = GameState::standard(
                augment_core::LATEST_RULESET_ID,
                augment_core::Mode::Standard,
                31,
            );
            let state = engine
                .acquire_for_setup(&state, id, Player::White)
                .unwrap()
                .state;
            let action = engine
                .legal_actions(&state)
                .unwrap()
                .into_iter()
                .find(|a| matches!(a, Action::UseCard { .. }))
                .unwrap();
            let view = action_view(&state, 0, &action);
            assert_eq!(view["selectionSteps"].as_array().unwrap().len(), 2);
            assert_eq!(view["targetSquares"].as_array().unwrap().len(), 2);
            assert_eq!(view["cardId"], id);
        }
        assert_eq!(piece_meta(PieceKind::Protestant, Player::White).1, "몬시뇰");
        assert!(piece_description("reaper").contains("두 번"));
        assert!(piece_description("campfire").contains("룩처럼"));
    }

    #[test]
    fn middle_card_terrain_is_visible_with_conveyor_activation_timing() {
        let mut state = GameState::standard(
            augment_core::LATEST_RULESET_ID,
            augment_core::Mode::Standard,
            31,
        );
        let square = "c4".parse().unwrap();
        state.card_runtime.barricades.insert(square);
        state.card_runtime.highway_cells.insert(square);
        state
            .card_runtime
            .file_conveyors
            .push((2, Player::White, 2));
        let cell = board_view(&state)
            .into_iter()
            .find(|c| c["square"] == "c4")
            .unwrap();
        assert_eq!(cell["terrain"]["wall"], true);
        assert_eq!(cell["terrain"]["highway"], true);
        assert_eq!(cell["terrain"]["fileConveyorUp"], true);
        assert_eq!(cell["terrain"]["fileConveyorPending"], true);
        state.turns_taken[Player::White.index()] = 1;
        state.turn = Player::Black;
        let cell = board_view(&state)
            .into_iter()
            .find(|c| c["square"] == "c4")
            .unwrap();
        assert_eq!(cell["terrain"]["fileConveyorPending"], true);
        state.turn = Player::White;
        let cell = board_view(&state)
            .into_iter()
            .find(|c| c["square"] == "c4")
            .unwrap();
        assert_eq!(cell["terrain"]["fileConveyorPending"], false);
    }

    #[test]
    fn middle_card_targets_are_selected_in_their_required_order() {
        let engine = augment_core::Engine::bundled_latest().unwrap();
        for (id, steps) in [
            (augment_core::FEUDAL_CONTRACT, 2),
            (augment_core::TWINS, 2),
            (augment_core::PANIC, 2),
            (augment_core::SACRIFICE, 2),
            (augment_core::FILE_CONVEYOR, 1),
        ] {
            let mut state = GameState::standard(
                augment_core::LATEST_RULESET_ID,
                augment_core::Mode::Standard,
                31,
            );
            state.completed_drafts = [true; 3];
            let state = engine
                .grant_card(&state, 1, id, Player::White)
                .unwrap()
                .state;
            let action = engine
                .legal_actions(&state)
                .unwrap()
                .into_iter()
                .find(|action| matches!(action, Action::UseCard { instance_id: 1, .. }))
                .unwrap();
            let view = action_view(&state, 0, &action);
            assert_eq!(
                view["selectionSteps"].as_array().unwrap().len(),
                steps,
                "{id}"
            );
            assert_eq!(
                view["targetSquares"].as_array().unwrap().len(),
                steps,
                "{id}"
            );
        }
    }

    #[test]
    fn named_card_choice_uses_the_catalog_name_instead_of_a_square() {
        let target = ActionTarget::Choice(augment_core::HIGHWAY.to_owned());
        assert_eq!(target_kind(&target), "choice");
        assert_eq!(target_label(&target), card_name(augment_core::HIGHWAY));
        assert!(target_squares(&target).is_empty());
    }
}
