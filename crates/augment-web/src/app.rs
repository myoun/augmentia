use augment_ai::{
    Evaluator, HeuristicEvaluator, MctsConfig, MctsTree, MctsV3Config, MctsV3Tree,
    OnnxNeuralEvaluator, V3OnnxRuntime,
};
use augment_core::{
    Action, ActionTarget, BUNDLED_RULESET_ID, CardActivation, Engine, Event, GamePhase, GameState,
    GameStatus, Mode, MoveAction, Piece, PieceAbility, PieceKind, Player, Square, implemented_card,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const DEFAULT_SEED: u64 = 20_260_807;
const V2_BASELINE_MODEL_PATH: &str = "models/baseline/augment_net_v2_canonical_baseline.onnx";
const V3_SMOKE_MODEL_PATH: &str = "artifacts/models/v3-smoke-step500.onnx";
const V3_SMOKE_METADATA_PATH: &str = "artifacts/models/v3-smoke-step500.metadata.json";
const RULESET_MANIFEST_PATH: &str = "rulesets/augment_chess_20260807/manifest.json";

#[derive(Clone, Debug)]
pub struct WebModelConfig {
    pub model_1: PathBuf,
    pub model_2: PathBuf,
    pub model_2_metadata: PathBuf,
    pub ruleset_manifest: PathBuf,
}

impl Default for WebModelConfig {
    fn default() -> Self {
        Self {
            model_1: V2_BASELINE_MODEL_PATH.into(),
            model_2: V3_SMOKE_MODEL_PATH.into(),
            model_2_metadata: V3_SMOKE_METADATA_PATH.into(),
            ruleset_manifest: RULESET_MANIFEST_PATH.into(),
        }
    }
}

pub struct GameSession {
    engine: Engine,
    state: GameState,
    seed: u64,
    history: Vec<String>,
    model_1: Result<OnnxNeuralEvaluator, String>,
    model_2: Result<V3OnnxRuntime, String>,
}

impl GameSession {
    #[cfg(test)]
    pub fn new(seed: u64) -> Result<Self, String> {
        let missing = PathBuf::from("test-models-are-not-loaded");
        Self::new_with_models(
            seed,
            &WebModelConfig {
                model_1: missing.join("model-1.onnx"),
                model_2: missing.join("model-2.onnx"),
                model_2_metadata: missing.join("model-2.metadata.json"),
                ruleset_manifest: missing.join("manifest.json"),
            },
        )
    }

    pub fn new_with_models(seed: u64, models: &WebModelConfig) -> Result<Self, String> {
        let engine = Engine::bundled_20260807().map_err(|error| error.to_string())?;
        let transition = engine
            .new_game_transition(Mode::Standard, seed)
            .map_err(|error| error.to_string())?;
        let model_1 = OnnxNeuralEvaluator::from_file(&models.model_1)
            .map_err(|error| format!("1번 모델을 불러오지 못했습니다: {error}"));
        let model_2 = load_v3_runtime(models)
            .map_err(|error| format!("2번 모델을 불러오지 못했습니다: {error}"));
        let mut session = Self {
            engine,
            state: transition.state,
            seed,
            model_1,
            model_2,
            history: Vec::new(),
        };
        session.record_events(&transition.events);
        Ok(session)
    }

    pub fn reset(&mut self, seed: u64) -> Result<(), String> {
        let transition = self
            .engine
            .new_game_transition(Mode::Standard, seed)
            .map_err(|error| error.to_string())?;
        self.state = transition.state;
        self.seed = seed;
        self.history.clear();
        self.record_events(&transition.events);
        Ok(())
    }

    pub fn view(&self) -> Result<Value, String> {
        let legal_actions = if self.state.status == GameStatus::Ongoing {
            self.engine
                .legal_actions(&self.state)
                .map_err(|error| error.to_string())?
        } else {
            Vec::new()
        };
        Ok(state_view(
            &self.state,
            &self.engine,
            self.model_1.as_ref().ok(),
            self.seed,
            &self.history,
            &legal_actions,
        ))
    }

    pub fn suggestions(
        &self,
        simulations: usize,
        fingerprint: Option<&str>,
    ) -> Result<Value, String> {
        let current = format!("{:016x}", self.state.fingerprint());
        if fingerprint.is_some_and(|fingerprint| fingerprint != current) {
            return Err("화면이 오래되었습니다. 최신 상태를 불러온 뒤 다시 시도하세요.".to_owned());
        }
        if self.state.status != GameStatus::Ongoing {
            return Ok(json!({ "fingerprint": current, "suggestions": [] }));
        }
        let legal_actions = self
            .engine
            .legal_actions(&self.state)
            .map_err(|error| error.to_string())?;
        let simulations = simulations.clamp(10, 1_000);
        let (model_1, model_2) = std::thread::scope(|scope| {
            let first = scope.spawn(|| {
                model_1_suggestion(
                    &self.model_1,
                    &self.state,
                    &self.engine,
                    &legal_actions,
                    simulations,
                )
            });
            let second = scope.spawn(|| {
                model_2_suggestion(
                    &self.model_2,
                    &self.state,
                    &self.engine,
                    &legal_actions,
                    simulations,
                )
            });
            (
                first.join().unwrap_or_else(|_| {
                    suggestion_error(
                        "model-1",
                        "1번 모델 · V2 기준",
                        "탐색 스레드가 중단됐습니다.",
                    )
                }),
                second.join().unwrap_or_else(|_| {
                    suggestion_error(
                        "model-2",
                        "2번 모델 · V3 smoke",
                        "탐색 스레드가 중단됐습니다.",
                    )
                }),
            )
        });
        Ok(json!({
            "fingerprint": current,
            "simulations": simulations,
            "suggestions": [model_1, model_2],
        }))
    }

    pub fn apply_index(
        &mut self,
        index: usize,
        fingerprint: Option<&str>,
    ) -> Result<Value, String> {
        let current = format!("{:016x}", self.state.fingerprint());
        if fingerprint.is_some_and(|fingerprint| fingerprint != current) {
            return Err("화면이 오래되었습니다. 최신 상태를 불러온 뒤 다시 시도하세요.".to_owned());
        }
        let legal_actions = self
            .engine
            .legal_actions(&self.state)
            .map_err(|error| error.to_string())?;
        let action = legal_actions
            .get(index)
            .cloned()
            .ok_or_else(|| "현재 존재하지 않는 행동입니다.".to_owned())?;
        let transition = self
            .engine
            .apply(&self.state, action)
            .map_err(|error| error.to_string())?;
        self.record_events(&transition.events);
        self.state = transition.state;
        self.view()
    }

    pub fn bot_move(&mut self, simulations: usize) -> Result<Value, String> {
        if self.state.status != GameStatus::Ongoing {
            return self.view();
        }
        let config = MctsConfig {
            c_puct: 2.0,
            num_simulations: simulations.clamp(10, 1000),
            temperature: 0.5,
        };
        let mut tree = MctsTree::new(config);

        let neural_eval = self.model_1.as_ref().map_err(Clone::clone)?;
        let failures_before = neural_eval.inference_failures();
        let search_result = tree.search(&self.state, &self.engine, neural_eval);
        if neural_eval.inference_failures() != failures_before {
            return Err("1번 모델 ONNX 추론이 실패해 자동 착수를 중단했습니다.".to_owned());
        }
        if let Some((chosen_action, _, _)) = search_result {
            let transition = self
                .engine
                .apply(&self.state, chosen_action)
                .map_err(|error| error.to_string())?;
            self.record_events(&transition.events);
            self.state = transition.state;
        }
        self.view()
    }

    fn record_events(&mut self, events: &[Event]) {
        self.history.extend(events.iter().filter_map(event_label));
        const LIMIT: usize = 120;
        if self.history.len() > LIMIT {
            self.history.drain(0..self.history.len() - LIMIT);
        }
    }
}

fn state_view(
    state: &GameState,
    engine: &Engine,
    neural_eval: Option<&OnnxNeuralEvaluator>,
    seed: u64,
    history: &[String],
    actions: &[Action],
) -> Value {
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
        "events": history,
        "eval": compute_eval(state, engine, neural_eval),
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
            "label": "강제 이동",
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

fn board_view(state: &GameState) -> Vec<Value> {
    let mut cells = BTreeMap::<Square, (Piece, bool, Square)>::new();
    for (square, piece) in state.board.iter() {
        cells.insert(square, (piece, true, square));
    }
    for (piece_id, status) in &state.card_runtime.large_pieces {
        let Some((_, piece)) = state.board.find_by_id(*piece_id) else {
            continue;
        };
        for square in large_piece_cells(status.anchor) {
            cells.insert(square, (piece, square == status.anchor, status.anchor));
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
            })
        })
        .collect()
}

fn piece_view(state: &GameState, piece: Piece, anchor: bool, anchor_square: Square) -> Value {
    let mut statuses = Vec::new();
    if state.card_runtime.shielded_piece_ids.contains(&piece.id) {
        statuses.push("보호막".to_owned());
    }
    if state.card_runtime.protected_piece_ids.contains(&piece.id) {
        statuses.push("영구 보호".to_owned());
    }
    if let Some(status) = state.card_runtime.frozen.get(&piece.id) {
        statuses.push(format!("빙결 {}턴", status.remaining_turns));
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
    if let Some(status) = state.card_runtime.large_pieces.get(&piece.id) {
        statuses.push(format!("HP {}/{}", status.hp, status.max_hp));
    }
    if let Some(mana) = state.card_runtime.wizard_mana.get(&piece.id) {
        statuses.push(format!("마나 {mana}"));
    }
    if let Some(souls) = state.card_runtime.reaper_souls.get(&piece.id) {
        statuses.push(format!("영혼 {souls}"));
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

    let (kind, name, code, unicode) = piece_meta(piece.kind, piece.owner);
    json!({
        "id": piece.id,
        "owner": player_key(piece.owner),
        "ownerLabel": player_name(piece.owner),
        "kind": kind,
        "name": name,
        "code": code,
        "symbol": unicode.unwrap_or(code),
        "isUnicode": unicode.is_some(),
        "anchor": anchor,
        "anchorSquare": anchor_square.to_string(),
        "captureCount": piece.capture_count,
        "statuses": statuses,
    })
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
            let definition = implemented_card(&card.definition_id);
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
                "name": definition.map_or(card.definition_id.as_str(), |definition| definition.name_ko),
                "stars": format_half_stars(card.half_stars),
                "charges": card.charges,
                "activation": definition.map_or("unknown", |definition| activation_key(definition.activation)),
                "available": !indices.is_empty(),
                "actionIndices": indices,
                "locked": locked,
                "openingAutoPending": state.card_runtime.opening_auto_card_ids.contains(&card.instance_id),
                "availableAfterTurn": card.available_after_turn,
            })
        })
        .collect()
}

fn action_view(state: &GameState, index: usize, action: &Action) -> Value {
    match action {
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

fn card_meta(id: &str) -> Value {
    implemented_card(id).map_or_else(
        || json!({ "id": id, "name": id, "stars": "?", "activation": "unknown" }),
        |definition| {
            json!({
                "id": definition.id,
                "name": definition.name_ko,
                "stars": format_half_stars(definition.half_stars),
                "activation": activation_key(definition.activation),
                "phase": format!("{:?}", definition.phase).to_ascii_lowercase(),
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

fn event_label(event: &Event) -> Option<String> {
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
        ActionTarget::None => Vec::new(),
        ActionTarget::Square(square) | ActionTarget::PieceAt(square) => {
            vec![square.to_string()]
        }
        ActionTarget::Squares(squares) => squares.iter().map(ToString::to_string).collect(),
    }
}

fn target_label(target: &ActionTarget) -> String {
    let squares = target_squares(target);
    if squares.is_empty() {
        "즉시 발동".to_owned()
    } else {
        squares.join(" · ")
    }
}

fn card_name(id: &str) -> &str {
    implemented_card(id).map_or(id, |definition| definition.name_ko)
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

fn piece_meta(
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
        PieceKind::Protestant => ("protestant", "프로테스탄트", "P"),
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
mod tests {
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
        let fingerprint = format!("{:016x}", session.state.fingerprint());
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
        let state = GameState::standard("test", Mode::Standard, 0);
        let action = Action::Move(MoveAction::new(
            "e1".parse().unwrap(),
            "g1".parse().unwrap(),
            None,
        ));
        let view = action_view(&state, 0, &action);
        assert_eq!(view["label"], "O-O · e1 → g1");
    }

    #[test]
    fn opening_auto_card_is_shown_as_pending_and_not_clickable() {
        let (mut session, offer_index) = (0..1_000)
            .find_map(|seed| {
                let session = GameSession::new(seed).ok()?;
                let GamePhase::Draft(draft) = &session.state.phase else {
                    return None;
                };
                let index = draft.offers.iter().position(|id| {
                    implemented_card(id).is_some_and(|card| {
                        card.phase == augment_core::CardPhase::Opening
                            && card.activation == CardActivation::Active
                    })
                })?;
                Some((session, index))
            })
            .expect("a bounded seed search must find an active OPENING offer");
        session.apply_index(offer_index, None).unwrap();
        let view = session.apply_index(0, None).unwrap();
        let card = view["cards"]["white"]
            .as_array()
            .unwrap()
            .iter()
            .find(|card| card["openingAutoPending"] == true)
            .unwrap();
        assert_eq!(card["openingAutoPending"], true);
        assert_eq!(card["available"], false);
        assert!(card["actionIndices"].as_array().unwrap().is_empty());
    }

    #[test]
    fn unavailable_models_are_reported_independently() {
        let missing = PathBuf::from("does-not-exist");
        let models = WebModelConfig {
            model_1: missing.join("model-1.onnx"),
            model_2: missing.join("model-2.onnx"),
            model_2_metadata: missing.join("model-2.metadata.json"),
            ruleset_manifest: missing.join("manifest.json"),
        };
        let session = GameSession::new_with_models(DEFAULT_SEED, &models).unwrap();
        let result = session.suggestions(10, None).unwrap();
        let suggestions = result["suggestions"].as_array().unwrap();
        assert_eq!(suggestions.len(), 2);
        assert_eq!(suggestions[0]["modelId"], "model-1");
        assert_eq!(suggestions[1]["modelId"], "model-2");
        assert_eq!(suggestions[0]["status"], "error");
        assert_eq!(suggestions[1]["status"], "error");
    }
}

fn load_v3_runtime(models: &WebModelConfig) -> Result<V3OnnxRuntime, String> {
    let ruleset_hash = sha256_file(&models.ruleset_manifest)?;
    V3OnnxRuntime::from_files(
        &models.model_2,
        &models.model_2_metadata,
        BUNDLED_RULESET_ID,
        &ruleset_hash,
    )
    .map_err(|error| error.to_string())
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn model_1_suggestion(
    evaluator: &Result<OnnxNeuralEvaluator, String>,
    state: &GameState,
    engine: &Engine,
    legal_actions: &[Action],
    simulations: usize,
) -> Value {
    let evaluator = match evaluator {
        Ok(evaluator) => evaluator,
        Err(error) => return suggestion_error("model-1", "1번 모델 · V2 기준", error),
    };
    let started = Instant::now();
    let failures_before = evaluator.inference_failures();
    let mut tree = MctsTree::new(MctsConfig {
        c_puct: 2.0,
        num_simulations: simulations,
        temperature: 0.0,
    });
    let Some((action, _, visits)) = tree.search(state, engine, evaluator) else {
        return suggestion_error("model-1", "1번 모델 · V2 기준", "탐색 결과가 없습니다.");
    };
    if evaluator.inference_failures() != failures_before {
        return suggestion_error(
            "model-1",
            "1번 모델 · V2 기준",
            "ONNX 추론이 실패해 휴리스틱 fallback이 발생했습니다.",
        );
    }
    let Some(action_index) = legal_actions
        .iter()
        .position(|candidate| candidate == &action)
    else {
        return suggestion_error(
            "model-1",
            "1번 모델 · V2 기준",
            "제안 행동이 현재 합법 행동 목록에 없습니다.",
        );
    };
    let visit_count = visits
        .iter()
        .find(|(candidate, _, _)| candidate == &action)
        .map_or(0, |(_, _, count)| *count);
    let root_value = tree
        .last_diagnostics
        .as_ref()
        .map_or(0.0, |diagnostics| diagnostics.root_raw_value);
    suggestion_ready(
        "model-1",
        "1번 모델 · V2 기준",
        state,
        action_index,
        &action,
        visit_count,
        root_value,
        simulations,
        started.elapsed().as_secs_f64() * 1_000.0,
        None,
    )
}

fn model_2_suggestion(
    evaluator: &Result<V3OnnxRuntime, String>,
    state: &GameState,
    engine: &Engine,
    legal_actions: &[Action],
    simulations: usize,
) -> Value {
    let evaluator = match evaluator {
        Ok(evaluator) => evaluator,
        Err(error) => return suggestion_error("model-2", "2번 모델 · V3 smoke", error),
    };
    let started = Instant::now();
    let mut tree = match MctsV3Tree::new(MctsV3Config {
        num_simulations: simulations,
        leaf_batch_size: 16,
        ..MctsV3Config::default()
    }) {
        Ok(tree) => tree,
        Err(error) => {
            return suggestion_error("model-2", "2번 모델 · V3 smoke", &error.to_string());
        }
    };
    let result = match tree.search(state, engine, evaluator) {
        Ok(result) => result,
        Err(error) => {
            return suggestion_error("model-2", "2번 모델 · V3 smoke", &error.to_string());
        }
    };
    let Some(action_index) = legal_actions
        .iter()
        .position(|candidate| candidate == &result.action)
    else {
        return suggestion_error(
            "model-2",
            "2번 모델 · V3 smoke",
            "제안 행동이 현재 합법 행동 목록에 없습니다.",
        );
    };
    let visit_count = result
        .visits
        .iter()
        .find(|(candidate, _, _)| candidate == &result.action)
        .map_or(0, |(_, _, count)| *count);
    let root_value = tree
        .last_diagnostics
        .as_ref()
        .map_or(0.0, |diagnostics| diagnostics.root_raw_value);
    suggestion_ready(
        "model-2",
        "2번 모델 · V3 smoke",
        state,
        action_index,
        &result.action,
        visit_count,
        root_value,
        simulations,
        started.elapsed().as_secs_f64() * 1_000.0,
        Some("일반 체스 policy-only smoke 모델 · value/카드/드래프트 미학습"),
    )
}

#[allow(clippy::too_many_arguments)]
fn suggestion_ready(
    model_id: &str,
    name: &str,
    state: &GameState,
    action_index: usize,
    action: &Action,
    visits: u32,
    root_value: f32,
    simulations: usize,
    elapsed_ms: f64,
    warning: Option<&str>,
) -> Value {
    json!({
        "modelId": model_id,
        "name": name,
        "status": "ready",
        "actionIndex": action_index,
        "action": action_view(state, action_index, action),
        "visits": visits,
        "rootValue": root_value,
        "simulations": simulations,
        "elapsedMs": elapsed_ms,
        "warning": warning,
    })
}

fn suggestion_error(model_id: &str, name: &str, error: &str) -> Value {
    json!({
        "modelId": model_id,
        "name": name,
        "status": "error",
        "error": error,
    })
}

fn compute_eval(
    state: &GameState,
    engine: &Engine,
    neural_eval: Option<&OnnxNeuralEvaluator>,
) -> Value {
    let raw_val = if let Some(neural) = neural_eval {
        neural.evaluate(state, engine).1
    } else {
        HeuristicEvaluator::new().evaluate(state, engine).1
    };

    let white_value = if state.turn == Player::White {
        raw_val
    } else {
        -raw_val
    };

    let score = (white_value * 10.0).clamp(-99.0, 99.0);
    let white_pct = ((white_value + 1.0) / 2.0 * 100.0).clamp(0.0, 100.0);
    let black_pct = 100.0 - white_pct;

    json!({
        "whiteValue": white_value,
        "score": if score > 0.0 { format!("+{:.1}", score) } else { format!("{:.1}", score) },
        "whitePercent": format!("{:.0}%", white_pct),
        "blackPercent": format!("{:.0}%", black_pct),
        "whiteRatio": white_pct,
    })
}
