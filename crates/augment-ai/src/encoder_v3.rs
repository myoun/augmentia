use crate::observation_v3::{CARD_RUNTIME_SOURCE_FIELDS, OBSERVATION_V3_SCHEMA_ID};
use augment_core::{
    BribeRevert, CardRuntimeState, ChoiceContinuation, DelayedHazard, DelayedHazardKind,
    DiceLockStatus, DraftPhase, EmptyLunchboxStatus, EnPassantState, FeudalContract,
    ForcedMoveContinuation, ForcedMoveSource, FrozenStatus, GamePhase, GameState, GameStatus,
    IMPLEMENTED_CARDS, InitiativeStatus, LargePieceStatus, LogStatus, MIDDLE_EXTRA_MOVE_SOURCE,
    Mode, MoveAction, NecromancyRevert, PendingIcbm, PendingOtherworld, PendingPanic,
    PendingPremove, PendingScarecrow, PendingVip, Piece, PieceKind, PlatformRuleState, Player,
    RepositionStatus, RoyalCommandStatus, Square, TimedPieceStatus, TrolleyDilemma, TwinBond,
    WindmillMode,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

// Preserve legacy planes0..117; kinds60..68 append self/opponent pairs at118..135.
pub const OBSERVATION_V3_SPATIAL_PLANES: usize = 136;
pub const OBSERVATION_V3_SPATIAL_SIZE: usize = OBSERVATION_V3_SPATIAL_PLANES * 64;
pub const OBSERVATION_V3_GLOBAL_FEATURES: usize = 32;

/// Exact string tokens that the bundled V3 encoder can emit independently of
/// card catalog data. Keep this list in Rust, next to the encoder that owns the
/// semantics, so ONNX export does not infer runtime coverage from a few sample
/// positions.
pub const OBSERVATION_V3_REQUIRED_TEXT_TOKENS: [&str; 80] = [
    "age_shared_turns",
    "anchor",
    "bishop",
    "bundles",
    "by",
    "card_effect",
    "campaign",
    "cell",
    "chaos",
    "choice:field-promotion:0",
    "choice:field-promotion:1",
    "choice:field-promotion:2",
    "choice:field-promotion:3",
    "choice:middle.trolley:0",
    "choice:middle.trolley:1",
    "clock_owner",
    "count",
    "draw",
    "effect",
    "effective",
    "elapsed_turns",
    "end",
    "equal_stars_after_deathmatch",
    "equal_stars_on_repetition",
    "expires_in_turns",
    "file_delta",
    "finish_turn",
    "first",
    "from",
    "grand",
    "guardian",
    "hp",
    "kind",
    "king_captured",
    "lightning",
    "limit",
    "lower_stars_after_deathmatch",
    "lower_stars_on_repetition",
    "max_hp",
    "meteor",
    "middle",
    MIDDLE_EXTRA_MOVE_SOURCE,
    "middle.trolley",
    "moves",
    "next_in_shared_turns",
    "no_legal_actions",
    "nonce",
    "occupation",
    "ongoing",
    "opening",
    "origin",
    "owner",
    "pawn",
    "piece",
    "piece_kind",
    "pieces",
    "platform",
    "player",
    "promotion",
    "queen",
    "rank_delta",
    "remaining_ply",
    "remaining_turns",
    "resume_turn",
    "roll_in_owner_turns",
    "rook",
    "second",
    "source",
    "square",
    "squares",
    "standard",
    "starts_in_turns",
    "to",
    "trigger_after",
    "trigger_in_turns",
    "trigger_player",
    "triggered_pieces",
    "used",
    "value",
    "won",
];

/// Complete finite token inventory for the bundled ruleset. The fixed encoder
/// vocabulary is augmented with every implemented card identifier because cards
/// also appear as draft offers, definitions, effect sources and cloned passives.
pub fn observation_v3_runtime_text_tokens() -> Vec<&'static str> {
    let mut tokens = OBSERVATION_V3_REQUIRED_TEXT_TOKENS.to_vec();
    tokens.extend(IMPLEMENTED_CARDS.iter().map(|definition| definition.id));
    tokens.sort_unstable();
    tokens.dedup();
    tokens
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelativePlayerV3 {
    SelfPlayer,
    Opponent,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationV3 {
    pub schema_id: String,
    pub ruleset_id: String,
    pub spatial_shape: [usize; 3],
    pub spatial: Vec<f32>,
    pub global_shape: [usize; 1],
    pub global: Vec<f32>,
    pub state: GlobalStateV3,
    pub pieces: Vec<BoardPieceV3>,
    pub cards: Vec<CardRecordV3>,
    pub phase: PhaseV3,
    pub runtime: Vec<RuntimeFeatureV3>,
    pub piece_slot_count: u32,
    pub card_slot_count: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GlobalStateV3 {
    pub mode_id: String,
    pub ply: u32,
    pub quiet_ply: u32,
    pub actions_remaining: u16,
    pub cards_used_this_turn: [u16; 2],
    pub half_star_totals: [u16; 2],
    pub turns_taken: [u32; 2],
    pub completed_drafts: [bool; 3],
    pub shared_turn_count: u32,
    pub current_repetition_count: u8,
    pub status: StatusV3,
    pub deathmatch: Option<DeathmatchV3>,
    pub en_passant: Option<EnPassantV3>,
    pub applied_rules: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StatusV3 {
    Ongoing,
    Won {
        winner: RelativePlayerV3,
        reason: String,
        detail: Option<String>,
    },
    Draw {
        reason: String,
        detail: Option<String>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DeathmatchV3 {
    pub age_shared_turns: u32,
    pub quiet_full_turns: u32,
    pub interval_full_turns: u32,
    pub progress_this_round: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EnPassantV3 {
    pub target: u8,
    pub captured: u8,
    pub pawn_owner: RelativePlayerV3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BoardPieceV3 {
    pub slot: u32,
    pub square: u8,
    pub owner: RelativePlayerV3,
    /// Canonical quantum shadow square, if the piece has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_square: Option<u8>,
    pub kind_code: u8,
    pub capture_count: u16,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CardRecordV3 {
    pub slot: u32,
    pub definition_id: String,
    pub owner: RelativePlayerV3,
    pub half_stars: u16,
    pub charges: u16,
    pub available_in_owner_turns: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PhaseV3 {
    Playing,
    ForcedMove {
        piece: u32,
        source_kind: String,
        source_id: Option<String>,
        continuation: String,
    },
    Draft {
        draft_phase: String,
        offers: Vec<String>,
        resume_turn: RelativePlayerV3,
    },
    AwaitingChoice {
        choice_slot: u32,
        owner: RelativePlayerV3,
        source_card: u32,
        handler_id: String,
        options: Vec<String>,
        continuation: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RuntimeFeatureV3 {
    pub field_id: u16,
    pub value: RuntimeValueV3,
}

impl RuntimeFeatureV3 {
    pub fn field_name(&self) -> Option<&'static str> {
        self.field_id
            .checked_sub(1)
            .and_then(|index| CARD_RUNTIME_SOURCE_FIELDS.get(usize::from(index)))
            .copied()
    }
}

/// Collect every free-text token that this observation can feed to the model
/// vocabulary. Used by contract tests and the runtime vocabulary exporter.
pub fn observation_text_tokens(observation: &ObservationV3) -> BTreeSet<String> {
    let mut tokens = BTreeSet::new();
    tokens.insert(observation.state.mode_id.clone());
    tokens.extend(observation.state.applied_rules.iter().cloned());
    match &observation.state.status {
        StatusV3::Ongoing => {}
        StatusV3::Won { reason, detail, .. } | StatusV3::Draw { reason, detail } => {
            tokens.insert(reason.clone());
            if let Some(detail) = detail {
                tokens.insert(detail.clone());
            }
        }
    }
    for card in &observation.cards {
        tokens.insert(card.definition_id.clone());
    }
    match &observation.phase {
        PhaseV3::Playing => {}
        PhaseV3::ForcedMove {
            source_kind,
            source_id,
            ..
        } => {
            tokens.insert(source_kind.clone());
            if let Some(source_id) = source_id {
                tokens.insert(source_id.clone());
            }
        }
        PhaseV3::Draft {
            draft_phase,
            offers,
            ..
        } => {
            tokens.insert(draft_phase.clone());
            tokens.extend(offers.iter().cloned());
        }
        PhaseV3::AwaitingChoice {
            handler_id,
            options,
            continuation,
            ..
        } => {
            tokens.insert(handler_id.clone());
            tokens.extend(options.iter().cloned());
            tokens.insert(continuation.clone());
        }
    }
    for feature in &observation.runtime {
        collect_runtime_text(&feature.value, &mut tokens);
    }
    tokens
}

fn collect_runtime_text(value: &RuntimeValueV3, tokens: &mut BTreeSet<String>) {
    match value {
        RuntimeValueV3::Text(text) => {
            tokens.insert(text.clone());
        }
        RuntimeValueV3::List(values) => {
            for value in values {
                collect_runtime_text(value, tokens);
            }
        }
        RuntimeValueV3::Record(fields) => {
            for field in fields {
                collect_runtime_text(&field.value, tokens);
            }
        }
        _ => {}
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NamedRuntimeValueV3 {
    pub name: String,
    pub value: RuntimeValueV3,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum RuntimeValueV3 {
    None,
    Bool(bool),
    Unsigned(u64),
    Signed(i64),
    Player(RelativePlayerV3),
    Piece(u32),
    Card(u32),
    PieceKind(u8),
    Square(u8),
    Text(String),
    List(Vec<RuntimeValueV3>),
    Record(Vec<NamedRuntimeValueV3>),
}

#[derive(Default)]
pub(crate) struct LocalIds {
    pieces: BTreeMap<u32, u32>,
    cards: BTreeMap<u32, u32>,
}

impl LocalIds {
    pub(crate) fn new(state: &GameState) -> Self {
        let mut ids = Self::default();
        let mut board_pieces = state
            .board
            .iter()
            .map(|(square, piece)| (canonical_square(square, state.turn).index(), piece.id))
            .collect::<Vec<_>>();
        board_pieces.sort_unstable();
        for (_, piece_id) in board_pieces {
            ids.piece(piece_id);
        }

        let mut cards = state.cards.iter().collect::<Vec<_>>();
        cards.sort_by_key(|card| {
            (
                relative_player(card.owner, state.turn) as u8,
                card.definition_id.as_str(),
                card.half_stars,
                card.charges,
                card.available_after_turn,
                card.instance_id,
            )
        });
        for card in cards {
            ids.card(card.instance_id);
        }
        ids
    }

    fn piece(&mut self, raw: u32) -> u32 {
        let next = self.pieces.len() as u32;
        *self.pieces.entry(raw).or_insert(next)
    }

    fn card(&mut self, raw: u32) -> u32 {
        let next = self.cards.len() as u32;
        *self.cards.entry(raw).or_insert(next)
    }

    pub(crate) fn piece_slot(&self, raw: u32) -> Option<u32> {
        self.pieces.get(&raw).copied()
    }

    pub(crate) fn card_slot(&self, raw: u32) -> Option<u32> {
        self.cards.get(&raw).copied()
    }

    pub(crate) fn raw_piece(&self, slot: u32) -> Option<u32> {
        self.pieces
            .iter()
            .find_map(|(raw, candidate)| (*candidate == slot).then_some(*raw))
    }

    pub(crate) fn raw_card(&self, slot: u32) -> Option<u32> {
        self.cards
            .iter()
            .find_map(|(raw, candidate)| (*candidate == slot).then_some(*raw))
    }
}

pub fn encode_observation_v3(state: &GameState) -> ObservationV3 {
    let mut ids = LocalIds::new(state);
    let pieces = encode_pieces(state, &mut ids);
    let cards = encode_cards(state, &mut ids);
    let phase = encode_phase(state, &mut ids);
    let runtime = encode_runtime(state, &mut ids);
    let global_state = encode_global_state(state);
    let spatial = encode_spatial(&pieces);
    let global = encode_global_dense(state, &global_state);

    debug_assert_eq!(runtime.len(), CARD_RUNTIME_SOURCE_FIELDS.len());
    debug_assert!(spatial.iter().all(|value| (0.0..=1.0).contains(value)));
    debug_assert!(global.iter().all(|value| (0.0..=1.0).contains(value)));

    ObservationV3 {
        schema_id: OBSERVATION_V3_SCHEMA_ID.to_owned(),
        ruleset_id: state.ruleset_id.clone(),
        spatial_shape: [OBSERVATION_V3_SPATIAL_PLANES, 8, 8],
        spatial,
        global_shape: [OBSERVATION_V3_GLOBAL_FEATURES],
        global,
        state: global_state,
        pieces,
        cards,
        phase,
        runtime,
        piece_slot_count: ids.pieces.len() as u32,
        card_slot_count: ids.cards.len() as u32,
    }
}

fn encode_pieces(state: &GameState, ids: &mut LocalIds) -> Vec<BoardPieceV3> {
    let mut pieces = state
        .board
        .iter()
        .map(|(square, piece)| BoardPieceV3 {
            slot: ids.piece(piece.id),
            square: square_value(square, state.turn),
            owner: relative_player(piece.owner, state.turn),
            shadow_square: state
                .card_runtime
                .quantum_positions
                .get(&piece.id)
                .map(|shadow| square_value(*shadow, state.turn)),
            kind_code: piece.kind.stable_code(),
            capture_count: piece.capture_count,
        })
        .collect::<Vec<_>>();
    pieces.sort_by_key(|piece| piece.square);
    pieces
}

fn encode_cards(state: &GameState, ids: &mut LocalIds) -> Vec<CardRecordV3> {
    let mut cards = state
        .cards
        .iter()
        .map(|card| CardRecordV3 {
            slot: ids.card(card.instance_id),
            definition_id: card.definition_id.clone(),
            owner: relative_player(card.owner, state.turn),
            half_stars: card.half_stars,
            charges: card.charges,
            available_in_owner_turns: card
                .available_after_turn
                .map(|turn| turn.saturating_sub(state.turns_taken[card.owner.index()])),
        })
        .collect::<Vec<_>>();
    cards.sort_by_key(|card| card.slot);
    cards
}

fn encode_phase(state: &GameState, ids: &mut LocalIds) -> PhaseV3 {
    match &state.phase {
        GamePhase::Playing => PhaseV3::Playing,
        GamePhase::ForcedMove {
            piece_id,
            source,
            continuation,
        } => {
            let (source_kind, source_id) = match source {
                ForcedMoveSource::Platform => ("platform".to_owned(), None),
                ForcedMoveSource::Effect(id) => ("effect".to_owned(), Some(id.clone())),
            };
            PhaseV3::ForcedMove {
                piece: ids.piece(*piece_id),
                source_kind,
                source_id,
                continuation: match continuation {
                    ForcedMoveContinuation::ResumeTurn => "resume_turn",
                    ForcedMoveContinuation::FinishTurn => "finish_turn",
                }
                .to_owned(),
            }
        }
        GamePhase::Draft(draft) => PhaseV3::Draft {
            draft_phase: match draft.phase {
                DraftPhase::Opening => "opening",
                DraftPhase::Middle => "middle",
                DraftPhase::End => "end",
            }
            .to_owned(),
            offers: draft.offers.clone(),
            resume_turn: relative_player(draft.resume_turn, state.turn),
        },
        GamePhase::AwaitingChoice(choice) => PhaseV3::AwaitingChoice {
            choice_slot: 0,
            owner: relative_player(choice.owner, state.turn),
            source_card: ids.card(choice.source_card_instance),
            handler_id: model_choice_handler_token(&choice.handler_id),
            options: (0..choice.options.len())
                .map(|index| model_choice_option_token(&choice.handler_id, index))
                .collect(),
            continuation: match choice.continuation {
                ChoiceContinuation::ResumeTurn => "resume_turn",
                ChoiceContinuation::FinishTurn => "finish_turn",
            }
            .to_owned(),
        },
    }
}

pub(crate) fn model_choice_option_token(handler_id: &str, option_index: usize) -> String {
    format!(
        "choice:{}:{option_index}",
        model_choice_handler_token(handler_id)
    )
}

pub(crate) fn model_choice_handler_token(handler_id: &str) -> String {
    augment_core::GameState::choice_handler_family(handler_id).to_owned()
}

fn encode_global_state(state: &GameState) -> GlobalStateV3 {
    let [self_index, opponent_index] = relative_indices(state.turn);
    GlobalStateV3 {
        mode_id: match &state.mode {
            Mode::Standard => "standard".to_owned(),
            Mode::Chaos => "chaos".to_owned(),
            Mode::Grand => "grand".to_owned(),
            Mode::Campaign(_) => "campaign".to_owned(),
        },
        ply: state.ply,
        quiet_ply: state.quiet_ply,
        actions_remaining: state.actions_remaining,
        cards_used_this_turn: [
            state.cards_used_this_turn[self_index],
            state.cards_used_this_turn[opponent_index],
        ],
        half_star_totals: [
            state.half_star_totals[self_index],
            state.half_star_totals[opponent_index],
        ],
        turns_taken: [
            state.turns_taken[self_index],
            state.turns_taken[opponent_index],
        ],
        completed_drafts: state.completed_drafts,
        shared_turn_count: state.shared_turn_count(),
        current_repetition_count: state.current_repetition_count(),
        status: encode_status(&state.status, state.turn),
        deathmatch: state.deathmatch.map(|deathmatch| DeathmatchV3 {
            age_shared_turns: state
                .shared_turn_count()
                .saturating_sub(deathmatch.started_at_shared_turn),
            quiet_full_turns: deathmatch.quiet_full_turns,
            interval_full_turns: deathmatch.interval_full_turns,
            progress_this_round: deathmatch.progress_this_round,
        }),
        en_passant: state.en_passant.map(|en_passant| EnPassantV3 {
            target: square_value(en_passant.target, state.turn),
            captured: square_value(en_passant.captured, state.turn),
            pawn_owner: relative_player(en_passant.pawn_owner, state.turn),
        }),
        applied_rules: state.applied_rules.clone(),
    }
}

fn encode_status(status: &GameStatus, perspective: Player) -> StatusV3 {
    match status {
        GameStatus::Ongoing => StatusV3::Ongoing,
        GameStatus::Won { winner, reason } => {
            let (reason, detail) = match reason {
                augment_core::WinReason::KingCaptured => ("king_captured", None),
                augment_core::WinReason::NoLegalActions => ("no_legal_actions", None),
                augment_core::WinReason::LowerStarsOnRepetition => {
                    ("lower_stars_on_repetition", None)
                }
                augment_core::WinReason::LowerStarsAfterDeathmatch => {
                    ("lower_stars_after_deathmatch", None)
                }
                augment_core::WinReason::CardEffect(id) => ("card_effect", Some(id.clone())),
            };
            StatusV3::Won {
                winner: relative_player(*winner, perspective),
                reason: reason.to_owned(),
                detail,
            }
        }
        GameStatus::Draw { reason } => {
            let (reason, detail) = match reason {
                augment_core::DrawReason::EqualStarsOnRepetition => {
                    ("equal_stars_on_repetition", None)
                }
                augment_core::DrawReason::EqualStarsAfterDeathmatch => {
                    ("equal_stars_after_deathmatch", None)
                }
                augment_core::DrawReason::CardEffect(id) => ("card_effect", Some(id.clone())),
            };
            StatusV3::Draw {
                reason: reason.to_owned(),
                detail,
            }
        }
    }
}

fn encode_spatial(pieces: &[BoardPieceV3]) -> Vec<f32> {
    let mut spatial = vec![0.0; OBSERVATION_V3_SPATIAL_SIZE];
    for piece in pieces {
        let plane = match (piece.kind_code, piece.owner) {
            (59..=68, RelativePlayerV3::SelfPlayer) => 116 + 2 * usize::from(piece.kind_code - 59),
            (59..=68, RelativePlayerV3::Opponent) => 117 + 2 * usize::from(piece.kind_code - 59),
            (1..=58, RelativePlayerV3::SelfPlayer) => usize::from(piece.kind_code - 1),
            (1..=58, RelativePlayerV3::Opponent) => 58 + usize::from(piece.kind_code - 1),
            _ => panic!("unreviewed spatial piece kind {}", piece.kind_code),
        };
        spatial[plane * 64 + usize::from(piece.square)] = 1.0;
    }
    spatial
}

fn encode_global_dense(state: &GameState, exact: &GlobalStateV3) -> Vec<f32> {
    let mut global = vec![0.0; OBSERVATION_V3_GLOBAL_FEATURES];
    global[0] = unit_count(u64::from(exact.actions_remaining));
    global[1] = unit_count(u64::from(exact.ply));
    global[2] = unit_count(u64::from(exact.quiet_ply));
    global[3] = unit_count(u64::from(exact.cards_used_this_turn[0]));
    global[4] = unit_count(u64::from(exact.cards_used_this_turn[1]));
    global[5] = unit_count(u64::from(exact.half_star_totals[0]));
    global[6] = unit_count(u64::from(exact.half_star_totals[1]));
    global[7] = unit_count(u64::from(exact.turns_taken[0]));
    global[8] = unit_count(u64::from(exact.turns_taken[1]));
    for (offset, completed) in exact.completed_drafts.into_iter().enumerate() {
        global[9 + offset] = f32::from(completed);
    }
    if let Some(deathmatch) = exact.deathmatch {
        global[12] = 1.0;
        global[13] = unit_count(u64::from(deathmatch.quiet_full_turns));
        global[14] = unit_count(u64::from(deathmatch.interval_full_turns));
        global[15] = f32::from(deathmatch.progress_this_round);
    }
    global[16] = f32::from(exact.en_passant.is_some());
    global[17] = unit_count(
        state
            .cards
            .iter()
            .filter(|card| card.owner == state.turn)
            .count() as u64,
    );
    global[18] = unit_count(
        state
            .cards
            .iter()
            .filter(|card| card.owner != state.turn)
            .count() as u64,
    );
    global[19] = unit_count(exact.applied_rules.len() as u64);
    global[20] = unit_count(u64::from(exact.shared_turn_count));
    global[21] = unit_count(u64::from(exact.current_repetition_count));
    global[match state.phase {
        GamePhase::Playing => 22,
        GamePhase::ForcedMove { .. } => 23,
        GamePhase::Draft(_) => 24,
        GamePhase::AwaitingChoice(_) => 25,
    }] = 1.0;
    global[match state.mode {
        Mode::Standard => 26,
        Mode::Chaos => 27,
        Mode::Grand => 28,
        Mode::Campaign(_) => 29,
    }] = 1.0;
    global[30] = f32::from(state.card_runtime.acceleration);
    global[31] = f32::from(state.card_runtime.revelation);
    global
}

fn unit_count(value: u64) -> f32 {
    let value = value as f64;
    (value / (value + 1.0)) as f32
}

fn canonical_square(square: Square, perspective: Player) -> Square {
    match perspective {
        Player::White => square,
        Player::Black => Square::new(square.file(), 7 - square.rank()).expect("mirrored square"),
    }
}

fn square_value(square: Square, perspective: Player) -> u8 {
    canonical_square(square, perspective).index() as u8
}

fn relative_player(player: Player, perspective: Player) -> RelativePlayerV3 {
    if player == perspective {
        RelativePlayerV3::SelfPlayer
    } else {
        RelativePlayerV3::Opponent
    }
}

fn relative_indices(perspective: Player) -> [usize; 2] {
    [perspective.index(), perspective.opponent().index()]
}

fn encode_runtime(state: &GameState, ids: &mut LocalIds) -> Vec<RuntimeFeatureV3> {
    encode_runtime_fields(state, &state.card_runtime, ids)
}

// Implemented below in source-field order. Keeping this function separate makes the contract
// reviewable and lets tests compare every emitted field directly with CARD_RUNTIME_SOURCE_FIELDS.
fn encode_runtime_fields(
    state: &GameState,
    runtime: &CardRuntimeState,
    ids: &mut LocalIds,
) -> Vec<RuntimeFeatureV3> {
    let mut fields = Vec::with_capacity(CARD_RUNTIME_SOURCE_FIELDS.len());
    macro_rules! push {
        ($name:literal, $value:expr) => {{
            let index = fields.len();
            assert_eq!(CARD_RUNTIME_SOURCE_FIELDS[index], $name);
            fields.push(RuntimeFeatureV3 {
                field_id: (index + 1) as u16,
                value: $value,
            });
        }};
    }

    push!(
        "opening_auto_card_ids",
        card_set(&runtime.opening_auto_card_ids, ids)
    );
    push!("hill_king", bool_pair(runtime.hill_king, state.turn));
    push!(
        "geneva_convention",
        bool_pair(runtime.geneva_convention, state.turn)
    );
    push!("pawn_sprint", bool_pair(runtime.pawn_sprint, state.turn));
    push!(
        "early_promotion",
        bool_pair(runtime.early_promotion, state.turn)
    );
    push!(
        "initiative",
        pair_with(&runtime.initiative, state.turn, |status| match status {
            None => RuntimeValueV3::None,
            Some(status) => initiative_value(status, state),
        })
    );
    push!(
        "dice_locks",
        pair_with(&runtime.dice_locks, state.turn, |status| match status {
            None => RuntimeValueV3::None,
            Some(status) => dice_lock_value(status),
        })
    );
    push!("moved_piece_ids", piece_set(&runtime.moved_piece_ids, ids));
    push!(
        "fresh_no_capture_until",
        piece_map(
            &runtime.fresh_no_capture_until,
            ids,
            |piece_id, deadline, _| {
                RuntimeValueV3::Unsigned(u64::from(remaining_piece_turn(
                    state, piece_id, *deadline,
                )))
            }
        )
    );
    push!(
        "disarmed",
        piece_map(&runtime.disarmed, ids, |_, remaining, _| {
            RuntimeValueV3::Unsigned(u64::from(*remaining))
        })
    );
    push!(
        "frozen",
        piece_map(&runtime.frozen, ids, |_, status, _| {
            frozen_value(status, state.turn)
        })
    );
    push!(
        "reposition",
        match runtime.reposition {
            None => RuntimeValueV3::None,
            Some(status) => reposition_value(&status, ids),
        }
    );
    push!("revelation", RuntimeValueV3::Bool(runtime.revelation));
    push!("parry_piece_ids", piece_set(&runtime.parry_piece_ids, ids));
    push!(
        "empty_lunchboxes",
        piece_map(&runtime.empty_lunchboxes, ids, |piece_id, status, _| {
            lunchbox_value(piece_id, status, state)
        })
    );
    push!(
        "platform",
        match &runtime.platform {
            None => RuntimeValueV3::None,
            Some(platform) => platform_value(platform, state, ids),
        }
    );
    push!(
        "herald_jump_locked_until",
        piece_map(
            &runtime.herald_jump_locked_until,
            ids,
            |piece_id, deadline, _| RuntimeValueV3::Unsigned(u64::from(remaining_piece_turn(
                state, piece_id, *deadline
            )))
        )
    );
    push!(
        "windmill_modes",
        piece_map(&runtime.windmill_modes, ids, |_, mode, _| {
            RuntimeValueV3::Text(
                match mode {
                    WindmillMode::Bishop => "bishop",
                    WindmillMode::Rook => "rook",
                }
                .to_owned(),
            )
        })
    );
    push!(
        "log_statuses",
        piece_map(&runtime.log_statuses, ids, |piece_id, status, _| {
            log_status_value(piece_id, status, state)
        })
    );
    push!(
        "wizard_mana",
        piece_map(&runtime.wizard_mana, ids, |_, mana, _| {
            RuntimeValueV3::Unsigned(u64::from(*mana))
        })
    );
    push!(
        "reaper_souls",
        piece_map(&runtime.reaper_souls, ids, |_, souls, _| {
            RuntimeValueV3::Unsigned(u64::from(*souls))
        })
    );
    push!(
        "shielded_piece_ids",
        piece_set(&runtime.shielded_piece_ids, ids)
    );
    push!(
        "protected_piece_ids",
        piece_set(&runtime.protected_piece_ids, ids)
    );
    push!(
        "idol_used_turn",
        piece_map(&runtime.idol_used_turn, ids, |piece_id, used_turn, _| {
            let owner = piece_owner(state, piece_id);
            RuntimeValueV3::Unsigned(u64::from(
                state.turns_taken[owner.index()].saturating_sub(*used_turn),
            ))
        })
    );
    push!(
        "idol_rest_piece_ids",
        piece_set(&runtime.idol_rest_piece_ids, ids)
    );
    push!(
        "delayed_hazards",
        RuntimeValueV3::List(
            runtime
                .delayed_hazards
                .iter()
                .map(|hazard| hazard_value(hazard, state.turn))
                .collect(),
        )
    );
    push!("skip_turns", bool_pair(runtime.skip_turns, state.turn));
    push!(
        "piece_origins",
        piece_map(&runtime.piece_origins, ids, |_, square, _| {
            RuntimeValueV3::Square(square_value(*square, state.turn))
        })
    );
    push!(
        "large_pieces",
        piece_map(&runtime.large_pieces, ids, |_, status, _| {
            large_piece_value(status, state.turn)
        })
    );
    push!(
        "loyalist_piece_ids",
        piece_set(&runtime.loyalist_piece_ids, ids)
    );
    push!("royal_piece_ids", piece_set(&runtime.royal_piece_ids, ids));
    push!(
        "calling_cards",
        piece_map(&runtime.calling_cards, ids, |_, player, _| {
            RuntimeValueV3::Player(relative_player(*player, state.turn))
        })
    );
    push!(
        "holdout_promotions",
        piece_map(&runtime.holdout_promotions, ids, |piece_id, deadline, _| {
            RuntimeValueV3::Unsigned(u64::from(remaining_piece_turn(state, piece_id, *deadline)))
        })
    );
    push!(
        "pending_otherworld",
        RuntimeValueV3::List(
            runtime
                .pending_otherworld
                .iter()
                .map(|pending| otherworld_value(pending, state, ids))
                .collect(),
        )
    );
    push!(
        "merchant_gold",
        piece_map(&runtime.merchant_gold, ids, |_, gold, _| {
            RuntimeValueV3::Unsigned(u64::from(*gold))
        })
    );

    for (name, values) in [
        ("democracy", runtime.democracy),
        ("fast_growth", runtime.fast_growth),
        ("fianchetto", runtime.fianchetto),
    ] {
        let index = fields.len();
        assert_eq!(CARD_RUNTIME_SOURCE_FIELDS[index], name);
        fields.push(RuntimeFeatureV3 {
            field_id: (index + 1) as u16,
            value: bool_pair(values, state.turn),
        });
    }

    push!(
        "field_promotion",
        bool_pair(runtime.field_promotion, state.turn)
    );
    push!(
        "field_promoted_piece_ids",
        piece_set(&runtime.field_promoted_piece_ids, ids)
    );
    push!(
        "frontline_response",
        bool_pair(runtime.frontline_response, state.turn)
    );
    push!("relay", bool_pair(runtime.relay, state.turn));
    push!("gomoku", bool_pair(runtime.gomoku, state.turn));
    push!(
        "magic_girl_surge",
        bool_pair(runtime.magic_girl_surge, state.turn)
    );
    push!(
        "free_move_no_capture",
        bool_pair(runtime.free_move_no_capture, state.turn)
    );
    push!(
        "siren_exposure",
        RuntimeValueV3::List(
            runtime
                .siren_exposure
                .iter()
                .map(|((siren, target), count)| {
                    record(vec![
                        ("first", RuntimeValueV3::Piece(ids.piece(*siren))),
                        ("second", RuntimeValueV3::Piece(ids.piece(*target))),
                        ("count", RuntimeValueV3::Unsigned(u64::from(*count))),
                    ])
                })
                .collect(),
        )
    );
    push!(
        "trickster_move_kinds",
        piece_map(&runtime.trickster_move_kinds, ids, |_, kind, _| {
            RuntimeValueV3::PieceKind(kind.stable_code())
        })
    );
    push!(
        "undead_resurrections",
        RuntimeValueV3::List(
            runtime
                .undead_resurrections
                .iter()
                .map(|pending| {
                    record(vec![
                        (
                            "piece",
                            runtime_piece_value(&pending.piece, ids, state.turn),
                        ),
                        (
                            "remaining_ply",
                            RuntimeValueV3::Unsigned(u64::from(
                                pending.due_ply.saturating_sub(state.ply),
                            )),
                        ),
                    ])
                })
                .collect(),
        )
    );
    push!(
        "judgment_exiles",
        RuntimeValueV3::List(
            runtime
                .judgment_exiles
                .iter()
                .map(|exile| {
                    record(vec![
                        ("piece", runtime_piece_value(&exile.piece, ids, state.turn)),
                        (
                            "square",
                            RuntimeValueV3::Square(square_value(exile.square, state.turn)),
                        ),
                        (
                            "resume_turn",
                            RuntimeValueV3::Text(
                                draft_phase_name(exile.return_at_draft).to_owned(),
                            ),
                        ),
                    ])
                })
                .collect(),
        )
    );
    push!(
        "move_replays",
        pair_with(&runtime.move_replays, state.turn, |replay| match replay {
            None => RuntimeValueV3::None,
            Some(replay) => record(vec![
                (
                    "pieces",
                    RuntimeValueV3::List(
                        replay
                            .cells
                            .iter()
                            .map(|cell| {
                                record(vec![
                                    (
                                        "square",
                                        RuntimeValueV3::Square(square_value(
                                            cell.square,
                                            state.turn,
                                        )),
                                    ),
                                    (
                                        "first",
                                        optional_runtime_piece_value(
                                            cell.before.as_ref(),
                                            ids,
                                            state.turn,
                                        ),
                                    ),
                                    (
                                        "second",
                                        optional_runtime_piece_value(
                                            cell.after.as_ref(),
                                            ids,
                                            state.turn,
                                        ),
                                    ),
                                ])
                            })
                            .collect(),
                    ),
                ),
                (
                    "first",
                    optional_en_passant_value(replay.en_passant_before.as_ref(), state.turn),
                ),
                (
                    "second",
                    optional_en_passant_value(replay.en_passant_after.as_ref(), state.turn),
                ),
            ]),
        })
    );

    for (name, values) in [
        ("iron_monarch", runtime.iron_monarch),
        ("pawn_conversion", runtime.pawn_conversion),
        ("backward_knight", runtime.backward_knight),
        ("breakthrough_pawns", runtime.breakthrough_pawns),
        ("corner_kick", runtime.corner_kick),
        ("en_passant_frenzy", runtime.en_passant_frenzy),
        ("file_surge", runtime.file_surge),
        ("horse_riding", runtime.horse_riding),
        ("knight_injury", runtime.knight_injury),
        ("pawn_leap", runtime.pawn_leap),
        ("radical_charge", runtime.radical_charge),
        ("retreat", runtime.retreat),
        ("rook_lift", runtime.rook_lift),
        ("encouragement", runtime.encouragement),
        ("exhaustion", runtime.exhaustion),
        ("overwhelm", runtime.overwhelm),
        ("imperial_studies", runtime.imperial_studies),
        ("mad_horse", runtime.mad_horse),
        ("queen_afterimage", runtime.queen_afterimage),
        ("religious_victory", runtime.religious_victory),
    ] {
        let index = fields.len();
        assert_eq!(CARD_RUNTIME_SOURCE_FIELDS[index], name);
        fields.push(RuntimeFeatureV3 {
            field_id: (index + 1) as u16,
            value: bool_pair(values, state.turn),
        });
    }

    push!(
        "basic_training_piece_ids",
        piece_set(&runtime.basic_training_piece_ids, ids)
    );
    push!(
        "promotion_rush_piece_ids",
        piece_set(&runtime.promotion_rush_piece_ids, ids)
    );
    push!(
        "evasion_piece_ids",
        piece_set(&runtime.evasion_piece_ids, ids)
    );
    push!("ghost_piece_ids", piece_set(&runtime.ghost_piece_ids, ids));
    push!(
        "inertia_piece_ids",
        piece_set(&runtime.inertia_piece_ids, ids)
    );
    push!(
        "ice_sheet",
        piece_map(&runtime.ice_sheet, ids, |_, status, _| {
            timed_piece_value(status, state.turn)
        })
    );
    push!(
        "severed",
        piece_map(&runtime.severed, ids, |_, status, _| {
            timed_piece_value(status, state.turn)
        })
    );
    push!(
        "royal_command",
        pair_with_player(
            &runtime.royal_command,
            state.turn,
            |player, status| match status {
                None => RuntimeValueV3::None,
                Some(status) => royal_command_value(status, state, player),
            }
        )
    );
    push!("socialism", u32_pair(runtime.socialism, state.turn));
    push!("taunt", u32_pair(runtime.taunt, state.turn));
    push!(
        "armistice_remaining_ply",
        RuntimeValueV3::Unsigned(u64::from(runtime.armistice_remaining_ply))
    );
    push!(
        "exhaustion_last_piece",
        pair_with(
            &runtime.exhaustion_last_piece,
            state.turn,
            |piece| match piece {
                None => RuntimeValueV3::None,
                Some(piece_id) => RuntimeValueV3::Piece(ids.piece(*piece_id)),
            }
        )
    );
    push!(
        "exhaustion_count",
        u8_pair(runtime.exhaustion_count, state.turn)
    );
    push!(
        "chameleon_piece_ids",
        piece_set(&runtime.chameleon_piece_ids, ids)
    );
    push!(
        "chimera_piece_ids",
        piece_set(&runtime.chimera_piece_ids, ids)
    );
    push!(
        "imperial_moves",
        piece_map(&runtime.imperial_moves, ids, |_, kinds, _| {
            kind_set(kinds)
        })
    );
    push!("mistake_card", bool_pair(runtime.mistake_card, state.turn));
    push!(
        "poisoned_piece_ids",
        piece_set(&runtime.poisoned_piece_ids, ids)
    );
    push!(
        "sacrifice_protection",
        piece_map(&runtime.sacrifice_protection, ids, |_, status, _| {
            timed_piece_value(status, state.turn)
        })
    );
    push!(
        "last_resistance",
        piece_map(&runtime.last_resistance, ids, |_, status, _| {
            timed_piece_value(status, state.turn)
        })
    );
    push!(
        "coronation_protection",
        piece_map(&runtime.coronation_protection, ids, |piece_id, start, _| {
            let elapsed = state
                .board
                .find_by_id(piece_id)
                .map(|(_, piece)| state.turns_taken[piece.owner.index()].saturating_sub(*start))
                .unwrap_or(0);
            RuntimeValueV3::Unsigned(u64::from(elapsed))
        })
    );
    push!(
        "queens_gambit_protection",
        piece_set(&runtime.queens_gambit_protection, ids)
    );
    push!(
        "potion_basic_training_piece_ids",
        piece_set(&runtime.potion_basic_training_piece_ids, ids)
    );
    push!(
        "poison_stun_turns",
        piece_map(&runtime.poison_stun_turns, ids, |_, remaining, _| {
            RuntimeValueV3::Unsigned(u64::from(*remaining))
        })
    );
    push!(
        "potion_manner_piece_ids",
        piece_set(&runtime.potion_manner_piece_ids, ids)
    );
    push!(
        "manner_captured_last_piece_ids",
        piece_set(&runtime.manner_captured_last_piece_ids, ids)
    );
    push!(
        "potion_saturation_piece_ids",
        piece_set(&runtime.potion_saturation_piece_ids, ids)
    );
    push!(
        "saturation_capture_counts",
        piece_map(&runtime.saturation_capture_counts, ids, |_, count, _| {
            RuntimeValueV3::Unsigned(u64::from(*count))
        })
    );
    push!("bishop_snipe", bool_pair(runtime.bishop_snipe, state.turn));
    push!(
        "explosive_piece_ids",
        piece_set(&runtime.explosive_piece_ids, ids)
    );
    push!(
        "trojan_piece_ids",
        piece_set(&runtime.trojan_piece_ids, ids)
    );
    push!(
        "feudal_contracts",
        RuntimeValueV3::List(
            runtime
                .feudal_contracts
                .iter()
                .map(|contract| feudal_value(contract, state.turn, ids))
                .collect(),
        )
    );
    push!(
        "gale_due_turns",
        pair_with_player(&runtime.gale_due_turns, state.turn, |player, deadlines| {
            RuntimeValueV3::List(
                deadlines
                    .iter()
                    .map(|deadline| {
                        RuntimeValueV3::Unsigned(u64::from(
                            deadline.saturating_sub(state.turns_taken[player.index()]),
                        ))
                    })
                    .collect(),
            )
        })
    );
    push!(
        "pending_panic",
        RuntimeValueV3::List(
            runtime
                .pending_panic
                .iter()
                .map(|pending| panic_value(pending, state.turn, ids))
                .collect(),
        )
    );
    push!(
        "pending_premoves",
        RuntimeValueV3::List(
            runtime
                .pending_premoves
                .iter()
                .map(|pending| premove_value(pending, state))
                .collect(),
        )
    );
    push!(
        "pending_scarecrows",
        RuntimeValueV3::List(
            runtime
                .pending_scarecrows
                .iter()
                .map(|pending| scarecrow_value(pending, state.turn))
                .collect(),
        )
    );
    push!(
        "quantum_pending",
        bool_pair(runtime.quantum_pending, state.turn)
    );
    push!(
        "quantum_positions",
        piece_map(&runtime.quantum_positions, ids, |_, square, _| {
            RuntimeValueV3::Square(square_value(*square, state.turn))
        })
    );
    push!(
        "stealth_piece_ids",
        piece_set(&runtime.stealth_piece_ids, ids)
    );
    push!(
        "submerged_piece_ids",
        piece_set(&runtime.submerged_piece_ids, ids)
    );
    push!("substitution", bool_pair(runtime.substitution, state.turn));
    push!("switcheroo", bool_pair(runtime.switcheroo, state.turn));
    push!(
        "twin_bonds",
        RuntimeValueV3::List(
            runtime
                .twin_bonds
                .iter()
                .map(|bond| twin_value(bond, ids))
                .collect(),
        )
    );
    push!(
        "witch_trials",
        piece_map(&runtime.witch_trials, ids, |_, status, _| {
            timed_piece_value(status, state.turn)
        })
    );
    push!(
        "pending_trolley_by",
        pair_with(
            &runtime.pending_trolley_by,
            state.turn,
            |player| match player {
                None => RuntimeValueV3::None,
                Some(player) => RuntimeValueV3::Player(relative_player(*player, state.turn)),
            }
        )
    );
    push!(
        "active_trolley",
        match &runtime.active_trolley {
            None => RuntimeValueV3::None,
            Some(trolley) => trolley_value(trolley, state.turn, ids),
        }
    );
    push!(
        "baby_bear_grow_turns",
        piece_map(&runtime.baby_bear_grow_turns, ids, |_, deadline, _| {
            RuntimeValueV3::Unsigned(u64::from(
                deadline.saturating_sub(state.shared_turn_count()),
            ))
        })
    );
    push!(
        "bear_retaliations",
        piece_map(&runtime.bear_retaliations, ids, |_, remaining, _| {
            RuntimeValueV3::Unsigned(u64::from(*remaining))
        })
    );
    push!(
        "bribe_reverts",
        RuntimeValueV3::List(
            runtime
                .bribe_reverts
                .iter()
                .map(|revert| bribe_value(revert, state, ids))
                .collect(),
        )
    );
    push!("charge_pawns", piece_set(&runtime.charge_pawns, ids));
    push!("coronation", bool_pair(runtime.coronation, state.turn));
    push!("frenzy_pawns", piece_set(&runtime.frenzy_pawns, ids));
    push!(
        "hallucination_until_ply",
        pair_with(&runtime.hallucination_until_ply, state.turn, |deadline| {
            deadline.map_or(RuntimeValueV3::None, |deadline| {
                RuntimeValueV3::Unsigned(u64::from(deadline.saturating_sub(state.ply)))
            })
        })
    );
    push!("moving", bool_pair(runtime.moving, state.turn));
    push!(
        "moving_consecutive",
        pair_with(&runtime.moving_consecutive, state.turn, |(piece, count)| {
            record(vec![
                (
                    "piece",
                    piece.map_or(RuntimeValueV3::None, |piece_id| {
                        RuntimeValueV3::Piece(ids.piece(piece_id))
                    }),
                ),
                ("count", RuntimeValueV3::Unsigned(u64::from(*count))),
            ])
        })
    );
    push!(
        "necromancy_reverts",
        RuntimeValueV3::List(
            runtime
                .necromancy_reverts
                .iter()
                .map(|revert| necromancy_value(revert, state, ids))
                .collect(),
        )
    );
    push!(
        "reverse_pawns_charges",
        u8_pair(runtime.reverse_pawns_charges, state.turn)
    );
    push!(
        "stake_statuses",
        piece_map(&runtime.stake_statuses, ids, |_, status, _| {
            timed_piece_value(status, state.turn)
        })
    );
    push!(
        "underpromotion",
        bool_pair(runtime.underpromotion, state.turn)
    );
    push!(
        "vip_pending",
        RuntimeValueV3::List(
            runtime
                .vip_pending
                .iter()
                .map(|pending| vip_value(pending, state, ids))
                .collect(),
        )
    );
    push!("barricades", square_set(&runtime.barricades, state.turn));
    push!(
        "chains",
        RuntimeValueV3::List(
            runtime
                .chains
                .iter()
                .map(|chain| {
                    RuntimeValueV3::List(
                        chain
                            .iter()
                            .map(|piece_id| RuntimeValueV3::Piece(ids.piece(*piece_id)))
                            .collect(),
                    )
                })
                .collect(),
        )
    );
    push!(
        "collapsed_files_ranks",
        RuntimeValueV3::Bool(runtime.collapsed_files_ranks)
    );
    push!(
        "desperado_pieces",
        piece_set(&runtime.desperado_pieces, ids)
    );
    push!("palaces", player_square_map(&runtime.palaces, state.turn));
    push!("portals", optional_square_pair(runtime.portals, state.turn));
    push!(
        "bunkers",
        piece_map(&runtime.bunkers, ids, |_, hp, _| {
            RuntimeValueV3::Unsigned(u64::from(*hp))
        })
    );
    push!("zugzwang", bool_pair(runtime.zugzwang, state.turn));
    push!("bina_mate", bool_pair(runtime.bina_mate, state.turn));
    push!("final_weapon", bool_pair(runtime.final_weapon, state.turn));
    push!(
        "icbm_pending",
        RuntimeValueV3::List(
            runtime
                .icbm_pending
                .iter()
                .map(|pending| icbm_value(pending, state, ids))
                .collect(),
        )
    );
    push!(
        "prophecy",
        pair_with(&runtime.prophecy, state.turn, |deadline| {
            deadline.map_or(RuntimeValueV3::None, |deadline| {
                RuntimeValueV3::Unsigned(u64::from(deadline.saturating_sub(state.ply)))
            })
        })
    );
    push!("racing_king", bool_pair(runtime.racing_king, state.turn));
    push!(
        "spies",
        piece_map(&runtime.spies, ids, |_, player, _| {
            RuntimeValueV3::Player(relative_player(*player, state.turn))
        })
    );
    push!(
        "ultimatum",
        runtime.ultimatum.map_or(RuntimeValueV3::None, |deadline| {
            RuntimeValueV3::Unsigned(u64::from(deadline.saturating_sub(state.ply)))
        })
    );
    push!("vanish", bool_pair(runtime.vanish, state.turn));
    push!("acceleration", RuntimeValueV3::Bool(runtime.acceleration));
    push!(
        "chaos_no_capture_until_ply",
        RuntimeValueV3::Unsigned(u64::from(
            runtime.chaos_no_capture_until_ply.saturating_sub(state.ply),
        ))
    );
    push!("black_holes", square_set(&runtime.black_holes, state.turn));
    push!(
        "camouflage_color",
        RuntimeValueV3::Bool(runtime.camouflage_color)
    );
    push!("conveyor", RuntimeValueV3::Bool(runtime.conveyor));
    push!("cool_guy_pieces", piece_set(&runtime.cool_guy_pieces, ids));
    push!("crown_pos", optional_square(runtime.crown_pos, state.turn));
    push!(
        "crown_holder",
        runtime
            .crown_holder
            .map_or(RuntimeValueV3::None, |(player, piece_id)| {
                record(vec![
                    (
                        "player",
                        RuntimeValueV3::Player(relative_player(player, state.turn)),
                    ),
                    ("piece", RuntimeValueV3::Piece(ids.piece(piece_id))),
                ])
            })
    );
    push!(
        "football_pos",
        optional_square(runtime.football_pos, state.turn)
    );
    push!(
        "high_ground_squares",
        square_set(&runtime.high_ground_squares, state.turn)
    );
    push!(
        "highways",
        RuntimeValueV3::List(
            runtime
                .highways
                .iter()
                .map(|file| RuntimeValueV3::Unsigned(u64::from(*file)))
                .collect(),
        )
    );
    push!("macho_chess", RuntimeValueV3::Bool(runtime.macho_chess));
    push!("mistake_rule", RuntimeValueV3::Bool(runtime.mistake_rule));
    push!(
        "monochrome_chess",
        RuntimeValueV3::Bool(runtime.monochrome_chess)
    );
    push!(
        "monster_rule",
        runtime
            .monster_rule
            .map_or(RuntimeValueV3::None, |piece_id| {
                RuntimeValueV3::Piece(ids.piece(piece_id))
            })
    );
    push!(
        "periodic_collapse_next",
        runtime
            .periodic_collapse_next
            .map_or(RuntimeValueV3::None, |deadline| {
                RuntimeValueV3::Unsigned(u64::from(deadline.saturating_sub(state.ply)))
            })
    );
    push!(
        "rule_portals",
        optional_square_pair(runtime.rule_portals, state.turn)
    );
    push!("recycling", RuntimeValueV3::Bool(runtime.recycling));
    push!("rule_bombs", square_set(&runtime.rule_bombs, state.turn));
    push!(
        "saturation_rule",
        RuntimeValueV3::Bool(runtime.saturation_rule)
    );
    push!("transcendence", RuntimeValueV3::Bool(runtime.transcendence));
    push!(
        "winter_kingdom_next",
        runtime
            .winter_kingdom_next
            .map_or(RuntimeValueV3::None, |deadline| {
                record(vec![
                    (
                        "clock_owner",
                        RuntimeValueV3::Player(relative_player(Player::White, state.turn)),
                    ),
                    (
                        "remaining_turns",
                        RuntimeValueV3::Unsigned(u64::from(
                            deadline.saturating_sub(state.turns_taken[Player::White.index()]),
                        )),
                    ),
                ])
            })
    );
    push!(
        "cloned_passives",
        pair_with(&runtime.cloned_passives, state.turn, |passives| {
            RuntimeValueV3::List(passives.iter().cloned().map(RuntimeValueV3::Text).collect())
        })
    );

    // Append-only contract: these fields were added after the original 169.
    push!(
        "capture_the_flag",
        match &runtime.capture_the_flag {
            None => RuntimeValueV3::None,
            Some(flags) => pair_with_player(&flags.flags, state.turn, |owner, square| {
                record(vec![
                    (
                        "owner",
                        RuntimeValueV3::Player(relative_player(owner, state.turn)),
                    ),
                    (
                        "square",
                        RuntimeValueV3::Square(square_value(*square, state.turn)),
                    ),
                    (
                        "occupation",
                        match flags.occupations[owner.index()] {
                            None => RuntimeValueV3::None,
                            Some(occupation) => record(vec![
                                (
                                    "piece",
                                    RuntimeValueV3::Piece(ids.piece(occupation.piece_id)),
                                ),
                                // CTF resolves only when owner turns are strictly greater than
                                // this threshold. Keep overdue (-1) distinct from due (0).
                                (
                                    "trigger_in_turns",
                                    RuntimeValueV3::Signed(
                                        i64::from(occupation.after_owner_turn)
                                            - i64::from(state.turns_taken[owner.index()]),
                                    ),
                                ),
                            ]),
                        },
                    ),
                ])
            }),
        }
    );
    push!(
        "pending_lobsters",
        RuntimeValueV3::List(
            runtime
                .pending_lobsters
                .iter()
                .map(|pending| record(vec![
                    (
                        "owner",
                        RuntimeValueV3::Player(relative_player(pending.owner, state.turn))
                    ),
                    (
                        "square",
                        RuntimeValueV3::Square(square_value(pending.square, state.turn))
                    ),
                    (
                        "remaining_ply",
                        RuntimeValueV3::Signed(i64::from(pending.due_ply) - i64::from(state.ply),)
                    ),
                ]))
                .collect(),
        )
    );
    // Queue order is meaningful: the last eligible portal replaces earlier ones.
    push!(
        "pending_portals",
        RuntimeValueV3::List(
            runtime
                .pending_portals
                .iter()
                .map(|pending| record(vec![
                    (
                        "owner",
                        RuntimeValueV3::Player(relative_player(pending.owner, state.turn))
                    ),
                    (
                        "squares",
                        RuntimeValueV3::List(
                            pending
                                .squares
                                .iter()
                                .map(|square| {
                                    RuntimeValueV3::Square(square_value(*square, state.turn))
                                })
                                .collect()
                        )
                    ),
                    (
                        "trigger_in_turns",
                        RuntimeValueV3::Signed(
                            i64::from(pending.due_turn)
                                - i64::from(state.turns_taken[pending.owner.index()]),
                        )
                    ),
                ]))
                .collect(),
        )
    );

    push!("majesty", bool_pair(runtime.majesty, state.turn));
    push!("proficiency", bool_pair(runtime.proficiency, state.turn));
    push!("killer_king", bool_pair(runtime.killer_king, state.turn));
    push!("disassembly", bool_pair(runtime.disassembly, state.turn));
    push!("overtake", bool_pair(runtime.overtake, state.turn));
    push!("reversal", bool_pair(runtime.reversal, state.turn));
    push!(
        "outpost_protected_piece_ids",
        piece_set(&runtime.outpost_protected_piece_ids, ids)
    );
    push!("symmetry", bool_pair(runtime.symmetry, state.turn));
    push!("mutation", bool_pair(runtime.mutation, state.turn));
    push!(
        "long_en_passant",
        bool_pair(runtime.long_en_passant, state.turn)
    );
    push!(
        "parrot_memory",
        pair_with(&runtime.parrot_memory, state.turn, |memory| {
            match memory {
                None => RuntimeValueV3::None,
                Some(memory) => record(vec![
                    (
                        "source",
                        RuntimeValueV3::PieceKind(memory.source.stable_code()),
                    ),
                    (
                        "effective",
                        RuntimeValueV3::PieceKind(memory.effective.stable_code()),
                    ),
                ]),
            }
        })
    );
    push!(
        "nullification_piece_ids",
        piece_set(&runtime.nullification_piece_ids, ids)
    );
    push!("vanguard", bool_pair(runtime.vanguard, state.turn));
    push!("infiltration", bool_pair(runtime.infiltration, state.turn));
    push!("resolve", bool_pair(runtime.resolve, state.turn));
    push!(
        "resolve_ready",
        bool_pair(runtime.resolve_ready, state.turn)
    );
    push!(
        "resolve_move_credit",
        bool_pair(runtime.resolve_move_credit, state.turn)
    );
    push!(
        "locust_swarm_piece_ids",
        piece_set(&runtime.locust_swarm_piece_ids, ids)
    );
    push!(
        "metalized_piece_ids",
        piece_set(&runtime.metalized_piece_ids, ids)
    );
    push!(
        "metal_cooldowns",
        piece_map(&runtime.metal_cooldowns, ids, |_, remaining, _| {
            RuntimeValueV3::Unsigned(u64::from(*remaining))
        })
    );
    push!(
        "recurrence_piece_ids",
        piece_set(&runtime.recurrence_piece_ids, ids)
    );
    push!(
        "pending_recurrences",
        RuntimeValueV3::List(
            runtime
                .pending_recurrences
                .iter()
                .map(|pending| {
                    record(vec![(
                        "piece",
                        runtime_piece_value(&pending.piece, ids, state.turn),
                    )])
                })
                .collect(),
        )
    );

    assert_eq!(fields.len(), CARD_RUNTIME_SOURCE_FIELDS.len());
    fields
}

fn named(name: &str, value: RuntimeValueV3) -> NamedRuntimeValueV3 {
    NamedRuntimeValueV3 {
        name: name.to_owned(),
        value,
    }
}

fn record(fields: Vec<(&str, RuntimeValueV3)>) -> RuntimeValueV3 {
    for (name, _) in &fields {
        assert!(
            OBSERVATION_V3_REQUIRED_TEXT_TOKENS.contains(name),
            "runtime record field {name:?} is missing from OBSERVATION_V3_REQUIRED_TEXT_TOKENS"
        );
    }
    RuntimeValueV3::Record(
        fields
            .into_iter()
            .map(|(name, value)| named(name, value))
            .collect(),
    )
}

fn pair_with<T>(
    values: &[T; 2],
    perspective: Player,
    mut encode: impl FnMut(&T) -> RuntimeValueV3,
) -> RuntimeValueV3 {
    pair_with_player(values, perspective, |_, value| encode(value))
}

fn pair_with_player<T>(
    values: &[T; 2],
    perspective: Player,
    mut encode: impl FnMut(Player, &T) -> RuntimeValueV3,
) -> RuntimeValueV3 {
    let [self_index, opponent_index] = relative_indices(perspective);
    RuntimeValueV3::List(vec![
        encode(perspective, &values[self_index]),
        encode(perspective.opponent(), &values[opponent_index]),
    ])
}

fn bool_pair(values: [bool; 2], perspective: Player) -> RuntimeValueV3 {
    pair_with(&values, perspective, |value| RuntimeValueV3::Bool(*value))
}

fn u32_pair(values: [u32; 2], perspective: Player) -> RuntimeValueV3 {
    pair_with(&values, perspective, |value| {
        RuntimeValueV3::Unsigned(u64::from(*value))
    })
}

fn u8_pair(values: [u8; 2], perspective: Player) -> RuntimeValueV3 {
    pair_with(&values, perspective, |value| {
        RuntimeValueV3::Unsigned(u64::from(*value))
    })
}

fn piece_set(values: &BTreeSet<u32>, ids: &mut LocalIds) -> RuntimeValueV3 {
    let mut slots = values.iter().map(|id| ids.piece(*id)).collect::<Vec<_>>();
    slots.sort_unstable();
    RuntimeValueV3::List(slots.into_iter().map(RuntimeValueV3::Piece).collect())
}

fn card_set(values: &BTreeSet<u32>, ids: &mut LocalIds) -> RuntimeValueV3 {
    let mut slots = values.iter().map(|id| ids.card(*id)).collect::<Vec<_>>();
    slots.sort_unstable();
    RuntimeValueV3::List(slots.into_iter().map(RuntimeValueV3::Card).collect())
}

fn square_set(values: &BTreeSet<Square>, perspective: Player) -> RuntimeValueV3 {
    let mut squares = values
        .iter()
        .map(|square| square_value(*square, perspective))
        .collect::<Vec<_>>();
    squares.sort_unstable();
    RuntimeValueV3::List(squares.into_iter().map(RuntimeValueV3::Square).collect())
}

fn kind_set(values: &BTreeSet<PieceKind>) -> RuntimeValueV3 {
    RuntimeValueV3::List(
        values
            .iter()
            .map(|kind| RuntimeValueV3::PieceKind(kind.stable_code()))
            .collect(),
    )
}

fn piece_map<T>(
    values: &BTreeMap<u32, T>,
    ids: &mut LocalIds,
    mut encode: impl FnMut(u32, &T, &mut LocalIds) -> RuntimeValueV3,
) -> RuntimeValueV3 {
    let mut entries = Vec::with_capacity(values.len());
    for (piece_id, value) in values {
        let slot = ids.piece(*piece_id);
        let value = encode(*piece_id, value, ids);
        entries.push((slot, value));
    }
    entries.sort_by_key(|(slot, _)| *slot);
    RuntimeValueV3::List(
        entries
            .into_iter()
            .map(|(slot, value)| {
                record(vec![
                    ("piece", RuntimeValueV3::Piece(slot)),
                    ("value", value),
                ])
            })
            .collect(),
    )
}

fn optional_square(value: Option<Square>, perspective: Player) -> RuntimeValueV3 {
    value.map_or(RuntimeValueV3::None, |square| {
        RuntimeValueV3::Square(square_value(square, perspective))
    })
}

fn runtime_piece_value(piece: &Piece, ids: &mut LocalIds, perspective: Player) -> RuntimeValueV3 {
    record(vec![
        ("piece", RuntimeValueV3::Piece(ids.piece(piece.id))),
        (
            "owner",
            RuntimeValueV3::Player(relative_player(piece.owner, perspective)),
        ),
        (
            "piece_kind",
            RuntimeValueV3::PieceKind(piece.kind.stable_code()),
        ),
        (
            "count",
            RuntimeValueV3::Unsigned(u64::from(piece.capture_count)),
        ),
    ])
}

fn optional_runtime_piece_value(
    piece: Option<&Piece>,
    ids: &mut LocalIds,
    perspective: Player,
) -> RuntimeValueV3 {
    piece.map_or(RuntimeValueV3::None, |piece| {
        runtime_piece_value(piece, ids, perspective)
    })
}

fn optional_en_passant_value(
    en_passant: Option<&EnPassantState>,
    perspective: Player,
) -> RuntimeValueV3 {
    en_passant.map_or(RuntimeValueV3::None, |en_passant| {
        record(vec![
            (
                "to",
                RuntimeValueV3::Square(square_value(en_passant.target, perspective)),
            ),
            (
                "from",
                RuntimeValueV3::Square(square_value(en_passant.captured, perspective)),
            ),
            (
                "owner",
                RuntimeValueV3::Player(relative_player(en_passant.pawn_owner, perspective)),
            ),
        ])
    })
}

fn draft_phase_name(phase: DraftPhase) -> &'static str {
    match phase {
        DraftPhase::Opening => "opening",
        DraftPhase::Middle => "middle",
        DraftPhase::End => "end",
    }
}

fn optional_square_pair(value: Option<[Square; 2]>, perspective: Player) -> RuntimeValueV3 {
    value.map_or(RuntimeValueV3::None, |squares| {
        RuntimeValueV3::List(
            squares
                .into_iter()
                .map(|square| RuntimeValueV3::Square(square_value(square, perspective)))
                .collect(),
        )
    })
}

fn player_square_map(
    values: &BTreeMap<Player, [Square; 2]>,
    perspective: Player,
) -> RuntimeValueV3 {
    RuntimeValueV3::List(
        [perspective, perspective.opponent()]
            .into_iter()
            .filter_map(|player| {
                values.get(&player).map(|squares| {
                    record(vec![
                        (
                            "player",
                            RuntimeValueV3::Player(relative_player(player, perspective)),
                        ),
                        (
                            "squares",
                            RuntimeValueV3::List(
                                squares
                                    .iter()
                                    .map(|square| {
                                        RuntimeValueV3::Square(square_value(*square, perspective))
                                    })
                                    .collect(),
                            ),
                        ),
                    ])
                })
            })
            .collect(),
    )
}

fn piece_owner(state: &GameState, piece_id: u32) -> Player {
    state
        .board
        .find_by_id(piece_id)
        .map_or(state.turn, |(_, piece)| piece.owner)
}

fn remaining_piece_turn(state: &GameState, piece_id: u32, deadline: u32) -> u32 {
    let owner = piece_owner(state, piece_id);
    deadline.saturating_sub(state.turns_taken[owner.index()])
}

fn initiative_value(status: &InitiativeStatus, state: &GameState) -> RuntimeValueV3 {
    record(vec![
        (
            "by",
            RuntimeValueV3::Player(relative_player(status.by, state.turn)),
        ),
        (
            "elapsed_turns",
            RuntimeValueV3::Unsigned(u64::from(
                state.turns_taken[status.by.index()].saturating_sub(status.start_turn),
            )),
        ),
        ("limit", RuntimeValueV3::Unsigned(u64::from(status.limit))),
    ])
}

fn dice_lock_value(status: &DiceLockStatus) -> RuntimeValueV3 {
    record(vec![
        (
            "piece_kind",
            RuntimeValueV3::PieceKind(status.kind.stable_code()),
        ),
        (
            "remaining_turns",
            RuntimeValueV3::Unsigned(u64::from(status.remaining_turns)),
        ),
    ])
}

fn frozen_value(status: &FrozenStatus, perspective: Player) -> RuntimeValueV3 {
    record(vec![
        (
            "source",
            RuntimeValueV3::Player(relative_player(status.source, perspective)),
        ),
        (
            "remaining_turns",
            RuntimeValueV3::Unsigned(u64::from(status.remaining_turns)),
        ),
    ])
}

fn reposition_value(status: &RepositionStatus, ids: &mut LocalIds) -> RuntimeValueV3 {
    record(vec![
        ("piece", RuntimeValueV3::Piece(ids.piece(status.piece_id))),
        ("used", RuntimeValueV3::Bool(status.used)),
    ])
}

fn lunchbox_value(
    piece_id: u32,
    status: &EmptyLunchboxStatus,
    state: &GameState,
) -> RuntimeValueV3 {
    record(vec![
        (
            "by",
            RuntimeValueV3::Player(relative_player(status.by, state.turn)),
        ),
        (
            "remaining_turns",
            RuntimeValueV3::Unsigned(u64::from(remaining_piece_turn(
                state,
                piece_id,
                status.deadline_turn,
            ))),
        ),
    ])
}

fn platform_value(
    status: &PlatformRuleState,
    state: &GameState,
    ids: &mut LocalIds,
) -> RuntimeValueV3 {
    record(vec![
        ("cell", optional_square(status.cell, state.turn)),
        (
            "age_shared_turns",
            RuntimeValueV3::Unsigned(u64::from(
                state.shared_turn_count().saturating_sub(status.spawned_at),
            )),
        ),
        (
            "next_in_shared_turns",
            RuntimeValueV3::Unsigned(u64::from(
                status.next_at.saturating_sub(state.shared_turn_count()),
            )),
        ),
        (
            "triggered_pieces",
            piece_set(&status.triggered_piece_ids, ids),
        ),
        ("nonce", RuntimeValueV3::Unsigned(u64::from(status.nonce))),
    ])
}

fn log_status_value(piece_id: u32, status: &LogStatus, state: &GameState) -> RuntimeValueV3 {
    record(vec![
        (
            "file_delta",
            RuntimeValueV3::Signed(i64::from(status.file_delta)),
        ),
        (
            "rank_delta",
            RuntimeValueV3::Signed(i64::from(status.rank_delta)),
        ),
        (
            "roll_in_owner_turns",
            RuntimeValueV3::Unsigned(u64::from(remaining_piece_turn(
                state,
                piece_id,
                status.roll_after_turn,
            ))),
        ),
    ])
}

fn hazard_value(status: &DelayedHazard, perspective: Player) -> RuntimeValueV3 {
    record(vec![
        (
            "kind",
            RuntimeValueV3::Text(
                match status.kind {
                    DelayedHazardKind::Lightning => "lightning",
                    DelayedHazardKind::Meteor => "meteor",
                }
                .to_owned(),
            ),
        ),
        (
            "owner",
            RuntimeValueV3::Player(relative_player(status.owner, perspective)),
        ),
        (
            "trigger_after",
            RuntimeValueV3::Player(relative_player(status.trigger_after, perspective)),
        ),
        (
            "squares",
            RuntimeValueV3::List(
                status
                    .squares
                    .iter()
                    .map(|square| RuntimeValueV3::Square(square_value(*square, perspective)))
                    .collect(),
            ),
        ),
    ])
}

fn otherworld_value(
    status: &PendingOtherworld,
    state: &GameState,
    ids: &mut LocalIds,
) -> RuntimeValueV3 {
    record(vec![
        ("piece", RuntimeValueV3::Piece(ids.piece(status.piece_id))),
        (
            "owner",
            RuntimeValueV3::Player(relative_player(status.owner, state.turn)),
        ),
        (
            "square",
            RuntimeValueV3::Square(square_value(status.square, state.turn)),
        ),
        (
            "origin",
            RuntimeValueV3::Square(square_value(status.origin, state.turn)),
        ),
        (
            "remaining_ply",
            RuntimeValueV3::Unsigned(u64::from(status.due_ply.saturating_sub(state.ply))),
        ),
    ])
}

fn large_piece_value(status: &LargePieceStatus, perspective: Player) -> RuntimeValueV3 {
    record(vec![
        (
            "anchor",
            RuntimeValueV3::Square(square_value(status.anchor, perspective)),
        ),
        ("hp", RuntimeValueV3::Unsigned(u64::from(status.hp))),
        ("max_hp", RuntimeValueV3::Unsigned(u64::from(status.max_hp))),
    ])
}

fn timed_piece_value(status: &TimedPieceStatus, perspective: Player) -> RuntimeValueV3 {
    record(vec![
        (
            "by",
            RuntimeValueV3::Player(relative_player(status.by, perspective)),
        ),
        (
            "remaining_turns",
            RuntimeValueV3::Unsigned(u64::from(status.remaining_turns)),
        ),
    ])
}

fn royal_command_value(
    status: &RoyalCommandStatus,
    state: &GameState,
    clock_owner: Player,
) -> RuntimeValueV3 {
    let clock = state.turns_taken[clock_owner.index()];
    record(vec![
        (
            "starts_in_turns",
            RuntimeValueV3::Unsigned(u64::from(status.starts_at_turn.saturating_sub(clock))),
        ),
        (
            "expires_in_turns",
            RuntimeValueV3::Unsigned(u64::from(status.expires_at_turn.saturating_sub(clock))),
        ),
    ])
}

fn feudal_value(
    status: &FeudalContract,
    perspective: Player,
    ids: &mut LocalIds,
) -> RuntimeValueV3 {
    record(vec![
        (
            "owner",
            RuntimeValueV3::Player(relative_player(status.owner, perspective)),
        ),
        ("pawn", RuntimeValueV3::Piece(ids.piece(status.pawn_id))),
        (
            "guardian",
            RuntimeValueV3::Piece(ids.piece(status.guardian_id)),
        ),
    ])
}

fn panic_value(status: &PendingPanic, perspective: Player, ids: &mut LocalIds) -> RuntimeValueV3 {
    record(vec![
        (
            "owner",
            RuntimeValueV3::Player(relative_player(status.owner, perspective)),
        ),
        (
            "pieces",
            RuntimeValueV3::List(
                status
                    .piece_ids
                    .iter()
                    .map(|piece_id| RuntimeValueV3::Piece(ids.piece(*piece_id)))
                    .collect(),
            ),
        ),
    ])
}

fn premove_value(status: &PendingPremove, state: &GameState) -> RuntimeValueV3 {
    record(vec![
        (
            "owner",
            RuntimeValueV3::Player(relative_player(status.owner, state.turn)),
        ),
        (
            "trigger_player",
            RuntimeValueV3::Player(relative_player(status.trigger_player, state.turn)),
        ),
        (
            "trigger_in_turns",
            RuntimeValueV3::Unsigned(u64::from(
                status
                    .trigger_turn
                    .saturating_sub(state.turns_taken[status.trigger_player.index()]),
            )),
        ),
        (
            "moves",
            RuntimeValueV3::List(
                status
                    .moves
                    .iter()
                    .map(|movement| move_value(movement, state.turn))
                    .collect(),
            ),
        ),
    ])
}

fn move_value(movement: &MoveAction, perspective: Player) -> RuntimeValueV3 {
    record(vec![
        (
            "from",
            RuntimeValueV3::Square(square_value(movement.from, perspective)),
        ),
        (
            "to",
            RuntimeValueV3::Square(square_value(movement.to, perspective)),
        ),
        (
            "promotion",
            movement.promotion.map_or(RuntimeValueV3::None, |kind| {
                RuntimeValueV3::PieceKind(kind.stable_code())
            }),
        ),
    ])
}

fn scarecrow_value(status: &PendingScarecrow, perspective: Player) -> RuntimeValueV3 {
    record(vec![
        (
            "owner",
            RuntimeValueV3::Player(relative_player(status.owner, perspective)),
        ),
        (
            "square",
            RuntimeValueV3::Square(square_value(status.square, perspective)),
        ),
        (
            "remaining_ply",
            RuntimeValueV3::Unsigned(u64::from(status.remaining_ply)),
        ),
    ])
}

fn twin_value(status: &TwinBond, ids: &mut LocalIds) -> RuntimeValueV3 {
    record(vec![
        ("first", RuntimeValueV3::Piece(ids.piece(status.first_id))),
        ("second", RuntimeValueV3::Piece(ids.piece(status.second_id))),
    ])
}

fn trolley_value(
    status: &TrolleyDilemma,
    perspective: Player,
    ids: &mut LocalIds,
) -> RuntimeValueV3 {
    record(vec![
        (
            "by",
            RuntimeValueV3::Player(relative_player(status.by, perspective)),
        ),
        (
            "bundles",
            RuntimeValueV3::List(
                status
                    .bundles
                    .iter()
                    .map(|bundle| {
                        RuntimeValueV3::List(
                            bundle
                                .iter()
                                .map(|piece_id| RuntimeValueV3::Piece(ids.piece(*piece_id)))
                                .collect(),
                        )
                    })
                    .collect(),
            ),
        ),
    ])
}

fn bribe_value(status: &BribeRevert, state: &GameState, ids: &mut LocalIds) -> RuntimeValueV3 {
    record(vec![
        ("piece", RuntimeValueV3::Piece(ids.piece(status.piece_id))),
        (
            "remaining_turns",
            RuntimeValueV3::Unsigned(u64::from(remaining_piece_turn(
                state,
                status.piece_id,
                status.due_turn,
            ))),
        ),
    ])
}

fn necromancy_value(
    status: &NecromancyRevert,
    state: &GameState,
    ids: &mut LocalIds,
) -> RuntimeValueV3 {
    record(vec![
        ("piece", RuntimeValueV3::Piece(ids.piece(status.piece_id))),
        (
            "remaining_ply",
            RuntimeValueV3::Unsigned(u64::from(status.due_ply.saturating_sub(state.ply))),
        ),
    ])
}

fn vip_value(status: &PendingVip, state: &GameState, ids: &mut LocalIds) -> RuntimeValueV3 {
    record(vec![
        ("piece", RuntimeValueV3::Piece(ids.piece(status.piece_id))),
        (
            "owner",
            RuntimeValueV3::Player(relative_player(status.owner, state.turn)),
        ),
        (
            "remaining_ply",
            RuntimeValueV3::Unsigned(u64::from(status.due_ply.saturating_sub(state.ply))),
        ),
    ])
}

fn icbm_value(status: &PendingIcbm, state: &GameState, ids: &mut LocalIds) -> RuntimeValueV3 {
    record(vec![
        ("queen", RuntimeValueV3::Piece(ids.piece(status.queen_id))),
        (
            "owner",
            RuntimeValueV3::Player(relative_player(status.owner, state.turn)),
        ),
        (
            "remaining_turns",
            RuntimeValueV3::Unsigned(u64::from(
                status
                    .due_turn
                    .saturating_sub(state.turns_taken[status.owner.index()]),
            )),
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation_v3::PIECE_KIND_VOCABULARY;
    use augment_core::{
        BUNDLED_RULESET_ID, Board, CardInstance, Engine, IMPLEMENTED_CARDS, PendingChoice, Piece,
    };

    fn playing_state(board: Board, turn: Player, seed: u64) -> GameState {
        let mut state =
            GameState::from_board(BUNDLED_RULESET_ID, Mode::Standard, board, turn, seed);
        state.phase = GamePhase::Playing;
        state
    }

    #[test]
    fn proficiency_is_public_canonical_and_distinguishes_both_owners() {
        let mut signatures = BTreeSet::new();
        for flags in [[false, false], [true, false], [false, true], [true, true]] {
            let mut white = september_state(Player::White, 3, 100);
            white.card_runtime.proficiency = flags;
            let mut black = september_state(Player::Black, 900, 700);
            black.card_runtime.proficiency = [flags[1], flags[0]];
            assert_eq!(public_bytes(&white), public_bytes(&black));
            signatures.insert(public_bytes(&white));
        }
        assert_eq!(
            signatures.len(),
            4,
            "both public owner flags must survive encoding"
        );
    }

    #[test]
    fn killer_king_is_public_canonical_and_distinguishes_both_owners() {
        let mut signatures = BTreeSet::new();
        for flags in [[false, false], [true, false], [false, true], [true, true]] {
            let mut white = september_state(Player::White, 3, 100);
            white.card_runtime.killer_king = flags;
            let mut black = september_state(Player::Black, 900, 700);
            black.card_runtime.killer_king = [flags[1], flags[0]];
            assert_eq!(public_bytes(&white), public_bytes(&black));
            signatures.insert(public_bytes(&white));
        }
        assert_eq!(
            signatures.len(),
            4,
            "both public owner flags must survive encoding"
        );
    }

    #[test]
    fn disassembly_field_176_survives_observation_json_roundtrip() {
        let mut state = september_state(Player::Black, 81, 200);
        state.card_runtime.disassembly = [true, false];
        let observation = encode_observation_v3(&state);
        assert_eq!(observation.runtime.len(), 194);
        assert_eq!(observation.runtime[175].field_id, 176);
        assert_eq!(observation.runtime[175].field_name(), Some("disassembly"));
        assert_eq!(
            observation.runtime[175].value,
            RuntimeValueV3::List(vec![
                RuntimeValueV3::Bool(false),
                RuntimeValueV3::Bool(true),
            ])
        );
        let restored: ObservationV3 =
            serde_json::from_slice(&serde_json::to_vec(&observation).unwrap()).unwrap();
        assert_eq!(observation, restored);
    }

    #[test]
    fn overtake_field_177_survives_observation_json_roundtrip() {
        let mut state = september_state(Player::White, 91, 300);
        state.card_runtime.overtake = [false, true];
        let observation = encode_observation_v3(&state);
        assert_eq!(observation.runtime.len(), 194);
        assert_eq!(observation.runtime[176].field_id, 177);
        assert_eq!(observation.runtime[176].field_name(), Some("overtake"));
        assert_eq!(
            observation.runtime[176].value,
            RuntimeValueV3::List(vec![
                RuntimeValueV3::Bool(false),
                RuntimeValueV3::Bool(true),
            ])
        );
        let restored: ObservationV3 =
            serde_json::from_slice(&serde_json::to_vec(&observation).unwrap()).unwrap();
        assert_eq!(observation, restored);
    }

    #[test]
    fn disassembly_is_public_canonical_and_distinguishes_both_owners() {
        let mut signatures = BTreeSet::new();
        for flags in [[false, false], [true, false], [false, true], [true, true]] {
            let mut white = september_state(Player::White, 3, 100);
            white.card_runtime.disassembly = flags;
            let mut black = september_state(Player::Black, 900, 700);
            black.card_runtime.disassembly = [flags[1], flags[0]];
            assert_eq!(public_bytes(&white), public_bytes(&black));
            signatures.insert(public_bytes(&white));
        }
        assert_eq!(
            signatures.len(),
            4,
            "both public owner flags must survive encoding"
        );
    }

    #[test]
    fn overtake_is_public_canonical_and_distinguishes_both_owners() {
        let mut signatures = BTreeSet::new();
        for flags in [[false, false], [true, false], [false, true], [true, true]] {
            let mut white = september_state(Player::White, 3, 100);
            white.card_runtime.overtake = flags;
            let mut black = september_state(Player::Black, 900, 700);
            black.card_runtime.overtake = [flags[1], flags[0]];
            assert_eq!(public_bytes(&white), public_bytes(&black));
            signatures.insert(public_bytes(&white));
        }
        assert_eq!(
            signatures.len(),
            4,
            "both public owner flags must survive encoding"
        );
    }

    #[test]
    fn latest_public_flags_outpost_and_clockwork_are_canonical() {
        for flag in 0..4 {
            let mut signatures = BTreeSet::new();
            for values in [[false, false], [false, true], [true, false], [true, true]] {
                let mut white = september_state(Player::White, 1, 100);
                let mut black = september_state(Player::Black, 999, 800);
                let reversed = [values[1], values[0]];
                match flag {
                    0 => {
                        white.card_runtime.reversal = values;
                        black.card_runtime.reversal = reversed;
                    }
                    1 => {
                        white.card_runtime.symmetry = values;
                        black.card_runtime.symmetry = reversed;
                    }
                    2 => {
                        white.card_runtime.mutation = values;
                        black.card_runtime.mutation = reversed;
                    }
                    _ => {
                        white.card_runtime.long_en_passant = values;
                        black.card_runtime.long_en_passant = reversed;
                    }
                }
                assert_eq!(public_bytes(&white), public_bytes(&black));
                signatures.insert(public_bytes(&white));
            }
            assert_eq!(signatures.len(), 4, "flag {flag}");
        }
        let mut signatures = BTreeSet::new();
        for protected in [vec![], vec![0], vec![1], vec![0, 1]] {
            let mut white = september_state(Player::White, 1, 100);
            let mut black = september_state(Player::Black, 999, 800);
            for slot in protected {
                white
                    .card_runtime
                    .outpost_protected_piece_ids
                    .insert([110, 120][slot]);
                black
                    .card_runtime
                    .outpost_protected_piece_ids
                    .insert([810, 820][slot]);
            }
            assert_eq!(public_bytes(&white), public_bytes(&black));
            signatures.insert(public_bytes(&white));
        }
        assert_eq!(signatures.len(), 4);
        // Reversing raw IDs must not change the sorted canonical set of board references.
        let mut white = september_state(Player::White, 1, 100);
        white
            .card_runtime
            .outpost_protected_piece_ids
            .extend([110, 120]);
        let mut permuted = white.clone();
        for (square, mut piece) in white.board.iter() {
            piece.id = if piece.id == 110 { 990 } else { 2 };
            permuted.board.set(square, Some(piece));
        }
        // Other public runtime references in this fixture must follow the same rename.
        let flags = permuted.card_runtime.capture_the_flag.as_mut().unwrap();
        for occupation in flags.occupations.iter_mut().flatten() {
            occupation.piece_id = if occupation.piece_id == 110 { 990 } else { 2 };
        }
        permuted.card_runtime.outpost_protected_piece_ids = [990, 2].into_iter().collect();
        permuted.card_runtime.piece_origins = white
            .card_runtime
            .piece_origins
            .iter()
            .map(|(id, square)| (if *id == 110 { 990 } else { 2 }, *square))
            .collect();
        assert_eq!(public_bytes(&white), public_bytes(&permuted));
        for owner in [Player::White, Player::Black] {
            let mut white_board = Board::empty();
            let mut black_board = Board::empty();
            white_board.set(
                "c4".parse().unwrap(),
                Some(Piece::new(owner, PieceKind::Clockwork).with_id(3)),
            );
            black_board.set(
                "c5".parse().unwrap(),
                Some(Piece::new(owner.opponent(), PieceKind::Clockwork).with_id(900)),
            );
            assert_eq!(
                public_bytes(&playing_state(white_board, Player::White, 7)),
                public_bytes(&playing_state(black_board, Player::Black, 800))
            );
        }
    }

    #[test]
    fn clockwork_planes_append_without_shifting_existing_opponent_kinds() {
        for owner in [Player::White, Player::Black] {
            for (code, kind) in PIECE_KIND_VOCABULARY {
                let mut board = Board::empty();
                board.set(
                    "d4".parse().unwrap(),
                    Some(Piece::new(owner, kind).with_id(90)),
                );
                let observation = encode_observation_v3(&playing_state(board, Player::White, 7));
                let plane = if code >= 59 {
                    116 + 2 * usize::from(code - 59) + owner.index()
                } else {
                    owner.index() * 58 + usize::from(code - 1)
                };
                assert_eq!(
                    observation.spatial[plane * 64 + 27],
                    1.0,
                    "{owner:?} {kind:?}"
                );
                assert_eq!(
                    observation
                        .spatial
                        .iter()
                        .filter(|value| **value != 0.0)
                        .count(),
                    1
                );
            }
        }
    }

    #[test]
    fn parrot_memory_uses_piece_kinds_not_piece_refs_and_roundtrips() {
        let mut state = september_state(Player::Black, 73, 100);
        state.card_runtime.parrot_memory = [
            None,
            Some(augment_core::ParrotMemoryStatus {
                source: PieceKind::Trickster,
                effective: PieceKind::Campfire,
            }),
        ];
        let observation = encode_observation_v3(&state);
        assert_eq!(observation.runtime[182].field_id, 183);
        assert_eq!(observation.runtime[182].field_name(), Some("parrot_memory"));
        assert_eq!(
            observation.runtime[182].value,
            RuntimeValueV3::List(vec![
                record(vec![
                    (
                        "source",
                        RuntimeValueV3::PieceKind(PieceKind::Trickster.stable_code())
                    ),
                    ("effective", RuntimeValueV3::PieceKind(65)),
                ]),
                RuntimeValueV3::None,
            ])
        );
        assert_eq!(observation.piece_slot_count, 2);
        let restored: ObservationV3 =
            serde_json::from_slice(&serde_json::to_vec(&observation).unwrap()).unwrap();
        assert_eq!(restored, observation);
        assert!(observation_v3_runtime_text_tokens().contains(&"effective"));
    }

    #[test]
    fn parrot_memory_public_source_effective_and_owner_do_not_collapse() {
        let mut signatures = BTreeSet::new();
        let choices = [
            None,
            Some(augment_core::ParrotMemoryStatus {
                source: PieceKind::Rook,
                effective: PieceKind::Bishop,
            }),
            Some(augment_core::ParrotMemoryStatus {
                source: PieceKind::Queen,
                effective: PieceKind::Bishop,
            }),
            Some(augment_core::ParrotMemoryStatus {
                source: PieceKind::Rook,
                effective: PieceKind::Knight,
            }),
        ];
        for own in choices {
            for opponent in choices {
                let mut white = september_state(Player::White, 1, 100);
                let mut black = september_state(Player::Black, 999, 800);
                white.card_runtime.parrot_memory = [own, opponent];
                black.card_runtime.parrot_memory = [opponent, own];
                assert_eq!(public_bytes(&white), public_bytes(&black));
                let observation = encode_observation_v3(&white);
                assert_eq!(
                    observation.piece_slot_count, 2,
                    "owner memory must not create piece refs"
                );
                signatures.insert(public_bytes(&white));
            }
        }
        assert_eq!(
            signatures.len(),
            16,
            "every owner/source/effective/presence must survive"
        );
    }

    #[test]
    fn six_new_piece_kinds_have_distinct_appended_planes_and_canonical_json() {
        let kinds = [
            PieceKind::Parrot,
            PieceKind::Paladin,
            PieceKind::Princess,
            PieceKind::Octopus,
            PieceKind::Hedgehog,
            PieceKind::Campfire,
        ];
        let mut seen_planes = BTreeSet::new();
        for (index, kind) in kinds.into_iter().enumerate() {
            assert_eq!(kind.stable_code(), 60 + index as u8);
            for owner in [Player::White, Player::Black] {
                let mut board = Board::empty();
                board.set(
                    "c4".parse().unwrap(),
                    Some(Piece::new(owner, kind).with_id(91)),
                );
                let mut mirrored = Board::empty();
                mirrored.set(
                    "c5".parse().unwrap(),
                    Some(Piece::new(owner.opponent(), kind).with_id(701)),
                );
                let observation = encode_observation_v3(&playing_state(board, Player::White, 1));
                assert_eq!(
                    observation,
                    encode_observation_v3(&playing_state(mirrored, Player::Black, 999))
                );
                assert_eq!(observation.spatial_shape, [136, 8, 8]);
                let plane = 118 + index * 2 + owner.index();
                seen_planes.insert(plane);
                assert_eq!(observation.spatial[plane * 64 + 26], 1.0);
                assert_eq!(
                    observation
                        .spatial
                        .iter()
                        .filter(|value| **value != 0.0)
                        .count(),
                    1
                );
                assert!(
                    observation.spatial[..118 * 64]
                        .iter()
                        .all(|value| *value == 0.0)
                );
                assert_eq!(observation.pieces[0].kind_code, 60 + index as u8);
                let restored: ObservationV3 =
                    serde_json::from_slice(&serde_json::to_vec(&observation).unwrap()).unwrap();
                assert_eq!(restored, observation);
            }
        }
        assert_eq!(seen_planes, (118..130).collect());
    }

    #[test]
    fn latest_public_fields_and_clockwork_survive_roundtrip() {
        let mut state = september_state(Player::White, 1, 100);
        let original = public_bytes(&state);
        for flag in 0..4 {
            let mut changed = state.clone();
            match flag {
                0 => changed.card_runtime.reversal[0] = true,
                1 => changed.card_runtime.symmetry[0] = true,
                2 => changed.card_runtime.mutation[0] = true,
                _ => changed.card_runtime.long_en_passant[0] = true,
            }
            assert_ne!(original, public_bytes(&changed), "flag {flag}");
        }
        state.card_runtime.outpost_protected_piece_ids.insert(110);
        assert_ne!(original, public_bytes(&state));
        state.board.set(
            "c4".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Clockwork).with_id(300)),
        );
        state.board.set(
            "f5".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::Clockwork).with_id(400)),
        );
        let observation = encode_observation_v3(&state);
        assert_eq!(observation.spatial_shape, [136, 8, 8]);
        assert_eq!(observation.spatial[116 * 64 + 26], 1.0);
        assert_eq!(observation.spatial[117 * 64 + 37], 1.0);
        assert_eq!(observation.runtime.len(), 194);
        let restored: ObservationV3 =
            serde_json::from_slice(&serde_json::to_vec(&observation).unwrap()).unwrap();
        assert_eq!(restored, observation);
    }

    #[test]
    fn majesty_is_public_canonical_and_distinguishes_both_owners() {
        let mut signatures = BTreeSet::new();
        for flags in [[false, false], [true, false], [false, true], [true, true]] {
            let mut white = september_state(Player::White, 3, 100);
            white.card_runtime.majesty = flags;
            let mut black = september_state(Player::Black, 900, 700);
            black.card_runtime.majesty = [flags[1], flags[0]];
            assert_eq!(public_bytes(&white), public_bytes(&black));
            signatures.insert(public_bytes(&white));
        }
        assert_eq!(
            signatures.len(),
            4,
            "both public owner flags must survive encoding"
        );
    }

    fn september_state(perspective: Player, seed: u64, id_offset: u32) -> GameState {
        let square = |value: &str| canonical_square(value.parse().unwrap(), perspective);
        let mut board = Board::empty();
        board.set(
            square("b2"),
            Some(Piece::new(perspective, PieceKind::Pawn).with_id(id_offset + 10)),
        );
        board.set(
            square("g7"),
            Some(Piece::new(perspective.opponent(), PieceKind::Rook).with_id(id_offset + 20)),
        );
        let mut state = playing_state(board, perspective, seed);
        state.ply = 23;
        state.turns_taken[perspective.index()] = 11;
        state.turns_taken[perspective.opponent().index()] = 8;
        let mut flags = augment_core::CaptureTheFlagState {
            flags: [square("c1"); 2],
            occupations: [None, None],
        };
        flags.flags[perspective.opponent().index()] = square("f8");
        flags.occupations[perspective.index()] = Some(augment_core::FlagOccupation {
            piece_id: id_offset + 20,
            after_owner_turn: 11,
        });
        flags.occupations[perspective.opponent().index()] = Some(augment_core::FlagOccupation {
            piece_id: id_offset + 10,
            after_owner_turn: 9,
        });
        state.card_runtime.capture_the_flag = Some(flags);
        for (owner, cell, due_ply) in [(perspective, "c4", 25), (perspective.opponent(), "e5", 22)]
        {
            state
                .card_runtime
                .pending_lobsters
                .push(augment_core::PendingLobster {
                    owner,
                    square: square(cell),
                    due_ply,
                });
        }
        for (owner, cells, due_turn) in [
            (perspective, ["a3", "h4"], 12),
            (perspective.opponent(), ["b4", "g5"], 7),
        ] {
            state
                .card_runtime
                .pending_portals
                .push(augment_core::PendingPortal {
                    owner,
                    squares: cells.map(square),
                    due_turn,
                });
        }
        state
    }

    fn public_bytes(state: &GameState) -> Vec<u8> {
        serde_json::to_vec(&encode_observation_v3(state)).unwrap()
    }

    #[test]
    fn september_public_fields_each_change_serialized_observation() {
        let state = september_state(Player::White, 3, 100);
        let original = public_bytes(&state);
        // Presence, both flag squares, both occupation presences/references/timers.
        for change in 0..11 {
            let mut changed = state.clone();
            if change == 0 {
                changed.card_runtime.capture_the_flag = None;
            } else {
                let flags = changed.card_runtime.capture_the_flag.as_mut().unwrap();
                let owner = (change - 1) / 5;
                match (change - 1) % 5 {
                    0 => flags.flags[owner] = "d3".parse().unwrap(),
                    1 => flags.occupations[owner] = None,
                    2 => {
                        flags.occupations[owner].as_mut().unwrap().piece_id =
                            if owner == 0 { 110 } else { 120 }
                    }
                    3 => flags.occupations[owner].as_mut().unwrap().after_owner_turn += 1,
                    _ => flags.occupations[owner].as_mut().unwrap().after_owner_turn -= 1,
                }
            }
            assert_ne!(original, public_bytes(&changed), "CTF change {change}");
        }
        // Queue presence, owner, each square, and timer are independent public facts.
        for index in 0..2 {
            for change in 0..5 {
                let mut changed = state.clone();
                if change == 0 {
                    changed.card_runtime.pending_portals.remove(index);
                } else {
                    let pending = &mut changed.card_runtime.pending_portals[index];
                    match change {
                        1 => pending.owner = pending.owner.opponent(),
                        2 => pending.squares[0] = "d3".parse().unwrap(),
                        3 => pending.squares[1] = "d3".parse().unwrap(),
                        _ => pending.due_turn += 1,
                    }
                }
                assert_ne!(
                    original,
                    public_bytes(&changed),
                    "portal {index} change {change}"
                );
            }
            for change in 0..4 {
                let mut changed = state.clone();
                if change == 0 {
                    changed.card_runtime.pending_lobsters.remove(index);
                } else {
                    let pending = &mut changed.card_runtime.pending_lobsters[index];
                    match change {
                        1 => pending.owner = pending.owner.opponent(),
                        2 => pending.square = "d3".parse().unwrap(),
                        _ => pending.due_ply += 1,
                    }
                }
                assert_ne!(
                    original,
                    public_bytes(&changed),
                    "lobster {index} change {change}"
                );
            }
        }
        let mut reordered = state.clone();
        reordered.card_runtime.pending_portals.reverse();
        assert_ne!(original, public_bytes(&reordered));
        let mut reordered = state.clone();
        reordered.card_runtime.pending_lobsters.reverse();
        assert_ne!(original, public_bytes(&reordered));
    }

    #[test]
    fn september_public_state_is_canonical_without_rng_or_raw_ids() {
        let white = september_state(Player::White, 10, 9000);
        let other_rng = september_state(Player::White, 900, 9000);
        let black = september_state(Player::Black, 999, 70);
        assert_ne!(white.fingerprint(), other_rng.fingerprint());
        assert_eq!(public_bytes(&white), public_bytes(&other_rng));
        assert_eq!(public_bytes(&white), public_bytes(&black));
        let encoded = encode_observation_v3(&white);
        assert_eq!(encoded.piece_slot_count, 2);
        assert_eq!(
            encoded.runtime[169].value,
            RuntimeValueV3::List(vec![
                record(vec![
                    (
                        "owner",
                        RuntimeValueV3::Player(RelativePlayerV3::SelfPlayer)
                    ),
                    ("square", RuntimeValueV3::Square(2)),
                    (
                        "occupation",
                        record(vec![
                            ("piece", RuntimeValueV3::Piece(1)),
                            ("trigger_in_turns", RuntimeValueV3::Signed(0)),
                        ])
                    ),
                ]),
                record(vec![
                    ("owner", RuntimeValueV3::Player(RelativePlayerV3::Opponent)),
                    ("square", RuntimeValueV3::Square(61)),
                    (
                        "occupation",
                        record(vec![
                            ("piece", RuntimeValueV3::Piece(0)),
                            ("trigger_in_turns", RuntimeValueV3::Signed(1)),
                        ])
                    ),
                ]),
            ])
        );
    }

    fn runtime_named<'a>(value: &'a RuntimeValueV3, name: &str) -> &'a RuntimeValueV3 {
        let RuntimeValueV3::Record(fields) = value else {
            panic!("expected record")
        };
        &fields
            .iter()
            .find(|field| field.name == name)
            .unwrap()
            .value
    }

    #[test]
    fn september_timers_use_correct_clock_and_signed_widened_subtraction() {
        // Both CTF strict-greater and portal/lobster greater-or-equal boundaries
        // retain -1/0/+1. Large counts must not underflow or wrap through i32.
        for (now, due, expected) in [
            (11, 10, -1),
            (11, 11, 0),
            (11, 12, 1),
            (u32::MAX, 0, -i64::from(u32::MAX)),
            (0, u32::MAX, i64::from(u32::MAX)),
        ] {
            let mut state = september_state(Player::Black, 30, 1);
            state.turns_taken[Player::Black.index()] = now;
            state.ply = now;
            state
                .card_runtime
                .capture_the_flag
                .as_mut()
                .unwrap()
                .occupations[Player::Black.index()]
            .as_mut()
            .unwrap()
            .after_owner_turn = due;
            state.card_runtime.pending_lobsters[0].due_ply = due;
            state.card_runtime.pending_portals[0].due_turn = due;
            let observation = encode_observation_v3(&state);
            for (field, timer, nested) in [
                (169, "trigger_in_turns", true),
                (170, "remaining_ply", false),
                (171, "trigger_in_turns", false),
            ] {
                let RuntimeValueV3::List(items) = &observation.runtime[field].value else {
                    panic!("list")
                };
                let item = if nested {
                    runtime_named(&items[0], "occupation")
                } else {
                    &items[0]
                };
                assert_eq!(
                    *runtime_named(item, timer),
                    RuntimeValueV3::Signed(expected)
                );
            }
        }
    }

    #[test]
    fn public_pending_lobster_fields_change_observation() {
        let base = playing_state(Board::empty(), Player::White, 7);
        let mut state = base.clone();
        state
            .card_runtime
            .pending_lobsters
            .push(augment_core::PendingLobster {
                owner: Player::White,
                square: "c4".parse().unwrap(),
                due_ply: 2,
            });
        assert_ne!(encode_observation_v3(&base), encode_observation_v3(&state));
        let original = encode_observation_v3(&state);
        for change in 0..3 {
            let mut changed = state.clone();
            let pending = &mut changed.card_runtime.pending_lobsters[0];
            match change {
                0 => pending.owner = Player::Black,
                1 => pending.square = "d4".parse().unwrap(),
                _ => pending.due_ply += 1,
            }
            assert_ne!(original, encode_observation_v3(&changed), "field {change}");
        }
    }

    #[test]
    fn dynamic_choice_labels_are_not_observation_text_tokens() {
        let mut state = GameState::standard(BUNDLED_RULESET_ID, Mode::Standard, 77);
        state.phase = GamePhase::AwaitingChoice(PendingChoice {
            id: 1,
            owner: Player::White,
            source_card_instance: 0,
            handler_id: "middle.trolley".to_owned(),
            options: vec!["희생: Pawn + Knight".to_owned(), "희생: Bishop".to_owned()],
            continuation: ChoiceContinuation::ResumeTurn,
        });

        let observation = encode_observation_v3(&state);
        assert!(matches!(
            observation.phase,
            PhaseV3::AwaitingChoice { options, .. }
                if options == ["choice:middle.trolley:0", "choice:middle.trolley:1"]
        ));
    }

    #[test]
    fn every_text_emitted_by_played_games_is_in_the_runtime_contract() {
        let contract = observation_v3_runtime_text_tokens()
            .into_iter()
            .map(str::to_owned)
            .collect::<BTreeSet<_>>();
        let engine = Engine::bundled_20260807().unwrap();
        let mut emitted = BTreeSet::new();
        for seed in 0..24u64 {
            let mut state = engine.new_game(Mode::Standard, seed).unwrap();
            for _ in 0..160 {
                if state.status != GameStatus::Ongoing {
                    break;
                }
                emitted.extend(observation_text_tokens(&encode_observation_v3(&state)));
                let actions = engine.legal_actions(&state).unwrap();
                if actions.is_empty() {
                    break;
                }
                // Rotate deterministically through the legal list instead of always
                // taking index 0 so forced moves, choices and cards are exercised.
                let index = (state.ply as usize + seed as usize) % actions.len();
                state = engine.apply(&state, actions[index].clone()).unwrap().state;
            }
        }
        assert!(!emitted.is_empty());
        let unknown: Vec<_> = emitted.difference(&contract).collect();
        assert!(
            unknown.is_empty(),
            "played-game tokens missing from the V3 runtime vocabulary contract: {unknown:?}"
        );
    }

    fn piece_with_captures(owner: Player, kind: PieceKind, id: u32, capture_count: u16) -> Piece {
        let mut piece = Piece::new(owner, kind).with_id(id);
        piece.capture_count = capture_count;
        piece
    }

    #[test]
    fn dense_shapes_ranges_and_runtime_coverage_are_fixed() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = engine.new_game(Mode::Standard, 42).unwrap();
        let observation = encode_observation_v3(&state);

        assert_eq!(observation.schema_id, OBSERVATION_V3_SCHEMA_ID);
        assert_eq!(observation.spatial_shape, [136, 8, 8]);
        assert_eq!(observation.spatial.len(), OBSERVATION_V3_SPATIAL_SIZE);
        assert_eq!(observation.global_shape, [32]);
        assert_eq!(observation.global.len(), OBSERVATION_V3_GLOBAL_FEATURES);
        assert!(
            observation
                .spatial
                .iter()
                .chain(&observation.global)
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        );
        assert_eq!(observation.runtime.len(), 194);
        assert_eq!(
            observation
                .runtime
                .iter()
                .map(|field| field.field_name().unwrap())
                .collect::<Vec<_>>(),
            CARD_RUNTIME_SOURCE_FIELDS
        );
        assert!(
            observation
                .runtime
                .iter()
                .enumerate()
                .all(|(index, field)| field.field_id == (index + 1) as u16)
        );
    }

    #[test]
    fn all_68_piece_kinds_have_distinct_self_planes() {
        // Per-kind boards avoid trying to fit68 distinct pieces in64 squares.
        let mut seen_codes = BTreeSet::new();
        let mut seen_planes = BTreeSet::new();
        for (code, kind) in PIECE_KIND_VOCABULARY {
            let mut board = Board::empty();
            board.set(
                "d4".parse().unwrap(),
                Some(Piece::new(Player::White, kind).with_id(10)),
            );
            let observation = encode_observation_v3(&playing_state(board, Player::White, 1));
            seen_codes.insert(observation.pieces[0].kind_code);
            let plane = if code <= 58 {
                usize::from(code - 1)
            } else {
                116 + 2 * usize::from(code - 59)
            };
            seen_planes.insert(plane);
            assert_eq!(observation.spatial[plane * 64 + 27], 1.0);
            assert_eq!(
                observation
                    .spatial
                    .iter()
                    .filter(|value| **value != 0.0)
                    .count(),
                1
            );
        }
        assert_eq!(seen_codes, (1..=68).collect());
        assert_eq!(seen_planes.len(), 68);
    }

    #[test]
    fn all_implemented_card_ids_survive_without_hashing_or_truncation() {
        let mut state = playing_state(Board::empty(), Player::White, 3);
        for (index, definition) in IMPLEMENTED_CARDS.iter().enumerate() {
            state.add_card(CardInstance {
                instance_id: index as u32 + 100,
                definition_id: definition.id.to_owned(),
                owner: if index % 2 == 0 {
                    Player::White
                } else {
                    Player::Black
                },
                half_stars: definition.half_stars,
                charges: 1,
                available_after_turn: None,
            });
        }

        let observation = encode_observation_v3(&state);
        let observed_ids = observation
            .cards
            .iter()
            .map(|card| card.definition_id.as_str())
            .collect::<BTreeSet<_>>();
        let expected_ids = IMPLEMENTED_CARDS
            .iter()
            .map(|definition| definition.id)
            .collect::<BTreeSet<_>>();

        assert_eq!(
            expected_ids.len(),
            IMPLEMENTED_CARDS.len(),
            "card IDs must be unique"
        );
        assert_eq!(observation.cards.len(), IMPLEMENTED_CARDS.len());
        assert_eq!(
            observation.card_slot_count as usize,
            IMPLEMENTED_CARDS.len()
        );
        assert_eq!(observed_ids, expected_ids);
    }

    #[test]
    fn raw_ids_and_rng_state_do_not_change_the_observation() {
        let make_board = |pawn_id, dragon_id| {
            let mut board = Board::empty();
            board.set(
                "b2".parse().unwrap(),
                Some(Piece::new(Player::White, PieceKind::Pawn).with_id(pawn_id)),
            );
            board.set(
                "g7".parse().unwrap(),
                Some(Piece::new(Player::Black, PieceKind::Dragon).with_id(dragon_id)),
            );
            board
        };
        let left = playing_state(make_board(10, 20), Player::White, 1);
        let right = playing_state(make_board(10_000, 90_000), Player::White, 999);

        assert_ne!(left.fingerprint(), right.fingerprint());
        assert_eq!(encode_observation_v3(&left), encode_observation_v3(&right));
    }

    #[test]
    fn color_swap_and_rank_mirror_are_canonical() {
        let mut white_board = Board::empty();
        white_board.set(
            "b2".parse().unwrap(),
            Some(piece_with_captures(Player::White, PieceKind::Pawn, 100, 2)),
        );
        white_board.set(
            "g7".parse().unwrap(),
            Some(piece_with_captures(
                Player::Black,
                PieceKind::Dragon,
                900,
                3,
            )),
        );

        let mut black_board = Board::empty();
        black_board.set(
            "b7".parse().unwrap(),
            Some(piece_with_captures(Player::Black, PieceKind::Pawn, 11, 2)),
        );
        black_board.set(
            "g2".parse().unwrap(),
            Some(piece_with_captures(Player::White, PieceKind::Dragon, 12, 3)),
        );

        let white = playing_state(white_board, Player::White, 5);
        let black = playing_state(black_board, Player::Black, 6);
        assert_eq!(encode_observation_v3(&white), encode_observation_v3(&black));
    }

    #[test]
    fn representative_cards_and_runtime_relations_are_canonical() {
        let mut white_board = Board::empty();
        white_board.set(
            "b2".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Wizard).with_id(100)),
        );
        white_board.set(
            "g7".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::King).with_id(900)),
        );
        let mut black_board = Board::empty();
        black_board.set(
            "b7".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::Wizard).with_id(11)),
        );
        black_board.set(
            "g2".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::King).with_id(12)),
        );

        let mut white = playing_state(white_board, Player::White, 5);
        let mut black = playing_state(black_board, Player::Black, 6);
        white.card_runtime.hill_king = [true, false];
        black.card_runtime.hill_king = [false, true];
        white.card_runtime.wizard_mana.insert(100, 3);
        black.card_runtime.wizard_mana.insert(11, 3);
        white.card_runtime.black_holes.insert("c3".parse().unwrap());
        black.card_runtime.black_holes.insert("c6".parse().unwrap());
        white.card_runtime.delayed_hazards.push(DelayedHazard {
            kind: DelayedHazardKind::Meteor,
            owner: Player::White,
            trigger_after: Player::Black,
            squares: vec!["d4".parse().unwrap(), "e5".parse().unwrap()],
        });
        black.card_runtime.delayed_hazards.push(DelayedHazard {
            kind: DelayedHazardKind::Meteor,
            owner: Player::Black,
            trigger_after: Player::White,
            squares: vec!["d5".parse().unwrap(), "e4".parse().unwrap()],
        });
        white.add_card(CardInstance {
            instance_id: 700,
            definition_id: "sprint".to_owned(),
            owner: Player::White,
            half_stars: 4,
            charges: 1,
            available_after_turn: Some(2),
        });
        black.add_card(CardInstance {
            instance_id: 8,
            definition_id: "sprint".to_owned(),
            owner: Player::Black,
            half_stars: 4,
            charges: 1,
            available_after_turn: Some(2),
        });

        assert_eq!(encode_observation_v3(&white), encode_observation_v3(&black));
    }

    #[test]
    fn distinct_runtime_meanings_do_not_collapse() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut base = engine.new_game(Mode::Standard, 4).unwrap();
        base.phase = GamePhase::Playing;
        let before = encode_observation_v3(&base);

        base.card_runtime.hill_king[base.turn.index()] = true;
        let after = encode_observation_v3(&base);
        assert_ne!(before, after);
        let field = after
            .runtime
            .iter()
            .find(|field| field.field_name() == Some("hill_king"))
            .unwrap();
        assert_eq!(
            field.value,
            RuntimeValueV3::List(vec![
                RuntimeValueV3::Bool(true),
                RuntimeValueV3::Bool(false),
            ])
        );
    }
}
