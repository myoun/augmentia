use crate::{Board, DeterministicRng, MoveAction, Piece, PieceKind, Player, Square};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Mode {
    Standard,
    Chaos,
    Grand,
    Campaign(String),
}

impl Mode {
    pub fn id(&self) -> &str {
        match self {
            Self::Standard => "standard",
            Self::Chaos => "chaos",
            Self::Grand => "grand",
            Self::Campaign(id) => id,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GamePhase {
    Playing,
    ForcedMove {
        piece_id: u32,
        source: ForcedMoveSource,
        continuation: ForcedMoveContinuation,
    },
    Draft(DraftState),
    AwaitingChoice(PendingChoice),
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DraftPhase {
    Opening,
    Middle,
    End,
}

impl DraftPhase {
    pub const fn index(self) -> usize {
        match self {
            Self::Opening => 0,
            Self::Middle => 1,
            Self::End => 2,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DraftState {
    pub phase: DraftPhase,
    pub offers: Vec<String>,
    pub resume_turn: Player,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ForcedMoveSource {
    Platform,
    Effect(String),
}

/// Stable model/runtime token for the extra move created by middle-card effects.
pub const MIDDLE_EXTRA_MOVE_SOURCE: &str = "middle-extra-move";

/// Stable model/runtime prefix for the dynamic field-promotion follow-up choice.
pub const FIELD_PROMOTION_CHOICE_PREFIX: &str = "field-promotion:";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ForcedMoveContinuation {
    ResumeTurn,
    FinishTurn,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChoiceContinuation {
    ResumeTurn,
    FinishTurn,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PendingChoice {
    pub id: u32,
    pub owner: Player,
    pub source_card_instance: u32,
    pub handler_id: String,
    pub options: Vec<String>,
    pub continuation: ChoiceContinuation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GameStatus {
    Ongoing,
    Won { winner: Player, reason: WinReason },
    Draw { reason: DrawReason },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WinReason {
    KingCaptured,
    NoLegalActions,
    LowerStarsOnRepetition,
    LowerStarsAfterDeathmatch,
    CardEffect(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DrawReason {
    EqualStarsOnRepetition,
    EqualStarsAfterDeathmatch,
    CardEffect(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CardInstance {
    pub instance_id: u32,
    pub definition_id: String,
    pub owner: Player,
    /// Stars are stored in half-star units so all comparisons are integer-only.
    pub half_stars: u16,
    pub charges: u16,
    /// MIDDLE and END draft cards unlock after their owner's first subsequent turn.
    pub available_after_turn: Option<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeathmatchState {
    pub started_at_shared_turn: u32,
    pub quiet_full_turns: u32,
    pub interval_full_turns: u32,
    pub progress_this_round: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EnPassantState {
    pub target: Square,
    pub captured: Square,
    pub pawn_owner: Player,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EmptyLunchboxStatus {
    pub by: Player,
    pub deadline_turn: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlatformRuleState {
    pub cell: Option<Square>,
    pub spawned_at: u32,
    pub next_at: u32,
    pub triggered_piece_ids: BTreeSet<u32>,
    pub nonce: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InitiativeStatus {
    pub by: Player,
    pub start_turn: u32,
    pub limit: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DiceLockStatus {
    pub kind: PieceKind,
    pub remaining_turns: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrozenStatus {
    pub source: Player,
    pub remaining_turns: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RepositionStatus {
    pub piece_id: u32,
    pub used: bool,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum WindmillMode {
    Bishop,
    Rook,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LogStatus {
    pub file_delta: i8,
    pub rank_delta: i8,
    pub roll_after_turn: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DelayedHazardKind {
    Lightning,
    Meteor,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DelayedHazard {
    pub kind: DelayedHazardKind,
    pub owner: Player,
    pub trigger_after: Player,
    pub squares: Vec<Square>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PendingOtherworld {
    pub piece_id: u32,
    pub owner: Player,
    pub square: Square,
    pub origin: Square,
    pub due_ply: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LargePieceStatus {
    pub anchor: Square,
    pub hp: u8,
    pub max_hp: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TimedPieceStatus {
    pub by: Player,
    pub remaining_turns: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoyalCommandStatus {
    pub starts_at_turn: u32,
    pub expires_at_turn: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FeudalContract {
    pub owner: Player,
    pub pawn_id: u32,
    pub guardian_id: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PendingPanic {
    pub owner: Player,
    pub piece_ids: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PendingPremove {
    pub owner: Player,
    pub trigger_player: Player,
    pub trigger_turn: u32,
    pub moves: Vec<MoveAction>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PendingScarecrow {
    pub owner: Player,
    pub square: Square,
    pub remaining_ply: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TwinBond {
    pub first_id: u32,
    pub second_id: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrolleyDilemma {
    pub by: Player,
    pub bundles: [Vec<u32>; 2],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BribeRevert {
    pub piece_id: u32,
    pub due_turn: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NecromancyRevert {
    pub piece_id: u32,
    pub due_ply: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PendingVip {
    pub piece_id: u32,
    pub owner: Player,
    pub due_ply: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PendingIcbm {
    pub queen_id: u32,
    pub owner: Player,
    pub due_turn: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JudgmentExile {
    pub piece: Piece,
    pub square: Square,
    pub return_at_draft: DraftPhase,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UndeadResurrection {
    pub piece: Piece,
    pub due_ply: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplayCell {
    pub square: Square,
    pub before: Option<Piece>,
    pub after: Option<Piece>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MoveReplayState {
    pub cells: Vec<ReplayCell>,
    pub en_passant_before: Option<EnPassantState>,
    pub en_passant_after: Option<EnPassantState>,
    /// The web client stores status flags on the piece objects contained in the board delta.
    /// Core keeps those flags in `CardRuntimeState`, so retain matching runtime snapshots here.
    /// Nested replay entries are stripped before these snapshots are stored.
    pub runtime_before: Box<CardRuntimeState>,
    pub runtime_after: Box<CardRuntimeState>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CardRuntimeState {
    /// Active OPENING cards acquired from the initial draft are forced after their owner's first
    /// move instead of being exposed as manual card actions.
    pub opening_auto_card_ids: BTreeSet<u32>,
    pub hill_king: [bool; 2],
    pub geneva_convention: [bool; 2],
    pub pawn_sprint: [bool; 2],
    pub early_promotion: [bool; 2],
    pub initiative: [Option<InitiativeStatus>; 2],
    pub dice_locks: [Option<DiceLockStatus>; 2],
    pub moved_piece_ids: BTreeSet<u32>,
    pub fresh_no_capture_until: BTreeMap<u32, u32>,
    pub disarmed: BTreeMap<u32, u32>,
    pub frozen: BTreeMap<u32, FrozenStatus>,
    pub reposition: Option<RepositionStatus>,
    pub revelation: bool,
    pub parry_piece_ids: BTreeSet<u32>,
    pub empty_lunchboxes: BTreeMap<u32, EmptyLunchboxStatus>,
    pub platform: Option<PlatformRuleState>,
    pub herald_jump_locked_until: BTreeMap<u32, u32>,
    pub windmill_modes: BTreeMap<u32, WindmillMode>,
    pub log_statuses: BTreeMap<u32, LogStatus>,
    pub wizard_mana: BTreeMap<u32, u8>,
    pub reaper_souls: BTreeMap<u32, u8>,
    pub shielded_piece_ids: BTreeSet<u32>,
    pub protected_piece_ids: BTreeSet<u32>,
    pub idol_used_turn: BTreeMap<u32, u32>,
    pub idol_rest_piece_ids: BTreeSet<u32>,
    pub delayed_hazards: Vec<DelayedHazard>,
    pub skip_turns: [bool; 2],
    pub piece_origins: BTreeMap<u32, Square>,
    pub large_pieces: BTreeMap<u32, LargePieceStatus>,
    pub loyalist_piece_ids: BTreeSet<u32>,
    pub royal_piece_ids: BTreeSet<u32>,
    pub calling_cards: BTreeMap<u32, Player>,
    pub holdout_promotions: BTreeMap<u32, u32>,
    pub pending_otherworld: Vec<PendingOtherworld>,
    pub merchant_gold: BTreeMap<u32, u16>,
    pub democracy: [bool; 2],
    pub fast_growth: [bool; 2],
    pub fianchetto: [bool; 2],
    pub field_promotion: [bool; 2],
    pub field_promoted_piece_ids: BTreeSet<u32>,
    pub frontline_response: [bool; 2],
    pub relay: [bool; 2],
    pub gomoku: [bool; 2],
    pub magic_girl_surge: [bool; 2],
    pub free_move_no_capture: [bool; 2],
    pub siren_exposure: BTreeMap<(u32, u32), u8>,
    pub trickster_move_kinds: BTreeMap<u32, PieceKind>,
    pub undead_resurrections: Vec<UndeadResurrection>,
    pub judgment_exiles: Vec<JudgmentExile>,
    pub move_replays: [Option<MoveReplayState>; 2],
    pub iron_monarch: [bool; 2],
    pub pawn_conversion: [bool; 2],
    pub backward_knight: [bool; 2],
    pub breakthrough_pawns: [bool; 2],
    pub corner_kick: [bool; 2],
    pub en_passant_frenzy: [bool; 2],
    pub file_surge: [bool; 2],
    pub horse_riding: [bool; 2],
    pub knight_injury: [bool; 2],
    pub pawn_leap: [bool; 2],
    pub radical_charge: [bool; 2],
    pub retreat: [bool; 2],
    pub rook_lift: [bool; 2],
    pub encouragement: [bool; 2],
    pub exhaustion: [bool; 2],
    pub overwhelm: [bool; 2],
    pub imperial_studies: [bool; 2],
    pub mad_horse: [bool; 2],
    pub queen_afterimage: [bool; 2],
    pub religious_victory: [bool; 2],
    pub basic_training_piece_ids: BTreeSet<u32>,
    pub promotion_rush_piece_ids: BTreeSet<u32>,
    pub evasion_piece_ids: BTreeSet<u32>,
    pub ghost_piece_ids: BTreeSet<u32>,
    pub inertia_piece_ids: BTreeSet<u32>,
    pub ice_sheet: BTreeMap<u32, TimedPieceStatus>,
    pub severed: BTreeMap<u32, TimedPieceStatus>,
    pub royal_command: [Option<RoyalCommandStatus>; 2],
    pub socialism: [u32; 2],
    pub taunt: [u32; 2],
    pub armistice_remaining_ply: u32,
    pub exhaustion_last_piece: [Option<u32>; 2],
    pub exhaustion_count: [u8; 2],
    pub chameleon_piece_ids: BTreeSet<u32>,
    pub chimera_piece_ids: BTreeSet<u32>,
    pub imperial_moves: BTreeMap<u32, BTreeSet<PieceKind>>,
    pub mistake_card: [bool; 2],
    pub poisoned_piece_ids: BTreeSet<u32>,
    pub sacrifice_protection: BTreeMap<u32, TimedPieceStatus>,
    pub last_resistance: BTreeMap<u32, TimedPieceStatus>,
    /// Owner turn number at which this temporary protection was granted.
    pub coronation_protection: BTreeMap<u32, u32>,
    pub queens_gambit_protection: BTreeSet<u32>,
    pub potion_basic_training_piece_ids: BTreeSet<u32>,
    pub poison_stun_turns: BTreeMap<u32, u8>,
    pub potion_manner_piece_ids: BTreeSet<u32>,
    pub manner_captured_last_piece_ids: BTreeSet<u32>,
    pub potion_saturation_piece_ids: BTreeSet<u32>,
    pub saturation_capture_counts: BTreeMap<u32, u8>,
    pub bishop_snipe: [bool; 2],
    pub explosive_piece_ids: BTreeSet<u32>,
    pub trojan_piece_ids: BTreeSet<u32>,
    pub feudal_contracts: Vec<FeudalContract>,
    pub gale_due_turns: [Vec<u32>; 2],
    pub pending_panic: Vec<PendingPanic>,
    pub pending_premoves: Vec<PendingPremove>,
    pub pending_scarecrows: Vec<PendingScarecrow>,
    pub quantum_pending: [bool; 2],
    pub quantum_positions: BTreeMap<u32, Square>,
    pub stealth_piece_ids: BTreeSet<u32>,
    pub submerged_piece_ids: BTreeSet<u32>,
    pub substitution: [bool; 2],
    pub switcheroo: [bool; 2],
    pub twin_bonds: Vec<TwinBond>,
    pub witch_trials: BTreeMap<u32, TimedPieceStatus>,
    pub pending_trolley_by: [Option<Player>; 2],
    pub active_trolley: Option<TrolleyDilemma>,
    pub baby_bear_grow_turns: BTreeMap<u32, u32>,
    pub bear_retaliations: BTreeMap<u32, u8>,
    pub bribe_reverts: Vec<BribeRevert>,
    pub charge_pawns: BTreeSet<u32>,
    pub coronation: [bool; 2],
    pub frenzy_pawns: BTreeSet<u32>,
    pub hallucination_until_ply: [Option<u32>; 2],
    pub moving: [bool; 2],
    pub moving_consecutive: [(Option<u32>, u8); 2],
    pub necromancy_reverts: Vec<NecromancyRevert>,
    pub reverse_pawns_charges: [u8; 2],
    pub stake_statuses: BTreeMap<u32, TimedPieceStatus>,
    pub underpromotion: [bool; 2],
    pub vip_pending: Vec<PendingVip>,
    pub barricades: BTreeSet<Square>,
    pub chains: Vec<[u32; 2]>,
    pub collapsed_files_ranks: bool,
    pub desperado_pieces: BTreeSet<u32>,
    pub palaces: BTreeMap<Player, [Square; 2]>,
    pub portals: Option<[Square; 2]>,
    pub bunkers: BTreeMap<u32, u8>,
    pub zugzwang: [bool; 2],
    pub bina_mate: [bool; 2],
    pub final_weapon: [bool; 2],
    pub icbm_pending: Vec<PendingIcbm>,
    pub prophecy: [Option<u32>; 2],
    pub racing_king: [bool; 2],
    pub spies: BTreeMap<u32, Player>,
    pub ultimatum: Option<u32>,
    pub vanish: [bool; 2],
    pub acceleration: bool,
    pub chaos_no_capture_until_ply: u32,
    pub black_holes: BTreeSet<Square>,
    pub camouflage_color: bool,
    pub conveyor: bool,
    pub cool_guy_pieces: BTreeSet<u32>,
    pub crown_pos: Option<Square>,
    pub crown_holder: Option<(Player, u32)>,
    pub football_pos: Option<Square>,
    pub high_ground_squares: BTreeSet<Square>,
    pub highways: BTreeSet<u8>,
    pub macho_chess: bool,
    pub mistake_rule: bool,
    pub monochrome_chess: bool,
    pub monster_rule: Option<u32>,
    pub periodic_collapse_next: Option<u32>,
    pub rule_portals: Option<[Square; 2]>,
    pub recycling: bool,
    pub rule_bombs: BTreeSet<Square>,
    pub saturation_rule: bool,
    pub transcendence: bool,
    pub winter_kingdom_next: Option<u32>,
    pub cloned_passives: [BTreeSet<String>; 2],
}

impl CardRuntimeState {
    fn has_middle_state(&self) -> bool {
        [
            self.backward_knight,
            self.breakthrough_pawns,
            self.corner_kick,
            self.en_passant_frenzy,
            self.file_surge,
            self.horse_riding,
            self.knight_injury,
            self.pawn_leap,
            self.radical_charge,
            self.retreat,
            self.rook_lift,
            self.encouragement,
            self.exhaustion,
            self.overwhelm,
            self.imperial_studies,
            self.mad_horse,
            self.queen_afterimage,
            self.religious_victory,
            self.mistake_card,
            self.bishop_snipe,
            self.quantum_pending,
            self.substitution,
            self.switcheroo,
        ]
        .into_iter()
        .any(|flags| flags != [false, false])
            || !self.basic_training_piece_ids.is_empty()
            || !self.promotion_rush_piece_ids.is_empty()
            || !self.evasion_piece_ids.is_empty()
            || !self.ghost_piece_ids.is_empty()
            || !self.inertia_piece_ids.is_empty()
            || !self.ice_sheet.is_empty()
            || !self.severed.is_empty()
            || self.royal_command != [None, None]
            || self.socialism != [0, 0]
            || self.taunt != [0, 0]
            || self.armistice_remaining_ply != 0
            || self.exhaustion_last_piece != [None, None]
            || self.exhaustion_count != [0, 0]
            || !self.chameleon_piece_ids.is_empty()
            || !self.chimera_piece_ids.is_empty()
            || !self.imperial_moves.is_empty()
            || !self.poisoned_piece_ids.is_empty()
            || !self.sacrifice_protection.is_empty()
            || !self.last_resistance.is_empty()
            || !self.coronation_protection.is_empty()
            || !self.queens_gambit_protection.is_empty()
            || !self.potion_basic_training_piece_ids.is_empty()
            || !self.poison_stun_turns.is_empty()
            || !self.potion_manner_piece_ids.is_empty()
            || !self.manner_captured_last_piece_ids.is_empty()
            || !self.potion_saturation_piece_ids.is_empty()
            || !self.saturation_capture_counts.is_empty()
            || !self.explosive_piece_ids.is_empty()
            || !self.trojan_piece_ids.is_empty()
            || !self.feudal_contracts.is_empty()
            || self
                .gale_due_turns
                .iter()
                .any(|entries| !entries.is_empty())
            || !self.pending_panic.is_empty()
            || !self.pending_premoves.is_empty()
            || !self.pending_scarecrows.is_empty()
            || !self.quantum_positions.is_empty()
            || !self.stealth_piece_ids.is_empty()
            || !self.submerged_piece_ids.is_empty()
            || !self.twin_bonds.is_empty()
            || !self.witch_trials.is_empty()
            || self.pending_trolley_by != [None, None]
            || self.active_trolley.is_some()
            || !self.baby_bear_grow_turns.is_empty()
            || !self.bear_retaliations.is_empty()
    }

    fn has_end_state(&self) -> bool {
        !self.bribe_reverts.is_empty()
            || !self.charge_pawns.is_empty()
            || self.coronation != [false, false]
            || !self.frenzy_pawns.is_empty()
            || self.hallucination_until_ply != [None, None]
            || self.moving != [false, false]
            || self.moving_consecutive != [(None, 0), (None, 0)]
            || !self.necromancy_reverts.is_empty()
            || self.reverse_pawns_charges != [0, 0]
            || !self.stake_statuses.is_empty()
            || self.underpromotion != [false, false]
            || !self.vip_pending.is_empty()
            || !self.barricades.is_empty()
            || !self.chains.is_empty()
            || self.collapsed_files_ranks
            || !self.desperado_pieces.is_empty()
            || !self.palaces.is_empty()
            || self.portals.is_some()
            || !self.bunkers.is_empty()
            || self.zugzwang != [false, false]
            || self.bina_mate != [false, false]
            || self.final_weapon != [false, false]
            || !self.icbm_pending.is_empty()
            || self.prophecy != [None, None]
            || self.racing_king != [false, false]
            || !self.spies.is_empty()
            || self.ultimatum.is_some()
            || self.vanish != [false, false]
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GameState {
    pub ruleset_id: String,
    pub mode: Mode,
    pub board: Board,
    pub turn: Player,
    pub ply: u32,
    pub quiet_ply: u32,
    pub actions_remaining: u16,
    pub cards_used_this_turn: [u16; 2],
    pub phase: GamePhase,
    pub status: GameStatus,
    pub cards: Vec<CardInstance>,
    pub half_star_totals: [u16; 2],
    pub turns_taken: [u32; 2],
    pub completed_drafts: [bool; 3],
    pub deathmatch: Option<DeathmatchState>,
    pub en_passant: Option<EnPassantState>,
    pub applied_rules: Vec<String>,
    pub card_runtime: CardRuntimeState,
    rng: DeterministicRng,
    next_piece_id: u32,
    next_card_instance_id: u32,
    next_choice_id: u32,
    turn_end_requested: bool,
    repetitions: BTreeMap<u64, u8>,
}

impl GameState {
    pub fn standard(ruleset_id: impl Into<String>, mode: Mode, seed: u64) -> Self {
        Self::from_board(ruleset_id, mode, Board::standard(), Player::White, seed)
    }

    pub fn from_board(
        ruleset_id: impl Into<String>,
        mode: Mode,
        mut board: Board,
        turn: Player,
        seed: u64,
    ) -> Self {
        let next_piece_id = board.assign_stable_ids();
        let mut state = Self {
            ruleset_id: ruleset_id.into(),
            mode,
            board,
            turn,
            ply: 0,
            quiet_ply: 0,
            actions_remaining: 1,
            cards_used_this_turn: [0, 0],
            phase: GamePhase::Playing,
            status: GameStatus::Ongoing,
            cards: Vec::new(),
            half_star_totals: [0, 0],
            turns_taken: [0, 0],
            completed_drafts: [false; 3],
            deathmatch: None,
            en_passant: None,
            applied_rules: Vec::new(),
            card_runtime: CardRuntimeState::default(),
            rng: DeterministicRng::new(seed),
            next_piece_id,
            next_card_instance_id: 1,
            next_choice_id: 1,
            turn_end_requested: false,
            repetitions: BTreeMap::new(),
        };
        state.card_runtime.piece_origins = state
            .board
            .iter()
            .map(|(square, piece)| (piece.id, square))
            .collect();
        state.record_current_position();
        state
    }

    pub fn add_card(&mut self, card: CardInstance) {
        self.half_star_totals[card.owner.index()] =
            self.half_star_totals[card.owner.index()].saturating_add(card.half_stars);
        self.cards.push(card);
        self.cards
            .sort_by_key(|card| (card.owner.index(), card.instance_id));
        // Acquiring a card changes the position identity. The caller controls when it should be
        // recorded because draft transitions may add more than one card atomically.
    }

    pub fn rng_state(&self) -> u64 {
        self.rng.state()
    }

    pub(crate) fn rng_mut(&mut self) -> &mut DeterministicRng {
        &mut self.rng
    }

    pub(crate) fn allocate_piece_id(&mut self) -> u32 {
        let id = self.next_piece_id.max(1);
        self.next_piece_id = id.saturating_add(1).max(1);
        id
    }

    pub(crate) fn allocate_card_instance_id(&mut self) -> u32 {
        let after_existing = self
            .cards
            .iter()
            .map(|card| card.instance_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
            .max(1);
        let id = self.next_card_instance_id.max(after_existing).max(1);
        self.next_card_instance_id = id.saturating_add(1).max(1);
        id
    }

    pub(crate) fn allocate_choice_id(&mut self) -> u32 {
        let id = self.next_choice_id.max(1);
        self.next_choice_id = id.saturating_add(1).max(1);
        id
    }

    pub(crate) fn request_turn_end(&mut self) {
        self.turn_end_requested = true;
    }

    pub(crate) fn take_turn_end_request(&mut self) -> bool {
        std::mem::take(&mut self.turn_end_requested)
    }

    pub(crate) fn reset_repetition_history(&mut self) {
        self.repetitions.clear();
    }

    pub(crate) fn mark_deathmatch_progress(&mut self) {
        if let Some(deathmatch) = &mut self.deathmatch {
            deathmatch.quiet_full_turns = 0;
            deathmatch.progress_this_round = true;
        }
    }

    pub fn shared_turn_count(&self) -> u32 {
        self.turns_taken[0].min(self.turns_taken[1])
    }

    /// Resolve the piece that owns a legal move source, including a quantum shadow square.
    pub fn move_source_piece(&self, square: Square) -> Option<Piece> {
        self.board.get(square).or_else(|| {
            self.card_runtime
                .quantum_positions
                .iter()
                .find_map(|(piece_id, shadow)| {
                    (*shadow == square)
                        .then(|| self.board.find_by_id(*piece_id).map(|(_, piece)| piece))
                        .flatten()
                })
        })
    }

    /// Stable choice-handler family used by the V3 model vocabulary.
    pub fn choice_handler_family(handler_id: &str) -> &str {
        if handler_id
            .strip_prefix(FIELD_PROMOTION_CHOICE_PREFIX)
            .is_some_and(|piece_id| piece_id.parse::<u32>().is_ok())
        {
            return crate::FIELD_PROMOTION;
        }
        handler_id
    }

    pub fn position_key(&self) -> u64 {
        let mut hash = StableHasher::new();
        hash.bytes(self.ruleset_id.as_bytes());
        hash.bytes(self.mode.id().as_bytes());
        hash.u8(self.turn.index() as u8);
        hash.u16(self.actions_remaining);
        hash.u16(self.cards_used_this_turn[0]);
        hash.u16(self.cards_used_this_turn[1]);
        hash.u8(u8::from(self.turn_end_requested));

        for index in 0..64 {
            let square = crate::Square::from_index(index).expect("0..64 is a valid square");
            match self.board.get(square) {
                None => hash.u8(0),
                Some(piece) => {
                    hash.u32(piece.id);
                    hash.u8(piece.kind.stable_code());
                    hash.u8(piece.owner.index() as u8);
                    hash.u16(piece.capture_count);
                }
            }
        }

        hash.u16(self.half_star_totals[0]);
        hash.u16(self.half_star_totals[1]);
        for card in &self.cards {
            hash.u32(card.instance_id);
            hash.bytes(card.definition_id.as_bytes());
            hash.u8(card.owner.index() as u8);
            hash.u16(card.half_stars);
            hash.u16(card.charges);
            hash.u32(card.available_after_turn.map_or(0, |turn| {
                turn.saturating_sub(self.turns_taken[card.owner.index()])
                    .saturating_add(1)
            }));
        }
        for completed in self.completed_drafts {
            hash.u8(u8::from(completed));
        }
        match self.deathmatch {
            None => hash.u8(0),
            Some(deathmatch) => {
                hash.u8(1);
                hash.u32(deathmatch.interval_full_turns);
            }
        }
        match self.en_passant {
            None => hash.u8(0),
            Some(en_passant) => {
                hash.u8(1);
                hash.u8(en_passant.target.index() as u8);
                hash.u8(en_passant.captured.index() as u8);
                hash.u8(en_passant.pawn_owner.index() as u8);
            }
        }
        for rule in &self.applied_rules {
            hash.bytes(rule.as_bytes());
        }
        for instance_id in &self.card_runtime.opening_auto_card_ids {
            hash.u32(*instance_id);
        }
        for enabled in self.card_runtime.hill_king {
            hash.u8(u8::from(enabled));
        }
        for enabled in self.card_runtime.geneva_convention {
            hash.u8(u8::from(enabled));
        }
        for enabled in self.card_runtime.pawn_sprint {
            hash.u8(u8::from(enabled));
        }
        for enabled in self.card_runtime.early_promotion {
            hash.u8(u8::from(enabled));
        }
        for player in [Player::White, Player::Black] {
            match self.card_runtime.initiative[player.index()] {
                None => hash.u8(0),
                Some(status) => {
                    hash.u8(1);
                    hash.u8(status.by.index() as u8);
                    hash.u32(status.limit.saturating_sub(
                        self.turns_taken[player.index()].saturating_sub(status.start_turn),
                    ));
                }
            }
            match self.card_runtime.dice_locks[player.index()] {
                None => hash.u8(0),
                Some(status) => {
                    hash.u8(1);
                    hash.u8(status.kind.stable_code());
                    hash.u32(status.remaining_turns);
                }
            }
        }
        for piece_id in &self.card_runtime.moved_piece_ids {
            hash.u32(*piece_id);
        }
        for (piece_id, deadline) in &self.card_runtime.fresh_no_capture_until {
            hash.u32(*piece_id);
            let owner = self
                .board
                .find_by_id(*piece_id)
                .map_or(self.turn, |(_, piece)| piece.owner);
            hash.u32(deadline.saturating_sub(self.turns_taken[owner.index()]));
        }
        for (piece_id, remaining_turns) in &self.card_runtime.disarmed {
            hash.u32(*piece_id);
            hash.u32(*remaining_turns);
        }
        for (piece_id, status) in &self.card_runtime.frozen {
            hash.u32(*piece_id);
            hash.u8(status.source.index() as u8);
            hash.u32(status.remaining_turns);
        }
        match self.card_runtime.reposition {
            None => hash.u8(0),
            Some(status) => {
                hash.u8(1);
                hash.u32(status.piece_id);
                hash.u8(u8::from(status.used));
            }
        }
        hash.u8(u8::from(self.card_runtime.revelation));
        for piece_id in &self.card_runtime.parry_piece_ids {
            hash.u32(*piece_id);
        }
        for (piece_id, status) in &self.card_runtime.empty_lunchboxes {
            hash.u32(*piece_id);
            hash.u8(status.by.index() as u8);
            let owner = self
                .board
                .find_by_id(*piece_id)
                .map_or(status.by.opponent(), |(_, piece)| piece.owner);
            hash.u32(
                status
                    .deadline_turn
                    .saturating_sub(self.turns_taken[owner.index()]),
            );
        }
        match &self.card_runtime.platform {
            None => hash.u8(0),
            Some(platform) => {
                hash.u8(1);
                hash.u8(platform.cell.map_or(u8::MAX, |square| square.index() as u8));
                hash.u32(platform.next_at.saturating_sub(self.shared_turn_count()));
                for piece_id in &platform.triggered_piece_ids {
                    hash.u32(*piece_id);
                }
            }
        }
        for (piece_id, deadline) in &self.card_runtime.herald_jump_locked_until {
            hash.u32(*piece_id);
            let owner = self
                .board
                .find_by_id(*piece_id)
                .map_or(self.turn, |(_, piece)| piece.owner);
            hash.u32(deadline.saturating_sub(self.turns_taken[owner.index()]));
        }
        for (piece_id, mode) in &self.card_runtime.windmill_modes {
            hash.u32(*piece_id);
            hash.u8(match mode {
                WindmillMode::Bishop => 0,
                WindmillMode::Rook => 1,
            });
        }
        for (piece_id, status) in &self.card_runtime.log_statuses {
            hash.u32(*piece_id);
            hash.u8(status.file_delta as u8);
            hash.u8(status.rank_delta as u8);
            let owner = self
                .board
                .find_by_id(*piece_id)
                .map_or(self.turn, |(_, piece)| piece.owner);
            hash.u32(
                status
                    .roll_after_turn
                    .saturating_sub(self.turns_taken[owner.index()]),
            );
        }
        for (piece_id, mana) in &self.card_runtime.wizard_mana {
            hash.u32(*piece_id);
            hash.u8(*mana);
        }
        for (piece_id, souls) in &self.card_runtime.reaper_souls {
            hash.u32(*piece_id);
            hash.u8(*souls);
        }
        for piece_id in &self.card_runtime.shielded_piece_ids {
            hash.u32(*piece_id);
        }
        for piece_id in &self.card_runtime.protected_piece_ids {
            hash.u32(*piece_id);
        }
        for (piece_id, status) in &self.card_runtime.large_pieces {
            hash.u32(*piece_id);
            hash.u8(status.anchor.index() as u8);
            hash.u8(status.hp);
            hash.u8(status.max_hp);
        }
        for piece_id in &self.card_runtime.loyalist_piece_ids {
            hash.u32(*piece_id);
        }
        for piece_id in &self.card_runtime.royal_piece_ids {
            hash.u32(*piece_id);
        }
        for (piece_id, owner) in &self.card_runtime.calling_cards {
            hash.u32(*piece_id);
            hash.u8(owner.index() as u8);
        }
        for (piece_id, ready) in &self.card_runtime.holdout_promotions {
            hash.u32(*piece_id);
            hash.u32(ready.saturating_sub(self.shared_turn_count()));
        }
        for pending in &self.card_runtime.pending_otherworld {
            hash.u32(pending.piece_id);
            hash.u8(pending.owner.index() as u8);
            hash.u8(pending.square.index() as u8);
            hash.u8(pending.origin.index() as u8);
            hash.u32(pending.due_ply.saturating_sub(self.ply));
        }
        for (piece_id, gold) in &self.card_runtime.merchant_gold {
            hash.u32(*piece_id);
            hash.u16(*gold);
        }
        // Middle-card state participates in repetition identity. BTree-backed collections have a
        // stable debug order; absolute turn deadlines are normalized separately below. Keeping
        // the empty case byte-for-byte unchanged preserves pre-middle snapshot position keys.
        if self.card_runtime.has_middle_state() {
            hash.bytes(b"middle-runtime-v1");
            hash.bytes(
                format!(
                    "{:?}",
                    (
                        self.card_runtime.backward_knight,
                        self.card_runtime.breakthrough_pawns,
                        self.card_runtime.corner_kick,
                        self.card_runtime.en_passant_frenzy,
                        self.card_runtime.file_surge,
                        self.card_runtime.horse_riding,
                        self.card_runtime.knight_injury,
                        self.card_runtime.pawn_leap,
                        self.card_runtime.radical_charge,
                        self.card_runtime.retreat,
                        self.card_runtime.rook_lift,
                        self.card_runtime.encouragement,
                    )
                )
                .as_bytes(),
            );
            hash.bytes(
                format!(
                    "{:?}",
                    (
                        self.card_runtime.exhaustion,
                        self.card_runtime.overwhelm,
                        self.card_runtime.imperial_studies,
                        self.card_runtime.mad_horse,
                        self.card_runtime.queen_afterimage,
                        self.card_runtime.religious_victory,
                        &self.card_runtime.basic_training_piece_ids,
                        &self.card_runtime.promotion_rush_piece_ids,
                        &self.card_runtime.evasion_piece_ids,
                        &self.card_runtime.ghost_piece_ids,
                        &self.card_runtime.inertia_piece_ids,
                        &self.card_runtime.ice_sheet,
                    )
                )
                .as_bytes(),
            );
            hash.bytes(
                format!(
                    "{:?}",
                    (
                        &self.card_runtime.severed,
                        self.card_runtime.socialism,
                        self.card_runtime.taunt,
                        self.card_runtime.armistice_remaining_ply,
                        self.card_runtime.exhaustion_last_piece,
                        self.card_runtime.exhaustion_count,
                        &self.card_runtime.chameleon_piece_ids,
                        &self.card_runtime.chimera_piece_ids,
                        &self.card_runtime.imperial_moves,
                        self.card_runtime.mistake_card,
                        &self.card_runtime.poisoned_piece_ids,
                        &self.card_runtime.sacrifice_protection,
                    )
                )
                .as_bytes(),
            );
            hash.bytes(
                format!(
                    "{:?}",
                    (
                        &self.card_runtime.last_resistance,
                        &self.card_runtime.coronation_protection,
                        &self.card_runtime.queens_gambit_protection,
                        &self.card_runtime.potion_basic_training_piece_ids,
                        &self.card_runtime.poison_stun_turns,
                        &self.card_runtime.potion_manner_piece_ids,
                        &self.card_runtime.manner_captured_last_piece_ids,
                        &self.card_runtime.potion_saturation_piece_ids,
                        &self.card_runtime.saturation_capture_counts,
                    )
                )
                .as_bytes(),
            );
            hash.bytes(
                format!(
                    "{:?}",
                    (
                        self.card_runtime.bishop_snipe,
                        &self.card_runtime.explosive_piece_ids,
                        &self.card_runtime.trojan_piece_ids,
                        &self.card_runtime.feudal_contracts,
                        &self.card_runtime.pending_panic,
                        &self.card_runtime.pending_scarecrows,
                        self.card_runtime.quantum_pending,
                        &self.card_runtime.quantum_positions,
                        &self.card_runtime.stealth_piece_ids,
                        &self.card_runtime.submerged_piece_ids,
                        self.card_runtime.substitution,
                        self.card_runtime.switcheroo,
                    )
                )
                .as_bytes(),
            );
            hash.bytes(
                format!(
                    "{:?}",
                    (
                        &self.card_runtime.twin_bonds,
                        &self.card_runtime.witch_trials,
                        self.card_runtime.pending_trolley_by,
                        &self.card_runtime.active_trolley,
                        &self.card_runtime.bear_retaliations,
                    )
                )
                .as_bytes(),
            );
            for player in [Player::White, Player::Black] {
                if let Some(status) = self.card_runtime.royal_command[player.index()] {
                    hash.u8(player.index() as u8);
                    hash.u32(
                        status
                            .starts_at_turn
                            .saturating_sub(self.turns_taken[player.index()]),
                    );
                    hash.u32(
                        status
                            .expires_at_turn
                            .saturating_sub(self.turns_taken[player.index()]),
                    );
                }
                for due in &self.card_runtime.gale_due_turns[player.index()] {
                    hash.u8(player.index() as u8);
                    hash.u32(due.saturating_sub(self.turns_taken[player.index()]));
                }
            }
            for pending in &self.card_runtime.pending_premoves {
                hash.u8(pending.owner.index() as u8);
                hash.u8(pending.trigger_player.index() as u8);
                hash.u32(
                    pending
                        .trigger_turn
                        .saturating_sub(self.turns_taken[pending.trigger_player.index()]),
                );
                hash.bytes(format!("{:?}", pending.moves).as_bytes());
            }
            for (piece_id, due) in &self.card_runtime.baby_bear_grow_turns {
                hash.u32(*piece_id);
                hash.u32(due.saturating_sub(self.shared_turn_count()));
            }
        }
        if self.card_runtime.has_end_state() {
            hash.bytes(b"end-runtime-v1");
            hash.bytes(
                format!(
                    "{:?}",
                    (
                        &self.card_runtime.bribe_reverts,
                        &self.card_runtime.charge_pawns,
                        self.card_runtime.coronation,
                        &self.card_runtime.frenzy_pawns,
                        self.card_runtime.hallucination_until_ply,
                        self.card_runtime.moving,
                    )
                )
                .as_bytes(),
            );
            hash.bytes(
                format!(
                    "{:?}",
                    (
                        self.card_runtime.moving_consecutive,
                        &self.card_runtime.necromancy_reverts,
                        self.card_runtime.reverse_pawns_charges,
                        &self.card_runtime.stake_statuses,
                        self.card_runtime.underpromotion,
                        &self.card_runtime.vip_pending,
                    )
                )
                .as_bytes(),
            );
            hash.bytes(
                format!(
                    "{:?}",
                    (
                        &self.card_runtime.barricades,
                        &self.card_runtime.chains,
                        self.card_runtime.collapsed_files_ranks,
                        &self.card_runtime.desperado_pieces,
                        &self.card_runtime.palaces,
                        self.card_runtime.portals,
                    )
                )
                .as_bytes(),
            );
            hash.bytes(
                format!(
                    "{:?}",
                    (
                        &self.card_runtime.bunkers,
                        self.card_runtime.zugzwang,
                        self.card_runtime.bina_mate,
                        self.card_runtime.final_weapon,
                        &self.card_runtime.icbm_pending,
                        self.card_runtime.prophecy,
                        self.card_runtime.racing_king,
                        &self.card_runtime.spies,
                        self.card_runtime.ultimatum,
                        self.card_runtime.vanish,
                    )
                )
                .as_bytes(),
            );
        }
        for (index, flags) in [
            self.card_runtime.democracy,
            self.card_runtime.fast_growth,
            self.card_runtime.fianchetto,
            self.card_runtime.iron_monarch,
            self.card_runtime.pawn_conversion,
        ]
        .into_iter()
        .enumerate()
        {
            if flags != [false, false] {
                hash.u8(index as u8);
                hash.u8(u8::from(flags[0]));
                hash.u8(u8::from(flags[1]));
            }
        }
        for (piece_id, turn) in &self.card_runtime.idol_used_turn {
            hash.u32(*piece_id);
            let owner = self
                .board
                .find_by_id(*piece_id)
                .map_or(self.turn, |(_, piece)| piece.owner);
            hash.u32(turn.saturating_sub(self.turns_taken[owner.index()]));
        }
        for piece_id in &self.card_runtime.idol_rest_piece_ids {
            hash.u32(*piece_id);
        }
        for hazard in &self.card_runtime.delayed_hazards {
            hash.u8(match hazard.kind {
                DelayedHazardKind::Lightning => 0,
                DelayedHazardKind::Meteor => 1,
            });
            hash.u8(hazard.owner.index() as u8);
            hash.u8(hazard.trigger_after.index() as u8);
            for square in &hazard.squares {
                hash.u8(square.index() as u8);
            }
        }
        if self.card_runtime.skip_turns != [false, false] {
            hash.u8(u8::from(self.card_runtime.skip_turns[0]));
            hash.u8(u8::from(self.card_runtime.skip_turns[1]));
        }
        let has_0818_runtime = self.card_runtime.field_promotion != [false, false]
            || !self.card_runtime.field_promoted_piece_ids.is_empty()
            || self.card_runtime.frontline_response != [false, false]
            || self.card_runtime.relay != [false, false]
            || self.card_runtime.gomoku != [false, false]
            || self.card_runtime.magic_girl_surge != [false, false]
            || self.card_runtime.free_move_no_capture != [false, false]
            || !self.card_runtime.siren_exposure.is_empty()
            || !self.card_runtime.trickster_move_kinds.is_empty()
            || !self.card_runtime.undead_resurrections.is_empty()
            || !self.card_runtime.judgment_exiles.is_empty()
            || self.card_runtime.chaos_no_capture_until_ply > self.ply;
        if has_0818_runtime {
            hash.bytes(b"augment-chess-20260818-runtime-v1");
            hash.bytes(
                format!(
                    "{:?}",
                    (
                        self.card_runtime.field_promotion,
                        &self.card_runtime.field_promoted_piece_ids,
                        self.card_runtime.frontline_response,
                        self.card_runtime.relay,
                        self.card_runtime.gomoku,
                        self.card_runtime.magic_girl_surge,
                        self.card_runtime.free_move_no_capture,
                        &self.card_runtime.siren_exposure,
                        &self.card_runtime.trickster_move_kinds,
                        &self.card_runtime.judgment_exiles,
                    )
                )
                .as_bytes(),
            );
            for resurrection in &self.card_runtime.undead_resurrections {
                hash.u32(resurrection.piece.id);
                hash.u8(resurrection.piece.kind.stable_code());
                hash.u8(resurrection.piece.owner.index() as u8);
                hash.u16(resurrection.piece.capture_count);
                hash.u32(resurrection.due_ply.saturating_sub(self.ply));
            }
            hash.u32(
                self.card_runtime
                    .chaos_no_capture_until_ply
                    .saturating_sub(self.ply),
            );
        }
        match &self.phase {
            GamePhase::Playing => hash.u8(0),
            GamePhase::ForcedMove {
                piece_id,
                source,
                continuation,
            } => {
                hash.u8(1);
                hash.u32(*piece_id);
                match source {
                    ForcedMoveSource::Platform => hash.u8(0),
                    ForcedMoveSource::Effect(id) => {
                        hash.u8(1);
                        hash.bytes(id.as_bytes());
                    }
                }
                hash.u8(match continuation {
                    ForcedMoveContinuation::ResumeTurn => 0,
                    ForcedMoveContinuation::FinishTurn => 1,
                });
            }
            GamePhase::Draft(draft) => {
                hash.u8(2);
                hash.u8(draft.phase.index() as u8);
                hash.u8(draft.resume_turn.index() as u8);
                for card in &draft.offers {
                    hash.bytes(card.as_bytes());
                }
            }
            GamePhase::AwaitingChoice(choice) => {
                hash.u8(3);
                hash.u32(choice.id);
                hash.u8(choice.owner.index() as u8);
                hash.u32(choice.source_card_instance);
                hash.bytes(choice.handler_id.as_bytes());
                hash.u8(match choice.continuation {
                    ChoiceContinuation::ResumeTurn => 0,
                    ChoiceContinuation::FinishTurn => 1,
                });
                for option in &choice.options {
                    hash.bytes(option.as_bytes());
                }
            }
        }
        hash.finish()
    }

    /// Fingerprint of replay-relevant state. This is a debugging invariant, not a cryptographic
    /// content digest.
    pub fn fingerprint(&self) -> u64 {
        let mut hash = StableHasher::new();
        hash.u64(self.position_key());
        hash.u32(self.ply);
        hash.u32(self.quiet_ply);
        hash.u64(self.rng.state());
        hash.u32(self.turns_taken[0]);
        hash.u32(self.turns_taken[1]);
        hash.u32(self.next_piece_id);
        hash.u32(self.next_card_instance_id);
        hash.u32(self.next_choice_id);
        for status in self.card_runtime.empty_lunchboxes.values() {
            hash.u32(status.deadline_turn);
        }
        if let Some(platform) = &self.card_runtime.platform {
            hash.u32(platform.spawned_at);
            hash.u32(platform.next_at);
            hash.u32(platform.nonce);
        }
        if let Some(deathmatch) = self.deathmatch {
            hash.u32(deathmatch.started_at_shared_turn);
            hash.u32(deathmatch.quiet_full_turns);
            hash.u8(u8::from(deathmatch.progress_this_round));
        }
        match &self.status {
            GameStatus::Ongoing => hash.u8(0),
            GameStatus::Won { winner, reason } => {
                hash.u8(1);
                hash.u8(winner.index() as u8);
                hash.bytes(format!("{reason:?}").as_bytes());
            }
            GameStatus::Draw { reason } => {
                hash.u8(2);
                hash.bytes(format!("{reason:?}").as_bytes());
            }
        }
        for (position, count) in &self.repetitions {
            hash.u64(*position);
            hash.u8(*count);
        }
        hash.finish()
    }

    pub(crate) fn record_current_position(&mut self) -> u8 {
        let key = self.position_key();
        let count = self.repetitions.entry(key).or_insert(0);
        *count = count.saturating_add(1);
        *count
    }

    pub fn current_repetition_count(&self) -> u8 {
        self.repetitions
            .get(&self.position_key())
            .copied()
            .unwrap_or(0)
    }

    pub fn has_king(&self, player: Player) -> bool {
        self.board.contains(player, PieceKind::King)
    }
}

struct StableHasher(u64);

impl StableHasher {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    const fn new() -> Self {
        Self(Self::FNV_OFFSET)
    }

    fn u8(&mut self, value: u8) {
        self.0 ^= u64::from(value);
        self.0 = self.0.wrapping_mul(Self::FNV_PRIME);
    }

    fn u16(&mut self, value: u16) {
        self.bytes(&value.to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.bytes(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes(&value.to_le_bytes());
    }

    fn bytes(&mut self, values: &[u8]) {
        self.u64_len(values.len() as u64);
        for value in values {
            self.u8(*value);
        }
    }

    fn u64_len(&mut self, value: u64) {
        for byte in value.to_le_bytes() {
            self.u8(byte);
        }
    }

    const fn finish(self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BUNDLED_RULESET_ID;

    #[test]
    fn observation_v3_review_gate_covers_every_game_state_field() {
        let state = GameState::standard(BUNDLED_RULESET_ID, Mode::Standard, 7);
        let GameState {
            ruleset_id: _,
            mode: _,
            board: _,
            turn: _,
            ply: _,
            quiet_ply: _,
            actions_remaining: _,
            cards_used_this_turn: _,
            phase: _,
            status: _,
            cards: _,
            half_star_totals: _,
            turns_taken: _,
            completed_drafts: _,
            deathmatch: _,
            en_passant: _,
            applied_rules: _,
            card_runtime: _,
            rng: _,
            next_piece_id: _,
            next_card_instance_id: _,
            next_choice_id: _,
            turn_end_requested: _,
            repetitions: _,
        } = state;
    }
    #[test]
    fn same_seed_and_position_have_same_fingerprint() {
        let left = GameState::standard(BUNDLED_RULESET_ID, Mode::Standard, 7);
        let right = GameState::standard(BUNDLED_RULESET_ID, Mode::Standard, 7);
        assert_eq!(left.fingerprint(), right.fingerprint());
    }

    #[test]
    fn rng_state_is_part_of_full_fingerprint_but_not_position_key() {
        let left = GameState::standard(BUNDLED_RULESET_ID, Mode::Standard, 7);
        let mut right = left.clone();
        right.rng_mut().next_u64();
        assert_eq!(left.position_key(), right.position_key());
        assert_ne!(left.fingerprint(), right.fingerprint());
    }
}
