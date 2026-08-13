use crate::{Board, DeterministicRng, PieceKind, Player, Square};
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

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CardRuntimeState {
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
    pub idol_used_turn: BTreeMap<u32, u32>,
    pub idol_rest_piece_ids: BTreeSet<u32>,
    pub delayed_hazards: Vec<DelayedHazard>,
    pub skip_turns: [bool; 2],
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
