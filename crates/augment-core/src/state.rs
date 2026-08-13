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
    },
    Draft {
        round: u16,
        offers: Vec<Vec<String>>,
    },
    AwaitingChoice(PendingChoice),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ForcedMoveSource {
    Platform,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PendingChoice {
    pub id: u32,
    pub owner: Player,
    pub source_card_instance: u32,
    pub options: Vec<String>,
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
    CardEffect(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DrawReason {
    EqualStarsOnRepetition,
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

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CardRuntimeState {
    pub hill_king: [bool; 2],
    pub geneva_convention: [bool; 2],
    pub parry_piece_ids: BTreeSet<u32>,
    pub empty_lunchboxes: BTreeMap<u32, EmptyLunchboxStatus>,
    pub platform: Option<PlatformRuleState>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GameState {
    pub ruleset_id: String,
    pub mode: Mode,
    pub board: Board,
    pub turn: Player,
    pub ply: u32,
    pub quiet_ply: u32,
    pub phase: GamePhase,
    pub status: GameStatus,
    pub cards: Vec<CardInstance>,
    pub half_star_totals: [u16; 2],
    pub turns_taken: [u32; 2],
    pub applied_rules: Vec<String>,
    pub card_runtime: CardRuntimeState,
    rng: DeterministicRng,
    next_piece_id: u32,
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
            phase: GamePhase::Playing,
            status: GameStatus::Ongoing,
            cards: Vec::new(),
            half_star_totals: [0, 0],
            turns_taken: [0, 0],
            applied_rules: Vec::new(),
            card_runtime: CardRuntimeState::default(),
            rng: DeterministicRng::new(seed),
            next_piece_id,
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

    pub fn shared_turn_count(&self) -> u32 {
        self.turns_taken[0].min(self.turns_taken[1])
    }

    pub fn position_key(&self) -> u64 {
        let mut hash = StableHasher::new();
        hash.bytes(self.ruleset_id.as_bytes());
        hash.bytes(self.mode.id().as_bytes());
        hash.u8(self.turn.index() as u8);

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
        match &self.phase {
            GamePhase::Playing => hash.u8(0),
            GamePhase::ForcedMove { piece_id, source } => {
                hash.u8(1);
                hash.u32(*piece_id);
                hash.u8(match source {
                    ForcedMoveSource::Platform => 0,
                });
            }
            GamePhase::Draft { round, offers } => {
                hash.u8(2);
                hash.u16(*round);
                for bundle in offers {
                    for card in bundle {
                        hash.bytes(card.as_bytes());
                    }
                    hash.u8(0xff);
                }
            }
            GamePhase::AwaitingChoice(choice) => {
                hash.u8(3);
                hash.u32(choice.id);
                hash.u8(choice.owner.index() as u8);
                hash.u32(choice.source_card_instance);
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
        for status in self.card_runtime.empty_lunchboxes.values() {
            hash.u32(status.deadline_turn);
        }
        if let Some(platform) = &self.card_runtime.platform {
            hash.u32(platform.spawned_at);
            hash.u32(platform.next_at);
            hash.u32(platform.nonce);
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
