use crate::{Action, DraftPhase, GameState, GameStatus, Piece, Player, Square};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Event {
    DraftStarted {
        phase: DraftPhase,
        player: Player,
        offers: Vec<String>,
    },
    DraftPicked {
        phase: DraftPhase,
        player: Player,
        offer_index: u16,
        definition_id: String,
    },
    DraftSkipped {
        phase: DraftPhase,
        player: Player,
    },
    DraftCompleted {
        phase: DraftPhase,
        resume_turn: Player,
    },
    DeathmatchStarted {
        shared_turn: u32,
        interval_full_turns: u32,
    },
    TurnStarted {
        player: Player,
        actions_remaining: u16,
    },
    TurnEnding {
        player: Player,
    },
    TurnEnded {
        player: Player,
    },
    ChoiceRequested {
        choice_id: u32,
        owner: Player,
        handler_id: String,
        options: Vec<String>,
    },
    ChoiceResolved {
        choice_id: u32,
        owner: Player,
        option_index: u16,
    },
    CardGranted {
        owner: Player,
        instance_id: u32,
        definition_id: String,
    },
    CardActivated {
        owner: Player,
        instance_id: u32,
        definition_id: String,
    },
    RuleApplied {
        definition_id: String,
    },
    CardUsed {
        owner: Player,
        instance_id: u32,
        definition_id: String,
    },
    RandomSample {
        purpose: String,
        /// Integer mantissa representing a browser-style value in `[0, 2^53)`.
        roll_53: u64,
    },
    PlatformSpawned {
        square: Option<Square>,
        shared_turn: u32,
        nonce: u32,
    },
    ForcedMoveStarted {
        player: Player,
        piece_id: u32,
        source: String,
    },
    ActionApplied {
        actor: Player,
        action: Action,
    },
    PieceCaptured {
        actor: Player,
        piece: Piece,
        at: Square,
    },
    PieceRemoved {
        actor: Player,
        piece: Piece,
        at: Square,
        cause: String,
    },
    EffectApplied {
        source_card_instance: u32,
        command: String,
    },
    TurnChanged {
        player: Player,
    },
    GameEnded {
        status: GameStatus,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Transition {
    pub state: GameState,
    pub events: Vec<Event>,
}
