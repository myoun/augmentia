use crate::{Action, GameState, GameStatus, Piece, Player, Square};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Event {
    CardGranted {
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
