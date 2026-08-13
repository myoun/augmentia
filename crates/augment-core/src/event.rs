use crate::{Action, GameState, GameStatus, Piece, Player, Square};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Event {
    ActionApplied {
        actor: Player,
        action: Action,
    },
    PieceCaptured {
        actor: Player,
        piece: Piece,
        at: Square,
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
