//! Deterministic engine kernel for Augment Chess.
//!
//! The crate intentionally has no UI, networking, wall-clock, or training-framework
//! dependencies. A complete game is a pure transition from `(GameState, Action)` to the next
//! state plus an ordered event log.

mod action;
mod cards;
mod effect;
mod engine;
mod event;
mod model;
mod notation;
mod rng;
mod ruleset;
mod state;

pub use action::{Action, ActionTarget, MoveAction};
pub use cards::{
    CardActivation, CardPhase, EMPTY_LUNCHBOX, GENEVA_CONVENTION, IMPLEMENTED_CARDS,
    ImplementedCardDefinition, KING_OF_THE_HILL, PARRY, PLATFORM, implemented_card,
};
pub use effect::{EffectCommand, EffectHandler, EffectRegistry, EffectRegistryError};
pub use engine::{Engine, EngineError};
pub use event::{Event, Transition};
pub use model::{Board, Piece, PieceKind, Player, Square, SquareParseError};
pub use notation::{
    NotationCatalog, NotationEntry, NotationError, NotationParsing, NotationSource,
};
pub use rng::DeterministicRng;
pub use ruleset::{
    Artifacts, BaseRules, Coverage, DraftConfig, ModeConfig, RulesetError, RulesetManifest,
    SourceLinks,
};
pub use state::{
    CardInstance, CardRuntimeState, DeathmatchState, DraftPhase, DraftState, DrawReason,
    EmptyLunchboxStatus, EnPassantState, ForcedMoveSource, GamePhase, GameState, GameStatus, Mode,
    PendingChoice, PlatformRuleState, WinReason,
};

/// Ruleset bundled into this build and used by the first engine milestone.
pub const BUNDLED_RULESET_ID: &str = "augment_chess_20260807";
pub const SUPPORTED_SCHEMA_VERSION: u32 = 1;
pub const SUPPORTED_ENGINE_API_VERSION: u32 = 1;
