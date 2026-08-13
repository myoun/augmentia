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

pub use action::{Action, ActionTarget, MoveAction, PieceAbility};
pub use cards::{
    AMAZON, ASSASSIN, CLEANUP_SACRIFICE, CONSTITUTIONAL_MONARCHY, CardActivation, CardPhase, DICE,
    DISARM, DRAGON, EAGLE, EARLY_PROMOTION, EASTERN_POLICY, EMPTY_LUNCHBOX, FREEZE,
    GENEVA_CONVENTION, GRASSHOPPER, HERALD, HOOK, IDOL, IMPLEMENTED_CARDS, INITIATIVE,
    ImplementedCardDefinition, JESTER, JUDGMENT, KING_OF_THE_HILL, KNIGHTMASTER,
    LOCAL_CONSCRIPTION, LOG, MISSIONARY, ORDINATION, PARRY, PEGASUS, PLATFORM, REAPER, REFORMATION,
    REPOSITION, REVELATION, SPRINT, STANDARD_BEARER, WINDMILL, WIZARD, implemented_card,
};
pub use effect::{
    ChoiceHandler, ChoiceRegistry, ChoiceRegistryError, EffectCommand, EffectHandler,
    EffectRegistry, EffectRegistryError,
};
pub use engine::{Engine, EngineError};
pub use event::{Event, Transition};
pub use model::{Board, Piece, PieceKind, Player, Square, SquareParseError};
pub use notation::{
    NotationCatalog, NotationEntry, NotationError, NotationParsing, NotationSource,
};
pub use rng::DeterministicRng;
pub use ruleset::{
    Artifacts, BaseRules, Coverage, DraftConfig, ModeConfig, OpeningRuleConfig, RulesetError,
    RulesetManifest, SourceLinks,
};
pub use state::{
    CardInstance, CardRuntimeState, ChoiceContinuation, DeathmatchState, DelayedHazard,
    DelayedHazardKind, DiceLockStatus, DraftPhase, DraftState, DrawReason, EmptyLunchboxStatus,
    EnPassantState, ForcedMoveContinuation, ForcedMoveSource, FrozenStatus, GamePhase, GameState,
    GameStatus, InitiativeStatus, LogStatus, Mode, PendingChoice, PlatformRuleState,
    RepositionStatus, WinReason, WindmillMode,
};

/// Ruleset bundled into this build and used by the first engine milestone.
pub const BUNDLED_RULESET_ID: &str = "augment_chess_20260807";
pub const SUPPORTED_SCHEMA_VERSION: u32 = 1;
pub const SUPPORTED_ENGINE_API_VERSION: u32 = 1;
