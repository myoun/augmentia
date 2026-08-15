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
    ALEKHINE_MACHINE_GUN, AMAZON, APPRENTICE_KNIGHTS, ARMISTICE, ASSASSIN, BABY_BEAR,
    BACKWARD_KNIGHT, BARRICADE, BASIC_TRAINING, BIG_ROOK, BINA_MATE, BLACK_MAGIC, BLUE_JEANS,
    BREAKTHROUGH_ORDER, BRIBE, CALLING_CARD, CANCELING, CASTLING, CHAIN, CHAMELEON_MUTATION,
    CHARGE, CHECKER, CHIMERA, CLEANUP_SACRIFICE, COLLAPSE, CONSCRIPTION, CONSTITUTIONAL_MONARCHY,
    CONVERSION, CORNER_KICK, CORONATION, CardActivation, CardPhase, DEATH_SQUAD, DEMOCRACY,
    DESPERADO, DICE, DISARM, DRAGON, DUTCH, EAGLE, EARLY_PROMOTION, EASTERN_POLICY,
    ELEPHANT_ESCAPE, EMERGENCY_EVACUATION, EMPTY_LUNCHBOX, EN_PASSANT_BANG, ENCOURAGEMENT, EVASION,
    EXHAUSTION, EXILE, FANATICAL_RITUAL, FAST_GROWTH, FEUDAL_CONTRACT, FIANCHETTO, FILE_SURGE,
    FINAL_WEAPON, FLEETING_DREAM, FREEZE, FRENZY, GALE, GENEVA_CONVENTION, GHOST, GRASSHOPPER,
    GUARD, HALLUCINATION, HERALD, HOLDOUT, HOMECOMING, HOOK, HORDE, HORSE_RIDING, ICBM, ICE_SHEET,
    IDOL, IMPERIAL_STUDIES, IMPLEMENTED_CARDS, INERTIA, INITIATIVE, INJURY, IRON_MONARCH,
    ImplementedCardDefinition, JESTER, JUDGMENT, KING_OF_THE_HILL, KNIGHTMASTER, KNIGHTMATE,
    LAST_RESISTANCE, LAST_STAND, LEAP, LOBSTER, LOCAL_CONSCRIPTION, LOG, LONDON_SYSTEM, LOYALIST,
    MAD_HORSE, MARTYRDOM, MERCHANT_GUILD, MISSIONARY, MISTAKE_CARD, MONGOLIAN_GAMBIT, MOVING,
    NECROMANCY, ORDINATION, OTHELLO, OTHERWORLD, OVERWHELM, PALACE, PANIC, PARRY, PAWN_CONVERSION,
    PAWN_STORM, PEGASUS, PLATFORM, POISONED_PAWN, PORTAL_GUN, PREMOVE, PROMOTION_RUSH, PROPHECY,
    QUANTUM_MECHANICS, QUEEN_AFTERIMAGE, QUEEN_CAVALRY, QUEENS_GAMBIT, QXE1, RACING_KING,
    RADICAL_CHARGE, RANDOM_ROULETTE, REAPER, REFORMATION, RELIGIOUS_VICTORY, REPOSITION, RETREAT,
    REVELATION, REVERSE_PAWNS, ROOK_LIFT, ROYAL_COMMAND, ROYAL_SHIELD, SACRIFICE, SCARECROW,
    SCHRODINGER_PAWNS, SEVERANCE, SNIPE, SOCIALISM, SPRINT, SPY, STAKE, STANDARD_BEARER, STEALTH,
    SUBMERGE, SUBSTITUTION, SUICIDE_BOMBER, SUMMON_COLOSSUS, SWITCHEROO, TAUNT, TRAITOR,
    TROJAN_HORSE, TROLLEY, TWINS, ULTIMATUM, UNDERGROUND_BUNKER, UNDERPROMOTION, VANISH, VIP,
    VORTEX, WINDMILL, WITCH_TRIAL, WIZARD, ZUGZWANG, implemented_card,
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
    BribeRevert, CardInstance, CardRuntimeState, ChoiceContinuation, DeathmatchState,
    DelayedHazard, DelayedHazardKind, DiceLockStatus, DraftPhase, DraftState, DrawReason,
    EmptyLunchboxStatus, EnPassantState, FeudalContract, ForcedMoveContinuation, ForcedMoveSource,
    FrozenStatus, GamePhase, GameState, GameStatus, InitiativeStatus, LargePieceStatus, LogStatus,
    Mode, NecromancyRevert, PendingChoice, PendingIcbm, PendingOtherworld, PendingPanic,
    PendingPremove, PendingScarecrow, PendingVip, PlatformRuleState, RepositionStatus,
    RoyalCommandStatus, TimedPieceStatus, TrolleyDilemma, TwinBond, WinReason, WindmillMode,
};

/// Ruleset bundled into this build and used by the first engine milestone.
pub const BUNDLED_RULESET_ID: &str = "augment_chess_20260807";
pub const SUPPORTED_SCHEMA_VERSION: u32 = 1;
pub const SUPPORTED_ENGINE_API_VERSION: u32 = 1;
