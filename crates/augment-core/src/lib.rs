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
    ACCELERATION, ALEKHINE_MACHINE_GUN, AMAZON, APPRENTICE_KNIGHTS, ARMISTICE, ASSASSIN, BABY_BEAR,
    BACKWARD_KNIGHT, BARRICADE, BASIC_TRAINING, BERSERKER, BIG_ROOK, BINA_MATE, BLACK_BOX,
    BLACK_HOLE, BLACK_MAGIC, BLUE_JEANS, BREAKTHROUGH_ORDER, BRIBE, CALLING_CARD, CAMOUFLAGE_COLOR,
    CANCELING, CASTLING, CHAIN, CHAMELEON_MUTATION, CHARGE, CHECKER, CHESS_960, CHESS_344200,
    CHESS_N_POW_30, CHIMERA, CLEANUP, CLEANUP_SACRIFICE, CLONE, COLLAPSE, CONSCRIPTION,
    CONSTITUTIONAL_MONARCHY, CONVERSION, CONVEYOR, COOL_GUY, CORNER_KICK, CORONATION, CROWN,
    CardActivation, CardPhase, DEATH_SQUAD, DEMOCRACY, DESPERADO, DIAGONAL_CHESS, DICE, DISARM,
    DRAGON, DUTCH, EAGLE, EARLY_PROMOTION, EASTERN_POLICY, ELEPHANT_ESCAPE, EMERGENCY_EVACUATION,
    EMPTY_LUNCHBOX, EN_PASSANT_BANG, ENCOURAGEMENT, EVASION, EXHAUSTION, EXILE, FANATICAL_RITUAL,
    FAST_GROWTH, FEUDAL_CONTRACT, FIANCHETTO, FIELD_PROMOTION, FILE_SURGE, FINAL_WEAPON,
    FLEETING_DREAM, FOOTBALL, FREEZE, FRENZY, FRONTLINE_RESPONSE, GALE, GENEVA_CONVENTION, GHOST,
    GOMOKU, GRASSHOPPER, GUARD, HALLUCINATION, HERALD, HIGH_GROUND, HIGHWAY, HOLDOUT, HOMECOMING,
    HOOK, HORDE, HORSE_RIDING, HYPOCRISY, ICBM, ICE_SHEET, IDOL, IMPERIAL_STUDIES,
    IMPLEMENTED_CARDS, INERTIA, INITIATIVE, INJURY, INSIGHT, IRON_MONARCH,
    ImplementedCardDefinition, JESTER, JOKER, JUDGMENT, KING_OF_THE_HILL, KNIGHTMASTER, KNIGHTMATE,
    LAST_RESISTANCE, LAST_STAND, LEAP, LOBSTER, LOCAL_CONSCRIPTION, LOG, LONDON_SYSTEM, LOYALIST,
    MACHO_CHESS, MAD_HORSE, MAGIC_GIRL, MARTYRDOM, MERCHANT_GUILD, MISSIONARY, MISTAKE,
    MISTAKE_CARD, MONGOLIAN_GAMBIT, MONOCHROME_CHESS, MONSTER_RULE, MOVING, NECROMANCY, ORDINATION,
    OTHELLO, OTHERWORLD, OVERWHELM, PALACE, PANIC, PARRY, PAWN_CONVERSION, PAWN_STORM, PEGASUS,
    PERIODIC_COLLAPSE, PLATFORM, POISONED_PAWN, PORTAL, PORTAL_GUN, PREMOVE, PROMOTION_RUSH,
    PROPHECY, QUANTUM_MECHANICS, QUEEN_AFTERIMAGE, QUEEN_CAVALRY, QUEENS_GAMBIT, QXE1, RACING_KING,
    RADICAL_CHARGE, RANDOM_ROULETTE, REAPER, RECYCLING, REFORMATION, RELAY, RELIGIOUS_VICTORY,
    REPLAY, REPOSITION, RETREAT, REVELATION, REVERSE_PAWNS, ROOK_LIFT, ROYAL_COMMAND, ROYAL_SHIELD,
    RULE_BOMBS, RULE_TICKET, SACRIFICE, SATURATION, SCARECROW, SCHRODINGER_PAWNS, SEVERANCE,
    SIEGE_RAM, SIREN, SLIME, SNIPE, SOCIALISM, SPRINT, SPY, STAKE, STANDARD_BEARER, STEALTH,
    SUBMERGE, SUBSTITUTION, SUICIDE_BOMBER, SUMMON_COLOSSUS, SUSPICIOUS_POTION, SWITCHEROO, TAUNT,
    TRAITOR, TRANSCENDENCE, TRICKSTER, TROJAN_HORSE, TROLLEY, TWINS, ULTIMATUM, UNDEAD,
    UNDERGROUND_BUNKER, UNDERPROMOTION, VANISH, VIP, VORTEX, WHITE_BOX, WINDMILL, WINTER_KINGDOM,
    WITCH_TRIAL, WIZARD, ZUGZWANG, implemented_card,
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
    FrozenStatus, GamePhase, GameState, GameStatus, InitiativeStatus, JudgmentExile,
    LargePieceStatus, LogStatus, Mode, MoveReplayState, NecromancyRevert, PendingChoice,
    PendingIcbm, PendingOtherworld, PendingPanic, PendingPremove, PendingScarecrow, PendingVip,
    PlatformRuleState, ReplayCell, RepositionStatus, RoyalCommandStatus, TimedPieceStatus,
    TrolleyDilemma, TwinBond, UndeadResurrection, WinReason, WindmillMode,
};

/// Ruleset bundled into this build and used by the first engine milestone.
pub const BUNDLED_RULESET_ID: &str = "augment_chess_20260818";
pub const SUPPORTED_SCHEMA_VERSION: u32 = 1;
pub const SUPPORTED_ENGINE_API_VERSION: u32 = 1;
