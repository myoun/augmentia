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
    BACKWARD_KNIGHT, BARRICADE, BASIC_TRAINING, BERSERKER, BIG_BISHOP, BIG_ROOK, BINA_MATE,
    BLACK_BOX, BLACK_HOLE, BLACK_MAGIC, BLUE_JEANS, BREAKTHROUGH_ORDER, BRIBE, BRUTUS,
    CALLING_CARD, CAMOUFLAGE_COLOR, CAMPFIRE, CANCELING, CAPTURE_THE_FLAG, CASTLING, CHAIN,
    CHAMELEON_MUTATION, CHARGE, CHECKER, CHESS_960, CHESS_344200, CHESS_N_POW_30, CHIMERA, CLEANUP,
    CLEANUP_SACRIFICE, CLOCKWORK, CLONE, COLLAPSE, CONSCRIPTION, CONSTITUTIONAL_MONARCHY,
    CONVERSION, CONVEYOR, COOL_GUY, CORNER_KICK, CORONATION, CROWN, CardActivation, CardPhase,
    DEATH_SQUAD, DEMOCRACY, DESPERADO, DIAGONAL_CHESS, DICE, DISARM, DISASSEMBLY, DRAGON, DUTCH,
    EAGLE, EARLY_PROMOTION, EASTERN_POLICY, ELEPHANT_ESCAPE, EMERGENCY_EVACUATION, EMPTY_LUNCHBOX,
    EN_PASSANT_BANG, ENCOURAGEMENT, EVASION, EXHAUSTION, EXILE, EXTINCTION, FALSE_START,
    FANATICAL_RITUAL, FAST_GROWTH, FEUDAL_CONTRACT, FIANCHETTO, FIELD_PROMOTION, FILE_SURGE,
    FINAL_WEAPON, FLEETING_DREAM, FOOTBALL, FREEZE, FRENZY, FRONTLINE_RESPONSE, GALE,
    GENEVA_CONVENTION, GHOST, GOMOKU, GRASSHOPPER, GUARD, HALLUCINATION, HEDGEHOG, HERALD,
    HIGH_GROUND, HIGHLANDER, HIGHWAY, HOLDOUT, HOMECOMING, HOOK, HORDE, HORSE_RIDING, HYPOCRISY,
    ICBM, ICE_SHEET, IDOL, IMPERIAL_STUDIES, IMPLEMENTED_CARDS, INERTIA, INFILTRATION, INITIATIVE,
    INJURY, INSIGHT, IRON_MONARCH, ImplementedCardDefinition, JESTER, JOKER, JUDGMENT, KILLER_KING,
    KING_OF_THE_HILL, KNIGHTMASTER, KNIGHTMATE, LAST_RESISTANCE, LAST_STAND, LEAP, LOBSTER,
    LOCAL_CONSCRIPTION, LOCUST_SWARM, LOG, LONDON_SYSTEM, LONG_EN_PASSANT, LOYALIST, MACHO_CHESS,
    MAD_HORSE, MAGIC_GIRL, MAJESTY, MARTYRDOM, MERCHANT_GUILD, METAL, MIRACLE, MISSIONARY, MISTAKE,
    MISTAKE_CARD, MONGOLIAN_GAMBIT, MONOCHROME_CHESS, MONSTER_RULE, MOVING, MUTATION, NECROMANCY,
    NULLIFICATION, OCTOPUS, ORDINATION, OTHELLO, OTHERWORLD, OUTPOST, OVERTAKE, OVERWHELM, PALACE,
    PALADIN, PANIC, PARROT, PARRY, PAWN_CONVERSION, PAWN_STORM, PEGASUS, PERIODIC_COLLAPSE,
    PLATFORM, POISONED_PAWN, PORTAL, PORTAL_GUN, PREMOVE, PRINCESS, PROFICIENCY, PROMOTION_RUSH,
    PROPHECY, QUANTUM_MECHANICS, QUEEN_AFTERIMAGE, QUEEN_CAVALRY, QUEENS_GAMBIT, QXE1, RACING_KING,
    RADICAL_CHARGE, RANDOM_ROULETTE, REAPER, RECURRENCE, RECYCLING, REFORMATION, RELAY,
    RELIGIOUS_VICTORY, REPLAY, REPOSITION, RESOLVE, RETREAT, REVELATION, REVERSAL, REVERSE_PAWNS,
    ROOK_LIFT, ROYAL_COMMAND, ROYAL_SHIELD, RULE_BOMBS, RULE_TICKET, SACRIFICE, SATURATION,
    SCARECROW, SCHRODINGER_PAWNS, SEVERANCE, SIEGE_RAM, SIREN, SLIME, SNIPE, SOCIALISM, SPRINT,
    SPY, STAKE, STANDARD_BEARER, STEALTH, SUBMERGE, SUBSTITUTION, SUICIDE_BOMBER, SUMMON_COLOSSUS,
    SUSPICIOUS_POTION, SWITCHEROO, SYMMETRY, TAUNT, THIEF, TRAITOR, TRANSCENDENCE, TRICKSTER,
    TROJAN_HORSE, TROLLEY, TWINS, ULTIMATUM, UNDEAD, UNDERGROUND_BUNKER, UNDERPROMOTION, VANGUARD,
    VANISH, VIP, VORTEX, WHITE_BOX, WINDMILL, WINTER_KINGDOM, WITCH_TRIAL, WIZARD, ZUGZWANG,
    implemented_card,
};
pub use effect::{
    ChoiceHandler, ChoiceRegistry, ChoiceRegistryError, EffectCommand, EffectHandler,
    EffectRegistry, EffectRegistryError,
};
pub use engine::{Engine, EngineError, SearchChanceContext, SearchTransition};
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
    EmptyLunchboxStatus, EnPassantState, FIELD_PROMOTION_CHOICE_PREFIX, FeudalContract,
    ForcedMoveContinuation, ForcedMoveSource, FrozenStatus, GamePhase, GameState, GameStatus,
    InitiativeStatus, JudgmentExile, LargePieceStatus, LogStatus, MIDDLE_EXTRA_MOVE_SOURCE, Mode,
    MoveReplayState, NecromancyRevert, ParrotMemoryStatus, PendingChoice, PendingIcbm,
    PendingLobster, PendingOtherworld, PendingPanic, PendingPortal, PendingPremove,
    PendingRecurrence, PendingScarecrow, PendingVip, PlatformRuleState, ReplayCell,
    RepositionStatus, RoyalCommandStatus, TimedPieceStatus, TrolleyDilemma, TwinBond,
    UndeadResurrection, WinReason, WindmillMode,
};
pub use state::{CaptureTheFlagState, FlagOccupation};

/// Ruleset bundled into this build and used by the first engine milestone.
pub const BUNDLED_RULESET_ID: &str = "augment_chess_20260818";
pub const SUPPORTED_SCHEMA_VERSION: u32 = 1;
pub const SUPPORTED_ENGINE_API_VERSION: u32 = 1;
