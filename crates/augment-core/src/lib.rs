//! Deterministic engine kernel for Augment Chess.
//!
//! The crate intentionally has no UI, networking, wall-clock, or training-framework
//! dependencies. A complete game is a pure transition from `(GameState, Action)` to the next
//! state plus an ordered event log.

mod action;
mod card;
mod card_eligibility;
mod card_ids;
mod compat;
mod effect;
mod engine;
mod event;
mod latest;
mod model;
mod notation;
mod rng;
mod ruleset;
mod state;

pub use action::{Action, ActionTarget, MoveAction, PieceAbility};
pub use card::{CardActivation, CardPhase, ImplementedCardDefinition, ruleset_card};
pub use card_eligibility::{
    AcquisitionContext, CardEligibility, EligibilityReason, EligibilityReasonCode,
    canonical_card_id, check_card_eligibility, mutually_exclusive_cards,
};
pub use card_ids::{
    ACCELERATION, ALEKHINE_MACHINE_GUN, AMAZON, APPRENTICE_KNIGHTS, ARCHBISHOP, ARMISTICE,
    ASCENSION, ASSASSIN, ASSEMBLY, BABY_BEAR, BACKWARD_KNIGHT, BARRICADE, BARRIER, BASIC_TRAINING,
    BEACON, BERSERKER, BIG_BISHOP, BIG_ROOK, BINA_MATE, BISHOP_INFILTRATION, BLACK_BOX, BLACK_HOLE,
    BLACK_MAGIC, BLOCKADE, BLUE_JEANS, BRAINWASH, BREAKTHROUGH_ORDER, BRIBE, BRUTUS, CALLING_CARD,
    CAMOUFLAGE_COLOR, CAMPFIRE, CANCELING, CAPTURE_THE_FLAG, CASTLING, CHAIN, CHAMELEON_MUTATION,
    CHANCELLOR, CHARGE, CHECKER, CHESS_960, CHESS_344200, CHESS_N_POW_30, CHIMERA, CLEANUP,
    CLEANUP_SACRIFICE, CLOCKWORK, CLONE, COLLAPSE, CONSCRIPTION, CONSTITUTIONAL_MONARCHY,
    CONVERSION, CONVEYOR, COOL_GUY, CORNER_KICK, CORONATION, COUNTERATTACK, CROWN, D4, DEATH_SQUAD,
    DEMOCRACY, DESPERADO, DIAGONAL_CHESS, DICE, DISARM, DISASSEMBLY, DON_QUIXOTE, DRAGON, DUTCH,
    E4, EAGLE, EARLY_PROMOTION, EASTERN_POLICY, ELEPHANT_ESCAPE, EMERGENCY_EVACUATION,
    EMPTY_LUNCHBOX, EN_PASSANT_BANG, ENCOURAGEMENT, EVASION, EXHAUSTION, EXILE, EXTINCTION,
    FALSE_START, FANATICAL_RITUAL, FAST_GROWTH, FEUDAL_CONTRACT, FIANCHETTO, FIELD_PROMOTION,
    FILE_CONVEYOR, FILE_SURGE, FINAL_WEAPON, FLEETING_DREAM, FOOTBALL, FREEZE, FRENZY,
    FRONTLINE_RESPONSE, GALE, GENEVA_CONVENTION, GHOST, GOMOKU, GRAPPLER, GRASSHOPPER, GREEK_GIFT,
    GRUDGE, GUARD, HALLUCINATION, HEDGEHOG, HERALD, HIGH_GROUND, HIGHLANDER, HIGHWAY, HOLDOUT,
    HOMECOMING, HOOK, HOPE, HORDE, HORSE_RIDING, HYPOCRISY, ICBM, ICE_KING, ICE_SHEET, IDOL,
    IMPERIAL_STUDIES, INERTIA, INFILTRATION, INITIATIVE, INJURY, INSIGHT, IQP, IRON_MONARCH,
    JESTER, JOKER, JUDGMENT, KILLER, KILLER_KING, KING_OF_THE_HILL, KNIGHTMASTER, KNIGHTMATE,
    LAST_RESISTANCE, LAST_STAND, LEAP, LOBSTER, LOCAL_CONSCRIPTION, LOCUST_SWARM, LOG,
    LONDON_SYSTEM, LONG_EN_PASSANT, LOYALIST, MACHO_CHESS, MAD_HORSE, MAGIC_GIRL, MAJESTY, MAN,
    MARTYRDOM, MEDIUM, MERCHANT_GUILD, METAL, MIRACLE, MISSIONARY, MISTAKE, MISTAKE_CARD,
    MONGOLIAN_GAMBIT, MONOCHROME_CHESS, MONSTER_RULE, MOVING, MUTATION, NECROMANCY, NULLIFICATION,
    OCTOPUS, ORDINATION, OTHELLO, OTHERWORLD, OUTPOST, OVERTAKE, OVERWHELM, PALACE, PALADIN, PANIC,
    PARROT, PARRY, PAWN_CHAIN, PAWN_CONVERSION, PAWN_STORM, PEGASUS, PENTA_CHECK,
    PERIODIC_COLLAPSE, PLATFORM, POISONED_PAWN, PORTAL, PORTAL_GUN, PREMOVE, PRINCESS, PROFICIENCY,
    PROMOTION_RUSH, PROPHECY, QUANTUM_MECHANICS, QUEEN_AFTERIMAGE, QUEEN_CAVALRY, QUEENS_GAMBIT,
    QXE1, RACING_KING, RADICAL_CHARGE, RANDOM_ROULETTE, REAPER, RECURRENCE, RECYCLING, REFORMATION,
    RELAY, RELIGIOUS_VICTORY, REPLAY, REPOSITION, RESOLVE, RETREAT, REVELATION, REVERSAL,
    REVERSE_PAWNS, REVOLVING_DOOR, ROCKET, ROLLER, ROOK_LIFT, ROYAL_COMMAND, ROYAL_SHIELD,
    RULE_BOMBS, RULE_TICKET, SACRIFICE, SATURATION, SCARECROW, SCHRODINGER_PAWNS, SEAL, SEVERANCE,
    SIEGE_RAM, SIREN, SLIME, SNIPE, SOCIALISM, SOLIDARITY, SPRINT, SPY, STAKE, STANDARD_BEARER,
    STEALTH, SUBMERGE, SUBSTITUTION, SUICIDE_BOMBER, SUMMON_COLOSSUS, SUSPICIOUS_POTION,
    SWITCHEROO, SYMMETRY, SYNCHRONIZATION, TABOO, TAUNT, THIEF, TRAITOR, TRANSCENDENCE,
    TRANSMISSION_TOWER, TRICKSTER, TROJAN_HORSE, TROLLEY, TWINS, ULTIMATUM, UNDEAD,
    UNDERGROUND_BUNKER, UNDERPROMOTION, VANGUARD, VANISH, VIGILANCE, VIP, VORTEX, WANTED,
    WHITE_BOX, WINDMILL, WINTER_KINGDOM, WITCH_TRIAL, WIZARD, ZUGZWANG,
};
pub use compat::card_definitions::{IMPLEMENTED_CARDS, implemented_card};
pub use effect::{
    ChoiceHandler, ChoiceRegistry, ChoiceRegistryError, EffectCommand, EffectHandler,
    EffectRegistry, EffectRegistryError,
};
pub use engine::{Engine, EngineError, LegalActionBatch, SearchChanceContext, SearchTransition};
pub use event::{Event, Transition};
pub use latest::{LatestCard, latest_catalog};
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
    AutomaticDrawState, BribeRevert, CardInstance, CardRuntimeState, ChoiceContinuation,
    DeathmatchState, DelayedHazard, DelayedHazardKind, DiceLockStatus, DraftPhase, DraftState,
    DrawReason, EmptyLunchboxStatus, EnPassantState, FIELD_PROMOTION_CHOICE_PREFIX, FeudalContract,
    ForcedMoveContinuation, ForcedMoveSource, FrozenStatus, GamePhase, GameState, GameStatus,
    InitiativeStatus, JudgmentExile, LargePieceStatus, LogStatus, MIDDLE_EXTRA_MOVE_SOURCE, Mode,
    MoveReplayState, NecromancyRevert, ParrotMemoryStatus, PendingChoice, PendingIcbm,
    PendingLobster, PendingOtherworld, PendingPanic, PendingPortal, PendingPremove,
    PendingRecurrence, PendingRuleTicket, PendingScarecrow, PendingVip, PlatformRuleState,
    ReplayCell, RepositionStatus, RoyalCommandStatus, TimedPieceStatus, TrolleyDilemma, TwinBond,
    UndeadResurrection, WinReason, WindmillMode,
};
pub use state::{CaptureTheFlagState, FlagOccupation};

/// Ruleset bundled into this build and used by the first engine milestone.
pub const BUNDLED_RULESET_ID: &str = "augment_chess_20260818";
/// Latest captured rules, with an explicitly limited executable card profile.
pub const LATEST_RULESET_ID: &str = "augment_chess_20261017";
pub const SUPPORTED_SCHEMA_VERSION: u32 = 1;
pub const SUPPORTED_ENGINE_API_VERSION: u32 = 1;
