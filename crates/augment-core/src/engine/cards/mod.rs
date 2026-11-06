//! Current executable card registry. Common turn/action framing stays in Engine.
use super::*;
use crate::card_eligibility::{AcquisitionContext, CardEligibility};
use crate::*;
pub use ascension::PendingAscension;
use eligibility_support::*;
pub use penta_check::PentaCheckProgress;
pub use seal::PendingSeal;
pub use taboo::PendingTaboo;
mod alekhine_machine_gun;
mod apprentice_knights;
mod basic_training;
mod big_rook;
mod bishop_infiltration;
mod blockade;
mod calling_card;
mod castling;
mod chameleon_mutation;
mod chimera;
mod counterattack;
mod d4;
mod dice;
mod disarm;
mod dutch;
mod e4;
mod elephant_escape;
mod eligibility_support;
pub(super) mod emergency_evacuation;
mod en_passant_bang;
mod encouragement;
mod evasion;
mod fanatical_ritual;
mod feudal_contract;
mod file_conveyor;
mod freeze;
mod frontline_response;
mod guard;
mod herald;
mod holdout;
mod horde;
mod ice_king;
mod ice_sheet;
mod inertia;
mod iron_monarch;
mod knightmate;
mod last_stand;
mod locust_swarm;
mod loyalist;
mod mad_horse;
mod martyrdom;
mod merchant_guild;
mod metal;
mod mistake_card;
mod opening_support;
mod panic;
mod pawn_conversion;
mod princess;
mod promotion_rush;
mod queen_cavalry;
mod queens_gambit;
mod random_status;
mod reposition;
mod rocket;
mod roller;
mod royal_command;
mod royal_shield;
mod sacrifice;
mod scarecrow;
mod severance;
mod snipe;
mod socialism;
mod solidarity;
mod suicide_bomber;
mod switcheroo;
mod target_status;
mod taunt;
mod thief;
mod trolley;
mod turn_effect_card;
mod twins;
mod vanguard;
mod vigilance;
mod vortex;
mod wanted;

use shared::*;
mod barrier;
mod beacon;
mod campfire;
mod constitutional_monarchy;
mod don_quixote;
mod eagle;
mod eastern_policy;
mod grappler;
mod idol;
mod jester;
mod killer;
mod lifecycle;
mod medium;
mod reaper;
mod reformation;
mod revolving_door;
mod runtime;
mod shared;
mod transform_support;
mod transmission_tower;
mod windmill;
mod wizard;
pub use runtime::OctoberRuntimeState;
#[cfg(test)]
mod tests;
use super::piece_effects::has_adjacent_enemy;
use super::piece_effects::is_ranged_piece;
use super::piece_effects::squares_target;
use super::piece_effects::suspicious_potion_effects;
use super::turn_effects::replay_move_matches_state;
use super::turn_effects::restore_replay_color_runtime;
use super::turn_effects::restore_replay_piece_runtime;
pub(crate) use grudge::grudge_kind_allowed;
mod amazon;
mod archbishop;
mod armistice;
mod ascension;
mod assassin;
mod assembly;
mod baby_bear;
mod backward_knight;
mod berserker;
mod big_bishop;
mod bina_mate;
mod blue_jeans;
mod breakthrough_order;
mod brutus;
mod camouflage_color;
mod capture_the_flag;
mod chancellor;
mod chaos_pieces;
mod checker;
mod cleanup;
mod clockwork;
mod clone;
mod conversion;
mod cool_guy;
mod corner_kick;
mod coronation;
mod desperado;
mod disassembly;
mod dragon;
mod early_promotion;
mod empty_lunchbox;
mod exhaustion;
mod extinction;
mod false_start;
mod fast_growth;
mod fianchetto;
mod field_promotion;
mod file_surge;
mod fleeting_dream;
mod frenzy;
mod geneva_convention;
mod ghost;
mod gomoku;
mod grasshopper;
mod greek_gift;
mod grudge;
mod hallucination;
mod hedgehog;
mod highlander;
mod homecoming;
mod hook;
mod hope;
mod horse_riding;
mod hypocrisy;
mod imperial_studies;
mod infiltration;
mod initiative;
mod injury;
mod insight;
mod iqp;
mod joker;
mod killer_king;
mod king_of_the_hill;
mod knightmaster;
mod local_conscription;
mod log;
mod london_system;
mod long_en_passant;
mod macho_chess;
mod magic_girl;
mod majesty;
mod man;
mod miracle;
mod missionary;
mod mongolian_gambit;
mod moving;
mod nullification;
mod octopus;
mod ordination;
mod otherworld;
mod outpost;
mod overtake;
mod overwhelm;
mod paladin;
mod parrot;
mod parry;
mod pawn_chain;
mod pawn_storm;
mod pegasus;
mod penta_check;
mod platform;
mod poisoned_pawn;
mod portal_gun;
mod proficiency;
mod prophecy;
mod quantum_mechanics;
mod queen_afterimage;
mod racing_king;
mod radical_charge;
mod recurrence;
mod recycling;
mod relay;
mod religious_victory;
mod replay;
mod retreat;
mod reversal;
mod reverse_pawns;
mod rook_lift;
mod rule_support;
mod saturation;
mod schrodinger_pawns;
mod seal;
mod siege_ram;
mod siren;
mod slime;
mod sprint;
mod spy;
mod standard_bearer;
mod stealth;
mod submerge;
mod substitution;
mod suspicious_potion;
mod symmetry;
mod taboo;
mod transcendence;
mod trickster;
mod trojan_horse;
mod undead;
mod underpromotion;
mod white_box;
pub use rule_support::RuleRuntime;
pub(in crate::engine) use rule_support::rule_sample;
use rule_support::*;
mod acceleration;
mod black_hole;
mod chess_344200;
mod chess_960;
pub(in crate::engine) mod chess_n_pow_30;
mod conveyor;
mod crown;
mod diagonal_chess;
mod football;
mod high_ground;
mod highway;
mod mistake;
mod monochrome_chess;
mod monster;
mod periodic_collapse;
mod portal;
mod revelation;
mod rule_bombs;
mod rule_ticket;
mod winter_kingdom;
type Eligibility = fn(&GameState, Player, AcquisitionContext) -> CardEligibility;
type Acquisition = fn(&Engine, &mut GameState, Player, &mut Vec<Event>) -> Result<(), EngineError>;
type Play = fn(
    &Engine,
    &mut GameState,
    Player,
    u32,
    &ActionTarget,
    &mut Vec<Event>,
) -> Result<(), EngineError>;
type Actions = fn(&Engine, &GameState, &CardInstance, &mut Vec<Action>);
type RuleInstall = fn(&Engine, &mut GameState, &mut Vec<Event>) -> Result<(), EngineError>;

#[derive(Clone, Copy)]
struct CardBehavior {
    id: &'static str,
    acquire: Option<Acquisition>,
    play: Option<Play>,
    actions: Option<Actions>,
    install: Option<RuleInstall>,
    eligibility: Eligibility,
}
impl CardBehavior {
    const fn active(
        id: &'static str,
        play: Play,
        actions: Actions,
        eligibility: Eligibility,
    ) -> Self {
        Self {
            id,
            acquire: None,
            play: Some(play),
            actions: Some(actions),
            install: None,
            eligibility,
        }
    }
    const fn passive(
        id: &'static str,
        acquire: Option<Acquisition>,
        eligibility: Eligibility,
    ) -> Self {
        Self {
            id,
            acquire,
            play: None,
            actions: None,
            install: None,
            eligibility,
        }
    }
    const fn rule(id: &'static str, install: RuleInstall, eligibility: Eligibility) -> Self {
        Self {
            id,
            acquire: None,
            play: None,
            actions: None,
            install: Some(install),
            eligibility,
        }
    }
}
const CARDS: &[CardBehavior] = &[
    acceleration::BEHAVIOR,
    alekhine_machine_gun::BEHAVIOR,
    amazon::BEHAVIOR,
    apprentice_knights::BEHAVIOR,
    archbishop::BEHAVIOR,
    armistice::BEHAVIOR,
    ascension::BEHAVIOR,
    assassin::BEHAVIOR,
    assembly::BEHAVIOR,
    baby_bear::BEHAVIOR,
    backward_knight::BEHAVIOR,
    barrier::BEHAVIOR,
    basic_training::BEHAVIOR,
    beacon::BEHAVIOR,
    berserker::BEHAVIOR,
    big_bishop::BEHAVIOR,
    big_rook::BEHAVIOR,
    bina_mate::BEHAVIOR,
    bishop_infiltration::BEHAVIOR,
    black_hole::BEHAVIOR,
    blockade::BEHAVIOR,
    blue_jeans::BEHAVIOR,
    breakthrough_order::BEHAVIOR,
    brutus::BEHAVIOR,
    calling_card::BEHAVIOR,
    camouflage_color::BEHAVIOR,
    campfire::BEHAVIOR,
    capture_the_flag::BEHAVIOR,
    castling::BEHAVIOR,
    chameleon_mutation::BEHAVIOR,
    chancellor::BEHAVIOR,
    checker::BEHAVIOR,
    chess_344200::BEHAVIOR,
    chess_960::BEHAVIOR,
    chess_n_pow_30::BEHAVIOR,
    chimera::BEHAVIOR,
    cleanup::BEHAVIOR,
    clockwork::BEHAVIOR,
    clone::BEHAVIOR,
    constitutional_monarchy::BEHAVIOR,
    conversion::BEHAVIOR,
    conveyor::BEHAVIOR,
    cool_guy::BEHAVIOR,
    corner_kick::BEHAVIOR,
    coronation::BEHAVIOR,
    counterattack::BEHAVIOR,
    crown::BEHAVIOR,
    d4::BEHAVIOR,
    desperado::BEHAVIOR,
    diagonal_chess::BEHAVIOR,
    dice::BEHAVIOR,
    disarm::BEHAVIOR,
    disassembly::BEHAVIOR,
    don_quixote::BEHAVIOR,
    dragon::BEHAVIOR,
    dutch::BEHAVIOR,
    e4::BEHAVIOR,
    eagle::BEHAVIOR,
    early_promotion::BEHAVIOR,
    eastern_policy::BEHAVIOR,
    elephant_escape::BEHAVIOR,
    emergency_evacuation::BEHAVIOR,
    empty_lunchbox::BEHAVIOR,
    en_passant_bang::BEHAVIOR,
    encouragement::BEHAVIOR,
    evasion::BEHAVIOR,
    exhaustion::BEHAVIOR,
    extinction::BEHAVIOR,
    false_start::BEHAVIOR,
    fanatical_ritual::BEHAVIOR,
    fast_growth::BEHAVIOR,
    feudal_contract::BEHAVIOR,
    fianchetto::BEHAVIOR,
    field_promotion::BEHAVIOR,
    file_conveyor::BEHAVIOR,
    file_surge::BEHAVIOR,
    fleeting_dream::BEHAVIOR,
    football::BEHAVIOR,
    freeze::BEHAVIOR,
    frenzy::BEHAVIOR,
    frontline_response::BEHAVIOR,
    geneva_convention::BEHAVIOR,
    ghost::BEHAVIOR,
    gomoku::BEHAVIOR,
    grappler::BEHAVIOR,
    grasshopper::BEHAVIOR,
    greek_gift::BEHAVIOR,
    grudge::BEHAVIOR,
    guard::BEHAVIOR,
    hallucination::BEHAVIOR,
    hedgehog::BEHAVIOR,
    herald::BEHAVIOR,
    high_ground::BEHAVIOR,
    highlander::BEHAVIOR,
    highway::BEHAVIOR,
    holdout::BEHAVIOR,
    homecoming::BEHAVIOR,
    hook::BEHAVIOR,
    hope::BEHAVIOR,
    horde::BEHAVIOR,
    horse_riding::BEHAVIOR,
    hypocrisy::BEHAVIOR,
    ice_king::BEHAVIOR,
    ice_sheet::BEHAVIOR,
    idol::BEHAVIOR,
    imperial_studies::BEHAVIOR,
    inertia::BEHAVIOR,
    infiltration::BEHAVIOR,
    initiative::BEHAVIOR,
    injury::BEHAVIOR,
    insight::BEHAVIOR,
    iqp::BEHAVIOR,
    iron_monarch::BEHAVIOR,
    jester::BEHAVIOR,
    joker::BEHAVIOR,
    killer::BEHAVIOR,
    killer_king::BEHAVIOR,
    king_of_the_hill::BEHAVIOR,
    knightmaster::BEHAVIOR,
    knightmate::BEHAVIOR,
    last_stand::BEHAVIOR,
    local_conscription::BEHAVIOR,
    locust_swarm::BEHAVIOR,
    log::BEHAVIOR,
    london_system::BEHAVIOR,
    long_en_passant::BEHAVIOR,
    loyalist::BEHAVIOR,
    macho_chess::BEHAVIOR,
    mad_horse::BEHAVIOR,
    magic_girl::BEHAVIOR,
    majesty::BEHAVIOR,
    man::BEHAVIOR,
    martyrdom::BEHAVIOR,
    medium::BEHAVIOR,
    merchant_guild::BEHAVIOR,
    metal::BEHAVIOR,
    miracle::BEHAVIOR,
    missionary::BEHAVIOR,
    mistake::BEHAVIOR,
    mistake_card::BEHAVIOR,
    mongolian_gambit::BEHAVIOR,
    monochrome_chess::BEHAVIOR,
    monster::BEHAVIOR,
    moving::BEHAVIOR,
    nullification::BEHAVIOR,
    octopus::BEHAVIOR,
    ordination::BEHAVIOR,
    otherworld::BEHAVIOR,
    outpost::BEHAVIOR,
    overtake::BEHAVIOR,
    overwhelm::BEHAVIOR,
    paladin::BEHAVIOR,
    panic::BEHAVIOR,
    parrot::BEHAVIOR,
    parry::BEHAVIOR,
    pawn_chain::BEHAVIOR,
    pawn_conversion::BEHAVIOR,
    pawn_storm::BEHAVIOR,
    pegasus::BEHAVIOR,
    penta_check::BEHAVIOR,
    periodic_collapse::BEHAVIOR,
    platform::BEHAVIOR,
    poisoned_pawn::BEHAVIOR,
    portal::BEHAVIOR,
    portal_gun::BEHAVIOR,
    princess::BEHAVIOR,
    proficiency::BEHAVIOR,
    promotion_rush::BEHAVIOR,
    prophecy::BEHAVIOR,
    quantum_mechanics::BEHAVIOR,
    queen_afterimage::BEHAVIOR,
    queen_cavalry::BEHAVIOR,
    queens_gambit::BEHAVIOR,
    racing_king::BEHAVIOR,
    radical_charge::BEHAVIOR,
    reaper::BEHAVIOR,
    recurrence::BEHAVIOR,
    recycling::BEHAVIOR,
    reformation::BEHAVIOR,
    relay::BEHAVIOR,
    religious_victory::BEHAVIOR,
    replay::BEHAVIOR,
    reposition::BEHAVIOR,
    retreat::BEHAVIOR,
    revelation::BEHAVIOR,
    reversal::BEHAVIOR,
    reverse_pawns::BEHAVIOR,
    revolving_door::BEHAVIOR,
    rocket::BEHAVIOR,
    roller::BEHAVIOR,
    rook_lift::BEHAVIOR,
    royal_command::BEHAVIOR,
    royal_shield::BEHAVIOR,
    rule_bombs::BEHAVIOR,
    rule_ticket::BEHAVIOR,
    sacrifice::BEHAVIOR,
    saturation::BEHAVIOR,
    scarecrow::BEHAVIOR,
    schrodinger_pawns::BEHAVIOR,
    seal::BEHAVIOR,
    severance::BEHAVIOR,
    siege_ram::BEHAVIOR,
    siren::BEHAVIOR,
    slime::BEHAVIOR,
    snipe::BEHAVIOR,
    socialism::BEHAVIOR,
    solidarity::BEHAVIOR,
    sprint::BEHAVIOR,
    spy::BEHAVIOR,
    standard_bearer::BEHAVIOR,
    stealth::BEHAVIOR,
    submerge::BEHAVIOR,
    substitution::BEHAVIOR,
    suicide_bomber::BEHAVIOR,
    suspicious_potion::BEHAVIOR,
    switcheroo::BEHAVIOR,
    symmetry::BEHAVIOR,
    taboo::BEHAVIOR,
    taunt::BEHAVIOR,
    thief::BEHAVIOR,
    transcendence::BEHAVIOR,
    transmission_tower::BEHAVIOR,
    trickster::BEHAVIOR,
    trojan_horse::BEHAVIOR,
    trolley::BEHAVIOR,
    twins::BEHAVIOR,
    undead::BEHAVIOR,
    underpromotion::BEHAVIOR,
    vanguard::BEHAVIOR,
    vigilance::BEHAVIOR,
    vortex::BEHAVIOR,
    wanted::BEHAVIOR,
    white_box::BEHAVIOR,
    windmill::BEHAVIOR,
    winter_kingdom::BEHAVIOR,
    wizard::BEHAVIOR,
];
fn find(id: &str) -> Option<&'static CardBehavior> {
    CARDS
        .binary_search_by_key(&id, |card| card.id)
        .ok()
        .map(|index| &CARDS[index])
}
pub(in crate::engine) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    id: &str,
    owner: Player,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let card = find(id).ok_or_else(|| EngineError::UnknownCard(id.into()))?;
    if let Some(apply) = card.acquire {
        apply(engine, state, owner, events)?;
    }
    Ok(())
}
pub(in crate::engine) fn play(
    engine: &Engine,
    state: &mut GameState,
    id: &str,
    actor: Player,
    instance_id: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let apply = find(id)
        .and_then(|card| card.play)
        .ok_or_else(|| EngineError::UnknownCard(id.into()))?;
    apply(engine, state, actor, instance_id, target, events)
}
pub(in crate::engine) fn actions(
    engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    if let Some(generate) = find(&card.definition_id).and_then(|entry| entry.actions) {
        generate(engine, state, card, actions);
    }
}
pub(in crate::engine) fn install(
    engine: &Engine,
    state: &mut GameState,
    id: &str,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let install = find(id)
        .and_then(|card| card.install)
        .ok_or_else(|| EngineError::UnknownCard(id.into()))?;
    install(engine, state, events)
}
pub(crate) fn eligibility(
    state: &GameState,
    id: &str,
    player: Player,
    context: AcquisitionContext,
) -> CardEligibility {
    find(id).map_or_else(
        || {
            CardEligibility::Ineligible(crate::EligibilityReason {
                code: crate::EligibilityReasonCode::UnknownCard,
                detail: id.into(),
                message_ko: "지원하지 않는 카드입니다.".into(),
            })
        },
        |card| (card.eligibility)(state, player, context),
    )
}

#[cfg(test)]
mod registry_tests {
    use super::*;

    #[test]
    fn registry_exactly_covers_supported_catalog_without_duplicate_ids() {
        let catalog_ids: BTreeSet<_> = crate::latest_catalog()
            .iter()
            .filter(|card| card.supported())
            .map(|card| card.id.as_str())
            .collect();
        let registry_ids: BTreeSet<_> = CARDS.iter().map(|card| card.id).collect();
        assert_eq!(catalog_ids, registry_ids);
        assert_eq!(CARDS.len(), registry_ids.len());
        assert!(CARDS.windows(2).all(|pair| pair[0].id < pair[1].id));
    }

    #[test]
    fn callbacks_match_catalog_activation() {
        let engine = Engine::bundled_latest().unwrap();
        for behavior in CARDS {
            let definition = engine.card_definition(behavior.id).unwrap();
            let (active, rule) = match definition.activation {
                CardActivation::Active => (true, false),
                CardActivation::Passive => (false, false),
                CardActivation::Rule => (false, true),
            };
            assert_eq!(behavior.play.is_some(), active, "{} play", behavior.id);
            assert_eq!(
                behavior.actions.is_some(),
                active,
                "{} actions",
                behavior.id
            );
            assert_eq!(behavior.install.is_some(), rule, "{} install", behavior.id);
            if active || rule {
                assert!(behavior.acquire.is_none(), "{} acquire", behavior.id);
            }
        }
        assert!(find("unsupported-card").is_none());
    }
}
