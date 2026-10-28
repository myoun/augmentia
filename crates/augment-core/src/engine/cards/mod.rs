//! Current executable card registry. Common turn/action framing stays in Engine.
use super::*;
use crate::card_eligibility::{AcquisitionContext, CardEligibility};
use crate::*;
pub use ascension::PendingAscension;
use eligibility_support::*;
pub use penta_check::PentaCheckProgress;
pub use seal::PendingSeal;
pub use taboo::PendingTaboo;
mod eligibility_support;

use shared::*;
mod lifecycle;
mod runtime;
mod shared;
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
    amazon::BEHAVIOR,
    archbishop::BEHAVIOR,
    armistice::BEHAVIOR,
    ascension::BEHAVIOR,
    assassin::BEHAVIOR,
    assembly::BEHAVIOR,
    baby_bear::BEHAVIOR,
    backward_knight::BEHAVIOR,
    berserker::BEHAVIOR,
    big_bishop::BEHAVIOR,
    bina_mate::BEHAVIOR,
    blue_jeans::BEHAVIOR,
    breakthrough_order::BEHAVIOR,
    brutus::BEHAVIOR,
    camouflage_color::BEHAVIOR,
    capture_the_flag::BEHAVIOR,
    chancellor::BEHAVIOR,
    checker::BEHAVIOR,
    cleanup::BEHAVIOR,
    clockwork::BEHAVIOR,
    clone::BEHAVIOR,
    conversion::BEHAVIOR,
    cool_guy::BEHAVIOR,
    corner_kick::BEHAVIOR,
    coronation::BEHAVIOR,
    desperado::BEHAVIOR,
    disassembly::BEHAVIOR,
    dragon::BEHAVIOR,
    early_promotion::BEHAVIOR,
    empty_lunchbox::BEHAVIOR,
    exhaustion::BEHAVIOR,
    extinction::BEHAVIOR,
    false_start::BEHAVIOR,
    fast_growth::BEHAVIOR,
    fianchetto::BEHAVIOR,
    field_promotion::BEHAVIOR,
    file_surge::BEHAVIOR,
    fleeting_dream::BEHAVIOR,
    frenzy::BEHAVIOR,
    geneva_convention::BEHAVIOR,
    ghost::BEHAVIOR,
    gomoku::BEHAVIOR,
    grasshopper::BEHAVIOR,
    greek_gift::BEHAVIOR,
    grudge::BEHAVIOR,
    hallucination::BEHAVIOR,
    hedgehog::BEHAVIOR,
    highlander::BEHAVIOR,
    homecoming::BEHAVIOR,
    hook::BEHAVIOR,
    hope::BEHAVIOR,
    horse_riding::BEHAVIOR,
    hypocrisy::BEHAVIOR,
    imperial_studies::BEHAVIOR,
    infiltration::BEHAVIOR,
    initiative::BEHAVIOR,
    injury::BEHAVIOR,
    insight::BEHAVIOR,
    iqp::BEHAVIOR,
    joker::BEHAVIOR,
    killer_king::BEHAVIOR,
    king_of_the_hill::BEHAVIOR,
    knightmaster::BEHAVIOR,
    local_conscription::BEHAVIOR,
    log::BEHAVIOR,
    london_system::BEHAVIOR,
    long_en_passant::BEHAVIOR,
    macho_chess::BEHAVIOR,
    magic_girl::BEHAVIOR,
    majesty::BEHAVIOR,
    man::BEHAVIOR,
    miracle::BEHAVIOR,
    missionary::BEHAVIOR,
    mongolian_gambit::BEHAVIOR,
    moving::BEHAVIOR,
    nullification::BEHAVIOR,
    octopus::BEHAVIOR,
    ordination::BEHAVIOR,
    otherworld::BEHAVIOR,
    outpost::BEHAVIOR,
    overtake::BEHAVIOR,
    overwhelm::BEHAVIOR,
    paladin::BEHAVIOR,
    parrot::BEHAVIOR,
    parry::BEHAVIOR,
    pawn_chain::BEHAVIOR,
    pawn_storm::BEHAVIOR,
    pegasus::BEHAVIOR,
    penta_check::BEHAVIOR,
    platform::BEHAVIOR,
    poisoned_pawn::BEHAVIOR,
    portal_gun::BEHAVIOR,
    proficiency::BEHAVIOR,
    prophecy::BEHAVIOR,
    quantum_mechanics::BEHAVIOR,
    queen_afterimage::BEHAVIOR,
    racing_king::BEHAVIOR,
    radical_charge::BEHAVIOR,
    recurrence::BEHAVIOR,
    recycling::BEHAVIOR,
    relay::BEHAVIOR,
    religious_victory::BEHAVIOR,
    replay::BEHAVIOR,
    retreat::BEHAVIOR,
    reversal::BEHAVIOR,
    reverse_pawns::BEHAVIOR,
    rook_lift::BEHAVIOR,
    saturation::BEHAVIOR,
    schrodinger_pawns::BEHAVIOR,
    seal::BEHAVIOR,
    siege_ram::BEHAVIOR,
    siren::BEHAVIOR,
    slime::BEHAVIOR,
    sprint::BEHAVIOR,
    spy::BEHAVIOR,
    standard_bearer::BEHAVIOR,
    stealth::BEHAVIOR,
    submerge::BEHAVIOR,
    substitution::BEHAVIOR,
    suspicious_potion::BEHAVIOR,
    symmetry::BEHAVIOR,
    taboo::BEHAVIOR,
    transcendence::BEHAVIOR,
    trickster::BEHAVIOR,
    trojan_horse::BEHAVIOR,
    undead::BEHAVIOR,
    underpromotion::BEHAVIOR,
    white_box::BEHAVIOR,
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
