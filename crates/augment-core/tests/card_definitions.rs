//! Version ownership through the same card lookup used by engine callers.
use augment_core::{CardActivation, Engine, LATEST_RULESET_ID, ruleset_card};
use serde_json::Value;

#[test]
fn current_definitions_use_their_own_catalog_and_execution_profile() {
    let engine = Engine::bundled_latest().unwrap();
    let catalog: Value = serde_json::from_str(include_str!(
        "../../../rulesets/augment_chess_20261102/cards/catalog.json"
    ))
    .unwrap();
    let profile: Value = serde_json::from_str(include_str!(
        "../../../rulesets/augment_chess_20261102/cards/implemented.json"
    ))
    .unwrap();
    let entries = profile["cards"].as_array().unwrap();
    assert_eq!(engine.card_definitions().len(), entries.len());
    for entry in entries {
        let id = entry["id"].as_str().unwrap();
        let canonical = catalog["cards"]
            .as_array()
            .unwrap()
            .iter()
            .find(|card| card["id"] == id)
            .unwrap();
        let definition = engine.card_definition(id).unwrap();
        assert_eq!(definition.id, id);
        assert_eq!(Some(definition.name_ko), canonical["name_ko"].as_str());
        assert_eq!(Some(definition.effect), canonical["effect"].as_str());
        assert_eq!(
            Some(u64::from(definition.half_stars)),
            canonical["half_stars"].as_u64()
        );
        assert_eq!(definition.ends_turn, canonical["turn_policy"] == "END_TURN");
        assert_eq!(
            Some(definition.turn_exclusive),
            entry["turn_exclusive"].as_bool()
        );
        assert_eq!(
            definition.activation == CardActivation::Rule,
            canonical["phase"] == "RULE"
        );
        // The phase decides when a card can be drafted, so a card listed in the profile must
        // carry the phase its catalog entry declares.
        let expected_phase = match canonical["phase"].as_str().unwrap() {
            "OPENING" => augment_core::CardPhase::Opening,
            "MIDDLE" => augment_core::CardPhase::Middle,
            "END" => augment_core::CardPhase::End,
            "PIECE" => augment_core::CardPhase::Piece,
            "RULE" => augment_core::CardPhase::Rule,
            other => panic!("unknown phase {other}"),
        };
        assert_eq!(definition.phase, expected_phase, "{id}");
        // Passive cards in this ruleset resolve on acquisition. `turn_policy` is only consulted
        // for a card that spends a charge, which a passive never has, so a passive's policy is
        // inert. The catalog still disagrees with the bundle on 18 of them (see the note in
        // `passive_cards_declare_inert_turn_policies`), so only the activation is asserted here.
        if canonical["phase"] != "RULE" {
            let expected_activation = match canonical["activation"].as_str().unwrap() {
                "PASSIVE" => CardActivation::Passive,
                "ACTIVE" => CardActivation::Active,
                other => panic!("unknown activation {other}"),
            };
            assert_eq!(definition.activation, expected_activation, "{id}");
            if expected_activation == CardActivation::Passive {
                assert_eq!(canonical["play_timing"], "ON_ACQUIRE", "{id}");
            }
        }
    }
}

/// The bundle's v2 card builder rejects a PASSIVE card whose `turnPolicy` is not
/// `NOT_APPLICABLE` (`Inconsistent v2 card activation`). Our catalog nevertheless records
/// `PRESERVE_TURN` for 18 passive cards.
///
/// The field is inert for them: a passive installs on acquisition and never has a charge to
/// spend, so `ends_turn` is never consulted. This test pins the known discrepancy so it cannot
/// grow silently, and so nobody mistakes the catalog for a faithful record of that field.
#[test]
fn passive_cards_declare_inert_turn_policies() {
    let catalog: Value = serde_json::from_str(include_str!(
        "../../../rulesets/augment_chess_20261102/cards/catalog.json"
    ))
    .unwrap();
    let profile: Value = serde_json::from_str(include_str!(
        "../../../rulesets/augment_chess_20261102/cards/implemented.json"
    ))
    .unwrap();
    let supported: std::collections::BTreeSet<&str> = profile["cards"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["id"].as_str().unwrap())
        .collect();
    let mut odd = Vec::new();
    for card in catalog["cards"].as_array().unwrap() {
        let id = card["id"].as_str().unwrap();
        if !supported.contains(id) || card["enabled"] != true || card["phase"] == "RULE" {
            continue;
        }
        if card["activation"] == "PASSIVE" && card["turn_policy"] != "NOT_APPLICABLE" {
            odd.push(id.to_owned());
        }
    }
    // None of these may be an END_TURN passive: that would mean a spent turn for a card that
    // cannot be spent.
    for id in &odd {
        let card = catalog["cards"]
            .as_array()
            .unwrap()
            .iter()
            .find(|candidate| candidate["id"] == id.as_str())
            .unwrap();
        assert_ne!(card["turn_policy"], "END_TURN", "{id}");
    }
    assert_eq!(
        odd.len(),
        18,
        "the known passive-with-policy set changed; re-check the catalog against the bundle: {odd:?}"
    );
}

/// The 2026-11-02 snapshot changed the transition contract of these cards. Whatever else moves,
/// their executable contract must keep matching the catalog, because that contract is what decides
/// when a card may be drafted, armed and spent.
#[test]
fn transition_changed_cards_keep_their_declared_contract() {
    let engine = Engine::bundled_latest().unwrap();
    let catalog: Value = serde_json::from_str(include_str!(
        "../../../rulesets/augment_chess_20261102/cards/catalog.json"
    ))
    .unwrap();
    // Cards whose activation, play timing, turn policy, selection or phase changed in this
    // snapshot, plus the ones it introduced.
    let changed = [
        "antichess",
        "ascension",
        "barrier",
        "berserker",
        "bombardment",
        "chameleon-mutation",
        "citadel",
        "death-squad",
        "hedgehog",
        "killer",
        "offering",
        "omega-chesstron",
        "overwhelm",
        "siren",
        "trojan-horse",
        "undead",
        "wanderer",
    ];
    for id in changed {
        let definition = engine
            .card_definition(id)
            .unwrap_or_else(|| panic!("{id} must be executable"));
        let canonical = catalog["cards"]
            .as_array()
            .unwrap()
            .iter()
            .find(|card| card["id"] == id)
            .unwrap_or_else(|| panic!("{id} must be in the catalog"));
        let activation = match canonical["activation"].as_str().unwrap() {
            "PASSIVE" => CardActivation::Passive,
            "ACTIVE" => CardActivation::Active,
            other => panic!("{id}: unknown activation {other}"),
        };
        assert_eq!(definition.activation, activation, "{id}");
        assert_eq!(
            definition.ends_turn,
            canonical["turn_policy"] == "END_TURN",
            "{id}"
        );
        // A passive resolves on acquisition; only an active card is spent from a charge.
        if activation == CardActivation::Passive {
            assert_eq!(canonical["play_timing"], "ON_ACQUIRE", "{id}");
            assert_eq!(canonical["selection"]["maxSquares"], 0, "{id}");
        }
    }
}

#[test]
fn historical_turn_exclusivity_does_not_leak_into_current_rules() {
    for id in [augment_core::PREMOVE, augment_core::TROLLEY] {
        let historical = ruleset_card("augment_chess_20260915", id).unwrap();
        assert!(historical.turn_exclusive);
    }
    // Both cards are now ported into the current profile, and neither may inherit
    // the September turn-exclusive policy: the current contract preserves the turn.
    for id in [augment_core::PREMOVE, augment_core::TROLLEY] {
        let current = ruleset_card(LATEST_RULESET_ID, id)
            .unwrap_or_else(|| panic!("{id} should be ported into the current profile"));
        assert!(
            !current.turn_exclusive,
            "{id} inherited historical exclusivity"
        );
    }
}
