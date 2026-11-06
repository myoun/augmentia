//! Version ownership through the same card lookup used by engine callers.
use augment_core::{CardActivation, Engine, LATEST_RULESET_ID, ruleset_card};
use serde_json::Value;

#[test]
fn current_definitions_use_their_own_catalog_and_execution_profile() {
    let engine = Engine::bundled_latest().unwrap();
    let catalog: Value = serde_json::from_str(include_str!(
        "../../../rulesets/augment_chess_20261017/cards/catalog.json"
    ))
    .unwrap();
    let profile: Value = serde_json::from_str(include_str!(
        "../../../rulesets/augment_chess_20261017/cards/implemented.json"
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
    }
}

#[test]
fn historical_turn_exclusivity_does_not_leak_into_current_rules() {
    for id in [augment_core::PREMOVE, augment_core::TROLLEY] {
        let historical = ruleset_card("augment_chess_20260915", id).unwrap();
        assert!(historical.turn_exclusive);
    }
    assert!(ruleset_card(LATEST_RULESET_ID, augment_core::PREMOVE).is_none());
    assert!(
        !ruleset_card(LATEST_RULESET_ID, augment_core::TROLLEY)
            .unwrap()
            .turn_exclusive
    );
}
