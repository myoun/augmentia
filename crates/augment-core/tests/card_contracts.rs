//! Before/after refactor contract through the public engine interface.
use augment_core::{
    AcquisitionContext, Action, Board, CardActivation, Engine, GameState, Mode, Piece, PieceKind,
    Player, check_card_eligibility, latest_catalog,
};
use serde_json::{Value, json};

fn contracts() -> Value {
    let engine = Engine::bundled_latest().unwrap();
    let mut cases = Vec::new();
    for scenario in 0..4 {
        let owner = if scenario < 2 {
            Player::White
        } else {
            Player::Black
        };
        let mut board = Board::standard();
        if scenario % 2 == 1 {
            // Real capture/target geometry as well as the normal initial layout.
            for (from, to) in [("d2", "d4"), ("d7", "d5"), ("c1", "e3"), ("b8", "c6")] {
                let piece = board.get(from.parse().unwrap());
                board.set(from.parse().unwrap(), None);
                board.set(to.parse().unwrap(), piece);
            }
            board.set(
                "f4".parse().unwrap(),
                Some(Piece::new(Player::Black, PieceKind::Pawn)),
            );
        }
        let mut base = GameState::from_board(
            augment_core::LATEST_RULESET_ID,
            Mode::Standard,
            board,
            owner,
            31,
        );
        base.completed_drafts = [true; 3];
        base.turns_taken = [6; 2];
        base.ply = 12;
        for card in latest_catalog().iter().filter(|card| card.supported()) {
            let definition = engine.card_definition(&card.id).unwrap();
            let setup = format!(
                "{:?}",
                check_card_eligibility(&base, &card.id, owner, AcquisitionContext::Setup)
            );
            let draft = format!(
                "{:?}",
                check_card_eligibility(&base, &card.id, owner, AcquisitionContext::Draft)
            );
            let acquired = if definition.activation == CardActivation::Rule {
                engine.apply_rule(&base, &card.id)
            } else {
                engine.grant_card(&base, 1, &card.id, owner)
            };
            let acquired = match acquired {
                Err(error) => json!({"error": error.to_string()}),
                Ok(transition) => {
                    let state = transition.state;
                    match engine.legal_actions(&state) {
                        Err(error) => json!({"error":error.to_string()}),
                        Ok(actions) => {
                            let results: Vec<_> = actions.iter().take(3).chain(actions.iter().filter(|action| matches!(action, Action::UseCard { .. })).take(3)).map(|action| match engine.apply(&state, action.clone()) {
                                Ok(next) => json!({"fingerprint":format!("{:016x}",next.state.fingerprint()),"events":format!("{:?}",next.events)}),
                                Err(error) => json!({"error":error.to_string()}),
                            }).collect();
                            json!({"fingerprint":format!("{:016x}",state.fingerprint()),"actions":format!("{actions:?}"),"events":format!("{:?}",transition.events),"results":results})
                        }
                    }
                }
            };
            cases.push(json!({"scenario":scenario,"id":card.id,"setup":setup,"draft":draft,"acquired":acquired}));
        }
    }
    json!(cases)
}

#[test]
fn supported_cards_preserve_acquisition_actions_events_and_state_identity() {
    let actual = contracts();
    let expected: Value =
        serde_json::from_str(include_str!("fixtures/card-contracts.json")).unwrap();
    let mut differences = Vec::new();
    for (actual, expected) in actual
        .as_array()
        .unwrap()
        .iter()
        .zip(expected.as_array().unwrap())
    {
        for field in ["setup", "draft", "acquired"] {
            if actual[field] != expected[field] {
                differences.push(format!(
                    "{} scenario {}: {field}",
                    actual["id"], actual["scenario"]
                ));
            }
        }
    }
    assert!(differences.is_empty(), "Changed contracts: {differences:?}");
    assert_eq!(
        actual.as_array().unwrap().len(),
        expected.as_array().unwrap().len()
    );
}
