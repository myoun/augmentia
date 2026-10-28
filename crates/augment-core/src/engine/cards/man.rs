use super::*;
pub(super) const ID: &str = crate::MAN;
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    _actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let target = target.clone();
    let (square, mut piece) = piece_target(&*state, &target)?;
    engine.clear_piece_kind_runtime(state, piece.id);
    state.card_runtime.trickster_move_kinds.remove(&piece.id);
    engine.clear_queens_gambit_protection(state, piece.id);
    piece.kind = PieceKind::Man;
    state.board.set(square, Some(piece));
    engine.mark_transformed_piece(state, piece);
    Ok(())
}

pub(super) fn actions(
    engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    for (square, piece) in state.board.iter() {
        if crate::latest::latest_minor_kind(piece.kind) && !engine.is_royal_piece(state, piece) {
            actions.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::PieceAt(square),
            });
        }
    }
}

pub(super) fn eligibility(
    state: &GameState,
    _player: Player,
    context: AcquisitionContext,
) -> CardEligibility {
    require(
        context == AcquisitionContext::Draft
            || state.board.iter().any(|(_, piece)| {
                crate::latest::latest_minor_kind(piece.kind) && !royal(state, piece)
            }),
        "변환 가능한 비왕실 마이너 기물이 필요합니다.",
    )
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

#[cfg(test)]
mod tests {
    use super::super::tests::interactions::*;
    use super::*;
    #[test]
    fn man_accepts_both_sides_native_minors_and_rejects_royals_and_majors() {
        let engine = Engine::bundled_latest().unwrap();
        let state = position(
            &[
                ("a1", Player::White, PieceKind::King),
                ("h8", Player::Black, PieceKind::King),
                ("d4", Player::White, PieceKind::Knight),
                ("f4", Player::Black, PieceKind::Wizard),
                ("b2", Player::White, PieceKind::Queen),
            ],
            Player::White,
        );
        let mut state = engine
            .grant_card(&state, 1, crate::MAN, Player::White)
            .unwrap()
            .state;
        let wizard = state.board.get("f4".parse().unwrap()).unwrap();
        state.card_runtime.wizard_mana.insert(wizard.id, 4);
        assert!(engine.legal_actions(&state).unwrap().contains(&card("d4")));
        assert!(!engine.legal_actions(&state).unwrap().contains(&card("b2")));
        assert!(!engine.legal_actions(&state).unwrap().contains(&card("a1")));
        let next = engine.apply(&state, card("f4")).unwrap().state;
        let man = next.board.get("f4".parse().unwrap()).unwrap();
        assert_eq!(
            (man.id, man.owner, man.kind),
            (wizard.id, Player::Black, PieceKind::Man)
        );
        assert!(!next.card_runtime.wizard_mana.contains_key(&wizard.id));
        state.card_runtime.royal_piece_ids.insert(wizard.id);
        assert!(!engine.legal_actions(&state).unwrap().contains(&card("f4")));
    }
}
