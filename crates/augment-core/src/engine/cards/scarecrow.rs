use super::*;
pub(super) const ID: &str = crate::SCARECROW;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (square, victim) = piece_target(state, target)?;
    engine.delete_piece_for_card(state, square, victim, actor, ID, events);
    let scarecrow = Piece::new(actor, PieceKind::Scarecrow).with_id(state.allocate_piece_id());
    state.board.set(square, Some(scarecrow));
    state.card_runtime.moved_piece_ids.insert(scarecrow.id);
    state
        .card_runtime
        .piece_origins
        .insert(scarecrow.id, square);
    state.card_runtime.protected_piece_ids.insert(scarecrow.id);
    state.card_runtime.scarecrow_reservations.insert(
        scarecrow.id,
        (actor, state.turns_taken[actor.index()].saturating_add(3)),
    );
    Ok(())
}
fn actions(engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    for (square, piece) in web_ordered_pieces(state) {
        if piece.owner == card.owner
            && !matches!(piece.kind, PieceKind::Football)
            && !engine.d4_restricts(state, card.owner, false, square)
        {
            out.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::PieceAt(square),
            });
        }
    }
}
fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    static ENGINE: std::sync::OnceLock<Engine> = std::sync::OnceLock::new();
    let engine = ENGINE.get_or_init(|| Engine::bundled_latest().expect("bundled latest ruleset"));
    require(
        (0..64).filter_map(Square::from_index).any(|s| {
            engine.card_landing_open(state, s) && !engine.d4_restricts(state, owner, false, s)
        }),
        "허수아비를 설치할 빈칸이 필요합니다.",
    )
}
