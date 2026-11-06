use super::*;
pub(super) fn play(
    _: &Engine,
    state: &mut GameState,
    actor: Player,
    instance_id: u32,
    target: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    if *target != ActionTarget::None {
        return Err(EngineError::Invariant("turn effect takes no target".into()));
    }
    let id = &state
        .cards
        .iter()
        .find(|c| c.instance_id == instance_id)
        .ok_or_else(|| EngineError::Invariant("missing turn effect card".into()))?
        .definition_id;
    match id.as_str() {
        crate::EN_PASSANT_BANG => state.card_runtime.en_passant_frenzy[actor.index()] = true,
        crate::SNIPE => state.card_runtime.bishop_snipe[actor.index()] = true,
        crate::SOCIALISM => state.card_runtime.socialism[actor.opponent().index()] = 1,
        crate::TAUNT => state.card_runtime.taunt[actor.opponent().index()] = 1,
        crate::SWITCHEROO => state.card_runtime.switcheroo[actor.index()] = true,
        crate::ROYAL_COMMAND => {
            let start = state.turns_taken[actor.index()].saturating_add(1);
            state.card_runtime.royal_command[actor.index()] = Some(RoyalCommandStatus {
                starts_at_turn: start,
                expires_at_turn: start.saturating_add(2),
            });
        }
        _ => return Err(EngineError::Invariant("unknown turn effect card".into())),
    }
    Ok(())
}
pub(super) fn actions(_: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    if state.turn == card.owner {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::None,
        });
    }
}
pub(super) fn eligibility(state: &GameState, owner: Player, id: &str) -> CardEligibility {
    let available = state.board.iter().any(|(_, p)| match id {
        crate::EN_PASSANT_BANG => p.owner == owner && p.kind == PieceKind::Pawn,
        crate::SNIPE => p.owner == owner && p.kind == PieceKind::Bishop,
        crate::ROYAL_COMMAND => {
            p.owner == owner && !royal(state, p) && p.kind != PieceKind::Merchant
        }
        crate::SOCIALISM => {
            p.owner == owner.opponent() && !royal(state, p) && p.kind != PieceKind::Merchant
        }
        crate::TAUNT => p.owner == owner.opponent() && p.kind != PieceKind::Football,
        crate::SWITCHEROO => p.owner == owner && p.kind == PieceKind::Pawn,
        _ => false,
    });
    require(
        available
            && (id != crate::SWITCHEROO
                || state.board.iter().any(|(_, p)| {
                    p.owner == owner && (royal(state, p) || p.kind == PieceKind::Merchant)
                })),
        "효과를 적용할 기물이 필요합니다.",
    )
}
