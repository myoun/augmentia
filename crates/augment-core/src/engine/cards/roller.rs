use super::*;
pub(super) const ID: &str = crate::ROLLER;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);
fn play(
    _: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    _: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    state.card_runtime.roller_active[actor.index()] = true;
    Ok(())
}
fn actions(_: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    if !state.card_runtime.roller_active[card.owner.index()] {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::None,
        });
    }
}
fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    enabled(state.card_runtime.roller_active[owner.index()]).unwrap_or(CardEligibility::Eligible)
}

fn add_segment(cells: &mut BTreeSet<Square>, from: Square, to: Square, teleport: bool) {
    cells.insert(from);
    cells.insert(to);
    if teleport || (from.file() != to.file() && from.rank() != to.rank()) {
        return;
    }
    let df = (to.file() as i8 - from.file() as i8).signum();
    let dr = (to.rank() as i8 - from.rank() as i8).signum();
    let mut cursor = from;
    while let Some(next) = cursor.offset(df, dr) {
        cells.insert(next);
        if next == to {
            break;
        }
        cursor = next;
    }
}

impl Engine {
    pub(in crate::engine) fn settle_roller(
        &self,
        before: &GameState,
        action: &Action,
        after: &mut GameState,
        events: &[Event],
    ) {
        if before.ruleset_id != crate::LATEST_RULESET_ID
            || !before.card_runtime.roller_active[before.turn.index()]
        {
            return;
        }
        let Action::Move(movement) = action else {
            return;
        };
        let Some(piece) = before.board.get(movement.from) else {
            return;
        };
        let blocked = events.iter().any(|event| matches!(event, Event::EffectApplied { command, .. } if ["shield-broken:", "metal-hit:", "encouragement-hit:", "vigilance-hit:", "hp-hit:", "mistake-reverse:"].iter().any(|prefix| command.starts_with(prefix))));
        if blocked {
            return;
        }
        let mut cells = BTreeSet::new();
        if piece.kind == PieceKind::Rook && movement.from != movement.to {
            let route = self.rule_portal_route(before, movement.from, piece, movement.to);
            if let Some(route) = route {
                add_segment(&mut cells, movement.from, route.entry, false);
                add_segment(&mut cells, route.exit, route.landing, false);
            } else {
                let teleport = (before.card_runtime.e4_active[piece.owner.index()]
                    && movement.to
                        == Square::new(4, if piece.owner == Player::White { 3 } else { 4 })
                            .unwrap())
                    || (before.card_runtime.symmetry[piece.owner.index()]
                        && movement.to.file() == 7 - movement.from.file()
                        && movement.to.rank() == movement.from.rank())
                    || before.board.get(movement.to).is_some_and(|target| {
                        (target.owner == piece.owner
                            && before.card_runtime.relay[piece.owner.index()])
                            || (target.owner != piece.owner
                                && target.kind == piece.kind
                                && before.card_runtime.substitution[piece.owner.index()])
                    });
                add_segment(&mut cells, movement.from, movement.to, teleport);
            }
        }
        if let Some((from, _, rook)) = self.standard_castling_rook_plan(before, piece, *movement)
            && rook.kind == PieceKind::Rook
            && let Some((to, _)) = after.board.find_by_id(rook.id)
            && from != to
        {
            add_segment(&mut cells, from, to, false);
        }
        after.card_runtime.highway_cells.extend(cells);
    }
}
