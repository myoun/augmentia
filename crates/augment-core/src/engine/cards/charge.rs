use super::*;

pub(super) const ID: &str = crate::CHARGE;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

/// `canChargePawn` (main-RjrZ5fFP.js:352608-352616): an own pawn that is neither
/// frozen nor staked and has both the first and second forward squares open.
fn can_charge(state: &GameState, owner: Player, square: Square, piece: Piece) -> bool {
    piece.owner == owner
        && piece.kind == PieceKind::Pawn
        && !state.card_runtime.frozen.contains_key(&piece.id)
        && !state.card_runtime.stake_statuses.contains_key(&piece.id)
        && charge_steps(state, piece, square).len() == 2
}

fn direction(piece: Piece) -> i8 {
    if piece.owner == Player::White { 1 } else { -1 }
}

fn open(state: &GameState, square: Square) -> bool {
    state.board.get(square).is_none()
        && !state.card_runtime.barricades.contains(&square)
        && !state.card_runtime.black_holes.contains(&square)
        && state.card_runtime.crown_pos != Some(square)
        && !(state.card_runtime.collapsed_files_ranks
            && (matches!(square.file(), 0 | 7) || matches!(square.rank(), 0 | 7)))
}

/// The required open forward squares, in order. `canChargePawn` needs both.
fn charge_steps(state: &GameState, piece: Piece, from: Square) -> Vec<Square> {
    let dir = direction(piece);
    let mut steps = Vec::with_capacity(2);
    for distance in [1_i8, 2] {
        let Some(square) = from.offset(0, dir * distance) else {
            break;
        };
        if !open(state, square) {
            break;
        }
        steps.push(square);
    }
    steps
}

/// Export for the shared pawn generator: the extra straight-ahead arrivals an
/// armed charging pawn may use (two and three squares, `addPawnDirectionalMoves`
/// main-RjrZ5fFP.js:345675-345682). Empty when this pawn is not charging.
///
/// The caller must route each square through `push_card_pawn_move` so promotion
/// choices are generated normally; a charge that fails to promote dies in
/// `resolve_end_after_move` (turn_effects.rs:35-39).
fn charge_advance(state: &GameState, piece: Piece, from: Square) -> Vec<Square> {
    if !state.card_runtime.charge_pawns.contains(&piece.id) || piece.kind != PieceKind::Pawn {
        return Vec::new();
    }
    let dir = direction(piece);
    let mut destinations = Vec::new();
    for distance in [2_i8, 3] {
        let Some(square) = from.offset(0, dir * distance) else {
            break;
        };
        if !open(state, square) {
            break;
        }
        destinations.push(square);
    }
    destinations
}

impl Engine {
    /// The 2- and 3-square straight arrivals for an armed charging pawn. Lead: call
    /// this from `generate_pawn_moves` and push each square with
    /// `push_card_pawn_move(..., promotion_rank, actions)`.
    #[allow(dead_code)] // Wired by the Lead in engine.rs generate_pawn_moves.
    pub(in crate::engine) fn charge_advance_destinations(
        &self,
        state: &GameState,
        piece: Piece,
        from: Square,
    ) -> Vec<Square> {
        charge_advance(state, piece, from)
    }
}

fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    instance_id: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (square, pawn) = piece_target(state, target)?;
    if !can_charge(state, actor, square, pawn) {
        return Err(EngineError::Invariant(
            "앞으로 3칸 전진할 수 있는 아군 폰을 골라야 합니다.".to_owned(),
        ));
    }
    if engine.is_royal_piece(state, pawn) {
        return Err(EngineError::Invariant(
            "왕실 기물은 질주할 수 없습니다.".to_owned(),
        ));
    }
    // `charge()` (main-RjrZ5fFP.js:352598-352607) first clears any pending dash and
    // then arms exactly one pawn for this turn.
    state.card_runtime.charge_pawns.clear();
    state.card_runtime.charge_pawns.insert(pawn.id);
    events.push(Event::EffectApplied {
        source_card_instance: instance_id,
        command: format!("charge:{}:{}", pawn.id, square),
    });
    Ok(())
}

fn actions(engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    for (square, piece) in web_ordered_pieces(state) {
        if !can_charge(state, card.owner, square, piece) || engine.is_royal_piece(state, piece) {
            continue;
        }
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::PieceAt(square),
        });
    }
}

/// `hasChargeCandidate` (main-RjrZ5fFP.js:352617-352623).
fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    require(
        web_ordered_pieces(state)
            .into_iter()
            .any(|(square, piece)| can_charge(state, owner, square, piece)),
        "앞으로 3칸 전진할 수 있는 아군 폰이 필요합니다.",
    )
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::{at, position};
    use super::*;

    fn card() -> CardInstance {
        CardInstance {
            instance_id: 1,
            definition_id: ID.to_owned(),
            owner: Player::White,
            half_stars: 0,
            charges: 1,
            available_after_turn: None,
        }
    }

    #[test]
    fn charge_arms_one_pawn_and_offers_only_pawns_with_two_open_squares() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("b2", Player::White, PieceKind::Pawn),
            ("c2", Player::White, PieceKind::Pawn),
            ("c3", Player::Black, PieceKind::Pawn),
        ]);
        let mut offered = Vec::new();
        actions(&engine, &state, &card(), &mut offered);
        let squares: BTreeSet<_> = offered
            .iter()
            .filter_map(|action| match action {
                Action::UseCard {
                    target: ActionTarget::PieceAt(square),
                    ..
                } => Some(*square),
                _ => None,
            })
            .collect();
        // b2 can dash; c2 is blocked by the enemy pawn two squares ahead.
        assert!(squares.contains(&"b2".parse().unwrap()));
        assert!(!squares.contains(&"c2".parse().unwrap()));
        let b2 = at(&state, "b2");
        play(
            &engine,
            &mut state,
            Player::White,
            1,
            &ActionTarget::PieceAt("b2".parse().unwrap()),
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(
            state
                .card_runtime
                .charge_pawns
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![b2.id]
        );
    }

    #[test]
    fn charge_advance_reaches_two_and_three_squares_and_stops_at_blocks() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("b2", Player::White, PieceKind::Pawn),
            ("d2", Player::White, PieceKind::Pawn),
        ]);
        // The dash needs both forward squares open, so b2 may land on b4 and b5.
        let b2 = at(&state, "b2");
        state.card_runtime.charge_pawns.insert(b2.id);
        assert_eq!(
            charge_advance(&state, b2, "b2".parse().unwrap()),
            vec!["b4".parse().unwrap(), "b5".parse().unwrap()]
        );
        assert_eq!(
            engine.charge_advance_destinations(&state, b2, "b2".parse().unwrap()),
            charge_advance(&state, b2, "b2".parse().unwrap())
        );
        // A blocker on the second square removes even the two-step arrival.
        let mut blocked = position(&[("b2", Player::White, PieceKind::Pawn)]);
        let b2 = at(&blocked, "b2");
        blocked.card_runtime.charge_pawns.insert(b2.id);
        blocked.board.set(
            "b4".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::Knight)),
        );
        assert!(charge_advance(&blocked, b2, "b2".parse().unwrap()).is_empty());
        // A blocker on the third square leaves the two-step arrival.
        let mut partial = position(&[("b2", Player::White, PieceKind::Pawn)]);
        let b2 = at(&partial, "b2");
        partial.card_runtime.charge_pawns.insert(b2.id);
        partial.board.set(
            "b5".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::Knight)),
        );
        assert_eq!(
            charge_advance(&partial, b2, "b2".parse().unwrap()),
            vec!["b4".parse().unwrap()]
        );
        // A pawn that was never armed gets nothing.
        let plain = position(&[("b2", Player::White, PieceKind::Pawn)]);
        assert!(charge_advance(&plain, at(&plain, "b2"), "b2".parse().unwrap()).is_empty());
        // Black pawns dash toward rank 1.
        let mut black = position(&[("b7", Player::Black, PieceKind::Pawn)]);
        let pawn = at(&black, "b7");
        black.card_runtime.charge_pawns.insert(pawn.id);
        assert_eq!(
            charge_advance(&black, pawn, "b7".parse().unwrap()),
            vec!["b5".parse().unwrap(), "b4".parse().unwrap()]
        );
    }

    #[test]
    fn charge_pawn_dies_after_a_dash_that_does_not_promote() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[("b2", Player::White, PieceKind::Pawn)]);
        let pawn = at(&state, "b2");
        play(
            &engine,
            &mut state,
            Player::White,
            1,
            &ActionTarget::PieceAt("b2".parse().unwrap()),
            &mut Vec::new(),
        )
        .unwrap();
        let mut dashed = state.clone();
        dashed.board.set("b2".parse().unwrap(), None);
        dashed.board.set("b4".parse().unwrap(), Some(pawn));
        // The dash is settled by the shared end-of-move pass once the move has landed
        // (turn_effects.rs:35-39): no promotion, so the pawn is removed.
        let mut events = Vec::new();
        engine.resolve_end_after_move(
            &mut dashed,
            pawn,
            MoveAction::new("b2".parse().unwrap(), "b4".parse().unwrap(), None),
            None,
            &mut events,
        );
        assert!(dashed.board.find_by_id(pawn.id).is_none());
        assert!(!dashed.card_runtime.charge_pawns.contains(&pawn.id));
        assert!(events.iter().any(|event| matches!(
            event,
            Event::PieceRemoved { cause, .. } if cause == crate::CHARGE
        )));
    }

    #[test]
    fn charge_pawn_survives_a_dash_that_promotes() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[("b6", Player::White, PieceKind::Pawn)]);
        let pawn = at(&state, "b6");
        play(
            &engine,
            &mut state,
            Player::White,
            1,
            &ActionTarget::PieceAt("b6".parse().unwrap()),
            &mut Vec::new(),
        )
        .unwrap();
        let mut promoted = state.clone();
        promoted.board.set("b6".parse().unwrap(), None);
        let mut queen = pawn;
        queen.kind = PieceKind::Queen;
        promoted.board.set("b8".parse().unwrap(), Some(queen));
        engine.resolve_end_after_move(
            &mut promoted,
            queen,
            MoveAction::new(
                "b6".parse().unwrap(),
                "b8".parse().unwrap(),
                Some(PieceKind::Queen),
            ),
            None,
            &mut Vec::new(),
        );
        assert!(promoted.board.find_by_id(pawn.id).is_some());
        assert!(!promoted.card_runtime.charge_pawns.contains(&pawn.id));
    }

    #[test]
    fn charge_rejects_enemy_wrong_kind_and_blocked_pawns() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("b2", Player::White, PieceKind::Pawn),
            ("g2", Player::White, PieceKind::Knight),
            ("g7", Player::Black, PieceKind::Pawn),
        ]);
        for target in ["g7", "g2"] {
            assert!(
                play(
                    &engine,
                    &mut state,
                    Player::White,
                    1,
                    &ActionTarget::PieceAt(target.parse().unwrap()),
                    &mut Vec::new(),
                )
                .is_err(),
                "{target}"
            );
        }
        state.board.set(
            "b4".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::Rook)),
        );
        assert!(
            play(
                &engine,
                &mut state,
                Player::White,
                1,
                &ActionTarget::PieceAt("b2".parse().unwrap()),
                &mut Vec::new(),
            )
            .is_err()
        );
        assert!(state.card_runtime.charge_pawns.is_empty());
    }
}
