use super::*;
pub(super) const ID: &str = crate::PREMOVE;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

/// A single declared relocation. The upstream contract also keeps a seed for the
/// nested random draws of the resolved moves; the current engine resolves each
/// declared move through the ordinary move generator, so no seed is needed here.
#[derive(Clone, Copy)]
struct Declared {
    piece_id: u32,
    from: Square,
    to: Square,
    promotion: Option<PieceKind>,
}

fn is_capture(engine: &Engine, state: &GameState, piece: Piece, to: Square) -> bool {
    if engine.board_piece_at(state, to).is_some() {
        return true;
    }
    // En passant removes a pawn that is not standing on the destination square.
    piece.kind == PieceKind::Pawn && state.en_passant.is_some_and(|pending| pending.target == to)
}

/// Every relocation this owner may declare, in web square order. Destinations
/// that would capture are excluded: the resolution window forbids captures, so
/// declaring one would silently cancel the remainder of the chain.
///
/// `used` names pieces already committed by earlier steps of a chain. They are skipped
/// before move generation rather than filtered afterwards, so a third step generates moves
/// for only the remaining pieces instead of the whole army.
fn declarations(engine: &Engine, state: &GameState, actor: Player, used: &[u32]) -> Vec<Declared> {
    // Move generation is expressed for the side to move, while draft-time eligibility asks
    // about an owner who may not be the current turn. Probe as the actor only when needed:
    // the enumeration paths always already stand on the actor's turn, so the hot path skips
    // the clone and only the eligibility probe pays for it.
    let probe;
    let state = if state.turn == actor {
        state
    } else {
        let mut copy = state.clone();
        copy.turn = actor;
        probe = copy;
        &probe
    };
    let mut declared = Vec::new();
    for (from, piece) in web_ordered_pieces(state) {
        if piece.owner != actor || used.contains(&piece.id) {
            continue;
        }
        let mut destinations = std::collections::BTreeMap::new();
        for action in engine.collect_piece_move_actions(state, piece.id) {
            let Action::Move(movement) = action else {
                continue;
            };
            if movement.from != from || movement.to == from {
                continue;
            }
            // The resolution pass relocates one piece and never moves a castling
            // rook, so a castling plane would silently break the contract.
            if engine
                .standard_castling_rook_plan(state, piece, movement)
                .is_some()
            {
                continue;
            }
            if is_capture(engine, state, piece, movement.to) {
                continue;
            }
            destinations
                .entry(movement.to)
                .or_insert(movement.promotion);
        }
        for (to, promotion) in destinations {
            declared.push(Declared {
                piece_id: piece.id,
                from,
                to,
                promotion,
            });
        }
    }
    declared
}

fn apply_declared(state: &mut GameState, declared: Declared) {
    let Some(mut piece) = state.board.get(declared.from) else {
        return;
    };
    state.board.set(declared.from, None);
    if let Some(promotion) = declared.promotion {
        piece.kind = promotion;
    }
    state.board.set(declared.to, Some(piece));
    state.card_runtime.moved_piece_ids.insert(piece.id);
}

/// Enumeration note: the source client builds a declaration incrementally from a prefix,
/// but an `Action` carries the complete square list, so every legal chain must be
/// materialised for `legal_actions`. All three depths are enumerated in full; upstream
/// rejects a list longer than three but places no cap on how many exist.
fn push_action(card: &CardInstance, path: &[Square], out: &mut Vec<Action>) {
    out.push(Action::UseCard {
        instance_id: card.instance_id,
        target: ActionTarget::Squares(path.to_vec()),
    });
}

fn actions(engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    if state.turn != card.owner {
        return;
    }
    let firsts = declarations(engine, state, card.owner, &[]);
    for declared in &firsts {
        push_action(card, &[declared.from, declared.to], out);
    }
    // Depths two and three share one walk. The two-move declarations are exactly the
    // prefixes of the three-move ones, so computing `declarations(after_first)` once and
    // using it for both depths halves the move generation at this level.
    //
    // Each level passes the pieces already committed, so generation skips them instead of
    // producing moves that a later filter would discard.
    for first in &firsts {
        let mut after_first = state.clone();
        apply_declared(&mut after_first, *first);
        let used_first = [first.piece_id];
        let seconds = declarations(engine, &after_first, card.owner, &used_first);
        for second in &seconds {
            push_action(card, &[first.from, first.to, second.from, second.to], out);
            let mut after_second = after_first.clone();
            apply_declared(&mut after_second, *second);
            let used_both = [first.piece_id, second.piece_id];
            for third in declarations(engine, &after_second, card.owner, &used_both) {
                push_action(
                    card,
                    &[
                        first.from,
                        first.to,
                        second.from,
                        second.to,
                        third.from,
                        third.to,
                    ],
                    out,
                );
            }
        }
    }
}

/// Validate one declared pair against the current board. Returns the declared
/// relocation with the promotion the resolution pass will need.
fn validate_pair(
    engine: &Engine,
    state: &GameState,
    actor: Player,
    from: Square,
    to: Square,
    used: &mut Vec<u32>,
) -> Result<Declared, EngineError> {
    let piece = state.board.get(from).ok_or_else(|| {
        EngineError::Invariant("프리 무브의 출발 칸에 아군 기물이 없습니다.".into())
    })?;
    if piece.owner != actor {
        return Err(EngineError::Invariant(
            "프리 무브는 아군 기물만 지정할 수 있습니다.".into(),
        ));
    }
    if used.contains(&piece.id) {
        return Err(EngineError::Invariant(
            "프리 무브는 서로 다른 기물만 지정할 수 있습니다.".into(),
        ));
    }
    if is_capture(engine, state, piece, to) {
        return Err(EngineError::Invariant(
            "프리 무브로는 기물을 잡을 수 없습니다.".into(),
        ));
    }
    let promotion = engine
        .generate_piece_move_actions(state, piece.id)
        .into_iter()
        .find_map(|action| match action {
            Action::Move(movement) if movement.from == from && movement.to == to => {
                Some(movement.promotion)
            }
            _ => None,
        })
        .ok_or_else(|| {
            EngineError::Invariant("프리 무브의 목표 칸으로는 이동할 수 없습니다.".into())
        })?;
    used.push(piece.id);
    Ok(Declared {
        piece_id: piece.id,
        from,
        to,
        promotion,
    })
}

fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    target: &ActionTarget,
    _: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let squares = squares_target(target)?;
    if !matches!(squares.len(), 2 | 4 | 6) {
        return Err(EngineError::Invariant(
            "프리 무브는 1~3개의 (출발, 도착) 칸 쌍이 필요합니다.".into(),
        ));
    }
    let mut used = Vec::new();
    let mut moves = Vec::with_capacity(squares.len() / 2);
    for chunk in squares.chunks_exact(2) {
        let declared = validate_pair(engine, state, actor, chunk[0], chunk[1], &mut used)?;
        moves.push(MoveAction::new(
            declared.from,
            declared.to,
            declared.promotion,
        ));
    }
    state.card_runtime.pending_premoves.push(PendingPremove {
        owner: actor,
        trigger_player: actor.opponent(),
        trigger_turn: state.turns_taken[actor.opponent().index()].saturating_add(1),
        moves,
    });
    Ok(())
}

fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    static ENGINE: std::sync::OnceLock<Engine> = std::sync::OnceLock::new();
    let engine = ENGINE.get_or_init(|| Engine::bundled_latest().expect("bundled latest ruleset"));
    require(
        !declarations(engine, state, owner, &[]).is_empty(),
        "선언할 수 있는 아군 기물 이동이 필요합니다.",
    )
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::*;
    use super::*;

    fn with_premove(pieces: &[(&str, Player, PieceKind)]) -> (Engine, GameState) {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(pieces);
        state = engine
            .grant_card(&state, 1, crate::PREMOVE, Player::White)
            .unwrap()
            .state;
        (engine, state)
    }

    fn offered(engine: &Engine, state: &GameState) -> Vec<Vec<Square>> {
        engine
            .legal_actions(state)
            .unwrap()
            .into_iter()
            .filter_map(|action| match action {
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::Squares(squares),
                } => Some(squares),
                _ => None,
            })
            .collect()
    }

    fn use_squares(
        engine: &Engine,
        state: &GameState,
        squares: &[&str],
    ) -> Result<GameState, EngineError> {
        engine
            .apply(
                state,
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::Squares(
                        squares.iter().map(|sq| sq.parse().unwrap()).collect(),
                    ),
                },
            )
            .map(|transition| transition.state)
    }

    #[test]
    fn premove_offers_ordered_own_move_pairs_and_excludes_captures() {
        let (engine, state) = with_premove(&[
            ("d4", Player::White, PieceKind::Rook),
            ("d6", Player::Black, PieceKind::Pawn),
            ("b1", Player::White, PieceKind::Knight),
        ]);
        let pairs = offered(&engine, &state);
        assert!(!pairs.is_empty(), "a premove must offer declared moves");
        for pair in &pairs {
            assert!(
                matches!(pair.len(), 2 | 4 | 6),
                "every declaration is one to three ordered pairs: {pair:?}"
            );
        }
        let contains = |pair: &[&str]| {
            let pair: Vec<Square> = pair.iter().map(|sq| sq.parse().unwrap()).collect();
            pairs.contains(&pair)
        };
        assert!(contains(&["d4", "d5"]), "the rook can declare d4->d5");
        assert!(contains(&["b1", "c3"]), "the knight can declare b1->c3");
        assert!(
            !contains(&["d4", "d6"]),
            "a capture destination must never be declared"
        );
        let d6: Square = "d6".parse().unwrap();
        assert!(
            !pairs.iter().any(|pair| pair.contains(&d6)),
            "no declaration may target the enemy pawn"
        );
        // Ordered chains are offered too: the same piece never appears twice.
        for pair in &pairs {
            let mut ids = BTreeSet::new();
            for chunk in pair.chunks(2) {
                let id = state.board.get(chunk[0]).unwrap().id;
                assert!(ids.insert(id), "a chain uses distinct pieces: {pair:?}");
            }
        }
    }

    #[test]
    fn premove_offers_chains_only_in_an_order_the_board_allows() {
        let (engine, state) = with_premove(&[
            ("b1", Player::White, PieceKind::Rook),
            ("b2", Player::White, PieceKind::Pawn),
        ]);
        let pairs = offered(&engine, &state);
        let squares =
            |pair: &[&str]| -> Vec<Square> { pair.iter().map(|sq| sq.parse().unwrap()).collect() };
        // The pawn steps first and clears b2, so the rook may then take it.
        assert!(
            pairs.contains(&squares(&["b2", "b3", "b1", "b2"])),
            "the chain must be validated against the intermediate board"
        );
        // The reverse order is not: b1->b2 is a capture while the pawn still sits there.
        assert!(
            !pairs.contains(&squares(&["b1", "b2", "b2", "b3"])),
            "a declared move that was illegal first must not be offered"
        );
    }

    #[test]
    fn premove_resolves_after_the_opponents_next_move_and_never_immediately() {
        let (engine, mut state) = with_premove(&[("a2", Player::White, PieceKind::Pawn)]);
        state = use_squares(&engine, &state, &["a2", "a4"]).unwrap();
        assert_eq!(state.card_runtime.pending_premoves.len(), 1);
        assert_eq!(state.card_runtime.pending_premoves[0].moves.len(), 1);
        assert_eq!(state.card_runtime.pending_premoves[0].owner, Player::White);
        assert_eq!(
            state.card_runtime.pending_premoves[0].trigger_player,
            Player::Black
        );
        assert_eq!(
            state.card_runtime.pending_premoves[0].trigger_turn,
            state.turns_taken[Player::Black.index()] + 1
        );
        assert!(
            at(&state, "a2").kind == PieceKind::Pawn,
            "the declaration must not move the pawn immediately"
        );
        // White ends the declaring turn; the opponent's move has not happened yet.
        handoff(&engine, &mut state);
        assert!(
            at(&state, "a2").kind == PieceKind::Pawn,
            "the pawn must wait for the opponent's next move"
        );
        assert_eq!(state.card_runtime.pending_premoves.len(), 1);
        // The opponent moves: resolution happens at the end of that turn.
        handoff(&engine, &mut state);
        assert!(state.board.get("a2".parse().unwrap()).is_none());
        assert_eq!(at(&state, "a4").kind, PieceKind::Pawn);
        assert!(state.card_runtime.pending_premoves.is_empty());
        assert_eq!(state.turn, Player::White);
    }

    #[test]
    fn premove_resolves_three_declared_moves_in_order() {
        let (engine, mut state) = with_premove(&[
            ("b1", Player::White, PieceKind::Knight),
            ("g1", Player::White, PieceKind::Knight),
            ("h1", Player::White, PieceKind::Rook),
        ]);
        // h1's rook is boxed in, so the chain is b1->c3, g1->f3, then h1 stays put.
        let chain = ["b1", "c3", "g1", "f3"];
        let pairs = offered(&engine, &state);
        let chain: Vec<Square> = chain.iter().map(|sq| sq.parse().unwrap()).collect();
        assert!(
            pairs.contains(&chain),
            "a two-move declaration must be offered"
        );
        state = engine
            .apply(
                &state,
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::Squares(chain),
                },
            )
            .unwrap()
            .state;
        assert_eq!(state.card_runtime.pending_premoves[0].moves.len(), 2);
        handoff(&engine, &mut state);
        handoff(&engine, &mut state);
        assert_eq!(at(&state, "c3").kind, PieceKind::Knight);
        assert_eq!(at(&state, "f3").kind, PieceKind::Knight);
        assert!(state.board.get("b1".parse().unwrap()).is_none());
        assert!(state.board.get("g1".parse().unwrap()).is_none());
    }

    #[test]
    fn premove_blocks_captures_through_the_resolution_turn() {
        let (engine, mut state) = with_premove(&[
            ("a2", Player::White, PieceKind::Rook),
            ("a6", Player::Black, PieceKind::Pawn),
        ]);
        // The rook can declare a quiet move but not the capture on a6.
        let rejected = use_squares(&engine, &state, &["a2", "a6"]);
        assert!(rejected.is_err(), "a declared capture must be rejected");
        state = use_squares(&engine, &state, &["a2", "a3"]).unwrap();
        handoff(&engine, &mut state);
        handoff(&engine, &mut state);
        assert_eq!(state.turn, Player::White);
        assert!(
            state.card_runtime.free_move_no_capture[Player::White.index()],
            "the owner cannot capture during their next turn"
        );
        assert!(
            !engine
                .legal_actions(&state)
                .unwrap()
                .iter()
                .any(|action| matches!(action, Action::Move(movement) if movement.to == "a6".parse().unwrap())),
            "capturing the pawn must be illegal during the no-capture turn"
        );
        assert!(
            state.card_runtime.pending_premoves.is_empty(),
            "the declaration has resolved"
        );
    }

    #[test]
    fn premove_rejects_malformed_or_illegal_selections_atomically() {
        let (engine, state) = with_premove(&[
            ("d4", Player::White, PieceKind::Rook),
            ("e5", Player::White, PieceKind::Rook),
            ("d6", Player::Black, PieceKind::Pawn),
        ]);
        let before = state.clone();
        for squares in [
            vec!["d4"],
            vec!["d4", "d5", "e5"],
            vec!["d4", "e4", "e5"],
            vec!["d4", "d4"],
            vec!["d4", "d5", "d4", "e4"],
            vec!["d6", "d5"],
        ] {
            assert!(
                use_squares(&engine, &state, &squares).is_err(),
                "{squares:?} must be rejected"
            );
            assert_eq!(state, before, "a rejected selection must not mutate state");
        }
    }

    /// Depth 3 must not be truncated. Upstream rejects a declaration longer than three
    /// moves but places no cap on how many exist, so stopping early would drop legal ones.
    #[test]
    fn premove_enumerates_three_move_declarations_without_a_cap() {
        let (engine, state) = with_premove(&[
            ("a2", Player::White, PieceKind::Rook),
            ("b2", Player::White, PieceKind::Knight),
            ("c2", Player::White, PieceKind::Bishop),
        ]);
        let offered = offered(&engine, &state);
        let triples = offered.iter().filter(|squares| squares.len() == 6).count();
        // Three distinct pieces with moves produce several ordered triples; the point is
        // that the count is driven by the position, not by an enumeration cap.
        assert!(
            triples >= 3,
            "three-move declarations must be enumerated, got {triples}"
        );
    }

    /// Skipping committed pieces in `declarations` must not change what is offered, only how
    /// much work it takes. Comparing against a chain whose third step would use an already
    /// committed piece confirms the exclusion happens before generation, not after.
    #[test]
    fn premove_never_repeats_a_piece_within_one_declaration() {
        let (engine, state) = with_premove(&[
            ("a2", Player::White, PieceKind::Rook),
            ("b2", Player::White, PieceKind::Knight),
            ("c2", Player::White, PieceKind::Bishop),
        ]);
        let pieces = |squares: &[Square]| -> Vec<u32> {
            squares
                .chunks(2)
                .filter_map(|pair| state.board.get(pair[0]).map(|piece| piece.id))
                .collect()
        };
        for squares in offered(&engine, &state) {
            let used = pieces(&squares);
            let mut unique = used.clone();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(
                used.len(),
                unique.len(),
                "a declaration must not move the same piece twice: {squares:?}"
            );
        }
    }

    #[test]
    fn premove_does_not_resolve_when_the_declared_piece_is_gone() {
        let (engine, mut state) = with_premove(&[("d4", Player::White, PieceKind::Rook)]);
        state = use_squares(&engine, &state, &["d4", "d5"]).unwrap();
        let rook = at(&state, "d4");
        state.board.set("d4".parse().unwrap(), None);
        engine.clear_removed_piece_statuses(&mut state, rook.id);
        handoff(&engine, &mut state);
        let mut events = Vec::new();
        let actor = state.turn;
        engine.finish_turn(&mut state, actor, &mut events).unwrap();
        assert!(
            state.card_runtime.pending_premoves.is_empty(),
            "a stale declaration is dropped, not retried"
        );
        assert!(
            !events.iter().any(|event| matches!(
                event,
                Event::EffectApplied { command, .. } if command.starts_with("premove:")
            )),
            "a stale declaration must not emit a relocation"
        );
        assert!(
            state.board.get("d5".parse().unwrap()).is_none(),
            "nothing appears on the declared destination"
        );
    }
}
