use super::*;
pub(super) const ID: &str = crate::BRAINWASH;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

/// Battle value used for the "cheaper than the sacrifice" comparison.
///
/// Reuses the repo's single value table (`revelation::combat_value`, which is the
/// core encoding of the upstream ENCYCLOPEDIA table). Upstream `pieceCombatValue`
/// returns `null` for unvalued pieces, so a `0` entry means "cannot participate".
fn combat_value(kind: PieceKind) -> Option<u16> {
    match super::revelation::combat_value(kind) {
        0 => None,
        value => Some(value),
    }
}

/// A sacrifice or victim is a legal brainwash participant only when it is a
/// non-royal, non-monster piece with a finite value.
fn valued(state: &GameState, piece: Piece) -> Option<u16> {
    if royal(state, piece) || piece.kind == PieceKind::Monster {
        return None;
    }
    combat_value(piece.kind)
}

/// (sacrifice square, sacrifice piece, victim square) triples in web order.
///
/// Upstream `brainwashVictims` (L1423-1430) has no reachability requirement: the
/// sacrifice may be any non-royal ally and the victim any non-royal enemy with a
/// strictly lower value, wherever they stand. The capture-permission helper is
/// therefore deliberately not consulted.
fn pairs(state: &GameState, owner: Player) -> Vec<(Square, Piece, Square)> {
    let own = web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, piece)| piece.owner == owner)
        .filter_map(|(square, piece)| valued(state, piece).map(|value| (square, piece, value)))
        .collect::<Vec<_>>();
    let enemy = web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, piece)| piece.owner == owner.opponent())
        .filter_map(|(square, piece)| valued(state, piece).map(|value| (square, piece, value)))
        .collect::<Vec<_>>();
    let mut out = Vec::new();
    for (source_square, source, source_value) in &own {
        for (victim_square, _, victim_value) in &enemy {
            if victim_value < source_value {
                out.push((*source_square, *source, *victim_square));
            }
        }
    }
    out
}

fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let [sacrifice_square, victim_square] = two_square_target(target)?;
    let source = state
        .board
        .get(sacrifice_square)
        .ok_or_else(|| EngineError::Invariant("brainwash sacrifice disappeared".to_owned()))?;
    let victim = state
        .board
        .get(victim_square)
        .ok_or_else(|| EngineError::Invariant("brainwash victim disappeared".to_owned()))?;
    let valid = source.owner == actor
        && valued(state, source).is_some()
        && victim.owner == actor.opponent()
        && valued(state, victim)
            .zip(valued(state, source))
            .is_some_and(|(victim_value, source_value)| victim_value < source_value);
    if !valid {
        return Err(EngineError::Invariant(
            "brainwash requires a cheaper non-royal enemy piece".to_owned(),
        ));
    }
    // The sacrifice is a real removal (it feeds reaper souls, undead resurrection
    // and capture reactions), so use the capture path with the opponent credited.
    engine.remove_piece_for_card(
        state,
        sacrifice_square,
        source,
        actor.opponent(),
        ID,
        events,
    );
    if state
        .board
        .get(victim_square)
        .is_some_and(|p| p.id == victim.id)
    {
        // Identity and kind are preserved; only ownership changes. Everything the
        // old owner had attached must not travel with the piece.
        engine.clear_queens_gambit_protection(state, victim.id);
        state.card_runtime.moved_piece_ids.insert(victim.id);
        state.card_runtime.royal_piece_ids.remove(&victim.id);
        state.card_runtime.protected_piece_ids.remove(&victim.id);
        state.card_runtime.shielded_piece_ids.remove(&victim.id);
        state
            .card_runtime
            .outpost_protected_piece_ids
            .remove(&victim.id);
        state.card_runtime.parry_piece_ids.remove(&victim.id);
        state.card_runtime.stealth_piece_ids.remove(&victim.id);
        state.card_runtime.submerged_piece_ids.remove(&victim.id);
        state.card_runtime.wanted_piece_ids.remove(&victim.id);
        state.card_runtime.loyalist_piece_ids.remove(&victim.id);
        state.card_runtime.calling_cards.remove(&victim.id);
        state.card_runtime.vigilance_protection.remove(&victim.id);
        state.card_runtime.sacrifice_protection.remove(&victim.id);
        state.card_runtime.last_resistance.remove(&victim.id);
        state.card_runtime.coronation_protection.remove(&victim.id);
        state
            .card_runtime
            .twin_bonds
            .retain(|bond| bond.first_id != victim.id && bond.second_id != victim.id);
        state
            .card_runtime
            .siren_exposure
            .retain(|(siren, target), _| *siren != victim.id && *target != victim.id);
        state
            .card_runtime
            .feudal_contracts
            .retain(|contract| contract.pawn_id != victim.id && contract.guardian_id != victim.id);
        state.rule_runtime.grappler_bound.remove(&victim.id);
        state.rule_runtime.winter_frozen.remove(&victim.id);
        state
            .rule_runtime
            .winter_preview
            .retain(|id| *id != victim.id);
        let mut converted = victim;
        converted.owner = actor;
        state.board.set(victim_square, Some(converted));
        events.push(Event::EffectApplied {
            source_card_instance: 0,
            command: format!("brainwash:{}:{}", converted.id, victim_square),
        });
    }
    Ok(())
}

fn actions(_engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    for (sacrifice, _, victim) in pairs(state, card.owner) {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::Squares(vec![sacrifice, victim]),
        });
    }
}

fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    let values = |color: Player| {
        state
            .board
            .iter()
            .filter(|(_, piece)| piece.owner == color)
            .filter_map(|(_, piece)| valued(state, piece))
            .collect::<Vec<_>>()
    };
    let own = values(owner);
    let enemy = values(owner.opponent());
    require(
        own.iter()
            .any(|source| enemy.iter().any(|victim| victim < source)),
        "더 약한 비왕실 상대 기물과 희생할 아군 기물이 필요합니다.",
    )
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::{at, movement, pair, position};
    use super::*;

    fn use_card(
        engine: &Engine,
        state: &mut GameState,
        actor: Player,
        target: ActionTarget,
    ) -> Result<(), EngineError> {
        play(engine, state, actor, 1, &target, &mut Vec::new())
    }

    fn card(owner: Player) -> CardInstance {
        CardInstance {
            instance_id: 1,
            definition_id: crate::BRAINWASH.to_owned(),
            owner,
            half_stars: 0,
            charges: 1,
            available_after_turn: None,
        }
    }

    #[test]
    fn brainwash_sacrifices_a_friendly_piece_and_converts_a_cheaper_enemy() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("d4", Player::White, PieceKind::Rook),
            ("e5", Player::Black, PieceKind::Knight),
            ("b1", Player::Black, PieceKind::Queen),
        ]);
        let rook = at(&state, "d4");
        let knight = at(&state, "e5");
        use_card(&engine, &mut state, Player::White, pair("d4", "e5")).unwrap();
        // Sacrifice is gone from the board.
        assert!(state.board.get("d4".parse().unwrap()).is_none());
        assert!(state.board.find_by_id(rook.id).is_none());
        // Victim survives with identity and kind, now owned by the actor.
        let converted = at(&state, "e5");
        assert_eq!(converted.id, knight.id, "piece identity is preserved");
        assert_eq!(converted.kind, PieceKind::Knight, "kind is preserved");
        assert_eq!(converted.owner, Player::White);
        assert!(state.card_runtime.moved_piece_ids.contains(&knight.id));
        // Unrelated enemy pieces are untouched.
        assert_eq!(at(&state, "b1").kind, PieceKind::Queen);
        assert_eq!(at(&state, "b1").owner, Player::Black);
        // END_TURN policy is enforced by the engine, not the handler; the handler
        // itself should not flip the turn.
        assert_eq!(state.turn, Player::White);
    }

    #[test]
    fn brainwash_excludes_royal_pieces_on_both_sides_and_requires_a_strict_lower_value() {
        let engine = Engine::bundled_latest().unwrap();
        // The enemy king is never a valid victim even though the queen "outvalues" it.
        let mut state = position(&[("d1", Player::White, PieceKind::Queen)]);
        let before = state.clone();
        assert!(use_card(&engine, &mut state, Player::White, pair("d1", "h8")).is_err());
        assert_eq!(state, before);
        // Equal-value exchange is rejected too.
        let mut state = position(&[
            ("d4", Player::White, PieceKind::Bishop),
            ("e5", Player::Black, PieceKind::Bishop),
        ]);
        assert!(use_card(&engine, &mut state, Player::White, pair("d4", "e5")).is_err());
        // The friendly king cannot be the sacrifice.
        let mut state = position(&[("d4", Player::Black, PieceKind::Bishop)]);
        assert!(use_card(&engine, &mut state, Player::White, pair("a1", "d4")).is_err());
        assert!(state.board.iter().any(|(_, p)| p.kind == PieceKind::King));
    }

    #[test]
    fn brainwash_does_not_leak_old_owner_status_onto_the_converted_piece() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("d4", Player::White, PieceKind::Rook),
            ("e5", Player::Black, PieceKind::Knight),
        ]);
        let knight = at(&state, "e5");
        // Statuses that must not survive the owner change.
        state.card_runtime.protected_piece_ids.insert(knight.id);
        state.card_runtime.shielded_piece_ids.insert(knight.id);
        state.card_runtime.parry_piece_ids.insert(knight.id);
        state.card_runtime.stealth_piece_ids.insert(knight.id);
        state.card_runtime.wanted_piece_ids.insert(knight.id);
        state.card_runtime.vigilance_protection.insert(
            knight.id,
            TimedPieceStatus {
                by: Player::Black,
                remaining_turns: 3,
            },
        );
        use_card(&engine, &mut state, Player::White, pair("d4", "e5")).unwrap();
        assert_eq!(at(&state, "e5").owner, Player::White);
        assert!(!state.card_runtime.protected_piece_ids.contains(&knight.id));
        assert!(!state.card_runtime.shielded_piece_ids.contains(&knight.id));
        assert!(!state.card_runtime.parry_piece_ids.contains(&knight.id));
        assert!(!state.card_runtime.stealth_piece_ids.contains(&knight.id));
        assert!(!state.card_runtime.wanted_piece_ids.contains(&knight.id));
        assert!(
            !state
                .card_runtime
                .vigilance_protection
                .contains_key(&knight.id)
        );
    }

    #[test]
    fn brainwash_offers_pairs_in_web_order_and_rejects_missing_targets() {
        let engine = Engine::bundled_latest().unwrap();
        let state = position(&[
            ("d4", Player::White, PieceKind::Queen),
            ("a3", Player::White, PieceKind::Rook),
            ("e5", Player::Black, PieceKind::Knight),
            ("b7", Player::Black, PieceKind::Pawn),
        ]);
        let mut out = Vec::new();
        actions(&engine, &state, &card(Player::White), &mut out);
        let targets = out
            .iter()
            .map(|action| match action {
                Action::UseCard {
                    target: ActionTarget::Squares(squares),
                    ..
                } => squares.clone(),
                other => panic!("unexpected action {other:?}"),
            })
            .collect::<Vec<_>>();
        // Web order is a8..h8 downward, so b7 (rank 6) precedes e5 (rank 4) on
        // both sides of each pair; sacrifices keep the same web order.
        assert_eq!(
            targets,
            vec![
                vec!["d4".parse().unwrap(), "b7".parse().unwrap()],
                vec!["d4".parse().unwrap(), "e5".parse().unwrap()],
                vec!["a3".parse().unwrap(), "b7".parse().unwrap()],
                vec!["a3".parse().unwrap(), "e5".parse().unwrap()],
            ]
        );
        // Too few selections are rejected.
        let mut state = position(&[
            ("d4", Player::White, PieceKind::Queen),
            ("e5", Player::Black, PieceKind::Knight),
        ]);
        for target in [
            ActionTarget::None,
            ActionTarget::Squares(vec!["d4".parse().unwrap()]),
            pair("d4", "h3"),
        ] {
            assert!(use_card(&engine, &mut state, Player::White, target).is_err());
        }
        assert_eq!(at(&state, "e5").owner, Player::Black);
        assert_eq!(at(&state, "d4").owner, Player::White);
        let _ = movement("d4", "d5");
    }

    #[test]
    fn brainwash_eligibility_needs_a_cheaper_enemy_piece() {
        // A queen can always find cheaper prey on a normal board.
        let only_king = position(&[]);
        assert!(!eligibility(&only_king, Player::White, AcquisitionContext::Draft).is_eligible());
        let mut state = position(&[]);
        state.board.set(
            "d4".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Rook)),
        );
        state.board.set(
            "d5".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::Pawn)),
        );
        assert!(eligibility(&state, Player::White, AcquisitionContext::Draft).is_eligible());
        let strong_enemy = {
            let mut state = position(&[]);
            state.board.set(
                "d4".parse().unwrap(),
                Some(Piece::new(Player::White, PieceKind::Pawn)),
            );
            state.board.set(
                "d5".parse().unwrap(),
                Some(Piece::new(Player::Black, PieceKind::Queen)),
            );
            state
        };
        assert!(
            !eligibility(&strong_enemy, Player::White, AcquisitionContext::Draft).is_eligible()
        );
    }

    /// The catalog declares END_TURN for brainwash, so using it must complete the actor's turn.
    #[test]
    fn brainwash_ends_the_turn() {
        let engine = Engine::bundled_latest().unwrap();
        // The victim must be worth less than the sacrifice, so give up a queen for a pawn.
        let mut state = position(&[
            ("b2", Player::White, PieceKind::Queen),
            ("d4", Player::Black, PieceKind::Pawn),
        ]);
        state.actions_remaining = 2;
        state = engine
            .grant_card(&state, 1, crate::BRAINWASH, Player::White)
            .expect("brainwash is supported")
            .state;
        let after = engine
            .apply(
                &state,
                Action::UseCard {
                    instance_id: 1,
                    target: pair("b2", "d4"),
                },
            )
            .expect("the swap is legal");
        assert!(
            after
                .events
                .iter()
                .any(|event| matches!(event, Event::TurnEnded { .. })),
            "using brainwash must end the turn: {:?}",
            after.events
        );
        assert_eq!(after.state.turn, Player::Black);
    }
}
