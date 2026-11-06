use super::*;
pub(super) const ID: &str = crate::VANISH;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

/// The live V15 settlement draws a vanishing king with this weight; every other
/// piece weighs 100 (JS:212186-212189, `weight: isVanishingKing ? 65 : 100`).
const KING_WEIGHT: u64 = 65;
const PIECE_WEIGHT: u64 = 100;

fn play(
    _engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    if target != &ActionTarget::None {
        return Err(EngineError::Invariant(
            "vanish must not have a target".into(),
        ));
    }
    state.card_runtime.vanish[actor.index()] = true;
    Ok(())
}

fn actions(_engine: &Engine, _state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    out.push(Action::UseCard {
        instance_id: card.instance_id,
        target: ActionTarget::None,
    });
}

fn eligibility(_state: &GameState, _player: Player, _: AcquisitionContext) -> CardEligibility {
    // Live `isCardDrawableForColor` (JS:235519): vanish is always drawable. The
    // global AlreadyAcquired check rejects a second copy.
    CardEligibility::Eligible
}

impl Engine {
    /// Settlement for `vanish`, called once per turn end after the moved piece
    /// resolves. Contract (catalog): once either side has played the card, every
    /// turn end removes one random piece from each side. Live
    /// `resolveVanishingForTurnStart` excludes kings while the game is on its
    /// first shared turn (`currentPly < 2`) and afterwards draws them with the
    /// reduced 0.65 weight, which is what this function reproduces.
    ///
    /// Each removal is a capture for the other side, matching
    /// `removeVanishingPiece`, which pushes the vanished piece onto the
    /// opponent's capture list. The caller must run `resolve_missing_king`
    /// afterwards so a vanished royal ends the game instead of leaving a
    /// kingless board.
    pub(in crate::engine) fn settle_vanish(
        &self,
        state: &mut GameState,
        _actor: Player,
        events: &mut Vec<Event>,
    ) {
        if !state.card_runtime.vanish[0] && !state.card_runtime.vanish[1] {
            return;
        }
        // Live `currentPly(position) < 2`: kings cannot vanish during the first
        // two half-turns. This runs before `ply` is incremented, so `ply == 0`
        // is the same window as the live check after the first move.
        let kings_excluded = state.ply == 0;
        for side in [Player::White, Player::Black] {
            if state.status != GameStatus::Ongoing {
                return;
            }
            let candidates = vanishing_candidates(state, side, kings_excluded);
            // The live random plan allocates a request per side even for an empty
            // pool, so the roll is consumed unconditionally.
            let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
            events.push(Event::RandomSample {
                purpose: "vanish".to_owned(),
                roll_53: roll,
            });
            if index.is_none() {
                continue;
            }
            let total = candidates
                .iter()
                .map(|(_, piece)| weight(state, *piece))
                .sum::<u64>();
            let mut cursor = roll % total;
            let mut chosen = candidates[candidates.len() - 1];
            for &(square, piece) in &candidates {
                let piece_weight = weight(state, piece);
                if cursor < piece_weight {
                    chosen = (square, piece);
                    break;
                }
                cursor -= piece_weight;
            }
            let (square, piece) = chosen;
            self.remove_piece_for_card(state, square, piece, side.opponent(), ID, events);
        }
    }
}

fn weight(state: &GameState, piece: Piece) -> u64 {
    if is_king(state, piece) {
        KING_WEIGHT
    } else {
        PIECE_WEIGHT
    }
}

/// `isKingAugmentRole` in the live client: royal identity pieces, plus merchants
/// once the September 18 balance made them king recipients.
fn is_king(state: &GameState, piece: Piece) -> bool {
    royal(state, piece) || piece.kind == PieceKind::Merchant
}

/// `vanishingPieceCandidates`: every piece of the side, installations excluded.
fn vanishing_candidates(
    state: &GameState,
    side: Player,
    kings_excluded: bool,
) -> Vec<(Square, Piece)> {
    web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, piece)| {
            piece.owner == side
                && !matches!(piece.kind, PieceKind::Barrier | PieceKind::Football)
                && !(kings_excluded && is_king(state, *piece))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::*;
    use super::*;

    fn card() -> CardInstance {
        CardInstance {
            instance_id: 1,
            definition_id: crate::VANISH.to_owned(),
            owner: Player::White,
            half_stars: 9,
            charges: 1,
            available_after_turn: None,
        }
    }

    fn tick(engine: &Engine, state: &mut GameState) -> Vec<Event> {
        let actor = state.turn;
        let mut events = Vec::new();
        engine.settle_vanish(state, actor, &mut events);
        events
    }

    fn removals(events: &[Event]) -> Vec<Piece> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::PieceRemoved { piece, cause, .. } if cause == crate::VANISH => Some(*piece),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn vanish_play_sets_the_flag_and_offers_a_targetless_action() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[("b1", Player::White, PieceKind::Rook)]);
        state.add_card(card());
        let actor = state.turn;
        super::play(
            &engine,
            &mut state,
            actor,
            1,
            &ActionTarget::None,
            &mut Vec::new(),
        )
        .unwrap();
        assert!(state.card_runtime.vanish[Player::White.index()]);
        assert!(!state.card_runtime.vanish[Player::Black.index()]);

        let mut out = Vec::new();
        super::actions(&engine, &state, &state.cards[0].clone(), &mut out);
        assert_eq!(
            out,
            vec![Action::UseCard {
                instance_id: 1,
                target: ActionTarget::None
            }]
        );
        assert!(
            engine
                .apply(
                    &state,
                    Action::UseCard {
                        instance_id: 1,
                        target: single("b1"),
                    }
                )
                .is_err(),
            "vanish takes no target"
        );
    }

    #[test]
    fn vanish_settles_only_when_played_and_removes_one_piece_from_each_side() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("b1", Player::White, PieceKind::Rook),
            ("c1", Player::White, PieceKind::Knight),
            ("b8", Player::Black, PieceKind::Rook),
            ("c8", Player::Black, PieceKind::Knight),
        ]);
        let events = tick(&engine, &mut state);
        assert!(events.is_empty(), "no settlement before the card is played");

        state.card_runtime.vanish[Player::White.index()] = true;
        state.turns_taken = [1, 1];
        let events = tick(&engine, &mut state);
        let removed = removals(&events);
        assert_eq!(removed.len(), 2, "one piece per side");
        assert!(removed.iter().any(|piece| piece.owner == Player::White));
        assert!(removed.iter().any(|piece| piece.owner == Player::Black));
        assert_eq!(
            state
                .board
                .iter()
                .filter(|(_, piece)| piece.owner == Player::White)
                .count(),
            2,
            "white keeps its king plus one piece"
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(
                    event,
                    Event::RandomSample { purpose, .. } if purpose == "vanish"
                ))
                .count(),
            2
        );
    }

    #[test]
    fn vanish_excludes_kings_on_the_first_shared_turn_then_allows_them() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[("b1", Player::White, PieceKind::Rook)]);
        state.card_runtime.vanish[Player::Black.index()] = true;
        // Shared turn 0: only the rook can vanish, never a king.
        let events = tick(&engine, &mut state);
        let removed = removals(&events);
        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0].kind, PieceKind::Rook);
        assert!(state.board.get("a1".parse().unwrap()).is_some());
        assert!(state.board.get("h8".parse().unwrap()).is_some());

        // Once both sides completed a turn, kings join the pool.
        let mut royal = position(&[]);
        royal.card_runtime.vanish[Player::White.index()] = true;
        royal.ply = 2;
        let events = tick(&engine, &mut royal);
        let removed = removals(&events);
        assert_eq!(removed.len(), 2);
        assert!(removed.iter().all(|piece| piece.kind == PieceKind::King));
    }

    #[test]
    fn vanish_removal_of_a_royal_is_finished_by_the_shared_missing_king_resolver() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[]);
        state.card_runtime.vanish[Player::White.index()] = true;
        state.ply = 2;
        let mut events = Vec::new();
        let actor = state.turn;
        engine.settle_vanish(&mut state, actor, &mut events);
        assert_eq!(
            state
                .board
                .iter()
                .filter(|(_, piece)| piece.kind == PieceKind::King)
                .count(),
            0
        );
        // The settlement itself does not decide the game; the caller's
        // `resolve_missing_king` (already part of the engine transition) does.
        // Live reports a simultaneous royal loss as an environmental draw only
        // when a RULE card is active; otherwise the shared resolver's ordering
        // ends the game on the first defeated side. Assert the invariant that
        // matters here: the game no longer continues with a kingless board.
        engine.resolve_missing_king(&mut state, &mut events);
        assert_ne!(state.status, GameStatus::Ongoing);
    }

    #[test]
    fn vanish_draw_is_seeded_and_reproducible() {
        let engine = Engine::bundled_latest().unwrap();
        let base = position(&[
            ("b1", Player::White, PieceKind::Rook),
            ("c1", Player::White, PieceKind::Knight),
            ("b8", Player::Black, PieceKind::Rook),
            ("c8", Player::Black, PieceKind::Knight),
        ]);
        let mut left = base.clone();
        left.card_runtime.vanish[Player::White.index()] = true;
        left.turns_taken = [1, 1];
        let mut right = left.clone();
        let left_events = tick(&engine, &mut left);
        let right_events = tick(&engine, &mut right);
        assert_eq!(left, right);
        assert_eq!(left_events, right_events);
    }

    #[test]
    fn vanish_is_always_drawable_and_rejects_a_second_acquisition() {
        // Live `isCardDrawableForColor` (JS:235519) returns true unconditionally
        // for vanish; only the global duplicate check stops a second copy.
        let empty = position(&[]);
        assert!(super::eligibility(&empty, Player::White, AcquisitionContext::Draft).is_eligible());
        let with_piece = position(&[("b1", Player::White, PieceKind::Rook)]);
        assert!(
            super::eligibility(&with_piece, Player::White, AcquisitionContext::Draft).is_eligible()
        );
        assert!(
            crate::check_card_eligibility(
                &with_piece,
                crate::VANISH,
                Player::White,
                AcquisitionContext::Draft
            )
            .is_eligible()
        );
        let mut second = with_piece.clone();
        second.add_card(super::tests::card());
        second.cards[0].owner = Player::Black;
        assert!(
            crate::check_card_eligibility(
                &second,
                crate::VANISH,
                Player::White,
                AcquisitionContext::Draft
            )
            .reason()
            .is_some()
        );
    }
}
