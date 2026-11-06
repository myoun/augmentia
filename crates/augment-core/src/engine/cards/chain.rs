use super::*;

pub(super) const ID: &str = crate::CHAIN;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

/// Upstream's `CHAIN_MAX_DISTANCE` (main-RjrZ5fFP.js:2026). `isWithinChainRange`
/// (L4426-4429) compares Chebyshev distance, so a bond survives at exactly 2.
const CHAIN_MAX_DISTANCE: u8 = 2;

fn chebyshev(a: Square, b: Square) -> u8 {
    a.file().abs_diff(b.file()).max(a.rank().abs_diff(b.rank()))
}

/// `isChainTargetPiece` (main-RjrZ5fFP.js:346100-346103) plus the large-footprint
/// check in `chainSelectionContract` (L198109): any enemy piece, royals included,
/// except a multi-cell body. Upstream's exclusion list is exactly
/// `largeFootprint || colossus || big-rook || big-bishop`.
fn chain_target(state: &GameState, actor: Player, piece: Piece) -> bool {
    piece.owner == actor.opponent()
        && !state.card_runtime.large_pieces.contains_key(&piece.id)
        && !matches!(
            piece.kind,
            PieceKind::Colossus | PieceKind::BigRook | PieceKind::BigBishop
        )
}

fn pair_key(first: u32, second: u32) -> (u32, u32) {
    if first <= second {
        (first, second)
    } else {
        (second, first)
    }
}

fn already_bound(state: &GameState, first: u32, second: u32) -> bool {
    let key = pair_key(first, second);
    state
        .card_runtime
        .chains
        .iter()
        .any(|pair| pair_key(pair[0], pair[1]) == key)
}

fn play(
    _: &Engine,
    state: &mut GameState,
    actor: Player,
    instance_id: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    // The catalog selection is `squares` with min/max 2, and `actions` emits exactly
    // that target shape. `chainSelectionContract` (main-RjrZ5fFP.js:198102-198117)
    // validates both enemies, range 2 and the no-duplicate-bond rule.
    let [first_square, second_square] = two_square_target(target)?;
    let first = state
        .board
        .get(first_square)
        .ok_or_else(|| EngineError::Invariant("사슬 대상이 없습니다.".to_owned()))?;
    let second = state
        .board
        .get(second_square)
        .ok_or_else(|| EngineError::Invariant("사슬 대상이 없습니다.".to_owned()))?;
    let legal = first_square != second_square
        && chain_target(state, actor, first)
        && chain_target(state, actor, second)
        && first.id != second.id
        && chebyshev(first_square, second_square) <= CHAIN_MAX_DISTANCE
        && !already_bound(state, first.id, second.id);
    if !legal {
        return Err(EngineError::Invariant(
            "사슬로 묶을 상대 기물 2개를 골라야 합니다.".to_owned(),
        ));
    }
    let pair = [first.id.min(second.id), first.id.max(second.id)];
    state.card_runtime.chains.push(pair);
    events.push(Event::EffectApplied {
        source_card_instance: instance_id,
        command: format!("chain:{}:{}", pair[0], pair[1]),
    });
    Ok(())
}

fn actions(_: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    // `chainPairCandidates` (main-RjrZ5fFP.js:346124-346137): every unbound enemy
    // pair within range, in web board order.
    let enemies: Vec<(Square, Piece)> = web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, piece)| chain_target(state, card.owner, *piece))
        .collect();
    for (index, (first_square, first)) in enemies.iter().enumerate() {
        for (second_square, second) in enemies.iter().skip(index + 1) {
            if chebyshev(*first_square, *second_square) > CHAIN_MAX_DISTANCE
                || already_bound(state, first.id, second.id)
            {
                continue;
            }
            out.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::Squares(vec![*first_square, *second_square]),
            });
        }
    }
}

/// `hasChainPairCandidate` (main-RjrZ5fFP.js:346138-346140), used as the draft
/// requirement `ENEMY_CHAIN_PAIR`.
fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    let enemies: Vec<(Square, Piece)> = web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, piece)| chain_target(state, owner, *piece))
        .collect();
    let has_pair = enemies
        .iter()
        .enumerate()
        .any(|(index, (first_square, first))| {
            enemies
                .iter()
                .skip(index + 1)
                .any(|(second_square, second)| {
                    chebyshev(*first_square, *second_square) <= CHAIN_MAX_DISTANCE
                        && !already_bound(state, first.id, second.id)
                })
        });
    require(has_pair, "사슬로 묶을 상대 기물 2개가 필요합니다.")
}

/// Exported predicate for the shared move filter.
///
/// `isChainMoveAllowed` (main-RjrZ5fFP.js:346176-346189) projects the mover's
/// destination and rejects the move when any currently binding bond would end up
/// beyond `CHAIN_MAX_DISTANCE`. A bond that is already out of range never restricts
/// anything again, matching `breakOutOfRangeChainBonds` (L346199-346206).
#[allow(dead_code)] // Wired by the Lead in filter_card_state_moves (shared.rs:242).
pub(in crate::engine) fn chain_allows_move(
    state: &GameState,
    piece_id: u32,
    destination: Square,
) -> bool {
    state.card_runtime.chains.iter().all(|pair| {
        let [first_id, second_id] = *pair;
        if first_id != piece_id && second_id != piece_id {
            return true;
        }
        let partner_id = if first_id == piece_id {
            second_id
        } else {
            first_id
        };
        let Some((mover_square, _)) = state.board.find_by_id(piece_id) else {
            return true;
        };
        let Some((partner_square, _)) = state.board.find_by_id(partner_id) else {
            return true;
        };
        if chebyshev(mover_square, partner_square) > CHAIN_MAX_DISTANCE {
            return true;
        }
        chebyshev(destination, partner_square) <= CHAIN_MAX_DISTANCE
    })
}

impl Engine {
    /// `breakOutOfRangeChainBonds` (main-RjrZ5fFP.js:346199-346206): drop any bond
    /// whose partner is gone or whose distance has grown past the limit.
    #[allow(dead_code)] // Wired by the Lead at the move-application site.
    pub(in crate::engine) fn clean_card_chains(&self, state: &mut GameState) {
        state.card_runtime.chains.retain(|pair| {
            let Some((first_square, _)) = state.board.find_by_id(pair[0]) else {
                return false;
            };
            let Some((second_square, _)) = state.board.find_by_id(pair[1]) else {
                return false;
            };
            first_square != second_square
                && chebyshev(first_square, second_square) <= CHAIN_MAX_DISTANCE
        });
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::{at, pair, position};
    use super::super::tests::target_procedures::position as seated;
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
    fn chain_play_links_two_enemy_pieces_within_two_squares() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("d5", Player::Black, PieceKind::Rook),
            ("e6", Player::Black, PieceKind::Knight),
            ("a8", Player::Black, PieceKind::Rook),
        ]);
        let first = at(&state, "d5");
        let second = at(&state, "e6");
        play(
            &engine,
            &mut state,
            Player::White,
            1,
            &pair("d5", "e6"),
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(
            state.card_runtime.chains,
            vec![[first.id.min(second.id), first.id.max(second.id)]]
        );
        // Three files apart is out of range, and the same pair cannot be bound twice.
        assert!(
            play(
                &engine,
                &mut state,
                Player::White,
                1,
                &pair("d5", "a8"),
                &mut Vec::new(),
            )
            .is_err()
        );
        assert!(
            play(
                &engine,
                &mut state,
                Player::White,
                1,
                &pair("e6", "d5"),
                &mut Vec::new(),
            )
            .is_err()
        );
        assert_eq!(state.card_runtime.chains.len(), 1);
    }

    #[test]
    fn chain_play_accepts_royals_and_rejects_own_pieces_and_wrong_target_shapes() {
        let engine = Engine::bundled_latest().unwrap();
        // Royal enemies are legal chain targets upstream (only large bodies are not).
        let mut royal = position(&[
            ("a8", Player::Black, PieceKind::King),
            ("b7", Player::Black, PieceKind::Rook),
        ]);
        play(
            &engine,
            &mut royal,
            Player::White,
            1,
            &pair("a8", "b7"),
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(royal.card_runtime.chains.len(), 1);

        let mut state = position(&[
            ("d5", Player::White, PieceKind::Rook),
            ("e6", Player::Black, PieceKind::Knight),
        ]);
        // Own piece and a single-square target are both invalid.
        for target in [
            pair("d5", "e6"),
            ActionTarget::PieceAt("e6".parse().unwrap()),
        ] {
            assert!(
                play(
                    &engine,
                    &mut state,
                    Player::White,
                    1,
                    &target,
                    &mut Vec::new()
                )
                .is_err()
            );
        }
        assert!(state.card_runtime.chains.is_empty());
    }

    #[test]
    fn chain_actions_and_eligibility_offer_only_in_range_distinct_pairs() {
        let engine = Engine::bundled_latest().unwrap();
        let state = position(&[
            ("d5", Player::Black, PieceKind::Rook),
            ("e6", Player::Black, PieceKind::Knight),
            ("a8", Player::Black, PieceKind::Rook),
        ]);
        let mut offered = Vec::new();
        actions(&engine, &state, &card(), &mut offered);
        // Exactly one in-range enemy pair exists (the far-flung rooks and king do not
        // pair up); `actions` emits it in web board order.
        assert_eq!(offered.len(), 1);
        let Action::UseCard {
            target: ActionTarget::Squares(squares),
            ..
        } = &offered[0]
        else {
            panic!("chain must offer a two-square target: {offered:?}");
        };
        assert_eq!(
            squares.iter().copied().collect::<BTreeSet<_>>(),
            ["d5".parse().unwrap(), "e6".parse().unwrap()]
                .into_iter()
                .collect::<BTreeSet<_>>()
        );
        assert!(eligibility(&state, Player::White, AcquisitionContext::Draft).is_eligible());
        // Enemies scattered beyond range leave no pair to chain.
        let scattered = seated(
            &[
                ("a8", Player::Black, PieceKind::Rook),
                ("h1", Player::Black, PieceKind::Knight),
                ("e4", Player::White, PieceKind::King),
            ],
            Player::White,
        );
        assert!(!eligibility(&scattered, Player::White, AcquisitionContext::Draft).is_eligible());
        let mut none = Vec::new();
        actions(&engine, &scattered, &card(), &mut none);
        assert!(none.is_empty());
    }

    #[test]
    fn chain_predicate_forbids_separating_a_live_bond_past_two_squares() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("d5", Player::Black, PieceKind::Rook),
            ("e6", Player::Black, PieceKind::Knight),
        ]);
        play(
            &engine,
            &mut state,
            Player::White,
            1,
            &pair("d5", "e6"),
            &mut Vec::new(),
        )
        .unwrap();
        let rook = at(&state, "d5");
        let knight = at(&state, "e6");
        // Both squares stay within two Chebyshev steps of the partner on e6.
        assert!(chain_allows_move(&state, rook.id, "d6".parse().unwrap()));
        assert!(chain_allows_move(&state, rook.id, "d4".parse().unwrap()));
        assert!(chain_allows_move(&state, rook.id, "d8".parse().unwrap()));
        // Three or more steps away breaks the bond.
        assert!(!chain_allows_move(&state, rook.id, "b5".parse().unwrap()));
        assert!(!chain_allows_move(&state, rook.id, "a5".parse().unwrap()));
        // The partner is symmetrically constrained.
        assert!(!chain_allows_move(&state, knight.id, "a6".parse().unwrap()));
        // An unrelated piece is never restricted.
        let king = at(&state, "a1");
        assert!(chain_allows_move(&state, king.id, "h8".parse().unwrap()));
    }

    #[test]
    fn chain_predicate_releases_a_lost_partner_and_cleaning_drops_dead_bonds() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("d5", Player::Black, PieceKind::Rook),
            ("e6", Player::Black, PieceKind::Knight),
        ]);
        play(
            &engine,
            &mut state,
            Player::White,
            1,
            &pair("d5", "e6"),
            &mut Vec::new(),
        )
        .unwrap();
        let rook = at(&state, "d5");

        // Removing the partner lifts the restriction and the bond is cleaned away.
        let mut orphaned = state.clone();
        orphaned.board.set("e6".parse().unwrap(), None);
        assert!(chain_allows_move(&orphaned, rook.id, "a5".parse().unwrap()));
        engine.clean_card_chains(&mut orphaned);
        assert!(orphaned.card_runtime.chains.is_empty());

        // A pair that drifted out of range stops binding and is dropped.
        let mut drifted = state.clone();
        drifted.board.set("d5".parse().unwrap(), None);
        drifted.board.set("a8".parse().unwrap(), Some(rook));
        assert!(chebyshev("a8".parse().unwrap(), "e6".parse().unwrap()) > CHAIN_MAX_DISTANCE);
        assert!(chain_allows_move(&drifted, rook.id, "b8".parse().unwrap()));
        engine.clean_card_chains(&mut drifted);
        assert!(drifted.card_runtime.chains.is_empty());

        // A healthy bond survives cleaning.
        let mut healthy = state.clone();
        engine.clean_card_chains(&mut healthy);
        assert_eq!(healthy.card_runtime.chains.len(), 1);
    }
}
