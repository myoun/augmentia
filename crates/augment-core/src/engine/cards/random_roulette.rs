use super::*;
pub(super) const ID: &str = crate::RANDOM_ROULETTE;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

/// Outcome pool in upstream source order (`RANDOM_ROULETTE_PIECE_TYPES` for the
/// captured catalog: the pre-audit pool plus the audit additions).
pub(super) const OUTCOMES: &[PieceKind] = &[
    PieceKind::Pawn,
    PieceKind::Knight,
    PieceKind::Bishop,
    PieceKind::Rook,
    PieceKind::Queen,
    PieceKind::Colossus,
    PieceKind::BigRook,
    PieceKind::Protestant,
    PieceKind::Herald,
    PieceKind::Cannon,
    PieceKind::Fanatic,
    PieceKind::PrimeMinister,
    PieceKind::Alibaba,
    PieceKind::Amazon,
    PieceKind::Cardinal,
    PieceKind::Unicorn,
    PieceKind::Jester,
    PieceKind::Camel,
    PieceKind::Log,
    PieceKind::Hook,
    PieceKind::Grasshopper,
    PieceKind::Dragon,
    PieceKind::Man,
    PieceKind::Assassin,
    PieceKind::Knightmaster,
    PieceKind::StandardBearer,
    PieceKind::Recruiter,
    PieceKind::Squire,
    PieceKind::Checker,
    PieceKind::CheckerKing,
    PieceKind::Wizard,
    PieceKind::Alfil,
    PieceKind::Guard,
    PieceKind::Reaper,
    PieceKind::Idol,
    PieceKind::BabyBear,
    PieceKind::Lobster,
    PieceKind::Missionary,
    PieceKind::Bear,
    PieceKind::Windmill,
    PieceKind::SiegeRam,
    PieceKind::MagicGirl,
    PieceKind::Berserker,
    PieceKind::Slime,
    PieceKind::Siren,
    PieceKind::Trickster,
    PieceKind::Undead,
    PieceKind::Campfire,
    PieceKind::Hedgehog,
    PieceKind::Princess,
    PieceKind::Thief,
    PieceKind::Brutus,
    PieceKind::Clockwork,
    PieceKind::Parrot,
    PieceKind::Paladin,
    PieceKind::Octopus,
    PieceKind::Grappler,
    PieceKind::RevolvingDoor,
    PieceKind::DonQuixote,
    PieceKind::Medium,
    PieceKind::BerolinaPawn,
    PieceKind::Chancellor,
    PieceKind::Archbishop,
    PieceKind::Killer,
    PieceKind::Beacon,
    PieceKind::TransmissionTower,
    PieceKind::Barrier,
];

/// `colossus`, `big-rook` and `big-bishop` need a free 2x2 landing body.
const LARGE_OUTCOMES: &[PieceKind] = &[
    PieceKind::Colossus,
    PieceKind::BigRook,
    PieceKind::BigBishop,
];

fn valid_target(engine: &Engine, state: &GameState, piece: Piece) -> bool {
    !engine.is_royal_piece(state, piece) && crate::latest::latest_major_kind(piece.kind)
}

fn large_piece_anchor_blocked(state: &GameState, piece_id: u32, cell: Square) -> bool {
    match state.board.get(cell) {
        Some(occupant) => occupant.id != piece_id,
        None => {
            state.card_runtime.black_holes.contains(&cell)
                // A 2x2 body cannot stand on a wall, a crown or a collapsed cell.
                || state.card_runtime.barricades.contains(&cell)
                || state.card_runtime.crown_pos == Some(cell)
                || (state.card_runtime.collapsed_files_ranks
                    && (matches!(cell.file(), 0 | 7) || matches!(cell.rank(), 0 | 7)))
                || state
                    .card_runtime
                    .pending_scarecrows
                    .iter()
                    .any(|pending| pending.square == cell)
                || state
                    .card_runtime
                    .pending_lobsters
                    .iter()
                    .any(|pending| pending.square == cell)
                || state
                    .card_runtime
                    .pending_portals
                    .iter()
                    .any(|pending| pending.squares.contains(&cell))
        }
    }
}

/// Anchors whose 2x2 body contains `square` and is otherwise free. A piece that
/// is already large keeps its current body; it cannot slide onto a new one.
fn large_anchors(state: &GameState, square: Square, piece: Piece) -> Vec<Square> {
    let candidates = if LARGE_OUTCOMES.contains(&piece.kind)
        || state.card_runtime.large_pieces.contains_key(&piece.id)
    {
        vec![square]
    } else {
        [
            Some(square),
            square.offset(-1, 0),
            square.offset(0, 1),
            square.offset(-1, 1),
        ]
        .into_iter()
        .flatten()
        .collect()
    };
    let mut seen = BTreeSet::new();
    candidates
        .into_iter()
        .filter(|anchor| seen.insert(*anchor))
        .filter(|anchor| {
            let cells = large_piece_cells(*anchor);
            cells.len() == 4
                && cells
                    .iter()
                    .all(|cell| !large_piece_anchor_blocked(state, piece.id, *cell))
        })
        .collect()
}

fn apply_transformed_runtime(
    engine: &Engine,
    state: &mut GameState,
    kind: PieceKind,
    piece_id: u32,
    square: Square,
    events: &mut Vec<Event>,
) {
    match kind {
        PieceKind::Colossus | PieceKind::BigRook | PieceKind::BigBishop => {
            let hp = if kind == PieceKind::Colossus { 3 } else { 2 };
            state.card_runtime.large_pieces.insert(
                piece_id,
                LargePieceStatus {
                    anchor: square,
                    hp,
                    max_hp: hp,
                },
            );
        }
        PieceKind::BabyBear => {
            state
                .card_runtime
                .baby_bear_grow_turns
                .insert(piece_id, state.shared_turn_count().saturating_add(7));
        }
        PieceKind::Bear | PieceKind::Hedgehog => {
            state.card_runtime.bear_retaliations.insert(piece_id, 2);
        }
        PieceKind::Thief => {
            state.card_runtime.submerged_piece_ids.insert(piece_id);
            state.card_runtime.wanted_piece_ids.insert(piece_id);
        }
        PieceKind::Trickster => {
            let kind = engine.choose_trickster_kind(state, events);
            state
                .card_runtime
                .trickster_move_kinds
                .insert(piece_id, kind);
        }
        _ => {}
    }
    if state.card_runtime.monochrome_chess {
        state
            .rule_runtime
            .mono_shades
            .insert(piece_id, (square.file() + square.rank()) % 2);
    }
}

fn play(
    engine: &Engine,
    state: &mut GameState,
    _: Player,
    _: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (square, piece) = piece_target(state, target)?;
    if !valid_target(engine, state, piece) {
        return Err(EngineError::Invariant(
            "랜덤룰렛은 비왕실 메이저 기물만 대상으로 합니다.".into(),
        ));
    }
    let anchors = large_anchors(state, square, piece);
    let outcomes: Vec<_> = OUTCOMES
        .iter()
        .copied()
        .filter(|kind| {
            *kind != piece.kind && (!LARGE_OUTCOMES.contains(kind) || !anchors.is_empty())
        })
        .collect();
    if outcomes.is_empty() {
        return Err(EngineError::Invariant(
            "사용할 수 있는 변형 결과가 없습니다.".into(),
        ));
    }
    let (roll, index) = state.rng_mut().web_choice_index(outcomes.len());
    events.push(Event::RandomSample {
        purpose: "random-roulette".to_owned(),
        roll_53: roll,
    });
    let kind = outcomes[index.expect("roulette outcomes are non-empty")];
    let destination = if LARGE_OUTCOMES.contains(&kind) {
        let (roll, index) = state.rng_mut().web_choice_index(anchors.len());
        events.push(Event::RandomSample {
            purpose: "random-roulette-large-anchor".to_owned(),
            roll_53: roll,
        });
        anchors[index.expect("large outcome requires an anchor")]
    } else {
        square
    };
    engine.clear_removed_piece_statuses(state, piece.id);
    if destination != square {
        state.board.set(square, None);
        state.board.set(destination, Some(piece));
    }
    let converted = engine.transform_current_piece(state, destination, kind, true)?;
    apply_transformed_runtime(engine, state, kind, converted.id, destination, events);
    Ok(())
}

fn actions(engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    for (square, piece) in web_ordered_pieces(state) {
        if valid_target(engine, state, piece) {
            out.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::PieceAt(square),
            });
        }
    }
}

fn eligibility(state: &GameState, _: Player, _: AcquisitionContext) -> CardEligibility {
    let engine = eligible_engine();
    require(
        state
            .board
            .iter()
            .any(|(_, piece)| valid_target(engine, state, piece)),
        "변형할 비왕실 메이저 기물이 필요합니다.",
    )
}

fn eligible_engine() -> &'static Engine {
    static ENGINE: std::sync::OnceLock<Engine> = std::sync::OnceLock::new();
    ENGINE.get_or_init(|| Engine::bundled_latest().expect("bundled latest ruleset"))
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::*;
    use super::*;

    fn with_roulette(pieces: &[(&str, Player, PieceKind)]) -> (Engine, GameState) {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(pieces);
        state = engine
            .grant_card(&state, 1, crate::RANDOM_ROULETTE, Player::White)
            .unwrap()
            .state;
        (engine, state)
    }

    fn offered(engine: &Engine, state: &GameState) -> Vec<Square> {
        engine
            .legal_actions(state)
            .unwrap()
            .into_iter()
            .filter_map(|action| match action {
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::PieceAt(square),
                } => Some(square),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn random_roulette_offers_every_non_royal_major_and_rejects_royal_targets() {
        let (engine, state) = with_roulette(&[
            ("d4", Player::White, PieceKind::Rook),
            ("d1", Player::White, PieceKind::Queen),
            ("c3", Player::White, PieceKind::Knight),
            ("f6", Player::Black, PieceKind::Amazon),
            ("a2", Player::White, PieceKind::Pawn),
        ]);
        let squares = offered(&engine, &state);
        let square = |text: &str| -> Square { text.parse().unwrap() };
        assert!(squares.contains(&square("d4")), "own rook is a target");
        assert!(squares.contains(&square("d1")), "own queen is a target");
        assert!(
            squares.contains(&square("f6")),
            "the contract does not restrict the owner of the target"
        );
        assert!(!squares.contains(&square("c3")), "a minor is not a major");
        assert!(!squares.contains(&square("a2")), "a pawn is not a major");
        assert!(
            !squares.iter().any(|square| {
                state
                    .board
                    .get(*square)
                    .is_some_and(|piece| engine.is_royal_piece(&state, piece))
            }),
            "royal pieces must never be offered"
        );
        for square in ["a1", "h8"] {
            let rejected = engine.apply(
                &state,
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::PieceAt(square.parse().unwrap()),
                },
            );
            assert!(rejected.is_err(), "royal target {square} must be rejected");
        }
    }

    #[test]
    fn random_roulette_never_keeps_the_current_kind_and_always_picks_a_pool_member() {
        let (engine, mut state) = with_roulette(&[("d4", Player::White, PieceKind::Rook)]);
        let original_id = state.board.get("d4".parse().unwrap()).unwrap().id;
        let mut seen = BTreeSet::new();
        for seed in 0..96_u64 {
            *state.rng_mut() = crate::DeterministicRng::new(seed);
            let next = engine
                .apply(
                    &state,
                    Action::UseCard {
                        instance_id: 1,
                        target: single("d4"),
                    },
                )
                .unwrap();
            let (square, _) = state.board.find_by_id(original_id).unwrap();
            assert_eq!(square, "d4".parse().unwrap(), "a small outcome stays put");
            let (_, converted) = next.state.board.find_by_id(original_id).unwrap();
            assert_eq!(converted.owner, Player::White, "owner is preserved");
            let (converted_square, _) = next.state.board.find_by_id(original_id).unwrap();
            assert_eq!(
                converted.kind,
                next.state.board.get(converted_square).unwrap().kind
            );
            assert_ne!(
                converted.kind,
                PieceKind::Rook,
                "the current kind is excluded"
            );
            assert!(
                OUTCOMES.contains(&converted.kind),
                "{:?} must come from the current outcome pool",
                converted.kind
            );
            seen.insert(converted.kind);
            assert!(
                next.events
                    .iter()
                    .any(|event| matches!(event, Event::RandomSample { purpose, .. } if purpose == "random-roulette")),
                "the draw must be recorded"
            );
            assert!(
                next.state
                    .card_runtime
                    .moved_piece_ids
                    .contains(&original_id)
            );
        }
        assert!(seen.len() > 8, "the pool must actually vary: {seen:?}");
    }

    #[test]
    fn random_roulette_creates_a_valid_large_body_and_a_cramped_board_excludes_large_outcomes() {
        let (engine, mut state) = with_roulette(&[("d4", Player::White, PieceKind::Rook)]);
        let rook_id = state.board.get("d4".parse().unwrap()).unwrap().id;
        let mut saw_large = false;
        for seed in 0..256_u64 {
            *state.rng_mut() = crate::DeterministicRng::new(seed);
            let transition = engine
                .apply(
                    &state,
                    Action::UseCard {
                        instance_id: 1,
                        target: single("d4"),
                    },
                )
                .unwrap();
            let next = transition.state;
            let (square, piece) = next.board.find_by_id(rook_id).unwrap();
            if let Some(status) = next.card_runtime.large_pieces.get(&rook_id).copied() {
                saw_large = true;
                assert_eq!(status.anchor, square);
                let cells = large_piece_cells(status.anchor);
                assert_eq!(cells.len(), 4);
                for cell in cells {
                    assert!(
                        next.board
                            .get(cell)
                            .is_none_or(|occupant| occupant.id == rook_id),
                        "a large body must own every footprint cell"
                    );
                }
                let expected_hp = if piece.kind == PieceKind::Colossus {
                    3
                } else {
                    2
                };
                assert_eq!(status.hp, expected_hp);
                assert_eq!(status.max_hp, expected_hp);
                assert!(
                    transition.events.iter().any(|event| matches!(
                        event,
                        Event::RandomSample { purpose, .. }
                            if purpose == "random-roulette-large-anchor"
                    )),
                    "the anchor draw must follow the outcome draw"
                );
            }
        }
        assert!(saw_large, "some seed must produce a large outcome");

        // Fill the board so that no 2x2 body around d4 is free: every large
        // outcome must then be excluded before the draw instead of failing.
        let mut state = state.clone();
        for file in 0..8 {
            for rank in 0..8 {
                let square = Square::new(file, rank).unwrap();
                if state.board.get(square).is_some() || square == "d4".parse().unwrap() {
                    continue;
                }
                // Leave only cells that can never form a clean 2x2 around d4.
                let near = square.file().abs_diff(3) <= 1 && square.rank().abs_diff(3) <= 1;
                if near {
                    state.card_runtime.barricades.insert(square);
                }
            }
        }
        let mut saw_large = false;
        for seed in 0..128_u64 {
            *state.rng_mut() = crate::DeterministicRng::new(seed);
            let next = engine
                .apply(
                    &state,
                    Action::UseCard {
                        instance_id: 1,
                        target: single("d4"),
                    },
                )
                .unwrap()
                .state;
            saw_large |= next.card_runtime.large_pieces.contains_key(&rook_id);
        }
        assert!(
            !saw_large,
            "without a free 2x2 body no large outcome may be drawn"
        );
    }

    #[test]
    fn random_roulette_initializes_transformed_piece_runtime() {
        let (engine, mut state) = with_roulette(&[("d4", Player::White, PieceKind::Rook)]);
        let piece_id = state.board.get("d4".parse().unwrap()).unwrap().id;
        let mut seen = BTreeSet::new();
        for seed in 0..512_u64 {
            *state.rng_mut() = crate::DeterministicRng::new(seed);
            let next = engine
                .apply(
                    &state,
                    Action::UseCard {
                        instance_id: 1,
                        target: single("d4"),
                    },
                )
                .unwrap()
                .state;
            let kind = next.board.find_by_id(piece_id).unwrap().1.kind;
            seen.insert(kind);
            let runtime = &next.card_runtime;
            match kind {
                PieceKind::Wizard => assert_eq!(runtime.wizard_mana.get(&piece_id), Some(&0)),
                PieceKind::Reaper => assert_eq!(runtime.reaper_souls.get(&piece_id), Some(&0)),
                PieceKind::Windmill => assert_eq!(
                    runtime.windmill_modes.get(&piece_id),
                    Some(&WindmillMode::Bishop)
                ),
                PieceKind::Bear | PieceKind::Hedgehog => {
                    assert_eq!(runtime.bear_retaliations.get(&piece_id), Some(&2))
                }
                PieceKind::BabyBear => assert!(
                    runtime
                        .baby_bear_grow_turns
                        .get(&piece_id)
                        .is_some_and(|due| *due > next.shared_turn_count())
                ),
                PieceKind::Trickster => {
                    assert!(runtime.trickster_move_kinds.contains_key(&piece_id))
                }
                PieceKind::Thief => {
                    assert!(runtime.submerged_piece_ids.contains(&piece_id));
                    assert!(runtime.wanted_piece_ids.contains(&piece_id));
                }
                _ => {}
            }
        }
        for expected in [
            PieceKind::Wizard,
            PieceKind::Reaper,
            PieceKind::Windmill,
            PieceKind::Hedgehog,
            PieceKind::Trickster,
        ] {
            assert!(
                seen.contains(&expected),
                "{expected:?} should be reachable in 512 draws"
            );
        }
    }

    #[test]
    fn random_roulette_rejects_invalid_targets_without_touching_state() {
        let (engine, state) = with_roulette(&[
            ("a2", Player::White, PieceKind::Pawn),
            ("c3", Player::White, PieceKind::Knight),
        ]);
        let before = state.clone();
        for square in ["a1", "a2", "c3", "h8"] {
            let rejected = engine.apply(
                &state,
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::PieceAt(square.parse().unwrap()),
                },
            );
            assert!(rejected.is_err(), "{square} is not a non-royal major");
            assert_eq!(state, before, "a rejected use must not mutate state");
        }
        // A missing piece square is rejected too.
        assert!(
            engine
                .apply(
                    &state,
                    Action::UseCard {
                        instance_id: 1,
                        target: ActionTarget::PieceAt("e5".parse().unwrap()),
                    },
                )
                .is_err()
        );
    }

    #[test]
    fn random_roulette_is_deterministic_for_a_seed() {
        let (engine, mut state) = with_roulette(&[("d4", Player::White, PieceKind::Rook)]);
        *state.rng_mut() = crate::DeterministicRng::new(1234);
        let action = Action::UseCard {
            instance_id: 1,
            target: single("d4"),
        };
        let first = engine.apply(&state, action.clone()).unwrap();
        let second = engine.apply(&state, action).unwrap();
        assert_eq!(first, second);
    }
}
