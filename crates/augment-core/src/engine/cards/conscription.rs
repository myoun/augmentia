use super::*;
pub(super) const ID: &str = crate::CONSCRIPTION;
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

/// The central four files the current contract spawns on.
const FILES: [u8; 4] = [2, 3, 4, 5];

/// Web rows are top-down; core ranks count from White's first rank.
fn spawn_rank(actor: Player) -> u8 {
    if actor == Player::White { 1 } else { 6 }
}

/// Cheap placement check usable from `eligibility`, which has no `Engine`.
fn approximately_open(state: &GameState, square: Square) -> bool {
    state.board.get(square).is_none()
        && !state.card_runtime.barricades.contains(&square)
        && !state.card_runtime.black_holes.contains(&square)
        && state.card_runtime.crown_pos != Some(square)
        && !(state.card_runtime.collapsed_files_ranks
            && (matches!(square.file(), 0 | 7) || matches!(square.rank(), 0 | 7)))
}

fn open_squares(engine: &Engine, state: &GameState, actor: Player) -> Vec<Square> {
    FILES
        .into_iter()
        .filter_map(|file| Square::new(file, spawn_rank(actor)))
        .filter(|square| engine.card_landing_open(state, *square))
        .collect()
}

fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _: u32,
    _: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let squares = open_squares(engine, state, actor);
    if squares.is_empty() {
        return Err(EngineError::Invariant(
            "conscription has no open central second-rank square".to_owned(),
        ));
    }
    for square in squares {
        let pawn = Piece::new(actor, PieceKind::Pawn).with_id(state.allocate_piece_id());
        state.board.set(square, Some(pawn));
        // Fresh spawns cannot capture until their owner's next turn.
        engine.register_spawned_piece(state, square, pawn, false);
        events.push(Event::EffectApplied {
            source_card_instance: 0,
            command: format!("conscription-spawn:{}:{}", pawn.id, square),
        });
    }
    Ok(())
}

fn actions(engine: &Engine, state: &GameState, card: &CardInstance, out: &mut Vec<Action>) {
    if !open_squares(engine, state, card.owner).is_empty() {
        out.push(Action::UseCard {
            instance_id: card.instance_id,
            target: ActionTarget::None,
        });
    }
}

fn eligibility(state: &GameState, owner: Player, _: AcquisitionContext) -> CardEligibility {
    require(
        FILES
            .into_iter()
            .filter_map(|file| Square::new(file, spawn_rank(owner)))
            .any(|square| approximately_open(state, square)),
        "두 번째 랭크의 중앙 네 파일에 빈칸이 필요합니다.",
    )
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::{position, single};
    use super::*;

    fn pawn_files(state: &GameState, owner: Player, rank: u8) -> Vec<u8> {
        let mut files = state
            .board
            .iter()
            .filter(|(square, piece)| {
                square.rank() == rank && piece.owner == owner && piece.kind == PieceKind::Pawn
            })
            .map(|(square, _)| square.file())
            .collect::<Vec<_>>();
        files.sort_unstable();
        files
    }

    fn use_card(engine: &Engine, state: &mut GameState, actor: Player) -> Result<(), EngineError> {
        play(
            engine,
            state,
            actor,
            1,
            &ActionTarget::None,
            &mut Vec::new(),
        )
    }

    #[test]
    fn conscription_fills_the_open_central_files_of_the_second_rank() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[]);
        use_card(&engine, &mut state, Player::White).unwrap();
        assert_eq!(pawn_files(&state, Player::White, 1), vec![2, 3, 4, 5]);
        for file in [0, 1, 6, 7] {
            assert!(
                state.board.get(Square::new(file, 1).unwrap()).is_none(),
                "file {file} is outside the central four"
            );
        }
        let spawned = state
            .board
            .iter()
            .filter(|(_, piece)| piece.kind == PieceKind::Pawn)
            .collect::<Vec<_>>();
        assert_eq!(spawned.len(), 4);
        for (_, piece) in spawned {
            assert!(state.card_runtime.piece_origins.contains_key(&piece.id));
            assert!(
                state
                    .card_runtime
                    .fresh_no_capture_until
                    .contains_key(&piece.id),
                "a spawned pawn cannot capture until its owner's next turn"
            );
        }
        // PRESERVE_TURN: the card user keeps the turn.
        assert_eq!(state.turn, Player::White);
    }

    #[test]
    fn conscription_skips_occupied_squares_and_requires_one_open_file() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[
            ("c2", Player::White, PieceKind::Rook),
            ("f2", Player::Black, PieceKind::Bishop),
        ]);
        use_card(&engine, &mut state, Player::White).unwrap();
        assert_eq!(pawn_files(&state, Player::White, 1), vec![3, 4]);
        assert_eq!(
            state.board.get("c2".parse().unwrap()).unwrap().kind,
            PieceKind::Rook
        );
        assert_eq!(
            state.board.get("f2".parse().unwrap()).unwrap().kind,
            PieceKind::Bishop
        );

        let mut blocked = position(&[
            ("c2", Player::White, PieceKind::Rook),
            ("d2", Player::White, PieceKind::Rook),
            ("e2", Player::White, PieceKind::Rook),
            ("f2", Player::White, PieceKind::Rook),
        ]);
        let before = blocked.clone();
        assert!(use_card(&engine, &mut blocked, Player::White).is_err());
        assert_eq!(blocked, before, "a rejected use must not mutate state");
        assert!(
            !eligibility(&blocked, Player::White, AcquisitionContext::Draft).is_eligible(),
            "no open central square means the card must not be offered"
        );

        // A blocked square also suppresses the offered action.
        let mut offered = position(&[
            ("c2", Player::White, PieceKind::Rook),
            ("d2", Player::White, PieceKind::Rook),
            ("e2", Player::White, PieceKind::Rook),
            ("f2", Player::White, PieceKind::Rook),
        ]);
        let mut out = Vec::new();
        let card = CardInstance {
            instance_id: 1,
            definition_id: crate::CONSCRIPTION.to_owned(),
            owner: Player::White,
            half_stars: 0,
            charges: 1,
            available_after_turn: None,
        };
        actions(&engine, &offered, &card, &mut out);
        assert!(out.is_empty());
        offered.board.set("c2".parse().unwrap(), None);
        actions(&engine, &offered, &card, &mut out);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn conscription_uses_the_acting_players_second_rank() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[]);
        state.turn = Player::Black;
        use_card(&engine, &mut state, Player::Black).unwrap();
        assert_eq!(pawn_files(&state, Player::Black, 6), vec![2, 3, 4, 5]);
        assert!(pawn_files(&state, Player::White, 1).is_empty());
        // Nothing leaked onto White's rank.
        assert!(
            state.board.iter().all(|(square, _)| square.rank() != 1),
            "black conscription must not touch White's second rank"
        );
    }

    #[test]
    fn conscription_rejects_placements_marked_by_other_card_state() {
        let engine = Engine::bundled_latest().unwrap();
        let mut state = position(&[]);
        state.card_runtime.barricades.insert("d2".parse().unwrap());
        state.card_runtime.black_holes.insert("e2".parse().unwrap());
        state.card_runtime.crown_pos = Some("c2".parse().unwrap());
        use_card(&engine, &mut state, Player::White).unwrap();
        assert_eq!(pawn_files(&state, Player::White, 1), vec![5]);
        // `single` is imported to keep the shared helper module exercised.
        let _ = single("d2");
    }
}
