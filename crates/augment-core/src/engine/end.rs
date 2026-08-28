use super::*;
use crate::*;

impl Engine {
    pub(super) fn apply_end_active_card(
        &self,
        state: &mut GameState,
        actor: Player,
        definition_id: &str,
        target: &ActionTarget,
        events: &mut Vec<Event>,
    ) -> Result<bool, EngineError> {
        let opponent = actor.opponent();
        match definition_id {
            HYPOCRISY => {
                let ActionTarget::Squares(squares) = target else {
                    return Err(EngineError::Invariant(
                        "hypocrisy requires four squares".to_owned(),
                    ));
                };
                for &square in squares {
                    let pawn =
                        Piece::new(opponent, PieceKind::Pawn).with_id(state.allocate_piece_id());
                    state.board.set(square, Some(pawn));
                    self.register_spawned_piece(state, square, pawn, true);
                }
            }
            GOMOKU => {
                state.card_runtime.gomoku[actor.index()] = true;
                self.resolve_gomoku_victory(state);
            }
            REPLAY => {
                let replay = state.card_runtime.move_replays[actor.index()]
                    .clone()
                    .ok_or_else(|| {
                        EngineError::Invariant("replay has no recorded move".to_owned())
                    })?;
                if !replay_move_matches_state(state, &replay) {
                    return Err(EngineError::Invariant(
                        "replay cells no longer match the recorded move".to_owned(),
                    ));
                }
                for cell in &replay.cells {
                    state.board.set(cell.square, cell.before);
                }
                if state.en_passant == replay.en_passant_after {
                    state.en_passant = replay.en_passant_before;
                }
                let mut current_runtime = state.card_runtime.clone();
                current_runtime.move_replays = [None, None];
                if current_runtime == *replay.runtime_after {
                    let other_replay = state.card_runtime.move_replays[opponent.index()].clone();
                    state.card_runtime = (*replay.runtime_before).clone();
                    // Playing Replay only consumes the current player's stored move. The other
                    // player's independent capture remains available, as in normalizeColorValues.
                    state.card_runtime.move_replays[opponent.index()] = other_replay;
                } else {
                    restore_replay_piece_runtime(&mut state.card_runtime, &replay);
                    restore_replay_color_runtime(&mut state.card_runtime, &replay, actor);
                }
                state.card_runtime.move_replays[actor.index()] = None;
            }
            BABY_BEAR => {
                let (square, queen) = piece_target(state, target)?;
                self.remove_piece_for_card(state, square, queen, actor, BABY_BEAR, events);
                let baby_bear =
                    Piece::new(actor, PieceKind::BabyBear).with_id(state.allocate_piece_id());
                state.board.set(square, Some(baby_bear));
                state
                    .card_runtime
                    .baby_bear_grow_turns
                    .insert(baby_bear.id, state.shared_turn_count().saturating_add(7));
                events.push(Event::EffectApplied {
                    source_card_instance: 0,
                    command: format!("baby-bear-spawn:{}:{}", baby_bear.id, square),
                });
            }
            BRIBE => {
                let (_, knight) = piece_target(state, target)?;
                if let Some((square, mut piece)) = state.board.find_by_id(knight.id) {
                    piece.kind = PieceKind::Amazon;
                    state.board.set(square, Some(piece));
                    state.card_runtime.bribe_reverts.push(BribeRevert {
                        piece_id: piece.id,
                        due_turn: state.turns_taken[actor.index()].saturating_add(3),
                    });
                    events.push(Event::EffectApplied {
                        source_card_instance: 0,
                        command: format!("bribe:{}:{}", piece.id, square),
                    });
                }
            }
            CHARGE => {
                let (_, pawn) = piece_target(state, target)?;
                state.card_runtime.charge_pawns.insert(pawn.id);
            }
            CORONATION => state.card_runtime.coronation[actor.index()] = true,
            FRENZY => {
                let (_, pawn) = piece_target(state, target)?;
                state.card_runtime.frenzy_pawns.insert(pawn.id);
            }
            HALLUCINATION => {
                state.card_runtime.hallucination_until_ply[actor.index()] =
                    Some(state.ply.saturating_add(5));
            }
            LAST_RESISTANCE => {
                for (_, piece) in state.board.iter() {
                    if piece.owner == actor && self.is_royal_piece(state, piece) {
                        state.card_runtime.protected_piece_ids.insert(piece.id);
                        state.card_runtime.last_resistance.insert(
                            piece.id,
                            TimedPieceStatus {
                                by: piece.owner,
                                remaining_turns: 3,
                            },
                        );
                    }
                }
            }
            MOVING => state.card_runtime.moving[actor.index()] = true,
            NECROMANCY => {
                let (pawn_square, pawn) = piece_target(state, target)?;
                let dead_candidates = [
                    PieceKind::Queen,
                    PieceKind::Rook,
                    PieceKind::Bishop,
                    PieceKind::Knight,
                ];
                let (roll, index) = state.rng_mut().web_choice_index(dead_candidates.len());
                events.push(Event::RandomSample {
                    purpose: "necromancy".to_owned(),
                    roll_53: roll,
                });
                let chosen_kind = dead_candidates[index.unwrap_or(0)];
                if let Some((square, mut piece)) = state.board.find_by_id(pawn.id) {
                    piece.kind = chosen_kind;
                    state.board.set(square, Some(piece));
                    state
                        .card_runtime
                        .necromancy_reverts
                        .push(NecromancyRevert {
                            piece_id: piece.id,
                            due_ply: state.ply.saturating_add(4),
                        });
                    events.push(Event::EffectApplied {
                        source_card_instance: 0,
                        command: format!("necromancy:{}:{:?}", pawn_square, chosen_kind),
                    });
                }
            }
            REVERSE_PAWNS => {
                state.card_runtime.reverse_pawns_charges[opponent.index()] = 3;
            }
            STAKE => {
                let (_, piece) = piece_target(state, target)?;
                state.card_runtime.stake_statuses.insert(
                    piece.id,
                    TimedPieceStatus {
                        by: piece.owner,
                        remaining_turns: 4,
                    },
                );
            }
            UNDERPROMOTION => state.card_runtime.underpromotion[actor.index()] = true,
            VIP => {
                let (_, pawn) = piece_target(state, target)?;
                state.card_runtime.vip_pending.push(PendingVip {
                    piece_id: pawn.id,
                    owner: pawn.owner,
                    due_ply: state.ply.saturating_add(3),
                });
            }
            BARRICADE => {
                let center = square_target(target)?;
                let mut barricade_squares = vec![center];
                for offset in [-1, 1] {
                    if let Some(sq) = center.offset(offset, 0)
                        && state.board.get(sq).is_none()
                    {
                        barricade_squares.push(sq);
                    }
                }
                for sq in barricade_squares {
                    state.card_runtime.barricades.insert(sq);
                }
            }
            CANCELING => {
                let home_square = if opponent == Player::White {
                    "e1".parse().unwrap()
                } else {
                    "e8".parse().unwrap()
                };
                let enemy_king = state
                    .board
                    .iter()
                    .find(|(_, p)| p.owner == opponent && self.is_royal_piece(state, *p));
                if let Some((current_square, king)) = enemy_king {
                    state.board.set(current_square, None);
                    if let Some(occupant) = state.board.get(home_square) {
                        self.remove_piece_for_card(
                            state,
                            home_square,
                            occupant,
                            actor,
                            CANCELING,
                            events,
                        );
                    }
                    state.board.set(home_square, Some(king));
                    events.push(Event::EffectApplied {
                        source_card_instance: 0,
                        command: format!(
                            "canceling:{}:{}->{}",
                            king.id, current_square, home_square
                        ),
                    });
                }
            }
            CHAIN => {
                let [first_square, second_square] = two_square_target(target)?;
                if let (Some(first), Some(second)) = (
                    state.board.get(first_square),
                    state.board.get(second_square),
                ) {
                    state.card_runtime.chains.push([first.id, second.id]);
                }
            }
            CONSCRIPTION => {
                let rank = if actor == Player::White { 1 } else { 6 };
                for file in 2..=5 {
                    if let Some(square) = Square::new(file, rank)
                        && state.board.get(square).is_none()
                    {
                        let pawn =
                            Piece::new(actor, PieceKind::Pawn).with_id(state.allocate_piece_id());
                        state.board.set(square, Some(pawn));
                        self.mark_fresh_no_capture(state, pawn);
                    }
                }
            }
            COLLAPSE => {
                state.card_runtime.collapsed_files_ranks = true;
                let outer = state
                    .board
                    .iter()
                    .filter(|(sq, _)| matches!(sq.file(), 0 | 7) || matches!(sq.rank(), 0 | 7))
                    .collect::<Vec<_>>();
                for (sq, piece) in outer {
                    self.remove_piece_for_card(state, sq, piece, actor, COLLAPSE, events);
                }
            }
            DESPERADO => {
                let (_, piece) = piece_target(state, target)?;
                state.card_runtime.desperado_pieces.insert(piece.id);
            }
            EXILE => {
                let (square, piece) = piece_target(state, target)?;
                if let Some(origin) = state.card_runtime.piece_origins.get(&piece.id).copied() {
                    state.board.set(square, None);
                    if let Some(occupant) = state.board.get(origin) {
                        self.remove_piece_for_card(state, origin, occupant, actor, EXILE, events);
                    }
                    state.board.set(origin, Some(piece));
                }
            }
            HOMECOMING => {
                let (square, piece) = piece_target(state, target)?;
                if let Some(origin) = state.card_runtime.piece_origins.get(&piece.id).copied()
                    && state.board.get(origin).is_none()
                {
                    state.board.set(square, None);
                    state.board.set(origin, Some(piece));
                }
            }
            LOBSTER => {
                let square = square_target(target)?;
                let lobster =
                    Piece::new(actor, PieceKind::Lobster).with_id(state.allocate_piece_id());
                state.board.set(square, Some(lobster));
                self.mark_fresh_no_capture(state, lobster);
            }
            PALACE => {
                if let Some((king_sq, _)) = state
                    .board
                    .iter()
                    .find(|(_, p)| p.owner == opponent && self.is_royal_piece(state, *p))
                {
                    let min_file = king_sq.file().saturating_sub(1);
                    let max_file = u8::min(7, king_sq.file().saturating_add(1));
                    let min_rank = king_sq.rank().saturating_sub(1);
                    let max_rank = u8::min(7, king_sq.rank().saturating_add(1));
                    let min_sq = Square::new(min_file, min_rank).unwrap();
                    let max_sq = Square::new(max_file, max_rank).unwrap();
                    state
                        .card_runtime
                        .palaces
                        .insert(opponent, [min_sq, max_sq]);
                }
            }
            PORTAL_GUN => {
                let [sq1, sq2] = two_square_target(target)?;
                state.card_runtime.portals = Some([sq1, sq2]);
            }
            UNDERGROUND_BUNKER => {
                if let Some((_, king)) = state
                    .board
                    .iter()
                    .find(|(_, p)| p.owner == actor && self.is_royal_piece(state, *p))
                {
                    state.card_runtime.bunkers.insert(king.id, 5);
                }
            }
            ZUGZWANG => {
                state.card_runtime.zugzwang[opponent.index()] = true;
            }
            BLUE_JEANS => {
                let major_pieces = state
                    .board
                    .iter()
                    .filter(|(_, p)| matches!(p.kind, PieceKind::Queen | PieceKind::Rook))
                    .collect::<Vec<_>>();
                for (sq, piece) in major_pieces {
                    self.remove_piece_for_card(state, sq, piece, actor, BLUE_JEANS, events);
                }
            }
            DEATH_SQUAD => {
                let file = match target {
                    ActionTarget::Square(sq) => sq.file(),
                    _ => 3,
                };
                let pawns = state
                    .board
                    .iter()
                    .filter(|(sq, p)| {
                        p.owner == actor && p.kind == PieceKind::Pawn && sq.file() == file
                    })
                    .collect::<Vec<_>>();
                for (sq, mut pawn) in pawns {
                    pawn.kind = PieceKind::Fanatic;
                    state.board.set(sq, Some(pawn));
                }
            }
            FINAL_WEAPON => state.card_runtime.final_weapon[actor.index()] = true,
            FLEETING_DREAM => {
                let promoted = state
                    .board
                    .iter()
                    .filter(|(_, p)| {
                        p.owner == opponent
                            && matches!(
                                p.kind,
                                PieceKind::Queen
                                    | PieceKind::Rook
                                    | PieceKind::Bishop
                                    | PieceKind::Knight
                                    | PieceKind::Amazon
                            )
                            && state
                                .card_runtime
                                .piece_origins
                                .get(&p.id)
                                .is_some_and(|orig| matches!(orig.rank(), 1 | 6))
                    })
                    .collect::<Vec<_>>();
                for (sq, mut piece) in promoted {
                    piece.kind = PieceKind::Pawn;
                    state.board.set(sq, Some(piece));
                }
            }
            ICBM => {
                let (_, queen) = piece_target(state, target)?;
                state.card_runtime.icbm_pending.push(PendingIcbm {
                    queen_id: queen.id,
                    owner: actor,
                    due_turn: state.turns_taken[actor.index()].saturating_add(1),
                });
            }
            OTHELLO => {
                let (enemy_sq, enemy) = piece_target(state, target)?;
                let mut converted = enemy;
                converted.owner = actor;
                state.board.set(enemy_sq, Some(converted));
                events.push(Event::EffectApplied {
                    source_card_instance: 0,
                    command: format!("othello:{}:{}", converted.id, enemy_sq),
                });
            }
            PROPHECY => {
                state.card_runtime.prophecy[actor.index()] = Some(state.ply.saturating_add(3));
            }
            RACING_KING => state.card_runtime.racing_king[actor.index()] = true,
            BINA_MATE => state.card_runtime.bina_mate[actor.index()] = true,
            SPY => {
                let (_, pawn) = piece_target(state, target)?;
                state.card_runtime.spies.insert(pawn.id, actor);
            }
            TRAITOR => {
                let enemy_pieces = state
                    .board
                    .iter()
                    .filter(|(_, p)| p.owner == opponent && !self.is_royal_piece(state, *p))
                    .collect::<Vec<_>>();
                let weakest = enemy_pieces.iter().min_by_key(|(_, p)| {
                    (
                        if p.kind == PieceKind::Pawn { 0 } else { 1 },
                        p.kind.stable_code(),
                    )
                });
                if let Some(&(sq, mut piece)) = weakest {
                    piece.owner = actor;
                    state.board.set(sq, Some(piece));
                    events.push(Event::EffectApplied {
                        source_card_instance: 0,
                        command: format!("traitor:{}:{}", piece.id, sq),
                    });
                }
            }
            ULTIMATUM => {
                state.card_runtime.ultimatum = Some(state.ply.saturating_add(4));
            }
            VANISH => {
                state.card_runtime.vanish[actor.index()] = true;
            }
            CLONE => {
                state.card_runtime.coronation[actor.index()] = true;
            }
            JOKER => {
                for card in &mut state.cards {
                    if card.owner == actor && card.charges == 0 {
                        card.charges = 1;
                        break;
                    }
                }
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    pub(super) fn generate_end_card_actions(
        &self,
        state: &GameState,
        card: &CardInstance,
        actor: Player,
        actions: &mut Vec<Action>,
    ) {
        let opponent = actor.opponent();
        let push_none = |actions: &mut Vec<Action>| {
            actions.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::None,
            })
        };
        match card.definition_id.as_str() {
            HYPOCRISY => {
                let squares = (0_u8..64)
                    .filter_map(Square::from_index)
                    .filter(|square| {
                        state.board.get(*square).is_none()
                            && !state.card_runtime.black_holes.contains(square)
                    })
                    .take(4)
                    .collect::<Vec<_>>();
                if squares.len() == 4 {
                    actions.push(Action::UseCard {
                        instance_id: card.instance_id,
                        target: ActionTarget::Squares(squares),
                    });
                }
            }
            GOMOKU => push_none(actions),
            REPLAY => {
                if state.card_runtime.move_replays[actor.index()]
                    .as_ref()
                    .is_some_and(|replay| replay_move_matches_state(state, replay))
                {
                    push_none(actions);
                }
            }
            BABY_BEAR => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == actor && piece.kind == PieceKind::Queen {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            BRIBE => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == actor && piece.kind == PieceKind::Knight {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            CHARGE | FRENZY => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == actor && piece.kind == PieceKind::Pawn {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            CORONATION | HALLUCINATION | MOVING | REVERSE_PAWNS | UNDERPROMOTION | CONSCRIPTION
            | COLLAPSE | PALACE | UNDERGROUND_BUNKER | ZUGZWANG | BLUE_JEANS | FINAL_WEAPON
            | FLEETING_DREAM | PROPHECY | RACING_KING | BINA_MATE | TRAITOR | ULTIMATUM
            | VANISH | CLONE | JOKER => {
                push_none(actions);
            }
            LAST_RESISTANCE => {
                if state
                    .board
                    .iter()
                    .any(|(_, p)| p.owner == actor && self.is_royal_piece(state, p))
                {
                    push_none(actions);
                }
            }
            NECROMANCY => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == actor && piece.kind == PieceKind::Pawn {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            STAKE => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == actor {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            VIP => {
                for (square, piece) in state.board.iter() {
                    if piece.kind == PieceKind::Pawn {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            BARRICADE | LOBSTER => {
                for index in 0..64 {
                    let square = Square::from_index(index).unwrap();
                    if state.board.get(square).is_none()
                        && !state.card_runtime.barricades.contains(&square)
                    {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::Square(square),
                        });
                    }
                }
            }
            CANCELING => {
                push_none(actions);
            }
            CHAIN => {
                let enemy_pieces = state
                    .board
                    .iter()
                    .filter(|(_, p)| p.owner == opponent)
                    .collect::<Vec<_>>();
                for i in 0..enemy_pieces.len() {
                    for j in (i + 1)..enemy_pieces.len() {
                        let (sq1, _) = enemy_pieces[i];
                        let (sq2, _) = enemy_pieces[j];
                        if sq1.file().abs_diff(sq2.file()) <= 2
                            && sq1.rank().abs_diff(sq2.rank()) <= 2
                        {
                            actions.push(Action::UseCard {
                                instance_id: card.instance_id,
                                target: ActionTarget::Squares(vec![sq1, sq2]),
                            });
                        }
                    }
                }
            }
            DESPERADO | HOMECOMING => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == actor && !self.is_royal_piece(state, piece) {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            EXILE => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == opponent && !self.is_royal_piece(state, piece) {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            PORTAL_GUN => {
                let empties = (0..64)
                    .filter_map(Square::from_index)
                    .filter(|sq| state.board.get(*sq).is_none())
                    .collect::<Vec<_>>();
                for i in 0..empties.len().min(8) {
                    for j in (i + 1)..empties.len().min(8) {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::Squares(vec![empties[i], empties[j]]),
                        });
                    }
                }
            }
            DEATH_SQUAD => {
                for file in 0..8 {
                    if state.board.iter().any(|(sq, p)| {
                        p.owner == actor && p.kind == PieceKind::Pawn && sq.file() == file
                    }) {
                        if let Some(sq) =
                            Square::new(file, if actor == Player::White { 1 } else { 6 })
                        {
                            actions.push(Action::UseCard {
                                instance_id: card.instance_id,
                                target: ActionTarget::Square(sq),
                            });
                        }
                    }
                }
            }
            ICBM => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == actor && piece.kind == PieceKind::Queen {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            OTHELLO => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == opponent {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            SPY => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == opponent && piece.kind == PieceKind::Pawn {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            _ => {}
        }
    }

    pub(super) fn resolve_end_after_move(
        &self,
        state: &mut GameState,
        piece: Piece,
        movement: MoveAction,
        captured: Option<(Square, Piece)>,
        events: &mut Vec<Event>,
    ) {
        let actor = piece.owner;
        if state.card_runtime.moving[actor.index()] {
            let (last_piece, count) = state.card_runtime.moving_consecutive[actor.index()];
            if last_piece == Some(piece.id) {
                let new_count = count.saturating_add(1);
                state.card_runtime.moving_consecutive[actor.index()] = (Some(piece.id), new_count);
                if new_count >= 5 {
                    state.card_runtime.evasion_piece_ids.insert(piece.id);
                }
            } else {
                state.card_runtime.moving_consecutive[actor.index()] = (Some(piece.id), 1);
            }
        }

        if captured.is_some() && state.card_runtime.frenzy_pawns.contains(&piece.id) {
            state.phase = GamePhase::ForcedMove {
                piece_id: piece.id,
                source: ForcedMoveSource::Effect(FRENZY.to_owned()),
                continuation: ForcedMoveContinuation::FinishTurn,
            };
        }

        if state.card_runtime.charge_pawns.remove(&piece.id) && movement.promotion.is_none() {
            if let Some((sq, p)) = state.board.find_by_id(piece.id) {
                self.remove_piece_for_card(state, sq, p, actor, CHARGE, events);
            }
        }

        if state.card_runtime.desperado_pieces.remove(&piece.id) {
            if let Some((sq, p)) = state.board.find_by_id(piece.id) {
                self.remove_piece_for_card(state, sq, p, actor, DESPERADO, events);
            }
        }

        if state.card_runtime.coronation[actor.index()]
            && movement.promotion == Some(PieceKind::Queen)
        {
            state.card_runtime.protected_piece_ids.insert(piece.id);
            state
                .card_runtime
                .coronation_protection
                .insert(piece.id, state.turns_taken[actor.index()]);
        }

        if state.card_runtime.underpromotion[actor.index()]
            && matches!(
                movement.promotion,
                Some(PieceKind::Knight | PieceKind::Bishop | PieceKind::Rook)
            )
        {
            state.phase = GamePhase::ForcedMove {
                piece_id: piece.id,
                source: ForcedMoveSource::Effect(UNDERPROMOTION.to_owned()),
                continuation: ForcedMoveContinuation::FinishTurn,
            };
        }

        if let Some(spy_owner) = state.card_runtime.spies.remove(&piece.id)
            && movement.promotion.is_some()
        {
            if let Some((sq, mut promoted)) = state.board.find_by_id(piece.id) {
                promoted.owner = spy_owner;
                state.board.set(sq, Some(promoted));
            }
        }

        if state.card_runtime.racing_king[actor.index()]
            && self.is_royal_piece(state, piece)
            && movement.to.rank() == if actor == Player::White { 7 } else { 0 }
        {
            state.status = GameStatus::Won {
                winner: actor,
                reason: WinReason::CardEffect("racing-king".to_owned()),
            };
        }

        if state.card_runtime.vanish[actor.index()] {
            let candidates = state
                .board
                .iter()
                .filter(|(_, p)| p.owner == actor && !self.is_royal_piece(state, *p))
                .collect::<Vec<_>>();
            if !candidates.is_empty() {
                let (roll, idx) = state.rng_mut().web_choice_index(candidates.len());
                events.push(Event::RandomSample {
                    purpose: "vanish".to_owned(),
                    roll_53: roll,
                });
                let (sq, p) = candidates[idx.unwrap_or(0)];
                self.remove_piece_for_card(state, sq, p, actor.opponent(), VANISH, events);
            }
        }

        for player in [Player::White, Player::Black] {
            if let Some(due_ply) = state.card_runtime.prophecy[player.index()] {
                if captured.is_some() {
                    state.card_runtime.prophecy[player.index()] = None;
                } else if state.ply >= due_ply {
                    state.status = GameStatus::Won {
                        winner: player,
                        reason: WinReason::CardEffect("prophecy".to_owned()),
                    };
                }
            }
        }

        if state.card_runtime.bina_mate[actor.index()] {
            if let Some((king_sq, _)) = state
                .board
                .iter()
                .find(|(_, p)| p.owner == actor.opponent() && self.is_royal_piece(state, *p))
            {
                let mut attackers = 0;
                for (_, friendly) in state.board.iter().filter(|(_, p)| p.owner == actor) {
                    if self
                        .generate_piece_move_actions(state, friendly.id)
                        .iter()
                        .any(|a| matches!(a, Action::Move(m) if m.to == king_sq))
                    {
                        attackers += 1;
                    }
                }
                if attackers >= 2 {
                    state.status = GameStatus::Won {
                        winner: actor,
                        reason: WinReason::CardEffect("bina-mate".to_owned()),
                    };
                }
            }
        }
    }

    pub(super) fn tick_end_statuses_after_turn(
        &self,
        state: &mut GameState,
        actor: Player,
        events: &mut Vec<Event>,
    ) {
        state.card_runtime.zugzwang[actor.index()] = false;

        let mut remaining_bribes = Vec::new();
        for bribe in std::mem::take(&mut state.card_runtime.bribe_reverts) {
            if state.turns_taken[actor.index()] >= bribe.due_turn {
                if let Some((sq, mut piece)) = state.board.find_by_id(bribe.piece_id) {
                    piece.kind = PieceKind::Knight;
                    state.board.set(sq, Some(piece));
                    events.push(Event::EffectApplied {
                        source_card_instance: 0,
                        command: format!("bribe-revert:{}:{}", piece.id, sq),
                    });
                }
            } else {
                remaining_bribes.push(bribe);
            }
        }
        state.card_runtime.bribe_reverts = remaining_bribes;

        let mut remaining_necro = Vec::new();
        for necro in std::mem::take(&mut state.card_runtime.necromancy_reverts) {
            if state.ply >= necro.due_ply {
                if let Some((sq, mut piece)) = state.board.find_by_id(necro.piece_id) {
                    piece.kind = PieceKind::Pawn;
                    state.board.set(sq, Some(piece));
                }
            } else {
                remaining_necro.push(necro);
            }
        }
        state.card_runtime.necromancy_reverts = remaining_necro;

        let mut remaining_vip = Vec::new();
        for vip in std::mem::take(&mut state.card_runtime.vip_pending) {
            if state.ply >= vip.due_ply {
                if let Some((sq, mut piece)) = state.board.find_by_id(vip.piece_id) {
                    piece.kind = PieceKind::Vip;
                    state.board.set(sq, Some(piece));
                }
            } else {
                remaining_vip.push(vip);
            }
        }
        state.card_runtime.vip_pending = remaining_vip;

        let due_stakes = state
            .card_runtime
            .stake_statuses
            .iter_mut()
            .filter_map(|(piece_id, status)| {
                if state
                    .board
                    .find_by_id(*piece_id)
                    .is_some_and(|(_, piece)| piece.owner == actor)
                {
                    status.remaining_turns = status.remaining_turns.saturating_sub(1);
                    (status.remaining_turns == 0).then_some(*piece_id)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        for piece_id in due_stakes {
            state.card_runtime.stake_statuses.remove(&piece_id);
            state.card_runtime.shielded_piece_ids.insert(piece_id);
        }

        let due_icbm = state
            .card_runtime
            .icbm_pending
            .iter()
            .filter(|icbm| state.turns_taken[icbm.owner.index()] >= icbm.due_turn)
            .cloned()
            .collect::<Vec<_>>();
        for icbm in due_icbm {
            let enemy_queen_sq = state
                .board
                .iter()
                .find(|(_, p)| p.owner == icbm.owner.opponent() && p.kind == PieceKind::Queen)
                .map(|(sq, _)| sq);
            if let Some(queen_sq) = enemy_queen_sq {
                let victims = state
                    .board
                    .iter()
                    .filter(|(sq, _)| {
                        sq.file().abs_diff(queen_sq.file()) <= 1
                            && sq.rank().abs_diff(queen_sq.rank()) <= 1
                    })
                    .collect::<Vec<_>>();
                for (sq, victim) in victims {
                    self.remove_piece_for_card(state, sq, victim, icbm.owner, ICBM, events);
                }
            }
        }
        state
            .card_runtime
            .icbm_pending
            .retain(|icbm| state.turns_taken[icbm.owner.index()] < icbm.due_turn);

        if let Some(due_ply) = state.card_runtime.ultimatum {
            if state.ply >= due_ply {
                let unmoved = state
                    .board
                    .iter()
                    .filter(|(_, p)| {
                        !self.is_royal_piece(state, *p)
                            && !state.card_runtime.moved_piece_ids.contains(&p.id)
                    })
                    .collect::<Vec<_>>();
                for (sq, victim) in unmoved {
                    self.remove_piece_for_card(state, sq, victim, actor, ULTIMATUM, events);
                }
                state.card_runtime.ultimatum = None;
            }
        }
    }
}

fn replay_move_matches_state(state: &GameState, replay: &MoveReplayState) -> bool {
    replay
        .cells
        .iter()
        .all(|cell| state.board.get(cell.square) == cell.after)
        && replay_piece_runtime_matches(&state.card_runtime, replay)
}

fn replay_piece_ids(replay: &MoveReplayState) -> Vec<u32> {
    let mut ids = replay
        .cells
        .iter()
        .flat_map(|cell| [cell.before, cell.after])
        .flatten()
        .map(|piece| piece.id)
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    ids
}

fn replay_piece_runtime_matches(current: &CardRuntimeState, replay: &MoveReplayState) -> bool {
    let after = &*replay.runtime_after;
    macro_rules! sets_match {
        ($id:expr; $($field:ident),+ $(,)?) => {
            $(if current.$field.contains(&$id) != after.$field.contains(&$id) { return false; })+
        };
    }
    macro_rules! maps_match {
        ($id:expr; $($field:ident),+ $(,)?) => {
            $(if current.$field.get(&$id) != after.$field.get(&$id) { return false; })+
        };
    }
    for id in replay_piece_ids(replay) {
        sets_match!(id;
            moved_piece_ids, parry_piece_ids, shielded_piece_ids, protected_piece_ids,
            idol_rest_piece_ids, loyalist_piece_ids, royal_piece_ids,
            basic_training_piece_ids, promotion_rush_piece_ids, evasion_piece_ids,
            ghost_piece_ids, inertia_piece_ids, chameleon_piece_ids, chimera_piece_ids,
            poisoned_piece_ids, queens_gambit_protection, potion_basic_training_piece_ids,
            potion_manner_piece_ids, manner_captured_last_piece_ids,
            potion_saturation_piece_ids, explosive_piece_ids, trojan_piece_ids,
            stealth_piece_ids, submerged_piece_ids, charge_pawns, desperado_pieces,
            field_promoted_piece_ids
        );
        maps_match!(id;
            fresh_no_capture_until, disarmed, frozen, empty_lunchboxes,
            herald_jump_locked_until, windmill_modes, log_statuses, wizard_mana,
            reaper_souls, idol_used_turn, piece_origins, large_pieces, calling_cards,
            holdout_promotions, merchant_gold, ice_sheet, severed, imperial_moves,
            sacrifice_protection, last_resistance, coronation_protection,
            poison_stun_turns, saturation_capture_counts, quantum_positions,
            witch_trials, baby_bear_grow_turns, bear_retaliations,
            trickster_move_kinds, stake_statuses, bunkers, spies
        );
    }
    true
}

fn restore_replay_piece_runtime(current: &mut CardRuntimeState, replay: &MoveReplayState) {
    let before = &*replay.runtime_before;
    macro_rules! restore_sets {
        ($id:expr; $($field:ident),+ $(,)?) => {
            $(if before.$field.contains(&$id) {
                current.$field.insert($id);
            } else {
                current.$field.remove(&$id);
            })+
        };
    }
    macro_rules! restore_maps {
        ($id:expr; $($field:ident),+ $(,)?) => {
            $(if let Some(value) = before.$field.get(&$id).cloned() {
                current.$field.insert($id, value);
            } else {
                current.$field.remove(&$id);
            })+
        };
    }
    for id in replay_piece_ids(replay) {
        restore_sets!(id;
            moved_piece_ids, parry_piece_ids, shielded_piece_ids, protected_piece_ids,
            idol_rest_piece_ids, loyalist_piece_ids, royal_piece_ids,
            basic_training_piece_ids, promotion_rush_piece_ids, evasion_piece_ids,
            ghost_piece_ids, inertia_piece_ids, chameleon_piece_ids, chimera_piece_ids,
            poisoned_piece_ids, queens_gambit_protection, potion_basic_training_piece_ids,
            potion_manner_piece_ids, manner_captured_last_piece_ids,
            potion_saturation_piece_ids, explosive_piece_ids, trojan_piece_ids,
            stealth_piece_ids, submerged_piece_ids, charge_pawns, desperado_pieces,
            field_promoted_piece_ids
        );
        restore_maps!(id;
            fresh_no_capture_until, disarmed, frozen, empty_lunchboxes,
            herald_jump_locked_until, windmill_modes, log_statuses, wizard_mana,
            reaper_souls, idol_used_turn, piece_origins, large_pieces, calling_cards,
            holdout_promotions, merchant_gold, ice_sheet, severed, imperial_moves,
            sacrifice_protection, last_resistance, coronation_protection,
            poison_stun_turns, saturation_capture_counts, quantum_positions,
            witch_trials, baby_bear_grow_turns, bear_retaliations,
            trickster_move_kinds, stake_statuses, bunkers, spies
        );
    }

    let ids = replay_piece_ids(replay);
    let touches = |id: u32| ids.binary_search(&id).is_ok();
    current
        .siren_exposure
        .retain(|(siren, target), _| !touches(*siren) && !touches(*target));
    current.siren_exposure.extend(
        before
            .siren_exposure
            .iter()
            .filter(|((siren, target), _)| touches(*siren) || touches(*target))
            .map(|(key, value)| (*key, *value)),
    );
    current
        .feudal_contracts
        .retain(|entry| !touches(entry.pawn_id) && !touches(entry.guardian_id));
    current.feudal_contracts.extend(
        before
            .feudal_contracts
            .iter()
            .filter(|entry| touches(entry.pawn_id) || touches(entry.guardian_id))
            .copied(),
    );
    current
        .twin_bonds
        .retain(|entry| !touches(entry.first_id) && !touches(entry.second_id));
    current.twin_bonds.extend(
        before
            .twin_bonds
            .iter()
            .filter(|entry| touches(entry.first_id) || touches(entry.second_id))
            .copied(),
    );
}

fn restore_replay_color_runtime(
    current: &mut CardRuntimeState,
    replay: &MoveReplayState,
    actor: Player,
) {
    let before = &*replay.runtime_before;
    let after = &*replay.runtime_after;
    let index = actor.index();
    macro_rules! restore_color_fields {
        ($($field:ident),+ $(,)?) => {
            $(if current.$field[index] == after.$field[index] {
                current.$field[index] = before.$field[index];
            })+
        };
    }
    restore_color_fields!(
        quantum_pending,
        switcheroo,
        moving,
        moving_consecutive,
        exhaustion_last_piece,
        exhaustion_count,
        zugzwang,
    );
    if current.crown_pos == after.crown_pos && current.crown_holder == after.crown_holder {
        current.crown_pos = before.crown_pos;
        current.crown_holder = before.crown_holder;
    }
    if current.ultimatum == after.ultimatum {
        current.ultimatum = before.ultimatum;
    }
    if current.undead_resurrections == after.undead_resurrections {
        current.undead_resurrections = before.undead_resurrections.clone();
    }
}
