//! Historical endgame card dispatch.
use crate::engine::turn_effects::*;
use crate::engine::*;
use crate::*;
impl Engine {
    pub(in crate::engine) fn apply_end_active_card(
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
                    self.clear_queens_gambit_protection(state, piece.id);
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
                        if state.ruleset_id == crate::LATEST_RULESET_ID {
                            state
                                .card_runtime
                                .remember_permanent_before_timed_protection(piece.id);
                        }
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
                    self.clear_queens_gambit_protection(state, piece.id);
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
                state.card_runtime.pending_lobsters.push(PendingLobster {
                    owner: actor,
                    square,
                    due_ply: state.ply.saturating_add(2),
                });
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
                state.card_runtime.pending_portals.push(PendingPortal {
                    owner: actor,
                    squares: [sq1, sq2],
                    due_turn: state.turns_taken[actor.index()].saturating_add(1),
                });
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
                    self.clear_queens_gambit_protection(state, pawn.id);
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
                    self.clear_queens_gambit_protection(state, piece.id);
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
            crate::REVERSAL => state.card_runtime.reversal[actor.index()] = true,
            crate::EXTINCTION => {
                let (_, selected) = piece_target(state, target)?;
                let kind = selected.kind;
                let victims = state
                    .board
                    .iter()
                    .filter(|(_, piece)| piece.kind == kind)
                    .collect::<Vec<_>>();
                for (square, piece) in victims {
                    self.delete_piece_for_card(state, square, piece, actor, EXTINCTION, events);
                }
            }
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
                    if card.owner == actor
                        && card.charges == 0
                        && (self.ruleset.id != crate::LATEST_RULESET_ID
                            || self
                                .card_definition(&card.definition_id)
                                .is_some_and(|d| d.activation == CardActivation::Active))
                    {
                        card.charges = 1;
                        break;
                    }
                }
            }
            _ => return Ok(false),
        }
        Ok(true)
    }
    pub(in crate::engine) fn generate_end_card_actions(
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
            GOMOKU | crate::REVERSAL => push_none(actions),
            crate::EXTINCTION => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == actor
                        && !self.is_royal_piece(state, piece)
                        && matches!(
                            piece.kind,
                            PieceKind::Knight | PieceKind::Bishop | PieceKind::Camel
                        )
                    {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
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
                    let concealed = state.board.get(square).is_some_and(|piece| {
                        state.card_runtime.stealth_piece_ids.contains(&piece.id)
                            && !self.is_royal_piece(state, piece)
                    });
                    if (state.board.get(square).is_none() || concealed)
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
                for i in 0..empties.len() {
                    for j in (i + 1)..empties.len() {
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
}
