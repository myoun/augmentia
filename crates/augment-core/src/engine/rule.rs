use super::*;
use crate::*;

impl Engine {
    pub(super) fn install_rule_ext(
        &self,
        state: &mut GameState,
        definition_id: &str,
        events: &mut Vec<Event>,
    ) -> Result<bool, EngineError> {
        match definition_id {
            ACCELERATION => {
                state.card_runtime.acceleration = true;
            }
            BLACK_HOLE => {
                for sq in ["d4", "d5", "e4", "e5"] {
                    state.card_runtime.black_holes.insert(sq.parse().unwrap());
                }
            }
            CAMOUFLAGE_COLOR => {
                state.card_runtime.camouflage_color = true;
            }
            CHESS_344200 => {
                let pieces = state.board.iter().map(|(_, p)| p).collect::<Vec<_>>();
                let mut squares = (0..64).filter_map(Square::from_index).collect::<Vec<_>>();
                for piece in pieces {
                    if let Some(sq) = squares.pop() {
                        state.board.set(sq, Some(piece));
                    }
                }
            }
            CHESS_960 => {
                for player in [Player::White, Player::Black] {
                    let rank = if player == Player::White { 0 } else { 7 };
                    let mut pieces = (0..8)
                        .filter_map(|f| Square::new(f, rank))
                        .filter_map(|sq| state.board.get(sq))
                        .collect::<Vec<_>>();
                    let (roll, idx) = state.rng_mut().web_choice_index(pieces.len().max(1));
                    events.push(Event::RandomSample {
                        purpose: "chess-960".to_owned(),
                        roll_53: roll,
                    });
                    let len = pieces.len().max(1);
                    if let Some(idx) = idx {
                        pieces.rotate_left(idx % len);
                    }
                    for (file, piece) in pieces.into_iter().enumerate() {
                        if let Some(sq) = Square::new(file as u8, rank) {
                            state.board.set(sq, Some(piece));
                        }
                    }
                }
            }
            CONVEYOR => {
                state.card_runtime.conveyor = true;
            }
            COOL_GUY => {
                state.card_runtime.cool_guy_pieces = BTreeSet::new();
            }
            CROWN => {
                let center_squares = ["d4", "d5", "e4", "e5"];
                let (roll, idx) = state.rng_mut().web_choice_index(center_squares.len());
                events.push(Event::RandomSample {
                    purpose: "crown-spawn".to_owned(),
                    roll_53: roll,
                });
                let sq: Square = center_squares[idx.unwrap_or(0)].parse().unwrap();
                state.card_runtime.crown_pos = Some(sq);
            }
            DIAGONAL_CHESS => {}
            FOOTBALL => {
                let center_squares = ["d4", "d5", "e4", "e5"];
                let sq: Square = center_squares[0].parse().unwrap();
                state.card_runtime.football_pos = Some(sq);
            }
            HIGH_GROUND => {
                for sq in ["c3", "c4", "c5", "c6", "f3", "f4", "f5", "f6"] {
                    state
                        .card_runtime
                        .high_ground_squares
                        .insert(sq.parse().unwrap());
                }
            }
            HIGHWAY => {
                state.card_runtime.highways.insert(1); // file b
                state.card_runtime.highways.insert(6); // file g
            }
            MACHO_CHESS => {
                state.card_runtime.macho_chess = true;
            }
            MISTAKE => {
                state.card_runtime.mistake_rule = true;
            }
            MONOCHROME_CHESS => {
                state.card_runtime.monochrome_chess = true;
                let knights = state
                    .board
                    .iter()
                    .filter(|(_, p)| p.kind == PieceKind::Knight)
                    .collect::<Vec<_>>();
                for (sq, mut piece) in knights {
                    piece.kind = PieceKind::Camel;
                    state.board.set(sq, Some(piece));
                }
            }
            MONSTER_RULE => {
                let center_squares = ["d4", "d5", "e4", "e5"];
                let sq: Square = center_squares[0].parse().unwrap();
                let monster = Piece::new(Player::White, PieceKind::Monster)
                    .with_id(state.allocate_piece_id());
                state.board.set(sq, Some(monster));
                state.card_runtime.monster_rule = Some(monster.id);
            }
            PERIODIC_COLLAPSE => {
                state.card_runtime.periodic_collapse_next = Some(state.ply.saturating_add(20));
            }
            PORTAL => {
                state.card_runtime.rule_portals =
                    Some(["b2".parse().unwrap(), "g7".parse().unwrap()]);
            }
            RECYCLING => {
                state.card_runtime.recycling = true;
            }
            RULE_BOMBS => {
                for sq in ["c4", "e4", "d5"] {
                    state.card_runtime.rule_bombs.insert(sq.parse().unwrap());
                }
            }
            SATURATION => {
                state.card_runtime.saturation_rule = true;
            }
            TRANSCENDENCE => {
                state.card_runtime.transcendence = true;
            }
            WINTER_KINGDOM => {
                state.card_runtime.winter_kingdom_next =
                    Some(state.turns_taken[0].saturating_add(3));
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    pub(super) fn resolve_rule_after_move(
        &self,
        state: &mut GameState,
        piece: Piece,
        movement: MoveAction,
        captured: Option<(Square, Piece)>,
        events: &mut Vec<Event>,
    ) {
        let actor = piece.owner;

        // Black Hole check
        if state.card_runtime.black_holes.contains(&movement.to) {
            if let Some((sq, p)) = state.board.find_by_id(piece.id) {
                self.remove_piece_for_card(state, sq, p, actor, BLACK_HOLE, events);
            }
        }

        // Bombs check
        if state.card_runtime.rule_bombs.remove(&movement.to) {
            let victims = state
                .board
                .iter()
                .filter(|(sq, _)| {
                    sq.file() == movement.to.file() || sq.rank() == movement.to.rank()
                })
                .collect::<Vec<_>>();
            for (sq, victim) in victims {
                self.remove_piece_for_card(state, sq, victim, actor, RULE_BOMBS, events);
            }
        }

        // Transcendence: Pawn -> Knight -> Rook -> Queen upon direct capture
        if captured.is_some() && state.card_runtime.transcendence {
            if let Some((sq, mut p)) = state.board.find_by_id(piece.id) {
                p.kind = match p.kind {
                    PieceKind::Pawn => PieceKind::Knight,
                    PieceKind::Knight => PieceKind::Rook,
                    PieceKind::Rook => PieceKind::Queen,
                    other => other,
                };
                state.board.set(sq, Some(p));
            }
        }

        // Crown check
        if let Some(crown_sq) = state.card_runtime.crown_pos {
            if movement.to == crown_sq {
                state.card_runtime.crown_pos = None;
                state.card_runtime.crown_holder = Some((actor, 1));
            }
        } else if let Some((holder, count)) = state.card_runtime.crown_holder {
            if holder == actor {
                let new_count = count.saturating_add(1);
                if new_count >= 10 {
                    state.status = GameStatus::Won {
                        winner: actor,
                        reason: WinReason::CardEffect("crown".to_owned()),
                    };
                } else {
                    state.card_runtime.crown_holder = Some((holder, new_count));
                }
            }
        }

        // Camouflage Color
        if state.card_runtime.camouflage_color {
            let is_light = (movement.to.file() + movement.to.rank()) % 2 != 0;
            if (actor == Player::White && is_light) || (actor == Player::Black && !is_light) {
                state.card_runtime.stealth_piece_ids.insert(piece.id);
            } else {
                state.card_runtime.stealth_piece_ids.remove(&piece.id);
            }
        }
    }
}
