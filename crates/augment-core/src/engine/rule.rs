use super::*;
use crate::*;

const CHESS_N_POW_30_PIECES_PER_SIDE: usize = 15;
const CHESS_N_POW_30_HOME_RANK_SLOTS: usize = 7;
const CHESS_N_POW_30_MIN_TOTAL: u8 = 39;
const CHESS_N_POW_30_MAX_TOTAL: u8 = 70;

pub(super) const CHESS_N_POW_30_KINDS: &[(PieceKind, u8, bool)] = &[
    (PieceKind::Queen, 9, true),
    (PieceKind::Rook, 5, true),
    (PieceKind::Bishop, 3, true),
    (PieceKind::Missionary, 3, false),
    (PieceKind::Knight, 3, false),
    (PieceKind::Pawn, 1, false),
    (PieceKind::Protestant, 3, true),
    (PieceKind::Herald, 5, true),
    (PieceKind::Cannon, 4, true),
    (PieceKind::Fanatic, 1, false),
    (PieceKind::PrimeMinister, 9, false),
    (PieceKind::Alibaba, 2, false),
    (PieceKind::Amazon, 13, true),
    (PieceKind::Cardinal, 7, true),
    (PieceKind::Unicorn, 4, false),
    (PieceKind::Jester, 9, true),
    (PieceKind::Camel, 2, false),
    (PieceKind::Log, 2, false),
    (PieceKind::Hook, 14, true),
    (PieceKind::Grasshopper, 5, false),
    (PieceKind::Dragon, 5, false),
    (PieceKind::Man, 4, false),
    (PieceKind::Assassin, 4, false),
    (PieceKind::Reaper, 9, false),
    (PieceKind::Knightmaster, 3, false),
    (PieceKind::StandardBearer, 2, false),
    (PieceKind::Guard, 2, false),
    (PieceKind::Recruiter, 9, false),
    (PieceKind::Squire, 1, false),
    (PieceKind::Checker, 2, false),
    (PieceKind::CheckerKing, 4, false),
    (PieceKind::Wizard, 9, false),
    (PieceKind::Alfil, 1, false),
    (PieceKind::Windmill, 4, true),
    (PieceKind::Idol, 9, true),
    (PieceKind::Lobster, 2, false),
    (PieceKind::BabyBear, 4, false),
    (PieceKind::Bear, 15, true),
    (PieceKind::SiegeRam, 5, false),
    (PieceKind::MagicGirl, 6, false),
    (PieceKind::Berserker, 7, false),
    (PieceKind::Slime, 5, false),
    (PieceKind::Siren, 9, false),
    (PieceKind::Trickster, 9, false),
    (PieceKind::Undead, 9, false),
];

fn chess_n_pow_30_lineup_ways() -> [[f64; 71]; 16] {
    let mut ways = [[0.0_f64; 71]; 16];
    ways[0][0] = 1.0;
    for slots in 1..=CHESS_N_POW_30_PIECES_PER_SIDE {
        for total in 0..=usize::from(CHESS_N_POW_30_MAX_TOTAL) {
            ways[slots][total] = CHESS_N_POW_30_KINDS
                .iter()
                .filter_map(|(_, value, _)| {
                    let value = usize::from(*value);
                    if value <= total {
                        Some(ways[slots - 1][total - value])
                    } else {
                        None
                    }
                })
                .sum();
        }
    }
    ways
}

fn weighted_index_f64(
    state: &mut GameState,
    events: &mut Vec<Event>,
    weights: &[f64],
    purpose: String,
) -> usize {
    let total = weights.iter().copied().sum::<f64>();
    debug_assert!(total > 0.0);
    let (roll, _) = state.rng_mut().web_choice_index(1);
    events.push(Event::RandomSample {
        purpose,
        roll_53: roll,
    });
    // Match JavaScript Number arithmetic and its subtractive weighted selection.
    let mut threshold = (roll as f64 / ((1_u64 << 53) as f64)) * total;
    for (index, weight) in weights.iter().copied().enumerate() {
        if threshold < weight {
            return index;
        }
        threshold -= weight;
    }
    weights.len().saturating_sub(1)
}

fn chess_n_pow_30_lineup(
    state: &mut GameState,
    events: &mut Vec<Event>,
    ways: &[[f64; 71]; 16],
    total_value: u8,
    label: &str,
) -> Vec<PieceKind> {
    let mut lineup = Vec::with_capacity(CHESS_N_POW_30_PIECES_PER_SIDE);
    let mut remaining = usize::from(total_value);
    for slots in (1..=CHESS_N_POW_30_PIECES_PER_SIDE).rev() {
        let weights = CHESS_N_POW_30_KINDS
            .iter()
            .map(|(_, value, _)| {
                let value = usize::from(*value);
                if value <= remaining {
                    ways[slots - 1][remaining - value]
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        let selected = weighted_index_f64(
            state,
            events,
            &weights,
            format!("chess-n-pow-30-lineup:{label}:{slots}"),
        );
        let (kind, value, _) = CHESS_N_POW_30_KINDS[selected];
        lineup.push(kind);
        remaining -= usize::from(value);
    }
    debug_assert_eq!(remaining, 0);
    lineup
}

fn shuffle_chess_n_pow_30(
    state: &mut GameState,
    events: &mut Vec<Event>,
    values: &mut [PieceKind],
    purpose: &str,
) {
    for index in (1..values.len()).rev() {
        let (roll, selected) = state.rng_mut().web_choice_index(index + 1);
        events.push(Event::RandomSample {
            purpose: format!("{purpose}:{index}"),
            roll_53: roll,
        });
        values.swap(index, selected.unwrap_or(0));
    }
}

fn arrange_chess_n_pow_30_lineup(
    state: &mut GameState,
    events: &mut Vec<Event>,
    lineup: Vec<PieceKind>,
    label: &str,
) -> Vec<PieceKind> {
    let mut remaining = lineup;
    let mut home_rank = Vec::with_capacity(CHESS_N_POW_30_HOME_RANK_SLOTS);
    while home_rank.len() < CHESS_N_POW_30_HOME_RANK_SLOTS {
        let weights = remaining
            .iter()
            .map(|kind| {
                let (_, value, ranged) = CHESS_N_POW_30_KINDS
                    .iter()
                    .find(|(candidate, _, _)| candidate == kind)
                    .copied()
                    .expect("chaos lineup kind is in the value table");
                0.5 + f64::from(value) * 0.1 + if ranged { 2.0 } else { 0.0 }
            })
            .collect::<Vec<_>>();
        let selected = weighted_index_f64(
            state,
            events,
            &weights,
            format!("chess-n-pow-30-home:{label}:{}", home_rank.len()),
        );
        home_rank.push(remaining.remove(selected));
    }
    shuffle_chess_n_pow_30(
        state,
        events,
        &mut home_rank,
        &format!("chess-n-pow-30-home-shuffle:{label}"),
    );
    shuffle_chess_n_pow_30(
        state,
        events,
        &mut remaining,
        &format!("chess-n-pow-30-pawn-shuffle:{label}"),
    );
    home_rank.extend(remaining);
    home_rank
}

impl Engine {
    pub(super) fn install_rule_ext(
        &self,
        state: &mut GameState,
        definition_id: &str,
        events: &mut Vec<Event>,
    ) -> Result<bool, EngineError> {
        match definition_id {
            CHESS_N_POW_30 => {
                let ways = chess_n_pow_30_lineup_ways();
                let feasible_totals = (CHESS_N_POW_30_MIN_TOTAL..=CHESS_N_POW_30_MAX_TOTAL)
                    .filter(|total| ways[CHESS_N_POW_30_PIECES_PER_SIDE][*total as usize] > 0.0)
                    .collect::<Vec<_>>();
                let (roll, total_index) = state.rng_mut().web_choice_index(feasible_totals.len());
                events.push(Event::RandomSample {
                    purpose: "chess-n-pow-30-total".to_owned(),
                    roll_53: roll,
                });
                let total_value = feasible_totals[total_index.unwrap_or(0)];
                let white_lineup =
                    chess_n_pow_30_lineup(state, events, &ways, total_value, "white");
                let mut black_lineup =
                    chess_n_pow_30_lineup(state, events, &ways, total_value, "black");
                for attempt in 0..8 {
                    if white_lineup != black_lineup {
                        break;
                    }
                    black_lineup = chess_n_pow_30_lineup(
                        state,
                        events,
                        &ways,
                        total_value,
                        &format!("black-reroll-{attempt}"),
                    );
                }
                let white_lineup =
                    arrange_chess_n_pow_30_lineup(state, events, white_lineup, "white");
                let black_lineup =
                    arrange_chess_n_pow_30_lineup(state, events, black_lineup, "black");
                state.board = crate::Board::empty();
                for (player, player_lineup) in
                    [(Player::White, white_lineup), (Player::Black, black_lineup)]
                {
                    let king_square = if player == Player::White { "e1" } else { "e8" }
                        .parse()
                        .expect("static king square");
                    let king =
                        Piece::new(player, PieceKind::King).with_id(state.allocate_piece_id());
                    state.board.set(king_square, Some(king));
                    self.register_spawned_piece(state, king_square, king, false);
                    let squares = if player == Player::White {
                        [
                            "a1", "b1", "c1", "d1", "f1", "g1", "h1", "a2", "b2", "c2", "d2", "e2",
                            "f2", "g2", "h2",
                        ]
                    } else {
                        [
                            "a8", "b8", "c8", "d8", "f8", "g8", "h8", "a7", "b7", "c7", "d7", "e7",
                            "f7", "g7", "h7",
                        ]
                    };
                    for (name, kind) in squares.into_iter().zip(player_lineup) {
                        let square = name.parse().expect("static chaos square");
                        let piece = Piece::new(player, kind).with_id(state.allocate_piece_id());
                        state.board.set(square, Some(piece));
                        self.register_spawned_piece(state, square, piece, true);
                        match kind {
                            PieceKind::Reaper => {
                                state.card_runtime.reaper_souls.insert(piece.id, 0);
                            }
                            PieceKind::Wizard => {
                                state.card_runtime.wizard_mana.insert(piece.id, 0);
                            }
                            PieceKind::Trickster => {
                                let kind = self.choose_trickster_kind(state, events);
                                state
                                    .card_runtime
                                    .trickster_move_kinds
                                    .insert(piece.id, kind);
                            }
                            _ => {}
                        }
                    }
                }
                state.en_passant = None;
                state.card_runtime.chaos_no_capture_until_ply = state.ply.saturating_add(4);
            }
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
                let mut kings = [None, None];
                let mut other_pieces = Vec::new();
                for (_, piece) in state.board.iter() {
                    if piece.kind == PieceKind::King && kings[piece.owner.index()].is_none() {
                        kings[piece.owner.index()] = Some(piece);
                    } else {
                        other_pieces.push(piece);
                    }
                }
                if kings.iter().any(Option::is_none) {
                    return Err(EngineError::Invariant(
                        "344200 chess requires one king per player".to_owned(),
                    ));
                }

                let mut available = (0..64).filter_map(Square::from_index).collect::<Vec<_>>();
                state.board = Board::empty();
                state.card_runtime.piece_origins.clear();

                for player in [Player::White, Player::Black] {
                    let home_rank = if player == Player::White { 0 } else { 7 };
                    let candidates = available
                        .iter()
                        .enumerate()
                        .filter_map(|(index, square)| (square.rank() == home_rank).then_some(index))
                        .collect::<Vec<_>>();
                    let (roll, choice) = state.rng_mut().web_choice_index(candidates.len());
                    events.push(Event::RandomSample {
                        purpose: format!("chess-344200-{}-king", player.index()),
                        roll_53: roll,
                    });
                    let available_index = candidates[choice.ok_or_else(|| {
                        EngineError::Invariant("344200 chess has no king square".to_owned())
                    })?];
                    let square = available.remove(available_index);
                    let king = kings[player.index()].expect("king presence was validated");
                    state.board.set(square, Some(king));
                    state.card_runtime.piece_origins.insert(king.id, square);
                }

                for (index, piece) in other_pieces.into_iter().enumerate() {
                    let (roll, choice) = state.rng_mut().web_choice_index(available.len());
                    events.push(Event::RandomSample {
                        purpose: format!("chess-344200-piece-{index}"),
                        roll_53: roll,
                    });
                    let square = available.remove(choice.ok_or_else(|| {
                        EngineError::Invariant("344200 chess ran out of squares".to_owned())
                    })?);
                    state.board.set(square, Some(piece));
                    state.card_runtime.piece_origins.insert(piece.id, square);
                }
                state.en_passant = None;
                state.reset_repetition_history();
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
