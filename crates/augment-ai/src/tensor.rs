use augment_core::{GamePhase, GameState, PieceKind, Player, Square};

pub const SPATIAL_CHANNELS: usize = 64;
pub const SPATIAL_SIZE: usize = SPATIAL_CHANNELS * 8 * 8; // 4,096 f32s
pub const GLOBAL_FEATURES_SIZE: usize = 128; // 128 f32s

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EncodedState {
    pub spatial: Vec<f32>,
    pub global: Vec<f32>,
}

impl EncodedState {
    pub fn new() -> Self {
        Self {
            spatial: vec![0.0; SPATIAL_SIZE],
            global: vec![0.0; GLOBAL_FEATURES_SIZE],
        }
    }

    #[inline]
    pub fn set_spatial(&mut self, channel: usize, square: Square, value: f32) {
        if channel < SPATIAL_CHANNELS {
            let idx = channel * 64 + square.index();
            self.spatial[idx] = value;
        }
    }
}

impl Default for EncodedState {
    fn default() -> Self {
        Self::new()
    }
}

fn card_id_hash(id: &str) -> f32 {
    let mut hash: u32 = 5381;
    for byte in id.bytes() {
        hash = ((hash << 5).wrapping_add(hash)).wrapping_add(byte as u32);
    }
    ((hash % 256) as f32 + 1.0) / 256.0
}

fn canonical_piece_channel(kind: PieceKind, is_my_piece: bool) -> usize {
    let base_idx = match kind {
        PieceKind::Pawn => 0,
        PieceKind::Knight => 1,
        PieceKind::Bishop => 2,
        PieceKind::Rook => 3,
        PieceKind::Queen => 4,
        PieceKind::King => 5,
        PieceKind::Amazon => 6,
        PieceKind::Dragon => 7,
        PieceKind::Herald => 8,
        PieceKind::Wizard => 9,
        PieceKind::Reaper => 10,
        PieceKind::Checker | PieceKind::CheckerKing => 11,
        PieceKind::Guard | PieceKind::Fanatic => 12,
        PieceKind::Bear | PieceKind::BabyBear => 13,
        PieceKind::Lobster | PieceKind::Vip => 14,
        _ => 15,
    };
    if is_my_piece { base_idx } else { 16 + base_idx }
}

#[inline]
pub fn canonical_square(sq: Square, turn: Player) -> Square {
    match turn {
        Player::White => sq,
        Player::Black => Square::new(sq.file(), 7 - sq.rank()).unwrap(),
    }
}

/// Encodes the GameState from the perspective of the side-to-move (Player who is taking action).
/// Channel 0..15: My Pieces
/// Channel 16..31: Opponent Pieces
/// Channel 32..35: Neutrals
/// Channel 36..41: Status Overlays (Shield, Protect, Freeze, Disarm, Evasion, Stealth)
/// Channel 42..48: Terrain & Rule Overlays
/// Channel 49..51: My Density, Opponent Density, King Difference
/// Channel 52..55: Tactical Ray Reach & Pin Geometries
pub fn encode_state(state: &GameState) -> EncodedState {
    let mut encoded = EncodedState::new();
    let current_turn = state.turn;

    // 1. Spatial Piece Channels
    for (square, piece) in state.board.iter() {
        let c_sq = canonical_square(square, current_turn);
        if matches!(
            piece.kind,
            PieceKind::Monster | PieceKind::Scarecrow | PieceKind::Crown | PieceKind::Football
        ) {
            let neutral_channel = match piece.kind {
                PieceKind::Monster => 32,
                PieceKind::Crown => 33,
                PieceKind::Football => 34,
                _ => 35,
            };
            encoded.set_spatial(neutral_channel, c_sq, 1.0);
        } else {
            let is_my = piece.owner == current_turn;
            let ch = canonical_piece_channel(piece.kind, is_my);
            encoded.set_spatial(ch, c_sq, 1.0);
        }

        // Status overlays (36..41)
        if state.card_runtime.shielded_piece_ids.contains(&piece.id) {
            encoded.set_spatial(36, c_sq, 1.0);
        }
        if state.card_runtime.protected_piece_ids.contains(&piece.id) {
            encoded.set_spatial(37, c_sq, 1.0);
        }
        if state.card_runtime.frozen.contains_key(&piece.id) {
            encoded.set_spatial(38, c_sq, 1.0);
        }
        if state.card_runtime.disarmed.contains_key(&piece.id) {
            encoded.set_spatial(39, c_sq, 1.0);
        }
        if state.card_runtime.evasion_piece_ids.contains(&piece.id) {
            encoded.set_spatial(40, c_sq, 1.0);
        }
        if state.card_runtime.stealth_piece_ids.contains(&piece.id) {
            encoded.set_spatial(41, c_sq, 1.0);
        }
    }

    // 2. Quantum shadows
    for shadow_sq in state.card_runtime.quantum_positions.values() {
        encoded.set_spatial(42, canonical_square(*shadow_sq, current_turn), 1.0);
    }

    // 3. Terrain & Rule Overlays (43..48)
    for bh in &state.card_runtime.black_holes {
        encoded.set_spatial(43, canonical_square(*bh, current_turn), 1.0);
    }
    for bar in &state.card_runtime.barricades {
        encoded.set_spatial(44, canonical_square(*bar, current_turn), 1.0);
    }
    for hg in &state.card_runtime.high_ground_squares {
        encoded.set_spatial(45, canonical_square(*hg, current_turn), 1.0);
    }
    for bomb in &state.card_runtime.rule_bombs {
        encoded.set_spatial(46, canonical_square(*bomb, current_turn), 1.0);
    }
    if let Some(platform) = &state.card_runtime.platform {
        if let Some(cell) = platform.cell {
            encoded.set_spatial(47, canonical_square(cell, current_turn), 1.0);
        }
    }
    if let Some(ep) = state.en_passant {
        encoded.set_spatial(48, canonical_square(ep.target, current_turn), 1.0);
    }

    // 4. Density & King Channels (49..51)
    for (square, piece) in state.board.iter() {
        let c_sq = canonical_square(square, current_turn);
        if piece.owner == current_turn {
            encoded.set_spatial(49, c_sq, 1.0); // My pieces density
            if piece.kind == PieceKind::King {
                encoded.set_spatial(51, c_sq, 1.0); // My King
            }
        } else {
            encoded.set_spatial(50, c_sq, 1.0); // Opponent pieces density
            if piece.kind == PieceKind::King {
                encoded.set_spatial(51, c_sq, -1.0); // Opponent King
            }
        }
    }

    // 5. Tactical Reach & Pin Geometries (52..55)
    for (square, piece) in state.board.iter() {
        let is_my = piece.owner == current_turn;
        let reach_ch = if is_my { 52 } else { 53 };

        let directions: &[(i8, i8)] = match piece.kind {
            PieceKind::Rook => &[(-1, 0), (1, 0), (0, -1), (0, 1)],
            PieceKind::Bishop => &[(-1, -1), (-1, 1), (1, -1), (1, 1)],
            PieceKind::Queen => &[
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ],
            PieceKind::Knight => &[
                (-2, -1),
                (-2, 1),
                (-1, -2),
                (-1, 2),
                (1, -2),
                (1, 2),
                (2, -1),
                (2, 1),
            ],
            _ => &[],
        };

        if piece.kind == PieceKind::Knight {
            for &(df, dr) in directions {
                if let Some(target_sq) = square.offset(df, dr) {
                    encoded.set_spatial(reach_ch, canonical_square(target_sq, current_turn), 0.5);
                }
            }
        } else {
            for &(df, dr) in directions {
                let mut cur = square;
                while let Some(next_sq) = cur.offset(df, dr) {
                    encoded.set_spatial(reach_ch, canonical_square(next_sq, current_turn), 0.5);
                    if let Some(p) = state.board.get(next_sq) {
                        if p.owner != piece.owner
                            && (p.kind == PieceKind::King || p.kind == PieceKind::Queen)
                        {
                            encoded.set_spatial(55, canonical_square(next_sq, current_turn), 1.0);
                            encoded.set_spatial(54, canonical_square(next_sq, current_turn), 1.0);
                        }
                        break;
                    }
                    cur = next_sq;
                }
            }
        }
    }

    // 6. Global Features (0..127) - Canonicalized to side-to-move
    encoded.global[0] = state.actions_remaining as f32 / 2.0;
    encoded.global[1] = (state.ply as f32 / 100.0).min(1.0);
    encoded.global[2] = (state.quiet_ply as f32 / 50.0).min(1.0);
    let my_idx = current_turn.index();
    let opp_idx = current_turn.opponent().index();
    encoded.global[3] =
        (state.half_star_totals[my_idx] as f32 - state.half_star_totals[opp_idx] as f32) / 20.0;
    if let Some(dm) = state.deathmatch {
        encoded.global[4] = 1.0;
        encoded.global[5] = dm.quiet_full_turns as f32 / 10.0;
    }
    encoded.global[6] = state.cards.len() as f32 / 6.0;

    // Castling Rights (7..10: My Kingside, My Queenside, Opp Kingside, Opp Queenside)
    let moved = &state.card_runtime.moved_piece_ids;
    let my_rank = if current_turn == Player::White { 0 } else { 7 };
    let opp_rank = if current_turn == Player::White { 7 } else { 0 };

    if let Some(k) = state.board.get(Square::new(4, my_rank).unwrap()) {
        if k.kind == PieceKind::King && !moved.contains(&k.id) {
            if let Some(r) = state.board.get(Square::new(7, my_rank).unwrap()) {
                if r.kind == PieceKind::Rook && !moved.contains(&r.id) {
                    encoded.global[7] = 1.0;
                }
            }
            if let Some(r) = state.board.get(Square::new(0, my_rank).unwrap()) {
                if r.kind == PieceKind::Rook && !moved.contains(&r.id) {
                    encoded.global[8] = 1.0;
                }
            }
        }
    }
    if let Some(k) = state.board.get(Square::new(4, opp_rank).unwrap()) {
        if k.kind == PieceKind::King && !moved.contains(&k.id) {
            if let Some(r) = state.board.get(Square::new(7, opp_rank).unwrap()) {
                if r.kind == PieceKind::Rook && !moved.contains(&r.id) {
                    encoded.global[9] = 1.0;
                }
            }
            if let Some(r) = state.board.get(Square::new(0, opp_rank).unwrap()) {
                if r.kind == PieceKind::Rook && !moved.contains(&r.id) {
                    encoded.global[10] = 1.0;
                }
            }
        }
    }

    // Phase Indicator (index 11)
    encoded.global[11] = match &state.phase {
        GamePhase::Playing => 0.0,
        GamePhase::ForcedMove { .. } => 0.33,
        GamePhase::Draft(_) => 0.66,
        GamePhase::AwaitingChoice(_) => 1.0,
    };

    // My Hand Cards (indices 12..17)
    let mut my_c = 0;
    for card in state.cards.iter().filter(|c| c.owner == current_turn) {
        if my_c < 3 {
            encoded.global[12 + my_c * 2] = card_id_hash(&card.definition_id);
            encoded.global[13 + my_c * 2] = card.charges as f32 / 3.0;
            my_c += 1;
        }
    }

    // Opponent Hand Cards (indices 18..23)
    let mut opp_c = 0;
    for card in state
        .cards
        .iter()
        .filter(|c| c.owner == current_turn.opponent())
    {
        if opp_c < 3 {
            encoded.global[18 + opp_c * 2] = card_id_hash(&card.definition_id);
            encoded.global[19 + opp_c * 2] = card.charges as f32 / 3.0;
            opp_c += 1;
        }
    }

    // Draft State Offers (indices 24..31)
    if let GamePhase::Draft(draft) = &state.phase {
        for (i, offer) in draft.offers.iter().enumerate().take(8) {
            encoded.global[24 + i] = card_id_hash(offer);
        }
    }

    encoded
}
