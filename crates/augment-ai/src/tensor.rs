use augment_core::{GameState, PieceKind, Player, Square};

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

fn piece_channel(kind: PieceKind, owner: Player) -> usize {
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
    if owner == Player::White {
        base_idx
    } else {
        16 + base_idx
    }
}

pub fn encode_state(state: &GameState) -> EncodedState {
    let mut encoded = EncodedState::new();

    // 1. Spatial Piece Channels
    for (square, piece) in state.board.iter() {
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
            encoded.set_spatial(neutral_channel, square, 1.0);
        } else {
            let ch = piece_channel(piece.kind, piece.owner);
            encoded.set_spatial(ch, square, 1.0);
        }

        // Status overlays
        if state.card_runtime.shielded_piece_ids.contains(&piece.id) {
            encoded.set_spatial(36, square, 1.0);
        }
        if state.card_runtime.protected_piece_ids.contains(&piece.id) {
            encoded.set_spatial(37, square, 1.0);
        }
        if state.card_runtime.frozen.contains_key(&piece.id) {
            encoded.set_spatial(38, square, 1.0);
        }
        if state.card_runtime.disarmed.contains_key(&piece.id) {
            encoded.set_spatial(39, square, 1.0);
        }
        if state.card_runtime.evasion_piece_ids.contains(&piece.id) {
            encoded.set_spatial(40, square, 1.0);
        }
        if state.card_runtime.stealth_piece_ids.contains(&piece.id) {
            encoded.set_spatial(41, square, 1.0);
        }
    }

    // 2. Quantum shadows
    for shadow_sq in state.card_runtime.quantum_positions.values() {
        encoded.set_spatial(42, *shadow_sq, 1.0);
    }

    // 3. Terrain & Rule Overlays
    for bh in &state.card_runtime.black_holes {
        encoded.set_spatial(43, *bh, 1.0);
    }
    for bar in &state.card_runtime.barricades {
        encoded.set_spatial(44, *bar, 1.0);
    }
    for hg in &state.card_runtime.high_ground_squares {
        encoded.set_spatial(45, *hg, 1.0);
    }
    for bomb in &state.card_runtime.rule_bombs {
        encoded.set_spatial(46, *bomb, 1.0);
    }
    if let Some(platform) = &state.card_runtime.platform {
        if let Some(cell) = platform.cell {
            encoded.set_spatial(47, cell, 1.0);
        }
    }
    if let Some(ep) = state.en_passant {
        encoded.set_spatial(48, ep.target, 1.0);
    }

    // 4. Global Scalar Features
    encoded.global[0] = if state.turn == Player::White {
        1.0
    } else {
        -1.0
    };
    encoded.global[1] = state.actions_remaining as f32 / 2.0;
    encoded.global[2] = (state.ply as f32 / 100.0).min(1.0);
    encoded.global[3] = (state.quiet_ply as f32 / 50.0).min(1.0);
    encoded.global[4] =
        (state.half_star_totals[0] as f32 - state.half_star_totals[1] as f32) / 20.0;
    if let Some(dm) = state.deathmatch {
        encoded.global[5] = 1.0;
        encoded.global[6] = dm.quiet_full_turns as f32 / 10.0;
    }
    encoded.global[7] = state.cards.len() as f32 / 6.0;

    encoded
}
