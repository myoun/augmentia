//! Reproducible features for a recorded decision.
//!
//! A policy cannot consume a `GameState`, and a recorded decision must be verifiable later without
//! the live engine. So each decision is reduced to a fixed-length vector derived only from public
//! state. Two rules govern this module:
//!
//! - **Public only.** No feature reads hidden information, because a policy trained on it could not
//!   legally run in a real game.
//! - **Reproducible.** The same state must always yield the same vector, so the encoding is
//!   positional and integer-valued rather than dependent on iteration order.
//!
//! Actions are encoded as (kind, from, to, promotion) so a policy can score an action without
//! reconstructing the engine's `Action` enum.

use augment_core::{Action, ActionTarget, GameState, PieceKind, Player};
use serde::Serialize;

/// Number of floats describing a position.
pub const POSITION_FEATURES: usize = 64;
/// Number of floats describing an action.
pub const ACTION_FEATURES: usize = 16;

/// A fixed-length description of a position.
#[derive(Clone, Debug, Serialize)]
pub struct PositionFeatures(pub Vec<f32>);

impl PositionFeatures {
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }
}

/// A fixed-length description of one candidate action.
#[derive(Clone, Debug, Serialize)]
pub struct ActionFeatures(pub Vec<f32>);

impl ActionFeatures {
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }
}

/// Piece ordering used by the encoding, so indices are stable.
const KINDS: [PieceKind; 16] = [
    PieceKind::Pawn,
    PieceKind::Knight,
    PieceKind::Bishop,
    PieceKind::Rook,
    PieceKind::Queen,
    PieceKind::King,
    PieceKind::Archbishop,
    PieceKind::Chancellor,
    PieceKind::Monster,
    PieceKind::Offering,
    PieceKind::Wanderer,
    PieceKind::OmegaChesstron,
    PieceKind::Paladin,
    PieceKind::Cannon,
    PieceKind::SiegeRam,
    PieceKind::Dragon,
];

/// Describe the position from `actor`'s point of view.
///
/// Blocks: per-piece-kind counts for both sides, king neighbourhood pressure, card resources,
/// turn budget, star totals, and phase flags. Any unused tail stays zero so the length is fixed.
pub fn position_features(state: &GameState, actor: Player) -> PositionFeatures {
    let mut features = vec![0.0f32; POSITION_FEATURES];
    let opponent = actor.opponent();

    // 0..16: our piece counts by kind. 16..32: theirs.
    for (_, piece) in state.board.iter() {
        let Some(index) = KINDS.iter().position(|kind| *kind == piece.kind) else {
            continue;
        };
        let base = if piece.owner == actor { 0 } else { 16 };
        features[base + index] += 1.0;
    }

    // 32: total material difference in pawn-like units.
    let value = |kind: PieceKind| -> f32 {
        match kind {
            PieceKind::Pawn => 1.0,
            PieceKind::Knight | PieceKind::Bishop => 3.0,
            PieceKind::Rook => 5.0,
            PieceKind::Queen => 9.0,
            PieceKind::King => 0.0,
            _ => 3.0,
        }
    };
    for (_, piece) in state.board.iter() {
        let signed = if piece.owner == actor { 1.0 } else { -1.0 };
        features[32] += signed * value(piece.kind);
    }

    // 33: are we the side to move.
    features[33] = if state.turn == actor { 1.0 } else { 0.0 };
    // 34: actions remaining this turn, normalised.
    features[34] = f32::from(state.actions_remaining).min(8.0) / 8.0;
    // 35: ply, capped, so a policy can tell opening from endgame.
    features[35] = (state.ply as f32 / 100.0).min(1.0);
    // 36: quiet plies, which drive the automatic draw.
    features[36] = (state.quiet_ply as f32 / 100.0).min(1.0);

    // 37..40: cards we hold, charges and installed. 40..43: theirs.
    for card in &state.cards {
        let base = if card.owner == actor { 37 } else { 40 };
        features[base] += 1.0;
        features[base + 1] += f32::from(card.charges).min(4.0) / 4.0;
        if card.charges == 0 {
            features[base + 2] += 1.0;
        }
    }

    // 43: stars we lead by, normalised around zero.
    let our_stars = f32::from(state.half_star_totals[actor.index()]);
    let their_stars = f32::from(state.half_star_totals[opponent.index()]);
    features[43] = ((our_stars - their_stars) / 20.0).clamp(-1.0, 1.0);

    // 44..48: which draft phases we have completed.
    for (index, done) in state.completed_drafts.iter().enumerate() {
        features[44 + index] = if *done { 1.0 } else { 0.0 };
    }

    // 48: king safety, counting enemy pieces within two files and ranks of our king.
    if let Some((king_square, _)) = state
        .board
        .iter()
        .find(|(_, piece)| piece.owner == actor && piece.kind == PieceKind::King)
    {
        let pressure = state
            .board
            .iter()
            .filter(|(square, piece)| {
                piece.owner == opponent
                    && piece.kind != PieceKind::King
                    && square.file().abs_diff(king_square.file()) <= 2
                    && square.rank().abs_diff(king_square.rank()) <= 2
            })
            .count();
        features[48] = (pressure as f32 / 8.0).min(1.0);
    }

    // 49: repetition count for this position, which is a draw signal.
    features[49] = (state.current_repetition_count() as f32 / 3.0).min(1.0);

    PositionFeatures(features)
}

/// Describe one candidate action.
///
/// The encoding is deliberately flat: a policy scores an action without needing the engine, so the
/// vector carries only what identifies and characterises it.
pub fn action_features(state: &GameState, action: &Action) -> ActionFeatures {
    let mut features = vec![0.0f32; ACTION_FEATURES];
    match action {
        Action::Move(movement) => {
            features[0] = 1.0;
            features[1] = f32::from(movement.from.file()) / 7.0;
            features[2] = f32::from(movement.from.rank()) / 7.0;
            features[3] = f32::from(movement.to.file()) / 7.0;
            features[4] = f32::from(movement.to.rank()) / 7.0;
            // Distance travelled, which separates short consolidating moves from long raids.
            let df = f32::from(movement.from.file().abs_diff(movement.to.file()));
            let dr = f32::from(movement.from.rank().abs_diff(movement.to.rank()));
            features[5] = ((df * df + dr * dr).sqrt() / 7.0).min(1.0);
            // A capture is the single most informative property of a move.
            features[6] = if state
                .board
                .get(movement.to)
                .is_some_and(|piece| piece.owner != state.turn)
            {
                1.0
            } else {
                0.0
            };
            features[7] = if movement.promotion.is_some() {
                1.0
            } else {
                0.0
            };
            // Which piece kind is moving, as a coarse class.
            if let Some(piece) = state.board.get(movement.from) {
                let index = KINDS
                    .iter()
                    .position(|kind| *kind == piece.kind)
                    .unwrap_or(0);
                features[8] = index as f32 / 16.0;
            }
        }
        Action::UseCard { .. } => {
            features[0] = 3.0;
            features[9] = 1.0;
        }
        Action::UsePieceAbility { .. } => {
            features[0] = 2.0;
            features[10] = 1.0;
        }
        Action::DraftPick { offer_index } => {
            features[0] = 4.0;
            features[11] = f32::from(*offer_index).min(3.0) / 3.0;
        }
        Action::ResolveChoice { option_index, .. } => {
            features[0] = 5.0;
            features[12] = f32::from(*option_index).min(3.0) / 3.0;
        }
        Action::SkipOptionalMove => {
            features[0] = 6.0;
        }
    }
    // 13: how many squares the action selects, which distinguishes a simple move from a multi-step
    // card declaration.
    features[13] = match action {
        Action::UseCard {
            target: ActionTarget::Squares(squares),
            ..
        }
        | Action::UsePieceAbility {
            target: ActionTarget::Squares(squares),
            ..
        } => (squares.len() as f32 / 8.0).min(1.0),
        Action::UseCard {
            target: ActionTarget::Square(_) | ActionTarget::PieceAt(_),
            ..
        }
        | Action::UsePieceAbility {
            target: ActionTarget::Square(_) | ActionTarget::PieceAt(_),
            ..
        } => 0.125,
        _ => 0.0,
    };
    ActionFeatures(features)
}
