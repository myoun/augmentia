use super::*;
use crate::{EligibilityReason, EligibilityReasonCode};
pub(super) fn denied(code: EligibilityReasonCode, detail: &str, message: &str) -> CardEligibility {
    CardEligibility::Ineligible(EligibilityReason {
        code,
        detail: detail.to_owned(),
        message_ko: message.to_owned(),
    })
}

pub(super) fn require(ok: bool, requirement: &str) -> CardEligibility {
    if ok {
        CardEligibility::Eligible
    } else {
        denied(
            EligibilityReasonCode::MissingBoardRequirement,
            requirement,
            requirement,
        )
    }
}

pub(super) fn enabled(flag: bool) -> Option<CardEligibility> {
    flag.then(|| {
        denied(
            EligibilityReasonCode::AlreadyEnabled,
            "runtime-enabled",
            "이미 적용 중인 효과입니다.",
        )
    })
}

// Do not use Engine::is_royal_piece: that helper additionally treats merchants
// as royal. JS isRoyalIdentityPiece does not; witch-trial excludes them separately.
pub(super) fn royal(state: &GameState, piece: Piece) -> bool {
    piece.kind != PieceKind::Merchant
        && (matches!(
            piece.kind,
            PieceKind::King | PieceKind::RoyalKnight | PieceKind::DarkWizard
        ) || state.card_runtime.royal_piece_ids.contains(&piece.id))
}

pub(super) fn queen(state: &GameState, piece: Piece) -> bool {
    // Core's Qxe1 heir is represented by royal_piece_ids, not regencyHeir.
    piece.kind == PieceKind::Queen && !state.card_runtime.royal_piece_ids.contains(&piece.id)
}
