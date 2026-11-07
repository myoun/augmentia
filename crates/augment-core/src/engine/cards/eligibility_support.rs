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

// ---------------------------------------------------------------------------
// Draft draw requirements.
//
// Upstream decides whether a card may appear in a draft with
// `v19DraftRequirementMet` (main bundle 55767-55903), a switch over the catalog's
// `draw_requirement`. These helpers answer the same questions from board state, so a
// card's `eligibility` can match the catalog instead of denying every acquisition.
//
// `ANY` and `ALWAYS` are unconditionally true. `REMOVABLE_EFFECT` and
// `REPLAYABLE_LAST_MOVE` have no arm in that switch, so they fall through to its
// `default: return true`; they stay unconditional here for the same reason.
// ---------------------------------------------------------------------------

/// Whether `owner` has a piece of `kind`.
pub(super) fn has_kind(state: &GameState, owner: Player, kind: PieceKind) -> bool {
    state
        .board
        .iter()
        .any(|(_, piece)| piece.owner == owner && piece.kind == kind)
}

/// Whether `owner` has any non-royal piece, which is what the draw gate calls
/// `nonRoyal`: a piece that is neither royal by identity nor a merchant.
pub(super) fn has_non_royal(state: &GameState, owner: Player) -> bool {
    state
        .board
        .iter()
        .any(|(_, piece)| piece.owner == owner && !royal(state, piece))
}

/// Whether `owner` has a plain bishop.
///
/// The `OWN_PLAIN_BISHOP` arm has a missionary-specific spawn check, but only under
/// `usesThreeCardBuffs`, which compares the catalog hash against a list of internal
/// hashes. This snapshot's hash is not in that list, so upstream takes the plain
/// `ownType("bishop")` branch here (main bundle 55783-55785).
pub(super) fn has_plain_bishop(state: &GameState, owner: Player) -> bool {
    has_kind(state, owner, PieceKind::Bishop)
}

/// Number of empty, placeable cells on the board.
pub(super) fn empty_square_count(state: &GameState) -> usize {
    (0_u8..64)
        .filter_map(Square::from_index)
        .filter(|square| state.board.get(*square).is_none())
        .count()
}

/// Whether both sides have a non-royal piece of the same kind.
pub(super) fn has_matching_type(state: &GameState, owner: Player) -> bool {
    let opponent = owner.opponent();
    let enemy_kinds: BTreeSet<PieceKind> = state
        .board
        .iter()
        .filter(|(_, piece)| piece.owner == opponent && !royal(state, *piece))
        .map(|(_, piece)| piece.kind)
        .collect();
    state.board.iter().any(|(_, piece)| {
        piece.owner == owner && !royal(state, piece) && enemy_kinds.contains(&piece.kind)
    })
}

/// Whether the opponent holds a piece in the knight family.
///
/// The upstream arm branches on `usesNativePieceClassification`, which is a catalog
/// hash comparison. This snapshot's hash is not among those hashes, so upstream takes
/// the explicit list at main bundle 55833-55843 plus any compound piece whose movement
/// includes a knight. This core has no compound-knight spec table, so the explicit
/// kinds are what can be matched; a future snapshot with a native-classification hash
/// would need the compound form instead.
pub(super) fn opponent_has_knight_family(state: &GameState, owner: Player) -> bool {
    state.board.iter().any(|(_, piece)| {
        piece.owner == owner.opponent()
            && matches!(
                piece.kind,
                PieceKind::Knight
                    | PieceKind::Knightmaster
                    | PieceKind::RoyalKnight
                    | PieceKind::Amazon
                    | PieceKind::Cardinal
                    | PieceKind::Camel
                    | PieceKind::Unicorn
            )
    })
}
