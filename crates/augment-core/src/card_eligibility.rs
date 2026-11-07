//! Checked acquisition eligibility, not activation legality or passive installation.
//!
//! Source: September 15 official JS `isCardDraftDrawableForColor` (54556),
//! `isCardDrawableForColor` (54562), and their helpers. Draft pool category,
//! milestone, AI-specific exclusions, and compatibility *between offers* belong
//! to the caller. Core has no cardBanIds/completeRandom/editor presentation-only
//! rule fields: callers must enforce external bans; global uniqueness here is
//! the normal (not completeRandom) policy. Unsupported predicates fail closed.

mod compat;

use crate::{CardActivation, CardPhase, GameState, Piece, PieceKind, Player};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AcquisitionContext {
    /// Normal draft acquisition; future-target exceptions apply.
    Draft,
    /// Editor/setup acquisition; RULE cards are permitted, targets are checked.
    Setup,
}

/// Stable machine-readable rejection codes. `detail` identifies the conflicting
/// card/rule or the official predicate that cannot yet be reconstructed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EligibilityReasonCode {
    UnknownCard,
    RuleCardInDraft,
    AlreadyAcquired,
    RuleAlreadyApplied,
    MutuallyExclusive,
    BlockedByRule,
    AlreadyEnabled,
    MissingBoardRequirement,
    UnsupportedPredicate,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EligibilityReason {
    pub code: EligibilityReasonCode,
    pub detail: String,
    pub message_ko: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CardEligibility {
    Eligible,
    Ineligible(EligibilityReason),
}

impl CardEligibility {
    pub const fn is_eligible(&self) -> bool {
        matches!(self, Self::Eligible)
    }

    pub fn reason(&self) -> Option<&EligibilityReason> {
        match self {
            Self::Eligible => None,
            Self::Ineligible(reason) => Some(reason),
        }
    }
}

fn denied(code: EligibilityReasonCode, detail: &str, message: &str) -> CardEligibility {
    CardEligibility::Ineligible(EligibilityReason {
        code,
        detail: detail.to_owned(),
        message_ko: message.to_owned(),
    })
}

fn require(ok: bool, requirement: &str) -> CardEligibility {
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

fn enabled(flag: bool) -> Option<CardEligibility> {
    flag.then(|| {
        denied(
            EligibilityReasonCode::AlreadyEnabled,
            "runtime-enabled",
            "이미 적용 중인 효과입니다.",
        )
    })
}

/// Official aliases; callers should also canonicalize before granting the card.
pub fn canonical_card_id(id: &str) -> &str {
    match id {
        "free-move" => "premove",
        "chess-45-pow-30" => "chess-n-pow-30",
        _ => id,
    }
}

const MERCHANT_EXCLUSIONS: &[&str] = &[
    "king-of-the-hill",
    "iron-monarch",
    "horse-riding",
    "racing-king",
    "switcheroo",
    "imperial-studies",
    "underground-bunker",
    "last-resistance",
    "black-magic",
    "knightmate",
    "qxe1",
    "castling",
    "encouragement",
];

/// Also use this while drawing offers: acquired-card checks alone do not prevent
/// two incompatible cards from appearing together in a single offer batch.
pub fn mutually_exclusive_cards(first: &str, second: &str) -> bool {
    let a = canonical_card_id(first);
    let b = canonical_card_id(second);
    if a == b {
        return false;
    }
    [
        &["big-rook", "london-system", "horde"][..],
        &["pawn-conversion", "en-passant-bang"][..],
        &["democracy", "mongolian-gambit"][..],
        &["democracy", "black-magic"][..],
        &["democracy", "last-stand"][..],
    ]
    .iter()
    .any(|group| group.contains(&a) && group.contains(&b))
        || (a == "merchant-guild" && MERCHANT_EXCLUSIONS.contains(&b))
        || (b == "merchant-guild" && MERCHANT_EXCLUSIONS.contains(&a))
}

fn is_large(piece: Piece) -> bool {
    matches!(
        piece.kind,
        PieceKind::Colossus | PieceKind::BigRook | PieceKind::BigBishop
    )
}

// Do not use Engine::is_royal_piece: that helper additionally treats merchants
// as royal. JS isRoyalIdentityPiece does not; witch-trial excludes them separately.
fn royal(state: &GameState, piece: Piece) -> bool {
    piece.kind != PieceKind::Merchant
        && (matches!(
            piece.kind,
            PieceKind::King | PieceKind::RoyalKnight | PieceKind::DarkWizard
        ) || state.card_runtime.royal_piece_ids.contains(&piece.id))
}

fn queen(state: &GameState, piece: Piece) -> bool {
    // Core's Qxe1 heir is represented by royal_piece_ids, not regencyHeir.
    piece.kind == PieceKind::Queen && !state.card_runtime.royal_piece_ids.contains(&piece.id)
}

/// Side-effect-free normal acquisition check. It deliberately does not require
/// `state.turn == player`, a playing phase, a charge, or an immediate legal action:
/// setup and draft acquisition are not card activation. Existing zero-charge
/// cards still count as acquired. Setup also uses global uniqueness, preventing
/// duplicate grants to the other player. RULE identity comes from catalog
/// activation/phase, never from an ID suffix (e.g. mistake-card vs mistake).
pub fn check_card_eligibility(
    state: &GameState,
    card_id: &str,
    player: Player,
    context: AcquisitionContext,
) -> CardEligibility {
    let id = canonical_card_id(card_id);
    let Some(definition) = crate::ruleset_card(&state.ruleset_id, id) else {
        return denied(
            if state.ruleset_id == crate::LATEST_RULESET_ID
                && crate::latest_catalog().iter().any(|card| card.id == id)
            {
                EligibilityReasonCode::UnsupportedPredicate
            } else {
                EligibilityReasonCode::UnknownCard
            },
            id,
            "이 룰셋에서 지원하지 않거나 비활성화된 카드입니다.",
        );
    };
    let is_rule =
        definition.activation == CardActivation::Rule || definition.phase == CardPhase::Rule;
    if is_rule && context == AcquisitionContext::Draft {
        return denied(
            EligibilityReasonCode::RuleCardInDraft,
            id,
            "규칙 카드는 일반 드래프트에서 획득할 수 없습니다.",
        );
    }
    if state
        .cards
        .iter()
        .any(|card| canonical_card_id(&card.definition_id) == id)
    {
        return denied(
            EligibilityReasonCode::AlreadyAcquired,
            id,
            "이미 어느 한 플레이어가 획득한 카드입니다.",
        );
    }
    if is_rule {
        return if state
            .applied_rules
            .iter()
            .any(|rule| canonical_card_id(rule) == id)
        {
            denied(
                EligibilityReasonCode::RuleAlreadyApplied,
                id,
                "이미 적용된 규칙입니다.",
            )
        } else {
            CardEligibility::Eligible
        };
    }
    // draftPoolForCategories checks against both players' acquired cards; setup
    // isCardDrawableForColor only checks the recipient's acquired cards.
    if let Some(card) = state.cards.iter().find(|card| {
        (context == AcquisitionContext::Draft || card.owner == player)
            && mutually_exclusive_cards(id, &card.definition_id)
    }) {
        return denied(
            EligibilityReasonCode::MutuallyExclusive,
            &card.definition_id,
            "이미 획득한 카드와 동시에 획득할 수 없습니다.",
        );
    }
    let diagonal = state
        .applied_rules
        .iter()
        .any(|rule| rule == "diagonal-chess");
    if (state.card_runtime.monochrome_chess && matches!(id, "horse-riding" | "horde"))
        || (diagonal && matches!(id, "big-rook" | "london-system" | "horde"))
    {
        return denied(
            EligibilityReasonCode::BlockedByRule,
            id,
            "현재 규칙과 호환되지 않는 카드입니다.",
        );
    }
    if matches!(id, "big-rook" | "big-bishop") {
        if let Some(rule) = state.applied_rules.iter().find(|rule| {
            matches!(
                canonical_card_id(rule),
                "diagonal-chess" | "chess-960" | "chess-344200" | "chess-n-pow-30"
            )
        }) {
            return denied(
                EligibilityReasonCode::BlockedByRule,
                rule,
                "현재 시작 배치 규칙에서는 대형 기물을 획득할 수 없습니다.",
            );
        }
    }
    if context == AcquisitionContext::Draft
        && state.card_runtime.macho_chess
        && id == "reverse-pawns"
    {
        return denied(
            EligibilityReasonCode::BlockedByRule,
            "macho-chess",
            "마초 체스에서는 폰 방향 반전을 드래프트할 수 없습니다.",
        );
    }
    // These precede drawable checks in the official draft wrapper. Keep them
    // even where setup activation predicates are not reconstructable in core.
    if context == AcquisitionContext::Draft
        && matches!(
            id,
            "insight"
                | "joker"
                | "zugzwang"
                | "othello"
                | "homecoming"
                | "judgment"
                | "necromancy"
                | "canceling"
        )
    {
        return CardEligibility::Eligible;
    }
    if state.ruleset_id == crate::LATEST_RULESET_ID {
        return crate::engine::cards::eligibility(state, id, player, context);
    }
    compat::check(state, id, player, context)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Board, CardInstance, Mode, Square};

    fn state() -> GameState {
        GameState::standard("augment_chess_20260915", Mode::Standard, 7)
    }
    fn check(state: &GameState, id: &str, context: AcquisitionContext) -> CardEligibility {
        check_card_eligibility(state, id, Player::White, context)
    }
    fn code(result: CardEligibility) -> EligibilityReasonCode {
        result.reason().expect("expected rejection").code
    }
    fn acquire(state: &mut GameState, id: &str, owner: Player) {
        state.add_card(CardInstance {
            instance_id: state.cards.len() as u32 + 1,
            definition_id: id.to_owned(),
            owner,
            half_stars: 0,
            charges: 0,
            available_after_turn: None,
        });
    }
    fn remove_kind(state: &mut GameState, owner: Player, kind: PieceKind) {
        let squares: Vec<_> = state
            .board
            .iter()
            .filter(|(_, p)| p.owner == owner && p.kind == kind)
            .map(|(sq, _)| sq)
            .collect();
        for sq in squares {
            state.board.set(sq, None);
        }
    }

    #[test]
    fn duplicates_are_global_even_when_spent_and_future_target() {
        let mut s = state();
        acquire(&mut s, "judgment", Player::Black);
        for context in [AcquisitionContext::Draft, AcquisitionContext::Setup] {
            assert_eq!(
                code(check(&s, "judgment", context)),
                EligibilityReasonCode::AlreadyAcquired
            );
        }
    }

    #[test]
    fn rule_catalog_identity_and_setup_duplicate() {
        let mut s = state();
        assert_eq!(
            code(check(&s, "mistake", AcquisitionContext::Draft)),
            EligibilityReasonCode::RuleCardInDraft
        );
        assert!(check(&s, "mistake", AcquisitionContext::Setup).is_eligible());
        assert_ne!(
            check(&s, "mistake-card", AcquisitionContext::Draft)
                .reason()
                .unwrap()
                .code,
            EligibilityReasonCode::RuleCardInDraft
        );
        s.applied_rules.push("mistake".into());
        assert_eq!(
            code(check(&s, "mistake", AcquisitionContext::Setup)),
            EligibilityReasonCode::RuleAlreadyApplied
        );
    }

    #[test]
    fn amazon_requires_plain_queen_and_knight_not_transformed_family() {
        let mut s = state();
        assert!(check(&s, "amazon", AcquisitionContext::Draft).is_eligible());
        remove_kind(&mut s, Player::White, PieceKind::Knight);
        s.board.set(
            "b1".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Amazon).with_id(100)),
        );
        assert_eq!(
            code(check(&s, "amazon", AcquisitionContext::Setup)),
            EligibilityReasonCode::MissingBoardRequirement
        );
        s.board.set(
            "b1".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Knight).with_id(100)),
        );
        let queen_id = s.board.get("d1".parse().unwrap()).unwrap().id;
        s.card_runtime.royal_piece_ids.insert(queen_id);
        assert!(!check(&s, "amazon", AcquisitionContext::Draft).is_eligible());
    }

    #[test]
    fn breakthrough_requires_pawn_not_immediate_move() {
        let mut s = state();
        // Initial pawns can all be blocked without changing acquisition eligibility.
        for file in 0..8 {
            s.board.set(
                Square::new(file, 2).unwrap(),
                Some(Piece::new(Player::White, PieceKind::Rook).with_id(100 + file as u32)),
            );
        }
        assert!(check(&s, "breakthrough-order", AcquisitionContext::Draft).is_eligible());
        remove_kind(&mut s, Player::White, PieceKind::Pawn);
        assert!(!check(&s, "breakthrough-order", AcquisitionContext::Draft).is_eligible());
    }

    #[test]
    fn witch_trial_excludes_royal_merchant_vip_colossus_but_not_big_rook() {
        let mut s = GameState::from_board(
            "augment_chess_20260915",
            Mode::Standard,
            Board::empty(),
            Player::White,
            1,
        );
        for (index, kind) in [
            PieceKind::King,
            PieceKind::RoyalKnight,
            PieceKind::DarkWizard,
            PieceKind::Merchant,
            PieceKind::Vip,
            PieceKind::Colossus,
        ]
        .into_iter()
        .enumerate()
        {
            s.board.set(
                Square::from_index(index as u8).unwrap(),
                Some(Piece::new(Player::Black, kind).with_id(index as u32 + 1)),
            );
        }
        assert!(!check(&s, "witch-trial", AcquisitionContext::Draft).is_eligible());
        s.board.set(
            "h8".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::BigRook).with_id(99)),
        );
        assert!(check(&s, "witch-trial", AcquisitionContext::Draft).is_eligible());
        s.card_runtime.royal_piece_ids.insert(99);
        assert!(!check(&s, "witch-trial", AcquisitionContext::Draft).is_eligible());
    }

    #[test]
    fn official_exclusive_pairs_are_symmetric_and_not_overgeneralized() {
        for (a, b) in [
            ("big-rook", "horde"),
            ("london-system", "horde"),
            ("pawn-conversion", "en-passant-bang"),
            ("democracy", "mongolian-gambit"),
            ("democracy", "black-magic"),
            ("democracy", "last-stand"),
        ] {
            assert!(mutually_exclusive_cards(a, b));
            assert!(mutually_exclusive_cards(b, a));
        }
        for id in MERCHANT_EXCLUSIONS {
            assert!(mutually_exclusive_cards("merchant-guild", id));
        }
        assert!(!mutually_exclusive_cards("big-bishop", "horde"));
        assert!(!mutually_exclusive_cards("democracy", "queens-gambit")); // white-box-only rule
        assert!(!mutually_exclusive_cards("horde", "horde"));
        let mut s = state();
        acquire(&mut s, "pawn-conversion", Player::Black);
        assert_eq!(
            code(check(&s, "en-passant-bang", AcquisitionContext::Draft)),
            EligibilityReasonCode::MutuallyExclusive
        );
        assert!(check(&s, "en-passant-bang", AcquisitionContext::Setup).is_eligible());
    }

    #[test]
    fn all_future_target_drafts_and_alekhine_survive_empty_board() {
        let s = GameState::from_board(
            "augment_chess_20260915",
            Mode::Standard,
            Board::empty(),
            Player::White,
            1,
        );
        for id in [
            "insight",
            "joker",
            "zugzwang",
            "othello",
            "homecoming",
            "judgment",
            "necromancy",
            "canceling",
            "alekhine-machine-gun",
        ] {
            assert!(
                check(&s, id, AcquisitionContext::Draft).is_eligible(),
                "{id}"
            );
        }
        assert!(!check(&s, "alekhine-machine-gun", AcquisitionContext::Setup).is_eligible());
        assert_eq!(
            code(check(&s, "necromancy", AcquisitionContext::Setup)),
            EligibilityReasonCode::UnsupportedPredicate
        );
    }

    #[test]
    fn rules_and_runtime_flags_use_real_core_fields() {
        let mut s = state();
        s.card_runtime.monochrome_chess = true;
        assert_eq!(
            code(check(&s, "horse-riding", AcquisitionContext::Setup)),
            EligibilityReasonCode::BlockedByRule
        );
        s.applied_rules.push("chess-960".into());
        assert_eq!(
            code(check(&s, "big-bishop", AcquisitionContext::Draft)),
            EligibilityReasonCode::BlockedByRule
        );
        s.card_runtime.fianchetto[0] = true;
        assert_eq!(
            code(check(&s, "fianchetto", AcquisitionContext::Draft)),
            EligibilityReasonCode::AlreadyEnabled
        );
        s.card_runtime.exhaustion[1] = true;
        assert_eq!(
            code(check(&s, "exhaustion", AcquisitionContext::Draft)),
            EligibilityReasonCode::AlreadyEnabled
        );
    }

    #[test]
    fn unknown_and_unmapped_are_explicit_not_silent_acceptance() {
        let s = state();
        assert_eq!(
            code(check(&s, "not-a-card", AcquisitionContext::Setup)),
            EligibilityReasonCode::UnknownCard
        );
        let result = check(&s, "freeze", AcquisitionContext::Draft);
        assert_eq!(
            result.reason().unwrap().code,
            EligibilityReasonCode::UnsupportedPredicate
        );
        assert!(!result.reason().unwrap().message_ko.is_empty());
        assert_eq!(canonical_card_id("free-move"), "premove");
    }

    /// The 46 cards that used to deny every acquisition with UnsupportedPredicate now
    /// resolve their catalog `draw_requirement` against the board. On a standard
    /// opening position every one of those requirements holds, so drafting must be
    /// allowed instead of silently excluding more than a sixth of the card pool.
    #[test]
    fn every_formerly_denied_card_can_be_drafted_from_the_opening_position() {
        let state = GameState::standard(crate::LATEST_RULESET_ID, Mode::Standard, 7);
        let previously_denied = [
            "camouflage-color",
            "capture-the-flag",
            "cleanup",
            "clone",
            "cool-guy",
            "coronation",
            "desperado",
            "disassembly",
            "empty-lunchbox",
            "extinction",
            "fleeting-dream",
            "hallucination",
            "homecoming",
            "hypocrisy",
            "infiltration",
            "injury",
            "insight",
            "joker",
            "killer-king",
            "long-en-passant",
            "macho-chess",
            "majesty",
            "miracle",
            "missionary",
            "moving",
            "nullification",
            "octopus",
            "outpost",
            "overtake",
            "parrot",
            "parry",
            "pawn-storm",
            "platform",
            "portal-gun",
            "proficiency",
            "quantum-mechanics",
            "recycling",
            "replay",
            "saturation",
            "schrodinger-pawns",
            "stealth",
            "submerge",
            "substitution",
            "suspicious-potion",
            "transcendence",
            "white-box",
        ];
        let mut rejected = Vec::new();
        for id in previously_denied {
            let result =
                check_card_eligibility(&state, id, Player::White, AcquisitionContext::Draft);
            if let Some(reason) = result.reason() {
                // RULE cards are correctly kept out of the ordinary draft by an earlier
                // gate; that rejection is unrelated to the draw requirement.
                if reason.code == EligibilityReasonCode::RuleCardInDraft {
                    continue;
                }
                rejected.push(format!("{id}: {:?}", reason.code));
            }
        }
        assert!(
            rejected.is_empty(),
            "cards that must be draftable were rejected: {rejected:?}"
        );
        // The draw requirement itself must no longer be the reason for any of them.
        for id in previously_denied {
            let result =
                check_card_eligibility(&state, id, Player::White, AcquisitionContext::Setup);
            if let Some(reason) = result.reason() {
                assert_ne!(
                    reason.code,
                    EligibilityReasonCode::UnsupportedPredicate,
                    "{id} still denies acquisition on an unmapped predicate"
                );
            }
        }
    }

    /// Three requirements are hash-gated upstream, so which branch applies depends on
    /// this snapshot's catalog hash rather than on preference. Pin the outcome each
    /// time so a snapshot change re-opens the question instead of silently drifting.
    #[test]
    fn hash_gated_draw_requirements_take_the_branch_this_snapshot_selects() {
        let state = GameState::standard(crate::LATEST_RULESET_ID, Mode::Standard, 7);

        // missionary: usesThreeCardBuffs is false for this hash, so the spawn check does
        // not apply and a plain bishop is enough. The standard position has bishops.
        assert!(
            check(&state, "missionary", AcquisitionContext::Draft).is_eligible(),
            "missionary must be draftable while the plain-bishop branch applies"
        );

        // suspicious-potion: ANY_POTION_PIECE only excludes wall/football/black-hole.
        // A board of nothing but excluded kinds must be the only rejection.
        assert!(check(&state, "suspicious-potion", AcquisitionContext::Draft).is_eligible());
        let mut only_footballs = GameState::standard(crate::LATEST_RULESET_ID, Mode::Standard, 7);
        for (square, _) in only_footballs.board.iter().collect::<Vec<_>>() {
            only_footballs.board.set(square, None);
        }
        let football = crate::Piece::new(Player::White, PieceKind::Football).with_id(900);
        let square = "d4".parse().unwrap();
        only_footballs.board.set(square, Some(football));
        assert!(
            !check(
                &only_footballs,
                "suspicious-potion",
                AcquisitionContext::Draft
            )
            .is_eligible(),
            "a board of only excluded kinds must not satisfy the potion requirement"
        );

        // injury: OPPONENT_KNIGHT_FAMILY uses the explicit kind list for this hash, and
        // the standard position has an enemy knight, so it is draftable.
        assert!(check(&state, "injury", AcquisitionContext::Draft).is_eligible());
    }

    /// Every draftable active card must honor its catalog `draw_requirement`: either it
    /// gates on a real predicate, delegates to a helper that does, or its requirement is
    /// one upstream resolves to `default: true`.
    ///
    /// This is the audit that found four cards still ignoring a real requirement after the
    /// layer-1 work covered the original forty-six, so it stays as a guard.
    #[test]
    fn delegation_cards_never_refuse_on_an_unmapped_predicate() {
        // Cards that delegate eligibility to a shared helper which applies the predicate.
        const DELEGATES: &[&str] = &[
            "disarm",
            "barrier",
            "berserker",
            "don-quixote",
            "hedgehog",
            "idol",
            "killer",
            "offering",
            "revolving-door",
            "siren",
            "undead",
            "wanderer",
            "evasion",
            "fanatical-ritual",
            "en-passant-bang",
            "royal-command",
            "snipe",
            "switcheroo",
            "taunt",
        ];
        let state = GameState::standard(crate::LATEST_RULESET_ID, Mode::Standard, 7);
        let mut refused = Vec::new();
        for id in DELEGATES {
            if let Some(reason) =
                check_card_eligibility(&state, id, Player::White, AcquisitionContext::Draft)
                    .reason()
            {
                if reason.code == EligibilityReasonCode::UnsupportedPredicate {
                    refused.push(*id);
                }
            }
        }
        assert!(
            refused.is_empty(),
            "delegation cards still refusing on an unmapped predicate: {refused:?}"
        );
    }
}
