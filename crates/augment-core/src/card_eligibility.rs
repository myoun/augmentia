//! Checked acquisition eligibility, not activation legality or passive installation.
//!
//! Source: September 15 official JS `isCardDraftDrawableForColor` (54556),
//! `isCardDrawableForColor` (54562), and their helpers. Draft pool category,
//! milestone, AI-specific exclusions, and compatibility *between offers* belong
//! to the caller. Core has no cardBanIds/completeRandom/editor presentation-only
//! rule fields: callers must enforce external bans; global uniqueness here is
//! the normal (not completeRandom) policy. Unsupported predicates fail closed.

use crate::{CardActivation, CardPhase, GameState, Piece, PieceKind, Player, implemented_card};

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
    let Some(definition) = implemented_card(id) else {
        return denied(
            EligibilityReasonCode::UnknownCard,
            id,
            "코어에서 구현되지 않은 카드입니다.",
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
    let rt = &state.card_runtime;
    let i = player.index();
    let count = |owner, kind| {
        state
            .board
            .iter()
            .filter(|(_, p)| p.owner == owner && p.kind == kind)
            .count()
    };
    let own = |kind| count(player, kind);
    let has_own = |predicate: &dyn Fn(Piece) -> bool| {
        state
            .board
            .iter()
            .any(|(_, p)| p.owner == player && predicate(p))
    };
    let has_enemy = |predicate: &dyn Fn(Piece) -> bool| {
        state
            .board
            .iter()
            .any(|(_, p)| p.owner == player.opponent() && predicate(p))
    };
    let own_queen = has_own(&|p| queen(state, p));
    let own_royal = has_own(&|p| royal(state, p));
    let status = match id {
        "king-of-the-hill" => rt.hill_king[i],
        "democracy" => rt.democracy[i],
        "fianchetto" => rt.fianchetto[i],
        "pawn-conversion" => rt.pawn_conversion[i],
        "rook-lift" => rt.rook_lift[i],
        "corner-kick" => rt.corner_kick[i],
        "religious-victory" => rt.religious_victory[i],
        "bina-mate" => rt.bina_mate[i],
        "overwhelm" => rt.overwhelm[i],
        "exhaustion" => rt.exhaustion[player.opponent().index()],
        "symmetry" => rt.symmetry[i],
        "mutation" => rt.mutation[i],
        "frontline-response" => rt.frontline_response[i],
        "relay" => rt.relay[i],
        "field-promotion" => rt.field_promotion[i],
        "geneva-convention" => rt.geneva_convention[i],
        "prophecy" => rt.prophecy[i].is_some(),
        _ => false,
    };
    if let Some(result) = enabled(status) {
        return result;
    }
    use PieceKind::*;
    match id {
        "frenzy" | "last-stand" | "early-promotion" | "fast-growth" | "otherworld" | "sprint"
        | "leap" | "retreat" | "underpromotion" | "final-weapon" | "en-passant-bang"
        | "pawn-conversion" | "ghost" | "breakthrough-order" | "death-squad" | "log"
        | "standard-bearer" | "field-promotion" => require(own(Pawn) > 0, "아군 폰이 필요합니다."),
        "vip" => require(
            own(Pawn) + count(player.opponent(), Pawn) > 0,
            "보드에 폰이 필요합니다.",
        ),
        "democracy" => require(
            own(Pawn) > 0
                || rt
                    .pending_recurrences
                    .iter()
                    .any(|entry| entry.piece.owner == player && entry.piece.kind == Pawn),
            "민주주의를 지탱할 아군 폰이 필요합니다.",
        ),
        "reverse-pawns" | "spy" => {
            require(count(player.opponent(), Pawn) > 0, "상대 폰이 필요합니다.")
        }
        "constitutional-monarchy"
        | "jester"
        | "wizard"
        | "reaper"
        | "idol"
        | "queen-afterimage"
        | "local-conscription"
        | "siren"
        | "undead"
        | "thief" => require(own_queen, "왕위 계승자가 아닌 아군 퀸이 필요합니다."),
        "amazon" => require(
            own_queen && own(Knight) >= 1,
            "아군 퀸과 나이트가 필요합니다.",
        ),
        "hook" => require(own_queen && own(Rook) > 0, "아군 퀸과 룩이 필요합니다."),
        "icbm" => require(
            own_queen && has_enemy(&|p| queen(state, p)),
            "아군과 상대 모두 퀸이 필요합니다.",
        ),
        "baby-bear" => require(
            has_own(&|p| queen(state, p) && !royal(state, p)),
            "비왕실 아군 퀸이 필요합니다.",
        ),
        "alekhine-machine-gun" => require(
            context == AcquisitionContext::Draft || (own_queen && own(Rook) >= 2),
            "아군 퀸과 룩 2개가 필요합니다.",
        ),
        "reformation" | "fianchetto" | "snipe" => {
            require(own(Bishop) > 0, "아군 비숍이 필요합니다.")
        }
        "ordination" | "religious-victory" => {
            require(own(Bishop) >= 2, "아군 비숍 2개가 필요합니다.")
        }
        "windmill" => require(
            own(Bishop) > 0 && own(Rook) > 0,
            "아군 비숍과 룩이 필요합니다.",
        ),
        "martyrdom" => require(
            own(Bishop) > 0 && own(Pawn) > 0,
            "아군 비숍과 폰이 필요합니다.",
        ),
        "dutch" => require(
            own(Rook) + own(Bishop) + own(Knight) > 0,
            "아군 룩, 비숍 또는 나이트가 필요합니다.",
        ),
        "radical-charge" | "backward-knight" | "file-surge" | "eagle" | "corner-kick"
        | "conversion" | "assassin" | "knightmaster" | "bribe" => {
            require(own(Knight) > 0, "아군 나이트가 필요합니다.")
        }
        "grasshopper" | "eastern-policy" => require(
            own(Knight) + own(Bishop) + own(Camel) > 0,
            "아군 나이트, 비숍 또는 낙타가 필요합니다.",
        ),
        "pegasus" | "dragon" | "herald" | "rook-lift" | "siege-ram" | "magic-girl"
        | "berserker" | "slime" | "trickster" => require(own(Rook) > 0, "아군 룩이 필요합니다."),
        "big-rook" | "big-bishop" => require(own(King) > 0, "아군 킹이 필요합니다."),
        "horse-riding" | "knightmate" | "horde" | "encouragement" | "racing-king"
        | "iron-monarch" | "imperial-studies" | "last-resistance" | "black-magic" | "prophecy" => {
            require(own_royal, "효과를 받을 아군 왕실 기물이 필요합니다.")
        }
        "palace" => require(
            has_enemy(&|p| royal(state, p)),
            "상대 왕실 기물이 필요합니다.",
        ),
        "underground-bunker" => require(
            has_own(&|p| royal(state, p) && !rt.bunkers.contains_key(&p.id)),
            "벙커가 없는 아군 왕실 기물이 필요합니다.",
        ),
        "witch-trial" => require(
            has_enemy(&|p| !royal(state, p) && !matches!(p.kind, Colossus | Merchant | Vip)),
            "킹, 상인, 귀빈, 거상이 아닌 상대 기물이 필요합니다.",
        ),
        "fanatical-ritual" => require(
            has_enemy(&|p| !royal(state, p) && !matches!(p.kind, Colossus | Merchant)),
            "왕실 기물, 상인, 거상이 아닌 상대 기물이 필요합니다.",
        ),
        "disarm" => require(
            has_enemy(&|p| !royal(state, p) && p.kind != Colossus),
            "비왕실 상대 기물이 필요합니다.",
        ),
        "royal-command" => require(
            has_own(&|p| !royal(state, p) && !matches!(p.kind, Colossus | Football)),
            "왕실 기물, 거상, 축구공이 아닌 아군 기물이 필요합니다.",
        ),
        "reposition" => require(
            has_own(&|p| p.kind != Football),
            "재배치할 아군 기물이 필요합니다.",
        ),
        "taunt" => require(
            has_enemy(&|p| p.kind != Football),
            "도발할 상대 기물이 필요합니다.",
        ),
        "traitor" => require(
            has_enemy(&|p| !matches!(p.kind, Merchant | Football)),
            "변절시킬 상대 기물이 필요합니다.",
        ),
        "royal-shield" => require(
            has_own(&|p| p.kind != Scarecrow && !rt.shielded_piece_ids.contains(&p.id)),
            "아직 보호막이 없는 아군 기물이 필요합니다.",
        ),
        "holdout" => require(
            has_own(&|p| p.kind == Pawn && !rt.holdout_promotions.contains_key(&p.id)),
            "존버 효과가 없는 아군 폰이 필요합니다.",
        ),
        "poisoned-pawn" => require(
            has_own(&|p| p.kind == Pawn && !rt.poisoned_piece_ids.contains(&p.id)),
            "독이 부여되지 않은 아군 폰이 필요합니다.",
        ),
        "trojan-horse" => require(
            has_own(&|p| p.kind == Knight && !rt.trojan_piece_ids.contains(&p.id)),
            "트로이 목마 효과가 없는 아군 나이트가 필요합니다.",
        ),
        "brutus" => require(
            has_own(&|p| p.kind == Rook && !royal(state, p)),
            "비왕실 아군 룩이 필요합니다.",
        ),
        "paladin" => require(
            has_own(&|p| p.kind == Knight && !royal(state, p)),
            "비왕실 아군 나이트가 필요합니다.",
        ),
        "clockwork" => require(
            has_own(&|p| matches!(p.kind, Knight | Bishop | Camel) && !royal(state, p)),
            "비왕실 아군 마이너 기물이 필요합니다.",
        ),
        "chimera" => require(
            has_own(&|p| {
                matches!(p.kind, Knight | Bishop) && !rt.chimera_piece_ids.contains(&p.id)
            }),
            "키메라 효과가 없는 아군 나이트 또는 비숍이 필요합니다.",
        ),
        "chameleon-mutation" => require(
            has_own(&|p| !royal(state, p) && p.kind != Merchant && !is_large(p)),
            "변이 가능한 비왕실 아군 기물이 필요합니다.",
        ),
        "mongolian-gambit" => require(
            has_own(&|p| !royal(state, p) && p.kind != Merchant),
            "상인이 아닌 비왕실 아군 기물이 필요합니다.",
        ),
        "vortex" => require(
            state
                .board
                .iter()
                .filter(|(_, p)| {
                    p.owner == player.opponent()
                        && !royal(state, *p)
                        && p.kind != Pawn
                        && !is_large(*p)
                })
                .count()
                >= 2,
            "뒤섞을 상대 비왕실 비폰 기물 2개가 필요합니다.",
        ),
        "suicide-bomber" => require(
            has_own(&|p| {
                matches!(p.kind, Pawn | Fanatic)
                    && !rt
                        .feudal_contracts
                        .iter()
                        .any(|contract| contract.pawn_id == p.id)
            }),
            "봉건 계약이 없는 아군 폰 또는 광신도가 필요합니다.",
        ),
        "feudal-contract" => require(
            has_own(&|p| {
                matches!(p.kind, Pawn | Fanatic)
                    && !rt.explosive_piece_ids.contains(&p.id)
                    && !rt
                        .feudal_contracts
                        .iter()
                        .any(|contract| contract.pawn_id == p.id)
            }) && has_own(&|p| !matches!(p.kind, Pawn | Fanatic) && !is_large(p)),
            "계약 가능한 아군 폰과 수호 기물이 필요합니다.",
        ),
        "merchant-guild" => require(
            state.board.iter().any(|(square, p)| {
                p.owner == player
                    && p.kind == King
                    && square
                        .offset(0, if player == Player::White { 2 } else { -2 })
                        .is_some_and(|to| {
                            state.board.get(to).is_none_or(|occupant| {
                                occupant.owner == player && !is_large(occupant)
                            })
                        })
            }),
            "킹 두 칸 앞에 상인을 배치할 수 있어야 합니다.",
        ),
        "exhaustion" | "bina-mate" | "overwhelm" | "king-of-the-hill" | "armistice" | "dice"
        | "initiative" | "symmetry" | "mutation" | "frontline-response" | "relay"
        | "geneva-convention" => CardEligibility::Eligible,
        // No broad `_ => true`: target metadata and some JS status fields do not
        // exist in ImplementedCardDefinition/GameState. A false positive here
        // would silently advertise compliance for those missing predicates.
        _ => denied(
            EligibilityReasonCode::UnsupportedPredicate,
            id,
            "공식 획득 조건의 런타임 매핑이 아직 검증되지 않아 획득을 허용하지 않습니다.",
        ),
    }
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
}
