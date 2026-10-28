//! Acquisition predicates for historical rulesets.
use super::*;
pub(super) fn check(
    state: &GameState,
    id: &str,
    player: Player,
    context: AcquisitionContext,
) -> CardEligibility {
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
        crate::ASSEMBLY => rt.october.assembly[i],
        crate::HOPE => rt.october.hope[i],
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
        "prophecy" => {
            rt.prophecy[i].is_some()
                || (state.ruleset_id == crate::LATEST_RULESET_ID
                    && rt.october.prophecy_remaining[i].is_some())
        }
        _ => false,
    };
    if let Some(result) = enabled(status) {
        return result;
    }
    use PieceKind::*;

    match id {
        crate::MAN => require(
            context == AcquisitionContext::Draft
                || state.board.iter().any(|(_, piece)| {
                    crate::latest::latest_minor_kind(piece.kind) && !royal(state, piece)
                }),
            "변환 가능한 비왕실 마이너 기물이 필요합니다.",
        ),
        crate::ASSEMBLY | crate::HOPE | crate::IQP | "highlander" => CardEligibility::Eligible,
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
