use super::*;
pub(super) const ID: &str = crate::GRUDGE;
pub(super) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    owner: Player,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let candidates: Vec<_> = web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, p)| {
            p.owner == owner
                && grudge_kind_allowed(p.kind)
                && !engine.is_royal_piece(state, *p)
                && !state.card_runtime.october.grudged_piece_ids.contains(&p.id)
        })
        .collect();
    if !candidates.is_empty() {
        let index = if candidates.len() == 1 {
            0
        } else {
            let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
            events.push(Event::RandomSample {
                purpose: "acquire.v19.grudge".into(),
                roll_53: roll,
            });
            index.unwrap()
        };
        state
            .card_runtime
            .october
            .grudged_piece_ids
            .insert(candidates[index].1.id);
    }
    Ok(())
}

pub(super) fn eligibility(
    state: &GameState,
    player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    let rt = &state.card_runtime;
    let has_own = |predicate: &dyn Fn(Piece) -> bool| {
        state
            .board
            .iter()
            .any(|(_, p)| p.owner == player && predicate(p))
    };
    require(
        has_own(&|p| {
            crate::engine::grudge_kind_allowed(p.kind)
                && !royal(state, p)
                && !rt.october.grudged_piece_ids.contains(&p.id)
        }),
        "원한을 부여할 비왕실 아군 비폰 기물이 필요합니다.",
    )
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

impl Engine {
    pub(in crate::engine) fn settle_grudge_captures(
        &self,
        state: &mut GameState,
        attacker: Piece,
        before: &crate::Board,
        rt: &CardRuntimeState,
        events: &[Event],
    ) {
        if state.ruleset_id != crate::LATEST_RULESET_ID
            || rt.october.grudged_piece_ids.is_empty()
            || state.board.find_by_id(attacker.id).is_none()
        {
            return;
        }
        let start = events
            .iter()
            .rposition(|e| matches!(e, Event::ActionApplied { .. }))
            .unwrap_or(0);
        for event in &events[start..] {
            let victim = match event {
                Event::PieceCaptured { actor, piece, .. } if *actor == attacker.owner => {
                    Some(*piece)
                }
                Event::PieceRemoved {
                    actor,
                    piece,
                    cause,
                    ..
                } if *actor == attacker.owner && cause == "colossus-sector" => Some(*piece),
                _ => None,
            };
            if let Some(victim) = victim
                && victim.owner != attacker.owner
                && rt.october.grudged_piece_ids.contains(&victim.id)
            {
                let mask = grudge_movement_mask(victim, before, rt, 0);
                if mask != 0 {
                    *state
                        .card_runtime
                        .october
                        .grudge_losses
                        .entry(attacker.id)
                        .or_default() |= mask;
                }
            }
        }
    }
}

pub(crate) fn grudge_kind_allowed(kind: PieceKind) -> bool {
    !matches!(
        kind,
        PieceKind::Pawn
            | PieceKind::King
            | PieceKind::RoyalKnight
            | PieceKind::DarkWizard
            | PieceKind::Merchant
            | PieceKind::Vip
            | PieceKind::Crown
            | PieceKind::Football
            | PieceKind::Monster
    )
}

pub(super) fn grudge_movement_mask(
    piece: Piece,
    board: &crate::Board,
    rt: &CardRuntimeState,
    depth: u8,
) -> u8 {
    if depth > 2 {
        return 0;
    }
    use PieceKind::*;
    let mut mask = match piece.kind {
        Rook | Herald | Cannon | BigRook | Campfire | SiegeRam | Slime | Hook | Brutus => 1,
        Bishop | BigBishop | Ferz | Alfil | Knightmaster | Princess | Missionary | Protestant
        | Cardinal => 2,
        Knight | RoyalKnight | Assassin | Unicorn | Paladin | Dragon => 4,
        Camel => 8,
        Alibaba => 3,
        Chancellor => 5,
        Archbishop => 6,
        Amazon | PrimeMinister => 7,
        Windmill => {
            if rt.windmill_modes.get(&piece.id) == Some(&WindmillMode::Rook) {
                1
            } else {
                2
            }
        }
        Trickster => {
            let mut p = piece;
            p.kind = rt
                .trickster_move_kinds
                .get(&piece.id)
                .copied()
                .unwrap_or(Rook);
            grudge_movement_mask(p, board, rt, depth + 1)
        }
        Parrot => rt.parrot_memory[piece.owner.index()].map_or(0, |memory| {
            let mut p = piece;
            p.kind = memory.effective;
            grudge_movement_mask(p, board, rt, depth + 1)
        }),
        BabyBear | Scarecrow | Football | Crown | Monster => 0,
        _ => 3,
    };
    if rt.reversal[piece.owner.index()] && matches!(piece.kind, Rook | Bishop) {
        mask = if piece.kind == Rook { 2 } else { 1 };
    }
    if piece.kind == Princess
        && board
            .iter()
            .any(|(_, p)| p.owner == piece.owner && p.kind == Queen)
    {
        mask = 3;
    }
    if (piece.kind == MagicGirl && rt.magic_girl_surge[piece.owner.index()])
        || (piece.kind == Berserker
            && board
                .iter()
                .filter(|(_, p)| p.owner == piece.owner && p.kind != Monster)
                .count()
                <= 5)
    {
        mask = 7;
    }
    mask & !rt
        .october
        .grudge_losses
        .get(&piece.id)
        .copied()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::super::tests::objectives::*;
    use super::*;
    #[test]
    fn grudge_acquisition_excludes_pawns_royals_and_vip_and_skips_singleton_rng() {
        let e = Engine::bundled_latest().unwrap();
        let s = position(&[
            ("b2", Player::White, PieceKind::Pawn),
            ("d4", Player::White, PieceKind::Rook),
            ("f1", Player::White, PieceKind::Vip),
        ]);
        let id = at(&s, "d4").id;
        let one = e.grant_card(&s, 1, crate::GRUDGE, Player::White).unwrap();
        assert_eq!(one.state.rng_state(), s.rng_state());
        assert_eq!(
            one.state.card_runtime.october.grudged_piece_ids,
            BTreeSet::from([id])
        );
        let mut multi = s.clone();
        multi.board.set(
            "c1".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Bishop).with_id(100)),
        );
        let a = e
            .grant_card(&multi, 1, crate::GRUDGE, Player::White)
            .unwrap();
        let b = e
            .grant_card(&multi, 1, crate::GRUDGE, Player::White)
            .unwrap();
        assert_eq!(a.state, b.state);
        assert_ne!(a.state.rng_state(), multi.rng_state());
        assert_eq!(
            a.events
                .iter()
                .filter(|v| matches!(v, Event::RandomSample { .. }))
                .count(),
            1
        );
        let mut royal = s.clone();
        royal.card_runtime.royal_piece_ids.insert(id);
        assert!(
            !crate::check_card_eligibility(
                &royal,
                crate::GRUDGE,
                Player::White,
                crate::AcquisitionContext::Setup
            )
            .is_eligible()
        );
    }
    #[test]
    fn grudge_capture_permanently_loses_victim_component_even_after_transform() {
        let e = Engine::bundled_latest().unwrap();
        let mut s = position(&[
            ("d4", Player::White, PieceKind::Queen),
            ("d6", Player::Black, PieceKind::Bishop),
        ]);
        s = e
            .grant_card(&s, 1, crate::GRUDGE, Player::Black)
            .unwrap()
            .state;
        let queen = at(&s, "d4");
        s = e.apply(&s, movement("d4", "d6")).unwrap().state;
        assert_eq!(s.card_runtime.october.grudge_losses[&queen.id], 2);
        s.turn = Player::White;
        assert!(has_move(&e, &s, "d6", "e6"));
        assert!(!has_move(&e, &s, "d6", "e7"));
        let mut p = at(&s, "d6");
        p.kind = PieceKind::Archbishop;
        s.board.set("d6".parse().unwrap(), Some(p));
        assert!(has_move(&e, &s, "d6", "f5"));
        assert!(!has_move(&e, &s, "d6", "e7"));
        let mut saved = CardRuntimeState::default();
        turn_effects::restore_piece_runtime(&mut saved, &s.card_runtime, &[queen.id]);
        assert_eq!(saved.october.grudge_losses[&queen.id], 2);
    }
    #[test]
    fn grudge_shield_contact_and_already_lost_victim_component_do_not_add_losses() {
        let e = Engine::bundled_latest().unwrap();
        let mut s = position(&[
            ("d4", Player::White, PieceKind::Queen),
            ("d6", Player::Black, PieceKind::Bishop),
        ]);
        s = e
            .grant_card(&s, 1, crate::GRUDGE, Player::Black)
            .unwrap()
            .state;
        let bishop = at(&s, "d6");
        s.card_runtime.shielded_piece_ids.insert(bishop.id);
        let contact = e.apply(&s, movement("d4", "d6")).unwrap().state;
        assert!(contact.card_runtime.october.grudge_losses.is_empty());
        s.card_runtime.shielded_piece_ids.clear();
        s.card_runtime.october.grudge_losses.insert(bishop.id, 2);
        let capture = e.apply(&s, movement("d4", "d6")).unwrap().state;
        assert!(capture.card_runtime.october.grudge_losses.is_empty());
    }
}
