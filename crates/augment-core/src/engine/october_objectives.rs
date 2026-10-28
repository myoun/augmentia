//! Scoped v19 movement losses, spatial pawn bonds and check objectives.
use super::*;
use crate::state::PentaCheckProgress;

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

fn distance(a: Square, b: Square) -> u8 {
    a.file().abs_diff(b.file()).max(a.rank().abs_diff(b.rank()))
}

/// Captured spatialPawnChainLinks: degree-two minimum-cost forest, deterministic ties.
fn pawn_links(mut pawns: Vec<(Square, u32)>, horde: Option<Player>) -> Vec<[u32; 2]> {
    pawns.sort_by_key(|(s, id)| (s.file(), 7 - s.rank(), *id));
    if let Some(owner) = horde {
        let mut rows = std::collections::BTreeMap::<u8, Vec<(Square, u32)>>::new();
        for p in pawns {
            rows.entry(p.0.rank()).or_default().push(p);
        }
        let ranks: Vec<_> = if owner == Player::White {
            rows.into_values().collect()
        } else {
            rows.into_values().rev().collect()
        };
        let mut path: Vec<(Square, u32)> = Vec::new();
        for mut rank in ranks {
            if let Some(last) = path.last()
                && distance(last.0, rank.last().unwrap().0) < distance(last.0, rank[0].0)
            {
                rank.reverse();
            }
            path.extend(rank);
        }
        return path
            .windows(2)
            .filter(|p| distance(p[0].0, p[1].0) <= 2)
            .map(|p| [p[0].1, p[1].1])
            .collect();
    }
    let n = pawns.len();
    let mut parent: Vec<_> = (0..n).collect();
    let mut neighbors = vec![Vec::<usize>::new(); n];
    let mut edges = Vec::new();
    for a in 0..n {
        for b in a + 1..n {
            let dr = pawns[a].0.rank().abs_diff(pawns[b].0.rank()) as u16;
            let dc = pawns[a].0.file().abs_diff(pawns[b].0.file()) as u16;
            if dr.max(dc) <= 2 {
                edges.push(((dr * dr + dc * dc) * 2 + u16::from(dr != 0), a, b));
            }
        }
    }
    edges.sort_unstable();
    fn root(parent: &[usize], mut i: usize) -> usize {
        while parent[i] != i {
            i = parent[i];
        }
        i
    }
    for (_, a, b) in edges {
        if neighbors[a].len() == 2 || neighbors[b].len() == 2 {
            continue;
        }
        let ra = root(&parent, a);
        let rb = root(&parent, b);
        if ra != rb {
            parent[ra] = rb;
            neighbors[a].push(b);
            neighbors[b].push(a);
        }
    }
    let mut seen = BTreeSet::new();
    let mut result = Vec::new();
    for start in 0..n {
        if neighbors[start].len() != 1 || seen.contains(&start) {
            continue;
        }
        let mut previous = None;
        let mut current = start;
        loop {
            seen.insert(current);
            let next = neighbors[current]
                .iter()
                .find(|&&i| Some(i) != previous)
                .copied();
            let Some(next) = next else {
                break;
            };
            result.push([pawns[current].1, pawns[next].1]);
            previous = Some(current);
            current = next;
        }
    }
    result
}

fn grudge_movement_mask(
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

impl Engine {
    pub(super) fn acquire_october_objective(
        &self,
        state: &mut GameState,
        id: &str,
        owner: Player,
        events: &mut Vec<Event>,
    ) {
        match id {
            crate::GRUDGE => {
                let candidates: Vec<_> = web_ordered_pieces(state)
                    .into_iter()
                    .filter(|(_, p)| {
                        p.owner == owner
                            && grudge_kind_allowed(p.kind)
                            && !self.is_royal_piece(state, *p)
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
            }
            crate::PAWN_CHAIN => {
                let enemy = owner.opponent();
                let pawns = web_ordered_pieces(state)
                    .into_iter()
                    .filter(|(_, p)| p.owner == enemy && p.kind == PieceKind::Pawn)
                    .map(|(s, p)| (s, p.id))
                    .collect();
                let horde = state
                    .cards
                    .iter()
                    .any(|c| c.owner == enemy && c.definition_id == HORDE);
                let links = pawn_links(pawns, horde.then_some(enemy));
                state.card_runtime.october.pawn_chain_links.extend(links);
            }
            crate::PENTA_CHECK => {
                state.card_runtime.october.penta_check[owner.index()] = Some(PentaCheckProgress {
                    count: 0,
                    opponent_checked: self.october_royal_checked(state, owner),
                })
            }
            _ => unreachable!(),
        }
    }

    pub(super) fn october_royal_checked(&self, state: &GameState, attacker: Player) -> bool {
        // v6OpponentKingAttacked uses the first royal in web board order. Keep
        // its actual identity and immunities; disable castling in attack probes.
        let Some((target, royal)) = web_ordered_pieces(state)
            .into_iter()
            .find(|(_, p)| p.owner == attacker.opponent() && self.is_royal_piece(state, *p))
        else {
            return false;
        };
        let mut probe = state.clone();
        probe.card_runtime.moved_piece_ids.extend(
            state
                .board
                .iter()
                .filter(|(_, p)| self.is_royal_piece(state, *p))
                .map(|(_, p)| p.id),
        );
        for (from, piece) in probe.board.iter().filter(|(_, p)| p.owner == attacker) {
            let mut actions = Vec::new();
            self.generate_moves_for_piece(&probe, from, piece, &mut actions);
            if actions.into_iter().any(|a| {
                if let Action::Move(m) = a {
                    m.to == target
                        || (matches!(piece.kind, PieceKind::Checker | PieceKind::CheckerKing)
                            && self
                                .checker_jump_capture(&probe, from, m.to, piece)
                                .is_some_and(|(_, p)| p.id == royal.id))
                        || (probe.card_runtime.large_pieces.contains_key(&piece.id)
                            && self
                                .large_landing_targets(&probe, piece, m.to)
                                .iter()
                                .any(|(_, p)| p.id == royal.id))
                } else {
                    false
                }
            }) {
                return true;
            }
        }
        false
    }

    pub(super) fn cancel_october_prophecies(&self, state: &mut GameState) {
        if state.ruleset_id == crate::LATEST_RULESET_ID {
            state.card_runtime.october.prophecy_remaining = [None; 2];
        }
    }

    pub(super) fn settle_october_checks(&self, state: &mut GameState) {
        if state.ruleset_id != crate::LATEST_RULESET_ID
            || (state.card_runtime.october.penta_check == [None; 2]
                && state.card_runtime.october.prophecy_remaining == [None; 2])
        {
            return;
        }
        let checked = [
            self.october_royal_checked(state, Player::White),
            self.october_royal_checked(state, Player::Black),
        ];
        if checked.contains(&true) {
            self.cancel_october_prophecies(state);
        }
        for owner in [Player::White, Player::Black] {
            if let Some(progress) = &mut state.card_runtime.october.penta_check[owner.index()] {
                if checked[owner.index()] && !progress.opponent_checked {
                    progress.count = (progress.count + 1).min(5);
                }
                progress.opponent_checked = checked[owner.index()];
            }
        }
    }

    pub(super) fn complete_october_objective_turn(&self, state: &mut GameState, _incoming: Player) {
        if state.ruleset_id != crate::LATEST_RULESET_ID || state.status != GameStatus::Ongoing {
            return;
        }
        self.settle_october_checks(state);
        for owner in [Player::White, Player::Black] {
            if let Some(remaining) =
                &mut state.card_runtime.october.prophecy_remaining[owner.index()]
            {
                *remaining = remaining.saturating_sub(1);
                if *remaining == 0 {
                    state.card_runtime.october.prophecy_remaining[owner.index()] = None;
                    state.status = GameStatus::Won {
                        winner: owner,
                        reason: WinReason::CardEffect(crate::PROPHECY.into()),
                    };
                    return;
                }
            }
        }
    }

    pub(super) fn apply_october_grudge_captures(
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

    pub(super) fn filter_october_state_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        if state.ruleset_id != crate::LATEST_RULESET_ID {
            return;
        }
        let rt = &state.card_runtime.october;
        if rt.grudge_losses.is_empty() && rt.pawn_chain_links.is_empty() {
            return;
        }
        actions.retain(|a| {
            let Action::Move(m) = a else {
                return true;
            };
            let ally = state
                .board
                .get(m.to)
                .filter(|p| p.owner == piece.owner && p.id != piece.id);
            let swapping = ally.is_some()
                && (self.piece_has_ability(state, piece, PieceKind::Dragon)
                    || state.card_runtime.relay[piece.owner.index()]
                    || state.card_runtime.substitution[piece.owner.index()]
                    || (self.is_royal_piece(state, piece)
                        && state.card_runtime.switcheroo[piece.owner.index()]));
            let castling = self.standard_castling_rook_plan(state, piece, *m);
            let df = from.file().abs_diff(m.to.file());
            let dr = from.rank().abs_diff(m.to.rank());
            let stationary = from == m.to;
            let component = match self.piece_ability_kind(state, piece) {
                PieceKind::Hook | PieceKind::Brutus => 1,
                PieceKind::Cardinal | PieceKind::Protestant => 2,
                _ => {
                    if df == 0 || dr == 0 {
                        1
                    } else if df == dr {
                        2
                    } else if u16::from(df) * u16::from(dr) == 2 {
                        4
                    } else if u16::from(df) * u16::from(dr) == 3 {
                        8
                    } else {
                        0
                    }
                }
            };
            if !swapping
                && !stationary
                && rt
                    .grudge_losses
                    .get(&piece.id)
                    .is_some_and(|loss| loss & component != 0)
            {
                return false;
            }
            for pair in &rt.pawn_chain_links {
                let Some((a, _)) = state.board.find_by_id(pair[0]) else {
                    continue;
                };
                let Some((b, _)) = state.board.find_by_id(pair[1]) else {
                    continue;
                };
                if distance(a, b) > 2 {
                    continue;
                }
                let destination = |id, old| {
                    if id == piece.id {
                        m.to
                    } else if swapping && ally.is_some_and(|p| p.id == id) {
                        from
                    } else if let Some((_, to, rook)) = castling
                        && rook.id == id
                    {
                        to
                    } else {
                        old
                    }
                };
                if distance(destination(pair[0], a), destination(pair[1], b)) > 2 {
                    return false;
                }
            }
            true
        });
    }

    pub(super) fn clean_october_chains(&self, state: &mut GameState, before: &crate::Board) {
        if state.ruleset_id != crate::LATEST_RULESET_ID {
            return;
        }
        state.card_runtime.october.pawn_chain_links.retain(|pair| {
            let Some((a, _)) = state.board.find_by_id(pair[0]) else {
                return false;
            };
            let Some((b, _)) = state.board.find_by_id(pair[1]) else {
                return false;
            };
            let unchanged = before.find_by_id(pair[0]).is_some_and(|(old, _)| old == a)
                && before.find_by_id(pair[1]).is_some_and(|(old, _)| old == b);
            unchanged || distance(a, b) <= 2
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn objective_geometry_matches_pinned_upstream_fixtures() {
        let e = Engine::bundled_latest().unwrap();
        let data: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/october-objectives.json"))
                .unwrap();
        for case in data["masks"].as_array().unwrap() {
            let kind: PieceKind = serde_json::from_value(case["kind"].clone()).unwrap();
            let mut s = position(&[("d4", Player::White, kind)]);
            let piece = at(&s, "d4");
            s.card_runtime.reversal[0] = case["flags"]["reversal"].as_bool().unwrap_or(false);
            s.card_runtime.magic_girl_surge[0] = case["flags"]["surge"].as_bool().unwrap_or(false);
            if case["flags"]["queen"] == true {
                s.board.set(
                    "e1".parse().unwrap(),
                    Some(Piece::new(Player::White, PieceKind::Queen).with_id(100)),
                );
            }
            if case["flags"]["army"] == true {
                for i in 0..6 {
                    s.board.set(
                        Square::new(i, 1).unwrap(),
                        Some(Piece::new(Player::White, PieceKind::Pawn).with_id(100 + i as u32)),
                    );
                }
            }
            if case["learned"] == "knight" {
                s.card_runtime
                    .trickster_move_kinds
                    .insert(piece.id, PieceKind::Knight);
            }
            if let Some(loss) = case["loss"].as_u64() {
                s.card_runtime
                    .october
                    .grudge_losses
                    .insert(piece.id, loss as u8);
            }
            assert_eq!(
                grudge_movement_mask(piece, &s.board, &s.card_runtime, 0),
                case["mask"].as_u64().unwrap() as u8,
                "{case}"
            );
        }
        for case in data["restrictions"].as_array().unwrap() {
            let kind: PieceKind = serde_json::from_value(case["kind"].clone()).unwrap();
            let mut s = position(&[("d4", Player::White, kind)]);
            let piece = at(&s, "d4");
            s.card_runtime
                .october
                .grudge_losses
                .insert(piece.id, case["loss"].as_u64().unwrap() as u8);
            let mut actions = (0..64)
                .map(|i| {
                    Action::Move(MoveAction::new(
                        "d4".parse().unwrap(),
                        Square::from_index(i).unwrap(),
                        None,
                    ))
                })
                .collect();
            e.filter_october_state_moves(&s, "d4".parse().unwrap(), piece, &mut actions);
            let actual: BTreeSet<_> = actions
                .into_iter()
                .filter_map(|a| {
                    if let Action::Move(m) = a {
                        Some(m.to.to_string())
                    } else {
                        None
                    }
                })
                .collect();
            let expected: BTreeSet<_> = case["expected"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| s.as_str().unwrap().to_owned())
                .collect();
            assert_eq!(actual, expected, "{case}");
        }
        for case in data["graphs"].as_array().unwrap() {
            let pawns = case["squares"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(i, s)| (s.as_str().unwrap().parse().unwrap(), i as u32 + 1))
                .collect();
            let horde = case["horde"].as_str().map(|s| {
                if s == "white" {
                    Player::White
                } else {
                    Player::Black
                }
            });
            let expected: Vec<[u32; 2]> = serde_json::from_value(case["expected"].clone()).unwrap();
            assert_eq!(pawn_links(pawns, horde), expected, "{case}");
        }
    }

    #[test]
    fn swaps_bypass_grudge_geometry_but_check_both_pawn_chain_destinations() {
        let e = Engine::bundled_latest().unwrap();
        let mut s = position(&[
            ("a2", Player::White, PieceKind::Dragon),
            ("c2", Player::White, PieceKind::Pawn),
            ("d4", Player::White, PieceKind::Pawn),
        ]);
        let dragon = at(&s, "a2");
        s.card_runtime.october.grudge_losses.insert(dragon.id, 4);
        s.card_runtime
            .october
            .pawn_chain_links
            .push([at(&s, "c2").id, at(&s, "d4").id]);
        assert!(!has_move(&e, &s, "a2", "b4"));
        assert!(!has_move(&e, &s, "a2", "c2"));
        assert!(has_move(&e, &s, "a2", "d4"));
    }
    fn position(extra: &[(&str, Player, PieceKind)]) -> GameState {
        let mut board = crate::Board::empty();
        for (s, o, k) in [
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
        ]
        .iter()
        .chain(extra)
        {
            board.set(s.parse().unwrap(), Some(Piece::new(*o, *k)));
        }
        let mut s = GameState::from_board(
            crate::LATEST_RULESET_ID,
            Mode::Standard,
            board,
            Player::White,
            17,
        );
        s.completed_drafts = [true; 3];
        s
    }
    fn at(s: &GameState, q: &str) -> Piece {
        s.board.get(q.parse().unwrap()).unwrap()
    }
    fn movement(a: &str, b: &str) -> Action {
        Action::Move(MoveAction::new(
            a.parse().unwrap(),
            b.parse().unwrap(),
            None,
        ))
    }
    fn has_move(e: &Engine, s: &GameState, a: &str, b: &str) -> bool {
        e.legal_actions(s).unwrap().contains(&movement(a, b))
    }
    fn arm(e: &Engine, s: &mut GameState) {
        *s = e
            .grant_card(s, 1, crate::PROPHECY, Player::White)
            .unwrap()
            .state;
        *s = e
            .apply(
                s,
                Action::UseCard {
                    instance_id: 1,
                    target: ActionTarget::None,
                },
            )
            .unwrap()
            .state;
    }

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
        end::restore_piece_runtime(&mut saved, &s.card_runtime, &[queen.id]);
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

    #[test]
    fn pawn_chain_initial_rank_links_and_two_square_leash_follow_identity() {
        let e = Engine::bundled_latest().unwrap();
        let mut s = position(&[]);
        for (i, file) in "abcdefgh".chars().enumerate() {
            s.board.set(
                format!("{file}7").parse().unwrap(),
                Some(Piece::new(Player::Black, PieceKind::Pawn).with_id(100 + i as u32)),
            );
        }
        s = e
            .grant_card(&s, 1, crate::PAWN_CHAIN, Player::White)
            .unwrap()
            .state;
        assert_eq!(s.card_runtime.october.pawn_chain_links.len(), 7);
        s.turn = Player::Black;
        assert!(has_move(&e, &s, "b7", "b5"));
        s = e.apply(&s, movement("b7", "b5")).unwrap().state;
        s.turn = Player::Black;
        assert!(!has_move(&e, &s, "b5", "b4"));
        assert_eq!(s.card_runtime.october.pawn_chain_links.len(), 7);
        let mut promoted = at(&s, "b5");
        promoted.kind = PieceKind::Rook;
        s.board.set("b5".parse().unwrap(), Some(promoted));
        assert!(!has_move(&e, &s, "b5", "b3"));
    }

    #[test]
    fn pawn_chain_breaks_on_capture_and_displacement_but_keeps_untouched_far_links() {
        let e = Engine::bundled_latest().unwrap();
        let mut s = position(&[
            ("b1", Player::White, PieceKind::Rook),
            ("b3", Player::Black, PieceKind::Pawn),
            ("c3", Player::Black, PieceKind::Pawn),
        ]);
        s = e
            .grant_card(&s, 1, crate::PAWN_CHAIN, Player::White)
            .unwrap()
            .state;
        assert_eq!(s.card_runtime.october.pawn_chain_links.len(), 1);
        let after = e.apply(&s, movement("b1", "b3")).unwrap().state;
        assert!(after.card_runtime.october.pawn_chain_links.is_empty());
        let before = s.board.clone();
        let p = at(&s, "b3");
        s.board.set("b3".parse().unwrap(), None);
        s.board.set("b7".parse().unwrap(), Some(p));
        e.clean_october_chains(&mut s, &before);
        assert!(s.card_runtime.october.pawn_chain_links.is_empty());
        s.card_runtime
            .october
            .pawn_chain_links
            .push([p.id, at(&s, "c3").id]);
        let before = s.board.clone();
        e.clean_october_chains(&mut s, &before);
        assert_eq!(s.card_runtime.october.pawn_chain_links.len(), 1);
    }

    #[test]
    fn penta_check_counts_five_new_checks_and_waits_until_owner_turn_start() {
        let e = Engine::bundled_latest().unwrap();
        let mut s = position(&[("a2", Player::White, PieceKind::Rook)]);
        s = e
            .grant_card(&s, 1, crate::PENTA_CHECK, Player::White)
            .unwrap()
            .state;
        for (i, (from, to, king_from, king_to)) in [
            ("a2", "h2", "h8", "g8"),
            ("h2", "g2", "g8", "f8"),
            ("g2", "f2", "f8", "e8"),
            ("f2", "e2", "e8", "d8"),
            ("e2", "d2", "d8", "c8"),
        ]
        .into_iter()
        .enumerate()
        {
            s = e.apply(&s, movement(from, to)).unwrap().state;
            assert_eq!(
                s.card_runtime.october.penta_check[0].unwrap().count,
                (i + 1) as u8
            );
            assert_eq!(s.status, GameStatus::Ongoing);
            s = e.apply(&s, movement(king_from, king_to)).unwrap().state;
            if i < 4 {
                assert_eq!(s.status, GameStatus::Ongoing);
            } else {
                assert_eq!(
                    s.status,
                    GameStatus::Won {
                        winner: Player::White,
                        reason: WinReason::CardEffect(crate::PENTA_CHECK.into())
                    }
                );
            }
        }
    }

    #[test]
    fn penta_check_does_not_count_initial_or_sustained_checks_and_observes_opponent_moves() {
        let e = Engine::bundled_latest().unwrap();
        let mut s = position(&[("h2", Player::White, PieceKind::Rook)]);
        s = e
            .grant_card(&s, 1, crate::PENTA_CHECK, Player::White)
            .unwrap()
            .state;
        assert_eq!(s.card_runtime.october.penta_check[0].unwrap().count, 0);
        s.actions_remaining = 2;
        s = e.apply(&s, movement("h2", "h3")).unwrap().state;
        assert_eq!(s.card_runtime.october.penta_check[0].unwrap().count, 0);
        s = e.apply(&s, movement("h3", "g3")).unwrap().state;
        assert!(
            !s.card_runtime.october.penta_check[0]
                .unwrap()
                .opponent_checked
        );
        s = e.apply(&s, movement("h8", "g8")).unwrap().state;
        assert_eq!(s.card_runtime.october.penta_check[0].unwrap().count, 1);
    }

    #[test]
    fn royal_check_respects_actual_identity_immunity_and_checker_jump_geometry() {
        let e = Engine::bundled_latest().unwrap();
        let mut s = position(&[("h2", Player::White, PieceKind::Rook)]);
        let king = at(&s, "h8");
        s.card_runtime.frozen.insert(
            king.id,
            FrozenStatus {
                source: Player::White,
                remaining_turns: 2,
            },
        );
        assert!(!e.october_royal_checked(&s, Player::White));
        s.card_runtime.frozen.clear();
        s.card_runtime.shielded_piece_ids.insert(king.id);
        assert!(e.october_royal_checked(&s, Player::White));
        let mut checker = position(&[("g5", Player::White, PieceKind::Checker)]);
        let king = at(&checker, "h8");
        checker.board.set("h8".parse().unwrap(), None);
        checker.board.set("f6".parse().unwrap(), Some(king));
        assert!(e.october_royal_checked(&checker, Player::White));
    }

    #[test]
    fn prophecy_counts_six_completed_turns_and_extra_actions_do_not_tick() {
        let e = Engine::bundled_latest().unwrap();
        let mut s = position(&[]);
        arm(&e, &mut s);
        s.actions_remaining = 2;
        s = e.apply(&s, movement("a1", "a2")).unwrap().state;
        assert_eq!(s.card_runtime.october.prophecy_remaining[0], Some(6));
        for (i, (a, b)) in [
            ("a2", "a3"),
            ("h8", "h7"),
            ("a3", "b3"),
            ("h7", "g7"),
            ("b3", "c3"),
            ("g7", "f7"),
        ]
        .into_iter()
        .enumerate()
        {
            s = e.apply(&s, movement(a, b)).unwrap().state;
            if i < 5 {
                assert_eq!(s.status, GameStatus::Ongoing);
                assert_eq!(
                    s.card_runtime.october.prophecy_remaining[0],
                    Some(5 - i as u8)
                );
            }
        }
        assert_eq!(
            s.status,
            GameStatus::Won {
                winner: Player::White,
                reason: WinReason::CardEffect(crate::PROPHECY.into())
            }
        );
    }

    #[test]
    fn prophecy_cancels_on_check_or_removal_before_last_tick_including_sacrifice() {
        let e = Engine::bundled_latest().unwrap();
        let mut s = position(&[
            ("a2", Player::White, PieceKind::Rook),
            ("a7", Player::Black, PieceKind::Pawn),
        ]);
        arm(&e, &mut s);
        s.card_runtime.october.prophecy_remaining[0] = Some(1);
        s = e.apply(&s, movement("a2", "a7")).unwrap().state;
        assert_eq!(s.status, GameStatus::Ongoing);
        assert_eq!(s.card_runtime.october.prophecy_remaining, [None; 2]);
        let mut s = position(&[("a2", Player::White, PieceKind::Rook)]);
        arm(&e, &mut s);
        s.card_runtime.october.prophecy_remaining[0] = Some(1);
        s = e.apply(&s, movement("a2", "h2")).unwrap().state;
        assert_eq!(s.status, GameStatus::Ongoing);
        assert_eq!(s.card_runtime.october.prophecy_remaining, [None; 2]);
        let mut s = position(&[("c1", Player::White, PieceKind::Bishop)]);
        arm(&e, &mut s);
        s = e
            .grant_card(&s, 2, crate::GREEK_GIFT, Player::White)
            .unwrap()
            .state;
        s = e
            .apply(
                &s,
                Action::UseCard {
                    instance_id: 2,
                    target: ActionTarget::PieceAt("c1".parse().unwrap()),
                },
            )
            .unwrap()
            .state;
        assert_eq!(s.card_runtime.october.prophecy_remaining, [None; 2]);
    }

    #[test]
    fn prophecy_started_while_checked_cancels_immediately_and_search_is_transactional() {
        let e = Engine::bundled_latest().unwrap();
        let mut s = position(&[("h2", Player::White, PieceKind::Rook)]);
        arm(&e, &mut s);
        assert_eq!(s.cards[0].charges, 0);
        assert_eq!(s.card_runtime.october.prophecy_remaining, [None; 2]);
        let mut quiet = position(&[]);
        arm(&e, &mut quiet);
        let before = quiet.clone();
        let mut chance = SearchChanceContext::new(31);
        let actual = e
            .apply_for_search(&quiet, movement("a1", "a2"), &mut chance)
            .unwrap();
        assert!(!actual.sampled_chance);
        assert_eq!(
            actual
                .transition
                .state
                .card_runtime
                .october
                .prophecy_remaining[0],
            Some(5)
        );
        assert_eq!(quiet, before);
        let mut changed = quiet.clone();
        changed.card_runtime.october.prophecy_remaining[0] = Some(4);
        assert_ne!(changed.position_key(), quiet.position_key());
    }
}
