use super::*;
pub(super) const ID: &str = crate::PAWN_CHAIN;
pub(super) fn acquire(
    _engine: &Engine,
    state: &mut GameState,
    owner: Player,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
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
    state.card_runtime.effects.pawn_chain_links.extend(links);
    Ok(())
}

pub(super) fn eligibility(
    _state: &GameState,
    _player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    CardEligibility::Eligible
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

impl Engine {
    pub(in crate::engine) fn clean_pawn_chains(
        &self,
        state: &mut GameState,
        before: &crate::Board,
    ) {
        if state.ruleset_id != crate::LATEST_RULESET_ID {
            return;
        }
        state.card_runtime.effects.pawn_chain_links.retain(|pair| {
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

pub(super) fn pawn_links(mut pawns: Vec<(Square, u32)>, horde: Option<Player>) -> Vec<[u32; 2]> {
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
    pub(in crate::engine) fn root(parent: &[usize], mut i: usize) -> usize {
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

#[cfg(test)]
mod tests {
    use super::super::tests::objectives::*;
    use super::*;
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
        assert_eq!(s.card_runtime.effects.pawn_chain_links.len(), 7);
        s.turn = Player::Black;
        assert!(has_move(&e, &s, "b7", "b5"));
        s = e.apply(&s, movement("b7", "b5")).unwrap().state;
        s.turn = Player::Black;
        assert!(!has_move(&e, &s, "b5", "b4"));
        assert_eq!(s.card_runtime.effects.pawn_chain_links.len(), 7);
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
        assert_eq!(s.card_runtime.effects.pawn_chain_links.len(), 1);
        let after = e.apply(&s, movement("b1", "b3")).unwrap().state;
        assert!(after.card_runtime.effects.pawn_chain_links.is_empty());
        let before = s.board.clone();
        let p = at(&s, "b3");
        s.board.set("b3".parse().unwrap(), None);
        s.board.set("b7".parse().unwrap(), Some(p));
        e.clean_pawn_chains(&mut s, &before);
        assert!(s.card_runtime.effects.pawn_chain_links.is_empty());
        s.card_runtime
            .effects
            .pawn_chain_links
            .push([p.id, at(&s, "c3").id]);
        let before = s.board.clone();
        e.clean_pawn_chains(&mut s, &before);
        assert_eq!(s.card_runtime.effects.pawn_chain_links.len(), 1);
    }
}
