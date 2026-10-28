use super::*;
pub(super) const ID: &str = crate::CHESS_N_POW_30;
pub(super) fn install(
    engine: &Engine,
    state: &mut GameState,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    mark_rule(state, ID);
    engine.install_chess_n_pow_30(state, events)
}
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::rule(ID, install, rule_eligibility);

#[derive(Clone, Copy)]
pub(super) struct SeededRandom(pub(super) u32);
impl SeededRandom {
    pub(super) fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(1831565813);
        let mut mixed = self.0;
        mixed = (mixed ^ (mixed >> 15)).wrapping_mul(mixed | 1);
        mixed ^= mixed.wrapping_add((mixed ^ (mixed >> 7)).wrapping_mul(mixed | 61));
        f64::from(mixed ^ (mixed >> 14)) / 4294967296.0
    }
    pub(super) fn bounded(&mut self, bound: u32) -> u32 {
        let limit = (1u64 << 32) / u64::from(bound) * u64::from(bound);
        loop {
            let word = (self.unit() * 4294967296.0) as u32;
            if u64::from(word) < limit {
                return word % bound;
            }
        }
    }
}
pub(super) const POOL: &[(PieceKind, u8, f64)] = &[
    (PieceKind::Queen, 9, 3.4),
    (PieceKind::Rook, 5, 3.0),
    (PieceKind::Bishop, 3, 2.8),
    (PieceKind::Missionary, 2, 0.7),
    (PieceKind::Knight, 3, 0.8),
    (PieceKind::Pawn, 1, 0.6),
    (PieceKind::Protestant, 3, 2.8),
    (PieceKind::Herald, 5, 3.0),
    (PieceKind::Cannon, 4, 2.9),
    (PieceKind::Fanatic, 1, 0.6),
    (PieceKind::PrimeMinister, 9, 1.4),
    (PieceKind::Alibaba, 2, 0.7),
    (PieceKind::Amazon, 13, 3.8),
    (PieceKind::Cardinal, 7, 3.2),
    (PieceKind::Unicorn, 5, 1.0),
    (PieceKind::Jester, 9, 3.4),
    (PieceKind::Camel, 2, 0.7),
    (PieceKind::Log, 2, 0.7),
    (PieceKind::Hook, 15, 4.0),
    (PieceKind::Grasshopper, 4, 0.9),
    (PieceKind::Dragon, 5, 1.0),
    (PieceKind::Man, 4, 0.9),
    (PieceKind::Assassin, 4, 0.9),
    (PieceKind::Reaper, 9, 1.4),
    (PieceKind::Knightmaster, 3, 0.8),
    (PieceKind::StandardBearer, 2, 0.7),
    (PieceKind::Guard, 2, 0.7),
    (PieceKind::Recruiter, 9, 1.4),
    (PieceKind::Squire, 1, 0.6),
    (PieceKind::Checker, 1, 0.6),
    (PieceKind::CheckerKing, 2, 0.7),
    (PieceKind::Wizard, 9, 1.4),
    (PieceKind::Alfil, 1, 0.6),
    (PieceKind::Windmill, 4, 2.9),
    (PieceKind::Idol, 9, 3.4),
    (PieceKind::Lobster, 2, 0.7),
    (PieceKind::BabyBear, 4, 0.9),
    (PieceKind::Bear, 17, 4.2),
    (PieceKind::SiegeRam, 5, 1.0),
    (PieceKind::MagicGirl, 6, 1.1),
    (PieceKind::Berserker, 7, 1.2000000000000002),
    (PieceKind::Slime, 5, 1.0),
    (PieceKind::Siren, 9, 1.4),
    (PieceKind::Trickster, 5, 1.0),
    (PieceKind::Undead, 9, 1.4),
    (PieceKind::Campfire, 4, 0.9),
    (PieceKind::Hedgehog, 9, 1.4),
    (PieceKind::Princess, 6, 1.1),
    (PieceKind::Thief, 9, 1.4),
    (PieceKind::Paladin, 4, 0.9),
    (PieceKind::Octopus, 5, 1.0),
    (PieceKind::Parrot, 5, 1.0),
    (PieceKind::Clockwork, 5, 3.0),
    (PieceKind::Brutus, 10, 3.5),
    (PieceKind::Grappler, 13, 1.8),
    (PieceKind::RevolvingDoor, 5, 1.0),
    (PieceKind::DonQuixote, 7, 1.2000000000000002),
    (PieceKind::Medium, 3, 0.8),
    (PieceKind::BerolinaPawn, 1, 0.6),
    (PieceKind::Chancellor, 8, 3.3),
    (PieceKind::Archbishop, 7, 3.2),
    (PieceKind::Killer, 3, 0.8),
    (PieceKind::Beacon, 5, 1.0),
    (PieceKind::TransmissionTower, 5, 1.0),
    (PieceKind::Barrier, 5, 1.0),
];
fn choose(random: &mut SeededRandom, weights: &[f64]) -> usize {
    let mut threshold = random.unit() * weights.iter().sum::<f64>();
    for (i, weight) in weights.iter().enumerate() {
        if threshold < *weight {
            return i;
        }
        threshold -= weight;
    }
    weights.iter().rposition(|w| *w > 0.0).unwrap()
}
fn shuffle(random: &mut SeededRandom, lineup: &mut [PieceKind]) {
    for i in (1..lineup.len()).rev() {
        let j = (random.unit() * (i + 1) as f64).floor() as usize;
        lineup.swap(i, j);
    }
}
fn lineup(random: &mut SeededRandom, target: usize, ways: &[[f64; 71]; 16]) -> Vec<PieceKind> {
    let mut remaining = target;
    let mut result = Vec::new();
    for slots in (1..=15).rev() {
        let weights: Vec<_> = POOL
            .iter()
            .map(|(_, v, _)| {
                if usize::from(*v) <= remaining {
                    ways[slots - 1][remaining - usize::from(*v)]
                } else {
                    0.0
                }
            })
            .collect();
        let i = choose(random, &weights);
        result.push(POOL[i].0);
        remaining -= usize::from(POOL[i].1);
    }
    result
}
fn arrange(random: &mut SeededRandom, lineup: Vec<PieceKind>) -> Option<Vec<PieceKind>> {
    let mut remaining = lineup;
    let mut home = Vec::new();
    while home.len() < 7 {
        let weights: Vec<_> = remaining
            .iter()
            .map(|k| POOL.iter().find(|(kind, _, _)| kind == k).unwrap().2)
            .collect();
        let i = choose(random, &weights);
        home.push(remaining.remove(i));
    }
    shuffle(random, &mut home);
    shuffle(random, &mut remaining);
    home.extend(remaining);
    for index in [3, 4, 11] {
        if home[index] != PieceKind::Brutus {
            continue;
        }
        let candidates: Vec<_> = (0..15)
            .filter(|i| ![3, 4, 11].contains(i) && home[*i] != PieceKind::Brutus)
            .collect();
        if candidates.is_empty() {
            return None;
        }
        let i = (random.unit() * candidates.len() as f64).floor() as usize;
        home.swap(index, candidates[i]);
    }
    Some(home)
}
fn seeded_setup(seed: u32) -> Option<(u8, [Vec<PieceKind>; 2], SeededRandom)> {
    let mut random = SeededRandom(seed);
    let mut ways = [[0.0; 71]; 16];
    ways[0][0] = 1.0;
    for slots in 1..=15 {
        for total in 0..=70 {
            ways[slots][total] = POOL
                .iter()
                .filter(|(_, v, _)| usize::from(*v) <= total)
                .map(|(_, v, _)| ways[slots - 1][total - usize::from(*v)])
                .sum();
        }
    }
    let totals: Vec<_> = (39..=70).filter(|v| ways[15][*v] > 0.0).collect();
    let total = totals[(random.unit() * totals.len() as f64).floor() as usize];
    let white = lineup(&mut random, total, &ways);
    let mut black = lineup(&mut random, total, &ways);
    for _ in 0..8 {
        if white != black {
            break;
        }
        black = lineup(&mut random, total, &ways);
    }
    Some((
        total as u8,
        [arrange(&mut random, white)?, arrange(&mut random, black)?],
        random,
    ))
}
impl Engine {
    pub(in crate::engine) fn install_chess_n_pow_30(
        &self,
        state: &mut GameState,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        let seed =
            rule_random(state, events, 1usize << 32, "rule.chess-n-pow-30.seed").unwrap() as u32;
        let (_, armies, mut local_random) = seeded_setup(seed).ok_or_else(|| {
            EngineError::Invariant("혼돈의 체스 배치를 생성할 수 없습니다.".into())
        })?;
        let old: Vec<_> = state.board.iter().map(|(_, p)| p.id).collect();
        for id in old {
            self.clear_removed_piece_statuses(state, id);
        }
        state.board = Board::empty();
        for (owner, army) in [Player::White, Player::Black].into_iter().zip(armies) {
            let home = if owner == Player::White { 0 } else { 7 };
            let pawn_rank = if owner == Player::White { 1 } else { 6 };
            let king = Piece::new(owner, PieceKind::King).with_id(state.allocate_piece_id());
            let square = Square::new(4, home).unwrap();
            state.board.set(square, Some(king));
            self.register_spawned_piece(state, square, king, false);
            let squares: Vec<_> = (0..8)
                .filter(|f| *f != 4)
                .map(|f| Square::new(f, home).unwrap())
                .chain((0..8).map(|f| Square::new(f, pawn_rank).unwrap()))
                .collect();
            for (square, kind) in squares.into_iter().zip(army) {
                let piece = Piece::new(owner, kind).with_id(state.allocate_piece_id());
                state.board.set(square, Some(piece));
                self.register_spawned_piece(state, square, piece, false);
                match kind {
                    PieceKind::BabyBear => {
                        state
                            .card_runtime
                            .baby_bear_grow_turns
                            .insert(piece.id, state.shared_turn_count().saturating_add(7));
                    }
                    PieceKind::Bear | PieceKind::Hedgehog => {
                        state.card_runtime.bear_retaliations.insert(piece.id, 2);
                    }
                    PieceKind::Thief => {
                        state.card_runtime.submerged_piece_ids.insert(piece.id);
                    }
                    PieceKind::Wizard => {
                        state.card_runtime.wizard_mana.insert(piece.id, 0);
                    }
                    PieceKind::Windmill => {
                        state
                            .card_runtime
                            .windmill_modes
                            .insert(piece.id, WindmillMode::Bishop);
                    }
                    PieceKind::Reaper => {
                        state.card_runtime.reaper_souls.insert(piece.id, 0);
                    }
                    PieceKind::Trickster => {
                        let kind = TRICKSTER_POOL
                            [(local_random.unit() * TRICKSTER_POOL.len() as f64).floor() as usize];
                        state
                            .card_runtime
                            .trickster_move_kinds
                            .insert(piece.id, kind);
                    }
                    _ => {}
                }
            }
        }
        state.rule_runtime.chaos_until_halfmove = Some(rule_halfmove(state) + 4);
        state.en_passant = None;
        Ok(())
    }
}

pub(in crate::engine) const TRICKSTER_POOL: &[PieceKind] = &[
    PieceKind::Queen,
    PieceKind::Rook,
    PieceKind::Bishop,
    PieceKind::Missionary,
    PieceKind::Knight,
    PieceKind::Pawn,
    PieceKind::Protestant,
    PieceKind::Herald,
    PieceKind::Cannon,
    PieceKind::Fanatic,
    PieceKind::PrimeMinister,
    PieceKind::Alibaba,
    PieceKind::Amazon,
    PieceKind::Cardinal,
    PieceKind::Unicorn,
    PieceKind::Jester,
    PieceKind::Camel,
    PieceKind::Hook,
    PieceKind::Grasshopper,
    PieceKind::Dragon,
    PieceKind::Man,
    PieceKind::Assassin,
    PieceKind::Reaper,
    PieceKind::Knightmaster,
    PieceKind::StandardBearer,
    PieceKind::Guard,
    PieceKind::Recruiter,
    PieceKind::Squire,
    PieceKind::Checker,
    PieceKind::CheckerKing,
    PieceKind::Wizard,
    PieceKind::Alfil,
    PieceKind::Windmill,
    PieceKind::Idol,
    PieceKind::Lobster,
    PieceKind::Bear,
    PieceKind::SiegeRam,
    PieceKind::MagicGirl,
    PieceKind::Berserker,
    PieceKind::Slime,
    PieceKind::Siren,
    PieceKind::Undead,
    PieceKind::Campfire,
    PieceKind::Hedgehog,
    PieceKind::Princess,
    PieceKind::Thief,
    PieceKind::Brutus,
    PieceKind::Clockwork,
    PieceKind::Parrot,
    PieceKind::Paladin,
    PieceKind::Octopus,
    PieceKind::Grappler,
    PieceKind::RevolvingDoor,
    PieceKind::DonQuixote,
    PieceKind::Medium,
    PieceKind::Chancellor,
    PieceKind::Archbishop,
    PieceKind::Killer,
    PieceKind::Beacon,
    PieceKind::TransmissionTower,
    PieceKind::Barrier,
    PieceKind::BerolinaPawn,
];

#[cfg(test)]
pub(super) fn seeded_armies(seed: u32) -> Option<(u8, [Vec<PieceKind>; 2])> {
    seeded_setup(seed).map(|(v, a, _)| (v, a))
}
