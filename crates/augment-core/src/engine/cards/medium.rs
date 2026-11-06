use super::transform_support::{self, TransformSource, TransformSpec};
use super::*;
pub(super) const ID: &str = crate::MEDIUM;
const SPEC: TransformSpec = TransformSpec {
    source: TransformSource::Minor,
    kind: PieceKind::Medium,
    allow_royal: false,
};
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance: u32,
    target: &ActionTarget,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let (square, piece) = piece_target(state, target)?;
    if !SPEC.accepts(state, actor, piece) {
        return Err(EngineError::Invariant(
            "invalid piece transformation target".into(),
        ));
    }
    engine.transform_current_piece(state, square, SPEC.kind, true)?;
    Ok(())
}
pub(super) fn actions(
    engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    out: &mut Vec<Action>,
) {
    transform_support::actions(engine, state, card, SPEC, out);
}
pub(super) fn eligibility(
    state: &GameState,
    owner: Player,
    _: AcquisitionContext,
) -> CardEligibility {
    transform_support::eligibility(state, owner, SPEC)
}
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);

impl Engine {
    pub(in crate::engine) fn medium_borrows_movement(
        &self,
        state: &GameState,
        piece: Piece,
        kind: PieceKind,
    ) -> bool {
        state.ruleset_id == crate::LATEST_RULESET_ID
            && piece.kind == PieceKind::Medium
            && state
                .rule_runtime
                .medium_memory
                .is_some_and(|m| m.source == kind)
    }

    pub(in crate::engine) fn generate_current_medium_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        out: &mut Vec<Action>,
    ) {
        let Some(memory) = state.rule_runtime.medium_memory else {
            return;
        };
        let orth = &[(-1, 0), (1, 0), (0, -1), (0, 1)];
        let diag = &[(-1, -1), (-1, 1), (1, -1), (1, 1)];
        let king = &[
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ];
        let mut moves = Vec::new();
        let mut quiet = false;
        match memory.source {
            PieceKind::Queen | PieceKind::Bear | PieceKind::Clockwork | PieceKind::Grappler => {
                self.generate_slider_moves(state, from, piece, king, &mut moves)
            }
            PieceKind::Rook | PieceKind::BigRook | PieceKind::RevolvingDoor => {
                self.generate_slider_moves(state, from, piece, orth, &mut moves)
            }
            PieceKind::Bishop | PieceKind::BigBishop => {
                self.generate_slider_moves(state, from, piece, diag, &mut moves)
            }
            PieceKind::Windmill => self.generate_slider_moves(
                state,
                from,
                piece,
                if state.rule_runtime.medium_windmill_mode == Some(WindmillMode::Rook) {
                    orth
                } else {
                    diag
                },
                &mut moves,
            ),
            PieceKind::Knight
            | PieceKind::RoyalKnight
            | PieceKind::Unicorn
            | PieceKind::DonQuixote => self.generate_knight_moves(state, from, piece, &mut moves),
            PieceKind::Dragon => {
                self.generate_knight_moves(state, from, piece, &mut moves);
                for (to, target) in state.board.iter() {
                    if to != from
                        && target.owner == piece.owner
                        && !matches!(
                            target.kind,
                            PieceKind::Slime
                                | PieceKind::Colossus
                                | PieceKind::BigRook
                                | PieceKind::BigBishop
                        )
                    {
                        moves.push(Action::Move(MoveAction::new(from, to, None)));
                    }
                }
            }
            PieceKind::Assassin => {
                self.generate_knight_moves(state, from, piece, &mut moves);
                for &(df, dr) in king {
                    let mut cursor = from;
                    while let Some(to) = cursor.offset(df, dr) {
                        if let Some(target) = self.movement_occupant(state, to) {
                            if target.owner != piece.owner
                                && self.is_royal_piece(state, target)
                                && self.can_capture(state, piece, target)
                            {
                                moves.push(Action::Move(MoveAction::new(from, to, None)));
                            }
                            break;
                        }
                        cursor = to;
                    }
                }
            }
            PieceKind::Vip => {
                self.generate_leaper_moves(state, from, piece, king, &mut moves);
                self.generate_leaper_moves(
                    state,
                    from,
                    piece,
                    &[(-2, -2), (-2, 2), (2, -2), (2, 2)],
                    &mut moves,
                );
            }
            PieceKind::Amazon => self.generate_amazon_moves(state, from, piece, &mut moves),
            PieceKind::Chancellor => {
                self.generate_slider_moves(state, from, piece, orth, &mut moves);
                self.generate_knight_moves(state, from, piece, &mut moves);
            }
            PieceKind::Archbishop => {
                self.generate_slider_moves(state, from, piece, diag, &mut moves);
                self.generate_knight_moves(state, from, piece, &mut moves);
            }
            PieceKind::Cardinal => self.generate_cardinal_moves(state, from, piece, &mut moves),
            PieceKind::PrimeMinister => {
                self.generate_prime_minister_moves(state, from, piece, &mut moves)
            }
            PieceKind::Protestant => self.generate_protestant_moves(state, from, piece, &mut moves),
            PieceKind::Cannon => self.generate_cannon_moves(state, from, piece, &mut moves),
            PieceKind::Hook | PieceKind::Brutus => {
                self.generate_hook_moves(state, from, piece, &mut moves)
            }
            PieceKind::Grasshopper => {
                self.generate_grasshopper_moves(state, from, piece, &mut moves)
            }
            PieceKind::Alibaba => self.generate_leaper_moves(
                state,
                from,
                piece,
                &[
                    (-2, -2),
                    (-2, 0),
                    (-2, 2),
                    (0, -2),
                    (0, 2),
                    (2, -2),
                    (2, 0),
                    (2, 2),
                ],
                &mut moves,
            ),
            PieceKind::Log => self.generate_leaper_moves(state, from, piece, king, &mut moves),
            PieceKind::Lobster => {
                let dir = if piece.owner == Player::White { 1 } else { -1 };
                self.generate_leaper_moves(
                    state,
                    from,
                    piece,
                    &[(-1, dir), (0, dir), (1, dir)],
                    &mut moves,
                );
            }
            PieceKind::Thief => {
                for &(df, dr) in orth {
                    for n in 1..=3 {
                        if let Some(to) = from.offset(df * n, dr * n) {
                            let target = self.movement_occupant(state, to);
                            let jumped = (1..n).any(|step| {
                                from.offset(df * step, dr * step)
                                    .is_some_and(|s| self.movement_occupant(state, s).is_some())
                            });
                            if target.is_none()
                                || !jumped
                                    && target.is_some_and(|p| self.can_capture(state, piece, p))
                            {
                                moves.push(Action::Move(MoveAction::new(from, to, None)));
                            }
                        }
                    }
                }
            }
            PieceKind::Herald => {
                for n in 1..=3 {
                    let offsets: Vec<_> = orth.iter().map(|(df, dr)| (df * n, dr * n)).collect();
                    self.generate_leaper_moves(state, from, piece, &offsets, &mut moves);
                }
                quiet = true;
            }
            PieceKind::SiegeRam => {
                for n in 1..=2 {
                    let offsets: Vec<_> = orth.iter().map(|(df, dr)| (df * n, dr * n)).collect();
                    self.generate_leaper_moves(state, from, piece, &offsets, &mut moves);
                }
            }
            PieceKind::Fanatic => {
                let dir = if piece.owner == Player::White { 1 } else { -1 };
                self.generate_slider_moves(state, from, piece, &[(0, dir)], &mut moves);
                moves.retain(
                    |a| matches!(a,Action::Move(m) if from.rank().abs_diff(m.to.rank())<=2),
                );
            }
            PieceKind::MagicGirl if state.card_runtime.magic_girl_surge[piece.owner.index()] => {
                self.generate_amazon_moves(state, from, piece, &mut moves)
            }
            PieceKind::Berserker => {
                let army = state
                    .board
                    .iter()
                    .filter(|(_, p)| p.owner == piece.owner && p.kind != PieceKind::Monster)
                    .count();
                if army <= 5 {
                    self.generate_amazon_moves(state, from, piece, &mut moves);
                } else {
                    self.generate_leaper_moves(state, from, piece, king, &mut moves);
                    if army <= 9 {
                        self.generate_slider_moves(state, from, piece, orth, &mut moves);
                    }
                }
            }
            PieceKind::BerolinaPawn => {
                let mut borrowed = piece;
                borrowed.kind = PieceKind::BerolinaPawn;
                self.generate_rule_dependency_moves(state, from, borrowed, &mut moves);
                for action in &mut moves {
                    if let Action::Move(m) = action {
                        m.promotion = None;
                    }
                }
            }
            PieceKind::Checker | PieceKind::CheckerKing => {
                let dir = if piece.owner == Player::White { 1 } else { -1 };
                let offsets: Vec<_> = diag
                    .iter()
                    .copied()
                    .filter(|(_, dr)| memory.source == PieceKind::CheckerKing || *dr == dir)
                    .collect();
                for (df, dr) in offsets {
                    if let Some(mid) = from.offset(df, dr) {
                        match self.movement_occupant(state, mid) {
                            None => moves.push(Action::Move(MoveAction::new(from, mid, None))),
                            Some(target) if self.can_capture(state, piece, target) => {
                                if let Some(to) = mid.offset(df, dr)
                                    && self.movement_occupant(state, to).is_none()
                                {
                                    moves.push(Action::Move(MoveAction::new(from, to, None)));
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            PieceKind::Campfire => {
                self.generate_slider_moves(state, from, piece, orth, &mut moves);
                quiet = true;
            }
            PieceKind::Idol | PieceKind::Jester => {
                self.generate_slider_moves(state, from, piece, king, &mut moves);
                quiet = true;
            }
            PieceKind::Beacon | PieceKind::Wizard => {
                self.generate_leaper_moves(state, from, piece, king, &mut moves);
                quiet = true;
            }
            PieceKind::Killer => {
                self.generate_slider_moves(state, from, piece, diag, &mut moves);
                moves.retain(
                    |a| matches!(a,Action::Move(m) if from.file().abs_diff(m.to.file())<=2),
                );
            }
            PieceKind::TransmissionTower
            | PieceKind::Barrier
            | PieceKind::Medium
            | PieceKind::Parrot
            | PieceKind::Trickster
            | PieceKind::BabyBear
            | PieceKind::Scarecrow
            | PieceKind::Football => {}
            // The existing base-memory generator owns the simple leaper/pawn families.
            _ => {
                let mut probe = state.clone();
                probe.card_runtime.parrot_memory[piece.owner.index()] = Some(memory);
                self.generate_parrot_memory_moves(&probe, from, piece, &mut moves);
            }
        }
        if quiet {
            moves.retain(
                |a| matches!(a,Action::Move(m) if self.movement_occupant(state,m.to).is_none()),
            );
        }
        out.extend(moves);
    }
}
