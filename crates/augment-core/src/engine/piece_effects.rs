use super::*;
use crate::*;

const KING_OFFSETS: &[(i8, i8)] = &[
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];
pub(super) const TROLLEY_CHOICE_HANDLER: &str = "middle.trolley";

#[derive(Clone, Copy)]
pub(super) struct MiddleCaptureContext {
    poisoned: bool,
    explosive: bool,
    trojan: bool,
    feudal_guardian_id: Option<u32>,
}

impl Engine {
    pub(super) fn alekhine_formation(&self, state: &GameState, actor: Player) -> Option<Vec<u32>> {
        if state.ruleset_id == crate::LATEST_RULESET_ID {
            let mut entries = state.board.iter().collect::<Vec<_>>();
            entries.sort_by_key(|(square, _)| (7 - square.rank(), square.file()));
            for (square, queen) in entries
                .iter()
                .copied()
                .filter(|(_, p)| p.owner == actor && p.kind == PieceKind::Queen)
            {
                let rooks = entries
                    .iter()
                    .copied()
                    .filter(|(s, p)| {
                        p.owner == actor && p.kind == PieceKind::Rook && s.file() == square.file()
                    })
                    .collect::<Vec<_>>();
                if rooks.len() >= 2
                    && rooks.iter().all(|(s, _)| {
                        if actor == Player::White {
                            s.rank() > square.rank()
                        } else {
                            s.rank() < square.rank()
                        }
                    })
                {
                    return Some(vec![queen.id, rooks[0].1.id, rooks[1].1.id]);
                }
            }
            return None;
        }
        let queens = state
            .board
            .iter()
            .filter(|(_, piece)| piece.owner == actor && piece.kind == PieceKind::Queen);
        for (queen_square, queen) in queens {
            let rooks = state
                .board
                .iter()
                .filter(|(_, piece)| piece.owner == actor && piece.kind == PieceKind::Rook)
                .collect::<Vec<_>>();
            for first in 0..rooks.len() {
                for second in first + 1..rooks.len() {
                    let a = rooks[first].0;
                    let b = rooks[second].0;
                    if (a.rank() == queen_square.rank()
                        && b.rank() == queen_square.rank()
                        && (a.file() < queen_square.file()) != (b.file() < queen_square.file()))
                        || (a.file() == queen_square.file()
                            && b.file() == queen_square.file()
                            && (a.rank() < queen_square.rank()) != (b.rank() < queen_square.rank()))
                    {
                        return Some(vec![queen.id, rooks[first].1.id, rooks[second].1.id]);
                    }
                }
            }
        }
        None
    }

    pub(super) fn apply_free_castling(
        &self,
        state: &mut GameState,
        actor: Player,
        target: &ActionTarget,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        let rook_square = square_target(target)?;
        let rook = state
            .board
            .get(rook_square)
            .ok_or_else(|| EngineError::Invariant("castling rook disappeared".to_owned()))?;
        let candidates = if state.ruleset_id == crate::LATEST_RULESET_ID {
            web_ordered_pieces(state)
        } else {
            state.board.iter().collect()
        };
        let (king_square, king) = candidates
            .into_iter()
            .find(|(_, piece)| piece.owner == actor && self.is_royal_piece(state, *piece))
            .ok_or_else(|| EngineError::Invariant("castling king disappeared".to_owned()))?;
        let (file_delta, rank_delta, distance) = aligned_distance(king_square, rook_square)
            .ok_or_else(|| EngineError::Invariant("castling pieces are not aligned".to_owned()))?;
        if distance < 3 {
            return Err(EngineError::Invariant(
                "castling rook is too close".to_owned(),
            ));
        }
        let king_to = king_square
            .offset(file_delta * 2, rank_delta * 2)
            .expect("aligned castling destination");
        let rook_to = king_square
            .offset(file_delta, rank_delta)
            .expect("aligned rook destination");
        state.board.set(king_square, None);
        state.board.set(rook_square, None);
        let mut enemy_king_captured = false;
        for (square, attacker) in [(king_to, king), (rook_to, rook)] {
            if let Some(victim) = state.board.get(square) {
                state.board.set(square, None);
                if state.ruleset_id == crate::LATEST_RULESET_ID {
                    enemy_king_captured |= victim.owner != actor && victim.kind == PieceKind::King;
                    if state.card_runtime.imperial_studies[actor.index()] {
                        state
                            .card_runtime
                            .imperial_moves
                            .entry(king.id)
                            .or_default()
                            .insert(victim.kind);
                    }
                }
                self.clear_removed_piece_statuses(state, victim.id);
                events.push(Event::PieceCaptured {
                    actor,
                    piece: victim,
                    at: square,
                });
            }
            state.board.set(square, Some(attacker));
            state.card_runtime.moved_piece_ids.insert(attacker.id);
        }
        if state.ruleset_id == crate::LATEST_RULESET_ID {
            state.card_runtime.castled_piece_ids.insert(king.id);
            if enemy_king_captured {
                state.status = GameStatus::Won {
                    winner: actor,
                    reason: WinReason::KingCaptured,
                };
            }
        }
        state.en_passant = None;
        Ok(())
    }

    pub(super) fn middle_piece_is_exhausted(&self, state: &GameState, piece: Piece) -> bool {
        state.card_runtime.exhaustion[piece.owner.index()]
            && !self.is_royal_piece(state, piece)
            && state.card_runtime.exhaustion_last_piece[piece.owner.index()] == Some(piece.id)
            && state.card_runtime.exhaustion_count[piece.owner.index()] >= 3
    }

    pub(super) fn middle_encouragement_protects(&self, state: &GameState, target: Piece) -> bool {
        if !state.card_runtime.encouragement[target.owner.index()] {
            return false;
        }
        if state.ruleset_id == crate::LATEST_RULESET_ID
            && matches!(target.kind, PieceKind::Scarecrow | PieceKind::Barrier)
        {
            return false;
        }
        let Some((target_square, _)) = state.board.find_by_id(target.id) else {
            return false;
        };
        if state.ruleset_id == crate::LATEST_RULESET_ID {
            return web_ordered_pieces(state)
                .into_iter()
                .find(|(_, p)| p.owner == target.owner && self.is_royal_piece(state, *p))
                .is_some_and(|(square, _)| {
                    square.file().abs_diff(target_square.file())
                        + square.rank().abs_diff(target_square.rank())
                        == 1
                });
        }
        state.board.iter().any(|(square, piece)| {
            piece.owner == target.owner
                && self.is_royal_piece(state, piece)
                && piece.id != target.id
                && squares_are_adjacent(square, target_square)
        })
    }

    /// Submerged status is lost when an enemy becomes adjacent. It is not
    /// restored when that enemy leaves; Octopus grants it only at turn completion.
    /// Parrot base movement copied from the owner's remembered last move
    /// (live parrotBaseMoves, JS 4205-4320). Quiet kinds never capture;
    /// capture-only entries never move into empty squares.
    pub(super) fn generate_parrot_memory_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        const KING: [(i8, i8); 8] = [
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ];
        const ORTH: [(i8, i8); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
        const DIAG: [(i8, i8); 4] = [(-1, -1), (-1, 1), (1, -1), (1, 1)];
        const KNIGHT: [(i8, i8); 8] = [
            (-2, -1),
            (-2, 1),
            (-1, -2),
            (-1, 2),
            (1, -2),
            (1, 2),
            (2, -1),
            (2, 1),
        ];
        let Some(memory) = state.card_runtime.parrot_memory[piece.owner.index()] else {
            return;
        };
        let owner = piece.owner;
        let dir: i8 = if owner == Player::White { 1 } else { -1 };
        #[allow(clippy::too_many_arguments)]
        fn push_leap(
            engine: &Engine,
            state: &GameState,
            piece: &Piece,
            owner: Player,
            from: Square,
            actions: &mut Vec<Action>,
            offsets: &[(i8, i8)],
            quiet: bool,
            capture_only: bool,
        ) {
            for &(df, dr) in offsets {
                let Some(to) = from.offset(df, dr) else {
                    continue;
                };
                match state.board.get(to) {
                    None => {
                        if !capture_only {
                            actions.push(Action::Move(MoveAction {
                                from,
                                to,
                                promotion: None,
                            }));
                        }
                    }
                    Some(target) => {
                        if !quiet
                            && target.owner != owner
                            && engine.can_capture(state, *piece, target)
                        {
                            actions.push(Action::Move(MoveAction {
                                from,
                                to,
                                promotion: None,
                            }));
                        }
                    }
                }
            }
        }
        fn push_ray(
            engine: &Engine,
            state: &GameState,
            piece: &Piece,
            owner: Player,
            from: Square,
            actions: &mut Vec<Action>,
            directions: &[(i8, i8)],
        ) {
            for &(df, dr) in directions {
                let mut step: i8 = 1;
                loop {
                    let file = i16::from(from.file()) + i16::from(df) * i16::from(step);
                    let rank = i16::from(from.rank()) + i16::from(dr) * i16::from(step);
                    let Some(to) = u8::try_from(file)
                        .ok()
                        .zip(u8::try_from(rank).ok())
                        .and_then(|(f, r)| Square::new(f, r))
                    else {
                        break;
                    };
                    match state.board.get(to) {
                        None => actions.push(Action::Move(MoveAction {
                            from,
                            to,
                            promotion: None,
                        })),
                        Some(target) => {
                            if target.owner != owner && engine.can_capture(state, *piece, target) {
                                actions.push(Action::Move(MoveAction {
                                    from,
                                    to,
                                    promotion: None,
                                }));
                            }
                            break;
                        }
                    }
                    step += 1;
                }
            }
        }
        match memory.source {
            PieceKind::Queen | PieceKind::Bear | PieceKind::Clockwork => {
                push_ray(self, state, &piece, owner, from, actions, &KING)
            }
            PieceKind::Rook => push_ray(self, state, &piece, owner, from, actions, &ORTH),
            PieceKind::Bishop => push_ray(self, state, &piece, owner, from, actions, &DIAG),
            PieceKind::Knight | PieceKind::Unicorn | PieceKind::Dragon | PieceKind::Assassin => {
                push_leap(
                    self, state, &piece, owner, from, actions, &KNIGHT, false, false,
                )
            }
            PieceKind::Paladin => push_leap(
                self, state, &piece, owner, from, actions, &KNIGHT, true, false,
            ),
            PieceKind::King
            | PieceKind::Octopus
            | PieceKind::Man
            | PieceKind::Guard
            | PieceKind::Reaper
            | PieceKind::Recruiter
            | PieceKind::Vip
            | PieceKind::Crown
            | PieceKind::DarkWizard
            | PieceKind::MagicGirl
            | PieceKind::Undead
            | PieceKind::Hedgehog
            | PieceKind::Siren
            | PieceKind::Berserker
            | PieceKind::Monster => push_leap(
                self, state, &piece, owner, from, actions, &KING, false, false,
            ),
            PieceKind::Ferz | PieceKind::Knightmaster | PieceKind::Princess => push_leap(
                self, state, &piece, owner, from, actions, &DIAG, false, false,
            ),
            PieceKind::Camel => push_leap(
                self,
                state,
                &piece,
                owner,
                from,
                actions,
                &[
                    (-3, -1),
                    (-3, 1),
                    (3, -1),
                    (3, 1),
                    (-1, -3),
                    (-1, 3),
                    (1, -3),
                    (1, 3),
                ],
                false,
                false,
            ),
            PieceKind::Alfil => push_leap(
                self,
                state,
                &piece,
                owner,
                from,
                actions,
                &[(-2, -2), (-2, 2), (2, -2), (2, 2)],
                false,
                false,
            ),
            PieceKind::Lobster => push_leap(
                self,
                state,
                &piece,
                owner,
                from,
                actions,
                &[(dir, -1), (dir, 0), (dir, 1)],
                false,
                false,
            ),
            PieceKind::Fanatic => {
                push_leap(
                    self,
                    state,
                    &piece,
                    owner,
                    from,
                    actions,
                    &[(dir, -1), (dir, 1)],
                    true,
                    false,
                );
                push_leap(
                    self,
                    state,
                    &piece,
                    owner,
                    from,
                    actions,
                    &[(dir, 0)],
                    false,
                    true,
                );
            }
            PieceKind::Slime => push_leap(
                self,
                state,
                &piece,
                owner,
                from,
                actions,
                &[(-3, 0), (3, 0), (0, -3), (0, 3)],
                false,
                false,
            ),
            PieceKind::Colossus => push_leap(
                self, state, &piece, owner, from, actions, &ORTH, false, false,
            ),
            PieceKind::SiegeRam => push_leap(
                self,
                state,
                &piece,
                owner,
                from,
                actions,
                &[(-2, 0), (2, 0), (0, -2), (0, 2)],
                false,
                false,
            ),
            PieceKind::Campfire => push_leap(
                self, state, &piece, owner, from, actions, &ORTH, true, false,
            ),
            PieceKind::Missionary => push_leap(
                self, state, &piece, owner, from, actions, &DIAG, true, false,
            ),
            PieceKind::Herald => push_leap(
                self,
                state,
                &piece,
                owner,
                from,
                actions,
                &[(-1, 0), (1, 0), (0, -1), (0, 1)],
                true,
                false,
            ),
            PieceKind::Jester | PieceKind::Idol => {
                // Quiet queen rays: walk manually to respect quiet semantics.
                for &(df, dr) in KING.iter() {
                    for n in 1..=7 {
                        let Some(to) = Square::new(
                            (from.file() as i8 + df * n) as u8,
                            (from.rank() as i8 + dr * n) as u8,
                        ) else {
                            break;
                        };
                        if state.board.get(to).is_some() {
                            break;
                        }
                        actions.push(Action::Move(MoveAction {
                            from,
                            to,
                            promotion: None,
                        }));
                    }
                }
            }
            PieceKind::Bat => {
                for &(df, dr) in ORTH.iter() {
                    for n in 1..=2 {
                        let Some(to) = from.offset(df * n, dr * n) else {
                            break;
                        };
                        match state.board.get(to) {
                            None => actions.push(Action::Move(MoveAction {
                                from,
                                to,
                                promotion: None,
                            })),
                            Some(target) => {
                                if target.owner != owner && self.can_capture(state, piece, target) {
                                    actions.push(Action::Move(MoveAction {
                                        from,
                                        to,
                                        promotion: None,
                                    }));
                                }
                                break;
                            }
                        }
                    }
                }
            }
            PieceKind::Wizard => push_leap(
                self, state, &piece, owner, from, actions, &KING, true, false,
            ),
            PieceKind::Parrot => {}
            PieceKind::Pawn | PieceKind::Squire | PieceKind::StandardBearer => {
                if let Some(to) = from.offset(0, dir)
                    && state.board.get(to).is_none()
                {
                    actions.push(Action::Move(MoveAction {
                        from,
                        to,
                        promotion: None,
                    }));
                }
                for df in [-1i8, 1] {
                    let Some(to) = from.offset(df, dir) else {
                        continue;
                    };
                    if let Some(target) = state.board.get(to)
                        && target.owner != owner
                        && self.can_capture(state, piece, target)
                    {
                        actions.push(Action::Move(MoveAction {
                            from,
                            to,
                            promotion: None,
                        }));
                    }
                }
            }
            PieceKind::Grasshopper => {
                for &(df, dr) in KING.iter() {
                    let mut cursor = from;
                    while let Some(next) = cursor.offset(df, dr)
                        && state.board.get(next).is_none()
                    {
                        cursor = next;
                    }
                    if let Some(screen) = cursor.offset(df, dr)
                        && state.board.get(screen).is_some()
                        && let Some(to) = screen.offset(df, dr)
                        && state.board.get(to).is_none()
                    {
                        actions.push(Action::Move(MoveAction {
                            from,
                            to,
                            promotion: None,
                        }));
                    }
                }
            }
            // Hook/Brutus/Protestant/Thief/Pegasus/Cannon/Windmill/Checker/PrimeMinister/
            // Amazon/Cardinal/Eagle-style patterns have no Rust piece-kind equivalent or are
            // out of this unit's evidence table; unhandled kinds yield no base moves.
            _ => {}
        }
    }

    pub(super) fn refresh_submerged_pieces(&self, state: &mut GameState) {
        let revealed = state
            .card_runtime
            .submerged_piece_ids
            .iter()
            .copied()
            .filter(|id| {
                state
                    .board
                    .find_by_id(*id)
                    .is_none_or(|(square, piece)| has_adjacent_enemy(state, square, piece.owner))
            })
            .collect::<Vec<_>>();
        for id in revealed {
            state.card_runtime.submerged_piece_ids.remove(&id);
        }
    }

    pub(super) fn complete_octopus_turn(&self, state: &mut GameState, actor: Player) {
        let submerged = state
            .board
            .iter()
            .filter_map(|(square, piece)| {
                (piece.owner == actor
                    && self.piece_has_ability(state, piece, PieceKind::Octopus)
                    && !has_adjacent_enemy(state, square, actor))
                .then_some(piece.id)
            })
            .collect::<Vec<_>>();
        state.card_runtime.submerged_piece_ids.extend(submerged);
        self.refresh_submerged_pieces(state);
    }

    pub(super) fn middle_submerged_protects(&self, state: &GameState, target: Piece) -> bool {
        state.card_runtime.submerged_piece_ids.contains(&target.id)
            && state
                .board
                .find_by_id(target.id)
                .is_some_and(|(square, piece)| !has_adjacent_enemy(state, square, piece.owner))
    }

    pub(super) fn generate_socialism_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        let direction = if piece.owner == Player::White { 1 } else { -1 };
        if let Some(to) = from.offset(0, direction)
            && self.board_piece_at(state, to).is_none()
        {
            actions.push(Action::Move(MoveAction::new(from, to, None)));
        }
        if state.ruleset_id == crate::LATEST_RULESET_ID
            && self.piece_has_ability(state, piece, PieceKind::Missionary)
            && state.card_runtime.socialism[piece.owner.index()] == 0
        {
            return;
        }
        for file_delta in [-1, 1] {
            if let Some(to) = from.offset(file_delta, direction)
                && self
                    .board_piece_at(state, to)
                    .is_some_and(|target| self.can_capture(state, piece, target))
            {
                actions.push(Action::Move(MoveAction::new(from, to, None)));
            }
        }
    }

    pub(super) fn augment_middle_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        if self.piece_has_ability(state, piece, PieceKind::Slime) {
            return;
        }
        let owner = piece.owner.index();
        if piece.kind == PieceKind::Bishop
            && state.card_runtime.bishop_infiltration_due[owner].is_some()
        {
            for &(df, dr) in BISHOP_DIRECTIONS {
                let mut cursor = from;
                while let Some(to) = cursor.offset(df, dr) {
                    let hole = state.card_runtime.black_holes.contains(&to)
                        && self.board_piece_at(state, to).is_none()
                        && !state.card_runtime.barricades.contains(&to)
                        && state.card_runtime.crown_pos != Some(to)
                        && !self.rule_cell_collapsed(state, to)
                        && !state
                            .card_runtime
                            .pending_lobsters
                            .iter()
                            .any(|p| p.square == to)
                        && !state
                            .card_runtime
                            .pending_scarecrows
                            .iter()
                            .any(|p| p.square == to)
                        && !state
                            .card_runtime
                            .pending_portals
                            .iter()
                            .any(|p| p.squares.contains(&to))
                        && !state.card_runtime.portals.is_some_and(|p| p.contains(&to))
                        && !state
                            .card_runtime
                            .rule_portals
                            .is_some_and(|p| p.contains(&to));
                    if self.card_landing_open(state, to) || hole {
                        actions.push(Action::Move(MoveAction::new(from, to, None)));
                    }
                    cursor = to;
                }
            }
        }
        if state
            .card_runtime
            .basic_training_piece_ids
            .contains(&piece.id)
        {
            self.generate_socialism_moves(state, from, piece, actions);
        }
        if state
            .card_runtime
            .promotion_rush_piece_ids
            .contains(&piece.id)
        {
            self.generate_slider_moves(state, from, piece, QUEEN_DIRECTIONS, actions);
        }
        if piece.kind == PieceKind::Pawn {
            // Long en passant (live JS:4032-4051): an own pawn standing on the rank the
            // just-double-stepped victim departed from may land on the en-passant target
            // from any file when the card is armed.
            if state.card_runtime.long_en_passant[owner]
                && let Some(en_passant) = &state.en_passant
                && en_passant.pawn_owner != piece.owner
                && en_passant.captured.rank() == from.rank()
            {
                let landing = en_passant.target;
                if state.board.get(landing).is_none() {
                    actions.push(Action::Move(MoveAction::new(from, landing, None)));
                }
            }
            // Pawn Leap vaults over an adjacent enemy piece and lands two squares
            // forward; it never captures that way (current contract, replacing the
            // old knight-offset behavior). The historical rulesets keep the old
            // knight-offset leap through the legacy path.
            if state.card_runtime.pawn_leap[owner] {
                if state.ruleset_id == crate::LATEST_RULESET_ID {
                    for to in self.leap_destinations(state, piece, from) {
                        actions.push(Action::Move(MoveAction::new(from, to, None)));
                    }
                } else {
                    self.generate_leaper_moves(state, from, piece, KNIGHT_OFFSETS, actions);
                }
            }
            if state.card_runtime.retreat[owner] {
                let backward = if piece.owner == Player::White { -1 } else { 1 };
                if let Some(to) = from.offset(0, backward)
                    && self.board_piece_at(state, to).is_none()
                {
                    actions.push(Action::Move(MoveAction::new(from, to, None)));
                }
            }
            if state.card_runtime.breakthrough_pawns[owner] {
                let direction = if piece.owner == Player::White { 1 } else { -1 };
                let mut cursor = from;
                while let Some(to) = cursor.offset(0, direction) {
                    if self.board_piece_at(state, to).is_some() {
                        break;
                    }
                    actions.push(Action::Move(MoveAction::new(from, to, None)));
                    cursor = to;
                }
            }
            if state.card_runtime.en_passant_frenzy[owner] {
                let direction = if piece.owner == Player::White { 1 } else { -1 };
                for file_delta in [-1, 1] {
                    let Some(adjacent) = from.offset(file_delta, 0) else {
                        continue;
                    };
                    let Some(target_piece) = self.board_piece_at(state, adjacent) else {
                        continue;
                    };
                    let Some(to) = from.offset(file_delta, direction) else {
                        continue;
                    };
                    if self.board_piece_at(state, to).is_none()
                        && (state.ruleset_id != crate::LATEST_RULESET_ID
                            || target_piece.owner != piece.owner)
                        && self.can_capture(state, piece, target_piece)
                    {
                        actions.push(Action::Move(MoveAction::new(from, to, None)));
                    }
                }
            }
        }
        if self.is_royal_piece(state, piece) && state.card_runtime.horse_riding[owner] {
            self.generate_leaper_moves(state, from, piece, KNIGHT_OFFSETS, actions);
        }
        if piece.kind == PieceKind::Knight
            && state.card_runtime.corner_kick[owner]
            && matches!(from.file(), 0 | 7)
            && matches!(from.rank(), 0 | 7)
        {
            self.generate_slider_moves(state, from, piece, BISHOP_DIRECTIONS, actions);
        }
        if state.card_runtime.royal_command[owner].is_some_and(|status| {
            let turn = state.turns_taken[owner];
            turn >= status.starts_at_turn && turn < status.expires_at_turn
        }) {
            self.generate_leaper_moves(state, from, piece, KING_OFFSETS, actions);
        }
        if let Some(kinds) = state.card_runtime.imperial_moves.get(&piece.id) {
            for kind in kinds {
                self.generate_middle_kind_moves(state, from, piece, *kind, actions);
            }
        }
        if piece.kind == PieceKind::Knight && state.card_runtime.mad_horse[owner] {
            for &(file, rank) in KNIGHT_OFFSETS {
                if let Some(to) = from.offset(file, rank)
                    && self.board_piece_at(state, to).is_some_and(|target| {
                        target.owner == piece.owner
                            && !self.is_royal_piece(state, target)
                            && (state.ruleset_id != crate::LATEST_RULESET_ID
                                || (self.capture_status_allows(state, piece)
                                    && !matches!(
                                        target.kind,
                                        PieceKind::Colossus
                                            | PieceKind::BigRook
                                            | PieceKind::BigBishop
                                            | PieceKind::Guard
                                            | PieceKind::Jester
                                    )
                                    && !state.card_runtime.shielded_piece_ids.contains(&target.id)
                                    && !state
                                        .card_runtime
                                        .protected_piece_ids
                                        .contains(&target.id)
                                    && !state
                                        .card_runtime
                                        .metalized_piece_ids
                                        .contains(&target.id)
                                    && !state
                                        .card_runtime
                                        .vigilance_protection
                                        .contains_key(&target.id)
                                    && !state
                                        .card_runtime
                                        .outpost_protected_piece_ids
                                        .contains(&target.id)
                                    && !state.card_runtime.frozen.contains_key(&target.id)
                                    && !state.rule_runtime.winter_frozen.contains(&target.id)
                                    && !self.middle_encouragement_protects(state, target)
                                    && !self.campfire_protects(state, target)))
                    })
                {
                    actions.push(Action::Move(MoveAction::new(from, to, None)));
                }
            }
        }
        if state.card_runtime.substitution[owner] {
            for (to, target) in state.board.iter() {
                if target.owner != piece.owner && target.kind == piece.kind {
                    actions.push(Action::Move(MoveAction::new(from, to, None)));
                }
            }
        }
        if self.is_royal_piece(state, piece) && state.card_runtime.switcheroo[owner] {
            for (to, target) in state.board.iter() {
                let allowed = state.ruleset_id != crate::LATEST_RULESET_ID
                    || (!state.rule_runtime.grappler_bound.contains_key(&target.id)
                        && !self.d4_restricts(state, piece.owner, false, from)
                        && state
                            .card_runtime
                            .october
                            .pawn_chain_links
                            .iter()
                            .all(|pair| {
                                let [a, b] = *pair;
                                let Some((a_pos, _)) = state.board.find_by_id(a) else {
                                    return true;
                                };
                                let Some((b_pos, _)) = state.board.find_by_id(b) else {
                                    return true;
                                };
                                let distance = |x: Square, y: Square| {
                                    x.file().abs_diff(y.file()).max(x.rank().abs_diff(y.rank()))
                                };
                                let new_pos = |id, square| {
                                    if id == piece.id {
                                        to
                                    } else if id == target.id {
                                        from
                                    } else {
                                        square
                                    }
                                };
                                distance(a_pos, b_pos) > 2
                                    || distance(new_pos(a, a_pos), new_pos(b, b_pos)) <= 2
                            }));
                if target.owner == piece.owner && target.kind == PieceKind::Pawn && allowed {
                    actions.push(Action::Move(MoveAction::new(from, to, None)));
                }
            }
        }
        if piece.kind == PieceKind::Bishop && state.card_runtime.bishop_snipe[owner] {
            let directions = if state.ruleset_id == crate::LATEST_RULESET_ID
                && state.card_runtime.reversal[owner]
            {
                ROOK_DIRECTIONS
            } else {
                BISHOP_DIRECTIONS
            };
            for &(file, rank) in directions {
                let mut cursor = from;
                let mut hurdle = false;
                while let Some(to) = cursor.offset(file, rank) {
                    if let Some(target) = self.board_piece_at(state, to) {
                        if hurdle {
                            if self.can_capture(state, piece, target) {
                                actions.push(Action::Move(MoveAction::new(from, to, None)));
                            }
                            break;
                        }
                        if state.ruleset_id == crate::LATEST_RULESET_ID
                            && target.kind != PieceKind::Pawn
                        {
                            break;
                        }
                        hurdle = true;
                    }
                    cursor = to;
                }
            }
        }
    }

    fn generate_middle_kind_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        kind: PieceKind,
        actions: &mut Vec<Action>,
    ) {
        match kind {
            PieceKind::Pawn => self.generate_socialism_moves(state, from, piece, actions),
            PieceKind::Knight => {
                self.generate_leaper_moves(state, from, piece, KNIGHT_OFFSETS, actions)
            }
            PieceKind::Bishop => {
                self.generate_slider_moves(state, from, piece, BISHOP_DIRECTIONS, actions)
            }
            PieceKind::Rook => {
                self.generate_slider_moves(state, from, piece, ROOK_DIRECTIONS, actions)
            }
            PieceKind::Queen => {
                self.generate_slider_moves(state, from, piece, QUEEN_DIRECTIONS, actions)
            }
            _ => {}
        }
    }

    fn paladin_radiance_contains(
        &self,
        state: &GameState,
        moving_owner: Player,
        square: Square,
    ) -> bool {
        state.board.iter().any(|(source, candidate)| {
            candidate.owner != moving_owner
                && self.piece_has_ability(state, candidate, PieceKind::Paladin)
                // JS lightSquare(row,col) uses row zero at rank 8; in Rust's
                // rank-one origin this is the opposite parity.
                && (source.file() + source.rank()) % 2 == 1
                && KING_OFFSETS.iter().any(|&(df, dr)| source.offset(df, dr) == Some(square))
        })
    }

    pub(super) fn paladin_radiance_blocks_move(
        &self,
        state: &GameState,
        piece: Piece,
        movement: MoveAction,
        check_path: bool,
    ) -> bool {
        if (movement.from.file() + movement.from.rank()) % 2 == 1 {
            return false;
        }
        if self.paladin_radiance_contains(state, piece.owner, movement.to) {
            return true;
        }
        if !check_path {
            return false;
        }
        let kind = self.piece_ability_kind(state, piece);
        if matches!(
            kind,
            PieceKind::Knight
                | PieceKind::Paladin
                | PieceKind::Camel
                | PieceKind::Alfil
                | PieceKind::Alibaba
                | PieceKind::Lobster
                | PieceKind::Slime
                | PieceKind::Colossus
                | PieceKind::SiegeRam
        ) {
            return false;
        }
        let df = movement.to.file() as i8 - movement.from.file() as i8;
        let dr = movement.to.rank() as i8 - movement.from.rank() as i8;
        if df != 0 && dr != 0 && df.abs() != dr.abs() {
            return false;
        }
        let steps = df.abs().max(dr.abs());
        let step_file = df.signum();
        let step_rank = dr.signum();
        (1..steps).any(|step| {
            movement
                .from
                .offset(step_file * step, step_rank * step)
                .is_some_and(|square| self.paladin_radiance_contains(state, piece.owner, square))
        })
    }

    pub(super) fn filter_middle_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        let owner = piece.owner.index();
        actions.retain(|action| {
            let Action::Move(movement) = action else {
                return true;
            };
            if self.paladin_radiance_blocks_move(state, piece, *movement, true) {
                return false;
            }
            if state
                .card_runtime
                .pending_scarecrows
                .iter()
                .any(|pending| pending.square == movement.to)
            {
                return false;
            }
            if self.fianchetto_destination_blocked(state, piece, from, movement.to) {
                return false;
            }
            let file_distance = movement.from.file().abs_diff(movement.to.file());
            let rank_distance = movement.from.rank().abs_diff(movement.to.rank());
            if piece.kind == PieceKind::DarkWizard
                && state
                    .card_runtime
                    .piece_origins
                    .get(&piece.id)
                    .is_some_and(|origin| {
                        movement.to.file().abs_diff(origin.file()) > 1
                            || movement.to.rank().abs_diff(origin.rank()) > 1
                    })
            {
                return false;
            }
            if state.card_runtime.inertia_piece_ids.contains(&piece.id)
                && file_distance <= 1
                && rank_distance <= 1
            {
                return false;
            }
            if state.card_runtime.severed.contains_key(&piece.id)
                && (file_distance > 1 || rank_distance > 1)
            {
                return false;
            }
            if state.card_runtime.taunt[owner] > 0 {
                let backwards = if piece.owner == Player::White {
                    movement.to.rank() < from.rank()
                } else {
                    movement.to.rank() > from.rank()
                };
                if backwards {
                    return false;
                }
            }
            if piece.kind == PieceKind::Knight && state.card_runtime.knight_injury[owner] {
                let file_delta = movement.to.file() as i8 - from.file() as i8;
                let rank_delta = movement.to.rank() as i8 - from.rank() as i8;
                if file_delta.unsigned_abs() == 2 && rank_delta.unsigned_abs() == 1 {
                    if from
                        .offset(file_delta.signum(), 0)
                        .is_some_and(|middle| self.board_piece_at(state, middle).is_some())
                    {
                        return false;
                    }
                } else if rank_delta.unsigned_abs() == 2
                    && file_delta.unsigned_abs() == 1
                    && from
                        .offset(0, rank_delta.signum())
                        .is_some_and(|middle| self.board_piece_at(state, middle).is_some())
                {
                    return false;
                }
            }
            true
        });

        if state.card_runtime.ice_sheet.contains_key(&piece.id) {
            let mut maximum = std::collections::BTreeMap::<(i8, i8), u8>::new();
            for action in actions.iter() {
                let Action::Move(movement) = action else {
                    continue;
                };
                let file = movement.to.file() as i8 - from.file() as i8;
                let rank = movement.to.rank() as i8 - from.rank() as i8;
                let direction = (file.signum(), rank.signum());
                maximum
                    .entry(direction)
                    .and_modify(|distance| {
                        *distance = (*distance).max(file.unsigned_abs().max(rank.unsigned_abs()))
                    })
                    .or_insert(file.unsigned_abs().max(rank.unsigned_abs()));
            }
            actions.retain(|action| {
                let Action::Move(movement) = action else {
                    return true;
                };
                let file = movement.to.file() as i8 - from.file() as i8;
                let rank = movement.to.rank() as i8 - from.rank() as i8;
                maximum
                    .get(&(file.signum(), rank.signum()))
                    .is_some_and(|maximum| *maximum == file.unsigned_abs().max(rank.unsigned_abs()))
            });
        }
    }

    pub(super) fn middle_evasion_destinations(
        &self,
        state: &GameState,
        origin: Square,
        attacker_landing: Square,
    ) -> Vec<Square> {
        KING_OFFSETS
            .iter()
            .filter_map(|(file, rank)| origin.offset(*file, *rank))
            .filter(|square| {
                *square != attacker_landing && self.board_piece_at(state, *square).is_none()
            })
            .collect()
    }

    pub(super) fn middle_capture_context(
        &self,
        state: &GameState,
        target: Piece,
    ) -> MiddleCaptureContext {
        MiddleCaptureContext {
            poisoned: state.card_runtime.poisoned_piece_ids.contains(&target.id),
            explosive: state.card_runtime.explosive_piece_ids.contains(&target.id),
            trojan: state.card_runtime.trojan_piece_ids.contains(&target.id),
            feudal_guardian_id: state
                .card_runtime
                .feudal_contracts
                .iter()
                .find(|contract| contract.pawn_id == target.id)
                .map(|contract| contract.guardian_id),
        }
    }

    pub(super) fn note_middle_piece_moved(&self, state: &mut GameState, piece: Piece) {
        let owner = piece.owner.index();
        if !state.card_runtime.exhaustion[owner] {
            return;
        }
        if state.card_runtime.exhaustion_last_piece[owner] == Some(piece.id) {
            state.card_runtime.exhaustion_count[owner] =
                state.card_runtime.exhaustion_count[owner].saturating_add(1);
        } else {
            state.card_runtime.exhaustion_last_piece[owner] = Some(piece.id);
            state.card_runtime.exhaustion_count[owner] = 1;
        }
    }

    pub(super) fn resolve_middle_after_move(
        &self,
        state: &mut GameState,
        original_piece: Piece,
        movement: MoveAction,
        captured: Option<(Square, Piece)>,
        context: Option<MiddleCaptureContext>,
        events: &mut Vec<Event>,
    ) -> bool {
        self.note_middle_piece_moved(state, original_piece);
        if state.ruleset_id == crate::LATEST_RULESET_ID {
            state
                .card_runtime
                .promotion_rush_piece_ids
                .remove(&original_piece.id);
            state
                .card_runtime
                .promotion_rush_capture_cleared
                .remove(&original_piece.id);
        }
        state
            .card_runtime
            .quantum_positions
            .remove(&original_piece.id);
        let owner = original_piece.owner;
        let captured_piece = captured.map(|(_, piece)| piece);

        if let Some(captured_piece) = captured_piece {
            if state
                .card_runtime
                .chameleon_piece_ids
                .contains(&original_piece.id)
                && let Some(mut attacker) = state.board.get(movement.to)
                && attacker.id == original_piece.id
                && (state.ruleset_id != crate::LATEST_RULESET_ID
                    || (!self.is_royal_piece(state, captured_piece)
                        && !matches!(
                            captured_piece.kind,
                            PieceKind::Colossus | PieceKind::BigRook | PieceKind::BigBishop
                        )))
            {
                if state.ruleset_id != crate::LATEST_RULESET_ID {
                    self.clear_queens_gambit_protection(state, attacker.id);
                }
                attacker.kind = captured_piece.kind;
                state.board.set(movement.to, Some(attacker));
            }
            if self.is_royal_piece(state, original_piece)
                && state.card_runtime.imperial_studies[owner.index()]
            {
                state
                    .card_runtime
                    .imperial_moves
                    .entry(original_piece.id)
                    .or_default()
                    .insert(captured_piece.kind);
            }
            if original_piece.kind == PieceKind::Queen
                && captured_piece.kind == PieceKind::Queen
                && state.card_runtime.queen_afterimage[owner.index()]
                && state.board.get(movement.from).is_none()
                && !self.d4_restricts(state, owner, false, movement.from)
            {
                let afterimage =
                    Piece::new(owner, PieceKind::Queen).with_id(state.allocate_piece_id());
                state.board.set(movement.from, Some(afterimage));
                self.mark_fresh_no_capture(state, afterimage);
                events.push(Event::EffectApplied {
                    source_card_instance: 0,
                    command: format!("queen-afterimage:{}:{}", afterimage.id, movement.from),
                });
            }
            if state
                .card_runtime
                .witch_trials
                .remove(&original_piece.id)
                .is_some()
            {
                state
                    .card_runtime
                    .shielded_piece_ids
                    .insert(original_piece.id);
            }
        }

        if state.ruleset_id == crate::LATEST_RULESET_ID
            && state
                .card_runtime
                .chimera_piece_ids
                .contains(&original_piece.id)
            && movement.from != movement.to
        {
            self.resolve_current_chimera(state, original_piece.id, events);
        } else if state.ruleset_id != crate::LATEST_RULESET_ID
            && state
                .card_runtime
                .chimera_piece_ids
                .contains(&original_piece.id)
            && let Some(mut chimera) = state.board.get(movement.to)
        {
            let (roll, selected) = state.rng_mut().web_choice_index(25);
            events.push(Event::RandomSample {
                purpose: "chimera".to_owned(),
                roll_53: roll,
            });
            let selected = selected.unwrap_or(0);
            chimera.kind = if selected == 24 {
                PieceKind::Queen
            } else {
                [
                    PieceKind::Pawn,
                    PieceKind::Knight,
                    PieceKind::Bishop,
                    PieceKind::Rook,
                ][selected % 4]
            };
            state.board.set(movement.to, Some(chimera));
        }

        if state.card_runtime.quantum_pending[owner.index()]
            && captured.is_none()
            && let Some(attacker) = state.board.get(movement.to)
        {
            let candidates = (0..64)
                .filter_map(Square::from_index)
                .filter(|square| state.board.get(*square).is_none())
                .collect::<Vec<_>>();
            let (roll, selected) = state.rng_mut().web_choice_index(candidates.len());
            events.push(Event::RandomSample {
                purpose: "quantum-mechanics".to_owned(),
                roll_53: roll,
            });
            if let Some(selected) = selected {
                state
                    .card_runtime
                    .quantum_positions
                    .insert(attacker.id, candidates[selected]);
            }
            state.card_runtime.quantum_pending[owner.index()] = false;
        }

        if original_piece.kind == PieceKind::Knight
            && state.card_runtime.radical_charge[owner.index()]
        {
            let file = movement.to.file() as i8 - movement.from.file() as i8;
            let rank = movement.to.rank() as i8 - movement.from.rank() as i8;
            let jumped = if file.unsigned_abs() == 2 {
                movement.from.offset(file.signum(), 0)
            } else if rank.unsigned_abs() == 2 {
                movement.from.offset(0, rank.signum())
            } else {
                None
            };
            if let Some(square) = jumped
                && let Some(target) = state.board.get(square)
                && target.owner != owner
            {
                self.remove_piece_for_card(state, square, target, owner, RADICAL_CHARGE, events);
            }
        }

        if let Some(context) = context {
            if context.poisoned
                && state.ruleset_id != crate::LATEST_RULESET_ID
                && let Some(attacker) = state.board.get(movement.to)
            {
                state.card_runtime.frozen.insert(
                    attacker.id,
                    FrozenStatus {
                        source: captured_piece.map_or(owner.opponent(), |piece| piece.owner),
                        remaining_turns: 3,
                    },
                );
            }
            if context.explosive {
                let center = captured.map_or(movement.to, |(square, _)| square);
                let victims = state
                    .board
                    .iter()
                    .filter(|(square, _)| {
                        square.file().abs_diff(center.file()) <= 1
                            && square.rank().abs_diff(center.rank()) <= 1
                    })
                    .filter(|(_, piece)| {
                        if state.ruleset_id == crate::LATEST_RULESET_ID {
                            let permanent =
                                state.card_runtime.protected_piece_ids.contains(&piece.id)
                                    && (state
                                        .card_runtime
                                        .queens_gambit_protection
                                        .contains(&piece.id)
                                        || !has_piece_protection_source(state, piece.id)
                                        || state
                                            .card_runtime
                                            .queens_gambit_previous_protection
                                            .contains(&piece.id));
                            !permanent
                                && !state
                                    .card_runtime
                                    .vigilance_protection
                                    .contains_key(&piece.id)
                                && piece.kind != PieceKind::Colossus
                        } else {
                            !state.card_runtime.frozen.contains_key(&piece.id)
                                && !matches!(
                                    piece.kind,
                                    PieceKind::Guard | PieceKind::Jester | PieceKind::Monster
                                )
                        }
                    })
                    .collect::<Vec<_>>();
                for (square, victim) in victims {
                    self.remove_piece_for_card(
                        state,
                        square,
                        victim,
                        if state.ruleset_id == crate::LATEST_RULESET_ID {
                            captured_piece.map_or(owner.opponent(), |p| p.owner)
                        } else {
                            owner
                        },
                        SUICIDE_BOMBER,
                        events,
                    );
                }
            }
            if context.trojan
                && let Some(attacker) = state.board.find_by_id(original_piece.id)
            {
                self.remove_piece_for_card(
                    state,
                    attacker.0,
                    attacker.1,
                    captured_piece.map_or(owner.opponent(), |piece| piece.owner),
                    TROJAN_HORSE,
                    events,
                );
                if state.board.get(movement.to).is_none() {
                    let pawn_owner = captured_piece.map_or(owner.opponent(), |piece| piece.owner);
                    let pawn =
                        Piece::new(pawn_owner, PieceKind::Pawn).with_id(state.allocate_piece_id());
                    state.board.set(movement.to, Some(pawn));
                    self.mark_fresh_no_capture(state, pawn);
                }
            }
            if let Some(guardian_id) = context.feudal_guardian_id
                && let Some((guardian_square, mut guardian)) = state.board.find_by_id(guardian_id)
                && let Some((attacker_square, attacker)) = state.board.find_by_id(original_piece.id)
            {
                state.board.set(guardian_square, None);
                self.remove_piece_for_card(
                    state,
                    attacker_square,
                    attacker,
                    guardian.owner,
                    FEUDAL_CONTRACT,
                    events,
                );
                guardian.capture_count = guardian.capture_count.saturating_add(1);
                state.board.set(attacker_square, Some(guardian));
                state.card_runtime.moved_piece_ids.insert(guardian.id);
            }
        }

        if state.ruleset_id != crate::LATEST_RULESET_ID
            && let Some(bond) = state
                .card_runtime
                .twin_bonds
                .iter()
                .find(|bond| {
                    bond.first_id == original_piece.id || bond.second_id == original_piece.id
                })
                .copied()
        {
            let partner_id = if bond.first_id == original_piece.id {
                bond.second_id
            } else {
                bond.first_id
            };
            if let Some((partner_square, partner)) = state.board.find_by_id(partner_id)
                && let Some(moved) = state.board.get(movement.to)
                && !self.piece_has_ability(state, partner, PieceKind::Slime)
                && !self.piece_has_ability(state, moved, PieceKind::Slime)
            {
                state.board.set(partner_square, Some(moved));
                state.board.set(movement.to, Some(partner));
            }
        }

        let captured_backward = captured_piece.is_some()
            && original_piece.kind == PieceKind::Knight
            && state.card_runtime.backward_knight[owner.index()]
            && if owner == Player::White {
                movement.to.rank() < movement.from.rank()
            } else {
                movement.to.rank() > movement.from.rank()
            };
        let captured_friend = captured_piece.is_some_and(|piece| piece.owner == owner);
        let mut rook_lift = original_piece.kind == PieceKind::Rook
            && state.card_runtime.rook_lift[owner.index()]
            && matches!(
                (movement.to.file(), movement.to.rank()),
                (0, 0) | (0, 7) | (7, 0) | (7, 7)
            );
        if state.ruleset_id == crate::LATEST_RULESET_ID {
            let count = state
                .card_runtime
                .october
                .rook_lift_chain
                .filter(|(id, _)| *id == original_piece.id)
                .map_or(0, |(_, count)| count);
            rook_lift &= captured_piece.is_none() && count < 3;
            state.card_runtime.october.rook_lift_chain =
                rook_lift.then_some((original_piece.id, count.saturating_add(1)));
        }
        let extra_move = captured_backward
            || (original_piece.kind == PieceKind::Knight
                && state.card_runtime.file_surge[owner.index()]
                && matches!(movement.to.file(), 0 | 7))
            || rook_lift
            || (original_piece.kind == PieceKind::Knight
                && state.card_runtime.mad_horse[owner.index()]
                && captured_friend);
        if state.ruleset_id == crate::LATEST_RULESET_ID
            && state
                .card_runtime
                .twin_bonds
                .iter()
                .any(|b| b.first_id == original_piece.id || b.second_id == original_piece.id)
        {
            let file_surge = self.piece_ability_kind(state, original_piece) == PieceKind::Knight
                && state.card_runtime.file_surge[owner.index()]
                && state
                    .board
                    .find_by_id(original_piece.id)
                    .is_some_and(|(s, _)| matches!(s.file(), 0 | 7));
            if state
                .card_runtime
                .twin_swap_pending
                .contains(&original_piece.id)
                || file_surge
            {
                state
                    .card_runtime
                    .twin_swap_pending
                    .insert(original_piece.id);
            } else {
                self.apply_twin_swap(state, original_piece.id);
            }
        }
        if extra_move && state.board.find_by_id(original_piece.id).is_some() {
            state.phase = GamePhase::ForcedMove {
                piece_id: original_piece.id,
                source: ForcedMoveSource::Effect(MIDDLE_EXTRA_MOVE_SOURCE.to_owned()),
                continuation: ForcedMoveContinuation::FinishTurn,
            };
            events.push(Event::ForcedMoveStarted {
                player: owner,
                piece_id: original_piece.id,
                source: MIDDLE_EXTRA_MOVE_SOURCE.to_owned(),
            });
        }
        self.resolve_religious_victory(state);
        state.board.find_by_id(original_piece.id).is_some()
    }

    fn apply_twin_swap(&self, state: &mut GameState, id: u32) {
        let Some(bond) = state
            .card_runtime
            .twin_bonds
            .iter()
            .find(|b| b.first_id == id || b.second_id == id)
            .copied()
        else {
            return;
        };
        let partner_id = if bond.first_id == id {
            bond.second_id
        } else {
            bond.first_id
        };
        let (Some((from, piece)), Some((to, partner))) = (
            state.board.find_by_id(id),
            state.board.find_by_id(partner_id),
        ) else {
            return;
        };
        if self.piece_has_ability(state, piece, PieceKind::Slime)
            || self.piece_has_ability(state, partner, PieceKind::Slime)
        {
            state
                .card_runtime
                .twin_bonds
                .retain(|b| b.first_id != id && b.second_id != id);
            state.card_runtime.twin_swap_pending.remove(&id);
            state.card_runtime.twin_swap_pending.remove(&partner_id);
            return;
        }
        let ep = state
            .en_passant
            .and_then(|e| state.board.get(e.captured).map(|p| (e.captured, p.id)));
        state.board.set(from, Some(partner));
        state.board.set(to, Some(piece));
        state.card_runtime.moved_piece_ids.extend([id, partner_id]);
        if ep.is_some_and(|(square, victim)| state.board.get(square).is_none_or(|p| p.id != victim))
        {
            state.en_passant = None;
        }
        state.card_runtime.twin_swap_pending.remove(&id);
        state.card_runtime.twin_swap_pending.remove(&partner_id);
    }

    pub(super) fn settle_twin_sequences(&self, state: &mut GameState) {
        let pending: Vec<_> = state
            .card_runtime
            .twin_swap_pending
            .iter()
            .copied()
            .collect();
        for id in pending {
            if matches!(state.phase, GamePhase::ForcedMove { piece_id, .. } if piece_id == id) {
                continue;
            }
            state.card_runtime.twin_swap_pending.remove(&id);
            self.apply_twin_swap(state, id);
        }
    }

    pub(super) fn tick_middle_statuses_after_turn(
        &self,
        state: &mut GameState,
        actor: Player,
        events: &mut Vec<Event>,
    ) {
        let owner = actor.index();
        state.card_runtime.evacuation_no_capture.retain(|id| {
            state
                .board
                .find_by_id(*id)
                .is_some_and(|(_, piece)| piece.owner != actor)
        });
        state.card_runtime.vigilance_protection.retain(|_, status| {
            if status.by == actor {
                status.remaining_turns = status.remaining_turns.saturating_sub(1);
            }
            status.remaining_turns > 0
        });
        if state.ruleset_id == crate::LATEST_RULESET_ID {
            state.card_runtime.promotion_rush_piece_ids.retain(|id| {
                state
                    .board
                    .find_by_id(*id)
                    .is_some_and(|(_, p)| p.owner != actor)
            });
            state
                .card_runtime
                .promotion_rush_capture_cleared
                .retain(|id| state.card_runtime.promotion_rush_piece_ids.contains(id));
        }
        state.card_runtime.counterattack_locked.retain(|id| {
            state
                .board
                .find_by_id(*id)
                .is_some_and(|(_, p)| p.owner != actor)
        });
        state.card_runtime.breakthrough_pawns[owner] = false;
        state.card_runtime.en_passant_frenzy[owner] = false;
        state.card_runtime.bishop_snipe[owner] = false;
        state.card_runtime.quantum_pending[owner] = false;
        state.card_runtime.substitution[owner] = false;
        state.card_runtime.overtake[owner] = false;
        state.card_runtime.disassembly[owner] = false;
        state.card_runtime.symmetry[owner] = false;
        state.card_runtime.e4_active[owner] = false;
        state.card_runtime.roller_active[owner] = false;
        state.card_runtime.solidarity_active[owner] = false;
        state.card_runtime.switcheroo[owner] = false;
        state.card_runtime.reposition_ready[owner] = false;
        state.card_runtime.mistake_card[owner] = false;
        state.card_runtime.socialism[owner] = state.card_runtime.socialism[owner].saturating_sub(1);
        state.card_runtime.taunt[owner] = state.card_runtime.taunt[owner].saturating_sub(1);
        state.card_runtime.armistice_remaining_ply =
            state.card_runtime.armistice_remaining_ply.saturating_sub(1);
        state.card_runtime.stealth_piece_ids.retain(|piece_id| {
            state
                .board
                .find_by_id(*piece_id)
                .is_some_and(|(_, piece)| piece.owner == actor)
        });

        if state.card_runtime.royal_command[owner]
            .is_some_and(|status| state.turns_taken[owner] >= status.expires_at_turn)
        {
            state.card_runtime.royal_command[owner] = None;
        }

        let timed_ids = |entries: &std::collections::BTreeMap<u32, TimedPieceStatus>,
                         state: &GameState| {
            entries
                .keys()
                .copied()
                .filter(|piece_id| {
                    state
                        .board
                        .find_by_id(*piece_id)
                        .is_some_and(|(_, piece)| piece.owner == actor)
                })
                .collect::<Vec<_>>()
        };
        for piece_id in timed_ids(&state.card_runtime.ice_sheet, state) {
            if let Some(status) = state.card_runtime.ice_sheet.get_mut(&piece_id) {
                status.remaining_turns = status.remaining_turns.saturating_sub(1);
                if status.remaining_turns == 0 {
                    state.card_runtime.ice_sheet.remove(&piece_id);
                }
            }
        }
        for piece_id in timed_ids(&state.card_runtime.severed, state) {
            if let Some(status) = state.card_runtime.severed.get_mut(&piece_id) {
                status.remaining_turns = status.remaining_turns.saturating_sub(1);
                if status.remaining_turns == 0 {
                    state.card_runtime.severed.remove(&piece_id);
                }
            }
        }
        for piece_id in timed_ids(&state.card_runtime.sacrifice_protection, state) {
            if let Some(status) = state.card_runtime.sacrifice_protection.get_mut(&piece_id) {
                status.remaining_turns = status.remaining_turns.saturating_sub(1);
                if status.remaining_turns == 0 {
                    state.card_runtime.sacrifice_protection.remove(&piece_id);
                    if !has_piece_protection_source(state, piece_id) {
                        state.card_runtime.protected_piece_ids.remove(&piece_id);
                    }
                }
            }
        }
        for piece_id in timed_ids(&state.card_runtime.last_resistance, state) {
            if let Some(status) = state.card_runtime.last_resistance.get_mut(&piece_id) {
                status.remaining_turns = status.remaining_turns.saturating_sub(1);
                if status.remaining_turns == 0 {
                    state.card_runtime.last_resistance.remove(&piece_id);
                    if !has_piece_protection_source(state, piece_id) {
                        state.card_runtime.protected_piece_ids.remove(&piece_id);
                    }
                }
            }
        }
        let witch_ids = timed_ids(&state.card_runtime.witch_trials, state);
        for piece_id in witch_ids {
            let expires = if let Some(status) = state.card_runtime.witch_trials.get_mut(&piece_id) {
                status.remaining_turns = status.remaining_turns.saturating_sub(1);
                status.remaining_turns == 0
            } else {
                false
            };
            if expires {
                state.card_runtime.witch_trials.remove(&piece_id);
                if let Some((square, piece)) = state.board.find_by_id(piece_id) {
                    self.remove_piece_for_card(
                        state,
                        square,
                        piece,
                        piece.owner.opponent(),
                        WITCH_TRIAL,
                        events,
                    );
                }
            }
        }

        let pending = std::mem::take(&mut state.card_runtime.pending_scarecrows);
        for mut scarecrow in pending {
            scarecrow.remaining_ply = scarecrow.remaining_ply.saturating_sub(1);
            if scarecrow.remaining_ply > 0 {
                state.card_runtime.pending_scarecrows.push(scarecrow);
            } else if state.board.get(scarecrow.square).is_none()
                && !self.d4_restricts(state, scarecrow.owner, false, scarecrow.square)
            {
                let piece = Piece::new(scarecrow.owner, PieceKind::Scarecrow)
                    .with_id(state.allocate_piece_id());
                state.board.set(scarecrow.square, Some(piece));
                state.card_runtime.moved_piece_ids.insert(piece.id);
                events.push(Event::EffectApplied {
                    source_card_instance: 0,
                    command: format!("scarecrow-spawn:{}:{}", piece.id, scarecrow.square),
                });
            }
        }

        // Gale is an END card now: it removes major pieces with no other major in
        // the surrounding eight squares, so the card owns the settlement.
        self.settle_gale(state, actor, events);

        let grown = state
            .card_runtime
            .baby_bear_grow_turns
            .iter()
            .filter_map(|(piece_id, due)| (state.shared_turn_count() >= *due).then_some(*piece_id))
            .collect::<Vec<_>>();
        for piece_id in grown {
            state.card_runtime.baby_bear_grow_turns.remove(&piece_id);
            if let Some((square, mut piece)) = state.board.find_by_id(piece_id) {
                self.clear_queens_gambit_protection(state, piece.id);
                piece.kind = PieceKind::Bear;
                state.board.set(square, Some(piece));
                state.card_runtime.bear_retaliations.insert(piece.id, 2);
                events.push(Event::EffectApplied {
                    source_card_instance: 0,
                    command: format!("baby-bear-grow:{}:{}", piece.id, square),
                });
            }
        }
    }

    pub(super) fn resolve_middle_turn_start(
        &self,
        state: &mut GameState,
        player: Player,
        events: &mut Vec<Event>,
    ) {
        let expired_coronation = state
            .card_runtime
            .coronation_protection
            .iter()
            .filter_map(|(piece_id, start_turn)| {
                state
                    .board
                    .find_by_id(*piece_id)
                    .is_some_and(|(_, piece)| {
                        piece.owner == player && state.turns_taken[player.index()] > *start_turn
                    })
                    .then_some(*piece_id)
            })
            .collect::<Vec<_>>();
        for piece_id in expired_coronation {
            state.card_runtime.coronation_protection.remove(&piece_id);
            if !has_piece_protection_source(state, piece_id) {
                state.card_runtime.protected_piece_ids.remove(&piece_id);
            }
        }
        let revealed = state
            .card_runtime
            .submerged_piece_ids
            .iter()
            .copied()
            .filter(|piece_id| {
                state
                    .board
                    .find_by_id(*piece_id)
                    .is_none_or(|(square, piece)| {
                        piece.owner == player && has_adjacent_enemy(state, square, piece.owner)
                    })
            })
            .collect::<Vec<_>>();
        for piece_id in revealed {
            state.card_runtime.submerged_piece_ids.remove(&piece_id);
        }

        let pending = std::mem::take(&mut state.card_runtime.pending_panic);
        for panic in pending {
            if panic.owner != player {
                state.card_runtime.pending_panic.push(panic);
                continue;
            }
            for piece_id in panic.piece_ids {
                let candidates = self
                    .generate_piece_move_actions(state, piece_id)
                    .into_iter()
                    .filter_map(|action| match action {
                        Action::Move(movement)
                            if state.board.get(movement.to).is_none()
                                && movement.promotion.is_none() =>
                        {
                            Some(movement)
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let (roll, selected) = if state.ruleset_id == crate::LATEST_RULESET_ID {
                    let mut candidates = candidates;
                    if state.board.find_by_id(piece_id).is_none_or(|(_, piece)| {
                        piece.owner != player
                            || self.piece_has_ability(state, piece, PieceKind::Slime)
                    }) {
                        candidates.clear();
                    }
                    candidates.sort_by_key(|movement| (7 - movement.to.rank(), movement.to.file()));
                    candidates.dedup_by_key(|movement| movement.to);
                    let (roll, random) = state.rng_mut().web_choice_index(64);
                    let selected =
                        (!candidates.is_empty()).then(|| random.unwrap() % candidates.len());
                    (roll, selected.map(|index| candidates[index]))
                } else {
                    let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
                    (roll, index.map(|index| candidates[index]))
                };
                events.push(Event::RandomSample {
                    purpose: "panic".to_owned(),
                    roll_53: roll,
                });
                if let Some(movement) = selected {
                    if let Some(piece) = state.board.get(movement.from) {
                        state.board.set(movement.from, None);
                        state.board.set(movement.to, Some(piece));
                        state.card_runtime.moved_piece_ids.insert(piece.id);
                        events.push(Event::EffectApplied {
                            source_card_instance: 0,
                            command: format!(
                                "panic:{}:{}->{}",
                                piece.id, movement.from, movement.to
                            ),
                        });
                    }
                }
            }
        }

        if state.ruleset_id == crate::LATEST_RULESET_ID
            && self.open_current_trolley(state, player, events)
        {
            return;
        }
        if let Some(by) = state.card_runtime.pending_trolley_by[player.index()].take()
            && let Some(bundles) = self.sample_middle_trolley_bundle_pair(state, player, events)
        {
            let labels = bundles.each_ref().map(|bundle| {
                let pieces = bundle
                    .iter()
                    .filter_map(|piece_id| {
                        state
                            .board
                            .find_by_id(*piece_id)
                            .map(|(_, piece)| format!("{:?}", piece.kind))
                    })
                    .collect::<Vec<_>>();
                format!("희생: {}", pieces.join(" + "))
            });
            state.card_runtime.active_trolley = Some(TrolleyDilemma { by, bundles });
            let choice_id = state.allocate_choice_id();
            state.phase = GamePhase::AwaitingChoice(PendingChoice {
                id: choice_id,
                owner: player,
                source_card_instance: 0,
                handler_id: TROLLEY_CHOICE_HANDLER.to_owned(),
                options: labels.to_vec(),
                continuation: ChoiceContinuation::ResumeTurn,
            });
            events.push(Event::ChoiceRequested {
                choice_id,
                owner: player,
                handler_id: TROLLEY_CHOICE_HANDLER.to_owned(),
                options: labels.to_vec(),
            });
        }
    }

    pub(super) fn resolve_middle_trolley_choice(
        &self,
        state: &mut GameState,
        option_index: usize,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        let dilemma = state.card_runtime.active_trolley.take().ok_or_else(|| {
            EngineError::Invariant("trolley choice has no active dilemma".to_owned())
        })?;
        let doomed = dilemma
            .bundles
            .get(option_index)
            .ok_or_else(|| EngineError::Invariant("trolley choice index is invalid".to_owned()))?;
        let mut removed = false;
        for piece_id in doomed {
            if let Some((square, piece)) = state.board.find_by_id(*piece_id) {
                self.remove_piece_for_card(state, square, piece, dilemma.by, TROLLEY, events);
                removed = true;
            }
        }
        if state.ruleset_id == crate::LATEST_RULESET_ID && removed {
            self.cancel_prophecies(state);
        }
        self.resolve_missing_king(state, events);
        Ok(())
    }

    #[allow(dead_code)]
    pub(super) fn has_trolley_bundle_pair(&self, state: &GameState, player: Player) -> bool {
        let pieces = state
            .board
            .iter()
            .filter(|(_, piece)| piece.owner == player && !self.is_royal_piece(state, *piece))
            .map(|(_, piece)| (piece.id, middle_piece_value(piece.kind)))
            .collect::<Vec<_>>();
        let mut by_score = std::collections::BTreeMap::<u8, Vec<Vec<u32>>>::new();
        collect_trolley_bundles(&pieces, 0, &mut Vec::new(), 0, &mut by_score);
        for low_score in 2..=8 {
            for high_score in low_score..=u8::min(10, low_score + 2) {
                let left = by_score.get(&low_score).into_iter().flatten();
                let right = by_score
                    .get(&high_score)
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>();
                for first in left {
                    for second in &right {
                        if (low_score != high_score || first.as_ptr() != second.as_ptr())
                            && first.iter().all(|piece_id| !second.contains(piece_id))
                        {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    fn sample_middle_trolley_bundle_pair(
        &self,
        state: &mut GameState,
        player: Player,
        events: &mut Vec<Event>,
    ) -> Option<[Vec<u32>; 2]> {
        let pieces = state
            .board
            .iter()
            .filter(|(_, piece)| piece.owner == player && !self.is_royal_piece(state, *piece))
            .map(|(_, piece)| (piece.id, middle_piece_value(piece.kind)))
            .collect::<Vec<_>>();
        let mut by_score = std::collections::BTreeMap::<u8, Vec<Vec<u32>>>::new();
        collect_trolley_bundles(&pieces, 0, &mut Vec::new(), 0, &mut by_score);
        let mut pairs = Vec::new();
        for low_score in 2..=8 {
            for high_score in low_score..=u8::min(10, low_score + 2) {
                let left = by_score.get(&low_score).into_iter().flatten();
                let right = by_score
                    .get(&high_score)
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>();
                for first in left {
                    for second in &right {
                        if (low_score != high_score || first.as_ptr() != second.as_ptr())
                            && first.iter().all(|piece_id| !second.contains(piece_id))
                        {
                            pairs.push([first.clone(), (*second).clone()]);
                        }
                    }
                }
            }
        }
        if pairs.is_empty() {
            return None;
        }
        let (roll, index) = state.rng_mut().web_choice_index(pairs.len());
        events.push(Event::RandomSample {
            purpose: "trolley-bundles".to_owned(),
            roll_53: roll,
        });
        Some(pairs[index.unwrap_or(0)].clone())
    }

    pub(super) fn resolve_due_premoves_after_turn(
        &self,
        state: &mut GameState,
        actor: Player,
        events: &mut Vec<Event>,
    ) {
        let pending = std::mem::take(&mut state.card_runtime.pending_premoves);
        for premove in pending {
            if premove.trigger_player != actor
                || state.turns_taken[actor.index()] < premove.trigger_turn
            {
                state.card_runtime.pending_premoves.push(premove);
                continue;
            }
            let previous_turn = state.turn;
            state.turn = premove.owner;
            state.card_runtime.free_move_no_capture[premove.owner.index()] = true;
            for movement in premove.moves {
                let legal = state.board.get(movement.from).is_some_and(|piece| {
                    piece.owner == premove.owner
                        && self
                            .generate_piece_move_actions(state, piece.id)
                            .contains(&Action::Move(movement))
                });
                if !legal {
                    break;
                }
                let Some(mut piece) = state.board.get(movement.from) else {
                    break;
                };
                state.board.set(movement.from, None);
                if let Some(promotion) = movement.promotion {
                    piece.kind = promotion;
                }
                state.board.set(movement.to, Some(piece));
                state.card_runtime.moved_piece_ids.insert(piece.id);
                // Each premove is a relocation, not an atomic multi-move teleport.
                self.refresh_submerged_pieces(state);
                events.push(Event::EffectApplied {
                    source_card_instance: 0,
                    command: format!("premove:{}:{}->{}", piece.id, movement.from, movement.to),
                });
            }
            state.turn = previous_turn;
        }
    }

    pub(super) fn resolve_religious_victory(&self, state: &mut GameState) {
        if state.status != GameStatus::Ongoing {
            return;
        }
        for player in [Player::White, Player::Black] {
            if !state.card_runtime.religious_victory[player.index()] {
                continue;
            }
            let own = state
                .board
                .iter()
                .filter(|(_, piece)| {
                    piece.owner == player
                        && matches!(piece.kind, PieceKind::Bishop | PieceKind::Protestant)
                })
                .count();
            let enemy = state
                .board
                .iter()
                .filter(|(_, piece)| {
                    piece.owner == player.opponent()
                        && matches!(piece.kind, PieceKind::Bishop | PieceKind::Protestant)
                })
                .count();
            if own >= enemy.saturating_add(3) {
                state.status = GameStatus::Won {
                    winner: player,
                    reason: WinReason::CardEffect(RELIGIOUS_VICTORY.to_owned()),
                };
                break;
            }
        }
    }
}

const KNIGHT_OFFSETS: &[(i8, i8)] = &[
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
];
const BISHOP_DIRECTIONS: &[(i8, i8)] = &[(-1, -1), (-1, 1), (1, -1), (1, 1)];
const ROOK_DIRECTIONS: &[(i8, i8)] = &[(-1, 0), (1, 0), (0, -1), (0, 1)];
const QUEEN_DIRECTIONS: &[(i8, i8)] = &[
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
];

fn collect_trolley_bundles(
    pieces: &[(u32, u8)],
    start: usize,
    selected: &mut Vec<u32>,
    score: u8,
    by_score: &mut std::collections::BTreeMap<u8, Vec<Vec<u32>>>,
) {
    if !selected.is_empty() && (2..=10).contains(&score) {
        let bucket = by_score.entry(score).or_default();
        if bucket.len() < 240 {
            bucket.push(selected.clone());
        }
    }
    if selected.len() >= 4 || score >= 10 {
        return;
    }
    for index in start..pieces.len() {
        let (piece_id, value) = pieces[index];
        if score.saturating_add(value) > 10 {
            continue;
        }
        selected.push(piece_id);
        collect_trolley_bundles(pieces, index + 1, selected, score + value, by_score);
        selected.pop();
    }
}

#[allow(dead_code)]
pub(super) fn push_square_combinations(
    actions: &mut Vec<Action>,
    instance_id: u32,
    candidates: &[Square],
    maximum: usize,
) {
    fn walk(
        actions: &mut Vec<Action>,
        instance_id: u32,
        candidates: &[Square],
        maximum: usize,
        start: usize,
        selected: &mut Vec<Square>,
    ) {
        if !selected.is_empty() {
            actions.push(Action::UseCard {
                instance_id,
                target: ActionTarget::Squares(selected.clone()),
            });
        }
        if selected.len() >= maximum {
            return;
        }
        for index in start..candidates.len() {
            selected.push(candidates[index]);
            walk(
                actions,
                instance_id,
                candidates,
                maximum,
                index + 1,
                selected,
            );
            selected.pop();
        }
    }
    walk(
        actions,
        instance_id,
        candidates,
        maximum,
        0,
        &mut Vec::new(),
    );
}

pub(in crate::engine) fn squares_target(target: &ActionTarget) -> Result<Vec<Square>, EngineError> {
    match target {
        ActionTarget::Squares(squares) if !squares.is_empty() => Ok(squares.clone()),
        ActionTarget::Square(square) | ActionTarget::PieceAt(square) => Ok(vec![*square]),
        _ => Err(EngineError::Invariant(
            "card requires one or more squares".to_owned(),
        )),
    }
}

pub(in crate::engine) fn is_ranged_piece(kind: PieceKind) -> bool {
    matches!(
        kind,
        PieceKind::Bishop
            | PieceKind::Rook
            | PieceKind::Queen
            | PieceKind::Bear
            | PieceKind::Amazon
            | PieceKind::Archbishop
            | PieceKind::Chancellor
            | PieceKind::Cardinal
            | PieceKind::Cannon
            | PieceKind::Herald
            | PieceKind::Hook
            | PieceKind::Protestant
            | PieceKind::Jester
            | PieceKind::Windmill
            | PieceKind::BigRook
            | PieceKind::Idol
    )
}

pub(super) fn suspicious_potion_effects(
    royal: bool,
    kind: PieceKind,
    ranged: bool,
) -> Vec<&'static str> {
    [
        ("sacrifice-protection", !royal),
        ("last-resistance", royal),
        ("coronation-protection", true),
        ("shield", true),
        ("evasion", true),
        ("parry", true),
        ("basic-training", !royal && kind != PieceKind::Pawn),
        ("stealth", true),
        ("loyalist", !royal),
        ("ghost", true),
        ("chameleon", !royal),
        ("submerge", true),
        ("freeze", true),
        ("chimera", !royal),
        ("witch-trial", !royal),
        ("stake", true),
        ("calling-card", !royal),
        ("empty-lunchbox", !royal),
        ("poison-stun", true),
        ("disarm", true),
        ("manner-no-capture", true),
        ("saturation-no-capture", true),
        ("explosive", kind == PieceKind::Pawn),
        ("poisoned-pawn", kind == PieceKind::Pawn),
        ("queens-gambit-protection", kind == PieceKind::Pawn),
        ("trojan-horse", kind == PieceKind::Knight),
        ("severance", ranged),
        ("inertia", ranged),
    ]
    .into_iter()
    .filter_map(|(effect, eligible)| eligible.then_some(effect))
    .collect()
}

#[allow(dead_code)]
pub(super) fn is_major_piece(kind: PieceKind) -> bool {
    matches!(
        kind,
        PieceKind::Queen
            | PieceKind::Rook
            | PieceKind::Amazon
            | PieceKind::Man
            | PieceKind::Colossus
            | PieceKind::BigRook
            | PieceKind::Herald
            | PieceKind::Jester
            | PieceKind::Recruiter
            | PieceKind::Hook
            | PieceKind::PrimeMinister
            | PieceKind::Assassin
            | PieceKind::Wizard
            | PieceKind::Windmill
            | PieceKind::Bear
    )
}

pub(super) fn has_piece_protection_source(state: &GameState, piece_id: u32) -> bool {
    state
        .card_runtime
        .sacrifice_protection
        .contains_key(&piece_id)
        || state.card_runtime.last_resistance.contains_key(&piece_id)
        || state
            .card_runtime
            .coronation_protection
            .contains_key(&piece_id)
        || state
            .card_runtime
            .queens_gambit_protection
            .contains(&piece_id)
        || state
            .card_runtime
            .queens_gambit_previous_protection
            .contains(&piece_id)
}

pub(super) fn middle_piece_value(kind: PieceKind) -> u8 {
    match kind {
        PieceKind::Archbishop => 7,
        PieceKind::Chancellor => 8,
        PieceKind::Pawn
        | PieceKind::Squire
        | PieceKind::Fanatic
        | PieceKind::Scarecrow
        | PieceKind::BabyBear => 1,
        PieceKind::Knight
        | PieceKind::Bishop
        | PieceKind::Alfil
        | PieceKind::Guard
        | PieceKind::Checker
        | PieceKind::CheckerKing
        | PieceKind::Man
        | PieceKind::Camel
        | PieceKind::Bat => 3,
        PieceKind::Ferz | PieceKind::Lobster => 2,
        PieceKind::Rook
        | PieceKind::Cardinal
        | PieceKind::Protestant
        | PieceKind::Windmill
        | PieceKind::Cannon
        | PieceKind::Herald => 5,
        PieceKind::Queen
        | PieceKind::Amazon
        | PieceKind::PrimeMinister
        | PieceKind::Dragon
        | PieceKind::Wizard
        | PieceKind::Jester
        | PieceKind::Hook
        | PieceKind::Bear => 9,
        PieceKind::King | PieceKind::RoyalKnight | PieceKind::Merchant | PieceKind::DarkWizard => {
            10
        }
        PieceKind::BigRook | PieceKind::Colossus => 12,
        _ => 4,
    }
}

pub(super) fn aligned_distance(from: Square, to: Square) -> Option<(i8, i8, u8)> {
    let file = to.file() as i8 - from.file() as i8;
    let rank = to.rank() as i8 - from.rank() as i8;
    if file == 0 && rank != 0 {
        Some((0, rank.signum(), rank.unsigned_abs()))
    } else if rank == 0 && file != 0 {
        Some((file.signum(), 0, file.unsigned_abs()))
    } else if file.unsigned_abs() == rank.unsigned_abs() {
        Some((file.signum(), rank.signum(), file.unsigned_abs()))
    } else {
        None
    }
}

pub(in crate::engine) fn has_adjacent_enemy(
    state: &GameState,
    square: Square,
    owner: Player,
) -> bool {
    KING_OFFSETS.iter().any(|(file, rank)| {
        square
            .offset(*file, *rank)
            .and_then(|target| state.board.get(target))
            .is_some_and(|piece| piece.owner != owner)
    })
}
