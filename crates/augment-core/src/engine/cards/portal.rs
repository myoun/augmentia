use super::*;
pub(super) const ID: &str = crate::PORTAL;
pub(super) fn install(
    engine: &Engine,
    state: &mut GameState,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    mark_rule(state, ID);
    engine.install_portal(state, events)
}
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::rule(ID, install, rule_eligibility);

#[derive(Clone, Copy)]
pub(in crate::engine) struct RulePortalRoute {
    pub entry: Square,
    pub exit: Square,
    pub landing: Square,
}

impl Engine {
    fn rule_portal_blocked(&self, state: &GameState, square: Square, piece: Piece) -> bool {
        self.rule_cell_collapsed(state, square)
            || state.card_runtime.barricades.contains(&square)
            || state.card_runtime.black_holes.contains(&square)
            || state.card_runtime.football_pos == Some(square)
            || state.card_runtime.crown_pos == Some(square)
            || self
                .board_piece_at(state, square)
                .is_some_and(|p| p.owner == piece.owner)
    }
    fn rule_portal_can_capture(
        &self,
        state: &GameState,
        piece: Piece,
        target: Piece,
        origin: Square,
    ) -> bool {
        if rule_active(state, crate::HIGH_GROUND)
            && state
                .board
                .find_by_id(target.id)
                .is_some_and(|(to, _)| state.card_runtime.high_ground_squares.contains(&to))
            && !state.card_runtime.high_ground_squares.contains(&origin)
        {
            return false;
        }
        let mut probe = state.clone();
        probe.rule_runtime.enabled.remove(crate::HIGH_GROUND);
        self.can_capture(&probe, piece, target)
    }
    fn rule_portal_directions(&self, state: &GameState, piece: Piece) -> Vec<(i8, i8)> {
        let native: &[(i8, i8)] = match self.piece_ability_kind(state, piece) {
            PieceKind::Rook | PieceKind::RevolvingDoor => &[(0, 1), (0, -1), (1, 0), (-1, 0)],
            PieceKind::Bishop | PieceKind::Archbishop => &[(1, 1), (1, -1), (-1, 1), (-1, -1)],
            PieceKind::Chancellor => &[(0, 1), (0, -1), (1, 0), (-1, 0)],
            PieceKind::Queen
            | PieceKind::Amazon
            | PieceKind::Grappler
            | PieceKind::Clockwork
            | PieceKind::PrimeMinister => &[
                (0, 1),
                (0, -1),
                (1, 0),
                (-1, 0),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ],
            _ => &[],
        };
        let mut directions = native.to_vec();
        if rule_active(state, crate::HIGHWAY)
            && state
                .board
                .find_by_id(piece.id)
                .is_some_and(|(s, _)| state.card_runtime.highways.contains(&s.file()))
        {
            directions.extend([(0, 1), (0, -1)]);
        }
        directions
    }
    pub(in crate::engine) fn rule_portal_route(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        to: Square,
    ) -> Option<RulePortalRoute> {
        if !rule_active(state, ID) || state.card_runtime.large_pieces.contains_key(&piece.id) {
            return None;
        }
        let pair = state.card_runtime.rule_portals?;
        if let Some(i) = pair.iter().position(|s| *s == to) {
            let exit = pair[1 - i];
            if self.rule_portal_blocked(state, exit, piece) {
                return None;
            }
            if self
                .board_piece_at(state, exit)
                .is_some_and(|p| !self.rule_portal_can_capture(state, piece, p, exit))
            {
                return None;
            }
            return Some(RulePortalRoute {
                entry: to,
                exit,
                landing: exit,
            });
        }
        for (df, dr) in self.rule_portal_directions(state, piece) {
            let mut cursor = from;
            let mut transit = None;
            while let Some(square) = cursor.offset(df, dr) {
                if let Some((entry, exit)) = transit {
                    if pair.contains(&square) {
                        break;
                    }
                    if square == to
                        && (self.board_piece_at(state, square).is_none()
                            || self.board_piece_at(state, square).is_some_and(|p| {
                                self.rule_portal_can_capture(state, piece, p, exit)
                            }))
                    {
                        return Some(RulePortalRoute {
                            entry,
                            exit,
                            landing: to,
                        });
                    }
                    if self.board_piece_at(state, square).is_some()
                        || self.rule_cell_collapsed(state, square)
                        || state.card_runtime.barricades.contains(&square)
                    {
                        break;
                    }
                    cursor = square;
                    continue;
                }
                if self.board_piece_at(state, square).is_some()
                    || state.card_runtime.barricades.contains(&square)
                {
                    break;
                }
                if let Some(i) = pair.iter().position(|s| *s == square) {
                    let exit = pair[1 - i];
                    if self.rule_portal_blocked(state, exit, piece)
                        || self.board_piece_at(state, exit).is_some()
                    {
                        break;
                    }
                    transit = Some((square, exit));
                    cursor = exit;
                    continue;
                }
                cursor = square;
            }
        }
        None
    }

    pub(in crate::engine) fn augment_rule_portal_moves(
        &self,
        state: &GameState,
        from: Square,
        piece: Piece,
        actions: &mut Vec<Action>,
    ) {
        if !rule_active(state, ID) {
            return;
        }
        let Some(pair) = state.card_runtime.rule_portals else {
            return;
        };
        if state.card_runtime.large_pieces.contains_key(&piece.id) {
            return;
        }
        // Direct entry is represented by the clicked entry square; apply resolves its exit.
        actions.retain(|a| {
            let Action::Move(m) = a else {
                return true;
            };
            if let Some(i) = pair.iter().position(|s| *s == m.to) {
                let exit = pair[1 - i];
                return self.rule_portal_blocked(state, exit, piece)
                    || self
                        .board_piece_at(state, exit)
                        .is_none_or(|p| self.rule_portal_can_capture(state, piece, p, exit));
            }
            if self.rule_portal_route(state, from, piece, m.to).is_some() {
                return true;
            }
            let directions = self.rule_portal_directions(state, piece);
            for (df, dr) in directions {
                let mut cursor = from;
                while let Some(cell) = cursor.offset(df, dr) {
                    if cell == m.to {
                        break;
                    }
                    if let Some(i) = pair.iter().position(|s| *s == cell) {
                        let exit = pair[1 - i];
                        if !self.rule_portal_blocked(state, exit, piece)
                            && self
                                .board_piece_at(state, exit)
                                .is_none_or(|p| self.rule_portal_can_capture(state, piece, p, exit))
                        {
                            // Only the ray containing this destination is mandatory.
                            let mut beyond = cell;
                            while let Some(next) = beyond.offset(df, dr) {
                                if next == m.to {
                                    return false;
                                }
                                beyond = next;
                            }
                        }
                        break;
                    }
                    if self.board_piece_at(state, cell).is_some() {
                        break;
                    }
                    cursor = cell;
                }
            }
            true
        });
        for index in 0..64 {
            let to = Square::from_index(index).unwrap();
            if to == from || pair.contains(&to) {
                continue;
            }
            if self.rule_portal_route(state, from, piece, to).is_some() {
                actions.push(Action::Move(MoveAction::new(from, to, None)));
            }
        }
    }
}

impl Engine {
    pub(in crate::engine) fn install_portal(
        &self,
        state: &mut GameState,
        _events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        state.card_runtime.rule_portals = Some(["c3".parse().unwrap(), "f6".parse().unwrap()]);
        Ok(())
    }
}
