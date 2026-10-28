use super::*;
pub(super) const ID: &str = crate::LONDON_SYSTEM;
pub(super) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    owner: Player,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    engine.install_london_system(state, owner, events);
    Ok(())
}

pub(super) fn eligibility(
    state: &GameState,
    player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    let has_own = |predicate: &dyn Fn(Piece) -> bool| {
        state
            .board
            .iter()
            .any(|(_, p)| p.owner == player && predicate(p))
    };
    let own_royal = has_own(&|p| royal(state, p));
    require(own_royal, "배치할 아군 왕실 기물이 필요합니다.")
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

impl Engine {
    pub(in crate::engine) fn install_london_system(
        &self,
        state: &mut GameState,
        owner: Player,
        events: &mut Vec<Event>,
    ) {
        // londonLayout: the five untouched home pawns and home royals retain unmoved identity.
        let white = [
            ("a1", PieceKind::Rook, false),
            ("d1", PieceKind::Queen, false),
            ("e1", PieceKind::King, false),
            ("h1", PieceKind::Rook, false),
            ("f4", PieceKind::Bishop, true),
            ("d3", PieceKind::Bishop, true),
            ("f3", PieceKind::Knight, true),
            ("d2", PieceKind::Knight, true),
            ("a2", PieceKind::Pawn, false),
            ("b2", PieceKind::Pawn, false),
            ("c3", PieceKind::Pawn, true),
            ("d4", PieceKind::Pawn, true),
            ("e3", PieceKind::Pawn, true),
            ("f2", PieceKind::Pawn, false),
            ("g2", PieceKind::Pawn, false),
            ("h2", PieceKind::Pawn, false),
        ];
        let layout: Vec<_> = white
            .iter()
            .map(|(name, kind, moved)| {
                let white: Square = name.parse().expect("London layout");
                let square = if owner == Player::White {
                    white
                } else {
                    Square::new(white.file(), 7 - white.rank()).unwrap()
                };
                (square, *kind, *moved)
            })
            .collect();
        self.install_card_layout(state, owner, &layout, &[], LONDON_SYSTEM, events);
    }
}
