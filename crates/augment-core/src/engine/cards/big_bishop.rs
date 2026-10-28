use super::*;
pub(super) const ID: &str = crate::BIG_BISHOP;
pub(super) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    owner: Player,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    engine.install_big_bishop(state, owner, events);
    Ok(())
}

pub(super) fn eligibility(
    state: &GameState,
    player: Player,
    _context: AcquisitionContext,
) -> CardEligibility {
    use PieceKind::*;
    let count = |owner, kind| {
        state
            .board
            .iter()
            .filter(|(_, p)| p.owner == owner && p.kind == kind)
            .count()
    };
    let own = |kind| count(player, kind);
    require(own(King) > 0, "아군 킹이 필요합니다.")
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

impl Engine {
    pub(in crate::engine) fn install_big_bishop(
        &self,
        state: &mut GameState,
        owner: Player,
        events: &mut Vec<Event>,
    ) {
        let horde = state
            .board
            .iter()
            .filter(|(_, p)| p.owner == owner && p.kind == PieceKind::Pawn)
            .count()
            > 8;
        let minor = if horde {
            PieceKind::Pawn
        } else if state.card_runtime.monochrome_chess {
            PieceKind::Camel
        } else {
            PieceKind::Knight
        };
        let anchor = opening_square(owner, "f2", "f8");
        let layout = [
            (anchor, PieceKind::BigBishop, false),
            (opening_square(owner, "f3", "f6"), PieceKind::Pawn, true),
            (opening_square(owner, "g3", "g6"), PieceKind::Pawn, true),
            (opening_square(owner, "h3", "h6"), minor, true),
        ];
        self.install_card_layout(
            state,
            owner,
            &layout,
            &large_piece_cells(anchor),
            BIG_BISHOP,
            events,
        );
    }
}
