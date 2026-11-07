use super::*;
pub(super) const ID: &str = crate::CHAMELEON_MUTATION;

/// Largest number of pieces the acquisition marks, per the catalog text
/// ("최대 3개"). Upstream loops `count < 3` over a shrinking candidate list
/// (main bundle 41601-41611).
const MAX_TARGETS: usize = 3;

/// Chameleon-eligible piece: an own piece that is not royal and not a large body.
///
/// Upstream's `balanceChameleonTargetType` excludes royals, merchants, walls,
/// footballs, black holes, coffins, colossi, big rooks and big bishops
/// (main bundle 1816-1818). The royal and large-body filters below cover the
/// kinds this core can represent.
fn valid(engine: &Engine, state: &GameState, owner: Player, piece: Piece) -> bool {
    piece.owner == owner
        && !engine.is_royal_piece(state, piece)
        && piece.kind != PieceKind::Merchant
        && piece.kind != PieceKind::Football
        && !matches!(
            piece.kind,
            PieceKind::Colossus | PieceKind::BigRook | PieceKind::BigBishop
        )
        && !state.card_runtime.large_pieces.contains_key(&piece.id)
}

/// The 2026-10-27 balance pass turned Chameleon Mutation into an opening passive
/// that marks up to three random own pieces on acquisition. Each mark is drawn
/// from the remaining candidates, so the same piece is never chosen twice and a
/// single candidate consumes no RNG (main bundle 41601-41611).
pub(super) fn acquire(
    engine: &Engine,
    state: &mut GameState,
    owner: Player,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let mut candidates: Vec<(Square, Piece)> = web_ordered_pieces(state)
        .into_iter()
        .filter(|(_, piece)| valid(engine, state, owner, *piece))
        .collect();
    if candidates.is_empty() {
        if matches!(state.phase, GamePhase::Draft(_)) {
            return Ok(());
        }
        return Err(EngineError::Invariant(
            "no applicable chameleon target".into(),
        ));
    }
    for count in 0..MAX_TARGETS {
        if candidates.is_empty() {
            break;
        }
        let index = if candidates.len() == 1 {
            0
        } else {
            let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
            events.push(Event::RandomSample {
                purpose: format!("acquire.chameleon-mutation.target.{count}"),
                roll_53: roll,
            });
            index.expect("nonempty candidate set")
        };
        let (_, piece) = candidates.remove(index);
        state.card_runtime.chameleon_piece_ids.insert(piece.id);
    }
    Ok(())
}

pub(super) fn eligibility(
    state: &GameState,
    owner: Player,
    _: AcquisitionContext,
) -> CardEligibility {
    let engine_agnostic = state.board.iter().any(|(_, piece)| {
        piece.owner == owner
            && !royal(state, piece)
            && piece.kind != PieceKind::Merchant
            && piece.kind != PieceKind::Football
            && !matches!(
                piece.kind,
                PieceKind::Colossus | PieceKind::BigRook | PieceKind::BigBishop
            )
            && !state.card_runtime.large_pieces.contains_key(&piece.id)
    });
    require(
        engine_agnostic,
        "카멜레온 효과를 부여할 아군 기물이 필요합니다.",
    )
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);
