use super::*;

pub(super) const ID: &str = crate::FINAL_WEAPON;

/// `acquire` arms the passive for its owner. Live `finalWeapon` (JS:354634-354637)
/// only sets the per-color flag; every rule effect is read from that flag.
fn acquire(
    _engine: &Engine,
    state: &mut GameState,
    owner: Player,
    _events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    state.card_runtime.final_weapon[owner.index()] = true;
    Ok(())
}

fn eligibility(state: &GameState, player: Player, _: AcquisitionContext) -> CardEligibility {
    if let Some(result) = enabled(state.card_runtime.final_weapon[player.index()]) {
        return result;
    }
    // Live `isCardDrawableForColor` (JS:235579): `!finalWeapon[color]` and at
    // least one own pawn.
    require(
        state
            .board
            .iter()
            .any(|(_, piece)| piece.owner == player && piece.kind == PieceKind::Pawn),
        "아군 폰이 필요합니다.",
    )
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::passive(ID, Some(acquire), eligibility);

/// True when the passive is armed for `owner`.
pub(in crate::engine) fn armed(state: &GameState, owner: Player) -> bool {
    state.card_runtime.final_weapon[owner.index()]
}

/// Final rank of the board for `owner` (rank index 7 for White, 0 for Black).
fn final_rank(owner: Player) -> u8 {
    match owner {
        Player::White => 7,
        Player::Black => 0,
    }
}

/// The promotion rank final-weapon forces, or `None` when the card is not armed.
///
/// Live `shouldPromote` (JS:359224-359229) short-circuits on
/// `state.finalWeapon[color]`: promotion happens only on the last rank and only
/// while the board is not collapsed, which also disables the early-promotion and
/// fast-growth advanced promotion rows. Live
/// `promotionContract` (JS:224228-224235) implements the same rule by setting
/// `promotionAdvance = 0` whenever final-weapon is applied.
///
/// ORDERING: this must be consulted BEFORE `early_promotion` and `fast_growth`
/// in the pawn-move generator. Those two advance the promotion rank only when
/// final-weapon is not armed.
pub(in crate::engine) fn forced_promotion_rank(state: &GameState, owner: Player) -> Option<u8> {
    armed(state, owner).then(|| final_rank(owner))
}

/// Whether an amazon promotion must be offered for `owner` on `to`.
///
/// Live `promotionChoicesFor` (JS:359238-359249) and the worker
/// `workerDeferredPromotionChoices` (JS:240618-240624) both append `"amazon"` to
/// the standard choices only for the final rank, and only when the board is not
/// collapsed.
pub(in crate::engine) fn amazon_promotion_allowed(
    state: &GameState,
    owner: Player,
    to: Square,
) -> bool {
    armed(state, owner)
        && !state.card_runtime.collapsed_files_ranks
        && to.rank() == final_rank(owner)
}

/// Standard promotion set, in the order the engine's pawn generator uses.
const STANDARD_PROMOTIONS: [PieceKind; 4] = [
    PieceKind::Knight,
    PieceKind::Bishop,
    PieceKind::Rook,
    PieceKind::Queen,
];

/// Inserts `kind` before a mutation `Monster` choice, or appends it.
fn insert_promotion(choices: &mut Vec<PieceKind>, kind: PieceKind) {
    if choices.contains(&kind) {
        return;
    }
    match choices
        .iter()
        .position(|existing| *existing == PieceKind::Monster)
    {
        Some(index) => choices.insert(index, kind),
        None => choices.push(kind),
    }
}

/// Appends the amazon choice to a generated promotion list, mirroring live
/// `promotionChoicesFor` (JS:359238-359249):
/// `[...PROMOTION_TYPES, "amazon", ...mutation]`.
///
/// ORDERING: final-weapon also suppresses fast-growth's minor-only narrowing.
/// Live `isWorkerFastGrowthPromotion` (JS:246255-246266) returns false on the
/// normal promotion row, and `promotionChoicesFor` only returns
/// `MINOR_PROMOTION_TYPES` from the `fastGrowthPromotion` branch, which never
/// fires on the final rank. The engine's `push_card_pawn_move` narrows the list
/// to knight/bishop whenever `fast_growth` is armed, so this function restores
/// the full standard set here and reports that the caller must push the amazon
/// choice through this hook rather than appending it separately.
pub(in crate::engine) fn append_amazon_promotion(
    state: &GameState,
    owner: Player,
    to: Square,
    choices: &mut Vec<PieceKind>,
) {
    if !amazon_promotion_allowed(state, owner, to) {
        return;
    }
    for kind in STANDARD_PROMOTIONS {
        insert_promotion(choices, kind);
    }
    insert_promotion(choices, PieceKind::Amazon);
}

#[cfg(test)]
mod tests {
    use super::super::tests::delayed::*;
    use super::*;

    fn armed_state(owner: Player) -> GameState {
        let mut state = position(&[("b2", Player::White, PieceKind::Pawn)]);
        state.card_runtime.final_weapon[owner.index()] = true;
        state
    }

    fn acquire_state(state: &mut GameState, owner: Player) {
        super::acquire(
            &Engine::bundled_latest().unwrap(),
            state,
            owner,
            &mut Vec::new(),
        )
        .unwrap();
    }

    #[test]
    fn final_weapon_acquisition_arms_the_owner_only() {
        let mut state = position(&[("b2", Player::White, PieceKind::Pawn)]);
        assert!(!super::armed(&state, Player::White));
        acquire_state(&mut state, Player::White);
        assert!(super::armed(&state, Player::White));
        assert!(!super::armed(&state, Player::Black));
        assert!(super::forced_promotion_rank(&state, Player::White).is_some());
        assert!(super::forced_promotion_rank(&state, Player::Black).is_none());
    }

    #[test]
    fn forced_promotion_rank_is_the_last_rank_and_overrides_early_and_fast_growth() {
        let state = position(&[]);
        assert_eq!(super::forced_promotion_rank(&state, Player::White), None);

        let mut white = armed_state(Player::White);
        assert_eq!(super::forced_promotion_rank(&white, Player::White), Some(7));
        // ORDERING: early-promotion and fast-growth would move the rank to 5/4;
        // final-weapon wins because live `shouldPromote` returns first.
        white.card_runtime.early_promotion[Player::White.index()] = true;
        assert_eq!(super::forced_promotion_rank(&white, Player::White), Some(7));
        white.card_runtime.fast_growth[Player::White.index()] = true;
        assert_eq!(super::forced_promotion_rank(&white, Player::White), Some(7));

        let mut black = armed_state(Player::Black);
        assert_eq!(super::forced_promotion_rank(&black, Player::Black), Some(0));
        black.card_runtime.early_promotion[Player::Black.index()] = true;
        assert_eq!(super::forced_promotion_rank(&black, Player::Black), Some(0));
    }

    #[test]
    fn amazon_choice_is_offered_only_on_the_locked_last_rank() {
        let state = armed_state(Player::White);
        let final_rank: Square = "b8".parse().unwrap();
        let mid_rank: Square = "b6".parse().unwrap();
        assert!(super::amazon_promotion_allowed(
            &state,
            Player::White,
            final_rank
        ));
        assert!(!super::amazon_promotion_allowed(
            &state,
            Player::White,
            mid_rank
        ));
        assert!(!super::amazon_promotion_allowed(
            &state,
            Player::Black,
            final_rank
        ));

        let mut choices = vec![
            PieceKind::Knight,
            PieceKind::Bishop,
            PieceKind::Rook,
            PieceKind::Queen,
        ];
        super::append_amazon_promotion(&state, Player::White, final_rank, &mut choices);
        assert_eq!(
            choices,
            vec![
                PieceKind::Knight,
                PieceKind::Bishop,
                PieceKind::Rook,
                PieceKind::Queen,
                PieceKind::Amazon,
            ]
        );
        // Idempotent: a second call does not duplicate the choice.
        super::append_amazon_promotion(&state, Player::White, final_rank, &mut choices);
        assert_eq!(choices.len(), 5);

        // Fast growth narrows the base choices off the normal promotion row, but
        // live `promotionChoicesFor` returns the full standard set plus amazon on
        // the final rank (the `fastGrowthPromotion` branch never fires there).
        let mut fast_growth = armed_state(Player::White);
        fast_growth.card_runtime.fast_growth[Player::White.index()] = true;
        let mut narrowed = vec![PieceKind::Knight, PieceKind::Bishop];
        super::append_amazon_promotion(&fast_growth, Player::White, final_rank, &mut narrowed);
        assert_eq!(
            narrowed,
            vec![
                PieceKind::Knight,
                PieceKind::Bishop,
                PieceKind::Rook,
                PieceKind::Queen,
                PieceKind::Amazon,
            ]
        );
        // Mutation's monster stays last.
        let mut with_monster = vec![
            PieceKind::Knight,
            PieceKind::Bishop,
            PieceKind::Rook,
            PieceKind::Queen,
            PieceKind::Monster,
        ];
        super::append_amazon_promotion(&fast_growth, Player::White, final_rank, &mut with_monster);
        assert_eq!(
            with_monster,
            vec![
                PieceKind::Knight,
                PieceKind::Bishop,
                PieceKind::Rook,
                PieceKind::Queen,
                PieceKind::Amazon,
                PieceKind::Monster,
            ]
        );
    }

    #[test]
    fn final_weapon_does_not_offer_amazon_when_collapsed_or_unarmed() {
        let mut state = armed_state(Player::White);
        state.card_runtime.collapsed_files_ranks = true;
        assert!(!super::amazon_promotion_allowed(
            &state,
            Player::White,
            "b8".parse().unwrap()
        ));

        let unarmed = position(&[("b2", Player::White, PieceKind::Pawn)]);
        let mut choices = vec![PieceKind::Queen];
        super::append_amazon_promotion(
            &unarmed,
            Player::White,
            "b8".parse().unwrap(),
            &mut choices,
        );
        assert_eq!(choices, vec![PieceKind::Queen]);
    }

    #[test]
    fn final_weapon_eligibility_requires_a_pawn_and_rejects_a_second_acquisition() {
        let pawnless = position(&[]);
        assert!(
            super::eligibility(&pawnless, Player::White, AcquisitionContext::Draft)
                .reason()
                .is_some(),
            "a king alone cannot promote"
        );
        let with_pawn = position(&[("b2", Player::White, PieceKind::Pawn)]);
        assert!(
            super::eligibility(&with_pawn, Player::White, AcquisitionContext::Draft).is_eligible()
        );
        // The opponent's pawn does not satisfy the owner's requirement.
        assert!(
            super::eligibility(&with_pawn, Player::Black, AcquisitionContext::Draft)
                .reason()
                .is_some()
        );
        let mut active = with_pawn.clone();
        acquire_state(&mut active, Player::White);
        assert!(
            super::eligibility(&active, Player::White, AcquisitionContext::Draft)
                .reason()
                .is_some()
        );
    }
}
