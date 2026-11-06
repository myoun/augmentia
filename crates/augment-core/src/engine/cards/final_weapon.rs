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

/// `Engine::generate_pawn_moves` hook: whether the mover's promotion rank is
/// locked to the last rank by final-weapon. Live `shouldPromote` returns on this
/// flag before `earlyPromotion`/`fastGrowth` can advance the rank.
pub(in crate::engine) fn locks_promotion_rank(state: &GameState) -> bool {
    armed(state, state.turn)
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

/// `Engine::push_card_pawn_move` hook: the extra amazon promotion choice, or
/// `None`. Only offered on the final rank; the promotion rank itself is locked
/// separately by `locks_promotion_rank`, so a fast-growth narrowed set still
/// gains the amazon option (live `workerDeferredPromotionChoices` appends it
/// after narrowing).
pub(in crate::engine) fn promotions(state: &GameState, owner: Player) -> Option<PieceKind> {
    let rank = forced_promotion_rank(state, owner)?;
    (amazon_promotion_allowed(state, owner, Square::new(0, rank)?)).then_some(PieceKind::Amazon)
}

impl Engine {
    /// `generate_pawn_moves` hook: true when the moving side has final-weapon
    /// armed, which locks its promotion rank to the last rank.
    ///
    /// Use this inherent method from `engine.rs` rather than the free function:
    /// `cards::final_weapon` is a private module of `cards`, so only items
    /// re-exported through an `impl Engine` block are reachable from the parent
    /// `engine` module without changing `cards/mod.rs`.
    pub(in crate::engine) fn final_weapon_locks_promotion_rank(&self, state: &GameState) -> bool {
        locks_promotion_rank(state)
    }

    /// `push_card_pawn_move` hook: the extra promotion kind final-weapon adds, if
    /// any. Live appends `"amazon"` to the standard choices on the locked rank.
    pub(in crate::engine) fn final_weapon_promotions(
        &self,
        state: &GameState,
        owner: Player,
    ) -> Option<PieceKind> {
        promotions(state, owner)
    }

    /// `push_card_pawn_move` hook: appends the amazon choice to `choices` when
    /// final-weapon allows it on `to`. Inserted before a mutation `Monster`
    /// choice; no duplicates.
    pub(in crate::engine) fn append_final_weapon_promotion(
        &self,
        state: &GameState,
        owner: Player,
        to: Square,
        choices: &mut Vec<PieceKind>,
    ) {
        append_amazon_promotion(state, owner, to, choices);
    }
}

/// Appends the amazon choice to a generated promotion list, mirroring live
/// ordering: standard choices first, then `amazon`, then `monster` from mutation
/// (JS:240620-240622 dedups in that order). Amazon is inserted before any
/// already-appended `Monster` so the caller does not have to order its own
/// pushes. The result never contains duplicates.
pub(in crate::engine) fn append_amazon_promotion(
    state: &GameState,
    owner: Player,
    to: Square,
    choices: &mut Vec<PieceKind>,
) {
    if !amazon_promotion_allowed(state, owner, to) || choices.contains(&PieceKind::Amazon) {
        return;
    }
    match choices.iter().position(|kind| *kind == PieceKind::Monster) {
        Some(index) => choices.insert(index, PieceKind::Amazon),
        None => choices.push(PieceKind::Amazon),
    }
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

        // Fast growth narrows the base choices; amazon is still appended, and it
        // stays after the base list rather than replacing it.
        let mut fast_growth = armed_state(Player::White);
        fast_growth.card_runtime.fast_growth[Player::White.index()] = true;
        let mut narrowed = vec![PieceKind::Bishop, PieceKind::Knight];
        super::append_amazon_promotion(&fast_growth, Player::White, final_rank, &mut narrowed);
        assert_eq!(
            narrowed,
            vec![PieceKind::Bishop, PieceKind::Knight, PieceKind::Amazon]
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
