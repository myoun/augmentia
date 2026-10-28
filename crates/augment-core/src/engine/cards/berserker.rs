use super::*;
pub(super) const ID: &str = crate::BERSERKER;
pub(super) fn play(
    engine: &Engine,
    state: &mut GameState,
    actor: Player,
    _instance_id: u32,
    target: &ActionTarget,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    let definition_id = ID.to_owned();
    let target = target.clone();
    let (target_square, mut target_piece) = piece_target(&*state, &target)?;
    if definition_id == MISSIONARY {
        let candidates = engine.missionary_spawn_candidates(&*state, target_square);
        let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
        let square = candidates[index.ok_or_else(|| {
            EngineError::Invariant("legal missionary card has no spawn square".to_owned())
        })?];
        events.push(Event::RandomSample {
            purpose: "missionary-second-spawn".to_owned(),
            roll_53: roll,
        });
        let summoned = Piece::new(actor, PieceKind::Missionary).with_id(state.allocate_piece_id());
        state.board.set(square, Some(summoned));
        engine.register_spawned_piece(state, square, summoned, true);
        engine.clear_queens_gambit_protection(state, target_piece.id);
    }
    engine.clear_queens_gambit_protection(state, target_piece.id);
    target_piece.kind = match definition_id.as_str() {
        CONSTITUTIONAL_MONARCHY => PieceKind::PrimeMinister,
        DRAGON => PieceKind::Dragon,
        EASTERN_POLICY => PieceKind::Cannon,
        HERALD => PieceKind::Herald,
        KNIGHTMASTER => PieceKind::Knightmaster,
        PEGASUS => PieceKind::Unicorn,
        IDOL => PieceKind::Idol,
        JESTER => PieceKind::Jester,
        LOCAL_CONSCRIPTION => PieceKind::Recruiter,
        REAPER => PieceKind::Reaper,
        WIZARD => PieceKind::Wizard,
        LOG => PieceKind::Log,
        MISSIONARY => PieceKind::Missionary,
        STANDARD_BEARER => PieceKind::StandardBearer,
        BERSERKER => PieceKind::Berserker,
        MAGIC_GIRL => PieceKind::MagicGirl,
        SIEGE_RAM => PieceKind::SiegeRam,
        SIREN => PieceKind::Siren,
        SLIME => PieceKind::Slime,
        TRICKSTER => PieceKind::Trickster,
        UNDEAD => PieceKind::Undead,
        crate::CLOCKWORK => PieceKind::Clockwork,
        crate::PARROT => PieceKind::Parrot,
        crate::PALADIN => PieceKind::Paladin,
        crate::PRINCESS => PieceKind::Princess,
        crate::OCTOPUS => PieceKind::Octopus,
        crate::HEDGEHOG => PieceKind::Hedgehog,
        crate::CAMPFIRE => PieceKind::Campfire,
        _ => unreachable!(),
    };
    state.board.set(target_square, Some(target_piece));
    state.card_runtime.moved_piece_ids.insert(target_piece.id);
    engine.mark_fresh_no_capture(state, target_piece);
    match target_piece.kind {
        PieceKind::Recruiter if state.ruleset_id == crate::LATEST_RULESET_ID => {
            target_piece.capture_count = 0;
            state.board.set(target_square, Some(target_piece));
        }
        PieceKind::Hedgehog => {
            state
                .card_runtime
                .bear_retaliations
                .insert(target_piece.id, 2);
        }
        PieceKind::Herald if state.shared_turn_count() >= 2 => {
            state.card_runtime.herald_jump_locked_until.insert(
                target_piece.id,
                state.turns_taken[actor.index()].saturating_add(1),
            );
        }
        PieceKind::Wizard => {
            state.card_runtime.wizard_mana.insert(target_piece.id, 0);
        }
        PieceKind::Reaper => {
            state.card_runtime.reaper_souls.insert(target_piece.id, 0);
        }
        PieceKind::Trickster => {
            let kind = engine.choose_trickster_kind(state, events);
            state
                .card_runtime
                .trickster_move_kinds
                .insert(target_piece.id, kind);
        }
        _ => {}
    }
    Ok(())
}

pub(super) fn actions(
    _engine: &Engine,
    state: &GameState,
    card: &CardInstance,
    actions: &mut Vec<Action>,
) {
    for (square, piece) in state.board.iter() {
        if piece.owner == state.turn && piece.kind == PieceKind::Rook {
            actions.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::PieceAt(square),
            });
        }
    }
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
    require(own(Rook) > 0, "아군 룩이 필요합니다.")
}

pub(super) const BEHAVIOR: CardBehavior = CardBehavior::active(ID, play, actions, eligibility);
