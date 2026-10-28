//! Historical ruleset dispatch. Current cards are implemented in `cards/`.
use super::*;
impl Engine {
    pub(super) fn install_compat_rule(
        &self,
        state: &mut GameState,
        definition_id: &str,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        match definition_id {
            crate::CAPTURE_THE_FLAG => self.install_capture_the_flag(state, events),
            PLATFORM => {
                let current_turn = state.shared_turn_count();
                state.card_runtime.platform = Some(PlatformRuleState {
                    cell: None,
                    spawned_at: current_turn,
                    next_at: current_turn.saturating_add(5),
                    triggered_piece_ids: BTreeSet::new(),
                    nonce: 0,
                });
                self.spawn_platform(state, current_turn, events);
            }
            REVELATION => {
                state.card_runtime.revelation = true;
                if let Some(deathmatch) = &mut state.deathmatch {
                    deathmatch.interval_full_turns = 5;
                    deathmatch.quiet_full_turns = deathmatch.quiet_full_turns.min(5);
                }
            }
            _ => {
                if !self.install_rule_ext(state, definition_id, events)? {
                    return Err(EngineError::Invariant(format!(
                        "implemented RULE {definition_id:?} has no installer"
                    )));
                }
            }
        }
        Ok(())
    }
    pub(super) fn apply_compat_active_card(
        &self,
        state: &mut GameState,
        actor: Player,
        definition_id: &str,
        target: &ActionTarget,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        let definition_id = definition_id.to_owned();
        let target = target.clone();
        match definition_id.as_str() {
            crate::MAN => {
                let (square, mut piece) = piece_target(&*state, &target)?;
                self.clear_piece_kind_runtime(state, piece.id);
                state.card_runtime.trickster_move_kinds.remove(&piece.id);
                self.clear_queens_gambit_protection(state, piece.id);
                piece.kind = PieceKind::Man;
                state.board.set(square, Some(piece));
                self.mark_transformed_piece(state, piece);
            }
            APPRENTICE_KNIGHTS
            | BIG_BISHOP
            | BIG_ROOK
            | CALLING_CARD
            | CHECKER
            | DEMOCRACY
            | DUTCH
            | ELEPHANT_ESCAPE
            | FAST_GROWTH
            | FIANCHETTO
            | FIELD_PROMOTION
            | GUARD
            | HOLDOUT
            | HORDE
            | IRON_MONARCH
            | KNIGHTMATE
            | LAST_STAND
            | LONDON_SYSTEM
            | LOYALIST
            | MARTYRDOM
            | MERCHANT_GUILD
            | OTHERWORLD
            | PAWN_CONVERSION
            | QUEEN_CAVALRY
            | QUEENS_GAMBIT
            | QXE1
            | SUMMON_COLOSSUS
            | THIEF
            | crate::FALSE_START
            | crate::LOCUST_SWARM => {
                self.apply_opening_card(state, actor, &definition_id, &target, events)?;
            }
            PARRY => {
                let (_, target_piece) = piece_target(&*state, &target)?;
                state.card_runtime.parry_piece_ids.insert(target_piece.id);
            }
            EMPTY_LUNCHBOX => {
                let (_, target_piece) = piece_target(&*state, &target)?;
                state.card_runtime.empty_lunchboxes.insert(
                    target_piece.id,
                    EmptyLunchboxStatus {
                        by: actor,
                        deadline_turn: state.turns_taken[target_piece.owner.index()]
                            .saturating_add(3),
                    },
                );
            }
            DISARM => {
                let (_, target_piece) = piece_target(&*state, &target)?;
                state.card_runtime.disarmed.insert(target_piece.id, 1);
            }
            DICE => {
                if target != ActionTarget::None {
                    return Err(EngineError::Invariant(
                        "dice must not have a target".to_owned(),
                    ));
                }
                let kinds = [
                    PieceKind::Pawn,
                    PieceKind::Knight,
                    PieceKind::Bishop,
                    PieceKind::Rook,
                    PieceKind::Queen,
                    PieceKind::King,
                ];
                let (roll, index) = state.rng_mut().web_choice_index(kinds.len());
                events.push(Event::RandomSample {
                    purpose: "dice".to_owned(),
                    roll_53: roll,
                });
                let kind = kinds[index.expect("six-sided die has a result")];
                state.card_runtime.dice_locks[actor.opponent().index()] = Some(DiceLockStatus {
                    kind,
                    remaining_turns: 2,
                });
            }
            FREEZE => {
                if target != ActionTarget::None {
                    return Err(EngineError::Invariant(
                        "freeze must not have a target".to_owned(),
                    ));
                }
                self.apply_freeze(state, actor, events);
            }
            REPOSITION => {
                let (_, target_piece) = piece_target(&*state, &target)?;
                state.card_runtime.reposition = Some(RepositionStatus {
                    piece_id: target_piece.id,
                    used: false,
                });
            }
            crate::METAL => {
                let (_, target_piece) = piece_target(&*state, &target)?;
                state
                    .card_runtime
                    .metalized_piece_ids
                    .insert(target_piece.id);
                state.card_runtime.metal_cooldowns.remove(&target_piece.id);
            }
            ASSASSIN | GRASSHOPPER => {
                let (target_square, mut target_piece) = piece_target(&*state, &target)?;
                self.clear_queens_gambit_protection(state, target_piece.id);
                target_piece.kind = if definition_id == ASSASSIN {
                    PieceKind::Assassin
                } else {
                    PieceKind::Grasshopper
                };
                state.board.set(target_square, Some(target_piece));
                state.card_runtime.moved_piece_ids.insert(target_piece.id);
                self.mark_fresh_no_capture(state, target_piece);
            }
            CONSTITUTIONAL_MONARCHY
            | DRAGON
            | EASTERN_POLICY
            | HERALD
            | KNIGHTMASTER
            | PEGASUS
            | IDOL
            | JESTER
            | LOCAL_CONSCRIPTION
            | REAPER
            | WIZARD
            | LOG
            | MISSIONARY
            | STANDARD_BEARER
            | BERSERKER
            | MAGIC_GIRL
            | SIEGE_RAM
            | SIREN
            | SLIME
            | TRICKSTER
            | UNDEAD
            | crate::CLOCKWORK
            | crate::PARROT
            | crate::PALADIN
            | crate::PRINCESS
            | crate::OCTOPUS
            | crate::HEDGEHOG
            | crate::CAMPFIRE => {
                let (target_square, mut target_piece) = piece_target(&*state, &target)?;
                if definition_id == MISSIONARY {
                    let candidates = self.missionary_spawn_candidates(&*state, target_square);
                    let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
                    let square = candidates[index.ok_or_else(|| {
                        EngineError::Invariant(
                            "legal missionary card has no spawn square".to_owned(),
                        )
                    })?];
                    events.push(Event::RandomSample {
                        purpose: "missionary-second-spawn".to_owned(),
                        roll_53: roll,
                    });
                    let summoned =
                        Piece::new(actor, PieceKind::Missionary).with_id(state.allocate_piece_id());
                    state.board.set(square, Some(summoned));
                    self.register_spawned_piece(state, square, summoned, true);
                    self.clear_queens_gambit_protection(state, target_piece.id);
                }
                self.clear_queens_gambit_protection(state, target_piece.id);
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
                self.mark_fresh_no_capture(state, target_piece);
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
                        let kind = self.choose_trickster_kind(state, events);
                        state
                            .card_runtime
                            .trickster_move_kinds
                            .insert(target_piece.id, kind);
                    }
                    _ => {}
                }
            }
            ORDINATION => {
                let (target_square, mut target_piece) = piece_target(&*state, &target)?;
                let sacrifice = web_ordered_pieces(&*state).into_iter().find(|(_, piece)| {
                    piece.owner == actor
                        && piece.kind == PieceKind::Bishop
                        && piece.id != target_piece.id
                });
                let (sacrifice_square, sacrificed) = sacrifice.ok_or_else(|| {
                    EngineError::Invariant("ordination requires a second bishop".to_owned())
                })?;
                self.remove_piece_for_card(
                    state,
                    sacrifice_square,
                    sacrificed,
                    actor.opponent(),
                    ORDINATION,
                    events,
                );
                self.clear_queens_gambit_protection(state, target_piece.id);
                target_piece.kind = PieceKind::Cardinal;
                state.board.set(target_square, Some(target_piece));
                state.card_runtime.moved_piece_ids.insert(target_piece.id);
                self.mark_fresh_no_capture(state, target_piece);
            }
            AMAZON | HOOK => {
                let [queen_square, sacrifice_square] = two_square_target(&target)?;
                let mut queen = state.board.get(queen_square).ok_or_else(|| {
                    EngineError::Invariant("piece card has no selected queen".to_owned())
                })?;
                let sacrificed = state.board.get(sacrifice_square).ok_or_else(|| {
                    EngineError::Invariant("piece card has no sacrifice".to_owned())
                })?;
                state.board.set(sacrifice_square, None);
                self.clear_removed_piece_statuses(state, sacrificed.id);
                events.push(Event::PieceRemoved {
                    actor: actor.opponent(),
                    piece: sacrificed,
                    at: sacrifice_square,
                    cause: definition_id.clone(),
                });
                self.resolve_reaper_deaths(
                    state,
                    &[(sacrifice_square, sacrificed)],
                    actor.opponent(),
                    events,
                );
                self.clear_queens_gambit_protection(state, queen.id);
                queen.kind = if definition_id == AMAZON {
                    PieceKind::Amazon
                } else {
                    PieceKind::Hook
                };
                state.board.set(queen_square, Some(queen));
                state.card_runtime.moved_piece_ids.insert(queen.id);
                self.mark_fresh_no_capture(state, queen);
            }
            EAGLE => {
                if target != ActionTarget::None {
                    return Err(EngineError::Invariant(
                        "alibaba must not have a target".to_owned(),
                    ));
                }
                let knights = web_ordered_pieces(&*state)
                    .into_iter()
                    .filter(|(_, piece)| piece.owner == actor && piece.kind == PieceKind::Knight)
                    .collect::<Vec<_>>();
                for &(square, mut knight) in &knights {
                    self.clear_queens_gambit_protection(state, knight.id);
                    knight.kind = PieceKind::Alibaba;
                    state.board.set(square, Some(knight));
                    state.card_runtime.moved_piece_ids.insert(knight.id);
                    self.mark_fresh_no_capture(state, knight);
                }
                for (origin, _) in knights {
                    let candidates = self.nearest_alibaba_candidates(&*state, origin, actor);
                    if candidates.is_empty() {
                        break;
                    }
                    let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
                    events.push(Event::RandomSample {
                        purpose: "alibaba-placement".to_owned(),
                        roll_53: roll,
                    });
                    let square = candidates[index.expect("non-empty Alibaba candidates")];
                    let piece =
                        Piece::new(actor, PieceKind::Alibaba).with_id(state.allocate_piece_id());
                    state.board.set(square, Some(piece));
                    state.card_runtime.moved_piece_ids.insert(piece.id);
                    self.mark_fresh_no_capture(state, piece);
                }
            }
            REFORMATION => {
                if target != ActionTarget::None {
                    return Err(EngineError::Invariant(
                        "reformation must not have a target".to_owned(),
                    ));
                }
                let bishops = web_ordered_pieces(&*state)
                    .into_iter()
                    .filter(|(_, piece)| piece.owner == actor && piece.kind == PieceKind::Bishop)
                    .collect::<Vec<_>>();
                for (square, mut bishop) in bishops {
                    self.clear_queens_gambit_protection(state, bishop.id);
                    bishop.kind = PieceKind::Protestant;
                    state.board.set(square, Some(bishop));
                    state.card_runtime.moved_piece_ids.insert(bishop.id);
                    self.mark_fresh_no_capture(state, bishop);
                }
            }
            WINDMILL => {
                let [bishop_square, rook_square] = two_square_target(&target)?;
                for square in [bishop_square, rook_square] {
                    let mut piece = state.board.get(square).ok_or_else(|| {
                        EngineError::Invariant("windmill has a missing selected piece".to_owned())
                    })?;
                    self.clear_queens_gambit_protection(state, piece.id);
                    piece.kind = PieceKind::Windmill;
                    state.board.set(square, Some(piece));
                    state.card_runtime.moved_piece_ids.insert(piece.id);
                    state
                        .card_runtime
                        .windmill_modes
                        .insert(piece.id, WindmillMode::Bishop);
                    self.mark_fresh_no_capture(state, piece);
                }
            }
            JUDGMENT => {
                let (target_square, target_piece) = piece_target(&*state, &target)?;
                state.board.set(target_square, None);
                let return_at_draft = if !state.completed_drafts[DraftPhase::Middle.index()] {
                    Some(DraftPhase::Middle)
                } else if !state.completed_drafts[DraftPhase::End.index()] {
                    Some(DraftPhase::End)
                } else {
                    None
                };
                if let Some(return_at_draft) = return_at_draft {
                    state.card_runtime.judgment_exiles.push(JudgmentExile {
                        piece: target_piece,
                        square: target_square,
                        return_at_draft,
                    });
                } else {
                    self.clear_removed_piece_statuses(state, target_piece.id);
                }
                let remover = if target_piece.owner == actor {
                    actor.opponent()
                } else {
                    actor
                };
                events.push(Event::PieceRemoved {
                    actor: remover,
                    piece: target_piece,
                    at: target_square,
                    cause: JUDGMENT.to_owned(),
                });
                if return_at_draft.is_none() {
                    self.grant_wizard_mana_for_loss(state, target_piece.owner);
                    self.resolve_reaper_deaths(
                        state,
                        &[(target_square, target_piece)],
                        remover,
                        events,
                    );
                }
            }
            CLEANUP_SACRIFICE => {
                let (selected_square, selected_piece) = piece_target(&*state, &target)?;
                let candidates = web_ordered_pieces(&*state)
                    .into_iter()
                    .filter(|(_, piece)| {
                        piece.owner == actor.opponent()
                            && piece.kind == selected_piece.kind
                            && !self.is_royal_piece(&*state, *piece)
                    })
                    .collect::<Vec<_>>();
                let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
                events.push(Event::RandomSample {
                    purpose: "cleanup-sacrifice".to_owned(),
                    roll_53: roll,
                });
                let (victim_square, victim_piece) = candidates[index.ok_or_else(|| {
                    EngineError::Invariant(
                        "legal cleanup sacrifice has no matching enemy".to_owned(),
                    )
                })?];
                state.board.set(victim_square, None);
                self.clear_removed_piece_statuses(state, victim_piece.id);
                events.push(Event::PieceRemoved {
                    actor,
                    piece: victim_piece,
                    at: victim_square,
                    cause: CLEANUP_SACRIFICE.to_owned(),
                });
                self.grant_wizard_mana_for_loss(state, victim_piece.owner);
                self.resolve_reaper_deaths(state, &[(victim_square, victim_piece)], actor, events);
                state.board.set(selected_square, None);
                self.clear_removed_piece_statuses(state, selected_piece.id);
                events.push(Event::PieceRemoved {
                    actor: actor.opponent(),
                    piece: selected_piece,
                    at: selected_square,
                    cause: CLEANUP_SACRIFICE.to_owned(),
                });
                self.grant_wizard_mana_for_loss(state, selected_piece.owner);
                self.resolve_reaper_deaths(
                    state,
                    &[(selected_square, selected_piece)],
                    actor.opponent(),
                    events,
                );
                self.break_initiative_by_capture(state, victim_piece.owner, actor);
            }
            _ => {
                if !self.apply_middle_active_card(state, actor, &definition_id, &target, events)?
                    && !self.apply_end_active_card(state, actor, &definition_id, &target, events)?
                {
                    return Err(EngineError::Invariant(format!(
                        "unsupported active card {definition_id:?} was legal"
                    )));
                }
            }
        }
        Ok(())
    }
    pub(super) fn generate_compat_card_actions(
        &self,
        state: &GameState,
        card: &CardInstance,
        actions: &mut Vec<Action>,
    ) {
        match card.definition_id.as_str() {
            crate::MAN => {
                for (square, piece) in state.board.iter() {
                    if crate::latest::latest_minor_kind(piece.kind)
                        && !self.is_royal_piece(state, piece)
                    {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            APPRENTICE_KNIGHTS
            | BIG_BISHOP
            | BIG_ROOK
            | CALLING_CARD
            | CHECKER
            | DEMOCRACY
            | DUTCH
            | ELEPHANT_ESCAPE
            | FAST_GROWTH
            | FIANCHETTO
            | FIELD_PROMOTION
            | GUARD
            | HORDE
            | IRON_MONARCH
            | KNIGHTMATE
            | LAST_STAND
            | LONDON_SYSTEM
            | MARTYRDOM
            | MERCHANT_GUILD
            | OTHERWORLD
            | PAWN_CONVERSION
            | QUEEN_CAVALRY
            | QXE1
            | SUMMON_COLOSSUS
            | THIEF
            | crate::FALSE_START
            | crate::LOCUST_SWARM => {
                if self.opening_card_none_legal(state, &card.definition_id, state.turn) {
                    actions.push(Action::UseCard {
                        instance_id: card.instance_id,
                        target: ActionTarget::None,
                    });
                }
            }
            HOLDOUT => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == state.turn && piece.kind == PieceKind::Pawn {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            LOYALIST => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == state.turn
                        && !self.is_royal_piece(state, piece)
                        && !matches!(
                            piece.kind,
                            PieceKind::Merchant | PieceKind::BigRook | PieceKind::Colossus
                        )
                        && !state.card_runtime.loyalist_piece_ids.contains(&piece.id)
                    {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            QUEENS_GAMBIT => {
                let protected_pawns = state.board.iter().any(|(square, piece)| {
                    piece.owner == state.turn
                        && piece.kind == PieceKind::Pawn
                        && matches!(square.file(), 2 | 3)
                        && !state.card_runtime.shielded_piece_ids.contains(&piece.id)
                });
                if protected_pawns {
                    for (square, piece) in state.board.iter() {
                        if piece.owner == state.turn
                            && piece.kind == PieceKind::Queen
                            && !self.is_royal_piece(state, piece)
                        {
                            actions.push(Action::UseCard {
                                instance_id: card.instance_id,
                                target: ActionTarget::PieceAt(square),
                            });
                        }
                    }
                }
            }
            PARRY => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == state.turn
                        && piece.kind != PieceKind::Monster
                        && !state.card_runtime.parry_piece_ids.contains(&piece.id)
                    {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            EMPTY_LUNCHBOX => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == state.turn.opponent()
                        && !self.is_royal_piece(state, piece)
                        && !state.card_runtime.empty_lunchboxes.contains_key(&piece.id)
                    {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            DISARM => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == state.turn.opponent() && !self.is_royal_piece(state, piece) {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            DICE => actions.push(Action::UseCard {
                instance_id: card.instance_id,
                target: ActionTarget::None,
            }),
            FREEZE => {
                let eligible = state
                    .board
                    .iter()
                    .filter(|(_, piece)| {
                        piece.owner == state.turn.opponent() && !self.is_royal_piece(state, *piece)
                    })
                    .collect::<Vec<_>>();
                if eligible.len() > 4
                    && eligible
                        .iter()
                        .any(|(_, piece)| !state.card_runtime.frozen.contains_key(&piece.id))
                {
                    actions.push(Action::UseCard {
                        instance_id: card.instance_id,
                        target: ActionTarget::None,
                    });
                }
            }
            REPOSITION => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == state.turn
                        && !self.is_royal_piece(state, piece)
                        && piece.kind != PieceKind::Monster
                    {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            crate::METAL => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == state.turn
                        && !self.is_royal_piece(state, piece)
                        && self.is_ranged_piece(state, piece)
                        && !state.card_runtime.metalized_piece_ids.contains(&piece.id)
                    {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            crate::CLOCKWORK => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == state.turn
                        && !self.is_royal_piece(state, piece)
                        && matches!(
                            piece.kind,
                            PieceKind::Knight | PieceKind::Bishop | PieceKind::Camel
                        )
                    {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            crate::PALADIN => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == state.turn && piece.kind == PieceKind::Knight {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            crate::HEDGEHOG => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == state.turn && piece.kind == PieceKind::Queen {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            ASSASSIN => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == state.turn && piece.kind == PieceKind::Knight {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            GRASSHOPPER => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == state.turn
                        && matches!(piece.kind, PieceKind::Knight | PieceKind::Bishop)
                    {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            CONSTITUTIONAL_MONARCHY
            | IDOL
            | JESTER
            | LOCAL_CONSCRIPTION
            | REAPER
            | WIZARD
            | SIREN
            | UNDEAD => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == state.turn && piece.kind == PieceKind::Queen {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            DRAGON
            | HERALD
            | BERSERKER
            | MAGIC_GIRL
            | SIEGE_RAM
            | SLIME
            | TRICKSTER
            | crate::PARROT
            | crate::PRINCESS
            | crate::OCTOPUS
            | crate::CAMPFIRE => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == state.turn && piece.kind == PieceKind::Rook {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            EASTERN_POLICY => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == state.turn
                        && matches!(piece.kind, PieceKind::Knight | PieceKind::Bishop)
                    {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            KNIGHTMASTER | PEGASUS => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == state.turn && piece.kind == PieceKind::Knight {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            LOG | STANDARD_BEARER => {
                for (square, piece) in state.board.iter() {
                    if piece.owner == state.turn && piece.kind == PieceKind::Pawn {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            MISSIONARY | ORDINATION => {
                let bishop_count = state
                    .board
                    .iter()
                    .filter(|(_, piece)| {
                        piece.owner == state.turn && piece.kind == PieceKind::Bishop
                    })
                    .count();
                for (square, piece) in state.board.iter() {
                    if piece.owner == state.turn
                        && piece.kind == PieceKind::Bishop
                        && (card.definition_id != ORDINATION || bishop_count >= 2)
                        && (card.definition_id != MISSIONARY
                            || !self.missionary_spawn_candidates(state, square).is_empty())
                    {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            AMAZON | HOOK => {
                let sacrifice_kind = if card.definition_id == AMAZON {
                    PieceKind::Knight
                } else {
                    PieceKind::Rook
                };
                let queens = state
                    .board
                    .iter()
                    .filter(|(_, piece)| {
                        piece.owner == state.turn && piece.kind == PieceKind::Queen
                    })
                    .collect::<Vec<_>>();
                let sacrifices = state
                    .board
                    .iter()
                    .filter(|(_, piece)| piece.owner == state.turn && piece.kind == sacrifice_kind)
                    .collect::<Vec<_>>();
                for (queen_square, _) in &queens {
                    for (sacrifice_square, _) in &sacrifices {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::Squares(vec![*queen_square, *sacrifice_square]),
                        });
                    }
                }
            }
            EAGLE => {
                if state
                    .board
                    .iter()
                    .any(|(_, piece)| piece.owner == state.turn && piece.kind == PieceKind::Knight)
                {
                    actions.push(Action::UseCard {
                        instance_id: card.instance_id,
                        target: ActionTarget::None,
                    });
                }
            }
            REFORMATION => {
                if state
                    .board
                    .iter()
                    .any(|(_, piece)| piece.owner == state.turn && piece.kind == PieceKind::Bishop)
                {
                    actions.push(Action::UseCard {
                        instance_id: card.instance_id,
                        target: ActionTarget::None,
                    });
                }
            }
            WINDMILL => {
                let bishops = state
                    .board
                    .iter()
                    .filter(|(_, piece)| {
                        piece.owner == state.turn && piece.kind == PieceKind::Bishop
                    })
                    .collect::<Vec<_>>();
                let rooks = state
                    .board
                    .iter()
                    .filter(|(_, piece)| piece.owner == state.turn && piece.kind == PieceKind::Rook)
                    .collect::<Vec<_>>();
                for (bishop_square, _) in &bishops {
                    for (rook_square, _) in &rooks {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::Squares(vec![*bishop_square, *rook_square]),
                        });
                    }
                }
            }
            JUDGMENT => {
                let maximum = state
                    .board
                    .iter()
                    .filter(|(_, piece)| {
                        piece.capture_count >= 2 && !self.is_royal_piece(state, *piece)
                    })
                    .map(|(_, piece)| piece.capture_count)
                    .max()
                    .unwrap_or(0);
                if maximum >= 2 {
                    for (square, piece) in state.board.iter() {
                        if piece.capture_count == maximum && !self.is_royal_piece(state, piece) {
                            actions.push(Action::UseCard {
                                instance_id: card.instance_id,
                                target: ActionTarget::PieceAt(square),
                            });
                        }
                    }
                }
            }
            CLEANUP_SACRIFICE => {
                for (square, piece) in state.board.iter() {
                    if piece.owner != state.turn || self.is_royal_piece(state, piece) {
                        continue;
                    }
                    let has_match = state.board.iter().any(|(_, candidate)| {
                        candidate.owner == state.turn.opponent()
                            && candidate.kind == piece.kind
                            && !self.is_royal_piece(state, candidate)
                    });
                    if has_match {
                        actions.push(Action::UseCard {
                            instance_id: card.instance_id,
                            target: ActionTarget::PieceAt(square),
                        });
                    }
                }
            }
            _ => {
                self.generate_middle_card_actions(state, card, actions);
                self.generate_end_card_actions(state, card, card.owner, actions);
            }
        }
    }
    pub(super) fn apply_opening_card(
        &self,
        state: &mut GameState,
        actor: Player,
        definition_id: &str,
        target: &ActionTarget,
        events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        let none = || {
            (target == &ActionTarget::None)
                .then_some(())
                .ok_or_else(|| {
                    EngineError::Invariant(format!("{definition_id} must not have a target"))
                })
        };
        match definition_id {
            APPRENTICE_KNIGHTS => {
                none()?;
                for origin in opening_squares(actor, &["b2", "g2"], &["b7", "g7"]) {
                    let found = state.board.iter().find(|(_, piece)| {
                        piece.owner == actor
                            && piece.kind == PieceKind::Pawn
                            && state.card_runtime.piece_origins.get(&piece.id) == Some(&origin)
                    });
                    if let Some((square, mut pawn)) = found {
                        pawn.kind = PieceKind::Squire;
                        state.board.set(square, Some(pawn));
                        self.mark_transformed_piece(state, pawn);
                    }
                }
            }
            CHECKER => {
                none()?;
                for origin in opening_squares(actor.opponent(), &["a2", "h2"], &["a7", "h7"]) {
                    let found = state.board.iter().find(|(_, piece)| {
                        piece.owner == actor.opponent()
                            && piece.kind == PieceKind::Pawn
                            && state.card_runtime.piece_origins.get(&piece.id) == Some(&origin)
                    });
                    if let Some((square, mut pawn)) = found {
                        pawn.kind = PieceKind::Checker;
                        state.board.set(square, Some(pawn));
                        self.mark_transformed_piece(state, pawn);
                    }
                }
            }
            ELEPHANT_ESCAPE => {
                none()?;
                for square in opening_squares(actor, &["a3", "h3"], &["a6", "h6"]) {
                    let piece =
                        Piece::new(actor, PieceKind::Alfil).with_id(state.allocate_piece_id());
                    state.board.set(square, Some(piece));
                    self.register_spawned_piece(state, square, piece, true);
                }
            }
            GUARD => {
                none()?;
                let king_origin = opening_square(actor, "e1", "e8");
                let pawn_origin = opening_square(actor, "e2", "e7");
                let Some((square, mut pawn)) = state.board.iter().find(|(_, piece)| {
                    piece.owner == actor
                        && matches!(piece.kind, PieceKind::Pawn | PieceKind::Fanatic)
                        && state.card_runtime.piece_origins.get(&piece.id) == Some(&pawn_origin)
                }) else {
                    return Err(EngineError::Invariant(format!(
                        "guard has no pawn originating in front of {king_origin}"
                    )));
                };
                pawn.kind = PieceKind::Guard;
                state.board.set(square, Some(pawn));
                self.mark_transformed_piece(state, pawn);
            }
            LAST_STAND => {
                none()?;
                let pawns = state
                    .board
                    .iter()
                    .filter(|(_, piece)| piece.owner == actor && piece.kind == PieceKind::Pawn)
                    .collect::<Vec<_>>();
                for (square, mut pawn) in pawns {
                    self.clear_queens_gambit_protection(state, pawn.id);
                    pawn.kind = PieceKind::Fanatic;
                    state.board.set(square, Some(pawn));
                }
            }
            QUEEN_CAVALRY => {
                none()?;
                let mut candidates = state
                    .board
                    .iter()
                    .filter(|(square, piece)| {
                        piece.owner == actor && piece.kind == PieceKind::Pawn && square.file() == 3
                    })
                    .collect::<Vec<_>>();
                candidates.sort_by_key(|(square, _)| match actor {
                    Player::White => square.rank(),
                    Player::Black => 7 - square.rank(),
                });
                let Some((square, mut pawn)) = candidates.first().copied() else {
                    return Err(EngineError::Invariant(
                        "queen cavalry has no pawn on the d-file".to_owned(),
                    ));
                };
                pawn.kind = PieceKind::Knight;
                state.card_runtime.shielded_piece_ids.remove(&pawn.id);
                state.board.set(square, Some(pawn));
                self.mark_transformed_piece(state, pawn);
            }
            DUTCH => {
                none()?;
                let pieces = state
                    .board
                    .iter()
                    .filter(|(_, piece)| {
                        piece.owner == actor
                            && matches!(
                                piece.kind,
                                PieceKind::Rook | PieceKind::Bishop | PieceKind::Knight
                            )
                    })
                    .collect::<Vec<_>>();
                for (square, mut piece) in pieces {
                    self.clear_queens_gambit_protection(state, piece.id);
                    piece.kind = PieceKind::Windmill;
                    state.board.set(square, Some(piece));
                    state
                        .card_runtime
                        .windmill_modes
                        .insert(piece.id, WindmillMode::Bishop);
                    self.mark_transformed_piece(state, piece);
                }
            }
            HORDE => {
                none()?;
                self.apply_horde_layout(state, actor, events)?;
            }
            KNIGHTMATE => {
                none()?;
                let pieces = state.board.iter().collect::<Vec<_>>();
                let mut found_king = false;
                for (square, mut piece) in pieces {
                    if piece.owner != actor {
                        continue;
                    }
                    if self.is_royal_piece(state, piece) {
                        piece.kind = PieceKind::RoyalKnight;
                        state.card_runtime.royal_piece_ids.insert(piece.id);
                        found_king = true;
                    } else if piece.kind == PieceKind::Knight {
                        piece.kind = PieceKind::Man;
                    } else {
                        continue;
                    }
                    state.board.set(square, Some(piece));
                    self.mark_transformed_piece(state, piece);
                }
                if !found_king {
                    return Err(EngineError::Invariant("knightmate has no king".to_owned()));
                }
            }
            LONDON_SYSTEM => {
                none()?;
                self.apply_london_layout(state, actor, events);
            }
            SUMMON_COLOSSUS => {
                none()?;
                self.apply_summon_colossus(state, actor, events)?;
            }
            BIG_ROOK => {
                none()?;
                self.apply_big_rook_layout(state, actor, events);
            }
            BIG_BISHOP => {
                none()?;
                self.apply_big_bishop_layout(state, actor, events);
            }
            CALLING_CARD => {
                none()?;
                let candidates = web_ordered_pieces(state)
                    .into_iter()
                    .filter(|(_, piece)| {
                        piece.owner == actor.opponent()
                            && piece.kind != PieceKind::Pawn
                            && !self.is_royal_piece(state, *piece)
                            && !matches!(piece.kind, PieceKind::Guard | PieceKind::Jester)
                    })
                    .collect::<Vec<_>>();
                let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
                events.push(Event::RandomSample {
                    purpose: "calling-card".to_owned(),
                    roll_53: roll,
                });
                let (_, piece) = candidates[index.ok_or_else(|| {
                    EngineError::Invariant("calling card has no target".to_owned())
                })?];
                state.card_runtime.calling_cards.insert(piece.id, actor);
            }
            DEMOCRACY => {
                none()?;
                state.card_runtime.democracy[actor.index()] = true;
            }
            FAST_GROWTH => {
                none()?;
                state.card_runtime.fast_growth[actor.index()] = true;
            }
            FIANCHETTO => {
                none()?;
                state.card_runtime.fianchetto[actor.index()] = true;
            }
            FIELD_PROMOTION => {
                none()?;
                state.card_runtime.field_promotion[actor.index()] = true;
            }
            HOLDOUT => {
                let (_, pawn) = piece_target(state, target)?;
                state
                    .card_runtime
                    .holdout_promotions
                    .insert(pawn.id, state.shared_turn_count().saturating_add(14));
            }
            IRON_MONARCH => {
                none()?;
                state.card_runtime.iron_monarch[actor.index()] = true;
            }
            LOYALIST => {
                let (_, piece) = piece_target(state, target)?;
                state.card_runtime.loyalist_piece_ids.insert(piece.id);
            }
            MARTYRDOM => {
                none()?;
                let bishops = state
                    .board
                    .iter()
                    .filter(|(_, piece)| piece.owner == actor && piece.kind == PieceKind::Bishop)
                    .collect::<Vec<_>>();
                for (square, bishop) in bishops {
                    self.remove_piece_for_card(
                        state,
                        square,
                        bishop,
                        actor.opponent(),
                        definition_id,
                        events,
                    );
                }
                for (_, pawn) in state
                    .board
                    .iter()
                    .filter(|(_, piece)| piece.owner == actor && piece.kind == PieceKind::Pawn)
                {
                    state.card_runtime.shielded_piece_ids.insert(pawn.id);
                }
            }
            MERCHANT_GUILD => {
                none()?;
                self.apply_merchant_guild(state, actor, events)?;
            }
            OTHERWORLD => {
                none()?;
                let pawns = web_ordered_pieces(state)
                    .into_iter()
                    .filter(|(_, piece)| piece.owner == actor && piece.kind == PieceKind::Pawn)
                    .collect::<Vec<_>>();
                let (roll, index) = state.rng_mut().web_choice_index(pawns.len());
                events.push(Event::RandomSample {
                    purpose: "otherworld-pawn".to_owned(),
                    roll_53: roll,
                });
                let (square, pawn) = pawns[index
                    .ok_or_else(|| EngineError::Invariant("otherworld has no pawn".to_owned()))?];
                let origin = state
                    .card_runtime
                    .piece_origins
                    .get(&pawn.id)
                    .copied()
                    .unwrap_or(square);
                state.board.set(square, None);
                self.clear_removed_piece_statuses(state, pawn.id);
                state
                    .card_runtime
                    .pending_otherworld
                    .push(PendingOtherworld {
                        piece_id: pawn.id,
                        owner: actor,
                        square,
                        origin,
                        due_ply: state.ply.saturating_add(28),
                    });
            }
            PAWN_CONVERSION => {
                none()?;
                state.card_runtime.pawn_conversion[actor.index()] = true;
            }
            QUEENS_GAMBIT => {
                let (queen_square, queen) = piece_target(state, target)?;
                self.remove_piece_for_card(
                    state,
                    queen_square,
                    queen,
                    actor.opponent(),
                    definition_id,
                    events,
                );
                let candidate_files = (0_u8..8)
                    .filter(|file| ![2, 3, 4].contains(file))
                    .collect::<Vec<_>>();
                let (roll, index) = state.rng_mut().web_choice_index(candidate_files.len());
                events.push(Event::RandomSample {
                    purpose: "queens-gambit-file".to_owned(),
                    roll_53: roll,
                });
                let random_file = candidate_files[index.unwrap_or(0)];
                for (square, pawn) in state.board.iter().filter(|(square, piece)| {
                    piece.owner == actor
                        && piece.kind == PieceKind::Pawn
                        && (square.file() == 3 || square.file() == random_file)
                }) {
                    let _ = square;
                    state.card_runtime.protected_piece_ids.insert(pawn.id);
                    state.card_runtime.queens_gambit_protection.insert(pawn.id);
                }
            }
            QXE1 => {
                none()?;
                self.apply_qxe1(state, actor, events)?;
            }
            THIEF => {
                none()?;
                let candidates = web_ordered_pieces(state)
                    .into_iter()
                    .filter(|(_, piece)| {
                        piece.owner == actor
                            && piece.kind == PieceKind::Queen
                            && !self.is_royal_piece(state, *piece)
                    })
                    .collect::<Vec<_>>();
                let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
                events.push(Event::RandomSample {
                    purpose: "thief-target".to_owned(),
                    roll_53: roll,
                });
                let (_, mut piece) = candidates[index.ok_or_else(|| {
                    EngineError::Invariant("thief has no nonroyal queen".to_owned())
                })?];
                let square = state
                    .board
                    .find_by_id(piece.id)
                    .expect("selected thief queen remains on board")
                    .0;
                self.clear_piece_kind_runtime(state, piece.id);
                piece.kind = PieceKind::Thief;
                state.board.set(square, Some(piece));
                state.card_runtime.submerged_piece_ids.insert(piece.id);
                self.mark_transformed_piece(state, piece);
            }
            crate::FALSE_START => {
                none()?;
                self.apply_false_start(state, actor);
            }
            crate::LOCUST_SWARM => {
                none()?;
                for (_, piece) in state.board.iter() {
                    if piece.owner == actor
                        && piece.kind != PieceKind::Pawn
                        && !state.card_runtime.moved_piece_ids.contains(&piece.id)
                    {
                        state.card_runtime.locust_swarm_piece_ids.insert(piece.id);
                    }
                }
            }
            _ => unreachable!("opening card dispatcher received {definition_id}"),
        }
        Ok(())
    }
    pub(super) fn apply_card_definition(
        &self,
        state: &mut GameState,
        definition_id: &str,
        owner: Player,
        events: &mut Vec<Event>,
    ) {
        match definition_id {
            crate::IQP => state.card_runtime.october.iqp_pending[owner.index()] = true,
            crate::HOPE => state.card_runtime.october.hope[owner.index()] = true,
            crate::ASSEMBLY => state.card_runtime.october.assembly[owner.index()] = true,
            crate::HIGHLANDER => self.resolve_highlander_victory(state),
            BRUTUS => {
                let candidates = web_ordered_pieces(state)
                    .into_iter()
                    .filter(|(_, piece)| {
                        piece.owner == owner
                            && piece.kind == PieceKind::Rook
                            && !self.is_royal_piece(state, *piece)
                    })
                    .collect::<Vec<_>>();
                let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
                events.push(Event::RandomSample {
                    purpose: "brutus-target".to_owned(),
                    roll_53: roll,
                });
                if let Some(index) = index {
                    let (square, mut piece) = candidates[index];
                    self.clear_piece_kind_runtime(state, piece.id);
                    piece.kind = PieceKind::Brutus;
                    state.board.set(square, Some(piece));
                    self.mark_transformed_piece(state, piece);
                }
            }
            crate::RECURRENCE => {
                let candidates = web_ordered_pieces(state)
                    .into_iter()
                    .filter(|(_, piece)| {
                        piece.owner == owner
                            && piece.kind != PieceKind::Pawn
                            && !state.card_runtime.recurrence_piece_ids.contains(&piece.id)
                    })
                    .collect::<Vec<_>>();
                let (roll, index) = state.rng_mut().web_choice_index(candidates.len());
                events.push(Event::RandomSample {
                    purpose: "recurrence-target".to_owned(),
                    roll_53: roll,
                });
                if let Some(index) = index {
                    state
                        .card_runtime
                        .recurrence_piece_ids
                        .insert(candidates[index].1.id);
                }
            }
            crate::MAJESTY => state.card_runtime.majesty[owner.index()] = true,
            KING_OF_THE_HILL => state.card_runtime.hill_king[owner.index()] = true,
            crate::KILLER_KING => state.card_runtime.killer_king[owner.index()] = true,
            crate::PROFICIENCY => state.card_runtime.proficiency[owner.index()] = true,
            crate::DISASSEMBLY => {}
            crate::MUTATION => state.card_runtime.mutation[owner.index()] = true,
            crate::LONG_EN_PASSANT => state.card_runtime.long_en_passant[owner.index()] = true,
            crate::VANGUARD => state.card_runtime.vanguard[owner.index()] = true,
            crate::INFILTRATION => state.card_runtime.infiltration[owner.index()] = true,
            crate::RESOLVE => state.card_runtime.resolve[owner.index()] = true,
            GENEVA_CONVENTION => state.card_runtime.geneva_convention[owner.index()] = true,
            SPRINT => state.card_runtime.pawn_sprint[owner.index()] = true,
            EARLY_PROMOTION => state.card_runtime.early_promotion[owner.index()] = true,
            INITIATIVE => {
                let target = owner.opponent();
                state.card_runtime.initiative[target.index()] = Some(InitiativeStatus {
                    by: owner,
                    start_turn: state.turns_taken[target.index()],
                    limit: if state.ruleset_id == crate::LATEST_RULESET_ID {
                        8
                    } else {
                        10
                    },
                });
            }
            BACKWARD_KNIGHT => state.card_runtime.backward_knight[owner.index()] = true,
            CORNER_KICK => state.card_runtime.corner_kick[owner.index()] = true,
            FILE_SURGE => state.card_runtime.file_surge[owner.index()] = true,
            HORSE_RIDING => {
                let knights = state
                    .board
                    .iter()
                    .filter(|(_, piece)| piece.owner == owner && piece.kind == PieceKind::Knight)
                    .collect::<Vec<_>>();
                for (square, piece) in knights {
                    self.delete_piece_for_card(state, square, piece, owner, HORSE_RIDING, events);
                }
                state.card_runtime.horse_riding[owner.index()] = true;
            }
            INJURY => state.card_runtime.knight_injury[owner.opponent().index()] = true,
            LEAP => state.card_runtime.pawn_leap[owner.index()] = true,
            RADICAL_CHARGE => state.card_runtime.radical_charge[owner.index()] = true,
            RETREAT => state.card_runtime.retreat[owner.index()] = true,
            ROOK_LIFT => state.card_runtime.rook_lift[owner.index()] = true,
            ENCOURAGEMENT => state.card_runtime.encouragement[owner.index()] = true,
            EXHAUSTION => state.card_runtime.exhaustion[owner.opponent().index()] = true,
            GHOST => {
                for (_, piece) in state.board.iter() {
                    if piece.owner == owner && piece.kind == PieceKind::Pawn {
                        state.card_runtime.ghost_piece_ids.insert(piece.id);
                    }
                }
            }
            OVERWHELM => state.card_runtime.overwhelm[owner.index()] = true,
            IMPERIAL_STUDIES => state.card_runtime.imperial_studies[owner.index()] = true,
            MAD_HORSE => state.card_runtime.mad_horse[owner.index()] = true,
            QUEEN_AFTERIMAGE => state.card_runtime.queen_afterimage[owner.index()] = true,
            RELIGIOUS_VICTORY => {
                state.card_runtime.religious_victory[owner.index()] = true;
                self.resolve_religious_victory(state);
            }
            CONVERSION => {
                let knights = state
                    .board
                    .iter()
                    .filter(|(_, piece)| piece.owner == owner && piece.kind == PieceKind::Knight)
                    .collect::<Vec<_>>();
                for (square, mut piece) in knights {
                    self.clear_queens_gambit_protection(state, piece.id);
                    piece.kind = PieceKind::Bishop;
                    state.board.set(square, Some(piece));
                    state.card_runtime.moved_piece_ids.insert(piece.id);
                }
            }
            MONGOLIAN_GAMBIT => {
                let pieces = state
                    .board
                    .iter()
                    .filter(|(_, piece)| {
                        piece.owner == owner
                            && !self.is_royal_piece(state, *piece)
                            && piece.kind != PieceKind::Merchant
                    })
                    .collect::<Vec<_>>();
                for (square, mut piece) in pieces {
                    self.clear_piece_kind_runtime(state, piece.id);
                    self.clear_queens_gambit_protection(state, piece.id);
                    piece.kind = PieceKind::Knight;
                    state.board.set(square, Some(piece));
                    state.card_runtime.moved_piece_ids.insert(piece.id);
                }
            }
            _ => {}
        }
    }
}

mod endgame;
mod midgame;
mod rules;

#[cfg(test)]
pub(super) use rules::CHESS_N_POW_30_KINDS;
