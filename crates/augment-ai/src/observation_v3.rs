use augment_core::PieceKind;

/// Stable semantic contract for the next observation encoder.
///
/// This identifier describes meaning, not a tensor layout. Changing field visibility,
/// perspective, identity handling, or normalization requires a new identifier.
pub const OBSERVATION_V3_SCHEMA_ID: &str = "augmentia.observation.v3.20260818";

/// Identifiers attached to retained V2 audit samples. They must never be used for V3 data.
pub const OBSERVATION_V2_SCHEMA_ID: &str = "augmentia.observation.canonical-v2";
pub const ACTION_V2_SCHEMA_ID: &str = "augmentia.action.fixed-32768.v2";
pub const SELF_PLAY_V2_RECORD_SCHEMA_ID: &str = "canonical_v2_selfplay_gen2_trace_v1";
pub const V2_BASELINE_MODEL_ID: &str = "augment_net_v2_canonical_baseline";

/// Public `GameState` fields reviewed by the ObservationV3 contract.
pub const GAME_STATE_SOURCE_FIELDS: [&str; 18] = [
    "ruleset_id",
    "mode",
    "board",
    "turn",
    "ply",
    "quiet_ply",
    "actions_remaining",
    "cards_used_this_turn",
    "phase",
    "status",
    "cards",
    "half_star_totals",
    "turns_taken",
    "completed_drafts",
    "deathmatch",
    "en_passant",
    "applied_rules",
    "card_runtime",
];

/// All public `CardRuntimeState` fields in source order.
///
/// Step 3 may choose different physical tensors or token collections, but it must assign every
/// entry here to an encoded or explicitly derived representation. No catch-all bucket is allowed.
pub const CARD_RUNTIME_SOURCE_FIELDS: [&str; 169] = [
    "opening_auto_card_ids",
    "hill_king",
    "geneva_convention",
    "pawn_sprint",
    "early_promotion",
    "initiative",
    "dice_locks",
    "moved_piece_ids",
    "fresh_no_capture_until",
    "disarmed",
    "frozen",
    "reposition",
    "revelation",
    "parry_piece_ids",
    "empty_lunchboxes",
    "platform",
    "herald_jump_locked_until",
    "windmill_modes",
    "log_statuses",
    "wizard_mana",
    "reaper_souls",
    "shielded_piece_ids",
    "protected_piece_ids",
    "idol_used_turn",
    "idol_rest_piece_ids",
    "delayed_hazards",
    "skip_turns",
    "piece_origins",
    "large_pieces",
    "loyalist_piece_ids",
    "royal_piece_ids",
    "calling_cards",
    "holdout_promotions",
    "pending_otherworld",
    "merchant_gold",
    "democracy",
    "fast_growth",
    "fianchetto",
    "field_promotion",
    "field_promoted_piece_ids",
    "frontline_response",
    "relay",
    "gomoku",
    "magic_girl_surge",
    "free_move_no_capture",
    "siren_exposure",
    "trickster_move_kinds",
    "undead_resurrections",
    "judgment_exiles",
    "move_replays",
    "iron_monarch",
    "pawn_conversion",
    "backward_knight",
    "breakthrough_pawns",
    "corner_kick",
    "en_passant_frenzy",
    "file_surge",
    "horse_riding",
    "knight_injury",
    "pawn_leap",
    "radical_charge",
    "retreat",
    "rook_lift",
    "encouragement",
    "exhaustion",
    "overwhelm",
    "imperial_studies",
    "mad_horse",
    "queen_afterimage",
    "religious_victory",
    "basic_training_piece_ids",
    "promotion_rush_piece_ids",
    "evasion_piece_ids",
    "ghost_piece_ids",
    "inertia_piece_ids",
    "ice_sheet",
    "severed",
    "royal_command",
    "socialism",
    "taunt",
    "armistice_remaining_ply",
    "exhaustion_last_piece",
    "exhaustion_count",
    "chameleon_piece_ids",
    "chimera_piece_ids",
    "imperial_moves",
    "mistake_card",
    "poisoned_piece_ids",
    "sacrifice_protection",
    "last_resistance",
    "coronation_protection",
    "queens_gambit_protection",
    "potion_basic_training_piece_ids",
    "poison_stun_turns",
    "potion_manner_piece_ids",
    "manner_captured_last_piece_ids",
    "potion_saturation_piece_ids",
    "saturation_capture_counts",
    "bishop_snipe",
    "explosive_piece_ids",
    "trojan_piece_ids",
    "feudal_contracts",
    "gale_due_turns",
    "pending_panic",
    "pending_premoves",
    "pending_scarecrows",
    "quantum_pending",
    "quantum_positions",
    "stealth_piece_ids",
    "submerged_piece_ids",
    "substitution",
    "switcheroo",
    "twin_bonds",
    "witch_trials",
    "pending_trolley_by",
    "active_trolley",
    "baby_bear_grow_turns",
    "bear_retaliations",
    "bribe_reverts",
    "charge_pawns",
    "coronation",
    "frenzy_pawns",
    "hallucination_until_ply",
    "moving",
    "moving_consecutive",
    "necromancy_reverts",
    "reverse_pawns_charges",
    "stake_statuses",
    "underpromotion",
    "vip_pending",
    "barricades",
    "chains",
    "collapsed_files_ranks",
    "desperado_pieces",
    "palaces",
    "portals",
    "bunkers",
    "zugzwang",
    "bina_mate",
    "final_weapon",
    "icbm_pending",
    "prophecy",
    "racing_king",
    "spies",
    "ultimatum",
    "vanish",
    "acceleration",
    "chaos_no_capture_until_ply",
    "black_holes",
    "camouflage_color",
    "conveyor",
    "cool_guy_pieces",
    "crown_pos",
    "crown_holder",
    "football_pos",
    "high_ground_squares",
    "highways",
    "macho_chess",
    "mistake_rule",
    "monochrome_chess",
    "monster_rule",
    "periodic_collapse_next",
    "rule_portals",
    "recycling",
    "rule_bombs",
    "saturation_rule",
    "transcendence",
    "winter_kingdom_next",
    "cloned_passives",
];

/// Private engine values that a model must not receive directly.
pub const INTERNAL_STATE_EXCLUSIONS: [(&str, &str); 6] = [
    ("rng", "would reveal future deterministic randomness"),
    ("next_piece_id", "allocation mechanism, not game meaning"),
    (
        "next_card_instance_id",
        "allocation mechanism, not game meaning",
    ),
    ("next_choice_id", "allocation mechanism, not game meaning"),
    (
        "turn_end_requested",
        "transient reducer control that is resolved before observation",
    ),
    (
        "repetitions",
        "history-map keys are internal; expose only current_repetition_count",
    ),
];

/// Engine-derived values included without exposing their private backing representation.
pub const DERIVED_SOURCE_FIELDS: [&str; 2] = ["shared_turn_count", "current_repetition_count"];

/// Complete, collision-free piece vocabulary for ObservationV3.
pub const PIECE_KIND_VOCABULARY: [(u8, PieceKind); 58] = [
    (1, PieceKind::Pawn),
    (2, PieceKind::Knight),
    (3, PieceKind::Bishop),
    (4, PieceKind::Rook),
    (5, PieceKind::Queen),
    (6, PieceKind::King),
    (7, PieceKind::Assassin),
    (8, PieceKind::Grasshopper),
    (9, PieceKind::Amazon),
    (10, PieceKind::PrimeMinister),
    (11, PieceKind::Dragon),
    (12, PieceKind::Cannon),
    (13, PieceKind::Herald),
    (14, PieceKind::Knightmaster),
    (15, PieceKind::Unicorn),
    (16, PieceKind::Hook),
    (17, PieceKind::Idol),
    (18, PieceKind::Jester),
    (19, PieceKind::Recruiter),
    (20, PieceKind::Reaper),
    (21, PieceKind::Wizard),
    (22, PieceKind::Alibaba),
    (23, PieceKind::Log),
    (24, PieceKind::Missionary),
    (25, PieceKind::Cardinal),
    (26, PieceKind::Protestant),
    (27, PieceKind::StandardBearer),
    (28, PieceKind::Windmill),
    (29, PieceKind::Squire),
    (30, PieceKind::BigRook),
    (31, PieceKind::Checker),
    (32, PieceKind::CheckerKing),
    (33, PieceKind::Alfil),
    (34, PieceKind::Guard),
    (35, PieceKind::Fanatic),
    (36, PieceKind::Colossus),
    (37, PieceKind::RoyalKnight),
    (38, PieceKind::Man),
    (39, PieceKind::Merchant),
    (40, PieceKind::DarkWizard),
    (41, PieceKind::Monster),
    (42, PieceKind::Scarecrow),
    (43, PieceKind::Camel),
    (44, PieceKind::Ferz),
    (45, PieceKind::Bat),
    (46, PieceKind::BabyBear),
    (47, PieceKind::Lobster),
    (48, PieceKind::Bear),
    (49, PieceKind::Vip),
    (50, PieceKind::Crown),
    (51, PieceKind::Football),
    (52, PieceKind::SiegeRam),
    (53, PieceKind::MagicGirl),
    (54, PieceKind::Berserker),
    (55, PieceKind::Slime),
    (56, PieceKind::Siren),
    (57, PieceKind::Trickster),
    (58, PieceKind::Undead),
];

#[cfg(test)]
mod tests {
    use super::*;
    use augment_core::{
        BribeRevert, CardInstance, CardRuntimeState, DeathmatchState, DelayedHazard,
        DiceLockStatus, DraftState, EmptyLunchboxStatus, EnPassantState, FeudalContract,
        FrozenStatus, InitiativeStatus, LargePieceStatus, LogStatus, NecromancyRevert,
        PendingChoice, PendingIcbm, PendingOtherworld, PendingPanic, PendingPremove,
        PendingScarecrow, PendingVip, Piece, PlatformRuleState, RepositionStatus,
        RoyalCommandStatus, TimedPieceStatus, TrolleyDilemma, TwinBond,
    };
    use std::collections::BTreeSet;

    #[test]
    fn source_field_inventories_are_unique_and_complete_at_the_frozen_count() {
        assert_eq!(GAME_STATE_SOURCE_FIELDS.len(), 18);
        assert_eq!(CARD_RUNTIME_SOURCE_FIELDS.len(), 169);
        assert_eq!(
            GAME_STATE_SOURCE_FIELDS
                .iter()
                .collect::<BTreeSet<_>>()
                .len(),
            GAME_STATE_SOURCE_FIELDS.len()
        );
        assert_eq!(
            CARD_RUNTIME_SOURCE_FIELDS
                .iter()
                .collect::<BTreeSet<_>>()
                .len(),
            CARD_RUNTIME_SOURCE_FIELDS.len()
        );
    }

    #[test]
    fn piece_vocabulary_matches_every_stable_code_without_collisions() {
        let codes = PIECE_KIND_VOCABULARY
            .iter()
            .map(|(code, kind)| {
                assert_eq!(*code, kind.stable_code());
                *code
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(codes.len(), PIECE_KIND_VOCABULARY.len());
        assert_eq!(codes.first(), Some(&1));
        assert_eq!(codes.last(), Some(&58));
    }

    #[test]
    fn card_runtime_destructure_forces_contract_review_when_a_field_is_added() {
        let runtime = CardRuntimeState::default();
        let CardRuntimeState {
            opening_auto_card_ids: _,
            hill_king: _,
            geneva_convention: _,
            pawn_sprint: _,
            early_promotion: _,
            initiative: _,
            dice_locks: _,
            moved_piece_ids: _,
            fresh_no_capture_until: _,
            disarmed: _,
            frozen: _,
            reposition: _,
            revelation: _,
            parry_piece_ids: _,
            empty_lunchboxes: _,
            platform: _,
            herald_jump_locked_until: _,
            windmill_modes: _,
            log_statuses: _,
            wizard_mana: _,
            reaper_souls: _,
            shielded_piece_ids: _,
            protected_piece_ids: _,
            idol_used_turn: _,
            idol_rest_piece_ids: _,
            delayed_hazards: _,
            skip_turns: _,
            piece_origins: _,
            large_pieces: _,
            loyalist_piece_ids: _,
            royal_piece_ids: _,
            calling_cards: _,
            holdout_promotions: _,
            pending_otherworld: _,
            merchant_gold: _,
            democracy: _,
            fast_growth: _,
            fianchetto: _,
            field_promotion: _,
            field_promoted_piece_ids: _,
            frontline_response: _,
            relay: _,
            gomoku: _,
            magic_girl_surge: _,
            free_move_no_capture: _,
            siren_exposure: _,
            trickster_move_kinds: _,
            undead_resurrections: _,
            judgment_exiles: _,
            move_replays: _,
            iron_monarch: _,
            pawn_conversion: _,
            backward_knight: _,
            breakthrough_pawns: _,
            corner_kick: _,
            en_passant_frenzy: _,
            file_surge: _,
            horse_riding: _,
            knight_injury: _,
            pawn_leap: _,
            radical_charge: _,
            retreat: _,
            rook_lift: _,
            encouragement: _,
            exhaustion: _,
            overwhelm: _,
            imperial_studies: _,
            mad_horse: _,
            queen_afterimage: _,
            religious_victory: _,
            basic_training_piece_ids: _,
            promotion_rush_piece_ids: _,
            evasion_piece_ids: _,
            ghost_piece_ids: _,
            inertia_piece_ids: _,
            ice_sheet: _,
            severed: _,
            royal_command: _,
            socialism: _,
            taunt: _,
            armistice_remaining_ply: _,
            exhaustion_last_piece: _,
            exhaustion_count: _,
            chameleon_piece_ids: _,
            chimera_piece_ids: _,
            imperial_moves: _,
            mistake_card: _,
            poisoned_piece_ids: _,
            sacrifice_protection: _,
            last_resistance: _,
            coronation_protection: _,
            queens_gambit_protection: _,
            potion_basic_training_piece_ids: _,
            poison_stun_turns: _,
            potion_manner_piece_ids: _,
            manner_captured_last_piece_ids: _,
            potion_saturation_piece_ids: _,
            saturation_capture_counts: _,
            bishop_snipe: _,
            explosive_piece_ids: _,
            trojan_piece_ids: _,
            feudal_contracts: _,
            gale_due_turns: _,
            pending_panic: _,
            pending_premoves: _,
            pending_scarecrows: _,
            quantum_pending: _,
            quantum_positions: _,
            stealth_piece_ids: _,
            submerged_piece_ids: _,
            substitution: _,
            switcheroo: _,
            twin_bonds: _,
            witch_trials: _,
            pending_trolley_by: _,
            active_trolley: _,
            baby_bear_grow_turns: _,
            bear_retaliations: _,
            bribe_reverts: _,
            charge_pawns: _,
            coronation: _,
            frenzy_pawns: _,
            hallucination_until_ply: _,
            moving: _,
            moving_consecutive: _,
            necromancy_reverts: _,
            reverse_pawns_charges: _,
            stake_statuses: _,
            underpromotion: _,
            vip_pending: _,
            barricades: _,
            chains: _,
            collapsed_files_ranks: _,
            desperado_pieces: _,
            palaces: _,
            portals: _,
            bunkers: _,
            zugzwang: _,
            bina_mate: _,
            final_weapon: _,
            icbm_pending: _,
            prophecy: _,
            racing_king: _,
            spies: _,
            ultimatum: _,
            vanish: _,
            acceleration: _,
            chaos_no_capture_until_ply: _,
            black_holes: _,
            camouflage_color: _,
            conveyor: _,
            cool_guy_pieces: _,
            crown_pos: _,
            crown_holder: _,
            football_pos: _,
            high_ground_squares: _,
            highways: _,
            macho_chess: _,
            mistake_rule: _,
            monochrome_chess: _,
            monster_rule: _,
            periodic_collapse_next: _,
            rule_portals: _,
            recycling: _,
            rule_bombs: _,
            saturation_rule: _,
            transcendence: _,
            winter_kingdom_next: _,
            cloned_passives: _,
        } = &runtime;
    }

    #[test]
    fn nested_payload_destructures_force_contract_review_when_a_field_is_added() {
        fn piece(value: &Piece) {
            let Piece {
                id: _,
                owner: _,
                kind: _,
                capture_count: _,
            } = value;
        }
        fn draft(value: &DraftState) {
            let DraftState {
                phase: _,
                offers: _,
                resume_turn: _,
            } = value;
        }
        fn choice(value: &PendingChoice) {
            let PendingChoice {
                id: _,
                owner: _,
                source_card_instance: _,
                handler_id: _,
                options: _,
                continuation: _,
            } = value;
        }
        fn card(value: &CardInstance) {
            let CardInstance {
                instance_id: _,
                definition_id: _,
                owner: _,
                half_stars: _,
                charges: _,
                available_after_turn: _,
            } = value;
        }
        fn deathmatch(value: &DeathmatchState) {
            let DeathmatchState {
                started_at_shared_turn: _,
                quiet_full_turns: _,
                interval_full_turns: _,
                progress_this_round: _,
            } = value;
        }
        fn en_passant(value: &EnPassantState) {
            let EnPassantState {
                target: _,
                captured: _,
                pawn_owner: _,
            } = value;
        }
        fn lunchbox(value: &EmptyLunchboxStatus) {
            let EmptyLunchboxStatus {
                by: _,
                deadline_turn: _,
            } = value;
        }
        fn platform(value: &PlatformRuleState) {
            let PlatformRuleState {
                cell: _,
                spawned_at: _,
                next_at: _,
                triggered_piece_ids: _,
                nonce: _,
            } = value;
        }
        fn initiative(value: &InitiativeStatus) {
            let InitiativeStatus {
                by: _,
                start_turn: _,
                limit: _,
            } = value;
        }
        fn dice(value: &DiceLockStatus) {
            let DiceLockStatus {
                kind: _,
                remaining_turns: _,
            } = value;
        }
        fn frozen(value: &FrozenStatus) {
            let FrozenStatus {
                source: _,
                remaining_turns: _,
            } = value;
        }
        fn reposition(value: &RepositionStatus) {
            let RepositionStatus {
                piece_id: _,
                used: _,
            } = value;
        }
        fn log_status(value: &LogStatus) {
            let LogStatus {
                file_delta: _,
                rank_delta: _,
                roll_after_turn: _,
            } = value;
        }
        fn hazard(value: &DelayedHazard) {
            let DelayedHazard {
                kind: _,
                owner: _,
                trigger_after: _,
                squares: _,
            } = value;
        }
        fn otherworld(value: &PendingOtherworld) {
            let PendingOtherworld {
                piece_id: _,
                owner: _,
                square: _,
                origin: _,
                due_ply: _,
            } = value;
        }
        fn large_piece(value: &LargePieceStatus) {
            let LargePieceStatus {
                anchor: _,
                hp: _,
                max_hp: _,
            } = value;
        }
        fn timed_piece(value: &TimedPieceStatus) {
            let TimedPieceStatus {
                by: _,
                remaining_turns: _,
            } = value;
        }
        fn royal_command(value: &RoyalCommandStatus) {
            let RoyalCommandStatus {
                starts_at_turn: _,
                expires_at_turn: _,
            } = value;
        }
        fn feudal(value: &FeudalContract) {
            let FeudalContract {
                owner: _,
                pawn_id: _,
                guardian_id: _,
            } = value;
        }
        fn panic(value: &PendingPanic) {
            let PendingPanic {
                owner: _,
                piece_ids: _,
            } = value;
        }
        fn premove(value: &PendingPremove) {
            let PendingPremove {
                owner: _,
                trigger_player: _,
                trigger_turn: _,
                moves: _,
            } = value;
        }
        fn scarecrow(value: &PendingScarecrow) {
            let PendingScarecrow {
                owner: _,
                square: _,
                remaining_ply: _,
            } = value;
        }
        fn twin(value: &TwinBond) {
            let TwinBond {
                first_id: _,
                second_id: _,
            } = value;
        }
        fn trolley(value: &TrolleyDilemma) {
            let TrolleyDilemma { by: _, bundles: _ } = value;
        }
        fn bribe(value: &BribeRevert) {
            let BribeRevert {
                piece_id: _,
                due_turn: _,
            } = value;
        }
        fn necromancy(value: &NecromancyRevert) {
            let NecromancyRevert {
                piece_id: _,
                due_ply: _,
            } = value;
        }
        fn vip(value: &PendingVip) {
            let PendingVip {
                piece_id: _,
                owner: _,
                due_ply: _,
            } = value;
        }
        fn icbm(value: &PendingIcbm) {
            let PendingIcbm {
                queen_id: _,
                owner: _,
                due_turn: _,
            } = value;
        }

        let _review_gates = (
            piece as fn(&Piece),
            draft as fn(&DraftState),
            choice as fn(&PendingChoice),
            card as fn(&CardInstance),
            deathmatch as fn(&DeathmatchState),
            en_passant as fn(&EnPassantState),
            lunchbox as fn(&EmptyLunchboxStatus),
            platform as fn(&PlatformRuleState),
            initiative as fn(&InitiativeStatus),
            dice as fn(&DiceLockStatus),
            frozen as fn(&FrozenStatus),
            reposition as fn(&RepositionStatus),
            log_status as fn(&LogStatus),
            hazard as fn(&DelayedHazard),
            otherworld as fn(&PendingOtherworld),
            large_piece as fn(&LargePieceStatus),
            timed_piece as fn(&TimedPieceStatus),
            royal_command as fn(&RoyalCommandStatus),
            feudal as fn(&FeudalContract),
            panic as fn(&PendingPanic),
            premove as fn(&PendingPremove),
            scarecrow as fn(&PendingScarecrow),
            twin as fn(&TwinBond),
            trolley as fn(&TrolleyDilemma),
            bribe as fn(&BribeRevert),
            necromancy as fn(&NecromancyRevert),
            vip as fn(&PendingVip),
            icbm as fn(&PendingIcbm),
        );
    }
}
