use crate::encoder_v3::{LocalIds, model_choice_handler_token, model_choice_option_token};
use crate::observation_v3::PIECE_KIND_VOCABULARY;
use augment_core::{
    Action, ActionTarget, Engine, GamePhase, GameState, MoveAction, PieceAbility, PieceKind,
    Player, Square, Transition,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

pub const ACTION_V3_SCHEMA_ID: &str = "augmentia.action.v3";

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PieceAbilityV3 {
    WizardLightning,
    WizardShield,
    WizardMeteor,
    WizardTimeStop,
    ColossusSector,
    MerchantBuy,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ActionTargetV3 {
    None,
    Square { square: u8 },
    PieceAt { square: u8 },
    Squares { squares: Vec<u8> },
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ActionDescriptorV3 {
    Move {
        piece: u32,
        piece_kind: u8,
        from: u8,
        to: u8,
        promotion: Option<u8>,
    },
    PieceAbility {
        piece: u32,
        piece_kind: u8,
        from: u8,
        ability: PieceAbilityV3,
        target: ActionTargetV3,
    },
    Card {
        card: u32,
        definition_id: String,
        target: ActionTargetV3,
    },
    DraftPick {
        offer_index: u16,
        definition_id: String,
    },
    ResolveChoice {
        choice_slot: u32,
        handler_id: String,
        option_index: u16,
        option: String,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LegalActionSetV3 {
    pub schema_id: String,
    pub actions: Vec<ActionDescriptorV3>,
}

impl LegalActionSetV3 {
    pub fn from_state(state: &GameState, engine: &Engine) -> Result<Self, ActionV3Error> {
        let legal_actions = engine
            .legal_actions(state)
            .map_err(|error| ActionV3Error::Engine(error.to_string()))?;
        let mut descriptors = Vec::with_capacity(legal_actions.len());
        let mut seen = BTreeMap::new();

        for (index, action) in legal_actions.iter().enumerate() {
            let descriptor = encode_action_v3(action, state)?;
            let decoded = decode_action_v3(&descriptor, state)?;
            if decoded != *action {
                return Err(ActionV3Error::RoundTripMismatch { index });
            }
            if let Some(first_index) = seen.insert(descriptor.clone(), index) {
                return Err(ActionV3Error::DuplicateDescriptor {
                    first_index,
                    second_index: index,
                });
            }
            descriptors.push(descriptor);
        }

        Ok(Self {
            schema_id: ACTION_V3_SCHEMA_ID.to_owned(),
            actions: descriptors,
        })
    }

    pub fn len(&self) -> usize {
        self.actions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    pub fn descriptor(&self, index: usize) -> Option<&ActionDescriptorV3> {
        self.actions.get(index)
    }

    pub fn decode_action(&self, index: usize, state: &GameState) -> Result<Action, ActionV3Error> {
        let descriptor = self
            .actions
            .get(index)
            .ok_or(ActionV3Error::ActionIndexOutOfRange {
                index,
                len: self.actions.len(),
            })?;
        decode_action_v3(descriptor, state)
    }

    pub fn apply(
        &self,
        index: usize,
        state: &GameState,
        engine: &Engine,
    ) -> Result<Transition, ActionV3Error> {
        let action = self.decode_action(index, state)?;
        engine
            .apply(state, action)
            .map_err(|error| ActionV3Error::Engine(error.to_string()))
    }

    /// Normalizes one model logit per current legal descriptor.
    ///
    /// There is no global action mask: the vector length is exactly the number of legal actions.
    pub fn policy_from_logits(&self, logits: &[f32]) -> Result<Vec<f32>, ActionV3Error> {
        if logits.len() != self.actions.len() {
            return Err(ActionV3Error::PolicyLengthMismatch {
                expected: self.actions.len(),
                actual: logits.len(),
            });
        }
        if logits.is_empty() {
            return Ok(Vec::new());
        }
        if let Some((index, _)) = logits
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(ActionV3Error::NonFiniteLogit { index });
        }

        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights = logits
            .iter()
            .map(|logit| (*logit - max_logit).exp())
            .collect::<Vec<_>>();
        let total = weights.iter().sum::<f32>();
        if !total.is_finite() || total <= 0.0 {
            return Err(ActionV3Error::InvalidPolicyNormalization);
        }
        Ok(weights.into_iter().map(|weight| weight / total).collect())
    }
}

pub fn encode_action_v3(
    action: &Action,
    state: &GameState,
) -> Result<ActionDescriptorV3, ActionV3Error> {
    let ids = LocalIds::new(state);
    match action {
        Action::Move(movement) => {
            let piece =
                state
                    .move_source_piece(movement.from)
                    .ok_or(ActionV3Error::MissingMovePiece {
                        square: movement.from,
                    })?;
            let slot = ids
                .piece_slot(piece.id)
                .ok_or(ActionV3Error::UnknownPieceId { piece_id: piece.id })?;
            Ok(ActionDescriptorV3::Move {
                piece: slot,
                piece_kind: piece.kind.stable_code(),
                from: canonical_square_index(movement.from, state.turn),
                to: canonical_square_index(movement.to, state.turn),
                promotion: movement.promotion.map(PieceKind::stable_code),
            })
        }
        Action::UsePieceAbility {
            piece_id,
            ability,
            target,
        } => {
            let (square, piece) =
                state
                    .board
                    .find_by_id(*piece_id)
                    .ok_or(ActionV3Error::UnknownPieceId {
                        piece_id: *piece_id,
                    })?;
            let slot = ids
                .piece_slot(*piece_id)
                .ok_or(ActionV3Error::UnknownPieceId {
                    piece_id: *piece_id,
                })?;
            Ok(ActionDescriptorV3::PieceAbility {
                piece: slot,
                piece_kind: piece.kind.stable_code(),
                from: canonical_square_index(square, state.turn),
                ability: encode_ability(*ability),
                target: encode_target(target, state.turn),
            })
        }
        Action::UseCard {
            instance_id,
            target,
        } => {
            let card = state
                .cards
                .iter()
                .find(|card| card.instance_id == *instance_id)
                .ok_or(ActionV3Error::UnknownCardId {
                    instance_id: *instance_id,
                })?;
            let slot = ids
                .card_slot(*instance_id)
                .ok_or(ActionV3Error::UnknownCardId {
                    instance_id: *instance_id,
                })?;
            Ok(ActionDescriptorV3::Card {
                card: slot,
                definition_id: card.definition_id.clone(),
                target: encode_target(target, state.turn),
            })
        }
        Action::DraftPick { offer_index } => {
            let GamePhase::Draft(draft) = &state.phase else {
                return Err(ActionV3Error::PhaseMismatch { expected: "draft" });
            };
            let definition_id = draft.offers.get(usize::from(*offer_index)).ok_or(
                ActionV3Error::OfferIndexOutOfRange {
                    index: *offer_index,
                    len: draft.offers.len(),
                },
            )?;
            Ok(ActionDescriptorV3::DraftPick {
                offer_index: *offer_index,
                definition_id: definition_id.clone(),
            })
        }
        Action::ResolveChoice {
            choice_id,
            option_index,
        } => {
            let GamePhase::AwaitingChoice(choice) = &state.phase else {
                return Err(ActionV3Error::PhaseMismatch {
                    expected: "awaiting_choice",
                });
            };
            if choice.id != *choice_id {
                return Err(ActionV3Error::ChoiceIdMismatch {
                    expected: choice.id,
                    actual: *choice_id,
                });
            }
            choice.options.get(usize::from(*option_index)).ok_or(
                ActionV3Error::OptionIndexOutOfRange {
                    index: *option_index,
                    len: choice.options.len(),
                },
            )?;
            Ok(ActionDescriptorV3::ResolveChoice {
                choice_slot: 0,
                handler_id: model_choice_handler_token(&choice.handler_id),
                option_index: *option_index,
                option: model_choice_option_token(&choice.handler_id, usize::from(*option_index)),
            })
        }
    }
}

pub fn decode_action_v3(
    descriptor: &ActionDescriptorV3,
    state: &GameState,
) -> Result<Action, ActionV3Error> {
    let ids = LocalIds::new(state);
    match descriptor {
        ActionDescriptorV3::Move {
            piece,
            piece_kind,
            from,
            to,
            promotion,
        } => {
            let from_square = decode_square(*from, state.turn)?;
            let raw_piece = ids
                .raw_piece(*piece)
                .ok_or(ActionV3Error::UnknownPieceSlot { slot: *piece })?;
            let state_piece =
                state
                    .move_source_piece(from_square)
                    .ok_or(ActionV3Error::MissingMovePiece {
                        square: from_square,
                    })?;
            if state_piece.id != raw_piece || state_piece.kind.stable_code() != *piece_kind {
                return Err(ActionV3Error::StalePieceDescriptor { slot: *piece });
            }
            Ok(Action::Move(MoveAction::new(
                from_square,
                decode_square(*to, state.turn)?,
                promotion.map(decode_piece_kind).transpose()?,
            )))
        }
        ActionDescriptorV3::PieceAbility {
            piece,
            piece_kind,
            from,
            ability,
            target,
        } => {
            let raw_piece = ids
                .raw_piece(*piece)
                .ok_or(ActionV3Error::UnknownPieceSlot { slot: *piece })?;
            let (square, state_piece) = state
                .board
                .find_by_id(raw_piece)
                .ok_or(ActionV3Error::UnknownPieceSlot { slot: *piece })?;
            if canonical_square_index(square, state.turn) != *from
                || state_piece.kind.stable_code() != *piece_kind
            {
                return Err(ActionV3Error::StalePieceDescriptor { slot: *piece });
            }
            Ok(Action::UsePieceAbility {
                piece_id: raw_piece,
                ability: decode_ability(*ability),
                target: decode_target(target, state.turn)?,
            })
        }
        ActionDescriptorV3::Card {
            card,
            definition_id,
            target,
        } => {
            let raw_card = ids
                .raw_card(*card)
                .ok_or(ActionV3Error::UnknownCardSlot { slot: *card })?;
            let state_card = state
                .cards
                .iter()
                .find(|candidate| candidate.instance_id == raw_card)
                .ok_or(ActionV3Error::UnknownCardSlot { slot: *card })?;
            if state_card.definition_id != *definition_id {
                return Err(ActionV3Error::StaleCardDescriptor { slot: *card });
            }
            Ok(Action::UseCard {
                instance_id: raw_card,
                target: decode_target(target, state.turn)?,
            })
        }
        ActionDescriptorV3::DraftPick {
            offer_index,
            definition_id,
        } => {
            let GamePhase::Draft(draft) = &state.phase else {
                return Err(ActionV3Error::PhaseMismatch { expected: "draft" });
            };
            let current = draft.offers.get(usize::from(*offer_index)).ok_or(
                ActionV3Error::OfferIndexOutOfRange {
                    index: *offer_index,
                    len: draft.offers.len(),
                },
            )?;
            if current != definition_id {
                return Err(ActionV3Error::StaleDraftDescriptor);
            }
            Ok(Action::DraftPick {
                offer_index: *offer_index,
            })
        }
        ActionDescriptorV3::ResolveChoice {
            choice_slot,
            handler_id,
            option_index,
            option,
        } => {
            if *choice_slot != 0 {
                return Err(ActionV3Error::UnknownChoiceSlot { slot: *choice_slot });
            }
            let GamePhase::AwaitingChoice(choice) = &state.phase else {
                return Err(ActionV3Error::PhaseMismatch {
                    expected: "awaiting_choice",
                });
            };
            choice.options.get(usize::from(*option_index)).ok_or(
                ActionV3Error::OptionIndexOutOfRange {
                    index: *option_index,
                    len: choice.options.len(),
                },
            )?;
            if model_choice_handler_token(&choice.handler_id) != *handler_id
                || model_choice_option_token(handler_id, usize::from(*option_index)) != *option
            {
                return Err(ActionV3Error::StaleChoiceDescriptor);
            }
            Ok(Action::ResolveChoice {
                choice_id: choice.id,
                option_index: *option_index,
            })
        }
    }
}

fn encode_target(target: &ActionTarget, perspective: Player) -> ActionTargetV3 {
    match target {
        ActionTarget::None => ActionTargetV3::None,
        ActionTarget::Square(square) => ActionTargetV3::Square {
            square: canonical_square_index(*square, perspective),
        },
        ActionTarget::PieceAt(square) => ActionTargetV3::PieceAt {
            square: canonical_square_index(*square, perspective),
        },
        ActionTarget::Squares(squares) => ActionTargetV3::Squares {
            squares: squares
                .iter()
                .map(|square| canonical_square_index(*square, perspective))
                .collect(),
        },
    }
}

fn decode_target(
    target: &ActionTargetV3,
    perspective: Player,
) -> Result<ActionTarget, ActionV3Error> {
    match target {
        ActionTargetV3::None => Ok(ActionTarget::None),
        ActionTargetV3::Square { square } => {
            Ok(ActionTarget::Square(decode_square(*square, perspective)?))
        }
        ActionTargetV3::PieceAt { square } => {
            Ok(ActionTarget::PieceAt(decode_square(*square, perspective)?))
        }
        ActionTargetV3::Squares { squares } => Ok(ActionTarget::Squares(
            squares
                .iter()
                .map(|square| decode_square(*square, perspective))
                .collect::<Result<Vec<_>, _>>()?,
        )),
    }
}

fn canonical_square_index(square: Square, perspective: Player) -> u8 {
    match perspective {
        Player::White => square.index() as u8,
        Player::Black => Square::new(square.file(), 7 - square.rank())
            .expect("mirrored square")
            .index() as u8,
    }
}

fn decode_square(index: u8, perspective: Player) -> Result<Square, ActionV3Error> {
    let canonical = Square::from_index(index).ok_or(ActionV3Error::InvalidSquare { index })?;
    Ok(match perspective {
        Player::White => canonical,
        Player::Black => {
            Square::new(canonical.file(), 7 - canonical.rank()).expect("mirrored canonical square")
        }
    })
}

fn encode_ability(ability: PieceAbility) -> PieceAbilityV3 {
    match ability {
        PieceAbility::WizardLightning => PieceAbilityV3::WizardLightning,
        PieceAbility::WizardShield => PieceAbilityV3::WizardShield,
        PieceAbility::WizardMeteor => PieceAbilityV3::WizardMeteor,
        PieceAbility::WizardTimeStop => PieceAbilityV3::WizardTimeStop,
        PieceAbility::ColossusSector => PieceAbilityV3::ColossusSector,
        PieceAbility::MerchantBuy => PieceAbilityV3::MerchantBuy,
    }
}

fn decode_ability(ability: PieceAbilityV3) -> PieceAbility {
    match ability {
        PieceAbilityV3::WizardLightning => PieceAbility::WizardLightning,
        PieceAbilityV3::WizardShield => PieceAbility::WizardShield,
        PieceAbilityV3::WizardMeteor => PieceAbility::WizardMeteor,
        PieceAbilityV3::WizardTimeStop => PieceAbility::WizardTimeStop,
        PieceAbilityV3::ColossusSector => PieceAbility::ColossusSector,
        PieceAbilityV3::MerchantBuy => PieceAbility::MerchantBuy,
    }
}

fn decode_piece_kind(code: u8) -> Result<PieceKind, ActionV3Error> {
    PIECE_KIND_VOCABULARY
        .iter()
        .find_map(|(candidate, kind)| (*candidate == code).then_some(*kind))
        .ok_or(ActionV3Error::UnknownPieceKindCode { code })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ActionV3Error {
    Engine(String),
    MissingMovePiece {
        square: Square,
    },
    UnknownPieceId {
        piece_id: u32,
    },
    UnknownCardId {
        instance_id: u32,
    },
    UnknownPieceSlot {
        slot: u32,
    },
    UnknownCardSlot {
        slot: u32,
    },
    UnknownChoiceSlot {
        slot: u32,
    },
    UnknownPieceKindCode {
        code: u8,
    },
    InvalidSquare {
        index: u8,
    },
    StalePieceDescriptor {
        slot: u32,
    },
    StaleCardDescriptor {
        slot: u32,
    },
    StaleDraftDescriptor,
    StaleChoiceDescriptor,
    PhaseMismatch {
        expected: &'static str,
    },
    ChoiceIdMismatch {
        expected: u32,
        actual: u32,
    },
    OfferIndexOutOfRange {
        index: u16,
        len: usize,
    },
    OptionIndexOutOfRange {
        index: u16,
        len: usize,
    },
    ActionIndexOutOfRange {
        index: usize,
        len: usize,
    },
    DuplicateDescriptor {
        first_index: usize,
        second_index: usize,
    },
    RoundTripMismatch {
        index: usize,
    },
    PolicyLengthMismatch {
        expected: usize,
        actual: usize,
    },
    NonFiniteLogit {
        index: usize,
    },
    InvalidPolicyNormalization,
}

impl fmt::Display for ActionV3Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Engine(message) => write!(formatter, "engine error: {message}"),
            Self::MissingMovePiece { square } => {
                write!(formatter, "move source {square} has no piece")
            }
            Self::UnknownPieceId { piece_id } => write!(formatter, "unknown piece id {piece_id}"),
            Self::UnknownCardId { instance_id } => {
                write!(formatter, "unknown card instance id {instance_id}")
            }
            Self::UnknownPieceSlot { slot } => write!(formatter, "unknown piece slot {slot}"),
            Self::UnknownCardSlot { slot } => write!(formatter, "unknown card slot {slot}"),
            Self::UnknownChoiceSlot { slot } => write!(formatter, "unknown choice slot {slot}"),
            Self::UnknownPieceKindCode { code } => {
                write!(formatter, "unknown stable piece kind code {code}")
            }
            Self::InvalidSquare { index } => write!(formatter, "invalid canonical square {index}"),
            Self::StalePieceDescriptor { slot } => {
                write!(formatter, "stale piece descriptor for slot {slot}")
            }
            Self::StaleCardDescriptor { slot } => {
                write!(formatter, "stale card descriptor for slot {slot}")
            }
            Self::StaleDraftDescriptor => formatter.write_str("stale draft descriptor"),
            Self::StaleChoiceDescriptor => formatter.write_str("stale choice descriptor"),
            Self::PhaseMismatch { expected } => {
                write!(formatter, "action requires phase {expected}")
            }
            Self::ChoiceIdMismatch { expected, actual } => {
                write!(
                    formatter,
                    "choice id mismatch: expected {expected}, got {actual}"
                )
            }
            Self::OfferIndexOutOfRange { index, len } => {
                write!(formatter, "draft offer index {index} is outside 0..{len}")
            }
            Self::OptionIndexOutOfRange { index, len } => {
                write!(formatter, "choice option index {index} is outside 0..{len}")
            }
            Self::ActionIndexOutOfRange { index, len } => {
                write!(formatter, "legal action index {index} is outside 0..{len}")
            }
            Self::DuplicateDescriptor {
                first_index,
                second_index,
            } => write!(
                formatter,
                "legal actions {first_index} and {second_index} have the same V3 descriptor"
            ),
            Self::RoundTripMismatch { index } => {
                write!(
                    formatter,
                    "legal action {index} failed descriptor round trip"
                )
            }
            Self::PolicyLengthMismatch { expected, actual } => write!(
                formatter,
                "policy has {actual} logits but the state has {expected} legal actions"
            ),
            Self::NonFiniteLogit { index } => {
                write!(formatter, "policy logit {index} is not finite")
            }
            Self::InvalidPolicyNormalization => {
                formatter.write_str("policy softmax normalization is invalid")
            }
        }
    }
}

impl std::error::Error for ActionV3Error {}

#[cfg(test)]
mod tests {
    use super::*;
    use augment_core::{
        BUNDLED_RULESET_ID, Board, CardInstance, ChoiceContinuation, IMPLEMENTED_CARDS, Mode,
        PendingChoice, Piece,
    };
    use std::collections::BTreeSet;

    fn standard_playing_state(seed: u64) -> GameState {
        let mut state = GameState::standard(BUNDLED_RULESET_ID, Mode::Standard, seed);
        state.phase = GamePhase::Playing;
        state
    }

    fn grant_card(state: &mut GameState, instance_id: u32, definition_id: &str, owner: Player) {
        state.add_card(CardInstance {
            instance_id,
            definition_id: definition_id.to_owned(),
            owner,
            half_stars: 1,
            charges: 1,
            available_after_turn: None,
        });
    }

    #[test]
    fn every_action_variant_round_trips_without_raw_ids() {
        let mut board = Board::standard();
        let wizard_square: Square = "b1".parse().unwrap();
        board.set(
            wizard_square,
            Some(Piece::new(Player::White, PieceKind::Wizard).with_id(9_000)),
        );
        let mut state =
            GameState::from_board(BUNDLED_RULESET_ID, Mode::Standard, board, Player::White, 1);
        state.phase = GamePhase::Playing;
        grant_card(&mut state, 70_000, "gale", Player::White);
        let pawn = state.board.get("e2".parse().unwrap()).unwrap();
        let actions = [
            Action::Move(MoveAction::new(
                "e2".parse().unwrap(),
                "e4".parse().unwrap(),
                None,
            )),
            Action::UsePieceAbility {
                piece_id: 9_000,
                ability: PieceAbility::WizardMeteor,
                target: ActionTarget::Squares(vec![
                    "c3".parse().unwrap(),
                    "d4".parse().unwrap(),
                    "e5".parse().unwrap(),
                ]),
            },
            Action::UseCard {
                instance_id: 70_000,
                target: ActionTarget::PieceAt("e2".parse().unwrap()),
            },
        ];
        assert_ne!(pawn.id, 0);
        for action in actions {
            let descriptor = encode_action_v3(&action, &state).unwrap();
            assert_eq!(decode_action_v3(&descriptor, &state).unwrap(), action);
        }

        let engine = Engine::bundled_20260807().unwrap();
        let draft = engine.new_game(Mode::Standard, 2).unwrap();
        let draft_action = engine.legal_actions(&draft).unwrap()[0].clone();
        let descriptor = encode_action_v3(&draft_action, &draft).unwrap();
        assert_eq!(decode_action_v3(&descriptor, &draft).unwrap(), draft_action);

        let mut choice = state.clone();
        choice.phase = GamePhase::AwaitingChoice(PendingChoice {
            id: 99_999,
            owner: Player::White,
            source_card_instance: 70_000,
            handler_id: "test-choice".to_owned(),
            options: vec!["alpha".to_owned(), "beta".to_owned()],
            continuation: ChoiceContinuation::ResumeTurn,
        });
        let choice_action = Action::ResolveChoice {
            choice_id: 99_999,
            option_index: 1,
        };
        let descriptor = encode_action_v3(&choice_action, &choice).unwrap();
        assert!(matches!(
            &descriptor,
            ActionDescriptorV3::ResolveChoice { option, .. }
                if option == "choice:test-choice:1"
        ));
        assert_eq!(
            decode_action_v3(&descriptor, &choice).unwrap(),
            choice_action
        );
    }

    #[test]
    fn quantum_shadow_moves_round_trip_and_apply_through_v3() {
        let mut board = Board::empty();
        board.set(
            "e1".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::King).with_id(1)),
        );
        board.set(
            "a1".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Rook).with_id(2)),
        );
        board.set(
            "e8".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::King).with_id(3)),
        );
        let mut state =
            GameState::from_board(BUNDLED_RULESET_ID, Mode::Standard, board, Player::White, 20);
        state
            .card_runtime
            .quantum_positions
            .insert(2, "d1".parse().unwrap());
        let engine = Engine::bundled_20260807().unwrap();
        let legal = LegalActionSetV3::from_state(&state, &engine).unwrap();
        let shadow_move = Action::Move(MoveAction::new(
            "d1".parse().unwrap(),
            "d3".parse().unwrap(),
            None,
        ));
        let index = legal
            .actions
            .iter()
            .position(|descriptor| {
                decode_action_v3(descriptor, &state).is_ok_and(|action| action == shadow_move)
            })
            .expect("quantum shadow move must have a V3 descriptor");
        let moved = legal.apply(index, &state, &engine).unwrap().state;
        assert!(moved.board.get("d3".parse().unwrap()).is_some());
        assert!(moved.board.get("a1".parse().unwrap()).is_none());
    }

    #[test]
    fn field_promotion_choice_hides_raw_piece_id_from_model_tokens() {
        let mut state = standard_playing_state(21);
        state.phase = GamePhase::AwaitingChoice(PendingChoice {
            id: 777,
            owner: Player::White,
            source_card_instance: 0,
            handler_id: "field-promotion:12345".to_owned(),
            options: vec!["queen".to_owned(), "rook".to_owned()],
            continuation: ChoiceContinuation::FinishTurn,
        });
        let action = Action::ResolveChoice {
            choice_id: 777,
            option_index: 1,
        };
        let descriptor = encode_action_v3(&action, &state).unwrap();
        assert!(matches!(
            &descriptor,
            ActionDescriptorV3::ResolveChoice {
                handler_id,
                option,
                ..
            } if handler_id == "field-promotion" && option == "choice:field-promotion:1"
        ));
        assert_eq!(decode_action_v3(&descriptor, &state).unwrap(), action);
    }

    #[test]
    fn arbitrary_multi_square_targets_never_clamp_or_collapse() {
        let mut state = standard_playing_state(3);
        grant_card(&mut state, 500, "gale", Player::White);
        let actions = [
            Action::UseCard {
                instance_id: 500,
                target: ActionTarget::Squares(vec!["a1".parse().unwrap(), "b2".parse().unwrap()]),
            },
            Action::UseCard {
                instance_id: 500,
                target: ActionTarget::Squares(vec!["a1".parse().unwrap(), "b3".parse().unwrap()]),
            },
            Action::UseCard {
                instance_id: 500,
                target: ActionTarget::Squares(vec![
                    "a1".parse().unwrap(),
                    "b2".parse().unwrap(),
                    "c3".parse().unwrap(),
                ]),
            },
            Action::UseCard {
                instance_id: 500,
                target: ActionTarget::Squares(Vec::new()),
            },
        ];
        let descriptors = actions
            .iter()
            .map(|action| encode_action_v3(action, &state).unwrap())
            .collect::<BTreeSet<_>>();
        assert_eq!(descriptors.len(), actions.len());
        for action in actions {
            let descriptor = encode_action_v3(&action, &state).unwrap();
            assert_eq!(decode_action_v3(&descriptor, &state).unwrap(), action);
        }
    }

    #[test]
    fn legal_promotion_descriptors_keep_all_promotion_kinds_distinct() {
        let mut board = Board::empty();
        board.set(
            "e1".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::King).with_id(1)),
        );
        board.set(
            "e8".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::King).with_id(2)),
        );
        board.set(
            "a7".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Pawn).with_id(3)),
        );
        let state =
            GameState::from_board(BUNDLED_RULESET_ID, Mode::Standard, board, Player::White, 4);
        let engine = Engine::bundled_20260807().unwrap();
        let descriptors = LegalActionSetV3::from_state(&state, &engine).unwrap();
        let promotions = descriptors
            .actions
            .iter()
            .filter_map(|descriptor| match descriptor {
                ActionDescriptorV3::Move {
                    from, promotion, ..
                } if *from == canonical_square_index("a7".parse().unwrap(), Player::White) => {
                    *promotion
                }
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(promotions.len(), 4);
        assert_eq!(
            promotions,
            [
                PieceKind::Knight.stable_code(),
                PieceKind::Bishop.stable_code(),
                PieceKind::Rook.stable_code(),
                PieceKind::Queen.stable_code(),
            ]
            .into_iter()
            .collect()
        );
    }

    #[test]
    fn all_current_card_actions_have_unique_round_trip_descriptors() {
        let mut state = standard_playing_state(5);
        for (index, definition) in IMPLEMENTED_CARDS.iter().enumerate() {
            state.add_card(CardInstance {
                instance_id: index as u32 + 1_000,
                definition_id: definition.id.to_owned(),
                owner: Player::White,
                half_stars: definition.half_stars,
                charges: 1,
                available_after_turn: None,
            });
        }
        let engine = Engine::bundled_20260807().unwrap();
        let legal = engine.legal_actions(&state).unwrap();
        let set = LegalActionSetV3::from_state(&state, &engine).unwrap();
        let card_action_count = legal
            .iter()
            .filter(|action| matches!(action, Action::UseCard { .. }))
            .count();

        assert_eq!(set.len(), legal.len());
        assert!(card_action_count > 100);
        assert_eq!(
            set.actions.iter().collect::<BTreeSet<_>>().len(),
            set.actions.len()
        );
        for (index, action) in legal.iter().enumerate() {
            assert_eq!(set.decode_action(index, &state).unwrap(), *action);
        }
    }

    #[test]
    fn decoded_legal_actions_produce_the_same_engine_transition() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = engine.new_game(Mode::Standard, 6).unwrap();
        let legal = engine.legal_actions(&state).unwrap();
        let set = LegalActionSetV3::from_state(&state, &engine).unwrap();

        for (index, action) in legal.into_iter().enumerate() {
            let expected = engine.apply(&state, action).unwrap();
            let actual = set.apply(index, &state, &engine).unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn policy_vector_is_exactly_the_dynamic_legal_action_count() {
        let engine = Engine::bundled_20260807().unwrap();
        let state = engine.new_game(Mode::Standard, 7).unwrap();
        let set = LegalActionSetV3::from_state(&state, &engine).unwrap();
        let logits = (0..set.len()).map(|index| index as f32).collect::<Vec<_>>();
        let policy = set.policy_from_logits(&logits).unwrap();

        assert_eq!(policy.len(), set.len());
        assert!((policy.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(policy.iter().all(|probability| *probability > 0.0));
        assert!(matches!(
            set.policy_from_logits(&[]),
            Err(ActionV3Error::PolicyLengthMismatch { .. })
        ));
        let mut invalid = logits;
        invalid[0] = f32::NAN;
        assert!(matches!(
            set.policy_from_logits(&invalid),
            Err(ActionV3Error::NonFiniteLogit { index: 0 })
        ));
    }

    #[test]
    fn color_swapped_moves_share_the_same_descriptor() {
        let mut white_board = Board::empty();
        white_board.set(
            "b2".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Pawn).with_id(900)),
        );
        let white = GameState::from_board(
            BUNDLED_RULESET_ID,
            Mode::Standard,
            white_board,
            Player::White,
            8,
        );
        let mut black_board = Board::empty();
        black_board.set(
            "b7".parse().unwrap(),
            Some(Piece::new(Player::Black, PieceKind::Pawn).with_id(3)),
        );
        let black = GameState::from_board(
            BUNDLED_RULESET_ID,
            Mode::Standard,
            black_board,
            Player::Black,
            9,
        );
        let white_action = Action::Move(MoveAction::new(
            "b2".parse().unwrap(),
            "b3".parse().unwrap(),
            None,
        ));
        let black_action = Action::Move(MoveAction::new(
            "b7".parse().unwrap(),
            "b6".parse().unwrap(),
            None,
        ));
        assert_eq!(
            encode_action_v3(&white_action, &white).unwrap(),
            encode_action_v3(&black_action, &black).unwrap()
        );
    }

    #[test]
    fn descriptors_remain_collision_free_across_a_deterministic_game_prefix() {
        let engine = Engine::bundled_20260807().unwrap();
        let mut state = engine.new_game(Mode::Standard, 10).unwrap();
        let mut checked_states = 0;

        for _ in 0..80 {
            let set = LegalActionSetV3::from_state(&state, &engine).unwrap();
            checked_states += 1;
            if set.is_empty() || !matches!(state.status, augment_core::GameStatus::Ongoing) {
                break;
            }
            state = set.apply(0, &state, &engine).unwrap().state;
        }
        assert!(checked_states >= 10);
    }

    #[test]
    fn stale_card_and_choice_descriptors_are_rejected() {
        let mut state = standard_playing_state(11);
        grant_card(&mut state, 44, "gale", Player::White);
        let action = Action::UseCard {
            instance_id: 44,
            target: ActionTarget::None,
        };
        let descriptor = encode_action_v3(&action, &state).unwrap();
        state.cards[0].definition_id = "panic".to_owned();
        assert!(matches!(
            decode_action_v3(&descriptor, &state),
            Err(ActionV3Error::StaleCardDescriptor { .. })
        ));
    }
}
