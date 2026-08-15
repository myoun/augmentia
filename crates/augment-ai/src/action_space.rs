use augment_core::{Action, ActionTarget, Engine, GameState, PieceAbility, PieceKind};

/// Total discrete action space dimension for the unified Policy head.
pub const ACTION_SPACE_SIZE: usize = 32_768;

const MOVE_OFFSET: usize = 0;
const MOVE_COUNT: usize = 64 * 64 * 6; // 24,576

const CARD_OFFSET: usize = MOVE_OFFSET + MOVE_COUNT;
const CARD_COUNT: usize = 3 * 256; // 768

const ABILITY_OFFSET: usize = CARD_OFFSET + CARD_COUNT;
const ABILITY_COUNT: usize = 6 * 65; // 390

const DRAFT_OFFSET: usize = ABILITY_OFFSET + ABILITY_COUNT;
const DRAFT_COUNT: usize = 8;

const CHOICE_OFFSET: usize = DRAFT_OFFSET + DRAFT_COUNT;
const CHOICE_COUNT: usize = 16;

pub const MULTI_SQ_OFFSET: usize = CHOICE_OFFSET + CHOICE_COUNT;

pub type ActionIndex = u16;

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ActionMask {
    bits: Vec<u64>,
}

impl ActionMask {
    pub fn new() -> Self {
        Self {
            bits: vec![0u64; ACTION_SPACE_SIZE.div_ceil(64)],
        }
    }

    pub fn set(&mut self, index: usize) {
        if index < ACTION_SPACE_SIZE {
            self.bits[index / 64] |= 1u64 << (index % 64);
        }
    }

    pub fn is_set(&self, index: usize) -> bool {
        if index < ACTION_SPACE_SIZE {
            (self.bits[index / 64] & (1u64 << (index % 64))) != 0
        } else {
            false
        }
    }

    pub fn count_ones(&self) -> usize {
        self.bits.iter().map(|b| b.count_ones() as usize).sum()
    }

    pub fn as_slice(&self) -> &[u64] {
        &self.bits
    }

    pub fn to_bool_vec(&self) -> Vec<bool> {
        (0..ACTION_SPACE_SIZE).map(|i| self.is_set(i)).collect()
    }
}

impl Default for ActionMask {
    fn default() -> Self {
        Self::new()
    }
}

fn promo_to_index(promo: Option<PieceKind>) -> usize {
    match promo {
        None => 0,
        Some(PieceKind::Knight) => 1,
        Some(PieceKind::Bishop) => 2,
        Some(PieceKind::Rook) => 3,
        Some(PieceKind::Queen) => 4,
        Some(PieceKind::Amazon) => 5,
        _ => 0,
    }
}

pub fn index_to_promo(idx: usize) -> Option<PieceKind> {
    match idx {
        1 => Some(PieceKind::Knight),
        2 => Some(PieceKind::Bishop),
        3 => Some(PieceKind::Rook),
        4 => Some(PieceKind::Queen),
        5 => Some(PieceKind::Amazon),
        _ => None,
    }
}

fn ability_to_index(ability: PieceAbility) -> usize {
    match ability {
        PieceAbility::WizardLightning => 0,
        PieceAbility::WizardShield => 1,
        PieceAbility::WizardMeteor => 2,
        PieceAbility::WizardTimeStop => 3,
        PieceAbility::ColossusSector => 4,
        PieceAbility::MerchantBuy => 5,
    }
}

pub fn index_to_ability(idx: usize) -> Option<PieceAbility> {
    match idx {
        0 => Some(PieceAbility::WizardLightning),
        1 => Some(PieceAbility::WizardShield),
        2 => Some(PieceAbility::WizardMeteor),
        3 => Some(PieceAbility::WizardTimeStop),
        4 => Some(PieceAbility::ColossusSector),
        5 => Some(PieceAbility::MerchantBuy),
        _ => None,
    }
}

pub fn encode_action(action: &Action, state: &GameState) -> ActionIndex {
    let raw_index = match action {
        Action::Move(movement) => {
            let promo_idx = promo_to_index(movement.promotion);
            MOVE_OFFSET + movement.from.index() * 64 * 6 + movement.to.index() * 6 + promo_idx
        }
        Action::UseCard {
            instance_id,
            target,
        } => {
            let slot = state
                .cards
                .iter()
                .filter(|c| c.owner == state.turn)
                .position(|c| c.instance_id == *instance_id)
                .unwrap_or(0)
                .min(2);
            match target {
                ActionTarget::None => CARD_OFFSET + slot * 256,
                ActionTarget::Square(sq) => CARD_OFFSET + slot * 256 + 1 + sq.index(),
                ActionTarget::PieceAt(sq) => CARD_OFFSET + slot * 256 + 65 + sq.index(),
                ActionTarget::Squares(sqs) => {
                    if sqs.len() == 2 {
                        let pair_idx = sqs[0].index() * 64 + sqs[1].index();
                        (MULTI_SQ_OFFSET + slot * 4096 + pair_idx).min(ACTION_SPACE_SIZE - 1)
                    } else if let Some(first) = sqs.first() {
                        CARD_OFFSET + slot * 256 + 129 + first.index()
                    } else {
                        CARD_OFFSET + slot * 256
                    }
                }
            }
        }
        Action::UsePieceAbility {
            ability, target, ..
        } => {
            let ab_idx = ability_to_index(*ability);
            let target_idx = match target {
                ActionTarget::None => 0,
                ActionTarget::Square(sq) | ActionTarget::PieceAt(sq) => 1 + sq.index(),
                ActionTarget::Squares(sqs) => sqs.first().map_or(0, |sq| 1 + sq.index()),
            };
            ABILITY_OFFSET + ab_idx * 65 + target_idx
        }
        Action::DraftPick { offer_index } => DRAFT_OFFSET + (*offer_index as usize).min(7),
        Action::ResolveChoice { option_index, .. } => {
            CHOICE_OFFSET + (*option_index as usize).min(15)
        }
    };
    (raw_index % ACTION_SPACE_SIZE) as ActionIndex
}

pub fn generate_action_mask(state: &GameState, engine: &Engine) -> ActionMask {
    let mut mask = ActionMask::new();
    if let Ok(legal_actions) = engine.legal_actions(state) {
        for action in &legal_actions {
            let idx = encode_action(action, state);
            mask.set(idx as usize);
        }
    }
    mask
}
