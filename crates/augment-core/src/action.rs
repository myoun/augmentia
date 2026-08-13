use crate::{PieceKind, Square};
use std::fmt;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Action {
    Move(MoveAction),
    UsePieceAbility {
        piece_id: u32,
        ability: PieceAbility,
        target: ActionTarget,
    },
    UseCard {
        instance_id: u32,
        target: ActionTarget,
    },
    DraftPick {
        offer_index: u16,
    },
    ResolveChoice {
        choice_id: u32,
        option_index: u16,
    },
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum PieceAbility {
    WizardLightning,
    WizardShield,
    WizardMeteor,
    WizardTimeStop,
    ColossusSector,
    MerchantBuy,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct MoveAction {
    pub from: Square,
    pub to: Square,
    pub promotion: Option<PieceKind>,
}

impl MoveAction {
    pub const fn new(from: Square, to: Square, promotion: Option<PieceKind>) -> Self {
        Self {
            from,
            to,
            promotion,
        }
    }
}

impl fmt::Display for MoveAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.from, self.to)?;
        if let Some(promotion) = self.promotion {
            write!(f, "={promotion:?}")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ActionTarget {
    None,
    Square(Square),
    PieceAt(Square),
    Squares(Vec<Square>),
}
