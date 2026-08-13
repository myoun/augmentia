use crate::{
    CardInstance, ChoiceContinuation, Event, GameState, PendingChoice, Piece, Player, Square,
};
use std::collections::BTreeMap;
use std::fmt;

/// A validated mutation requested by a card handler.
///
/// Keeping mutations as data gives the engine a single deterministic reduction path and makes
/// handler ordering visible in replay logs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EffectCommand {
    AdjustHalfStars {
        player: Player,
        delta: i16,
    },
    SpawnPiece {
        square: Square,
        piece: Piece,
        replace: bool,
    },
    RemovePiece {
        square: Square,
    },
    GrantActions {
        player: Player,
        count: u16,
    },
    ForceMove {
        player: Player,
        piece_id: u32,
        source: String,
        finish_turn: bool,
    },
    RequestChoice {
        owner: Player,
        handler_id: String,
        options: Vec<String>,
        continuation: ChoiceContinuation,
    },
    EndTurn {
        player: Player,
    },
    Win {
        winner: Player,
        reason: String,
    },
    Draw {
        reason: String,
    },
}

pub trait EffectHandler: Send + Sync {
    fn definition_id(&self) -> &str;

    /// React to one already-applied engine event. Returned commands are reduced in vector order.
    fn react(&self, card: &CardInstance, event: &Event, state: &GameState) -> Vec<EffectCommand>;
}

pub trait ChoiceHandler: Send + Sync {
    fn handler_id(&self) -> &str;

    fn resolve(
        &self,
        choice: &PendingChoice,
        option_index: u16,
        state: &GameState,
    ) -> Vec<EffectCommand>;
}

#[derive(Default)]
pub struct EffectRegistry {
    handlers: BTreeMap<String, Box<dyn EffectHandler>>,
}

impl EffectRegistry {
    pub fn register<H>(&mut self, handler: H) -> Result<(), EffectRegistryError>
    where
        H: EffectHandler + 'static,
    {
        let id = handler.definition_id().to_owned();
        if self.handlers.contains_key(&id) {
            return Err(EffectRegistryError::DuplicateDefinition(id));
        }
        self.handlers.insert(id, Box::new(handler));
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.handlers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }

    pub(crate) fn get(&self, definition_id: &str) -> Option<&dyn EffectHandler> {
        self.handlers.get(definition_id).map(Box::as_ref)
    }
}

#[derive(Default)]
pub struct ChoiceRegistry {
    handlers: BTreeMap<String, Box<dyn ChoiceHandler>>,
}

impl ChoiceRegistry {
    pub fn register<H>(&mut self, handler: H) -> Result<(), ChoiceRegistryError>
    where
        H: ChoiceHandler + 'static,
    {
        let id = handler.handler_id().to_owned();
        if self.handlers.contains_key(&id) {
            return Err(ChoiceRegistryError::DuplicateHandler(id));
        }
        self.handlers.insert(id, Box::new(handler));
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.handlers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }

    pub(crate) fn get(&self, handler_id: &str) -> Option<&dyn ChoiceHandler> {
        self.handlers.get(handler_id).map(Box::as_ref)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChoiceRegistryError {
    DuplicateHandler(String),
}

impl fmt::Display for ChoiceRegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateHandler(id) => write!(f, "choice handler {id:?} is already registered"),
        }
    }
}

impl std::error::Error for ChoiceRegistryError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EffectRegistryError {
    DuplicateDefinition(String),
}

impl fmt::Display for EffectRegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateDefinition(id) => {
                write!(
                    f,
                    "effect handler for definition {id:?} is already registered"
                )
            }
        }
    }
}

impl std::error::Error for EffectRegistryError {}
