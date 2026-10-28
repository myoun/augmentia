use super::*;
pub(super) const ID: &str = crate::REVELATION;
pub(super) fn install(
    engine: &Engine,
    state: &mut GameState,
    events: &mut Vec<Event>,
) -> Result<(), EngineError> {
    mark_rule(state, ID);
    engine.install_revelation(state, events)
}
pub(super) const BEHAVIOR: CardBehavior = CardBehavior::rule(ID, install, rule_eligibility);

impl Engine {
    pub(in crate::engine) fn install_revelation(
        &self,
        _state: &mut GameState,
        _events: &mut Vec<Event>,
    ) -> Result<(), EngineError> {
        // Material adjudication replaces the historical shortened-deathmatch effect.
        Ok(())
    }
}

pub(super) fn combat_value(kind: PieceKind) -> u16 {
    match kind {
        PieceKind::BerolinaPawn => 1,
        PieceKind::Scarecrow => 3,
        PieceKind::Monster => 4,
        PieceKind::Killer => 3,
        PieceKind::Beacon => 5,
        PieceKind::TransmissionTower => 5,
        PieceKind::Barrier => 5,
        PieceKind::Queen => 9,
        PieceKind::Rook => 5,
        PieceKind::Bishop => 3,
        PieceKind::Missionary => 2,
        PieceKind::Knight => 3,
        PieceKind::Pawn => 1,
        PieceKind::Protestant => 3,
        PieceKind::Herald => 5,
        PieceKind::Cannon => 4,
        PieceKind::Fanatic => 1,
        PieceKind::PrimeMinister => 9,
        PieceKind::Alibaba => 2,
        PieceKind::Amazon => 13,
        PieceKind::Chancellor => 8,
        PieceKind::Archbishop => 7,
        PieceKind::Cardinal => 7,
        PieceKind::Unicorn => 5,
        PieceKind::Jester => 9,
        PieceKind::Camel => 2,
        PieceKind::Log => 2,
        PieceKind::Hook => 15,
        PieceKind::Grasshopper => 4,
        PieceKind::Dragon => 5,
        PieceKind::Man => 4,
        PieceKind::Assassin => 4,
        PieceKind::Reaper => 9,
        PieceKind::Knightmaster => 3,
        PieceKind::StandardBearer => 2,
        PieceKind::Guard => 2,
        PieceKind::Recruiter => 9,
        PieceKind::Squire => 1,
        PieceKind::Checker => 1,
        PieceKind::CheckerKing => 2,
        PieceKind::Wizard => 9,
        PieceKind::Alfil => 1,
        PieceKind::Windmill => 4,
        PieceKind::Idol => 9,
        PieceKind::Lobster => 2,
        PieceKind::BabyBear => 4,
        PieceKind::Bear => 17,
        PieceKind::SiegeRam => 5,
        PieceKind::MagicGirl => 6,
        PieceKind::Berserker => 7,
        PieceKind::Slime => 5,
        PieceKind::Siren => 9,
        PieceKind::Trickster => 5,
        PieceKind::Undead => 9,
        PieceKind::Paladin => 4,
        PieceKind::Octopus => 5,
        PieceKind::Brutus => 10,
        PieceKind::Clockwork => 5,
        PieceKind::Parrot => 5,
        PieceKind::Thief => 9,
        PieceKind::Campfire => 4,
        PieceKind::Hedgehog => 9,
        PieceKind::Princess => 6,
        PieceKind::Colossus => 12,
        PieceKind::BigRook => 8,
        PieceKind::BigBishop => 8,
        PieceKind::Grappler => 13,
        PieceKind::RevolvingDoor => 5,
        PieceKind::DonQuixote => 7,
        PieceKind::Medium => 3,
        _ => 0,
    }
}
impl Engine {
    pub(in crate::engine) fn settle_rule_revelation(&self, state: &mut GameState) {
        if !rule_active(state, ID)
            || state.status != GameStatus::Ongoing
            || state.turns_taken[1] < 60
        {
            return;
        }
        let mut totals = [0u32; 2];
        for (_, piece) in state.board.iter() {
            if state.card_runtime.monster_rule != Some(piece.id) {
                totals[piece.owner.index()] += u32::from(combat_value(piece.kind));
            }
        }
        state.status = if totals[0] == totals[1] {
            GameStatus::Draw {
                reason: crate::DrawReason::CardEffect(ID.into()),
            }
        } else {
            GameStatus::Won {
                winner: if totals[0] > totals[1] {
                    Player::White
                } else {
                    Player::Black
                },
                reason: WinReason::CardEffect(ID.into()),
            }
        };
    }
}
