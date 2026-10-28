use augment_core::{Board, GameState, Mode, Piece, PieceKind, Player, Square};
use serde_json::{Value, json};

pub const PIECE_KINDS: &[PieceKind] = &[
    PieceKind::Pawn,
    PieceKind::Knight,
    PieceKind::Bishop,
    PieceKind::Rook,
    PieceKind::Queen,
    PieceKind::King,
    PieceKind::Assassin,
    PieceKind::Grasshopper,
    PieceKind::Amazon,
    PieceKind::PrimeMinister,
    PieceKind::Dragon,
    PieceKind::Cannon,
    PieceKind::Herald,
    PieceKind::Knightmaster,
    PieceKind::Unicorn,
    PieceKind::Hook,
    PieceKind::Idol,
    PieceKind::Jester,
    PieceKind::Recruiter,
    PieceKind::Reaper,
    PieceKind::Wizard,
    PieceKind::Alibaba,
    PieceKind::Log,
    PieceKind::Missionary,
    PieceKind::Cardinal,
    PieceKind::Protestant,
    PieceKind::StandardBearer,
    PieceKind::Windmill,
    PieceKind::Squire,
    PieceKind::BigRook,
    PieceKind::Checker,
    PieceKind::CheckerKing,
    PieceKind::Alfil,
    PieceKind::Guard,
    PieceKind::Fanatic,
    PieceKind::Colossus,
    PieceKind::RoyalKnight,
    PieceKind::Man,
    PieceKind::Merchant,
    PieceKind::DarkWizard,
    PieceKind::Monster,
    PieceKind::Scarecrow,
    PieceKind::Camel,
    PieceKind::Ferz,
    PieceKind::Bat,
    PieceKind::BabyBear,
    PieceKind::Lobster,
    PieceKind::Bear,
    PieceKind::Vip,
    PieceKind::Crown,
    PieceKind::Football,
    PieceKind::SiegeRam,
    PieceKind::MagicGirl,
    PieceKind::Berserker,
    PieceKind::Slime,
    PieceKind::Siren,
    PieceKind::Trickster,
    PieceKind::Undead,
    PieceKind::Clockwork,
    PieceKind::Parrot,
    PieceKind::Paladin,
    PieceKind::Princess,
    PieceKind::Octopus,
    PieceKind::Hedgehog,
    PieceKind::Campfire,
    PieceKind::Thief,
    PieceKind::Brutus,
    PieceKind::BigBishop,
    PieceKind::Archbishop,
    PieceKind::Chancellor,
];

pub fn piece_catalog() -> Value {
    json!(PIECE_KINDS.iter().map(|kind| {
        let (key, name, code, symbol) = super::view::piece_meta(*kind, Player::White);
        let black_symbol = super::view::piece_meta(*kind, Player::Black).3;
        json!({"kind": key, "name": name, "code": code, "description": super::view::piece_description(key), "whiteSymbol": symbol.unwrap_or(code),
            "blackSymbol": black_symbol.unwrap_or(code), "isUnicode": symbol.is_some()})
    }).collect::<Vec<_>>())
}

pub fn edited_state(payload: &Value, ruleset: &str, seed: u64) -> Result<GameState, String> {
    let owner = |v: &Value| match v.as_str() {
        Some("white") => Ok(Player::White),
        Some("black") => Ok(Player::Black),
        _ => Err("기물 색과 시작 차례는 white 또는 black이어야 합니다.".to_owned()),
    };
    let turn = owner(&payload["turn"])?;
    let cells = payload["pieces"]
        .as_array()
        .ok_or("기물 목록이 필요합니다.")?;
    if cells.len() > 64 {
        return Err("기물은 최대 64개입니다.".into());
    }
    let mut board = Board::empty();
    for cell in cells {
        let square = cell["square"]
            .as_str()
            .ok_or("기물 위치가 필요합니다.")?
            .parse::<Square>()
            .map_err(|e| e.to_string())?;
        if board.get(square).is_some() {
            return Err(format!("{square}에 기물이 중복되었습니다."));
        }
        let kind = PIECE_KINDS
            .iter()
            .find(|k| Some(super::view::piece_meta(**k, Player::White).0) == cell["kind"].as_str())
            .copied()
            .ok_or("지원하지 않는 기물입니다.")?;
        board.set(square, Some(Piece::new(owner(&cell["owner"])?, kind)));
    }
    // Rebuild IDs, origins, repetition state and runtime rather than mutating a live board.
    let mut state = GameState::from_board(ruleset, Mode::Standard, board, turn, seed);
    state.completed_drafts = if payload["futureDrafts"] == true {
        [true, false, false]
    } else {
        [true; 3]
    };
    if payload["allowCastling"] != true {
        state
            .card_runtime
            .moved_piece_ids
            .extend(state.board.iter().map(|(_, p)| p.id));
    }
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn editor_normalizes_ids_turn_and_disables_implicit_castling() {
        let payload = json!({"turn":"black", "pieces":[
            {"square":"e1", "kind":"king", "owner":"white"},
            {"square":"e8", "kind":"king", "owner":"black"},
            {"square":"h1", "kind":"rook", "owner":"white"}]});
        let state = edited_state(&payload, "test", 1).unwrap();
        assert_eq!(state.turn, Player::Black);
        assert_eq!(state.completed_drafts, [true; 3]);
        let ids = state
            .board
            .iter()
            .map(|(_, p)| p.id)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(ids.len(), 3);
        assert!(!ids.contains(&0));
        assert_eq!(state.card_runtime.moved_piece_ids, ids);
        let mut duplicate = payload.clone();
        duplicate["pieces"]
            .as_array_mut()
            .unwrap()
            .push(payload["pieces"][0].clone());
        assert!(edited_state(&duplicate, "test", 1).is_err());
    }
}
