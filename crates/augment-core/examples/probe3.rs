use augment_core::*;
use std::str::FromStr;

fn minimal(pieces: &[(&str, Player, PieceKind)], turn: Player, seed: u64) -> GameState {
    let mut board = Board::empty();
    for (s, o, k) in pieces {
        board.set(s.parse().unwrap(), Some(Piece::new(*o, *k)));
    }
    GameState::from_board(LATEST_RULESET_ID, Mode::Standard, board, turn, seed)
}

fn has(engine: &Engine, st: &GameState, from: &str, to: &str) -> bool {
    engine
        .legal_actions(st)
        .unwrap()
        .contains(&Action::Move(MoveAction::new(
            Square::from_str(from).unwrap(),
            Square::from_str(to).unwrap(),
            None,
        )))
}

fn main() {
    let engine = Engine::bundled_latest().unwrap();
    for card in ["parrot", "campfire"] {
        for kind in [PieceKind::Rook, PieceKind::Knight, PieceKind::Bishop] {
            let st = minimal(
                &[
                    ("a1", Player::White, PieceKind::King),
                    ("h8", Player::Black, PieceKind::King),
                    ("d4", Player::White, kind),
                    ("g7", Player::Black, PieceKind::Rook),
                    ("b2", Player::White, PieceKind::Pawn),
                ],
                Player::White,
                936,
            );
            let g = match engine.grant_card(&st, 1, card, Player::White) {
                Ok(t) => t.state,
                Err(e) => {
                    println!("{card}/{kind:?}: grant err {e:?}");
                    continue;
                }
            };
            let actions: Vec<String> = engine
                .legal_actions(&g)
                .unwrap()
                .iter()
                .filter_map(|a| match a {
                    Action::UseCard {
                        instance_id: 1,
                        target,
                    } => Some(format!("{target:?}")),
                    _ => None,
                })
                .collect();
            println!(
                "{card}/{kind:?}: d4={:?} use_actions={actions:?}",
                g.board.get(Square::from_str("d4").unwrap()).map(|p| p.kind)
            );
        }
    }
    // campfire after use on a knight
    let st = minimal(
        &[
            ("a1", Player::White, PieceKind::King),
            ("h8", Player::Black, PieceKind::King),
            ("d4", Player::White, PieceKind::Knight),
            ("g7", Player::Black, PieceKind::Rook),
            ("b2", Player::White, PieceKind::Pawn),
            ("d5", Player::White, PieceKind::Pawn),
            ("d8", Player::Black, PieceKind::Rook),
        ],
        Player::White,
        941,
    );
    let g = engine
        .grant_card(&st, 1, "campfire", Player::White)
        .unwrap()
        .state;
    let used = engine
        .apply(
            &g,
            Action::UseCard {
                instance_id: 1,
                target: ActionTarget::PieceAt(Square::from_str("d4").unwrap()),
            },
        )
        .unwrap()
        .state;
    println!(
        "campfire used: d4={:?} d4e4={} d4e5={}",
        used.board
            .get(Square::from_str("d4").unwrap())
            .map(|p| p.kind),
        has(&engine, &used, "d4", "e4"),
        has(&engine, &used, "d4", "e5")
    );
}
