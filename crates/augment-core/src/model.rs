use std::fmt;
use std::str::FromStr;

pub const BOARD_FILES: u8 = 8;
pub const BOARD_RANKS: u8 = 8;
pub const BOARD_SQUARES: usize = 64;

#[derive(
    Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize,
)]
pub enum Player {
    White,
    Black,
}

impl Player {
    pub const fn opponent(self) -> Self {
        match self {
            Self::White => Self::Black,
            Self::Black => Self::White,
        }
    }

    pub const fn index(self) -> usize {
        match self {
            Self::White => 0,
            Self::Black => 1,
        }
    }
}

impl fmt::Display for Player {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::White => f.write_str("white"),
            Self::Black => f.write_str("black"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Square(u8);

impl Square {
    pub fn new(file: u8, rank: u8) -> Option<Self> {
        (file < BOARD_FILES && rank < BOARD_RANKS).then_some(Self(rank * BOARD_FILES + file))
    }

    pub fn from_index(index: u8) -> Option<Self> {
        (usize::from(index) < BOARD_SQUARES).then_some(Self(index))
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub const fn file(self) -> u8 {
        self.0 % BOARD_FILES
    }

    pub const fn rank(self) -> u8 {
        self.0 / BOARD_FILES
    }

    pub fn offset(self, file_delta: i8, rank_delta: i8) -> Option<Self> {
        let file = self.file() as i8 + file_delta;
        let rank = self.rank() as i8 + rank_delta;
        if (0..BOARD_FILES as i8).contains(&file) && (0..BOARD_RANKS as i8).contains(&rank) {
            Self::new(file as u8, rank as u8)
        } else {
            None
        }
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let file = char::from(b'a' + self.file());
        let rank = char::from(b'1' + self.rank());
        write!(f, "{file}{rank}")
    }
}

impl FromStr for Square {
    type Err = SquareParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let bytes = value.as_bytes();
        if bytes.len() != 2
            || !(b'a'..=b'h').contains(&bytes[0])
            || !(b'1'..=b'8').contains(&bytes[1])
        {
            return Err(SquareParseError(value.to_owned()));
        }
        Ok(Self((bytes[1] - b'1') * BOARD_FILES + (bytes[0] - b'a')))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SquareParseError(String);

impl fmt::Display for SquareParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid square {:?}; expected a1 through h8", self.0)
    }
}

impl std::error::Error for SquareParseError {}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PieceKind {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
    Assassin,
    Grasshopper,
    Amazon,
    PrimeMinister,
    Dragon,
    Cannon,
    Herald,
    Knightmaster,
    Unicorn,
    Hook,
    Idol,
    Jester,
    Recruiter,
    Reaper,
    Wizard,
    Alibaba,
    Log,
    Missionary,
    Cardinal,
    Protestant,
    StandardBearer,
    Windmill,
    Squire,
    BigRook,
    Checker,
    CheckerKing,
    Alfil,
    Guard,
    Fanatic,
    Colossus,
    RoyalKnight,
    Man,
    Merchant,
    DarkWizard,
    Monster,
    Scarecrow,
    Camel,
    Ferz,
    Bat,
    BabyBear,
    Lobster,
    Bear,
    Vip,
    Crown,
    Football,
    SiegeRam,
    MagicGirl,
    Berserker,
    Slime,
    Siren,
    Trickster,
    Undead,
    Clockwork,
    Parrot,
    Paladin,
    Princess,
    Octopus,
    Hedgehog,
    Campfire,
    Thief,
    Brutus,
    BigBishop,
}

impl PieceKind {
    pub const fn stable_code(self) -> u8 {
        match self {
            Self::Pawn => 1,
            Self::Knight => 2,
            Self::Bishop => 3,
            Self::Rook => 4,
            Self::Queen => 5,
            Self::King => 6,
            Self::Assassin => 7,
            Self::Grasshopper => 8,
            Self::Amazon => 9,
            Self::PrimeMinister => 10,
            Self::Dragon => 11,
            Self::Cannon => 12,
            Self::Herald => 13,
            Self::Knightmaster => 14,
            Self::Unicorn => 15,
            Self::Hook => 16,
            Self::Idol => 17,
            Self::Jester => 18,
            Self::Recruiter => 19,
            Self::Reaper => 20,
            Self::Wizard => 21,
            Self::Alibaba => 22,
            Self::Log => 23,
            Self::Missionary => 24,
            Self::Cardinal => 25,
            Self::Protestant => 26,
            Self::StandardBearer => 27,
            Self::Windmill => 28,
            Self::Squire => 29,
            Self::BigRook => 30,
            Self::Checker => 31,
            Self::CheckerKing => 32,
            Self::Alfil => 33,
            Self::Guard => 34,
            Self::Fanatic => 35,
            Self::Colossus => 36,
            Self::RoyalKnight => 37,
            Self::Man => 38,
            Self::Merchant => 39,
            Self::DarkWizard => 40,
            Self::Monster => 41,
            Self::Scarecrow => 42,
            Self::Camel => 43,
            Self::Ferz => 44,
            Self::Bat => 45,
            Self::BabyBear => 46,
            Self::Lobster => 47,
            Self::Bear => 48,
            Self::Vip => 49,
            Self::Crown => 50,
            Self::Football => 51,
            Self::SiegeRam => 52,
            Self::MagicGirl => 53,
            Self::Berserker => 54,
            Self::Slime => 55,
            Self::Siren => 56,
            Self::Trickster => 57,
            Self::Undead => 58,
            Self::Clockwork => 59,
            Self::Parrot => 60,
            Self::Paladin => 61,
            Self::Princess => 62,
            Self::Octopus => 63,
            Self::Hedgehog => 64,
            Self::Campfire => 65,
            Self::Thief => 66,
            Self::Brutus => 67,
            Self::BigBishop => 68,
        }
    }

    pub const fn is_valid_promotion(self) -> bool {
        matches!(self, Self::Knight | Self::Bishop | Self::Rook | Self::Queen)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Piece {
    /// Stable per-game identity used by card statuses and replay-compatible quirks.
    pub id: u32,
    pub owner: Player,
    pub kind: PieceKind,
    pub capture_count: u16,
}

impl Piece {
    pub const fn new(owner: Player, kind: PieceKind) -> Self {
        Self {
            id: 0,
            owner,
            kind,
            capture_count: 0,
        }
    }

    pub const fn with_id(mut self, id: u32) -> Self {
        self.id = id;
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Board {
    cells: [Option<Piece>; BOARD_SQUARES],
}

impl Board {
    pub const fn empty() -> Self {
        Self {
            cells: [None; BOARD_SQUARES],
        }
    }

    pub fn standard() -> Self {
        let mut board = Self::empty();
        let back_rank = [
            PieceKind::Rook,
            PieceKind::Knight,
            PieceKind::Bishop,
            PieceKind::Queen,
            PieceKind::King,
            PieceKind::Bishop,
            PieceKind::Knight,
            PieceKind::Rook,
        ];

        for (file, kind) in back_rank.into_iter().enumerate() {
            board.set(
                Square::new(file as u8, 0).expect("standard board square"),
                Some(Piece::new(Player::White, kind)),
            );
            board.set(
                Square::new(file as u8, 1).expect("standard board square"),
                Some(Piece::new(Player::White, PieceKind::Pawn)),
            );
            board.set(
                Square::new(file as u8, 6).expect("standard board square"),
                Some(Piece::new(Player::Black, PieceKind::Pawn)),
            );
            board.set(
                Square::new(file as u8, 7).expect("standard board square"),
                Some(Piece::new(Player::Black, kind)),
            );
        }
        board
    }

    pub fn get(&self, square: Square) -> Option<Piece> {
        self.cells[square.index()]
    }

    pub fn set(&mut self, square: Square, piece: Option<Piece>) -> Option<Piece> {
        std::mem::replace(&mut self.cells[square.index()], piece)
    }

    pub fn iter(&self) -> impl Iterator<Item = (Square, Piece)> + '_ {
        self.cells.iter().enumerate().filter_map(|(index, piece)| {
            piece.map(|piece| {
                (
                    Square::from_index(index as u8).expect("board index is always valid"),
                    piece,
                )
            })
        })
    }

    pub fn contains(&self, owner: Player, kind: PieceKind) -> bool {
        self.iter()
            .any(|(_, piece)| piece.owner == owner && piece.kind == kind)
    }

    pub fn find_by_id(&self, id: u32) -> Option<(Square, Piece)> {
        (id != 0)
            .then(|| self.iter().find(|(_, piece)| piece.id == id))
            .flatten()
    }

    pub(crate) fn assign_stable_ids(&mut self) -> u32 {
        let mut used = std::collections::BTreeSet::new();
        let mut next_id = self
            .iter()
            .map(|(_, piece)| piece.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
            .max(1);

        for index in 0..BOARD_SQUARES as u8 {
            let square = Square::from_index(index).expect("board index is valid");
            let Some(mut piece) = self.get(square) else {
                continue;
            };
            if piece.id == 0 || !used.insert(piece.id) {
                while used.contains(&next_id) || next_id == 0 {
                    next_id = next_id.saturating_add(1).max(1);
                }
                piece.id = next_id;
                used.insert(next_id);
                next_id = next_id.saturating_add(1).max(1);
                self.set(square, Some(piece));
            }
        }
        next_id
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn algebraic_square_round_trip() {
        for rank in 0..8 {
            for file in 0..8 {
                let square = Square::new(file, rank).unwrap();
                assert_eq!(square.to_string().parse::<Square>().unwrap(), square);
            }
        }
    }

    #[test]
    fn standard_board_has_32_pieces_and_two_kings() {
        let board = Board::standard();
        assert_eq!(board.iter().count(), 32);
        assert!(board.contains(Player::White, PieceKind::King));
        assert!(board.contains(Player::Black, PieceKind::King));
    }

    #[test]
    fn duplicate_and_missing_ids_are_normalized_in_square_order() {
        let mut board = Board::empty();
        board.set(
            "a1".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::King).with_id(7)),
        );
        board.set(
            "b1".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Rook).with_id(7)),
        );
        board.set(
            "c1".parse().unwrap(),
            Some(Piece::new(Player::White, PieceKind::Bishop)),
        );
        let next = board.assign_stable_ids();
        let ids: Vec<_> = board.iter().map(|(_, piece)| piece.id).collect();
        assert_eq!(ids.len(), 3);
        assert!(ids.iter().all(|id| *id != 0));
        assert_eq!(
            ids.iter().collect::<std::collections::BTreeSet<_>>().len(),
            3
        );
        assert!(next > *ids.iter().max().unwrap());
    }
}
