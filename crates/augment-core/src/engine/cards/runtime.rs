//! Storage for card effects. Field order and Debug type name are part of the
//! existing position-key encoding; keep them stable during structural refactors.
use super::*;
use std::collections::BTreeMap;
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CardEffectRuntimeState {
    pub assembly: [bool; 2],
    pub hope: [bool; 2],
    pub iqp_pending: [bool; 2],
    /// Number of retained moves already granted in this rook-lift chain (max 3).
    pub rook_lift_chain: Option<(u32, u8)>,
    pub pending_seals: Vec<PendingSeal>,
    pub bound_piece_ids: BTreeSet<u32>,
    pub greek_gift_pending: [bool; 2],
    pub pending_taboo: Vec<PendingTaboo>,
    pub pending_ascensions: Vec<PendingAscension>,
    pub grudged_piece_ids: BTreeSet<u32>,
    /// Rook=1, bishop=2, knight=4, camel=8. Loss survives transformations.
    pub grudge_losses: BTreeMap<u32, u8>,
    pub pawn_chain_links: Vec<[u32; 2]>,
    pub penta_check: [Option<PentaCheckProgress>; 2],
    pub prophecy_remaining: [Option<u8>; 2],
}
