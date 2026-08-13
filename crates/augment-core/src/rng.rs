/// Small deterministic PRNG whose state is owned by `GameState`.
///
/// SplitMix64 is used for reproducibility, not cryptographic security. Never replace this
/// algorithm in place: doing so changes replay semantics and requires a new engine API version.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    pub const ALGORITHM: &'static str = "splitmix64-v1";

    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub const fn state(self) -> u64 {
        self.state
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }

    pub fn choose_index(&mut self, len: usize) -> Option<usize> {
        if len == 0 {
            None
        } else {
            Some((self.next_u64() % len as u64) as usize)
        }
    }

    /// Draw the 53-bit mantissa used by browser-style `[0, 1)` decisions.
    pub(crate) fn next_unit_53(&mut self) -> u64 {
        self.next_u64() >> 11
    }

    /// Mirrors `items[Math.floor(Math.random() * items.length)] || null`.
    ///
    /// The web implementation consumes a random value even for an empty list; preserving that
    /// quirk matters for all later random outcomes in the same game.
    pub(crate) fn web_choice_index(&mut self, len: usize) -> (u64, Option<usize>) {
        let roll = self.next_unit_53();
        let index = (len > 0).then(|| ((u128::from(roll) * len as u128) >> 53) as usize);
        (roll, index)
    }

    pub(crate) fn web_probability(&mut self, numerator: u64, denominator: u64) -> (u64, bool) {
        debug_assert!(denominator > 0 && numerator <= denominator);
        let roll = self.next_unit_53();
        let triggered =
            u128::from(roll) * u128::from(denominator) < u128::from(numerator) * (1_u128 << 53);
        (roll, triggered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_produces_same_stream() {
        let mut left = DeterministicRng::new(42);
        let mut right = DeterministicRng::new(42);
        for _ in 0..100 {
            assert_eq!(left.next_u64(), right.next_u64());
        }
    }
}
