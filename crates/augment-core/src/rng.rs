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
