//! Assertion helpers shared by the crate's test modules.

/// The bit patterns of `values`, so `assert_eq!` on them asserts bit-identical results.
pub fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().copied().map(f64::to_bits).collect()
}

#[cfg(feature = "diffsol")]
pub fn block_bits(blocks: &[Vec<f64>]) -> Vec<Vec<u64>> {
    blocks.iter().map(|block| bits(block)).collect()
}
