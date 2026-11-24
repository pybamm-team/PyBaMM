//! Assertion helpers shared by the crate's test modules.

/// The bit patterns of `values`, so `assert_eq!` on them asserts bit-identical results.
pub fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().copied().map(f64::to_bits).collect()
}
