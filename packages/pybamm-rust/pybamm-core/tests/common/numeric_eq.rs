#[allow(dead_code)]
#[track_caller]
pub fn assert_bitwise_eq(lhs: &[f64], rhs: &[f64]) {
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "length mismatch: {} vs {}",
        lhs.len(),
        rhs.len()
    );
    for (i, (l, r)) in lhs.iter().zip(rhs.iter()).enumerate() {
        assert_eq!(
            l.to_bits(),
            r.to_bits(),
            "bit mismatch at index {i}: {l} vs {r}"
        );
    }
}

/// Conservative-mode simplification (and the `zero_propagate` pass it feeds)
/// is value-exact but may normalise the sign of a zero result — see the doc
/// comments on `SimplifyMode::Conservative` and `zero_propagate`. Compare
/// under that contract: a real value divergence still fails, a ±0
/// difference does not. Every other proptest must stay strictly bitwise.
#[allow(dead_code)]
#[track_caller]
pub fn assert_conservative_eq(lhs: &[f64], rhs: &[f64]) {
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "length mismatch: {} vs {}",
        lhs.len(),
        rhs.len()
    );
    for (i, (l, r)) in lhs.iter().zip(rhs.iter()).enumerate() {
        if *l == 0.0 && *r == 0.0 {
            continue; // ±0 both permitted under the sign-of-zero carve-out
        }
        assert_eq!(
            l.to_bits(),
            r.to_bits(),
            "mismatch at index {i} (beyond the sign-of-zero carve-out): {l} vs {r}"
        );
    }
}

#[allow(dead_code)]
#[track_caller]
pub fn assert_close(lhs: &[f64], rhs: &[f64], rtol: f64, atol: f64) {
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "length mismatch: {} vs {}",
        lhs.len(),
        rhs.len()
    );
    for (i, (l, r)) in lhs.iter().zip(rhs.iter()).enumerate() {
        assert!(
            !l.is_nan() && !r.is_nan(),
            "NaN mismatch at index {i}: {l} vs {r}"
        );
        if !l.is_finite() || !r.is_finite() {
            assert!(
                l.to_bits() == r.to_bits(),
                "non-finite mismatch at index {i}: {l} vs {r}"
            );
            continue;
        }
        let tol = rtol.mul_add(l.abs().max(r.abs()), atol);
        assert!(
            (l - r).abs() <= tol,
            "mismatch at index {i}: {l} vs {r} (diff={}, tol={tol})",
            (l - r).abs()
        );
    }
}
