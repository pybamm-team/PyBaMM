//! Fused multiply-add accumulation for matrix products.
//!
//! Stencil rows cancel large terms, so an unfused product's rounding lands at their scale.
//! A CPU without an FMA instruction accumulates unfused, as a software `mul_add` halves the speed.

// Products must inline into `pulp`'s feature-enabled wrapper to compile as FMA instructions.
#![allow(clippy::inline_always)]

/// How a matrix product accumulates `acc + a * b`.
pub trait Accumulate {
    fn mul_acc(acc: f64, a: f64, b: f64) -> f64;
}

/// `acc + a * b` with a single rounding.
pub enum Fused {}

/// `acc + a * b` with the product and the sum each rounded.
pub enum Unfused {}

impl Accumulate for Fused {
    #[inline(always)]
    fn mul_acc(acc: f64, a: f64, b: f64) -> f64 {
        a.mul_add(b, acc)
    }
}

impl Accumulate for Unfused {
    #[inline(always)]
    fn mul_acc(acc: f64, a: f64, b: f64) -> f64 {
        acc + a * b
    }
}

/// A matrix-product loop, generic over its accumulation.
pub trait Product {
    type Output;
    fn run<A: Accumulate>(self) -> Self::Output;
}

/// The widest instruction set this CPU supports, detected once.
#[cfg(target_arch = "x86_64")]
static ARCH: std::sync::LazyLock<pulp::Arch> = std::sync::LazyLock::new(pulp::Arch::new);

/// Whether the target guarantees an FMA instruction.
#[cfg(not(target_arch = "x86_64"))]
const TARGET_FMA: bool = cfg!(any(target_arch = "aarch64", target_feature = "fma"));

/// Hands a [`Product`] to `pulp`, which compiles it for the detected instruction set.
#[cfg(target_arch = "x86_64")]
struct Fma<P>(P);

#[cfg(target_arch = "x86_64")]
impl<P: Product> pulp::WithSimd for Fma<P> {
    type Output = P::Output;

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, _simd: S) -> P::Output {
        self.0.run::<Fused>()
    }
}

/// Run `product`, fused where the CPU has an FMA instruction.
#[inline]
pub fn dispatch<P: Product>(product: P) -> P::Output {
    // Baseline x86-64 lacks FMA, so `pulp` compiles the fused loop for the detected set.
    #[cfg(target_arch = "x86_64")]
    {
        match *ARCH {
            pulp::Arch::Scalar => product.run::<Unfused>(),
            arch => arch.dispatch(Fma(product)),
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        if TARGET_FMA {
            product.run::<Fused>()
        } else {
            product.run::<Unfused>()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `-(1 + 2^-29) + (1 + 2^-30)^2`, whose exact value `2^-60` an unfused product rounds away.
    struct CancellingRow;

    impl Product for CancellingRow {
        type Output = f64;
        fn run<A: Accumulate>(self) -> f64 {
            let (a, b) = (1.0 + 2f64.powi(-29), 1.0 + 2f64.powi(-30));
            A::mul_acc(A::mul_acc(0.0, -1.0, a), b, b)
        }
    }

    #[test]
    fn a_cancelling_row_rounds_once_per_term_only_when_fused() {
        assert_eq!(
            CancellingRow.run::<Fused>().to_bits(),
            2f64.powi(-60).to_bits()
        );
        assert_eq!(CancellingRow.run::<Unfused>().to_bits(), 0f64.to_bits());
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn aarch64_always_fuses() {
        assert_eq!(dispatch(CancellingRow).to_bits(), 2f64.powi(-60).to_bits());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn an_x86_cpu_fuses_exactly_when_it_has_avx2_and_fma() {
        // Guards the detection, so a CPU that could fuse never silently runs unfused.
        let has_fma = std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma");
        let expected = if has_fma { 2f64.powi(-60) } else { 0.0 };
        assert_eq!(dispatch(CancellingRow).to_bits(), expected.to_bits());
    }
}
