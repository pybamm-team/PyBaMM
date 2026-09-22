//! The residual tape `CompiledModel::new` compiles must be bit-identical to a
//! direct evaluation of the DAG

#![cfg(feature = "serialize")]

use pybamm_core::{CompiledExpr, CompiledModel, CompiledModelOptions, CsrData, DagSnapshot, Shape};

const SPM: &[u8] = include_bytes!("../benches/fixtures/spm.bin");
const SPME: &[u8] = include_bytes!("../benches/fixtures/spme.bin");
const DFN: &[u8] = include_bytes!("../benches/fixtures/dfn.bin");

const TRIALS: usize = 400;

/// xorshift64*, so the sweep is reproducible without a rand dependency.
struct XorShift(u64);

impl XorShift {
    const fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A positive state spanning 1e-6..1e4, plus an occasional exact zero.
    fn state(&mut self) -> f64 {
        let draw = self.next_u64();
        if draw.is_multiple_of(17) {
            return 0.0;
        }
        let exponent = (draw % 11) as i32 - 6;
        self.unit().mul_add(0.9, 0.1) * 10f64.powi(exponent)
    }

    fn states(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.state()).collect()
    }
}

fn identity_mass(n: usize) -> CsrData {
    CsrData::try_new(
        (0..=n).collect(),
        (0..n).collect(),
        vec![1.0; n],
        Shape::matrix(n, n),
    )
    .expect("identity mass matrix")
}

fn check(name: &str, bytes: &[u8]) {
    let snap = DagSnapshot::from_bytes(bytes).expect("fixture decodes");
    let mass = snap
        .mass_matrix
        .clone()
        .unwrap_or_else(|| identity_mass(snap.n_states));
    let model = CompiledModel::new(
        snap.dag().arena(),
        snap.dag().root(),
        mass,
        snap.n_states,
        snap.packed_input_width,
        CompiledModelOptions::default(),
    )
    .expect("a valid fixture");
    let mut ws = model.create_workspace();

    let reference = CompiledExpr::new(snap.dag());
    let mut scratch = vec![0.0; reference.scratch_len()];

    let inputs = vec![0.0; snap.packed_input_width.get()];
    // f, not M*y' - f: `yp - (yp - f) == f` is not a floating-point identity.
    let mut f = vec![0.0; model.output_len()];
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);

    for trial in 0..TRIALS {
        let y = rng.states(snap.n_states);
        let t = rng.unit() * 3600.0;

        let want = reference.eval(&mut scratch, t, &y, &[], &inputs).to_vec();
        model.eval_rhs(&mut ws, t, &y, &inputs, &mut f);

        for (i, (&got, &want_i)) in f.iter().zip(&want).enumerate() {
            assert_eq!(
                got.to_bits(),
                want_i.to_bits(),
                "{name} trial {trial}: residual bit mismatch at output {i}: \
                 compiled {got} vs raw-DAG {want_i}. The rhs compile path has \
                 gained a pass that moves bits; see the guard in CompiledModel::new."
            );
        }
    }
}

#[test]
fn residual_is_compiled_bit_exactly_spm() {
    check("spm", SPM);
}

#[test]
fn residual_is_compiled_bit_exactly_spme() {
    check("spme", SPME);
}

#[test]
fn residual_is_compiled_bit_exactly_dfn() {
    check("dfn", DFN);
}
