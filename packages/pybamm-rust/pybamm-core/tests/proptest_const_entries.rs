//! Soundness of the constant-entry classifier, and of assembling on it.
//!
//! Two properties, both over random graphs at random states. First: an entry
//! the classifier calls constant must equal what the tangent tape produces for
//! that column, at every state — a wrong constant does not crash, it quietly
//! degrades Newton convergence, so this is the property the whole scheme rests
//! on. Second: assembling with the split on must reproduce the unsplit
//! assembly entry for entry.
//!
//! Non-finite tape values are exempt from both: where a tape overflows, a term
//! the fold drops as exactly zero evaluates to `inf * 0.0` and poisons the
//! sweep, so the two legitimately disagree (see `const_entries`).

mod common;

use common::cases::{TangentCase, arb_split_eval_case, targeted_split_eval_cases};
use proptest::prelude::*;
use pybamm_core::const_entries::classify_constant_entries;
use pybamm_core::jacobian::{JacobianData, JacobianScratch};
use pybamm_core::{
    CompiledExpr, TangentInputs, TypedIr, detect_sparsity_per_output, simplify_pipeline,
    tangent_wrt_states,
};

/// States to probe each case at, spread so a coefficient that merely looks
/// constant at one point does not survive. Contractive only: the generators
/// promise a finite tape at their own `y`, and scaling up walks into the
/// overflow regime where the two paths are allowed to differ.
fn probe_states(y: &[f64]) -> Vec<Vec<f64>> {
    let mut states = vec![y.to_vec()];
    for factor in [0.83_f64, 0.61, 0.42, 0.25] {
        states.push(
            y.iter()
                .enumerate()
                .map(|(i, value)| value * factor.powi(1 + i32::try_from(i % 3).expect("small")))
                .collect(),
        );
    }
    states
}

/// Every column of `d(root)/dy` the tape produces, one unit seed at a time.
///
/// Takes an already-compiled tape: the compile is state-independent, and the
/// callers probe several states per case.
fn sweep_columns(
    expr: &CompiledExpr,
    scratch: &mut [f64],
    n_states: usize,
    t: f64,
    y: &[f64],
) -> Vec<Vec<f64>> {
    let mut cache = expr.eval_primal(scratch, t, y, &[], &[]);
    (0..n_states)
        .map(|col| {
            let mut seed = vec![0.0; n_states];
            seed[col] = 1.0;
            cache
                .eval_tangent(&TangentInputs {
                    dy: Some(&seed),
                    dp: None,
                })
                .to_vec()
        })
        .collect()
}

/// Returns how many entries the classifier proved, so a caller can check the
/// property is not passing vacuously.
fn check_constants_match_the_tape(case: &TangentCase) -> usize {
    let n_states = case.n_states;
    let n_rows = CompiledExpr::new(&case.arena, case.root).output_len();

    let mut diff_arena = case.arena.clone();
    let tangent_root = tangent_wrt_states(&mut diff_arena, case.root);
    let (diff_arena, tangent_root) = simplify_pipeline(diff_arena, tangent_root);
    let pattern = detect_sparsity_per_output(&case.arena, case.root, n_rows, n_states);
    let (varying, entries) = classify_constant_entries(&diff_arena, tangent_root, &pattern);

    assert_eq!(
        varying.iter().filter(|&&v| v).count() + entries.len(),
        pattern.nnz(),
        "every pattern entry is either swept or known"
    );

    let rows = pattern.entry_rows();
    // The tape the fold is checked against is the one just built, compiled once.
    let expr = CompiledExpr::from_ir(TypedIr::from_arena_split_eval(&diff_arena, tangent_root));
    let mut scratch = vec![0.0; expr.scratch_len()];

    for y in probe_states(&case.y) {
        let columns = sweep_columns(&expr, &mut scratch, n_states, 0.5, &y);
        for &(csr_idx, value) in &entries {
            let (row, col) = (rows[csr_idx], pattern.indices[csr_idx]);
            let swept = columns[col][row];
            if !swept.is_finite() {
                continue;
            }
            assert!(
                value.to_bits() == swept.to_bits() || (value == 0.0 && swept == 0.0),
                "entry ({row}, {col}) folded to {value}, tape gave {swept} at y={y:?}"
            );
        }
    }
    entries.len()
}

#[allow(clippy::float_cmp)] // exact equality is the point: pins the two paths
fn check_split_assembly_matches_unsplit(case: &TangentCase) {
    let n_states = case.n_states;
    let n_rows = CompiledExpr::new(&case.arena, case.root).output_len();
    let split = JacobianData::new_wrt_states(&case.arena, case.root, n_rows, n_states);
    let reference = JacobianData::new_wrt_states_unsplit(&case.arena, case.root, n_rows, n_states);

    // Scratch and layout depend only on the artifact, so they are minted once
    // here rather than per probe state.
    let buffers = |jac: &JacobianData| {
        (
            JacobianScratch::new(jac),
            vec![f64::NAN; jac.layout().n_slots()],
        )
    };
    let (mut split_bufs, mut reference_bufs) = (buffers(&split), buffers(&reference));
    let assemble = |jac: &JacobianData, bufs: &mut (JacobianScratch, Vec<f64>), y: &[f64]| {
        let (scratch, data) = bufs;
        jac.assemble_into(scratch, jac.layout(), 0.5, y, &[], &[], data);
        data.clone()
    };

    for y in probe_states(&case.y) {
        let (actual, expected) = (
            assemble(&split, &mut split_bufs, &y),
            assemble(&reference, &mut reference_bufs, &y),
        );
        for (csc_idx, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
            if !want.is_finite() {
                continue;
            }
            let (row, col) = split.csc().csc_to_csr_map[csc_idx];
            assert!(
                got == want,
                "entry ({row}, {col}): split gave {got}, unsplit {want} at y={y:?}"
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn constants_match_the_tape(case in arb_split_eval_case()) {
        check_constants_match_the_tape(&case);
    }

    #[test]
    fn split_assembly_matches_unsplit(case in arb_split_eval_case()) {
        check_split_assembly_matches_unsplit(&case);
    }
}

#[test]
fn targeted_cases_classify_soundly() {
    let mut proved = 0;
    for case in targeted_split_eval_cases() {
        proved += check_constants_match_the_tape(&case);
        check_split_assembly_matches_unsplit(&case);
    }
    assert!(
        proved > 0,
        "the targeted cases must exercise the fold at all"
    );
}
