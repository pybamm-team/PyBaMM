//! Soundness of the constant-entry classifier, and of assembling on it.
//! Non-finite tape values are exempt: an overflow turns a folded zero into `inf * 0.0`.

mod common;

use common::cases::{
    ALL_EXTRAS, DagCase, TangentCase, arb_eval_case_with, arb_smooth_tangent_case_with,
    arb_split_eval_case_with, targeted_split_eval_cases,
};
use proptest::prelude::*;
use pybamm_core::const_entries::classify_constant_entries;
use pybamm_core::jacobian::{JacobianData, JacobianScratch};
use pybamm_core::{
    Arena, CompiledExpr, Dag, DagRef, NodeId, SplitCompiledExpr, TangentInputs,
    detect_sparsity_per_output, simplify_pipeline, tangent_wrt_states,
};

/// One generated expression and the point it was drawn at.
#[derive(Clone, Debug)]
struct Case {
    arena: Arena,
    root: NodeId,
    y: Vec<f64>,
    inputs: Vec<f64>,
}

impl From<TangentCase> for Case {
    fn from(case: TangentCase) -> Self {
        let TangentCase {
            arena,
            root,
            y,
            inputs,
            ..
        } = case;
        Self {
            arena,
            root,
            y,
            inputs,
        }
    }
}

impl From<DagCase> for Case {
    fn from(case: DagCase) -> Self {
        let DagCase {
            arena,
            root,
            y,
            inputs,
            ..
        } = case;
        Self {
            arena,
            root,
            y,
            inputs,
        }
    }
}

/// Every extra, at positive states, at signed ones near zero, and with comparisons and steps.
fn arb_case() -> impl Strategy<Value = Case> {
    prop_oneof![
        arb_split_eval_case_with(ALL_EXTRAS).prop_map(Case::from),
        arb_smooth_tangent_case_with(ALL_EXTRAS).prop_map(Case::from),
        arb_eval_case_with(ALL_EXTRAS).prop_map(Case::from),
    ]
}

/// A probe point `(t, y, inputs)`.
type Probe = (f64, Vec<f64>, Vec<f64>);

/// Probe points; contractive only, since scaling up can overflow the tape, and `|t| <= 1.5`.
fn probe_points(y: &[f64], inputs: &[f64]) -> Vec<Probe> {
    let mut points = vec![(0.5, y.to_vec(), inputs.to_vec())];
    for (factor, t) in [(0.83_f64, -1.25), (0.61, 1.5), (0.42, 0.0), (0.25, -0.5)] {
        let scaled = |values: &[f64]| -> Vec<f64> {
            values
                .iter()
                .enumerate()
                .map(|(i, value)| value * factor.powi(1 + i32::try_from(i % 3).expect("small")))
                .collect()
        };
        points.push((t, scaled(y), scaled(inputs)));
    }
    points
}

/// Every column of `d(root)/dy` the tape produces, one unit seed at a time.
fn sweep_columns(
    expr: &SplitCompiledExpr,
    scratch: &mut [f64],
    n_states: usize,
    (t, y, inputs): &Probe,
) -> Vec<Vec<f64>> {
    let mut cache = expr.eval_primal(scratch, *t, y, &[], inputs);
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

/// Returns the number of proved entries, so callers can rule out a vacuous pass.
fn check_constants_match_the_tape(case: &Case) -> usize {
    let n_states = case.y.len();
    let n_rows = CompiledExpr::new(DagRef::new(&case.arena, case.root)).output_len();

    let mut tangent = Dag::new(case.arena.clone(), case.root);
    tangent.rewrite_in_place(tangent_wrt_states);
    let simplified = simplify_pipeline(tangent);
    let pattern = detect_sparsity_per_output(&case.arena, case.root, n_rows, n_states);
    let (varying, entries) =
        classify_constant_entries(simplified.arena(), simplified.root(), &pattern);

    assert_eq!(
        varying.iter().filter(|&&v| v).count() + entries.len(),
        pattern.nnz(),
        "every pattern entry is either swept or known"
    );

    let rows = pattern.entry_rows();
    let expr = SplitCompiledExpr::compile(simplified.as_ref());
    let mut scratch = vec![0.0; expr.scratch_len()];

    for probe in probe_points(&case.y, &case.inputs) {
        let columns = sweep_columns(&expr, &mut scratch, n_states, &probe);
        for &(csr_idx, value) in &entries {
            let (row, col) = (rows[csr_idx], pattern.indices()[csr_idx]);
            let swept = columns[col][row];
            if !swept.is_finite() {
                continue;
            }
            assert!(
                value.to_bits() == swept.to_bits() || (value == 0.0 && swept == 0.0),
                "entry ({row}, {col}) folded to {value}, tape gave {swept} at {probe:?}"
            );
        }
    }
    entries.len()
}

#[allow(clippy::float_cmp)]
fn check_split_assembly_matches_unsplit(case: &Case) {
    let n_states = case.y.len();
    let n_rows = CompiledExpr::new(DagRef::new(&case.arena, case.root)).output_len();
    let split = JacobianData::new_wrt_states(&case.arena, case.root, n_rows, n_states);
    let reference =
        JacobianData::new_wrt_states_sweeping_constants(&case.arena, case.root, n_rows, n_states);

    let buffers = |jac: &JacobianData| {
        (
            JacobianScratch::new(jac),
            vec![f64::NAN; jac.layout().n_slots()],
        )
    };
    let (mut split_bufs, mut reference_bufs) = (buffers(&split), buffers(&reference));
    let assemble = |jac: &JacobianData, bufs: &mut (JacobianScratch, Vec<f64>), probe: &Probe| {
        let ((scratch, data), (t, y, inputs)) = (bufs, probe);
        jac.assemble_into(scratch, jac.layout(), *t, y, &[], inputs, data);
        data.clone()
    };

    for probe in probe_points(&case.y, &case.inputs) {
        let (actual, expected) = (
            assemble(&split, &mut split_bufs, &probe),
            assemble(&reference, &mut reference_bufs, &probe),
        );
        for ((row, col), (&got, &want)) in split.csc().entries().zip(actual.iter().zip(&expected)) {
            if !want.is_finite() {
                continue;
            }
            assert!(
                got == want,
                "entry ({row}, {col}): split gave {got}, unsplit {want} at {probe:?}"
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn constants_match_the_tape(case in arb_case()) {
        check_constants_match_the_tape(&case);
    }

    #[test]
    fn split_assembly_matches_unsplit(case in arb_case()) {
        check_split_assembly_matches_unsplit(&case);
    }
}

#[test]
fn targeted_cases_classify_soundly() {
    let mut proved = 0;
    for case in targeted_split_eval_cases().into_iter().map(Case::from) {
        proved += check_constants_match_the_tape(&case);
        check_split_assembly_matches_unsplit(&case);
    }
    assert!(
        proved > 0,
        "the targeted cases must exercise the fold at all"
    );
}
