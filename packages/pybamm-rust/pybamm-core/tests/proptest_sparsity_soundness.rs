//! The detected pattern must hold every entry a forward sweep can make non-zero or NaN,
//! which colouring relies on.

mod common;

use common::cases::{
    ALL_EXTRAS, arb_eval_case_with, arb_smooth_tangent_case_with, arb_split_eval_case_with,
};
use proptest::prelude::*;
use pybamm_core::{
    Arena, CompiledExpr, DagRef, NodeId, SplitCompiledExpr, TangentInputs,
    detect_sparsity_per_output, tangent_wrt_states,
};

/// One forward sweep per unit seed, against the pattern's rows.
///
/// A row a zero seed already makes NaN is seed-independent, so it constrains nothing.
fn check_pattern_covers_every_sweep(arena: &Arena, root: NodeId, y: &[f64], inputs: &[f64]) {
    let n_states = y.len();
    let n_rows = CompiledExpr::new(DagRef::new(arena, root)).output_len();
    let pattern = detect_sparsity_per_output(arena, root, n_rows, n_states);

    let mut tangent_arena = arena.clone();
    let tangent_root = tangent_wrt_states(&mut tangent_arena, root);
    let compiled = SplitCompiledExpr::compile(DagRef::new(&tangent_arena, tangent_root));
    let mut scratch = vec![0.0; compiled.scratch_len()];
    let mut sweep = |seed: &[f64]| {
        let tangent = TangentInputs {
            dy: Some(seed),
            dp: None,
        };
        compiled
            .eval_with_tangent(&mut scratch, 0.5, y, &[], inputs, &tangent)
            .to_vec()
    };

    let baseline = sweep(&vec![0.0; n_states]);
    for col in 0..n_states {
        let mut seed = vec![0.0; n_states];
        seed[col] = 1.0;
        for (row, &value) in sweep(&seed).iter().enumerate() {
            let produced = value.is_nan() || value.abs() > 0.0;
            let columns = &pattern.indices()[pattern.row_span(row)];
            assert!(
                !produced || baseline[row].is_nan() || columns.binary_search(&col).is_ok(),
                "seed {col} gives row {row} the value {value}, outside its pattern {columns:?}"
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn pattern_covers_smooth_sweeps(case in arb_smooth_tangent_case_with(ALL_EXTRAS)) {
        check_pattern_covers_every_sweep(&case.arena, case.root, &case.y, &case.inputs);
    }

    #[test]
    fn pattern_covers_positive_state_sweeps(case in arb_split_eval_case_with(ALL_EXTRAS)) {
        check_pattern_covers_every_sweep(&case.arena, case.root, &case.y, &case.inputs);
    }

    /// Adds comparisons, which the pattern leaves out.
    #[test]
    fn pattern_covers_comparison_sweeps(case in arb_eval_case_with(ALL_EXTRAS)) {
        check_pattern_covers_every_sweep(&case.arena, case.root, &case.y, &case.inputs);
    }
}
