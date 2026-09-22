mod common;

use common::cases::{arb_eval_case, targeted_eval_cases};
use common::numeric_eq::assert_bitwise_eq;
use proptest::prelude::*;
use pybamm_core::{Arena, CompiledExpr, DagRef, NodeId, TypedIr};

/// The slot-reusing tape must match the no-reuse (SSA) one bitwise, an independent
/// oracle for the liveness-driven slot allocator.
#[track_caller]
fn assert_reuse_layout_matches_pinned(arena: &Arena, root: NodeId, y: &[f64]) {
    let eval = |ir: TypedIr| {
        let expr = CompiledExpr::from_ir(ir);
        let mut s = vec![0.0; expr.scratch_len()];
        expr.eval(&mut s, 0.0, y, &[], &[]).to_vec()
    };
    let dag = DagRef::new(arena, root);
    assert_bitwise_eq(
        &eval(TypedIr::from_dag(dag)),
        &eval(TypedIr::from_dag_pinned(dag)),
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(250))]

    #[test]
    fn reuse_layout_matches_pinned_layout(case in arb_eval_case()) {
        assert_reuse_layout_matches_pinned(&case.arena, case.root, &case.y);
    }

    #[test]
    fn repeated_eval_on_same_compiled_expr(case in arb_eval_case()) {
        let expr = CompiledExpr::new(DagRef::new(&case.arena, case.root));
        let mut s = vec![0.0; expr.scratch_len()];

        let _ = expr.eval(&mut s, 0.0, &case.y, &[], &[]).to_vec();

        // Eval with shifted input, then repeat — catches buffer state leaks
        let y2: Vec<f64> = case.y.iter().map(|v| v + 0.1).collect();
        let first = expr.eval(&mut s, 0.0, &y2, &[], &[]).to_vec();
        let second = expr.eval(&mut s, 0.0, &y2, &[], &[]).to_vec();
        assert_bitwise_eq(&first, &second);
    }
}

#[test]
fn targeted_cases_reuse_layout_matches_pinned_layout() {
    for case in targeted_eval_cases() {
        assert_reuse_layout_matches_pinned(&case.arena, case.root, &case.y);
    }
}
