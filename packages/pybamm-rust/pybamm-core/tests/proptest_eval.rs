mod common;

use common::cases::{arb_eval_case, eval_dag, targeted_eval_cases};
use common::numeric_eq::assert_bitwise_eq;
use proptest::prelude::*;
use pybamm_core::{CompiledExpr, TypedIr};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(250))]

    #[test]
    fn eval_is_deterministic(case in arb_eval_case()) {
        let result1 = eval_dag(&case.arena, case.root, 0.0, &case.y, &[]);
        let result2 = eval_dag(&case.arena, case.root, 0.0, &case.y, &[]);
        assert_bitwise_eq(&result1, &result2);
    }

    #[test]
    fn independent_compilations_agree(case in arb_eval_case()) {
        let ir1 = TypedIr::from_arena(&case.arena, case.root);
        let ir2 = TypedIr::from_arena(&case.arena, case.root);

        let expr1 = CompiledExpr::from_ir(ir1);
        let expr2 = CompiledExpr::from_ir(ir2);
        let mut s1 = vec![0.0; expr1.scratch_len()];
        let mut s2 = vec![0.0; expr2.scratch_len()];

        let result1 = expr1.eval(&mut s1, 0.0, &case.y, &[], &[]).to_vec();
        let result2 = expr2.eval(&mut s2, 0.0, &case.y, &[], &[]).to_vec();
        assert_bitwise_eq(&result1, &result2);
    }

    #[test]
    fn repeated_eval_on_same_compiled_expr(case in arb_eval_case()) {
        let ir = TypedIr::from_arena(&case.arena, case.root);
        let expr = CompiledExpr::from_ir(ir);
        let mut s = vec![0.0; expr.scratch_len()];

        // Warm up with one eval
        let _ = expr.eval(&mut s, 0.0, &case.y, &[], &[]).to_vec();

        // Eval with shifted input, then repeat — catches buffer state leaks
        let y2: Vec<f64> = case.y.iter().map(|v| v + 0.1).collect();
        let first = expr.eval(&mut s, 0.0, &y2, &[], &[]).to_vec();
        let second = expr.eval(&mut s, 0.0, &y2, &[], &[]).to_vec();
        assert_bitwise_eq(&first, &second);
    }
}

#[test]
fn targeted_eval_cases_are_stable() {
    for case in targeted_eval_cases() {
        let r1 = eval_dag(&case.arena, case.root, 0.0, &case.y, &[]);
        let r2 = eval_dag(&case.arena, case.root, 0.0, &case.y, &[]);
        assert_bitwise_eq(&r1, &r2);
    }
}
