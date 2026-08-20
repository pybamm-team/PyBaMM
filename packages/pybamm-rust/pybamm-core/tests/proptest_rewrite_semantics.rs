mod common;

use common::cases::{arb_eval_case, duplicate_subexpr_case, eval_dag, targeted_eval_cases};
use common::numeric_eq::{assert_bitwise_eq, assert_conservative_eq};
use proptest::prelude::*;
use pybamm_core::{cse, simplify, zero_propagate};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(250))]

    #[test]
    fn cse_preserves_eval(case in arb_eval_case()) {
        let original = eval_dag(&case.arena, case.root, 0.0, &case.y, &[]);
        let (cse_arena, cse_root) = cse(&case.arena, case.root);
        let after = eval_dag(&cse_arena, cse_root, 0.0, &case.y, &[]);
        assert_bitwise_eq(&original, &after);
    }

    #[test]
    fn simplify_conservative_preserves_eval(case in arb_eval_case()) {
        let original = eval_dag(&case.arena, case.root, 0.0, &case.y, &[]);
        let mut arena_copy = case.arena.clone();
        let simplified_root = simplify(&mut arena_copy, case.root);
        let after = eval_dag(&arena_copy, simplified_root, 0.0, &case.y, &[]);
        assert_conservative_eq(&original, &after);
    }

    #[test]
    fn zero_propagate_preserves_eval(case in arb_eval_case()) {
        let original = eval_dag(&case.arena, case.root, 0.0, &case.y, &[]);
        let (zp_arena, zp_root) = zero_propagate(&case.arena, case.root);
        let after = eval_dag(&zp_arena, zp_root, 0.0, &case.y, &[]);
        assert_conservative_eq(&original, &after);
    }
}

#[test]
fn rewrite_passes_preserve_eval_on_targeted_cases() {
    for case in targeted_eval_cases() {
        let original = eval_dag(&case.arena, case.root, 0.0, &case.y, &[]);

        let (cse_arena, cse_root) = cse(&case.arena, case.root);
        let cse_after = eval_dag(&cse_arena, cse_root, 0.0, &case.y, &[]);
        assert_bitwise_eq(&original, &cse_after);

        let mut arena_copy = case.arena.clone();
        let simplified_root = simplify(&mut arena_copy, case.root);
        let simplified_after = eval_dag(&arena_copy, simplified_root, 0.0, &case.y, &[]);
        assert_bitwise_eq(&original, &simplified_after);

        let (zp_arena, zp_root) = zero_propagate(&case.arena, case.root);
        let zp_after = eval_dag(&zp_arena, zp_root, 0.0, &case.y, &[]);
        assert_bitwise_eq(&original, &zp_after);
    }
}

#[test]
fn cse_reduces_known_duplicate_graph() {
    let case = duplicate_subexpr_case();
    let (cse_arena, _) = cse(&case.arena, case.root);
    assert!(
        cse_arena.len() < case.arena.len(),
        "CSE failed to reduce known duplicates: {} -> {}",
        case.arena.len(),
        cse_arena.len(),
    );
}
