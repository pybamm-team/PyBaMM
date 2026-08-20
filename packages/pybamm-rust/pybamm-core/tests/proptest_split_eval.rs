mod common;

use common::cases::{TangentCase, arb_split_eval_case, targeted_split_eval_cases};
use common::numeric_eq::assert_bitwise_eq;
use proptest::prelude::*;
use pybamm_core::{
    CompiledExpr, SimplifyMode, TangentInputs, TypedIr, cse, dce, simplify_with_mode,
    tangent_wrt_states, zero_propagate,
};

// Reusable check functions (called from both proptests and targeted tests)

fn check_split_eval_matches_monolithic(case: &TangentCase) {
    let mut arena = case.arena.clone();
    let tangent_root = tangent_wrt_states(&mut arena, case.root);

    for seed in &case.seeds {
        let tangent_inputs = TangentInputs {
            dy: Some(seed),
            dp: None,
        };

        // Monolithic eval
        let ir_mono = TypedIr::from_arena(&arena, tangent_root);
        let mono = CompiledExpr::from_ir(ir_mono);
        let mut s_mono = vec![0.0; mono.scratch_len()];
        let mono_result = mono
            .eval_with_tangent(&mut s_mono, 0.0, &case.y, &[], &[], &tangent_inputs)
            .to_vec();

        // Partitioned tape
        let ir_split = TypedIr::from_arena_split_eval(&arena, tangent_root);
        let split = CompiledExpr::from_ir(ir_split);
        let mut s_split = vec![0.0; split.scratch_len()];
        let mut cache = split.eval_primal(&mut s_split, 0.0, &case.y, &[], &[]);
        let split_result = cache.eval_tangent(&tangent_inputs).to_vec();

        assert_bitwise_eq(&mono_result, &split_result);
    }
}

fn check_repeated_tangent_stable(case: &TangentCase) {
    let mut arena = case.arena.clone();
    let tangent_root = tangent_wrt_states(&mut arena, case.root);

    let ir = TypedIr::from_arena_split_eval(&arena, tangent_root);
    let compiled = CompiledExpr::from_ir(ir);
    let mut s = vec![0.0; compiled.scratch_len()];
    let mut cache = compiled.eval_primal(&mut s, 0.0, &case.y, &[], &[]);

    for seed in &case.seeds {
        let tangent_inputs = TangentInputs {
            dy: Some(seed),
            dp: None,
        };

        let first = cache.eval_tangent(&tangent_inputs).to_vec();
        let second = cache.eval_tangent(&tangent_inputs).to_vec();
        assert_bitwise_eq(&first, &second);
    }
}

fn check_split_eval_after_pipeline(case: &TangentCase) {
    let mut arena = case.arena.clone();
    let root = tangent_wrt_states(&mut arena, case.root);
    let root = simplify_with_mode(&mut arena, root, SimplifyMode::Aggressive);
    let (arena, root) = zero_propagate(&arena, root);
    let (arena, root) = cse(&arena, root);
    let (arena, root) = dce(&arena, root);

    for seed in &case.seeds {
        let tangent_inputs = TangentInputs {
            dy: Some(seed),
            dp: None,
        };

        // Monolithic tape
        let ir_mono = TypedIr::from_arena(&arena, root);
        let mono = CompiledExpr::from_ir(ir_mono);
        let mut s_mono = vec![0.0; mono.scratch_len()];
        let mono_result = mono
            .eval_with_tangent(&mut s_mono, 0.0, &case.y, &[], &[], &tangent_inputs)
            .to_vec();

        // Partitioned tape
        let ir_split = TypedIr::from_arena_split_eval(&arena, root);
        let split = CompiledExpr::from_ir(ir_split);
        let mut s_split = vec![0.0; split.scratch_len()];
        let mut cache = split.eval_primal(&mut s_split, 0.0, &case.y, &[], &[]);
        let split_result = cache.eval_tangent(&tangent_inputs).to_vec();

        assert_bitwise_eq(&mono_result, &split_result);
    }
}

// Proptest properties

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// `eval_primal` then `PrimalCache::eval_tangent` must produce bitwise
    /// identical tangent output to monolithic `eval_with_tangent`.
    #[test]
    fn split_eval_matches_monolithic(case in arb_split_eval_case()) {
        check_split_eval_matches_monolithic(&case);
    }

    /// Repeated tangent-only evaluation after a single primal must be stable.
    #[test]
    fn split_eval_repeated_tangent_is_stable(case in arb_split_eval_case()) {
        check_repeated_tangent_stable(&case);
    }

    /// The split tape must still agree with monolithic after the production
    /// derivative pipeline `jacobian.rs` runs: tangent_wrt_states ->
    /// simplify(Aggressive) -> zero_propagate -> cse -> dce ->
    /// from_arena_split_eval.
    #[test]
    fn split_eval_after_production_pipeline(case in arb_split_eval_case()) {
        check_split_eval_after_pipeline(&case);
    }
}

// Targeted test: named Conditional, Index, wide fan-out, sparse matmul cases

#[test]
fn targeted_split_eval_cases_pass() {
    for case in targeted_split_eval_cases() {
        check_split_eval_matches_monolithic(&case);
        check_repeated_tangent_stable(&case);
        check_split_eval_after_pipeline(&case);
    }
}
