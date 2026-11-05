//! Forward-mode tangents against central finite differences and the unoptimized DAG.

mod common;

use common::cases::{
    ALL_EXTRAS, TangentCase, arb_smooth_tangent_case, arb_smooth_tangent_case_with, eval_dag,
    targeted_smooth_tangent_cases,
};
use common::numeric_eq::assert_close;
use proptest::prelude::*;
use pybamm_core::{
    Arena, DagRef, Node, SplitCompiledExpr, TangentInputs, cse, dce, simplify, tangent_wrt_states,
    zero_propagate,
};

const FD_STEP: f64 = 1e-5;
/// Deep smooth compositions amplify FD truncation to ~1e-4 relative.
const RTOL: f64 = 1e-4;
/// Chain-rule factors amplify FD truncation on small derivatives to ~1.5e-6.
const ATOL: f64 = 2e-6;

/// Forward-mode AD agrees with central finite differences for every seed direction.
fn check_ad_matches_fd(case: &TangentCase) {
    let primal = eval_dag(&case.arena, case.root, 0.0, &case.y, &case.inputs);
    assert!(
        primal.iter().all(|v| v.is_finite()),
        "smooth generator produced non-finite primal output: {primal:?}"
    );

    // The AD oracle must not depend on the rewrite pipeline.
    let mut ad_arena = case.arena.clone();
    let tangent_root = tangent_wrt_states(&mut ad_arena, case.root);

    for seed in &case.seeds {
        let compiled = SplitCompiledExpr::compile(DagRef::new(&ad_arena, tangent_root));
        let mut s = vec![0.0; compiled.scratch_len()];
        let mut cache = compiled.eval_primal(&mut s, 0.0, &case.y, &[], &case.inputs);
        let tangent_inputs = TangentInputs {
            dy: Some(seed),
            dp: None,
        };
        let ad_result = cache.eval_tangent(&tangent_inputs).to_vec();

        let y_plus: Vec<f64> = case
            .y
            .iter()
            .zip(seed.iter())
            .map(|(yi, si)| yi + FD_STEP * si)
            .collect();
        let y_minus: Vec<f64> = case
            .y
            .iter()
            .zip(seed.iter())
            .map(|(yi, si)| yi - FD_STEP * si)
            .collect();

        let f_plus = eval_dag(&case.arena, case.root, 0.0, &y_plus, &case.inputs);
        let f_minus = eval_dag(&case.arena, case.root, 0.0, &y_minus, &case.inputs);

        assert!(
            f_plus.iter().chain(f_minus.iter()).all(|v| v.is_finite()),
            "smooth generator produced non-finite FD probe values:\n\
             seed={seed:?}\n\
             f_plus={f_plus:?}\n\
             f_minus={f_minus:?}"
        );

        let fd_result: Vec<f64> = f_plus
            .iter()
            .zip(f_minus.iter())
            .map(|(fp, fm)| (fp - fm) / (2.0 * FD_STEP))
            .collect();

        assert_close(&ad_result, &fd_result, RTOL, ATOL);
    }
}

/// The optimized tangent DAG agrees with the unoptimized one for every seed direction.
fn check_optimized_matches_unoptimized(case: &TangentCase) {
    let mut unopt_arena = case.arena.clone();
    let unopt_root = tangent_wrt_states(&mut unopt_arena, case.root);

    let mut opt_arena = case.arena.clone();
    let opt_root = tangent_wrt_states(&mut opt_arena, case.root);
    let opt_root = simplify(&mut opt_arena, opt_root);
    let folded = zero_propagate(DagRef::new(&opt_arena, opt_root));
    let shared = cse(folded.as_ref());
    let optimized = dce(shared.as_ref());

    for seed in &case.seeds {
        let tangent_inputs = TangentInputs {
            dy: Some(seed),
            dp: None,
        };

        let compiled_unopt = SplitCompiledExpr::compile(DagRef::new(&unopt_arena, unopt_root));
        let mut s_unopt = vec![0.0; compiled_unopt.scratch_len()];
        let mut cache_unopt =
            compiled_unopt.eval_primal(&mut s_unopt, 0.0, &case.y, &[], &case.inputs);
        let unopt_result = cache_unopt.eval_tangent(&tangent_inputs).to_vec();

        let compiled_opt = SplitCompiledExpr::compile(optimized.as_ref());
        let mut s_opt = vec![0.0; compiled_opt.scratch_len()];
        let mut cache_opt = compiled_opt.eval_primal(&mut s_opt, 0.0, &case.y, &[], &case.inputs);
        let opt_result = cache_opt.eval_tangent(&tangent_inputs).to_vec();

        assert!(
            unopt_result.iter().all(|v| v.is_finite()) && opt_result.iter().all(|v| v.is_finite()),
            "smooth derivative generator produced non-finite tangent output:\n\
             seed={seed:?}\n\
             unoptimized={unopt_result:?}\n\
             optimized={opt_result:?}"
        );

        assert_eq!(
            unopt_result.len(),
            opt_result.len(),
            "optimization pipeline changed output dimensionality:\n\
             seed={seed:?}\n\
             unoptimized ({} outputs)={unopt_result:?}\n\
             optimized ({} outputs)={opt_result:?}",
            unopt_result.len(),
            opt_result.len(),
        );
        assert_close(&unopt_result, &opt_result, 1e-12, 1e-14);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn ad_matches_finite_differences(case in arb_smooth_tangent_case()) {
        check_ad_matches_fd(&case);
    }

    #[test]
    fn optimized_derivative_matches_unoptimized(case in arb_smooth_tangent_case()) {
        check_optimized_matches_unoptimized(&case);
    }

    /// Every [`Extra`](common::cases::Extra) the smooth mode admits.
    #[test]
    fn extended_ops_match_finite_differences(case in arb_smooth_tangent_case_with(ALL_EXTRAS)) {
        check_ad_matches_fd(&case);
        check_optimized_matches_unoptimized(&case);
    }
}

#[test]
fn targeted_smooth_tangent_cases_pass() {
    for case in targeted_smooth_tangent_cases() {
        check_ad_matches_fd(&case);
        check_optimized_matches_unoptimized(&case);
    }
}

/// Too stiff for the FD oracle, so AD is checked against the analytic derivative.
#[test]
fn ad_exact_on_stiff_double_exponential() {
    let mut arena = Arena::new();
    let two = arena.alloc(Node::Scalar(2.0));
    let sv0 = arena.alloc(Node::StateVector { start: 0, end: 2 });
    let sub = arena.alloc(Node::Sub(two, sv0));
    let sinh1 = arena.alloc(Node::Sinh(sub));
    let sinh2 = arena.alloc(Node::Sinh(sinh1));
    let cos = arena.alloc(Node::Cos(sinh2));
    let sv1 = arena.alloc(Node::StateVector { start: 0, end: 2 });
    let neg1 = arena.alloc(Node::Neg(sv1));
    let sub1 = arena.alloc(Node::Sub(cos, neg1));
    let sv2 = arena.alloc(Node::StateVector { start: 0, end: 2 });
    let neg2 = arena.alloc(Node::Neg(sv2));
    let root = arena.alloc(Node::Sub(sub1, neg2));

    let y = [0.0, -1.074_148_380_395_331_3];
    let tangent_root = tangent_wrt_states(&mut arena, root);
    let compiled = SplitCompiledExpr::compile(DagRef::new(&arena, tangent_root));
    let mut s = vec![0.0; compiled.scratch_len()];
    let mut cache = compiled.eval_primal(&mut s, 0.0, &y, &[], &[]);

    let analytic: Vec<f64> = y
        .iter()
        .map(|yi| {
            let u = 2.0 - yi;
            let w = u.sinh();
            (w.sinh().sin() * w.cosh()).mul_add(u.cosh(), 2.0)
        })
        .collect();

    for (i, &di) in analytic.iter().enumerate() {
        let mut seed = vec![0.0; y.len()];
        seed[i] = 1.0;
        let tangent_inputs = TangentInputs {
            dy: Some(&seed),
            dp: None,
        };
        let ad = cache.eval_tangent(&tangent_inputs).to_vec();

        let mut expected = vec![0.0; y.len()];
        expected[i] = di;
        assert_close(&ad, &expected, 1e-12, 1e-12);
    }
}
