mod common;

use common::cases::{
    TangentCase, arb_smooth_tangent_case, eval_dag, targeted_smooth_tangent_cases,
};
use common::numeric_eq::assert_close;
use proptest::prelude::*;
use pybamm_core::{
    Arena, ArrayData, CompiledExpr, CsrData, CubicInterpolantData, InterpolantData, JacobianData,
    JacobianScratch, NdInterpolantData, Node, NodeId, Shape, SimplifyMode, TangentInputs, TypedIr,
    adjoint::AdjointTape, cse, dce, simplify_with_mode, tangent_wrt_states, zero_propagate,
};

const FD_STEP: f64 = 1e-5;
/// Central FD with step h=1e-5 has O(h^2) truncation ≈ 1e-10.
/// Deep smooth compositions (12 layers) amplify higher-order terms
/// to ~1e-4 relative error on moderate-valued derivatives.
const RTOL: f64 = 1e-4;
/// For small derivatives (~1e-3), truncation error ~1e-10 is amplified
/// by chain-rule factors to ~1.5e-6 absolute. Observed worst case across
/// 10k+ proptest trials is 1.52e-6, so 2e-6 gives margin without
/// hiding real errors.
const ATOL: f64 = 2e-6;

// Reusable check functions (called from both proptests and targeted tests)

/// Check that forward-mode AD agrees with central finite differences
/// for every seed direction in the case.
///
/// Panics on mismatch (suitable for both proptest and `#[test]` contexts).
fn check_ad_matches_fd(case: &TangentCase) {
    // Smooth generator contract: primal evaluation must be finite.
    let primal = eval_dag(&case.arena, case.root, 0.0, &case.y, &[]);
    assert!(
        primal.iter().all(|v| v.is_finite()),
        "smooth generator produced non-finite primal output: {primal:?}"
    );

    // This check is the AD oracle, so it must not depend on the rewrite pipeline
    // beyond `tangent_wrt_states` itself.
    let mut ad_arena = case.arena.clone();
    let tangent_root = tangent_wrt_states(&mut ad_arena, case.root);

    // Test every seed direction in the case: basis + dense directions.
    for seed in &case.seeds {
        // AD evaluation via split-eval
        let ir = TypedIr::from_arena_split_eval(&ad_arena, tangent_root);
        let compiled = CompiledExpr::from_ir(ir);
        let mut s = vec![0.0; compiled.scratch_len()];
        let mut cache = compiled.eval_primal(&mut s, 0.0, &case.y, &[], &[]);
        let tangent_inputs = TangentInputs {
            dy: Some(seed),
            dp: None,
        };
        let ad_result = cache.eval_tangent(&tangent_inputs).to_vec();

        // Central finite difference: (f(y + h*seed) - f(y - h*seed)) / (2h)
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

        let f_plus = eval_dag(&case.arena, case.root, 0.0, &y_plus, &[]);
        let f_minus = eval_dag(&case.arena, case.root, 0.0, &y_minus, &[]);

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

/// Reverse gradient of a scalar row vs (a) the forward-JVP row assembled through
/// the crate's trusted forward-mode path, tight (1e-12/1e-14), catching any
/// AD-path divergence, and (b) central FD: loose, independent. `ctx` labels
/// the failing case. Row of `J @ e_j` is `df/dy_j`.
fn assert_reverse_matches_forward_and_fd(ctx: &str, arena: &Arena, root: NodeId, y: &[f64]) {
    let n = y.len();

    // Reverse row.
    let tape = AdjointTape::new(arena, root, n);
    let mut scratch = vec![0.0; tape.scratch_len()];
    let mut bar = vec![0.0; tape.scratch_len()];
    let mut grad = vec![0.0; n];
    tape.assemble(&mut scratch, &mut bar, &mut grad, 0.5, y, &[], &[]);

    // Forward-JVP row via the trusted forward-mode path, and central FD.
    let mut ad_arena = arena.clone();
    let tangent_root = tangent_wrt_states(&mut ad_arena, root);
    let compiled = CompiledExpr::from_ir(TypedIr::from_arena_split_eval(&ad_arena, tangent_root));
    let mut s = vec![0.0; compiled.scratch_len()];
    let mut seed = vec![0.0; n];
    let primal = CompiledExpr::new(arena, root);
    let mut ps = vec![0.0; primal.scratch_len()];
    let eps = 1e-6;

    for j in 0..n {
        // Tight forward-JVP oracle: row of J @ e_j.
        seed.fill(0.0);
        seed[j] = 1.0;
        let mut cache = compiled.eval_primal(&mut s, 0.5, y, &[], &[]);
        let jvp = cache.eval_tangent(&TangentInputs {
            dy: Some(&seed),
            dp: None,
        })[0];
        assert!(
            (grad[j] - jvp).abs() <= 1e-12f64.mul_add(jvp.abs(), 1e-14),
            "{ctx} col {j}: reverse {} vs forward-JVP {jvp}",
            grad[j]
        );

        // Independent central-FD oracle.
        let mut yp = y.to_vec();
        let mut ym = y.to_vec();
        yp[j] += eps;
        ym[j] -= eps;
        let fp = primal.eval(&mut ps, 0.5, &yp, &[], &[])[0];
        let fm = primal.eval(&mut ps, 0.5, &ym, &[], &[])[0];
        let fd = (fp - fm) / (2.0 * eps);
        assert!(
            (grad[j] - fd).abs() <= 1e-4 * (1.0 + fd.abs()),
            "{ctx} col {j}: reverse {} vs fd {fd}",
            grad[j]
        );
    }
}

/// The crate intentionally treats first-derivative interpolant nodes as
/// first-order terminals. Verify reverse and forward AD both return zero.
fn assert_reverse_and_forward_are_zero(ctx: &str, arena: &Arena, root: NodeId, y: &[f64]) {
    let n = y.len();
    let tape = AdjointTape::new(arena, root, n);
    let mut scratch = vec![0.0; tape.scratch_len()];
    let mut bar = vec![0.0; tape.scratch_len()];
    let mut grad = vec![0.0; n];
    tape.assemble(&mut scratch, &mut bar, &mut grad, 0.5, y, &[], &[]);

    let mut ad_arena = arena.clone();
    let tangent_root = tangent_wrt_states(&mut ad_arena, root);
    let compiled = CompiledExpr::from_ir(TypedIr::from_arena_split_eval(&ad_arena, tangent_root));
    let mut ad_scratch = vec![0.0; compiled.scratch_len()];
    let mut seed = vec![0.0; n];
    for j in 0..n {
        seed.fill(0.0);
        seed[j] = 1.0;
        let result = compiled.eval_with_tangent(
            &mut ad_scratch,
            0.5,
            y,
            &[],
            &[],
            &TangentInputs {
                dy: Some(&seed),
                dp: None,
            },
        );
        assert!(
            grad[j].abs() <= f64::EPSILON,
            "{ctx} reverse col {j}: {}",
            grad[j]
        );
        assert!(
            result[0].abs() <= f64::EPSILON,
            "{ctx} forward col {j}: {}",
            result[0]
        );
    }
}

/// Check that the optimized derivative DAG (after the production pipeline)
/// agrees with the unoptimized derivative DAG for every seed direction.
///
/// Production pipeline: `tangent_wrt_states` -> simplify(Aggressive)
/// -> `zero_propagate` -> cse -> dce
fn check_optimized_matches_unoptimized(case: &TangentCase) {
    // Build unoptimized tangent
    let mut unopt_arena = case.arena.clone();
    let unopt_root = tangent_wrt_states(&mut unopt_arena, case.root);

    // Build optimized tangent (mirrors model.rs production pipeline)
    let mut opt_arena = case.arena.clone();
    let opt_root = tangent_wrt_states(&mut opt_arena, case.root);
    let opt_root = simplify_with_mode(&mut opt_arena, opt_root, SimplifyMode::Aggressive);
    let (opt_arena, opt_root) = zero_propagate(&opt_arena, opt_root);
    let (opt_arena, opt_root) = cse(&opt_arena, opt_root);
    let (opt_arena, opt_root) = dce(&opt_arena, opt_root);

    for seed in &case.seeds {
        let tangent_inputs = TangentInputs {
            dy: Some(seed),
            dp: None,
        };

        // Unoptimized eval
        let ir_unopt = TypedIr::from_arena_split_eval(&unopt_arena, unopt_root);
        let compiled_unopt = CompiledExpr::from_ir(ir_unopt);
        let mut s_unopt = vec![0.0; compiled_unopt.scratch_len()];
        let mut cache_unopt = compiled_unopt.eval_primal(&mut s_unopt, 0.0, &case.y, &[], &[]);
        let unopt_result = cache_unopt.eval_tangent(&tangent_inputs).to_vec();

        // Optimized eval
        let ir_opt = TypedIr::from_arena_split_eval(&opt_arena, opt_root);
        let compiled_opt = CompiledExpr::from_ir(ir_opt);
        let mut s_opt = vec![0.0; compiled_opt.scratch_len()];
        let mut cache_opt = compiled_opt.eval_primal(&mut s_opt, 0.0, &case.y, &[], &[]);
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

// Proptest properties

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Forward-mode AD must agree with central finite differences
    /// on smooth, finite-valued expressions.
    #[test]
    fn ad_matches_finite_differences(case in arb_smooth_tangent_case()) {
        check_ad_matches_fd(&case);
    }

    /// The optimized derivative DAG (after the production pipeline)
    /// must agree with the unoptimized derivative DAG.
    ///
    /// Production pipeline: tangent_wrt_states -> simplify(Aggressive)
    /// -> zero_propagate -> cse -> dce
    #[test]
    fn optimized_derivative_matches_unoptimized(case in arb_smooth_tangent_case()) {
        check_optimized_matches_unoptimized(&case);
    }

    /// Assembled df/dy matches central finite differences column-by-column
    /// on smooth random trees.
    #[test]
    fn jacobian_data_wrt_states_matches_fd(case in arb_smooth_tangent_case()) {
        let TangentCase { arena, root, y, n_states, .. } = case;
        prop_assume!(n_states > 0);
        let primal = CompiledExpr::new(&arena, root);
        let n_rows = primal.output_len();
        let jac = JacobianData::new_wrt_states(&arena, root, n_rows, n_states);

        // The production driver, at the lane width this tape would really run.
        let layout = jac.layout();
        let mut scratch = JacobianScratch::new(&jac);
        let mut data = vec![0.0; layout.n_slots()];
        jac.assemble_into(&mut scratch, layout, 0.5, &y, &[], &[], &mut data);

        let mut s = vec![0.0; primal.scratch_len()];
        let eps = 1e-6;
        for col in 0..jac.n_cols() {
            let mut yp = y.clone(); yp[col] += eps;
            let mut ym = y.clone(); ym[col] -= eps;
            let fp = primal.eval(&mut s, 0.5, &yp, &[], &[]).to_vec();
            let fm = primal.eval(&mut s, 0.5, &ym, &[], &[]).to_vec();
            let (lo, hi) = (jac.csc().colptr[col], jac.csc().colptr[col + 1]);
            for (&dk, &row) in data[lo..hi].iter().zip(&jac.csc().rowind[lo..hi]) {
                let fd = (fp[row] - fm[row]) / (2.0 * eps);
                prop_assert!((dk - fd).abs() <= 1e-4 * (1.0 + fd.abs()),
                    "entry ({},{}): assembled {} vs fd {}", row, col, dk, fd);
            }
        }
    }

    /// Reverse gradient of each output row matches the forward-JVP row (tight) and
    /// central FD (loose) across generated smooth expressions.
    #[test]
    fn reverse_row_gradient_matches_forward_and_fd(case in arb_smooth_tangent_case()) {
        let TangentCase { mut arena, root, y, n_states, .. } = case;
        prop_assume!(n_states > 0);
        let n_rows = CompiledExpr::new(&arena, root).output_len();
        for r in 0..n_rows {
            let row = arena.alloc(Node::Index { child: root, start: r, end: r + 1 });
            assert_reverse_matches_forward_and_fd(&format!("row {r}"), &arena, row, &y);
        }
    }

    /// df/dp on random trees via parameter grafting: for g = f * p0 + p1,
    /// dg/dp0 == f(y) and dg/dp1 == 1, exactly, no FD tolerance needed.
    #[test]
    fn jacobian_data_wrt_params_matches_grafted_analytic(case in arb_smooth_tangent_case()) {
        let TangentCase { mut arena, root, y, n_states, .. } = case;
        prop_assume!(n_states > 0);
        let p0 = arena.alloc(Node::InputParameter {
            name: "p0".into(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let p1 = arena.alloc(Node::InputParameter {
            name: "p1".into(),
            index: 1,
            offset: 1,
            width: 1,
        });
        let scaled = arena.alloc(Node::Mul(root, p0));
        let grafted = arena.alloc(Node::Add(scaled, p1));

        let primal = CompiledExpr::new(&arena, grafted);
        let n_rows = primal.output_len();
        let jac = JacobianData::new_wrt_params(&arena, grafted, n_rows, 2);

        let p = [1.7, -0.3];
        let layout = jac.layout();
        let mut scratch = JacobianScratch::new(&jac);
        let mut data = vec![0.0; layout.n_slots()];
        jac.assemble_into(&mut scratch, layout, 0.5, &y, &[], &p, &mut data);

        // Reference: f(y) on the un-grafted tree.
        let base = CompiledExpr::new(&arena, root);
        let mut s = vec![0.0; base.scratch_len()];
        let f_val = base.eval(&mut s, 0.5, &y, &[], &[]).to_vec();

        for col in 0..2 {
            let (lo, hi) = (jac.csc().colptr[col], jac.csc().colptr[col + 1]);
            for (&dk, &row) in data[lo..hi].iter().zip(&jac.csc().rowind[lo..hi]) {
                let expected = if col == 0 { f_val[row] } else { 1.0 };
                prop_assert!((dk - expected).abs() <= 1e-10 * (1.0 + expected.abs()),
                    "dg/dp{} row {}: assembled {} vs analytic {}", col, row, dk, expected);
            }
        }
    }
}

// Targeted test: named deep smooth-composition and shape-stress cases

#[test]
fn targeted_smooth_tangent_cases_pass() {
    for case in targeted_smooth_tangent_cases() {
        check_ad_matches_fd(&case);
        check_optimized_matches_unoptimized(&case);
    }
}

/// AD must stay exact on a stiff double-exponential composition where the FD
/// oracle is invalid: f(y) = cos(sinh(sinh(2 - y))) + 2y has a diagonal Jacobian
/// entry of ~2.6e5, so an h=1e-5 central-difference probe sweeps the cosine
/// argument by ~2.6 radians. Checked against the analytic derivative instead.
#[test]
fn ad_exact_on_stiff_double_exponential() {
    use pybamm_core::{Arena, Node};

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
    let ir = TypedIr::from_arena_split_eval(&arena, tangent_root);
    let compiled = CompiledExpr::from_ir(ir);
    let mut s = vec![0.0; compiled.scratch_len()];
    let mut cache = compiled.eval_primal(&mut s, 0.0, &y, &[], &[]);

    // Analytic: df_i/dy_i = sin(sinh(sinh(u))) * cosh(sinh(u)) * cosh(u) + 2
    // with u = 2 - y_i; off-diagonal entries are exactly zero.
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

#[test]
fn reverse_targeted_instruction_families() {
    fn idx(a: &mut Arena, y: NodeId, i: usize) -> NodeId {
        a.alloc(Node::Index {
            child: y,
            start: i,
            end: i + 1,
        })
    }
    fn vec3(a: &mut Arena, y: NodeId) -> NodeId {
        a.alloc(Node::Index {
            child: y,
            start: 1,
            end: 4,
        })
    }
    // Sum a width-`w` vector to a scalar via a constant 1×w ones row (MatMul).
    fn sum_row(a: &mut Arena, v: NodeId, w: usize) -> NodeId {
        let ones = a.alloc(Node::SparseMatrix(Box::new(
            CsrData::try_new(
                vec![0, w],
                (0..w).collect(),
                vec![1.0; w],
                Shape::matrix(1, w),
            )
            .unwrap(),
        )));
        a.alloc(Node::MatMul(ones, v))
    }

    // y chosen so every domain is valid (log/sqrt need >0; min/max branches differ).
    type CaseBuilder = fn(&mut Arena, NodeId) -> NodeId;
    let y = [2.25_f64, 3.25, 4.25, 5.25];
    let cases: Vec<(&str, CaseBuilder)> = vec![
        // Binary scalar-scalar partials the smooth generator never emits.
        ("sub", |a, y| {
            let x0 = idx(a, y, 0);
            let x1 = idx(a, y, 1);
            a.alloc(Node::Sub(x0, x1))
        }),
        ("div", |a, y| {
            let x0 = idx(a, y, 0);
            let x1 = idx(a, y, 1);
            a.alloc(Node::Div(x0, x1))
        }),
        ("pow", |a, y| {
            let x0 = idx(a, y, 0);
            let x1 = idx(a, y, 1);
            a.alloc(Node::Pow(x0, x1))
        }),
        ("min", |a, y| {
            let x0 = idx(a, y, 0);
            let x1 = idx(a, y, 1);
            a.alloc(Node::Minimum(x0, x1))
        }),
        ("max", |a, y| {
            let x0 = idx(a, y, 0);
            let x1 = idx(a, y, 1);
            a.alloc(Node::Maximum(x0, x1))
        }),
        ("modulo", |a, y| {
            let x0 = idx(a, y, 0);
            let x1 = idx(a, y, 1);
            a.alloc(Node::Modulo(x0, x1))
        }),
        ("hypot", |a, y| {
            let x0 = idx(a, y, 0);
            let x1 = idx(a, y, 1);
            a.alloc(Node::Hypot(x0, x1))
        }),
        ("equal_heaviside", |a, y| {
            let x0 = idx(a, y, 0);
            let x1 = idx(a, y, 1);
            a.alloc(Node::EqualHeaviside(x0, x1))
        }),
        ("not_equal_heaviside", |a, y| {
            let x0 = idx(a, y, 0);
            let x1 = idx(a, y, 1);
            a.alloc(Node::NotEqualHeaviside(x0, x1))
        }),
        ("equality", |a, y| {
            let x0 = idx(a, y, 0);
            let x1 = idx(a, y, 1);
            a.alloc(Node::Equality(x0, x1))
        }),
        // All four broadcast kinds (scalar*scalar via `mul_ss`, then s*v, v*s, v*v summed).
        ("mul_ss", |a, y| {
            let x0 = idx(a, y, 0);
            let x1 = idx(a, y, 1);
            a.alloc(Node::Mul(x0, x1))
        }),
        ("mul_sv", |a, y| {
            let x = idx(a, y, 0);
            let v = vec3(a, y);
            let p = a.alloc(Node::Mul(x, v));
            sum_row(a, p, 3)
        }),
        ("mul_vs", |a, y| {
            let v = vec3(a, y);
            let x = idx(a, y, 0);
            let p = a.alloc(Node::Mul(v, x));
            sum_row(a, p, 3)
        }),
        ("mul_vv", |a, y| {
            let v = vec3(a, y);
            let p = a.alloc(Node::Mul(v, v));
            sum_row(a, p, 3)
        }),
        // Unary derivatives not exercised elsewhere (valid at y0=2).
        ("neg", |a, y| {
            let x = idx(a, y, 0);
            a.alloc(Node::Neg(x))
        }),
        ("abs", |a, y| {
            let x = idx(a, y, 0);
            a.alloc(Node::Abs(x))
        }),
        ("sqrt", |a, y| {
            let x = idx(a, y, 0);
            a.alloc(Node::Sqrt(x))
        }),
        ("log", |a, y| {
            let x = idx(a, y, 0);
            a.alloc(Node::Log(x))
        }),
        ("sign", |a, y| {
            let x = idx(a, y, 0);
            a.alloc(Node::Sign(x))
        }),
        ("floor", |a, y| {
            let x = idx(a, y, 0);
            a.alloc(Node::Floor(x))
        }),
        ("ceiling", |a, y| {
            let x = idx(a, y, 0);
            a.alloc(Node::Ceiling(x))
        }),
        // Reductions route the scalar bar to the argmax/argmin element.
        ("max_reduce", |a, y| {
            let v = vec3(a, y);
            a.alloc(Node::MaxReduce(v))
        }),
        ("min_reduce", |a, y| {
            let v = vec3(a, y);
            a.alloc(Node::MinReduce(v))
        }),
        // Concat then reduce (bar split across source ranges).
        ("concat", |a, y| {
            let x0 = idx(a, y, 0);
            let x1 = idx(a, y, 1);
            let c = a.alloc(Node::Concat(vec![x0, x1]));
            sum_row(a, c, 2)
        }),
        // Dense matmul: constant 1×3 row @ vec3 (transpose-scatter adjoint).
        ("dense_matmul", |a, y| {
            let v = vec3(a, y);
            let mat = a.alloc(Node::Array(Box::new(
                ArrayData::try_new(vec![2.0, -1.0, 0.5], Shape::matrix(1, 3)).unwrap(),
            )));
            a.alloc(Node::MatMul(mat, v))
        }),
        // 1-D linear interpolant of a scalar (interp'(x) · bar adjoint).
        ("interp_1d_linear", |a, y| {
            let x = idx(a, y, 0);
            let data = InterpolantData::try_new(vec![0.0, 2.5, 5.0], vec![1.0, 4.0, 9.0]).unwrap();
            a.alloc(Node::Interpolant1DLinear {
                data: Box::new(data),
                child: x,
            })
        }),
        ("interp_1d_cubic", |a, y| {
            let x = idx(a, y, 0);
            let data =
                CubicInterpolantData::try_new(vec![0.0, 5.0], vec![[1.0, 2.0, 3.0, 4.0]]).unwrap();
            a.alloc(Node::Interpolant1DCubic {
                data: Box::new(data),
                child: x,
            })
        }),
        ("interp_nd", |a, y| {
            let x0 = idx(a, y, 0);
            let x1 = idx(a, y, 1);
            let data = NdInterpolantData::try_new(
                vec![vec![0.0, 5.0], vec![0.0, 5.0]],
                vec![1.0, 4.0, 3.0, 2.0],
                2,
            )
            .unwrap();
            a.alloc(Node::InterpolantNd {
                data: Box::new(data),
                children: vec![x0, x1],
            })
        }),
        // Conditional routes bar to the selected branch (selector 1.0 → branch 0).
        ("conditional", |a, y| {
            let selector = a.alloc(Node::Scalar(1.0));
            let b0 = idx(a, y, 0);
            let b1 = idx(a, y, 1);
            a.alloc(Node::Conditional {
                selector,
                branches: vec![b0, b1],
            })
        }),
    ];

    for (label, build) in cases {
        let mut arena = Arena::new();
        let yv = arena.alloc(Node::StateVector { start: 0, end: 4 });
        let root = build(&mut arena, yv);
        assert_reverse_matches_forward_and_fd(label, &arena, root, &y);
    }
}

#[test]
fn reverse_derivative_interpolants_follow_first_order_policy() {
    let mut linear_arena = Arena::new();
    let linear_y = linear_arena.alloc(Node::StateVector { start: 0, end: 1 });
    let linear = linear_arena.alloc(Node::Interpolant1DLinearDeriv {
        slopes: vec![2.0, 3.0].into_boxed_slice(),
        x_data: vec![0.0, 2.0, 5.0].into_boxed_slice(),
        child: linear_y,
    });
    assert_reverse_and_forward_are_zero("linear derivative", &linear_arena, linear, &[1.0]);

    let mut cubic_arena = Arena::new();
    let cubic_y = cubic_arena.alloc(Node::StateVector { start: 0, end: 1 });
    let cubic_data =
        CubicInterpolantData::try_new(vec![0.0, 5.0], vec![[1.0, 2.0, 3.0, 4.0]]).unwrap();
    let cubic = cubic_arena.alloc(Node::Interpolant1DCubicDeriv {
        data: Box::new(cubic_data),
        child: cubic_y,
    });
    assert_reverse_and_forward_are_zero("cubic derivative", &cubic_arena, cubic, &[1.0]);

    let mut nd_arena = Arena::new();
    let nd_y0 = nd_arena.alloc(Node::StateVector { start: 0, end: 1 });
    let nd_y1 = nd_arena.alloc(Node::StateVector { start: 1, end: 2 });
    let nd_data = NdInterpolantData::try_new(
        vec![vec![0.0, 5.0], vec![0.0, 5.0]],
        vec![1.0, 4.0, 3.0, 2.0],
        2,
    )
    .unwrap();
    let nd = nd_arena.alloc(Node::InterpolantNdPartial {
        data: Box::new(nd_data),
        children: vec![nd_y0, nd_y1],
        axis: 0,
    });
    assert_reverse_and_forward_are_zero("ND partial", &nd_arena, nd, &[1.0, 2.0]);
}
