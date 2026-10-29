mod common;

use common::cases::{
    ALL_EXTRAS, DagCase, TangentCase, arb_eval_case_with, arb_smooth_tangent_case,
    arb_smooth_tangent_case_with, eval_dag, targeted_smooth_tangent_cases,
};
use common::numeric_eq::assert_close;
use proptest::prelude::*;
use pybamm_core::{
    Arena, ArrayData, CompiledExpr, CsrData, CubicInterpolantData, DagRef, InterpolantData,
    JacobianData, JacobianScratch, NdInterpolantData, Node, NodeId, Shape, SplitCompiledExpr,
    TangentInputs, adjoint::AdjointTape, cse, dce, simplify, tangent_wrt_states, zero_propagate,
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

/// The gradient of a scalar `root` from one reverse pass.
fn reverse_gradient(arena: &Arena, root: NodeId, y: &[f64], inputs: &[f64]) -> Vec<f64> {
    let tape = AdjointTape::new(DagRef::new(arena, root), y.len());
    let mut scratch = vec![0.0; tape.scratch_len()];
    let mut bar = vec![0.0; tape.scratch_len()];
    let mut grad = vec![0.0; y.len()];
    tape.assemble(&mut scratch, &mut bar, &mut grad, 0.5, y, &[], inputs);
    grad
}

/// Reverse gradient of a scalar row vs the forward-JVP row (tight) and central FD (loose).
fn assert_reverse_matches_forward_and_fd(
    ctx: &str,
    arena: &Arena,
    root: NodeId,
    y: &[f64],
    inputs: &[f64],
) {
    let n = y.len();
    let grad = reverse_gradient(arena, root, y, inputs);

    let mut ad_arena = arena.clone();
    let tangent_root = tangent_wrt_states(&mut ad_arena, root);
    let compiled = SplitCompiledExpr::compile(DagRef::new(&ad_arena, tangent_root));
    let mut s = vec![0.0; compiled.scratch_len()];
    let mut seed = vec![0.0; n];
    let primal = CompiledExpr::new(DagRef::new(arena, root));
    let mut ps = vec![0.0; primal.scratch_len()];
    let eps = 1e-6;

    for j in 0..n {
        seed.fill(0.0);
        seed[j] = 1.0;
        let mut cache = compiled.eval_primal(&mut s, 0.5, y, &[], inputs);
        let jvp = cache.eval_tangent(&TangentInputs {
            dy: Some(&seed),
            dp: None,
        })[0];
        assert!(
            (grad[j] - jvp).abs() <= 1e-12f64.mul_add(jvp.abs(), 1e-14),
            "{ctx} col {j}: reverse {} vs forward-JVP {jvp}",
            grad[j]
        );

        let mut yp = y.to_vec();
        let mut ym = y.to_vec();
        yp[j] += eps;
        ym[j] -= eps;
        let fp = primal.eval(&mut ps, 0.5, &yp, &[], inputs)[0];
        let fm = primal.eval(&mut ps, 0.5, &ym, &[], inputs)[0];
        let fd = (fp - fm) / (2.0 * eps);
        assert!(
            (grad[j] - fd).abs() <= 1e-4 * (1.0 + fd.abs()),
            "{ctx} col {j}: reverse {} vs fd {fd}",
            grad[j]
        );
    }
}

/// Derivative-interpolant nodes are first-order terminals: both AD directions return zero.
fn assert_reverse_and_forward_are_zero(ctx: &str, arena: &Arena, root: NodeId, y: &[f64]) {
    let n = y.len();
    let grad = reverse_gradient(arena, root, y, &[]);

    let mut ad_arena = arena.clone();
    let tangent_root = tangent_wrt_states(&mut ad_arena, root);
    let compiled = SplitCompiledExpr::compile(DagRef::new(&ad_arena, tangent_root));
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

/// The assembled state Jacobian, zeros off the pattern included, matches central FD.
fn check_jacobian_matches_fd(case: &TangentCase) {
    let TangentCase {
        arena,
        root,
        y,
        n_states,
        inputs,
        ..
    } = case;
    let primal = CompiledExpr::new(DagRef::new(arena, *root));
    let n_rows = primal.output_len();
    let jac = JacobianData::new_wrt_states(arena, *root, n_rows, *n_states);

    let layout = jac.layout();
    let mut scratch = JacobianScratch::new(&jac);
    let mut data = vec![0.0; layout.n_slots()];
    jac.assemble_into(&mut scratch, layout, 0.5, y, &[], inputs, &mut data);

    let mut s = vec![0.0; primal.scratch_len()];
    let eps = 1e-6;
    for col in 0..jac.n_cols() {
        let mut assembled = vec![0.0; n_rows];
        for k in jac.csc().colptr()[col]..jac.csc().colptr()[col + 1] {
            assembled[jac.csc().rowind()[k]] = data[k];
        }
        let mut yp = y.clone();
        yp[col] += eps;
        let mut ym = y.clone();
        ym[col] -= eps;
        let fp = primal.eval(&mut s, 0.5, &yp, &[], inputs).to_vec();
        let fm = primal.eval(&mut s, 0.5, &ym, &[], inputs).to_vec();
        for (row, &dk) in assembled.iter().enumerate() {
            let fd = (fp[row] - fm[row]) / (2.0 * eps);
            assert!(
                (dk - fd).abs() <= 1e-4 * (1.0 + fd.abs()),
                "entry ({row},{col}): assembled {dk} vs fd {fd}"
            );
        }
    }
}

/// Each output row's reverse gradient matches its forward sweeps and central FD.
fn check_reverse_rows(case: TangentCase) {
    let TangentCase {
        mut arena,
        root,
        y,
        inputs,
        ..
    } = case;
    let n_rows = CompiledExpr::new(DagRef::new(&arena, root)).output_len();
    for r in 0..n_rows {
        let row = arena.alloc(Node::Index {
            child: root,
            start: r,
            end: r + 1,
        });
        assert_reverse_matches_forward_and_fd(&format!("row {r}"), &arena, row, &y, &inputs);
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

    #[test]
    fn jacobian_data_wrt_states_matches_fd(case in arb_smooth_tangent_case()) {
        check_jacobian_matches_fd(&case);
    }

    #[test]
    fn reverse_row_gradient_matches_forward_and_fd(case in arb_smooth_tangent_case()) {
        check_reverse_rows(case);
    }

    /// Every [`Extra`](common::cases::Extra) the smooth mode admits.
    #[test]
    fn extended_ops_match_finite_differences(case in arb_smooth_tangent_case_with(ALL_EXTRAS)) {
        check_ad_matches_fd(&case);
        check_optimized_matches_unoptimized(&case);
        check_jacobian_matches_fd(&case);
        check_reverse_rows(case);
    }

    /// Steps, `Modulo` and comparisons too, which no finite difference can judge.
    #[test]
    fn forward_and_reverse_rows_agree_with_every_extra(case in arb_eval_case_with(ALL_EXTRAS)) {
        let DagCase { mut arena, root, y, inputs, .. } = case;
        let mut ad_arena = arena.clone();
        let tangent_root = tangent_wrt_states(&mut ad_arena, root);
        let tangent = SplitCompiledExpr::compile(DagRef::new(&ad_arena, tangent_root));
        let mut scratch = vec![0.0; tangent.scratch_len()];
        let unseeded = TangentInputs { dy: Some(&vec![0.0; y.len()]), dp: None };
        let baseline = tangent.eval_with_tangent(&mut scratch, 0.5, &y, &[], &inputs, &unseeded);
        for (r, value) in baseline.iter().enumerate() {
            // A NaN no seed is needed for, say `Hypot` of two zero literals, is no derivative.
            if value.is_nan() {
                continue;
            }
            let row = arena.alloc(Node::Index {
                child: root,
                start: r,
                end: r + 1,
            });
            assert_modes_agree(&format!("row {r}"), &arena, row, &y, &inputs);
        }
    }

    /// For g = f * p0 + p1, dg/dp0 == f(y) and dg/dp1 == 1 exactly.
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

        let primal = CompiledExpr::new(DagRef::new(&arena, grafted));
        let n_rows = primal.output_len();
        let jac = JacobianData::new_wrt_params(&arena, grafted, n_rows, 2);

        let p = [1.7, -0.3];
        let layout = jac.layout();
        let mut scratch = JacobianScratch::new(&jac);
        let mut data = vec![0.0; layout.n_slots()];
        jac.assemble_into(&mut scratch, layout, 0.5, &y, &[], &p, &mut data);

        let base = CompiledExpr::new(DagRef::new(&arena, root));
        let mut s = vec![0.0; base.scratch_len()];
        let f_val = base.eval(&mut s, 0.5, &y, &[], &[]).to_vec();

        for col in 0..2 {
            let (lo, hi) = (jac.csc().colptr()[col], jac.csc().colptr()[col + 1]);
            for (&dk, &row) in data[lo..hi].iter().zip(&jac.csc().rowind()[lo..hi]) {
                let expected = if col == 0 { f_val[row] } else { 1.0 };
                prop_assert!((dk - expected).abs() <= 1e-10 * (1.0 + expected.abs()),
                    "dg/dp{} row {}: assembled {} vs analytic {}", col, row, dk, expected);
            }
        }
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
        ("max_reduce", |a, y| {
            let v = vec3(a, y);
            a.alloc(Node::MaxReduce(v))
        }),
        ("min_reduce", |a, y| {
            let v = vec3(a, y);
            a.alloc(Node::MinReduce(v))
        }),
        ("concat", |a, y| {
            let x0 = idx(a, y, 0);
            let x1 = idx(a, y, 1);
            let c = a.alloc(Node::Concat(vec![x0, x1]));
            sum_row(a, c, 2)
        }),
        ("dense_matmul", |a, y| {
            let v = vec3(a, y);
            let mat = a.alloc(Node::Array(Box::new(
                ArrayData::try_new(vec![2.0, -1.0, 0.5], Shape::matrix(1, 3)).unwrap(),
            )));
            a.alloc(Node::MatMul(mat, v))
        }),
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
        assert_reverse_matches_forward_and_fd(label, &arena, root, &y, &[]);
    }
}

/// Forward sweeps of a scalar `root` against its reverse gradient: NaN for NaN, finite values
/// close, and an infinity only against a value past `sqrt(f64::MAX)` of the same sign.
///
/// Skips a non-finite primal, where no derivative is meant, and a column whose sweep
/// multiplies another column's non-finite partial by a zero seed.
fn assert_modes_agree(label: &str, arena: &Arena, root: NodeId, y: &[f64], inputs: &[f64]) {
    if !eval_dag(arena, root, 0.5, y, inputs)[0].is_finite() {
        return;
    }
    let reverse = reverse_gradient(arena, root, y, inputs);
    let mut ad_arena = arena.clone();
    let tangent_root = tangent_wrt_states(&mut ad_arena, root);
    let compiled = SplitCompiledExpr::compile(DagRef::new(&ad_arena, tangent_root));
    let mut scratch = vec![0.0; compiled.scratch_len()];
    for (col, &want) in reverse.iter().enumerate() {
        if (0..y.len()).any(|other| other != col && !reverse[other].is_finite()) {
            continue;
        }
        let mut seed = vec![0.0; y.len()];
        seed[col] = 1.0;
        let tangent = TangentInputs {
            dy: Some(&seed),
            dp: None,
        };
        let got = compiled.eval_with_tangent(&mut scratch, 0.5, y, &[], inputs, &tangent)[0];
        let agree = if got.is_nan() || want.is_nan() {
            got.is_nan() && want.is_nan()
        } else if got.is_finite() && want.is_finite() {
            (got - want).abs() <= 1e-12 * (1.0 + want.abs())
        } else if got.is_finite() || want.is_finite() {
            // One mode overflows a product the other never forms, like `Hypot`'s `b * db`.
            got.signum() == want.signum() && got.abs().min(want.abs()) >= f64::MAX.sqrt()
        } else {
            got.to_bits() == want.to_bits()
        };
        assert!(
            agree,
            "{label} at y = {y:?}: column {col} forward {got} vs reverse {want}"
        );
    }
}

/// Every elementwise op at zeros, a negative value and ties, in both AD modes.
#[test]
fn forward_and_reverse_agree_at_edge_values() {
    const EDGES: [f64; 3] = [0.0, -1.0, 2.0];
    type Unary = fn(NodeId) -> Node;
    type Binary = fn(NodeId, NodeId) -> Node;
    let unary: [(&str, Unary); 16] = [
        ("neg", Node::Neg),
        ("abs", Node::Abs),
        ("sqrt", Node::Sqrt),
        ("exp", Node::Exp),
        ("log", Node::Log),
        ("sin", Node::Sin),
        ("cos", Node::Cos),
        ("tanh", Node::Tanh),
        ("sinh", Node::Sinh),
        ("cosh", Node::Cosh),
        ("arcsinh", Node::Arcsinh),
        ("arctan", Node::Arctan),
        ("erf", Node::Erf),
        ("sign", Node::Sign),
        ("floor", Node::Floor),
        ("ceiling", Node::Ceiling),
    ];
    let binary: [(&str, Binary); 12] = [
        ("add", Node::Add),
        ("sub", Node::Sub),
        ("mul", Node::Mul),
        ("div", Node::Div),
        ("pow", Node::Pow),
        ("min", Node::Minimum),
        ("max", Node::Maximum),
        ("modulo", Node::Modulo),
        ("hypot", Node::Hypot),
        ("equal_heaviside", Node::EqualHeaviside),
        ("not_equal_heaviside", Node::NotEqualHeaviside),
        ("equality", Node::Equality),
    ];
    let state = |arena: &mut Arena, i: usize| {
        arena.alloc(Node::StateVector {
            start: i,
            end: i + 1,
        })
    };
    let input = |arena: &mut Arena| {
        arena.alloc(Node::InputParameter {
            name: "p".into(),
            index: 0,
            offset: 0,
            width: 1,
        })
    };

    for (label, op) in unary {
        for x in EDGES {
            let mut arena = Arena::new();
            let y0 = state(&mut arena, 0);
            let root = arena.alloc(op(y0));
            assert_modes_agree(label, &arena, root, &[x], &[]);
        }
    }
    // `a / b` rounds up to an integer the exact quotient lies below, so it truncates wrongly.
    let rounded_quotients = [[1.0, 0.1], [2.0, 0.2]];
    for (label, op) in binary {
        let edges = EDGES.into_iter().flat_map(|a| EDGES.map(|b| [a, b]));
        for point in edges.chain(rounded_quotients) {
            let mut arena = Arena::new();
            let (y0, y1) = (state(&mut arena, 0), state(&mut arena, 1));
            let root = arena.alloc(op(y0, y1));
            assert_modes_agree(label, &arena, root, &point, &[]);
        }
    }
    // Constant exponents, literal and input.
    for exponent in [0.0, 0.5, -1.0, 2.0] {
        for x in EDGES {
            let mut arena = Arena::new();
            let y0 = state(&mut arena, 0);
            let literal = arena.alloc(Node::Scalar(exponent));
            let root = arena.alloc(Node::Pow(y0, literal));
            assert_modes_agree("pow literal", &arena, root, &[x], &[]);

            let mut arena = Arena::new();
            let y0 = state(&mut arena, 0);
            let p = input(&mut arena);
            let root = arena.alloc(Node::Pow(y0, p));
            assert_modes_agree("pow input", &arena, root, &[x], &[exponent]);
        }
    }
    // An input base; at 0 its own infinite partial meets a zero tangent in the raw sweep.
    for base in [-1.0, 0.5, 2.0] {
        for x in EDGES {
            let mut arena = Arena::new();
            let y0 = state(&mut arena, 0);
            let p = input(&mut arena);
            let root = arena.alloc(Node::Pow(p, y0));
            assert_modes_agree("input pow", &arena, root, &[x], &[base]);
        }
    }
}

/// A knot gap below `f64::EPSILON` must not floor the slope to zero in either AD direction.
#[test]
fn reverse_matches_forward_over_a_sub_ulp_interpolant_segment() {
    let y = [5e-18];
    let mut arena = Arena::new();
    let state = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let data = InterpolantData::try_new(vec![0.0, 1e-17, 1.0], vec![0.0, 1.0, 2.0])
        .expect("valid interpolant");
    let root = arena.alloc(Node::Interpolant1DLinear {
        data: Box::new(data),
        child: state,
    });

    let tape = AdjointTape::new(DagRef::new(&arena, root), 1);
    let mut scratch = vec![0.0; tape.scratch_len()];
    let mut bar = vec![0.0; tape.scratch_len()];
    let mut grad = [0.0];
    tape.assemble(&mut scratch, &mut bar, &mut grad, 0.5, &y, &[], &[]);

    let mut ad_arena = arena.clone();
    let tangent_root = tangent_wrt_states(&mut ad_arena, root);
    let compiled = SplitCompiledExpr::compile(DagRef::new(&ad_arena, tangent_root));
    let mut ad_scratch = vec![0.0; compiled.scratch_len()];
    let jvp = compiled.eval_with_tangent(
        &mut ad_scratch,
        0.5,
        &y,
        &[],
        &[],
        &TangentInputs {
            dy: Some(&[1.0]),
            dp: None,
        },
    )[0];

    assert_eq!(grad[0].to_bits(), 1e17_f64.to_bits());
    assert_eq!(grad[0].to_bits(), jvp.to_bits());
}

#[test]
fn reverse_derivative_interpolants_follow_first_order_policy() {
    let mut linear_arena = Arena::new();
    let linear_y = linear_arena.alloc(Node::StateVector { start: 0, end: 1 });
    let table = InterpolantData::try_new(vec![0.0, 2.0, 5.0], vec![0.0, 4.0, 13.0])
        .expect("valid interpolant");
    let linear = linear_arena.alloc(Node::Interpolant1DLinearDeriv {
        data: Box::new(table),
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
