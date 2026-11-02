//! Differential test: `eval_batch` must match scalar `eval` bitwise (same float ops, same order).

mod common;

use common::cases::{DagCase, arb_eval_case, targeted_eval_cases};
use proptest::prelude::*;
use pybamm_core::ir::{BinaryOp, UnaryOp};
use pybamm_core::{
    Arena, ArrayData, BatchEvalError, CompiledExpr, CsrData, CubicInterpolantData, DagRef,
    InterpolantData, NdInterpolantData, Node, NodeId, Shape, tangent_wrt_states,
};

/// Assert `eval_batch` over `k` lanes equals `k` scalar `eval`s, bit for bit.
#[track_caller]
fn assert_batch_matches_scalar(
    arena: &Arena,
    root: NodeId,
    n_states: usize,
    k: usize,
    ts: &[f64],
    y_cols: &[f64],
    inputs: &[f64],
) {
    let expr = CompiledExpr::new(DagRef::new(arena, root));
    let out_len = expr.output_len();

    let mut scalar = vec![0.0_f64; out_len * k];
    let mut s = vec![0.0_f64; expr.scratch_len()];
    for l in 0..k {
        let y = &y_cols[l * n_states..(l + 1) * n_states];
        let res = expr.eval(&mut s, ts[l], y, &[], inputs);
        scalar[l * out_len..(l + 1) * out_len].copy_from_slice(res);
    }

    let mut batch_scratch = vec![0.0_f64; expr.scratch_len() * k];
    let root_slice = expr
        .eval_batch(&mut batch_scratch, k, ts, y_cols, n_states, inputs)
        .expect("primal tape must batch-evaluate");

    for l in 0..k {
        for e in 0..out_len {
            let got = root_slice[e * k + l];
            let want = scalar[l * out_len + e];
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "lane {l}, elem {e}: batch {got} != scalar {want} (k={k})"
            );
        }
    }
}

/// `(case, k, ts[k], y_cols[n_states*k])` with `k` sampled from {1, 2, 7, 32}.
fn arb_batch_case() -> impl Strategy<Value = (DagCase, usize, Vec<f64>, Vec<f64>)> {
    arb_eval_case()
        .prop_flat_map(|case| (Just(case), prop::sample::select(vec![1_usize, 2, 7, 32])))
        .prop_flat_map(|(case, k)| {
            let n_states = case.n_states;
            (
                Just(case),
                Just(k),
                prop::collection::vec(-2.0_f64..5.0, k),
                prop::collection::vec(0.1_f64..10.0, n_states * k),
            )
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    #[test]
    fn batch_matches_scalar_random_dag((case, k, ts, y_cols) in arb_batch_case()) {
        assert_batch_matches_scalar(&case.arena, case.root, case.n_states, k, &ts, &y_cols, &[]);
    }
}

#[test]
fn targeted_cases_batch_match_scalar() {
    for case in targeted_eval_cases() {
        let n = case.n_states;
        for &k in &[1_usize, 2, 7, 32] {
            // Stay near the case's own `y` so guarded ops (sqrt/div) stay in range.
            let mut y_cols = vec![0.0_f64; n * k];
            for l in 0..k {
                for i in 0..n {
                    y_cols[l * n + i] =
                        case.y[i].mul_add(0.03_f64.mul_add(l as f64, 1.0), 0.01 * i as f64);
                }
            }
            let ts: Vec<f64> = (0..k).map(|l| l as f64 * 0.1).collect();
            assert_batch_matches_scalar(&case.arena, case.root, n, k, &ts, &y_cols, &[]);
        }
    }
}

/// Build a column-major `(n_states, k)` state matrix from a closure.
fn state_cols(n_states: usize, k: usize, f: impl Fn(usize, usize) -> f64) -> Vec<f64> {
    let mut cols = vec![0.0_f64; n_states * k];
    for l in 0..k {
        for i in 0..n_states {
            cols[l * n_states + i] = f(i, l);
        }
    }
    cols
}

#[test]
fn all_binary_ops_match_scalar() {
    let ops = [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Pow,
        BinaryOp::Minimum,
        BinaryOp::Maximum,
        BinaryOp::Modulo,
        BinaryOp::Hypot,
        BinaryOp::EqualHeaviside,
        BinaryOp::NotEqualHeaviside,
        BinaryOp::Equality,
    ];
    let k = 7;
    let ts: Vec<f64> = (0..k).map(|l| l as f64 * 0.3).collect();
    let y_cols = state_cols(2, k, |i, l| 0.5 + i as f64 + l as f64 * 0.11);
    for op in ops {
        let mut arena = Arena::new();
        let a = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let b = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let root = alloc_binary(&mut arena, op, a, b);
        assert_batch_matches_scalar(&arena, root, 2, k, &ts, &y_cols, &[]);
    }
}

#[test]
fn all_unary_ops_match_scalar() {
    let ops = [
        UnaryOp::Neg,
        UnaryOp::Abs,
        UnaryOp::Sqrt,
        UnaryOp::Exp,
        UnaryOp::Log,
        UnaryOp::Sin,
        UnaryOp::Cos,
        UnaryOp::Tanh,
        UnaryOp::Sinh,
        UnaryOp::Cosh,
        UnaryOp::Arcsinh,
        UnaryOp::Arctan,
        UnaryOp::Erf,
        UnaryOp::Sign,
        UnaryOp::Floor,
        UnaryOp::Ceiling,
    ];
    let k = 5;
    let ts: Vec<f64> = (0..k).map(|l| l as f64 * 0.2).collect();
    // Positive so Sqrt/Log stay in-domain.
    let y_cols = state_cols(1, k, |_, l| 0.3 + l as f64 * 0.37);
    for op in ops {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let root = alloc_unary(&mut arena, op, x);
        assert_batch_matches_scalar(&arena, root, 1, k, &ts, &y_cols, &[]);
    }
}

#[test]
fn broadcast_scalar_vector_per_lane() {
    let mut arena = Arena::new();
    let s = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let v = arena.alloc(Node::StateVector { start: 1, end: 4 });
    let root = arena.alloc(Node::Mul(s, v));
    let k = 4;
    let ts = vec![0.0; k];
    let y_cols = state_cols(4, k, |i, l| 1.0 + i as f64 * 0.5 + l as f64);
    assert_batch_matches_scalar(&arena, root, 4, k, &ts, &y_cols, &[]);
}

#[test]
fn reduce_argselect_tie_breaking_per_lane() {
    // Lane 0: [5,5,3] (max tie at 0), lane 1: [1,5,5] (max tie at 1), lane 2: [3,1,1]
    // (min tie at 1), lane 3: [NaN,2,2] (NaN skipped, tie at 1).
    let picks = [
        [5.0, 5.0, 3.0],
        [1.0, 5.0, 5.0],
        [3.0, 1.0, 1.0],
        [f64::NAN, 2.0, 2.0],
    ];
    let k = picks.len();
    let y_cols = state_cols(
        6,
        k,
        |i, l| if i < 3 { picks[l][i] } else { 10.0 * i as f64 },
    );
    for is_max in [true, false] {
        let mut arena = Arena::new();
        let picker = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let basis = arena.alloc(Node::StateVector { start: 3, end: 6 });
        let root = arena.alloc(Node::ReduceArgSelect {
            basis,
            picker,
            is_max,
        });
        assert_batch_matches_scalar(&arena, root, 6, k, &vec![0.0; k], &y_cols, &[]);
    }
}

#[test]
fn max_and_min_reduce_match_scalar() {
    let mut arena = Arena::new();
    let v = arena.alloc(Node::StateVector { start: 0, end: 4 });
    let max = arena.alloc(Node::MaxReduce(v));
    let min = arena.alloc(Node::MinReduce(v));
    let root = arena.alloc(Node::Concat(vec![max, min]));
    let k = 5;
    let y_cols = state_cols(4, k, |i, l| match (i, l) {
        (0, 3) => f64::NAN,
        (2, 4) => f64::NEG_INFINITY,
        _ => ((i * 7 + l * 3) % 5) as f64 - 2.0,
    });
    assert_batch_matches_scalar(&arena, root, 4, k, &vec![0.0; k], &y_cols, &[]);
}

#[test]
fn dense_matmul_and_constant_loads_match_scalar() {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
    let matrix = arena.alloc(Node::Array(Box::new(
        ArrayData::try_new(vec![1.5, -2.0, 0.25, 3.0, 0.1, -0.7], Shape::matrix(2, 3))
            .expect("valid array"),
    )));
    let product = arena.alloc(Node::MatMul(matrix, y));
    let offset = arena.alloc(Node::Array(Box::new(
        ArrayData::try_new(vec![0.3, -0.9], Shape::vector(2)).expect("valid array"),
    )));
    let shifted = arena.alloc(Node::Add(product, offset));
    let zeros = arena.alloc(Node::ZeroVector { len: 2 });
    let root = arena.alloc(Node::Concat(vec![shifted, zeros, offset]));
    let k = 4;
    let y_cols = state_cols(3, k, |i, l| 0.2 + i as f64 * 0.7 - l as f64 * 0.45);
    assert_batch_matches_scalar(&arena, root, 3, k, &vec![0.0; k], &y_cols, &[]);
}

#[test]
fn interp_1d_variants_match_scalar() {
    let linear = InterpolantData::try_new(vec![0.0, 0.1, 0.3, 0.7], vec![0.2, -1.3, 2.9, 0.05])
        .expect("valid interpolant");
    let cubic = CubicInterpolantData::try_new(
        vec![0.0, 0.4, 1.0],
        vec![[1.0, -2.0, 0.5, 3.0], [0.4, 1.5, -1.0, 0.25]],
    )
    .expect("valid interpolant");
    let mut arena = Arena::new();
    let x = arena.alloc(Node::StateVector { start: 0, end: 3 });
    let value_linear = arena.alloc(Node::Interpolant1DLinear {
        data: Box::new(linear.clone()),
        child: x,
    });
    let slope = arena.alloc(Node::Interpolant1DLinearDeriv {
        data: Box::new(linear),
        child: x,
    });
    let value = arena.alloc(Node::Interpolant1DCubic {
        data: Box::new(cubic.clone()),
        child: x,
    });
    let derivative = arena.alloc(Node::Interpolant1DCubicDeriv {
        data: Box::new(cubic),
        child: x,
    });
    let root = arena.alloc(Node::Concat(vec![value_linear, slope, value, derivative]));
    // Below, inside and above both tables' domains.
    let k = 6;
    let y_cols = state_cols(3, k, |i, l| -0.3 + l as f64 * 0.27 + i as f64 * 0.1);
    assert_batch_matches_scalar(&arena, root, 3, k, &vec![0.0; k], &y_cols, &[]);
}

#[test]
fn sparse_matmul_matches_scalar() {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 4 });
    let mat = arena.alloc(Node::SparseMatrix(Box::new(
        CsrData::try_new(
            vec![0, 2, 4],
            vec![0, 1, 2, 3],
            vec![1.0, -1.0, 2.0, 3.0],
            Shape::matrix(2, 4),
        )
        .expect("valid matrix"),
    )));
    let root = arena.alloc(Node::MatMul(mat, y));
    let k = 6;
    let ts = vec![0.0; k];
    let y_cols = state_cols(4, k, |i, l| 0.5 + i as f64 + l as f64 * 0.13);
    assert_batch_matches_scalar(&arena, root, 4, k, &ts, &y_cols, &[]);
}

#[test]
fn nd_interpolant_and_partial_match_scalar() {
    // Axis 1's length-1 source broadcasts over axis 0's three-element one.
    let data = NdInterpolantData::try_new(
        vec![vec![0.0, 1.0, 3.0], vec![-1.0, 2.0]],
        vec![1.0, -2.0, 0.5, 3.0, -1.5, 0.25, 2.0, 4.0],
        2,
    )
    .expect("valid interpolant");
    let mut arena = Arena::new();
    let x0 = arena.alloc(Node::StateVector { start: 0, end: 3 });
    let x1 = arena.alloc(Node::StateVector { start: 3, end: 4 });
    let value = arena.alloc(Node::InterpolantNd {
        data: Box::new(data.clone()),
        children: vec![x0, x1],
    });
    let partial = arena.alloc(Node::InterpolantNdPartial {
        data: Box::new(data),
        children: vec![x0, x1],
        axis: 1,
    });
    let root = arena.alloc(Node::Concat(vec![value, partial]));
    let k = 5;
    let ts = vec![0.0; k];
    let y_cols = state_cols(4, k, |i, l| -0.5 + i as f64 * 0.9 + l as f64 * 0.45);
    assert_batch_matches_scalar(&arena, root, 4, k, &ts, &y_cols, &[]);
}

#[test]
fn time_and_concat_match_scalar() {
    let mut arena = Arena::new();
    let t = arena.alloc(Node::Time);
    let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let root = arena.alloc(Node::Concat(vec![t, y0]));
    let k = 4;
    let ts: Vec<f64> = (0..k).map(|l| 0.1 + l as f64 * 0.9).collect();
    let y_cols = state_cols(1, k, |_, l| l as f64 * 2.0 - 1.0);
    assert_batch_matches_scalar(&arena, root, 1, k, &ts, &y_cols, &[]);
}

#[test]
#[should_panic(expected = "eval_batch needs n_states = 2 values for each of 2 lanes")]
fn a_state_matrix_of_the_wrong_length_is_refused() {
    // Three values for two lanes used to stride both lanes over one state each.
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
    let expr = CompiledExpr::new(DagRef::new(&arena, y));
    let mut scratch = vec![0.0; expr.scratch_len() * 2];
    let _ = expr.eval_batch(&mut scratch, 2, &[0.0, 0.0], &[1.0, 2.0, 3.0], 2, &[]);
}

#[test]
fn tangent_load_returns_error() {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let two = arena.alloc(Node::Scalar(2.0));
    let root = arena.alloc(Node::Mul(y, two));
    let troot = tangent_wrt_states(&mut arena, root);
    let expr = CompiledExpr::new(DagRef::new(&arena, troot));
    let mut scratch = vec![0.0_f64; expr.scratch_len() * 2];
    let err = expr.eval_batch(&mut scratch, 2, &[0.0, 0.0], &[1.0, 2.0], 1, &[]);
    assert_eq!(err, Err(BatchEvalError::NonPrimalInstruction));
}

/// An `InputParameter` selector, so every lane agrees on the branch.
fn input_selector(arena: &mut Arena, name: &str, index: usize) -> NodeId {
    arena.alloc(Node::InputParameter {
        name: name.to_string(),
        index,
        offset: index,
        width: 1,
    })
}

#[test]
fn dispatch_skips_blocks_no_lane_selects() {
    // An `InputParameter` selector agrees across lanes, so only one block runs.
    let mut arena = Arena::new();
    let selector = input_selector(&mut arena, "s", 0);
    let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let b1 = arena.alloc(Node::Sin(y));
    let mut b2 = y;
    for _ in 0..6 {
        b2 = arena.alloc(Node::Exp(b2));
    }
    let root = arena.alloc(Node::Conditional {
        selector,
        branches: vec![b1, b2],
    });
    let k = 4;
    let ts = vec![0.0; k];
    let y_cols = state_cols(1, k, |_, l| 0.1 + l as f64 * 0.05);
    for sel in [1.0_f64, 2.0, 0.0] {
        assert_batch_matches_scalar(&arena, root, 1, k, &ts, &y_cols, &[sel]);
    }
}

#[test]
fn dispatch_skips_unneeded_block_even_when_it_would_error() {
    // Branch 2 would error if evaluated, so passing proves its block is skipped.
    let mut arena = Arena::new();
    let selector = input_selector(&mut arena, "s", 0);
    let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let b1 = arena.alloc(Node::Sin(y));
    let y_dot = arena.alloc(Node::StateVectorDot { start: 0, end: 1 });
    let b2 = arena.alloc(Node::Exp(y_dot));
    let root = arena.alloc(Node::Conditional {
        selector,
        branches: vec![b1, b2],
    });
    let expr = CompiledExpr::new(DagRef::new(&arena, root));
    let k = 4;
    let ts = vec![0.0; k];
    let y_cols = state_cols(1, k, |_, l| 0.1 + l as f64 * 0.05);
    let mut scratch = vec![0.0_f64; expr.scratch_len() * k];
    let result = expr.eval_batch(&mut scratch, k, &ts, &y_cols, 1, &[1.0]);
    assert!(
        result.is_ok(),
        "branch 2 must never run when no lane selects it: {result:?}"
    );
}

#[test]
fn two_dispatches_in_one_tape_match_scalar() {
    // The union-dispatch loop must resume after the first `Dispatch`.
    let mut arena = Arena::new();
    let sel1 = input_selector(&mut arena, "s1", 0);
    let sel2 = input_selector(&mut arena, "s2", 1);
    let y1 = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let y2 = arena.alloc(Node::StateVector { start: 1, end: 2 });
    let a1 = arena.alloc(Node::Sin(y1));
    let a2 = arena.alloc(Node::Cos(y1));
    let cond1 = arena.alloc(Node::Conditional {
        selector: sel1,
        branches: vec![a1, a2],
    });
    let b1 = arena.alloc(Node::Exp(y2));
    let b2 = arena.alloc(Node::Sqrt(y2));
    let cond2 = arena.alloc(Node::Conditional {
        selector: sel2,
        branches: vec![b1, b2],
    });
    let root = arena.alloc(Node::Add(cond1, cond2));
    let expr = CompiledExpr::new(DagRef::new(&arena, root));
    assert_eq!(
        expr.ir().branch_block_lens().len(),
        4,
        "two two-branch dispatches, or this test covers nothing"
    );
    let k = 4;
    let ts = vec![0.0; k];
    let y_cols = state_cols(2, k, |i, l| match i {
        0 => 0.2 + l as f64 * 0.1,
        _ => 0.5 + l as f64 * 0.2,
    });
    for sels in [[1.0, 1.0], [2.0, 2.0], [1.0, 2.0], [0.0, 1.0]] {
        assert_batch_matches_scalar(&arena, root, 2, k, &ts, &y_cols, &sels);
    }
}

#[test]
fn dispatch_on_boundary_and_nan_selectors_matches_scalar() {
    // Half-integer boundaries, NaN, negative and infinite selectors match no branch.
    let mut arena = Arena::new();
    let selector = input_selector(&mut arena, "s", 0);
    let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let b1 = arena.alloc(Node::Sin(y));
    let b2 = arena.alloc(Node::Cos(y));
    let root = arena.alloc(Node::Conditional {
        selector,
        branches: vec![b1, b2],
    });
    let k = 4;
    let ts = vec![0.0; k];
    let y_cols = state_cols(1, k, |_, l| 0.1 + l as f64 * 0.05);
    for sel in [0.5_f64, 1.5, 2.5, -1.0, f64::NAN, f64::INFINITY] {
        assert_batch_matches_scalar(&arena, root, 1, k, &ts, &y_cols, &[sel]);
    }

    let lane_selector = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
    let lb1 = arena.alloc(Node::Sin(y1));
    let lb2 = arena.alloc(Node::Cos(y1));
    let lane_root = arena.alloc(Node::Conditional {
        selector: lane_selector,
        branches: vec![lb1, lb2],
    });
    let weird = [f64::NAN, 0.5, 1.5, 2.5, -1.0, f64::INFINITY, 1.0, 2.0];
    let kw = weird.len();
    let y_cols = state_cols(2, kw, |i, l| if i == 0 { weird[l] } else { 0.3 });
    assert_batch_matches_scalar(&arena, lane_root, 2, kw, &vec![0.0; kw], &y_cols, &[]);
}

fn alloc_binary(arena: &mut Arena, op: BinaryOp, a: NodeId, b: NodeId) -> NodeId {
    arena.alloc(match op {
        BinaryOp::Add => Node::Add(a, b),
        BinaryOp::Sub => Node::Sub(a, b),
        BinaryOp::Mul => Node::Mul(a, b),
        BinaryOp::Div => Node::Div(a, b),
        BinaryOp::Pow => Node::Pow(a, b),
        BinaryOp::Minimum => Node::Minimum(a, b),
        BinaryOp::Maximum => Node::Maximum(a, b),
        BinaryOp::Modulo => Node::Modulo(a, b),
        BinaryOp::Hypot => Node::Hypot(a, b),
        BinaryOp::EqualHeaviside => Node::EqualHeaviside(a, b),
        BinaryOp::NotEqualHeaviside => Node::NotEqualHeaviside(a, b),
        BinaryOp::Equality => Node::Equality(a, b),
    })
}

fn alloc_unary(arena: &mut Arena, op: UnaryOp, x: NodeId) -> NodeId {
    arena.alloc(match op {
        UnaryOp::Neg => Node::Neg(x),
        UnaryOp::Abs => Node::Abs(x),
        UnaryOp::Sqrt => Node::Sqrt(x),
        UnaryOp::Exp => Node::Exp(x),
        UnaryOp::Log => Node::Log(x),
        UnaryOp::Sin => Node::Sin(x),
        UnaryOp::Cos => Node::Cos(x),
        UnaryOp::Tanh => Node::Tanh(x),
        UnaryOp::Sinh => Node::Sinh(x),
        UnaryOp::Cosh => Node::Cosh(x),
        UnaryOp::Arcsinh => Node::Arcsinh(x),
        UnaryOp::Arctan => Node::Arctan(x),
        UnaryOp::Erf => Node::Erf(x),
        UnaryOp::Sign => Node::Sign(x),
        UnaryOp::Floor => Node::Floor(x),
        UnaryOp::Ceiling => Node::Ceiling(x),
    })
}
