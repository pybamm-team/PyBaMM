use proptest::prelude::*;
use pybamm_core::{Arena, CompiledExpr, Node, NodeId, TypedIr};

// Case structs

#[derive(Clone, Debug)]
pub struct DagCase {
    pub arena: Arena,
    pub root: NodeId,
    pub y: Vec<f64>,
    pub n_states: usize,
}

#[derive(Clone, Debug)]
// shared scaffolding: not every integration-test binary reads every field
#[allow(dead_code)]
pub struct TangentCase {
    pub arena: Arena,
    pub root: NodeId,
    pub y: Vec<f64>,
    pub seeds: Vec<Vec<f64>>,
    pub n_states: usize,
}

#[allow(dead_code)]
pub fn eval_dag(arena: &Arena, root: NodeId, t: f64, y: &[f64], inputs: &[f64]) -> Vec<f64> {
    let ir = TypedIr::from_arena(arena, root);
    let expr = CompiledExpr::from_ir(ir);
    let mut s = vec![0.0; expr.scratch_len()];
    expr.eval(&mut s, t, y, &[], inputs).to_vec()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
// shared scaffolding: not every integration-test binary uses every class
#[allow(dead_code)]
enum ValueClass {
    AnyFinite,
    PositiveFinite,
    NonZeroFinite,
    BoundedSmall,
    SmoothSafe,
    SelectorScalar,
}

#[derive(Clone, Copy, Debug)]
enum UnarySpec {
    Neg,
    Abs,
    Sqrt,
    Exp,
    Sin,
    Cos,
    Tanh,
    Sinh,
    Cosh,
    Arcsinh,
    Arctan,
    Erf,
}

#[derive(Clone, Copy, Debug)]
enum BinarySpec {
    Add,
    Sub,
    Mul,
    Div,
    EqualHeaviside,
}

#[derive(Clone, Debug)]
// shared scaffolding: not every integration-test binary builds every variant
#[allow(dead_code)]
enum ExprSpec {
    Scalar(f64, ValueClass),
    StateSlice {
        start: usize,
        end: usize,
        class: ValueClass,
    },
    Unary {
        op: UnarySpec,
        child: Box<Self>,
        len: usize,
        class: ValueClass,
    },
    Binary {
        op: BinarySpec,
        lhs: Box<Self>,
        rhs: Box<Self>,
        len: usize,
        class: ValueClass,
    },
    Concat(Vec<Self>, usize),
    Index {
        child: Box<Self>,
        start: usize,
        end: usize,
        class: ValueClass,
    },
    Conditional {
        selector: Box<Self>,
        branches: Vec<Self>,
        len: usize,
        class: ValueClass,
    },
    Interpolant1DLinear {
        x_data: Vec<f64>,
        y_data: Vec<f64>,
        child: Box<Self>,
        len: usize,
        class: ValueClass,
    },
    SparseMatMul {
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<f64>,
        nrows: usize,
        ncols: usize,
        rhs: Box<Self>,
        class: ValueClass,
    },
}

#[derive(Clone, Debug)]
struct TypedEntry {
    spec: ExprSpec,
    len: usize,
    class: ValueClass,
}

const fn can_broadcast(a_len: usize, b_len: usize) -> bool {
    a_len == b_len || a_len == 1 || b_len == 1
}

fn lower_expr(arena: &mut Arena, spec: &ExprSpec) -> NodeId {
    match spec {
        ExprSpec::Scalar(v, _) => arena.alloc(Node::Scalar(*v)),
        ExprSpec::StateSlice { start, end, .. } => arena.alloc(Node::StateVector {
            start: *start,
            end: *end,
        }),
        ExprSpec::Unary { op, child, .. } => {
            let child = lower_expr(arena, child);
            match op {
                UnarySpec::Neg => arena.alloc(Node::Neg(child)),
                UnarySpec::Abs => arena.alloc(Node::Abs(child)),
                UnarySpec::Sqrt => arena.alloc(Node::Sqrt(child)),
                UnarySpec::Exp => arena.alloc(Node::Exp(child)),
                UnarySpec::Sin => arena.alloc(Node::Sin(child)),
                UnarySpec::Cos => arena.alloc(Node::Cos(child)),
                UnarySpec::Tanh => arena.alloc(Node::Tanh(child)),
                UnarySpec::Sinh => arena.alloc(Node::Sinh(child)),
                UnarySpec::Cosh => arena.alloc(Node::Cosh(child)),
                UnarySpec::Arcsinh => arena.alloc(Node::Arcsinh(child)),
                UnarySpec::Arctan => arena.alloc(Node::Arctan(child)),
                UnarySpec::Erf => arena.alloc(Node::Erf(child)),
            }
        },
        ExprSpec::Binary { op, lhs, rhs, .. } => {
            let lhs = lower_expr(arena, lhs);
            let rhs = lower_expr(arena, rhs);
            match op {
                BinarySpec::Add => arena.alloc(Node::Add(lhs, rhs)),
                BinarySpec::Sub => arena.alloc(Node::Sub(lhs, rhs)),
                BinarySpec::Mul => arena.alloc(Node::Mul(lhs, rhs)),
                BinarySpec::Div => arena.alloc(Node::Div(lhs, rhs)),
                BinarySpec::EqualHeaviside => arena.alloc(Node::EqualHeaviside(lhs, rhs)),
            }
        },
        ExprSpec::Concat(children, _) => {
            let children: Vec<NodeId> = children.iter().map(|c| lower_expr(arena, c)).collect();
            arena.alloc(Node::Concat(children))
        },
        ExprSpec::Index {
            child, start, end, ..
        } => {
            let child = lower_expr(arena, child);
            arena.alloc(Node::Index {
                child,
                start: *start,
                end: *end,
            })
        },
        ExprSpec::Conditional {
            selector, branches, ..
        } => {
            let selector = lower_expr(arena, selector);
            let branches: Vec<NodeId> = branches.iter().map(|b| lower_expr(arena, b)).collect();
            arena.alloc(Node::Conditional { selector, branches })
        },
        ExprSpec::Interpolant1DLinear {
            x_data,
            y_data,
            child,
            ..
        } => {
            let child = lower_expr(arena, child);
            arena.alloc(Node::Interpolant1DLinear {
                data: Box::new(
                    pybamm_core::InterpolantData::try_new(x_data.clone(), y_data.clone())
                        .expect("valid interpolant"),
                ),
                child,
            })
        },
        ExprSpec::SparseMatMul {
            indptr,
            indices,
            data,
            nrows,
            ncols,
            rhs,
            ..
        } => {
            let mat = arena.alloc(Node::SparseMatrix(Box::new(
                pybamm_core::CsrData::try_new(
                    indptr.clone(),
                    indices.clone(),
                    data.clone(),
                    pybamm_core::Shape::matrix(*nrows, *ncols),
                )
                .expect("valid test matrix"),
            )));
            let rhs = lower_expr(arena, rhs);
            arena.alloc(Node::MatMul(mat, rhs))
        },
    }
}

#[allow(clippy::cast_possible_truncation)]
fn seed_set_from_entropy(n_states: usize, entropy: &[u8]) -> Vec<Vec<f64>> {
    let mut seeds: Vec<Vec<f64>> = (0..n_states)
        .map(|j| {
            let mut v = vec![0.0; n_states];
            v[j] = 1.0;
            v
        })
        .collect();

    seeds.push(vec![1.0; n_states]);
    seeds.push(
        (0..n_states)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect(),
    );

    let mut dense = vec![0.0; n_states];
    for (i, slot) in dense.iter_mut().enumerate() {
        let b = entropy
            .get(i)
            .copied()
            .unwrap_or_else(|| (i as u8).wrapping_mul(17));
        *slot = (f64::from(b) / 127.5) - 1.0;
    }
    let max_abs = dense
        .iter()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()))
        .max(1e-12);
    for slot in &mut dense {
        *slot /= max_abs;
    }
    seeds.push(dense);
    seeds
}

const fn unary_input_class(op: UnarySpec, mode: GenMode) -> ValueClass {
    match op {
        UnarySpec::Exp => ValueClass::BoundedSmall,
        // Sinh/Cosh grow like Exp, so unbounded inputs push derivatives past the
        // FD oracle's validity: sinh(sinh(x)) sweeps the probe radians of a cos.
        UnarySpec::Sinh | UnarySpec::Cosh => match mode {
            GenMode::Smooth => ValueClass::BoundedSmall,
            GenMode::Full | GenMode::TangentSafe => ValueClass::AnyFinite,
        },
        UnarySpec::Sqrt => ValueClass::PositiveFinite,
        UnarySpec::Arcsinh
        | UnarySpec::Arctan
        | UnarySpec::Erf
        | UnarySpec::Sin
        | UnarySpec::Cos
        | UnarySpec::Tanh
        | UnarySpec::Neg
        | UnarySpec::Abs => ValueClass::AnyFinite,
    }
}

const fn unary_output_class(op: UnarySpec, child_class: ValueClass) -> ValueClass {
    match op {
        UnarySpec::Exp | UnarySpec::Sqrt | UnarySpec::Abs => ValueClass::PositiveFinite,
        UnarySpec::Sin | UnarySpec::Cos | UnarySpec::Tanh | UnarySpec::Erf | UnarySpec::Arctan => {
            ValueClass::BoundedSmall
        },
        UnarySpec::Neg => match child_class {
            ValueClass::BoundedSmall => ValueClass::BoundedSmall,
            ValueClass::NonZeroFinite => ValueClass::NonZeroFinite,
            _ => ValueClass::AnyFinite,
        },
        UnarySpec::Sinh | UnarySpec::Cosh | UnarySpec::Arcsinh => ValueClass::AnyFinite,
    }
}

const fn binary_output_class(op: BinarySpec, a: ValueClass, b: ValueClass) -> ValueClass {
    match op {
        BinarySpec::Add | BinarySpec::Sub => {
            if matches!(a, ValueClass::BoundedSmall) && matches!(b, ValueClass::BoundedSmall) {
                ValueClass::BoundedSmall
            } else {
                ValueClass::AnyFinite
            }
        },
        BinarySpec::Div => {
            if matches!(a, ValueClass::PositiveFinite) && matches!(b, ValueClass::PositiveFinite) {
                ValueClass::PositiveFinite
            } else {
                ValueClass::AnyFinite
            }
        },
        BinarySpec::Mul | BinarySpec::EqualHeaviside => ValueClass::AnyFinite,
    }
}

const fn class_satisfies(actual: ValueClass, required: ValueClass) -> bool {
    use ValueClass::*;
    match required {
        AnyFinite => true,
        BoundedSmall => matches!(actual, BoundedSmall | SelectorScalar),
        PositiveFinite => matches!(actual, PositiveFinite),
        NonZeroFinite => matches!(actual, NonZeroFinite | PositiveFinite),
        SmoothSafe => matches!(actual, SmoothSafe | BoundedSmall | PositiveFinite),
        SelectorScalar => matches!(actual, SelectorScalar),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GenMode {
    Full,
    TangentSafe,
    Smooth,
}

#[allow(clippy::cast_possible_truncation)]
fn build_random_spec(entropy: &[u8], n_states: usize, mode: GenMode) -> ExprSpec {
    let mut pool: Vec<TypedEntry> = Vec::new();
    let state_class = if mode == GenMode::Smooth {
        ValueClass::BoundedSmall
    } else {
        ValueClass::PositiveFinite
    };

    pool.push(TypedEntry {
        spec: ExprSpec::StateSlice {
            start: 0,
            end: n_states,
            class: state_class,
        },
        len: n_states,
        class: state_class,
    });

    for i in 0..n_states.min(4) {
        pool.push(TypedEntry {
            spec: ExprSpec::StateSlice {
                start: i,
                end: i + 1,
                class: state_class,
            },
            len: 1,
            class: state_class,
        });
    }

    let scalar_vals = [0.5, 1.0, 2.0, -1.0];
    for &v in &scalar_vals {
        let class = if v > 0.0 {
            ValueClass::PositiveFinite
        } else {
            ValueClass::NonZeroFinite
        };
        pool.push(TypedEntry {
            spec: ExprSpec::Scalar(v, class),
            len: 1,
            class,
        });
    }

    let smooth_unary = [
        UnarySpec::Neg,
        UnarySpec::Sin,
        UnarySpec::Cos,
        UnarySpec::Tanh,
        UnarySpec::Sinh,
        UnarySpec::Cosh,
        UnarySpec::Arcsinh,
        UnarySpec::Arctan,
        UnarySpec::Erf,
    ];
    let extra_unary = [UnarySpec::Abs, UnarySpec::Sqrt];
    let exp_unary = [UnarySpec::Exp];

    let smooth_binary = [BinarySpec::Add, BinarySpec::Sub, BinarySpec::Mul];

    let target_successful_ops = if mode == GenMode::Smooth { 12 } else { 16 };
    let max_attempts = target_successful_ops * 3;
    let mut successful_ops = 0;
    let mut attempts = 0;
    let mut last_good_idx = 0;
    let mut byte_cursor = 0;

    let next_byte = |cursor: &mut usize| -> u8 {
        let b = entropy
            .get(*cursor)
            .copied()
            .unwrap_or_else(|| (*cursor as u8).wrapping_mul(31));
        *cursor += 1;
        b
    };

    while successful_ops < target_successful_ops && attempts < max_attempts {
        attempts += 1;
        let op_kind = next_byte(&mut byte_cursor) % 5;

        match op_kind {
            0 | 1 => {
                let all_unary: Vec<UnarySpec> = if mode == GenMode::Smooth {
                    smooth_unary
                        .iter()
                        .chain(exp_unary.iter())
                        .copied()
                        .collect()
                } else {
                    smooth_unary
                        .iter()
                        .chain(exp_unary.iter())
                        .chain(extra_unary.iter())
                        .copied()
                        .collect()
                };
                let op = all_unary[next_byte(&mut byte_cursor) as usize % all_unary.len()];
                let required = unary_input_class(op, mode);

                let child_idx = next_byte(&mut byte_cursor) as usize % pool.len();
                if class_satisfies(pool[child_idx].class, required) {
                    let child = &pool[child_idx];
                    let out_class = unary_output_class(op, child.class);
                    let out_len = child.len;
                    pool.push(TypedEntry {
                        spec: ExprSpec::Unary {
                            op,
                            child: Box::new(child.spec.clone()),
                            len: out_len,
                            class: out_class,
                        },
                        len: out_len,
                        class: out_class,
                    });
                    last_good_idx = pool.len() - 1;
                    successful_ops += 1;
                }
            },
            2 | 3 => {
                let all_binary: Vec<BinarySpec> = match mode {
                    GenMode::Smooth => smooth_binary.to_vec(),
                    GenMode::TangentSafe => {
                        let mut v = smooth_binary.to_vec();
                        v.push(BinarySpec::Div);
                        v
                    },
                    GenMode::Full => {
                        let mut v = smooth_binary.to_vec();
                        v.push(BinarySpec::Div);
                        v.push(BinarySpec::EqualHeaviside);
                        v
                    },
                };
                let op = all_binary[next_byte(&mut byte_cursor) as usize % all_binary.len()];

                let a_idx = next_byte(&mut byte_cursor) as usize % pool.len();
                let b_idx = next_byte(&mut byte_cursor) as usize % pool.len();
                let a = &pool[a_idx];
                let b = &pool[b_idx];

                if !can_broadcast(a.len, b.len) {
                    continue;
                }

                let ok = match op {
                    BinarySpec::Div => class_satisfies(b.class, ValueClass::NonZeroFinite),
                    _ => true,
                };

                if ok {
                    let out_len = a.len.max(b.len);
                    let out_class = binary_output_class(op, a.class, b.class);
                    pool.push(TypedEntry {
                        spec: ExprSpec::Binary {
                            op,
                            lhs: Box::new(a.spec.clone()),
                            rhs: Box::new(b.spec.clone()),
                            len: out_len,
                            class: out_class,
                        },
                        len: out_len,
                        class: out_class,
                    });
                    last_good_idx = pool.len() - 1;
                    successful_ops += 1;
                }
            },
            4 => {
                let do_index = next_byte(&mut byte_cursor) % 2 == 0;
                if do_index {
                    let child_idx = next_byte(&mut byte_cursor) as usize % pool.len();
                    let child = &pool[child_idx];
                    if child.len > 1 {
                        let start = next_byte(&mut byte_cursor) as usize % child.len;
                        let max_end = child.len;
                        let end =
                            start + 1 + (next_byte(&mut byte_cursor) as usize % (max_end - start));
                        let end = end.min(max_end);
                        pool.push(TypedEntry {
                            spec: ExprSpec::Index {
                                child: Box::new(child.spec.clone()),
                                start,
                                end,
                                class: child.class,
                            },
                            len: end - start,
                            class: child.class,
                        });
                        last_good_idx = pool.len() - 1;
                        successful_ops += 1;
                    }
                } else {
                    let n_parts = 2 + (next_byte(&mut byte_cursor) as usize % 3);
                    let parts: Vec<TypedEntry> = (0..n_parts)
                        .map(|_| {
                            let idx = next_byte(&mut byte_cursor) as usize % pool.len();
                            pool[idx].clone()
                        })
                        .collect();
                    let total_len: usize = parts.iter().map(|p| p.len).sum();
                    let specs: Vec<ExprSpec> = parts.iter().map(|p| p.spec.clone()).collect();
                    pool.push(TypedEntry {
                        spec: ExprSpec::Concat(specs, total_len),
                        len: total_len,
                        class: ValueClass::AnyFinite,
                    });
                    last_good_idx = pool.len() - 1;
                    successful_ops += 1;
                }
            },
            _ => {},
        }
    }

    pool[last_good_idx].spec.clone()
}

#[allow(clippy::needless_pass_by_value)]
fn build_random_eval_case(entropy: Vec<u8>, n_states: usize, y: Vec<f64>) -> DagCase {
    let spec = build_random_spec(&entropy, n_states, GenMode::Full);
    let mut arena = Arena::new();
    let root = lower_expr(&mut arena, &spec);
    DagCase {
        arena,
        root,
        y,
        n_states,
    }
}

#[allow(clippy::needless_pass_by_value)]
fn build_random_smooth_tangent_case(entropy: Vec<u8>, n_states: usize, y: Vec<f64>) -> TangentCase {
    let spec = build_random_spec(&entropy, n_states, GenMode::Smooth);
    let mut arena = Arena::new();
    let root = lower_expr(&mut arena, &spec);
    let seeds = seed_set_from_entropy(n_states, &entropy);
    TangentCase {
        arena,
        root,
        y,
        seeds,
        n_states,
    }
}

#[allow(clippy::needless_pass_by_value)]
fn build_random_split_eval_case(entropy: Vec<u8>, n_states: usize, y: Vec<f64>) -> TangentCase {
    let spec = build_random_spec(&entropy, n_states, GenMode::TangentSafe);
    let mut arena = Arena::new();
    let root = lower_expr(&mut arena, &spec);
    let seeds = seed_set_from_entropy(n_states, &entropy);
    TangentCase {
        arena,
        root,
        y,
        seeds,
        n_states,
    }
}

// Targeted shape builders

fn tangent_case_from_dag(case: DagCase, entropy: &[u8]) -> TangentCase {
    TangentCase {
        arena: case.arena,
        root: case.root,
        y: case.y,
        seeds: seed_set_from_entropy(case.n_states, entropy),
        n_states: case.n_states,
    }
}

/// Deep linear chain: y[0..n] + 1 + 1 + ... (depth additions).
#[allow(dead_code)]
pub fn deep_chain_case(n_states: usize, depth: usize) -> DagCase {
    let mut arena = Arena::new();
    let mut cur = arena.alloc(Node::StateVector {
        start: 0,
        end: n_states,
    });
    for _ in 0..depth {
        let one = arena.alloc(Node::Scalar(1.0));
        cur = arena.alloc(Node::Add(cur, one));
    }
    let y = vec![1.0; n_states];
    DagCase {
        arena,
        root: cur,
        y,
        n_states,
    }
}

/// Wide fan-out: Concat(2*y[0], 2*y[1], ..., 2*y[fanout-1]).
#[allow(dead_code)]
pub fn wide_concat_case(fanout: usize) -> DagCase {
    let mut arena = Arena::new();
    let mut terms = Vec::with_capacity(fanout);
    for i in 0..fanout {
        let yi = arena.alloc(Node::StateVector {
            start: i,
            end: i + 1,
        });
        let two = arena.alloc(Node::Scalar(2.0));
        terms.push(arena.alloc(Node::Mul(yi, two)));
    }
    let root = arena.alloc(Node::Concat(terms));
    let y: Vec<f64> = (0..fanout)
        .map(|i| (i as f64).mul_add(0.03, 0.05))
        .collect();
    DagCase {
        arena,
        root,
        y,
        n_states: fanout,
    }
}

#[allow(dead_code)]
pub fn broadcast_mix_case(n_states: usize) -> DagCase {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector {
        start: 0,
        end: n_states,
    });
    let one = arena.alloc(Node::Scalar(1.0));
    let shifted = arena.alloc(Node::Add(y, one));
    let scale = arena.alloc(Node::Scalar(0.5));
    let root = arena.alloc(Node::Mul(shifted, scale));
    DagCase {
        arena,
        root,
        y: vec![1.0; n_states],
        n_states,
    }
}

#[allow(dead_code)]
pub fn index_slice_case() -> DagCase {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 4 });
    let root = arena.alloc(Node::Index {
        child: y,
        start: 1,
        end: 3,
    });
    DagCase {
        arena,
        root,
        y: vec![0.5, 1.0, 1.5, 2.0],
        n_states: 4,
    }
}

#[allow(dead_code)]
pub fn conditional_case() -> DagCase {
    let mut arena = Arena::new();
    let selector = arena.alloc(Node::Scalar(1.0));
    let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
    let s2 = arena.alloc(Node::Scalar(2.0));
    let b1 = arena.alloc(Node::Add(y0, s2));
    let s3 = arena.alloc(Node::Scalar(3.0));
    let b2 = arena.alloc(Node::Mul(y1, s3));
    let root = arena.alloc(Node::Conditional {
        selector,
        branches: vec![b1, b2],
    });
    DagCase {
        arena,
        root,
        y: vec![1.0, 2.0],
        n_states: 2,
    }
}

#[allow(dead_code)]
pub fn interpolant_case() -> DagCase {
    let mut arena = Arena::new();
    let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let root = arena.alloc(Node::Interpolant1DLinear {
        data: Box::new(
            pybamm_core::InterpolantData::try_new(vec![0.0, 1.0, 2.0], vec![0.0, 10.0, 20.0])
                .expect("valid interpolant"),
        ),
        child: x,
    });
    DagCase {
        arena,
        root,
        y: vec![1.5],
        n_states: 1,
    }
}

#[allow(dead_code)]
pub fn sparse_matmul_case() -> DagCase {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector { start: 0, end: 4 });
    let mat = arena.alloc(Node::SparseMatrix(Box::new(
        pybamm_core::CsrData::try_new(
            vec![0, 2, 4],
            vec![0, 1, 2, 3],
            vec![1.0, -1.0, 2.0, 3.0],
            pybamm_core::Shape::matrix(2, 4),
        )
        .expect("valid test matrix"),
    )));
    let root = arena.alloc(Node::MatMul(mat, y));
    DagCase {
        arena,
        root,
        y: vec![0.5, 1.0, 1.5, 2.0],
        n_states: 4,
    }
}

/// Intentional structural duplicate: (3*y[0]) + (3*y[0]) built twice.
#[allow(dead_code)]
pub fn duplicate_subexpr_case() -> DagCase {
    let mut arena = Arena::new();
    let y1 = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let s1 = arena.alloc(Node::Scalar(3.0));
    let m1 = arena.alloc(Node::Mul(y1, s1));
    let y2 = arena.alloc(Node::StateVector { start: 0, end: 1 });
    let s2 = arena.alloc(Node::Scalar(3.0));
    let m2 = arena.alloc(Node::Mul(y2, s2));
    let root = arena.alloc(Node::Add(m1, m2));
    DagCase {
        arena,
        root,
        y: vec![1.25],
        n_states: 1,
    }
}

/// Deep smooth composition: repeatedly apply tanh(0.5*x + 0.25).
#[allow(dead_code)]
pub fn smooth_composition_case(n_states: usize, depth: usize) -> TangentCase {
    let mut arena = Arena::new();
    let mut cur = arena.alloc(Node::StateVector {
        start: 0,
        end: n_states,
    });
    for _ in 0..depth {
        let half = arena.alloc(Node::Scalar(0.5));
        let scaled = arena.alloc(Node::Mul(cur, half));
        let bias = arena.alloc(Node::Scalar(0.25));
        let shifted = arena.alloc(Node::Add(scaled, bias));
        cur = arena.alloc(Node::Tanh(shifted));
    }
    TangentCase {
        arena,
        root: cur,
        y: vec![0.2; n_states],
        seeds: seed_set_from_entropy(n_states, &[31, 32, 33, 34]),
        n_states,
    }
}

#[allow(dead_code)]
pub fn targeted_eval_cases() -> Vec<DagCase> {
    vec![
        deep_chain_case(1, 64),
        deep_chain_case(8, 100),
        wide_concat_case(32),
        wide_concat_case(64),
        broadcast_mix_case(8),
        index_slice_case(),
        conditional_case(),
        interpolant_case(),
        sparse_matmul_case(),
        duplicate_subexpr_case(),
    ]
}

#[allow(dead_code)]
pub fn targeted_smooth_tangent_cases() -> Vec<TangentCase> {
    vec![
        smooth_composition_case(1, 64),
        smooth_composition_case(4, 32),
        tangent_case_from_dag(wide_concat_case(8), &[5, 6, 7, 8]),
        tangent_case_from_dag(broadcast_mix_case(6), &[9, 10, 11, 12]),
        tangent_case_from_dag(index_slice_case(), &[13, 14, 15, 16]),
    ]
}

#[allow(dead_code)]
pub fn targeted_split_eval_cases() -> Vec<TangentCase> {
    vec![
        tangent_case_from_dag(wide_concat_case(8), &[17, 18, 19, 20]),
        tangent_case_from_dag(index_slice_case(), &[21, 22, 23, 24]),
        tangent_case_from_dag(conditional_case(), &[25, 26, 27, 28]),
        tangent_case_from_dag(sparse_matmul_case(), &[29, 30, 31, 32]),
    ]
}

// Proptest strategies

fn arb_random_eval_case() -> impl Strategy<Value = DagCase> {
    (2_usize..=8).prop_flat_map(|n_states| {
        (
            prop::collection::vec(any::<u8>(), 8..48),
            prop::collection::vec(0.1_f64..10.0, n_states),
        )
            .prop_map(move |(entropy, y)| build_random_eval_case(entropy, n_states, y))
    })
}

#[allow(dead_code)]
pub fn arb_eval_case() -> impl Strategy<Value = DagCase> {
    arb_random_eval_case()
}

fn arb_random_smooth_tangent_case() -> impl Strategy<Value = TangentCase> {
    (2_usize..=6).prop_flat_map(|n_states| {
        (
            prop::collection::vec(any::<u8>(), 8..36),
            prop::collection::vec(-1.5_f64..1.5, n_states),
        )
            .prop_map(move |(entropy, y)| build_random_smooth_tangent_case(entropy, n_states, y))
    })
}

#[allow(dead_code)]
pub fn arb_smooth_tangent_case() -> impl Strategy<Value = TangentCase> {
    arb_random_smooth_tangent_case()
}

#[allow(dead_code)]
pub fn arb_split_eval_case() -> impl Strategy<Value = TangentCase> {
    (2_usize..=6).prop_flat_map(|n_states| {
        (
            prop::collection::vec(any::<u8>(), 8..36),
            prop::collection::vec(0.1_f64..10.0, n_states),
        )
            .prop_map(move |(entropy, y)| build_random_split_eval_case(entropy, n_states, y))
    })
}
