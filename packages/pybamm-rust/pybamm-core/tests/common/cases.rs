use proptest::prelude::*;
use pybamm_core::{Arena, CompiledExpr, DagRef, Node, NodeId, TypedIr};

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct DagCase {
    pub arena: Arena,
    pub root: NodeId,
    pub y: Vec<f64>,
    pub n_states: usize,
    /// Packed input-parameter values; empty unless [`Extra::Inputs`] is on.
    pub inputs: Vec<f64>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct TangentCase {
    pub arena: Arena,
    pub root: NodeId,
    pub y: Vec<f64>,
    pub seeds: Vec<Vec<f64>>,
    pub n_states: usize,
    /// Packed input-parameter values; empty unless [`Extra::Inputs`] is on.
    pub inputs: Vec<f64>,
}

/// An opt-in generator extension; with none, every mode draws exactly its original cases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Extra {
    /// `InputParameter` leaves of widths 1, 2 and 3, read from the case's `inputs`.
    Inputs,
    /// `Pow` of a positive base by a literal or an input-parameter exponent.
    Pow,
    /// `Minimum` and `Maximum`.
    MinMax,
    /// `0.0`, `-0.0` and `ZeroVector` leaves, never a `Div` denominator.
    Zeros,
    /// A sparse `MatMul` of random pattern, a `Conditional` on a literal or input selector,
    /// a 2-knot linear interpolant, and `MaxReduce`/`MinReduce`.
    Structural,
    /// `Log` of a positive argument and `Hypot` with a literal leg; away from the smooth mode,
    /// also `Modulo` by a positive divisor, `Sign` and `Floor`, and in the full mode `<` and `==`.
    Elementwise,
}

#[allow(dead_code)]
pub const ALL_EXTRAS: &[Extra] = &[
    Extra::Inputs,
    Extra::Pow,
    Extra::MinMax,
    Extra::Zeros,
    Extra::Structural,
    Extra::Elementwise,
];

/// Widths of the [`Extra::Inputs`] leaves, packed in order.
const INPUT_WIDTHS: [usize; 3] = [1, 2, 3];
const N_INPUTS: usize = 6;
/// Literal exponents for [`Extra::Pow`]: some `simplify` lowers, some it leaves as `pow`.
const POW_EXPONENTS: [f64; 6] = [2.0, 3.0, 4.0, -1.0, -2.0, 0.5];

#[allow(dead_code)]
pub fn eval_dag(arena: &Arena, root: NodeId, t: f64, y: &[f64], inputs: &[f64]) -> Vec<f64> {
    let ir = TypedIr::from_dag(DagRef::new(arena, root));
    let expr = CompiledExpr::from_ir(ir);
    let mut s = vec![0.0; expr.scratch_len()];
    expr.eval(&mut s, t, y, &[], inputs).to_vec()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValueClass {
    AnyFinite,
    PositiveFinite,
    NonZeroFinite,
    BoundedSmall,
    Zero,
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
    Log,
    Sign,
    Floor,
}

#[derive(Clone, Copy, Debug)]
enum BinarySpec {
    Add,
    Sub,
    Mul,
    Div,
    EqualHeaviside,
    Pow,
    Minimum,
    Maximum,
    Modulo,
    Hypot,
    NotEqualHeaviside,
    Equality,
}

#[derive(Clone, Debug)]
enum ExprSpec {
    Scalar(f64),
    Input {
        index: usize,
        offset: usize,
        width: usize,
    },
    ZeroVector(usize),
    StateSlice {
        start: usize,
        end: usize,
    },
    Unary {
        op: UnarySpec,
        child: Box<Self>,
    },
    Binary {
        op: BinarySpec,
        lhs: Box<Self>,
        rhs: Box<Self>,
    },
    Concat(Vec<Self>),
    Index {
        child: Box<Self>,
        start: usize,
        end: usize,
    },
    Conditional {
        selector: Box<Self>,
        branches: Vec<Self>,
    },
    Interpolant1DLinear {
        x_data: Vec<f64>,
        y_data: Vec<f64>,
        child: Box<Self>,
    },
    SparseMatMul {
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<f64>,
        nrows: usize,
        ncols: usize,
        rhs: Box<Self>,
    },
    Reduce {
        max: bool,
        child: Box<Self>,
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
        ExprSpec::Scalar(v) => arena.alloc(Node::Scalar(*v)),
        &ExprSpec::Input {
            index,
            offset,
            width,
        } => arena.alloc(Node::InputParameter {
            name: format!("p{index}"),
            index,
            offset,
            width,
        }),
        &ExprSpec::ZeroVector(len) => arena.alloc(Node::ZeroVector { len }),
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
                UnarySpec::Log => arena.alloc(Node::Log(child)),
                UnarySpec::Sign => arena.alloc(Node::Sign(child)),
                UnarySpec::Floor => arena.alloc(Node::Floor(child)),
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
                BinarySpec::Pow => arena.alloc(Node::Pow(lhs, rhs)),
                BinarySpec::Minimum => arena.alloc(Node::Minimum(lhs, rhs)),
                BinarySpec::Maximum => arena.alloc(Node::Maximum(lhs, rhs)),
                BinarySpec::Modulo => arena.alloc(Node::Modulo(lhs, rhs)),
                BinarySpec::Hypot => arena.alloc(Node::Hypot(lhs, rhs)),
                BinarySpec::NotEqualHeaviside => arena.alloc(Node::NotEqualHeaviside(lhs, rhs)),
                BinarySpec::Equality => arena.alloc(Node::Equality(lhs, rhs)),
            }
        },
        ExprSpec::Concat(children) => {
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
        ExprSpec::Reduce { max, child } => {
            let child = lower_expr(arena, child);
            if *max {
                arena.alloc(Node::MaxReduce(child))
            } else {
                arena.alloc(Node::MinReduce(child))
            }
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
        // Sinh/Cosh grow like Exp, so unbounded inputs break the FD oracle.
        UnarySpec::Sinh | UnarySpec::Cosh => match mode {
            GenMode::Smooth => ValueClass::BoundedSmall,
            GenMode::Full | GenMode::TangentSafe => ValueClass::AnyFinite,
        },
        UnarySpec::Sqrt | UnarySpec::Log => ValueClass::PositiveFinite,
        UnarySpec::Sign
        | UnarySpec::Floor
        | UnarySpec::Arcsinh
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
        UnarySpec::Sin
        | UnarySpec::Cos
        | UnarySpec::Tanh
        | UnarySpec::Erf
        | UnarySpec::Arctan
        | UnarySpec::Sign => ValueClass::BoundedSmall,
        UnarySpec::Neg => match child_class {
            ValueClass::BoundedSmall => ValueClass::BoundedSmall,
            ValueClass::NonZeroFinite => ValueClass::NonZeroFinite,
            _ => ValueClass::AnyFinite,
        },
        UnarySpec::Sinh
        | UnarySpec::Cosh
        | UnarySpec::Arcsinh
        | UnarySpec::Log
        | UnarySpec::Floor => ValueClass::AnyFinite,
    }
}

const fn binary_output_class(op: BinarySpec, a: ValueClass, b: ValueClass) -> ValueClass {
    match op {
        BinarySpec::Add | BinarySpec::Sub | BinarySpec::Minimum | BinarySpec::Maximum => {
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
        // A power is never a base again, so towers cannot overflow.
        BinarySpec::Mul
        | BinarySpec::EqualHeaviside
        | BinarySpec::Pow
        | BinarySpec::Modulo
        | BinarySpec::Hypot
        | BinarySpec::NotEqualHeaviside
        | BinarySpec::Equality => ValueClass::AnyFinite,
    }
}

const fn class_satisfies(actual: ValueClass, required: ValueClass) -> bool {
    use ValueClass::*;
    match required {
        AnyFinite => true,
        BoundedSmall => matches!(actual, BoundedSmall | Zero),
        PositiveFinite => matches!(actual, PositiveFinite),
        NonZeroFinite => matches!(actual, NonZeroFinite | PositiveFinite),
        Zero => matches!(actual, Zero),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GenMode {
    Full,
    TangentSafe,
    Smooth,
}

#[allow(clippy::cast_possible_truncation)]
fn build_random_spec(entropy: &[u8], n_states: usize, mode: GenMode, extras: &[Extra]) -> ExprSpec {
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
        },
        len: n_states,
        class: state_class,
    });

    for i in 0..n_states.min(4) {
        pool.push(TypedEntry {
            spec: ExprSpec::StateSlice {
                start: i,
                end: i + 1,
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
            spec: ExprSpec::Scalar(v),
            len: 1,
            class,
        });
    }

    add_extra_leaves(&mut pool, n_states, extras);

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
    let elementwise = extras.contains(&Extra::Elementwise);
    let structural = extras.contains(&Extra::Structural);

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
        let op_kind = next_byte(&mut byte_cursor) % if structural { 6 } else { 5 };

        match op_kind {
            0 | 1 => {
                let mut all_unary: Vec<UnarySpec> = if mode == GenMode::Smooth {
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
                if elementwise {
                    all_unary.push(UnarySpec::Log);
                    // Steps would put a jump inside a finite-difference stencil.
                    if mode != GenMode::Smooth {
                        all_unary.extend([UnarySpec::Sign, UnarySpec::Floor]);
                    }
                }
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
                        },
                        len: out_len,
                        class: out_class,
                    });
                    last_good_idx = pool.len() - 1;
                    successful_ops += 1;
                }
            },
            2 | 3 => {
                let mut all_binary: Vec<BinarySpec> = match mode {
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
                if extras.contains(&Extra::Pow) {
                    all_binary.push(BinarySpec::Pow);
                }
                if extras.contains(&Extra::MinMax) {
                    all_binary.extend([BinarySpec::Minimum, BinarySpec::Maximum]);
                }
                if elementwise {
                    all_binary.push(BinarySpec::Hypot);
                    if mode != GenMode::Smooth {
                        all_binary.push(BinarySpec::Modulo);
                    }
                    if mode == GenMode::Full {
                        all_binary.extend([BinarySpec::NotEqualHeaviside, BinarySpec::Equality]);
                    }
                }
                let op = all_binary[next_byte(&mut byte_cursor) as usize % all_binary.len()];

                let a_idx = next_byte(&mut byte_cursor) as usize % pool.len();
                let b_idx = next_byte(&mut byte_cursor) as usize % pool.len();
                if matches!(op, BinarySpec::Pow) {
                    let base = &pool[a_idx];
                    if !class_satisfies(base.class, ValueClass::PositiveFinite) {
                        continue;
                    }
                    let exponent = pow_exponent(next_byte(&mut byte_cursor), extras);
                    pool.push(TypedEntry {
                        spec: ExprSpec::Binary {
                            op,
                            lhs: Box::new(base.spec.clone()),
                            rhs: Box::new(exponent),
                        },
                        len: base.len,
                        class: ValueClass::AnyFinite,
                    });
                    last_good_idx = pool.len() - 1;
                    successful_ops += 1;
                    continue;
                }
                if matches!(op, BinarySpec::Hypot) {
                    pool.push(hypot_entry(&pool[a_idx], next_byte(&mut byte_cursor)));
                    last_good_idx = pool.len() - 1;
                    successful_ops += 1;
                    continue;
                }
                let a = &pool[a_idx];
                let b = &pool[b_idx];

                if !can_broadcast(a.len, b.len) {
                    continue;
                }

                let ok = match op {
                    BinarySpec::Div => class_satisfies(b.class, ValueClass::NonZeroFinite),
                    BinarySpec::Modulo => class_satisfies(b.class, ValueClass::PositiveFinite),
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
                        spec: ExprSpec::Concat(specs),
                        len: total_len,
                        class: ValueClass::AnyFinite,
                    });
                    last_good_idx = pool.len() - 1;
                    successful_ops += 1;
                }
            },
            5 => {
                if let Some(entry) =
                    structural_entry(&pool, &mut || next_byte(&mut byte_cursor), extras)
                {
                    pool.push(entry);
                    last_good_idx = pool.len() - 1;
                    successful_ops += 1;
                }
            },
            _ => {},
        }
    }

    pool[last_good_idx].spec.clone()
}

/// `Hypot` of `leg` and a positive literal, in the order `byte` picks.
///
/// A literal leg keeps it off the origin, where both modes' partials are `0 / 0`.
fn hypot_entry(leg: &TypedEntry, byte: u8) -> TypedEntry {
    const LITERAL_LEGS: [f64; 3] = [0.5, 1.0, 2.0];
    let literal = ExprSpec::Scalar(LITERAL_LEGS[usize::from(byte / 2) % LITERAL_LEGS.len()]);
    let leg_spec = Box::new(leg.spec.clone());
    let (lhs, rhs) = if byte.is_multiple_of(2) {
        (leg_spec, Box::new(literal))
    } else {
        (Box::new(literal), leg_spec)
    };
    TypedEntry {
        spec: ExprSpec::Binary {
            op: BinarySpec::Hypot,
            lhs,
            rhs,
        },
        len: leg.len,
        class: ValueClass::PositiveFinite,
    }
}

/// One [`Extra::Structural`] node over `pool`, or `None` when the drawn operands do not fit.
fn structural_entry(
    pool: &[TypedEntry],
    next_byte: &mut impl FnMut() -> u8,
    extras: &[Extra],
) -> Option<TypedEntry> {
    fn pick(pool: &[TypedEntry], byte: u8) -> &TypedEntry {
        &pool[usize::from(byte) % pool.len()]
    }
    match next_byte() % 4 {
        0 => {
            let (first, second) = (pick(pool, next_byte()), pick(pool, next_byte()));
            if first.len != second.len {
                return None;
            }
            let class = if [first.class, second.class]
                .iter()
                .all(|&class| class_satisfies(class, ValueClass::BoundedSmall))
            {
                ValueClass::BoundedSmall
            } else {
                ValueClass::AnyFinite
            };
            // A literal window centre always picks a branch; the input does unless it lands on
            // a half-way value, where the zeros it gives are still bounded.
            let selector = match next_byte() % 3 {
                0 => ExprSpec::Scalar(1.0),
                1 => ExprSpec::Scalar(2.0),
                _ if extras.contains(&Extra::Inputs) => ExprSpec::Input {
                    index: 0,
                    offset: 0,
                    width: 1,
                },
                _ => ExprSpec::Scalar(2.0),
            };
            Some(TypedEntry {
                spec: ExprSpec::Conditional {
                    selector: Box::new(selector),
                    branches: vec![first.spec.clone(), second.spec.clone()],
                },
                len: first.len,
                class,
            })
        },
        1 => {
            let child = pick(pool, next_byte());
            let x0 = f64::from(next_byte() % 4) - 2.0;
            let (y0, y1) = (
                f64::from(next_byte() % 5) - 2.0,
                f64::from(next_byte() % 5) - 2.0,
            );
            Some(TypedEntry {
                spec: ExprSpec::Interpolant1DLinear {
                    x_data: vec![x0, x0 + 1.5],
                    y_data: vec![y0, y1],
                    child: Box::new(child.spec.clone()),
                },
                len: child.len,
                class: ValueClass::AnyFinite,
            })
        },
        2 => {
            const COEFFICIENTS: [f64; 5] = [0.5, -1.0, 1.0, 2.0, -0.25];
            let rhs = pick(pool, next_byte());
            let (nrows, ncols) = (1 + usize::from(next_byte() % 4), rhs.len);
            let (mut indptr, mut indices, mut data) = (vec![0], Vec::new(), Vec::new());
            for _ in 0..nrows {
                for col in 0..ncols {
                    let byte = next_byte();
                    if byte.is_multiple_of(2) {
                        indices.push(col);
                        data.push(COEFFICIENTS[usize::from(byte / 2) % COEFFICIENTS.len()]);
                    }
                }
                indptr.push(indices.len());
            }
            Some(TypedEntry {
                spec: ExprSpec::SparseMatMul {
                    indptr,
                    indices,
                    data,
                    nrows,
                    ncols,
                    rhs: Box::new(rhs.spec.clone()),
                },
                len: nrows,
                class: ValueClass::AnyFinite,
            })
        },
        _ => {
            let child = pick(pool, next_byte());
            Some(TypedEntry {
                spec: ExprSpec::Reduce {
                    max: next_byte().is_multiple_of(2),
                    child: Box::new(child.spec.clone()),
                },
                len: 1,
                class: child.class,
            })
        },
    }
}

/// Push the leaves `extras` add: input parameters, then zeros of each length present.
fn add_extra_leaves(pool: &mut Vec<TypedEntry>, n_states: usize, extras: &[Extra]) {
    let mut lens = vec![n_states];
    if extras.contains(&Extra::Inputs) {
        let mut offset = 0;
        for (index, width) in INPUT_WIDTHS.into_iter().enumerate() {
            pool.push(TypedEntry {
                spec: ExprSpec::Input {
                    index,
                    offset,
                    width,
                },
                len: width,
                class: ValueClass::PositiveFinite,
            });
            offset += width;
            lens.push(width);
        }
    }
    if extras.contains(&Extra::Zeros) {
        for zero in [0.0, -0.0] {
            pool.push(TypedEntry {
                spec: ExprSpec::Scalar(zero),
                len: 1,
                class: ValueClass::Zero,
            });
        }
        lens.sort_unstable();
        lens.dedup();
        for len in lens.into_iter().filter(|&len| len > 1) {
            pool.push(TypedEntry {
                spec: ExprSpec::ZeroVector(len),
                len,
                class: ValueClass::Zero,
            });
        }
    }
}

/// A literal exponent, or the width-1 input parameter when inputs are on.
fn pow_exponent(byte: u8, extras: &[Extra]) -> ExprSpec {
    let choice = usize::from(byte) % (POW_EXPONENTS.len() + 1);
    match POW_EXPONENTS.get(choice) {
        Some(&exponent) => ExprSpec::Scalar(exponent),
        None if extras.contains(&Extra::Inputs) => ExprSpec::Input {
            index: 0,
            offset: 0,
            width: 1,
        },
        None => ExprSpec::Scalar(2.0),
    }
}

/// Lower a random `mode` spec drawn from `entropy`.
fn random_dag(entropy: &[u8], n_states: usize, mode: GenMode, extras: &[Extra]) -> (Arena, NodeId) {
    let spec = build_random_spec(entropy, n_states, mode, extras);
    let mut arena = Arena::new();
    let root = lower_expr(&mut arena, &spec);
    (arena, root)
}

#[allow(clippy::needless_pass_by_value)]
fn build_random_eval_case(entropy: Vec<u8>, n_states: usize, y: Vec<f64>) -> DagCase {
    build_eval_case(&entropy, n_states, y, Vec::new(), &[])
}

fn build_eval_case(
    entropy: &[u8],
    n_states: usize,
    y: Vec<f64>,
    inputs: Vec<f64>,
    extras: &[Extra],
) -> DagCase {
    let (arena, root) = random_dag(entropy, n_states, GenMode::Full, extras);
    DagCase {
        arena,
        root,
        y,
        n_states,
        inputs,
    }
}

#[allow(clippy::needless_pass_by_value)]
fn build_random_smooth_tangent_case(entropy: Vec<u8>, n_states: usize, y: Vec<f64>) -> TangentCase {
    build_tangent_case(&entropy, n_states, y, Vec::new(), GenMode::Smooth, &[])
}

#[allow(clippy::needless_pass_by_value)]
fn build_random_split_eval_case(entropy: Vec<u8>, n_states: usize, y: Vec<f64>) -> TangentCase {
    build_tangent_case(&entropy, n_states, y, Vec::new(), GenMode::TangentSafe, &[])
}

fn build_tangent_case(
    entropy: &[u8],
    n_states: usize,
    y: Vec<f64>,
    inputs: Vec<f64>,
    mode: GenMode,
    extras: &[Extra],
) -> TangentCase {
    let (arena, root) = random_dag(entropy, n_states, mode, extras);
    TangentCase {
        arena,
        root,
        y,
        seeds: seed_set_from_entropy(n_states, entropy),
        n_states,
        inputs,
    }
}

fn tangent_case_from_dag(case: DagCase, entropy: &[u8]) -> TangentCase {
    TangentCase {
        arena: case.arena,
        root: case.root,
        y: case.y,
        seeds: seed_set_from_entropy(case.n_states, entropy),
        n_states: case.n_states,
        inputs: case.inputs,
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
        inputs: Vec::new(),
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
        inputs: Vec::new(),
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
        inputs: Vec::new(),
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
        inputs: Vec::new(),
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
        inputs: Vec::new(),
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
        inputs: Vec::new(),
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
        inputs: Vec::new(),
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
        inputs: Vec::new(),
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
        inputs: Vec::new(),
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

/// Input values for `extras`: positive, so an input can be a `Pow` exponent or a divisor.
fn arb_inputs(extras: &'static [Extra]) -> impl Strategy<Value = Vec<f64>> {
    let len = if extras.contains(&Extra::Inputs) {
        N_INPUTS
    } else {
        0
    };
    prop::collection::vec(0.5_f64..2.0, len)
}

/// [`arb_eval_case`] with `extras` switched on.
#[allow(dead_code)]
pub fn arb_eval_case_with(extras: &'static [Extra]) -> impl Strategy<Value = DagCase> {
    (2_usize..=8).prop_flat_map(move |n_states| {
        (
            prop::collection::vec(any::<u8>(), 8..48),
            prop::collection::vec(0.1_f64..10.0, n_states),
            arb_inputs(extras),
        )
            .prop_map(move |(entropy, y, inputs)| {
                build_eval_case(&entropy, n_states, y, inputs, extras)
            })
    })
}

/// [`arb_smooth_tangent_case`] with `extras` switched on.
#[allow(dead_code)]
pub fn arb_smooth_tangent_case_with(
    extras: &'static [Extra],
) -> impl Strategy<Value = TangentCase> {
    (2_usize..=6).prop_flat_map(move |n_states| {
        (
            prop::collection::vec(any::<u8>(), 8..36),
            prop::collection::vec(-1.5_f64..1.5, n_states),
            arb_inputs(extras),
        )
            .prop_map(move |(entropy, y, inputs)| {
                build_tangent_case(&entropy, n_states, y, inputs, GenMode::Smooth, extras)
            })
    })
}

/// [`arb_split_eval_case`] with `extras` switched on.
#[allow(dead_code)]
pub fn arb_split_eval_case_with(extras: &'static [Extra]) -> impl Strategy<Value = TangentCase> {
    (2_usize..=6).prop_flat_map(move |n_states| {
        (
            prop::collection::vec(any::<u8>(), 8..36),
            prop::collection::vec(0.1_f64..10.0, n_states),
            arb_inputs(extras),
        )
            .prop_map(move |(entropy, y, inputs)| {
                build_tangent_case(&entropy, n_states, y, inputs, GenMode::TangentSafe, extras)
            })
    })
}
