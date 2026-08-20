//! Expression simplification pass for symbolic differentiation.
//!
//! Without simplification, derivative expressions explode. For example,
//! `d(a*b*c)/dx` produces terms like `Mul(Scalar(0), b)` that should fold to zero.
//!
//! This module provides two modes:
//! - **Conservative** (default): Skips rules that could change NaN/infinity behavior
//! - **Aggressive**: Applies all algebraic simplifications
//!
//! # Examples
//!
//! ```
//! use pybamm_core::{Arena, Node, simplify};
//!
//! let mut arena = Arena::new();
//! let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
//! let zero = arena.alloc(Node::Scalar(0.0));
//! let expr = arena.alloc(Node::Add(x, zero));
//!
//! let simplified = simplify(&mut arena, expr);
//! assert_eq!(simplified, x);
//! ```

use crate::arena::{Arena, NodeId, NodeMap};
use crate::eval::{erf_approx, sign};
use crate::node::Node;

/// Controls which simplification rules are applied.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum SimplifyMode {
    /// Value-preserving rewrites only. Rules that would change a NaN result are
    /// skipped: `0 * x -> 0` (x=Inf), `0 / x -> 0` and `x / x -> 1` (x=0), and
    /// `x - x -> 0` (x=Inf). So are the two sub rules that would change a zero's
    /// sign: `x - 0 -> x` from `-0.0` and `0 - x -> -x` from `+0.0`.
    ///
    /// Values are exact but the sign of a zero result is not: `-0.0 + 0.0`
    /// is `+0.0` where folding the `+ 0` away yields `-0.0`. No `PyBaMM` path
    /// distinguishes them, so the fold stays rather than bloating every tape
    /// `zero_propagate` feeds.
    #[default]
    Conservative,

    /// Apply all algebraic simplifications.
    /// Use when you know inputs are well-behaved (no NaN/Inf).
    Aggressive,
}

/// Constant-fold and apply algebraic identities in `Conservative` mode.
#[must_use]
pub fn simplify(arena: &mut Arena, root: NodeId) -> NodeId {
    simplify_with_mode(arena, root, SimplifyMode::Conservative)
}

/// As [`simplify`], with the rule set chosen by `mode`.
#[must_use]
pub fn simplify_with_mode(arena: &mut Arena, root: NodeId, mode: SimplifyMode) -> NodeId {
    let mut memo: NodeMap<NodeId> = NodeMap::new(arena.len());
    let mut lens: NodeMap<usize> = NodeMap::new(arena.len());
    simplify_node(arena, root, mode, &mut memo, &mut lens)
}

/// Check if a node is a scalar with a specific value.
///
/// Exact literal match: tiny nonzero coefficients (e.g. 1e-17) must not be
/// treated as 0/1. Mirrors `zero_propagate`'s exact `== 0.0` policy.
#[allow(clippy::float_cmp)]
fn is_scalar(arena: &Arena, id: NodeId, value: f64) -> bool {
    match arena.get(id) {
        Node::Scalar(v) => *v == value,
        _ => false,
    }
}

/// Check if a node is the scalar 0.0.
fn is_zero(arena: &Arena, id: NodeId) -> bool {
    is_scalar(arena, id, 0.0)
}

/// Check if a node is the scalar `+0.0`, distinguished from `-0.0`.
///
/// `is_zero` cannot tell the two apart (`-0.0 == 0.0`), but the additive
/// identities are only bit-exact for one sign of zero each.
fn is_positive_zero(arena: &Arena, id: NodeId) -> bool {
    matches!(arena.get(id), Node::Scalar(v) if v.to_bits() == 0.0_f64.to_bits())
}

/// Check if a node is the scalar `-0.0`. See [`is_positive_zero`].
fn is_negative_zero(arena: &Arena, id: NodeId) -> bool {
    matches!(arena.get(id), Node::Scalar(v) if v.to_bits() == (-0.0_f64).to_bits())
}

/// Check if a node is the scalar 1.0.
fn is_one(arena: &Arena, id: NodeId) -> bool {
    is_scalar(arena, id, 1.0)
}

/// Check if a node is the scalar -1.0.
fn is_neg_one(arena: &Arena, id: NodeId) -> bool {
    is_scalar(arena, id, -1.0)
}

/// Get the scalar value if the node is a scalar.
fn get_scalar(arena: &Arena, id: NodeId) -> Option<f64> {
    match arena.get(id) {
        Node::Scalar(v) => Some(*v),
        _ => None,
    }
}

/// Broadcast length of a node, mirroring `ir::infer_sizes` semantics.
///
/// Memoized via `lens`. Used by folds that replace an expression with a
/// constant, so the replacement keeps the expression's length; also reused by
/// `jacobian::mask_scalar_rows` and `row_extract` to walk `Concat` offsets.
pub(crate) fn node_len(arena: &Arena, id: NodeId, lens: &mut NodeMap<usize>) -> usize {
    if let Some(&len) = lens.get(id) {
        return len;
    }
    let len = match arena.get(id) {
        Node::Scalar(_)
        | Node::Time
        | Node::TangentParameter { .. }
        | Node::MaxReduce(_)
        | Node::MinReduce(_)
        | Node::ReduceArgSelect { .. } => 1,
        // Width, not 1: a vector input broadcasts like any other vector, which
        // is how `infer_sizes` sizes it.
        Node::InputParameter { width, .. } => *width,
        Node::Array(arr) => arr.data.len(),
        Node::ZeroVector { len } => *len,
        Node::SparseMatrix(_) => 0,
        Node::StateVector { start, end }
        | Node::StateVectorDot { start, end }
        | Node::TangentStateVector { start, end }
        | Node::Index { start, end, .. } => end - start,
        Node::Add(a, b)
        | Node::Sub(a, b)
        | Node::Mul(a, b)
        | Node::Div(a, b)
        | Node::Pow(a, b)
        | Node::Minimum(a, b)
        | Node::Maximum(a, b)
        | Node::Modulo(a, b)
        | Node::Hypot(a, b)
        | Node::EqualHeaviside(a, b)
        | Node::NotEqualHeaviside(a, b)
        | Node::Equality(a, b) => {
            let (a, b) = (*a, *b);
            node_len(arena, a, lens).max(node_len(arena, b, lens))
        },
        Node::MatMul(a, b) => match arena.get(*a) {
            Node::SparseMatrix(csr) => csr.shape.rows,
            Node::Array(arr) => arr.shape.rows,
            _ => node_len(arena, *b, lens),
        },
        Node::Concat(children) => {
            let mut total = 0;
            for &c in children {
                total += node_len(arena, c, lens);
            }
            total
        },
        Node::Neg(a)
        | Node::Abs(a)
        | Node::Sqrt(a)
        | Node::Exp(a)
        | Node::Log(a)
        | Node::Sin(a)
        | Node::Cos(a)
        | Node::Tanh(a)
        | Node::Sinh(a)
        | Node::Cosh(a)
        | Node::Arcsinh(a)
        | Node::Arctan(a)
        | Node::Erf(a)
        | Node::Sign(a)
        | Node::Floor(a)
        | Node::Ceiling(a) => node_len(arena, *a, lens),
        Node::Interpolant1DLinear { child, .. }
        | Node::Interpolant1DLinearDeriv { child, .. }
        | Node::Interpolant1DCubic { child, .. }
        | Node::Interpolant1DCubicDeriv { child, .. } => node_len(arena, *child, lens),
        Node::InterpolantNd { children, .. } | Node::InterpolantNdPartial { children, .. } => {
            let mut max = 1;
            for &c in children {
                max = max.max(node_len(arena, c, lens));
            }
            max
        },
        Node::Conditional { branches, .. } => {
            let mut max = 1;
            for &b in branches {
                max = max.max(node_len(arena, b, lens));
            }
            max
        },
    };
    lens.insert(id, len);
    len
}

/// Allocate a zero with the given broadcast length.
fn zero_of_len(arena: &mut Arena, len: usize) -> NodeId {
    if len == 1 {
        arena.alloc(Node::Scalar(0.0))
    } else {
        arena.alloc(Node::ZeroVector { len })
    }
}

/// Recursively simplify a node, using memoization to avoid redundant work.
fn simplify_node(
    arena: &mut Arena,
    id: NodeId,
    mode: SimplifyMode,
    memo: &mut NodeMap<NodeId>,
    lens: &mut NodeMap<usize>,
) -> NodeId {
    // Check memo first
    if let Some(&cached) = memo.get(id) {
        return cached;
    }

    let result = match arena.get(id).clone() {
        // Leaves - no simplification needed
        Node::Scalar(_)
        | Node::Array(_)
        | Node::ZeroVector { .. }
        | Node::SparseMatrix(_)
        | Node::StateVector { .. }
        | Node::StateVectorDot { .. }
        | Node::InputParameter { .. }
        | Node::Time
        | Node::TangentStateVector { .. }
        | Node::TangentParameter { .. } => id,

        // Binary operations
        Node::Add(lhs, rhs) => simplify_add(arena, lhs, rhs, mode, memo, lens),
        Node::Sub(lhs, rhs) => simplify_sub(arena, lhs, rhs, mode, memo, lens),
        Node::Mul(lhs, rhs) => simplify_mul(arena, lhs, rhs, mode, memo, lens),
        Node::Div(lhs, rhs) => simplify_div(arena, lhs, rhs, mode, memo, lens),
        Node::Pow(base, exp) => simplify_pow(arena, base, exp, mode, memo, lens),

        // Other binary ops - just simplify children, no algebraic rules
        Node::MatMul(lhs, rhs) => {
            let lhs_s = simplify_node(arena, lhs, mode, memo, lens);
            let rhs_s = simplify_node(arena, rhs, mode, memo, lens);
            if lhs_s == lhs && rhs_s == rhs {
                id
            } else {
                arena.alloc(Node::MatMul(lhs_s, rhs_s))
            }
        },
        Node::Minimum(lhs, rhs) => {
            let lhs_s = simplify_node(arena, lhs, mode, memo, lens);
            let rhs_s = simplify_node(arena, rhs, mode, memo, lens);
            // Constant folding
            if let (Some(a), Some(b)) = (get_scalar(arena, lhs_s), get_scalar(arena, rhs_s)) {
                return arena.alloc(Node::Scalar(a.min(b)));
            }
            if lhs_s == lhs && rhs_s == rhs {
                id
            } else {
                arena.alloc(Node::Minimum(lhs_s, rhs_s))
            }
        },
        Node::Maximum(lhs, rhs) => {
            let lhs_s = simplify_node(arena, lhs, mode, memo, lens);
            let rhs_s = simplify_node(arena, rhs, mode, memo, lens);
            // Constant folding
            if let (Some(a), Some(b)) = (get_scalar(arena, lhs_s), get_scalar(arena, rhs_s)) {
                return arena.alloc(Node::Scalar(a.max(b)));
            }
            if lhs_s == lhs && rhs_s == rhs {
                id
            } else {
                arena.alloc(Node::Maximum(lhs_s, rhs_s))
            }
        },
        Node::Modulo(lhs, rhs) => {
            let lhs_s = simplify_node(arena, lhs, mode, memo, lens);
            let rhs_s = simplify_node(arena, rhs, mode, memo, lens);
            // Constant folding
            if let (Some(a), Some(b)) = (get_scalar(arena, lhs_s), get_scalar(arena, rhs_s)) {
                return arena.alloc(Node::Scalar(a % b));
            }
            if lhs_s == lhs && rhs_s == rhs {
                id
            } else {
                arena.alloc(Node::Modulo(lhs_s, rhs_s))
            }
        },
        Node::Hypot(lhs, rhs) => {
            let lhs_s = simplify_node(arena, lhs, mode, memo, lens);
            let rhs_s = simplify_node(arena, rhs, mode, memo, lens);
            // Constant folding
            if let (Some(a), Some(b)) = (get_scalar(arena, lhs_s), get_scalar(arena, rhs_s)) {
                return arena.alloc(Node::Scalar(a.hypot(b)));
            }
            if lhs_s == lhs && rhs_s == rhs {
                id
            } else {
                arena.alloc(Node::Hypot(lhs_s, rhs_s))
            }
        },
        Node::EqualHeaviside(lhs, rhs) => {
            let lhs_s = simplify_node(arena, lhs, mode, memo, lens);
            let rhs_s = simplify_node(arena, rhs, mode, memo, lens);
            if lhs_s == lhs && rhs_s == rhs {
                id
            } else {
                arena.alloc(Node::EqualHeaviside(lhs_s, rhs_s))
            }
        },
        Node::NotEqualHeaviside(lhs, rhs) => {
            let lhs_s = simplify_node(arena, lhs, mode, memo, lens);
            let rhs_s = simplify_node(arena, rhs, mode, memo, lens);
            if lhs_s == lhs && rhs_s == rhs {
                id
            } else {
                arena.alloc(Node::NotEqualHeaviside(lhs_s, rhs_s))
            }
        },
        Node::Equality(lhs, rhs) => {
            let lhs_s = simplify_node(arena, lhs, mode, memo, lens);
            let rhs_s = simplify_node(arena, rhs, mode, memo, lens);
            if lhs_s == lhs && rhs_s == rhs {
                id
            } else {
                arena.alloc(Node::Equality(lhs_s, rhs_s))
            }
        },

        // Unary operations
        Node::Neg(child) => simplify_neg(arena, child, mode, memo, lens),
        Node::Abs(child) => simplify_unary(arena, child, mode, memo, lens, f64::abs, Node::Abs),
        Node::Sqrt(child) => simplify_unary(arena, child, mode, memo, lens, f64::sqrt, Node::Sqrt),
        Node::Exp(child) => simplify_unary(arena, child, mode, memo, lens, f64::exp, Node::Exp),
        Node::Log(child) => simplify_unary(arena, child, mode, memo, lens, f64::ln, Node::Log),
        Node::Sin(child) => simplify_unary(arena, child, mode, memo, lens, f64::sin, Node::Sin),
        Node::Cos(child) => simplify_unary(arena, child, mode, memo, lens, f64::cos, Node::Cos),
        Node::Tanh(child) => simplify_unary(arena, child, mode, memo, lens, f64::tanh, Node::Tanh),
        Node::Sinh(child) => simplify_unary(arena, child, mode, memo, lens, f64::sinh, Node::Sinh),
        Node::Cosh(child) => simplify_unary(arena, child, mode, memo, lens, f64::cosh, Node::Cosh),
        Node::Arcsinh(child) => {
            simplify_unary(arena, child, mode, memo, lens, f64::asinh, Node::Arcsinh)
        },
        Node::Arctan(child) => {
            simplify_unary(arena, child, mode, memo, lens, f64::atan, Node::Arctan)
        },
        Node::Erf(child) => simplify_unary(arena, child, mode, memo, lens, erf_approx, Node::Erf),
        Node::Sign(child) => simplify_unary(arena, child, mode, memo, lens, sign, Node::Sign),
        Node::Floor(child) => {
            simplify_unary(arena, child, mode, memo, lens, f64::floor, Node::Floor)
        },
        Node::Ceiling(child) => {
            simplify_unary(arena, child, mode, memo, lens, f64::ceil, Node::Ceiling)
        },
        Node::MaxReduce(child) => {
            let child_s = simplify_node(arena, child, mode, memo, lens);
            if child_s == child {
                id
            } else {
                arena.alloc(Node::MaxReduce(child_s))
            }
        },
        Node::MinReduce(child) => {
            let child_s = simplify_node(arena, child, mode, memo, lens);
            if child_s == child {
                id
            } else {
                arena.alloc(Node::MinReduce(child_s))
            }
        },
        Node::ReduceArgSelect {
            basis,
            picker,
            is_max,
        } => {
            let basis_s = simplify_node(arena, basis, mode, memo, lens);
            // basis all-zero -> selecting any element yields the scalar 0
            if is_zero(arena, basis_s) || matches!(arena.get(basis_s), Node::ZeroVector { .. }) {
                arena.alloc(Node::Scalar(0.0))
            } else {
                let picker_s = simplify_node(arena, picker, mode, memo, lens);
                if basis_s == basis && picker_s == picker {
                    id
                } else {
                    arena.alloc(Node::ReduceArgSelect {
                        basis: basis_s,
                        picker: picker_s,
                        is_max,
                    })
                }
            }
        },

        // Structural operations
        Node::Index { child, start, end } => {
            let child_s = simplify_node(arena, child, mode, memo, lens);
            if child_s == child {
                id
            } else {
                arena.alloc(Node::Index {
                    child: child_s,
                    start,
                    end,
                })
            }
        },
        Node::Concat(children) => {
            let children_s: Vec<NodeId> = children
                .iter()
                .map(|&c| simplify_node(arena, c, mode, memo, lens))
                .collect();
            if children_s == children {
                id
            } else {
                arena.alloc(Node::Concat(children_s))
            }
        },

        Node::Interpolant1DLinear { data, child } => {
            let child_s = simplify_node(arena, child, mode, memo, lens);
            if child_s == child {
                id
            } else {
                arena.alloc(Node::Interpolant1DLinear {
                    data,
                    child: child_s,
                })
            }
        },
        Node::Interpolant1DLinearDeriv {
            slopes,
            x_data,
            child,
        } => {
            let child_s = simplify_node(arena, child, mode, memo, lens);
            if child_s == child {
                id
            } else {
                arena.alloc(Node::Interpolant1DLinearDeriv {
                    slopes,
                    x_data,
                    child: child_s,
                })
            }
        },
        Node::Interpolant1DCubic { data, child } => {
            let child_s = simplify_node(arena, child, mode, memo, lens);
            if child_s == child {
                id
            } else {
                arena.alloc(Node::Interpolant1DCubic {
                    data,
                    child: child_s,
                })
            }
        },
        Node::Interpolant1DCubicDeriv { data, child } => {
            let child_s = simplify_node(arena, child, mode, memo, lens);
            if child_s == child {
                id
            } else {
                arena.alloc(Node::Interpolant1DCubicDeriv {
                    data,
                    child: child_s,
                })
            }
        },
        Node::InterpolantNd { data, children } => {
            let children_s: Vec<NodeId> = children
                .iter()
                .map(|&c| simplify_node(arena, c, mode, memo, lens))
                .collect();
            if children_s == children {
                id
            } else {
                arena.alloc(Node::InterpolantNd {
                    data,
                    children: children_s,
                })
            }
        },
        Node::InterpolantNdPartial {
            data,
            children,
            axis,
        } => {
            let children_s: Vec<NodeId> = children
                .iter()
                .map(|&c| simplify_node(arena, c, mode, memo, lens))
                .collect();
            if children_s == children {
                id
            } else {
                arena.alloc(Node::InterpolantNdPartial {
                    data,
                    children: children_s,
                    axis,
                })
            }
        },

        Node::Conditional { selector, branches } => {
            let selector_s = simplify_node(arena, selector, mode, memo, lens);
            let branches_s: Vec<NodeId> = branches
                .iter()
                .map(|&b| simplify_node(arena, b, mode, memo, lens))
                .collect();
            if selector_s == selector && branches_s == branches {
                id
            } else {
                arena.alloc(Node::Conditional {
                    selector: selector_s,
                    branches: branches_s,
                })
            }
        },
    };

    memo.insert(id, result);
    result
}

/// `x + 0 -> x`, `0 + x -> x`, `const + const -> const`.
fn simplify_add(
    arena: &mut Arena,
    lhs: NodeId,
    rhs: NodeId,
    mode: SimplifyMode,
    memo: &mut NodeMap<NodeId>,
    lens: &mut NodeMap<usize>,
) -> NodeId {
    let lhs_s = simplify_node(arena, lhs, mode, memo, lens);
    let rhs_s = simplify_node(arena, rhs, mode, memo, lens);

    // Constant folding
    if let (Some(a), Some(b)) = (get_scalar(arena, lhs_s), get_scalar(arena, rhs_s)) {
        return arena.alloc(Node::Scalar(a + b));
    }

    // 0 + x -> x
    if is_zero(arena, lhs_s) {
        return rhs_s;
    }

    // x + 0 -> x
    if is_zero(arena, rhs_s) {
        return lhs_s;
    }

    arena.alloc(Node::Add(lhs_s, rhs_s))
}

/// `x - 0 -> x`, `0 - x -> -x`, `x - x -> 0`, `const - const -> const`.
fn simplify_sub(
    arena: &mut Arena,
    lhs: NodeId,
    rhs: NodeId,
    mode: SimplifyMode,
    memo: &mut NodeMap<NodeId>,
    lens: &mut NodeMap<usize>,
) -> NodeId {
    let lhs_s = simplify_node(arena, lhs, mode, memo, lens);
    let rhs_s = simplify_node(arena, rhs, mode, memo, lens);

    // Constant folding
    if let (Some(a), Some(b)) = (get_scalar(arena, lhs_s), get_scalar(arena, rhs_s)) {
        return arena.alloc(Node::Scalar(a - b));
    }

    // x - 0 -> x, exact only from +0.0 (x = -0.0 would flip positive).
    // Aggressive already normalises zero's sign, as `0 - x` below does.
    if is_positive_zero(arena, rhs_s) || (mode == SimplifyMode::Aggressive && is_zero(arena, rhs_s))
    {
        return lhs_s;
    }

    // 0 - x -> -x, exact only from -0.0: `(+0.0) - (+0.0)` is `+0.0` where
    // `-(+0.0)` is `-0.0`, so from +0.0 it is aggressive-only like `x - x`.
    if is_negative_zero(arena, lhs_s) || (mode == SimplifyMode::Aggressive && is_zero(arena, lhs_s))
    {
        return arena.alloc(Node::Neg(rhs_s));
    }

    // x - x -> 0 (aggressive only: Inf - Inf = NaN); zero keeps x's length
    if mode == SimplifyMode::Aggressive && lhs_s == rhs_s {
        let len = node_len(arena, lhs_s, lens);
        return zero_of_len(arena, len);
    }

    arena.alloc(Node::Sub(lhs_s, rhs_s))
}

/// `1 * x -> x`, `x * 1 -> x`, `-1 * x -> -x`, `const * const -> const`, and
/// `0 * x -> 0` in aggressive mode.
fn simplify_mul(
    arena: &mut Arena,
    lhs: NodeId,
    rhs: NodeId,
    mode: SimplifyMode,
    memo: &mut NodeMap<NodeId>,
    lens: &mut NodeMap<usize>,
) -> NodeId {
    let lhs_s = simplify_node(arena, lhs, mode, memo, lens);
    let rhs_s = simplify_node(arena, rhs, mode, memo, lens);

    // Constant folding
    if let (Some(a), Some(b)) = (get_scalar(arena, lhs_s), get_scalar(arena, rhs_s)) {
        return arena.alloc(Node::Scalar(a * b));
    }

    // 1 * x -> x
    if is_one(arena, lhs_s) {
        return rhs_s;
    }

    // x * 1 -> x
    if is_one(arena, rhs_s) {
        return lhs_s;
    }

    // -1 * x -> -x
    if is_neg_one(arena, lhs_s) {
        return arena.alloc(Node::Neg(rhs_s));
    }

    // x * -1 -> -x
    if is_neg_one(arena, rhs_s) {
        return arena.alloc(Node::Neg(lhs_s));
    }

    // Aggressive mode only: 0 * x -> 0, x * 0 -> 0
    // The zero keeps the broadcast length of the product.
    if mode == SimplifyMode::Aggressive && (is_zero(arena, lhs_s) || is_zero(arena, rhs_s)) {
        let len = node_len(arena, lhs_s, lens).max(node_len(arena, rhs_s, lens));
        return zero_of_len(arena, len);
    }

    arena.alloc(Node::Mul(lhs_s, rhs_s))
}

/// `x / 1 -> x`, `const / const -> const`, and `0 / x -> 0`, `x / x -> 1` in
/// aggressive mode.
fn simplify_div(
    arena: &mut Arena,
    lhs: NodeId,
    rhs: NodeId,
    mode: SimplifyMode,
    memo: &mut NodeMap<NodeId>,
    lens: &mut NodeMap<usize>,
) -> NodeId {
    let lhs_s = simplify_node(arena, lhs, mode, memo, lens);
    let rhs_s = simplify_node(arena, rhs, mode, memo, lens);

    // Constant folding (uses IEEE semantics, so 1/0 = inf, 0/0 = nan)
    if let (Some(a), Some(b)) = (get_scalar(arena, lhs_s), get_scalar(arena, rhs_s)) {
        return arena.alloc(Node::Scalar(a / b));
    }

    // x / 1 -> x
    if is_one(arena, rhs_s) {
        return lhs_s;
    }

    // Aggressive mode only
    if mode == SimplifyMode::Aggressive {
        // 0 / x -> 0 (the zero keeps the broadcast length of the quotient)
        if is_zero(arena, lhs_s) {
            let len = node_len(arena, rhs_s, lens);
            return zero_of_len(arena, len);
        }

        // x / x -> 1 (scalar x only: a vector x / x is a ones vector)
        if lhs_s == rhs_s && node_len(arena, lhs_s, lens) == 1 {
            return arena.alloc(Node::Scalar(1.0));
        }
    }

    arena.alloc(Node::Div(lhs_s, rhs_s))
}

/// `x^0 -> 1`, `x^1 -> x`, `1^x -> 1`, `const^const -> const`.
fn simplify_pow(
    arena: &mut Arena,
    base: NodeId,
    exp: NodeId,
    mode: SimplifyMode,
    memo: &mut NodeMap<NodeId>,
    lens: &mut NodeMap<usize>,
) -> NodeId {
    let base_s = simplify_node(arena, base, mode, memo, lens);
    let exp_s = simplify_node(arena, exp, mode, memo, lens);

    // Constant folding
    if let (Some(a), Some(b)) = (get_scalar(arena, base_s), get_scalar(arena, exp_s)) {
        return arena.alloc(Node::Scalar(a.powf(b)));
    }

    // x^0 -> 1 (scalar base only: a vector x^0 is a ones vector)
    if is_zero(arena, exp_s) && node_len(arena, base_s, lens) == 1 {
        return arena.alloc(Node::Scalar(1.0));
    }

    // x^1 -> x
    if is_one(arena, exp_s) {
        return base_s;
    }

    // 1^x -> 1 (scalar exponent only: result broadcasts to the exponent's length)
    if is_one(arena, base_s) && node_len(arena, exp_s, lens) == 1 {
        return base_s;
    }

    // powf is ~30x an elementwise multiply, and a multiply/divide chain keeps
    // the IEEE special cases (sign of zero, infinities, NaN) exact.
    if let Some(e) = get_scalar(arena, exp_s)
        && let Some(id) = lower_int_pow(arena, base_s, e)
    {
        return id;
    }

    arena.alloc(Node::Pow(base_s, exp_s))
}

/// Lower `base^e` for e in {2, 3, 4, -1, -2} to a Mul/Div chain, else `None`.
#[allow(clippy::float_cmp)] // exact literal exponents only
fn lower_int_pow(arena: &mut Arena, base: NodeId, e: f64) -> Option<NodeId> {
    if e == 2.0 {
        return Some(arena.alloc(Node::Mul(base, base)));
    }
    if e == 3.0 {
        let sq = arena.alloc(Node::Mul(base, base));
        return Some(arena.alloc(Node::Mul(sq, base)));
    }
    if e == 4.0 {
        let sq = arena.alloc(Node::Mul(base, base));
        return Some(arena.alloc(Node::Mul(sq, sq)));
    }
    if e == -1.0 {
        let one = arena.alloc(Node::Scalar(1.0));
        return Some(arena.alloc(Node::Div(one, base)));
    }
    if e == -2.0 {
        // (1/x)*(1/x), not 1/(x*x): squaring first overflows to 0 for large
        // finite x (e.g. 1e160), whereas this form stays representable like powf.
        let one = arena.alloc(Node::Scalar(1.0));
        let inv = arena.alloc(Node::Div(one, base));
        return Some(arena.alloc(Node::Mul(inv, inv)));
    }
    None
}

/// `-const -> const`, `--x -> x`.
fn simplify_neg(
    arena: &mut Arena,
    child: NodeId,
    mode: SimplifyMode,
    memo: &mut NodeMap<NodeId>,
    lens: &mut NodeMap<usize>,
) -> NodeId {
    let child_s = simplify_node(arena, child, mode, memo, lens);

    // Constant folding
    if let Some(v) = get_scalar(arena, child_s) {
        return arena.alloc(Node::Scalar(-v));
    }

    // --x -> x (double negation)
    if let Node::Neg(inner) = arena.get(child_s) {
        return *inner;
    }

    arena.alloc(Node::Neg(child_s))
}

/// Helper for unary operations with constant folding
fn simplify_unary<F, G>(
    arena: &mut Arena,
    child: NodeId,
    mode: SimplifyMode,
    memo: &mut NodeMap<NodeId>,
    lens: &mut NodeMap<usize>,
    fold_fn: F,
    constructor: G,
) -> NodeId
where
    F: Fn(f64) -> f64,
    G: Fn(NodeId) -> Node,
{
    let child_s = simplify_node(arena, child, mode, memo, lens);

    // Constant folding
    if let Some(v) = get_scalar(arena, child_s) {
        return arena.alloc(Node::Scalar(fold_fn(v)));
    }

    arena.alloc(constructor(child_s))
}

/// Common Subexpression Elimination
///
/// Identifies structurally equivalent subtrees and reuses them.
/// Returns a new arena containing only unique nodes and the remapped root.
///
/// This is important for derivative expressions which duplicate primal computations.
#[must_use]
pub fn cse(arena: &Arena, root: NodeId) -> (Arena, NodeId) {
    use rustc_hash::FxHashMap;
    use rustc_hash::FxHasher;
    use std::hash::Hasher;

    use crate::node::structural_hash;

    let mut new_arena = Arena::new();
    let mut old_to_new: NodeMap<NodeId> = NodeMap::new(arena.len());
    let mut hash_to_candidates: FxHashMap<u64, Vec<NodeId>> = FxHashMap::default();

    let order = arena.topological_order(root);

    for old_id in order {
        let node = arena.get(old_id);
        let remapped = node.map_children(|c| {
            old_to_new
                .get(c)
                .copied()
                .expect("child precedes parent in topo order")
        });
        let mut hasher = FxHasher::default();
        structural_hash(&remapped, &mut hasher, |c| c);
        let hash = hasher.finish();

        let canonical = hash_to_candidates.get(&hash).and_then(|cands| {
            cands
                .iter()
                .copied()
                .find(|&existing| new_arena.get(existing) == &remapped)
        });

        if let Some(id) = canonical {
            old_to_new.insert(old_id, id);
        } else {
            let new_id = new_arena.alloc(remapped);
            old_to_new.insert(old_id, new_id);
            hash_to_candidates.entry(hash).or_default().push(new_id);
        }
    }

    let new_root = old_to_new
        .get(root)
        .copied()
        .expect("root must be processed");
    (new_arena, new_root)
}

/// The full compile-prep simplification pipeline:
/// aggressive simplify -> zero propagation -> CSE -> DCE.
///
/// The single home for the pass order shared by primal compiles, tangent
/// tapes, jacobian prep, sensitivities and algebraic blocks. Takes the
/// arena by value, callers clone (or own) the arena they want rewritten.
#[must_use]
pub fn simplify_pipeline(mut arena: Arena, root: NodeId) -> (Arena, NodeId) {
    use crate::zero_propagate::zero_propagate;

    let root = simplify_with_mode(&mut arena, root, SimplifyMode::Aggressive);
    let (arena, root) = zero_propagate(&arena, root);
    let (arena, root) = cse(&arena, root);
    dce(&arena, root)
}

/// Dead Code Elimination
///
/// Removes nodes not reachable from the root.
/// Returns a new arena containing only reachable nodes and the remapped root.
#[must_use]
pub fn dce(arena: &Arena, root: NodeId) -> (Arena, NodeId) {
    let mut new_arena = Arena::new();
    let mut old_to_new: NodeMap<NodeId> = NodeMap::new(arena.len());

    // Get topological order (only visits reachable nodes)
    let order = arena.topological_order(root);

    for old_id in order {
        let node = arena.get(old_id);
        let remapped_node = node.map_children(|c| {
            old_to_new
                .get(c)
                .copied()
                .expect("child precedes parent in topo order")
        });
        let new_id = new_arena.alloc(remapped_node);
        old_to_new.insert(old_id, new_id);
    }

    let new_root = old_to_new
        .get(root)
        .copied()
        .expect("root must be processed");
    (new_arena, new_root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::Node;

    #[test]
    fn simplify_pipeline_matches_manual_pass_order() {
        use crate::zero_propagate::zero_propagate;

        // decorated: (sin(y0) * 1.0) + 0.0, must reduce like the manual
        // simplify -> zero_propagate -> cse -> dce chain
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let s = arena.alloc(Node::Sin(y0));
        let one = arena.alloc(Node::Scalar(1.0));
        let zero = arena.alloc(Node::Scalar(0.0));
        let m = arena.alloc(Node::Mul(s, one));
        let decorated = arena.alloc(Node::Add(m, zero));

        let (pipe_arena, pipe_root) = simplify_pipeline(arena.clone(), decorated);

        let mut manual = arena;
        let r = simplify_with_mode(&mut manual, decorated, SimplifyMode::Aggressive);
        let (za, r) = zero_propagate(&manual, r);
        let (ca, r) = cse(&za, r);
        let (ma, r) = dce(&ca, r);

        assert_eq!(pipe_arena.len(), ma.len());
        assert_eq!(pipe_root, r);
        assert_eq!(pipe_arena.get(pipe_root), ma.get(r));
    }

    #[test]
    fn test_fold_multiply_by_one() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let one = arena.alloc(Node::Scalar(1.0));
        let expr = arena.alloc(Node::Mul(x, one));

        let simplified = simplify(&mut arena, expr);

        assert_eq!(simplified, x);
    }

    #[test]
    fn test_fold_one_times_x() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let one = arena.alloc(Node::Scalar(1.0));
        let expr = arena.alloc(Node::Mul(one, x));

        let simplified = simplify(&mut arena, expr);

        assert_eq!(simplified, x);
    }

    #[test]
    fn test_fold_add_zero() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let zero = arena.alloc(Node::Scalar(0.0));
        let expr = arena.alloc(Node::Add(x, zero));

        let simplified = simplify(&mut arena, expr);

        assert_eq!(simplified, x);
    }

    #[test]
    fn test_fold_zero_plus_x() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let zero = arena.alloc(Node::Scalar(0.0));
        let expr = arena.alloc(Node::Add(zero, x));

        let simplified = simplify(&mut arena, expr);

        assert_eq!(simplified, x);
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact equality is the point: pins value-exactness
    fn conservative_simplify_may_normalise_the_sign_of_zero() {
        // `-0.0 + 0.0` is `+0.0`; folding the `+ 0` away yields `-0.0`. Both
        // are zero, so the fold is permitted.
        use crate::eval::CompiledExpr;

        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let neg_y = arena.alloc(Node::Neg(y));
        let zero = arena.alloc(Node::Scalar(0.0));
        let root = arena.alloc(Node::Add(neg_y, zero));

        let mut folded = arena.clone();
        let folded_root = simplify_with_mode(&mut folded, root, SimplifyMode::Conservative);

        let before = CompiledExpr::new(&arena, root);
        let after = CompiledExpr::new(&folded, folded_root);
        let mut s1 = vec![0.0; before.scratch_len()];
        let mut s2 = vec![0.0; after.scratch_len()];
        let a = before.eval(&mut s1, 0.0, &[0.0], &[], &[])[0];
        let b = after.eval(&mut s2, 0.0, &[0.0], &[], &[])[0];

        assert_eq!(a, b, "values must be equal");
        assert!(a == 0.0 && b == 0.0, "both must be zero");
        // The sign is explicitly NOT guaranteed; assert only that we know which
        // way it went, so a future change to the contract fails loudly here.
        assert!(a.is_sign_positive(), "unfolded -0.0 + 0.0 is +0.0");
        assert!(b.is_sign_negative(), "folded form keeps -0.0");
    }

    #[test]
    fn test_fold_sub_zero() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let zero = arena.alloc(Node::Scalar(0.0));
        let expr = arena.alloc(Node::Sub(x, zero));

        let simplified = simplify(&mut arena, expr);

        assert_eq!(simplified, x);
    }

    /// `(+0.0) - x -> -x` is sign-of-zero lossy, so conservative mode keeps the
    /// `Sub`; aggressive mode still folds it.
    #[test]
    fn test_fold_zero_minus_x() {
        let build = || {
            let mut arena = Arena::new();
            let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
            let zero = arena.alloc(Node::Scalar(0.0));
            let expr = arena.alloc(Node::Sub(zero, x));
            (arena, x, expr)
        };

        let (mut arena, _x, expr) = build();
        let conservative = simplify(&mut arena, expr);
        assert!(
            matches!(arena.get(conservative), Node::Sub(_, _)),
            "conservative mode must not turn (+0) - x into -x: for x = +0 that \
             yields -0 where the subtraction yields +0"
        );

        let (mut arena, x, expr) = build();
        let aggressive = simplify_with_mode(&mut arena, expr, SimplifyMode::Aggressive);
        match arena.get(aggressive) {
            Node::Neg(inner) => assert_eq!(*inner, x),
            other => panic!("aggressive mode should fold to Neg, got {other:?}"),
        }
    }

    /// `-0.0 - x` really is `-x` for every `x`, so that one folds even in
    /// conservative mode.
    #[test]
    fn test_fold_negative_zero_minus_x_is_exact() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let neg_zero = arena.alloc(Node::Scalar(-0.0));
        let expr = arena.alloc(Node::Sub(neg_zero, x));

        let simplified = simplify(&mut arena, expr);

        match arena.get(simplified) {
            Node::Neg(inner) => assert_eq!(*inner, x),
            other => panic!("expected Neg node, got {other:?}"),
        }
    }

    /// `x - (-0.0)` is `x + 0.0`, which loses a negative-zero `x`, so only the
    /// `+0.0` form may be folded away.
    #[test]
    fn test_x_minus_negative_zero_is_not_folded() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let neg_zero = arena.alloc(Node::Scalar(-0.0));
        let expr = arena.alloc(Node::Sub(x, neg_zero));

        let simplified = simplify(&mut arena, expr);

        assert!(
            matches!(arena.get(simplified), Node::Sub(_, _)),
            "x - (-0.0) must keep its Sub: for x = -0 it evaluates to +0"
        );
    }

    /// Aggressive mode ignores the sign of zero already (`x - x -> 0` below
    /// treats `Inf - Inf` as an acceptable NaN loss), so it should fold a
    /// vector `x - (-0.0)` too, matching `0 - x`'s existing aggressive-only
    /// arm. This also covers the broadcast shape (`x` a vector, the zero a
    /// scalar) that `simplify` folds without a shape guard.
    #[test]
    #[allow(clippy::float_cmp)] // exact equality is the point: pins value-exactness
    fn test_aggressive_fold_vector_minus_negative_zero() {
        use crate::eval::CompiledExpr;

        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let neg_zero = arena.alloc(Node::Scalar(-0.0));
        let expr = arena.alloc(Node::Sub(x, neg_zero));

        let mut folded_arena = arena.clone();
        let simplified = simplify_with_mode(&mut folded_arena, expr, SimplifyMode::Aggressive);
        assert_eq!(simplified, x);

        // Tape shape alone doesn't prove the fold is value-preserving; check
        // the trade is confined to a zero element's sign, others untouched.
        let before = CompiledExpr::new(&arena, expr);
        let after = CompiledExpr::new(&folded_arena, simplified);
        let mut s1 = vec![0.0; before.scratch_len()];
        let mut s2 = vec![0.0; after.scratch_len()];
        let y = [-0.0, 3.5];
        let unfolded = before.eval(&mut s1, 0.0, &y, &[], &[]).to_vec();
        let folded = after.eval(&mut s2, 0.0, &y, &[], &[]).to_vec();

        assert_eq!(unfolded, folded, "values must be equal element-wise");
        assert!(
            unfolded[0].is_sign_positive(),
            "x - (-0.0) turns -0.0 positive"
        );
        assert!(folded[0].is_sign_negative(), "the fold keeps -0.0's sign");
        assert_eq!(
            unfolded[1], 3.5,
            "a nonzero element passes through untouched"
        );
    }

    /// Regression for the shrunk `simplify_conservative_preserves_eval`
    /// counterexample: `sin((2 - 2) - (y - y))` is `sin(+0.0)`, and conservative
    /// simplification must not turn it into `sin(-(y - y))` = `sin(-0.0)`.
    #[test]
    fn test_conservative_simplify_preserves_sign_of_zero() {
        use crate::eval::CompiledExpr;

        let mut arena = Arena::new();
        let two_a = arena.alloc(Node::Scalar(2.0));
        let two_b = arena.alloc(Node::Scalar(2.0));
        let const_zero = arena.alloc(Node::Sub(two_a, two_b));
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y_again = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let state_zero = arena.alloc(Node::Sub(y, y_again));
        let difference = arena.alloc(Node::Sub(const_zero, state_zero));
        let root = arena.alloc(Node::Sin(difference));

        let before = CompiledExpr::new(&arena, root);
        let mut arena_copy = arena.clone();
        let simplified = simplify(&mut arena_copy, root);
        let after = CompiledExpr::new(&arena_copy, simplified);

        let mut s1 = vec![0.0; before.scratch_len()];
        let mut s2 = vec![0.0; after.scratch_len()];
        let a = before.eval(&mut s1, 0.0, &[0.1], &[], &[])[0];
        let b = after.eval(&mut s2, 0.0, &[0.1], &[], &[])[0];
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "conservative simplify changed the sign of zero: {a} vs {b}"
        );
        assert_eq!(a.to_bits(), 0.0_f64.to_bits(), "expected +0.0");
    }

    #[test]
    fn test_conservative_does_not_fold_x_minus_x() {
        // Conservative mode should NOT fold x - x -> 0 because Inf - Inf = NaN
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let expr = arena.alloc(Node::Sub(x, x));

        let simplified = simplify(&mut arena, expr);

        match arena.get(simplified) {
            Node::Sub(_, _) => {},
            _ => panic!("Conservative mode should not simplify x - x"),
        }
    }

    #[test]
    fn test_aggressive_folds_x_minus_x() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let expr = arena.alloc(Node::Sub(x, x));

        let simplified = simplify_with_mode(&mut arena, expr, SimplifyMode::Aggressive);

        match arena.get(simplified) {
            Node::Scalar(v) => assert!((*v).abs() < f64::EPSILON),
            _ => panic!("Aggressive mode should simplify x - x to 0"),
        }
    }

    #[test]
    fn test_fold_div_by_one() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let one = arena.alloc(Node::Scalar(1.0));
        let expr = arena.alloc(Node::Div(x, one));

        let simplified = simplify(&mut arena, expr);

        assert_eq!(simplified, x);
    }

    #[test]
    fn test_fold_neg_one_times_x() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let neg_one = arena.alloc(Node::Scalar(-1.0));
        let expr = arena.alloc(Node::Mul(neg_one, x));

        let simplified = simplify(&mut arena, expr);

        match arena.get(simplified) {
            Node::Neg(inner) => assert_eq!(*inner, x),
            _ => panic!("Expected Neg node"),
        }
    }

    #[test]
    fn test_fold_pow_zero() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let zero = arena.alloc(Node::Scalar(0.0));
        let expr = arena.alloc(Node::Pow(x, zero));

        let simplified = simplify(&mut arena, expr);

        match arena.get(simplified) {
            Node::Scalar(v) => assert!((*v - 1.0).abs() < f64::EPSILON),
            _ => panic!("Expected Scalar(1.0)"),
        }
    }

    #[test]
    fn test_fold_pow_one() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let one = arena.alloc(Node::Scalar(1.0));
        let expr = arena.alloc(Node::Pow(x, one));

        let simplified = simplify(&mut arena, expr);

        assert_eq!(simplified, x);
    }

    #[test]
    fn test_fold_one_pow_x() {
        let mut arena = Arena::new();
        let one = arena.alloc(Node::Scalar(1.0));
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let expr = arena.alloc(Node::Pow(one, x));

        let simplified = simplify(&mut arena, expr);

        assert_eq!(simplified, one);
    }

    #[test]
    fn test_double_negation() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let neg_x = arena.alloc(Node::Neg(x));
        let neg_neg_x = arena.alloc(Node::Neg(neg_x));

        let simplified = simplify(&mut arena, neg_neg_x);

        assert_eq!(simplified, x);
    }

    /// Evaluate the simplified form of `y0 ^ e` at `y`.
    fn eval_pow_of_state(e: f64, y: f64) -> f64 {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let exp = arena.alloc(Node::Scalar(e));
        let expr = arena.alloc(Node::Pow(x, exp));
        let simplified = simplify(&mut arena, expr);
        let compiled = crate::eval::CompiledExpr::new(&arena, simplified);
        let mut scratch = vec![0.0; compiled.scratch_len()];
        compiled.eval(&mut scratch, 0.0, &[y], &[], &[])[0]
    }

    #[test]
    fn test_int_pow_lowers_to_mul_chain() {
        // x^2 becomes Mul(x, x), not a runtime powf.
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let two = arena.alloc(Node::Scalar(2.0));
        let expr = arena.alloc(Node::Pow(x, two));
        let simplified = simplify(&mut arena, expr);
        match arena.get(simplified) {
            Node::Mul(a, b) => {
                assert_eq!(*a, x);
                assert_eq!(*b, x);
            },
            other => panic!("expected Mul(x, x), got {other:?}"),
        }
    }

    #[test]
    fn test_int_pow_chain_values() {
        for e in [2.0, 3.0, 4.0, -1.0, -2.0] {
            for y in [0.7, -1.3, 2.5] {
                let got = eval_pow_of_state(e, y);
                let want = y.powf(e);
                assert!(
                    (got - want).abs() <= 2.0 * f64::EPSILON * want.abs(),
                    "y={y}, e={e}: chain {got} != powf {want}"
                );
            }
        }
    }

    #[test]
    fn test_int_pow_chain_preserves_special_cases() {
        // The chains must keep IEEE special cases (the reason they are safe
        // in Conservative mode): signed zero, infinities, NaN.
        let cases = [
            (2.0, f64::NEG_INFINITY),
            (3.0, f64::NEG_INFINITY),
            (3.0, -0.0),
            (-1.0, 0.0),
            (-1.0, -0.0),
            (-1.0, f64::INFINITY),
            (-2.0, 0.0),
        ];
        for (e, y) in cases {
            let got = eval_pow_of_state(e, y);
            let want = y.powf(e);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "y={y}, e={e}: chain {got} != powf {want}"
            );
        }
        // NaN propagates (payload bits are not portable, so assert NaN-ness).
        for e in [2.0, -2.0] {
            assert!(eval_pow_of_state(e, f64::NAN).is_nan());
        }
    }

    #[test]
    fn test_non_integer_pow_stays_pow() {
        // No chain form: 0.5 and 1.3 must remain runtime Pow (sqrt rewrites
        // would change pow(-inf, 0.5) semantics, so simplify never emits them).
        for e in [0.5, 1.3, 5.0, -3.0] {
            let mut arena = Arena::new();
            let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
            let exp = arena.alloc(Node::Scalar(e));
            let expr = arena.alloc(Node::Pow(x, exp));
            let simplified = simplify(&mut arena, expr);
            match arena.get(simplified) {
                Node::Pow(_, _) => {},
                other => panic!("x^{e} must stay Pow, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_constant_fold_add() {
        let mut arena = Arena::new();
        let two = arena.alloc(Node::Scalar(2.0));
        let three = arena.alloc(Node::Scalar(3.0));
        let expr = arena.alloc(Node::Add(two, three));

        let simplified = simplify(&mut arena, expr);

        match arena.get(simplified) {
            Node::Scalar(v) => assert!((*v - 5.0).abs() < f64::EPSILON),
            _ => panic!("Expected Scalar(5.0)"),
        }
    }

    #[test]
    fn test_constant_fold_sub() {
        let mut arena = Arena::new();
        let five = arena.alloc(Node::Scalar(5.0));
        let three = arena.alloc(Node::Scalar(3.0));
        let expr = arena.alloc(Node::Sub(five, three));

        let simplified = simplify(&mut arena, expr);

        match arena.get(simplified) {
            Node::Scalar(v) => assert!((*v - 2.0).abs() < f64::EPSILON),
            _ => panic!("Expected Scalar(2.0)"),
        }
    }

    #[test]
    fn test_constant_fold_mul() {
        let mut arena = Arena::new();
        let three = arena.alloc(Node::Scalar(3.0));
        let four = arena.alloc(Node::Scalar(4.0));
        let expr = arena.alloc(Node::Mul(three, four));

        let simplified = simplify(&mut arena, expr);

        match arena.get(simplified) {
            Node::Scalar(v) => assert!((*v - 12.0).abs() < f64::EPSILON),
            _ => panic!("Expected Scalar(12.0)"),
        }
    }

    #[test]
    fn test_constant_fold_div() {
        let mut arena = Arena::new();
        let twelve = arena.alloc(Node::Scalar(12.0));
        let four = arena.alloc(Node::Scalar(4.0));
        let expr = arena.alloc(Node::Div(twelve, four));

        let simplified = simplify(&mut arena, expr);

        match arena.get(simplified) {
            Node::Scalar(v) => assert!((*v - 3.0).abs() < f64::EPSILON),
            _ => panic!("Expected Scalar(3.0)"),
        }
    }

    #[test]
    fn test_constant_fold_pow() {
        let mut arena = Arena::new();
        let two = arena.alloc(Node::Scalar(2.0));
        let three = arena.alloc(Node::Scalar(3.0));
        let expr = arena.alloc(Node::Pow(two, three));

        let simplified = simplify(&mut arena, expr);

        match arena.get(simplified) {
            Node::Scalar(v) => assert!((*v - 8.0).abs() < f64::EPSILON),
            _ => panic!("Expected Scalar(8.0)"),
        }
    }

    #[test]
    fn test_constant_fold_neg() {
        let mut arena = Arena::new();
        let five = arena.alloc(Node::Scalar(5.0));
        let expr = arena.alloc(Node::Neg(five));

        let simplified = simplify(&mut arena, expr);

        match arena.get(simplified) {
            Node::Scalar(v) => assert!((*v + 5.0).abs() < f64::EPSILON),
            _ => panic!("Expected Scalar(-5.0)"),
        }
    }

    #[test]
    fn test_constant_fold_unary_ops() {
        let mut arena = Arena::new();

        // Test sin(0) = 0
        let zero = arena.alloc(Node::Scalar(0.0));
        let sin_zero = arena.alloc(Node::Sin(zero));
        let simplified = simplify(&mut arena, sin_zero);
        match arena.get(simplified) {
            Node::Scalar(v) => assert!(v.abs() < f64::EPSILON),
            _ => panic!("Expected Scalar(0.0)"),
        }

        // Test cos(0) = 1
        let cos_zero = arena.alloc(Node::Cos(zero));
        let simplified = simplify(&mut arena, cos_zero);
        match arena.get(simplified) {
            Node::Scalar(v) => assert!((*v - 1.0).abs() < f64::EPSILON),
            _ => panic!("Expected Scalar(1.0)"),
        }

        // Test exp(0) = 1
        let exp_zero = arena.alloc(Node::Exp(zero));
        let simplified = simplify(&mut arena, exp_zero);
        match arena.get(simplified) {
            Node::Scalar(v) => assert!((*v - 1.0).abs() < f64::EPSILON),
            _ => panic!("Expected Scalar(1.0)"),
        }

        // Test sqrt(4) = 2
        let four = arena.alloc(Node::Scalar(4.0));
        let sqrt_four = arena.alloc(Node::Sqrt(four));
        let simplified = simplify(&mut arena, sqrt_four);
        match arena.get(simplified) {
            Node::Scalar(v) => assert!((*v - 2.0).abs() < f64::EPSILON),
            _ => panic!("Expected Scalar(2.0)"),
        }
    }

    #[test]
    fn test_nested_simplification() {
        // (x + 0) * 1 -> x
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let zero = arena.alloc(Node::Scalar(0.0));
        let one = arena.alloc(Node::Scalar(1.0));
        let x_plus_zero = arena.alloc(Node::Add(x, zero));
        let expr = arena.alloc(Node::Mul(x_plus_zero, one));

        let simplified = simplify(&mut arena, expr);

        assert_eq!(simplified, x);
    }

    #[test]
    fn test_nested_simplification_complex() {
        // ((x * 1) + 0) - (y * 0)
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let zero = arena.alloc(Node::Scalar(0.0));
        let one = arena.alloc(Node::Scalar(1.0));

        let x_times_one = arena.alloc(Node::Mul(x, one));
        let lhs = arena.alloc(Node::Add(x_times_one, zero));
        let y_times_zero = arena.alloc(Node::Mul(y, zero));
        let expr = arena.alloc(Node::Sub(lhs, y_times_zero));

        // Conservative keeps `y * 0` as a Mul, so the Sub survives.
        let simplified_conservative = simplify(&mut arena, expr);
        match arena.get(simplified_conservative) {
            Node::Sub(l, _r) => assert_eq!(*l, x),
            _ => panic!("Expected Sub node in conservative mode"),
        }

        let simplified_aggressive = simplify_with_mode(&mut arena, expr, SimplifyMode::Aggressive);
        assert_eq!(simplified_aggressive, x);
    }

    #[test]
    fn test_conservative_does_not_fold_zero_times_x() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let zero = arena.alloc(Node::Scalar(0.0));
        let expr = arena.alloc(Node::Mul(zero, x));

        let simplified = simplify(&mut arena, expr);

        match arena.get(simplified) {
            Node::Mul(_, _) => {},
            _ => panic!("Conservative mode should not simplify 0 * x"),
        }
    }

    #[test]
    fn test_aggressive_folds_zero_times_x() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let zero = arena.alloc(Node::Scalar(0.0));
        let expr = arena.alloc(Node::Mul(zero, x));

        let simplified = simplify_with_mode(&mut arena, expr, SimplifyMode::Aggressive);

        match arena.get(simplified) {
            Node::Scalar(v) => assert!(v.abs() < f64::EPSILON),
            _ => panic!("Aggressive mode should simplify 0 * x to 0"),
        }
    }

    #[test]
    fn test_aggressive_folds_x_times_zero() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let zero = arena.alloc(Node::Scalar(0.0));
        let expr = arena.alloc(Node::Mul(x, zero));

        let simplified = simplify_with_mode(&mut arena, expr, SimplifyMode::Aggressive);

        match arena.get(simplified) {
            Node::Scalar(v) => assert!(v.abs() < f64::EPSILON),
            _ => panic!("Aggressive mode should simplify x * 0 to 0"),
        }
    }

    #[test]
    fn test_aggressive_does_not_fold_tiny_scalar_coefficient() {
        // 1e-17 is below f64::EPSILON but is a genuine nonzero coefficient;
        // it must NOT be folded away as if it were exactly zero.
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let tiny = arena.alloc(Node::Scalar(1e-17));
        let expr = arena.alloc(Node::Mul(tiny, x));

        let simplified = simplify_with_mode(&mut arena, expr, SimplifyMode::Aggressive);

        match arena.get(simplified) {
            Node::Mul(_, _) => {}, // expected: coefficient survives
            other => panic!("tiny nonzero coefficient must survive; got {other:?}"),
        }
    }

    #[test]
    fn test_conservative_does_not_fold_zero_div_x() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let zero = arena.alloc(Node::Scalar(0.0));
        let expr = arena.alloc(Node::Div(zero, x));

        let simplified = simplify(&mut arena, expr);

        match arena.get(simplified) {
            Node::Div(_, _) => {},
            _ => panic!("Conservative mode should not simplify 0 / x"),
        }
    }

    #[test]
    fn test_aggressive_folds_zero_div_x() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let zero = arena.alloc(Node::Scalar(0.0));
        let expr = arena.alloc(Node::Div(zero, x));

        let simplified = simplify_with_mode(&mut arena, expr, SimplifyMode::Aggressive);

        match arena.get(simplified) {
            Node::Scalar(v) => assert!(v.abs() < f64::EPSILON),
            _ => panic!("Aggressive mode should simplify 0 / x to 0"),
        }
    }

    #[test]
    fn test_conservative_does_not_fold_x_div_x() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let expr = arena.alloc(Node::Div(x, x));

        let simplified = simplify(&mut arena, expr);

        match arena.get(simplified) {
            Node::Div(_, _) => {},
            _ => panic!("Conservative mode should not simplify x / x"),
        }
    }

    #[test]
    fn test_aggressive_folds_x_div_x() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let expr = arena.alloc(Node::Div(x, x));

        let simplified = simplify_with_mode(&mut arena, expr, SimplifyMode::Aggressive);

        match arena.get(simplified) {
            Node::Scalar(v) => assert!((*v - 1.0).abs() < f64::EPSILON),
            _ => panic!("Aggressive mode should simplify x / x to 1"),
        }
    }

    #[test]
    fn test_aggressive_zero_times_vector_preserves_shape() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let zero = arena.alloc(Node::Scalar(0.0));
        let expr = arena.alloc(Node::Mul(zero, x));

        let simplified = simplify_with_mode(&mut arena, expr, SimplifyMode::Aggressive);

        assert_eq!(arena.get(simplified), &Node::ZeroVector { len: 3 });
    }

    #[test]
    fn test_aggressive_vector_times_zero_preserves_shape() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let zero = arena.alloc(Node::Scalar(0.0));
        let expr = arena.alloc(Node::Mul(x, zero));

        let simplified = simplify_with_mode(&mut arena, expr, SimplifyMode::Aggressive);

        assert_eq!(arena.get(simplified), &Node::ZeroVector { len: 3 });
    }

    #[test]
    fn test_aggressive_vector_sub_self_preserves_shape() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let expr = arena.alloc(Node::Sub(x, x));

        let simplified = simplify_with_mode(&mut arena, expr, SimplifyMode::Aggressive);

        assert_eq!(arena.get(simplified), &Node::ZeroVector { len: 3 });
    }

    #[test]
    fn test_aggressive_zero_div_vector_preserves_shape() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let zero = arena.alloc(Node::Scalar(0.0));
        let expr = arena.alloc(Node::Div(zero, x));

        let simplified = simplify_with_mode(&mut arena, expr, SimplifyMode::Aggressive);

        assert_eq!(arena.get(simplified), &Node::ZeroVector { len: 3 });
    }

    #[test]
    fn test_aggressive_vector_div_self_not_folded_to_scalar() {
        // x / x on a vector is a ones *vector*; without a shaped ones node
        // the fold must not fire.
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let expr = arena.alloc(Node::Div(x, x));

        let simplified = simplify_with_mode(&mut arena, expr, SimplifyMode::Aggressive);

        match arena.get(simplified) {
            Node::Div(_, _) => {},
            other => panic!("vector x / x must stay Div, got {other:?}"),
        }
    }

    #[test]
    fn test_vector_pow_zero_not_folded_to_scalar() {
        // x^0 on a vector is a ones *vector*; the fold must not fire.
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let zero = arena.alloc(Node::Scalar(0.0));
        let expr = arena.alloc(Node::Pow(x, zero));

        let simplified = simplify(&mut arena, expr);

        match arena.get(simplified) {
            Node::Pow(_, _) => {},
            other => panic!("vector x^0 must stay Pow, got {other:?}"),
        }
    }

    #[test]
    fn test_one_pow_vector_not_folded_to_scalar() {
        // 1^x broadcasts to len(x); folding to scalar 1 narrows the output.
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let one = arena.alloc(Node::Scalar(1.0));
        let expr = arena.alloc(Node::Pow(one, x));

        let simplified = simplify(&mut arena, expr);

        match arena.get(simplified) {
            Node::Pow(_, _) => {},
            other => panic!("1^x with vector x must stay Pow, got {other:?}"),
        }
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact fold/runtime agreement is the property under test
    fn test_erf_fold_zero_is_exact() {
        let mut arena = Arena::new();
        let zero = arena.alloc(Node::Scalar(0.0));
        let erf = arena.alloc(Node::Erf(zero));

        let simplified = simplify(&mut arena, erf);

        match arena.get(simplified) {
            Node::Scalar(v) => assert_eq!(*v, 0.0, "erf(0) must fold to exactly 0"),
            other => panic!("Expected folded Scalar, got {other:?}"),
        }
    }

    #[test]
    fn test_erf_fold_bitwise_matches_runtime() {
        let mut arena = Arena::new();
        let c = arena.alloc(Node::Scalar(0.5));
        let erf = arena.alloc(Node::Erf(c));

        let compiled = crate::eval::CompiledExpr::new(&arena, erf);
        let mut scratch = vec![0.0; compiled.scratch_len()];
        let runtime = compiled.eval(&mut scratch, 0.0, &[], &[], &[])[0];

        let simplified = simplify(&mut arena, erf);

        match arena.get(simplified) {
            Node::Scalar(v) => assert_eq!(v.to_bits(), runtime.to_bits()),
            other => panic!("Expected folded Scalar, got {other:?}"),
        }
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact fold/runtime agreement is the property under test
    fn test_sign_fold_zero_matches_runtime() {
        // Runtime sign(0) = 0; f64::signum(0.0) is 1, so the fold must not use it.
        let mut arena = Arena::new();
        let zero = arena.alloc(Node::Scalar(0.0));
        let sign = arena.alloc(Node::Sign(zero));

        let simplified = simplify(&mut arena, sign);

        match arena.get(simplified) {
            Node::Scalar(v) => assert_eq!(*v, 0.0, "sign(0) must fold to 0"),
            other => panic!("Expected folded Scalar, got {other:?}"),
        }
    }

    #[test]
    fn test_simplify_index() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 10 });
        let zero = arena.alloc(Node::Scalar(0.0));
        let x_plus_zero = arena.alloc(Node::Add(x, zero));
        let expr = arena.alloc(Node::Index {
            child: x_plus_zero,
            start: 0,
            end: 5,
        });

        let simplified = simplify(&mut arena, expr);

        match arena.get(simplified) {
            Node::Index { child, start, end } => {
                assert_eq!(*child, x);
                assert_eq!(*start, 0);
                assert_eq!(*end, 5);
            },
            _ => panic!("Expected Index node"),
        }
    }

    #[test]
    fn test_simplify_concat() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 5 });
        let y = arena.alloc(Node::StateVector { start: 5, end: 10 });
        let zero = arena.alloc(Node::Scalar(0.0));
        let x_plus_zero = arena.alloc(Node::Add(x, zero));
        let expr = arena.alloc(Node::Concat(vec![x_plus_zero, y]));

        let simplified = simplify(&mut arena, expr);

        match arena.get(simplified) {
            Node::Concat(children) => {
                assert_eq!(children.len(), 2);
                assert_eq!(children[0], x);
                assert_eq!(children[1], y);
            },
            _ => panic!("Expected Concat node"),
        }
    }

    #[test]
    fn test_memoization() {
        // Ensure same subexpression is not processed twice
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let zero = arena.alloc(Node::Scalar(0.0));
        let x_plus_zero = arena.alloc(Node::Add(x, zero));
        // Use x_plus_zero twice
        let expr = arena.alloc(Node::Add(x_plus_zero, x_plus_zero));

        let simplified = simplify(&mut arena, expr);

        // Should simplify to x + x
        match arena.get(simplified) {
            Node::Add(l, r) => {
                assert_eq!(*l, x);
                assert_eq!(*r, x);
            },
            _ => panic!("Expected Add node"),
        }
    }

    #[test]
    fn test_leaf_nodes_unchanged() {
        let mut arena = Arena::new();

        let scalar = arena.alloc(Node::Scalar(42.0));
        assert_eq!(simplify(&mut arena, scalar), scalar);

        let sv = arena.alloc(Node::StateVector { start: 0, end: 10 });
        assert_eq!(simplify(&mut arena, sv), sv);

        let time = arena.alloc(Node::Time);
        assert_eq!(simplify(&mut arena, time), time);

        let param = arena.alloc(Node::InputParameter {
            name: "k".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        assert_eq!(simplify(&mut arena, param), param);
    }

    #[test]
    fn test_cse_deduplicates_identical_subtrees() {
        // (x + y) * (x + y)
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let sum1 = arena.alloc(Node::Add(x, y));
        let sum2 = arena.alloc(Node::Add(x, y));
        let product = arena.alloc(Node::Mul(sum1, sum2));

        let (new_arena, new_root) = cse(&arena, product);

        // 5 nodes in, 4 out: the two identical `x + y` subtrees collapse.
        assert_eq!(new_arena.len(), 4);

        match new_arena.get(new_root) {
            Node::Mul(l, r) => {
                assert_eq!(l, r, "Both operands should be the same deduplicated node");
            },
            _ => panic!("Expected Mul node"),
        }
    }

    #[test]
    fn test_cse_preserves_different_subtrees() {
        // Build: (x + y) * (x - y)
        // These are different, should not be merged
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let sum = arena.alloc(Node::Add(x, y));
        let diff = arena.alloc(Node::Sub(x, y));
        let product = arena.alloc(Node::Mul(sum, diff));

        let (new_arena, new_root) = cse(&arena, product);

        // All 5 nodes should be preserved (x, y shared, but sum and diff are different)
        assert_eq!(new_arena.len(), 5);

        match new_arena.get(new_root) {
            Node::Mul(l, r) => {
                assert_ne!(l, r, "Different operations should not be merged");
            },
            _ => panic!("Expected Mul node"),
        }
    }

    #[test]
    fn test_cse_handles_scalars() {
        // Build: 2.0 + 2.0
        // Both scalars are identical and should be deduplicated
        let mut arena = Arena::new();
        let two1 = arena.alloc(Node::Scalar(2.0));
        let two2 = arena.alloc(Node::Scalar(2.0));
        let sum = arena.alloc(Node::Add(two1, two2));

        let (new_arena, new_root) = cse(&arena, sum);

        // Should have: one scalar(2.0), one Add = 2 nodes
        assert_eq!(new_arena.len(), 2);

        match new_arena.get(new_root) {
            Node::Add(l, r) => {
                assert_eq!(l, r, "Identical scalars should be deduplicated");
            },
            _ => panic!("Expected Add node"),
        }
    }

    #[test]
    fn test_cse_nested_common_subexpressions() {
        // Build: sin(x + y) + sin(x + y)
        // Both sin(x + y) should be deduplicated
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let sum1 = arena.alloc(Node::Add(x, y));
        let sum2 = arena.alloc(Node::Add(x, y));
        let sin1 = arena.alloc(Node::Sin(sum1));
        let sin2 = arena.alloc(Node::Sin(sum2));
        let result = arena.alloc(Node::Add(sin1, sin2));

        let (new_arena, _new_root) = cse(&arena, result);

        // Should have: x, y, Add(x,y), Sin(Add), Add(Sin,Sin) = 5 nodes
        // (both sum1/sum2 collapse to one, both sin1/sin2 collapse to one)
        assert_eq!(new_arena.len(), 5);
    }

    #[test]
    fn test_dce_removes_unreachable_nodes() {
        // Build arena with some unreachable nodes
        let mut arena = Arena::new();
        let _unreachable1 = arena.alloc(Node::Scalar(999.0)); // Not referenced
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let _unreachable2 = arena.alloc(Node::Scalar(888.0)); // Not referenced
        let one = arena.alloc(Node::Scalar(1.0));
        let root = arena.alloc(Node::Add(x, one));

        // Original arena has 5 nodes
        assert_eq!(arena.len(), 5);

        let (new_arena, new_root) = dce(&arena, root);

        // DCE should remove unreachable nodes, keeping only x, one, and Add
        assert_eq!(new_arena.len(), 3);

        // Verify structure is preserved
        match new_arena.get(new_root) {
            Node::Add(l, r) => {
                match new_arena.get(*l) {
                    Node::StateVector { start: 0, end: 1 } => {},
                    _ => panic!("Expected StateVector"),
                }
                match new_arena.get(*r) {
                    Node::Scalar(v) => assert!((*v - 1.0).abs() < f64::EPSILON),
                    _ => panic!("Expected Scalar(1.0)"),
                }
            },
            _ => panic!("Expected Add node"),
        }
    }

    #[test]
    fn test_dce_preserves_all_reachable_nodes() {
        // Build a tree where all nodes are reachable
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let sum = arena.alloc(Node::Add(x, y));
        let neg = arena.alloc(Node::Neg(sum));
        let root = arena.alloc(Node::Abs(neg));

        assert_eq!(arena.len(), 5);

        let (new_arena, _new_root) = dce(&arena, root);

        // All nodes are reachable, so none should be removed
        assert_eq!(new_arena.len(), 5);
    }

    #[test]
    fn test_dce_handles_diamond_pattern() {
        // Build diamond: root = (x + y) * (x + y) where x and y are shared
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let sum = arena.alloc(Node::Add(x, y));
        let root = arena.alloc(Node::Mul(sum, sum)); // Same node referenced twice

        assert_eq!(arena.len(), 4);

        let (new_arena, new_root) = dce(&arena, root);

        // All 4 nodes are reachable
        assert_eq!(new_arena.len(), 4);

        // Structure should be preserved with shared reference
        match new_arena.get(new_root) {
            Node::Mul(l, r) => {
                assert_eq!(l, r, "Both operands should reference the same Add node");
            },
            _ => panic!("Expected Mul node"),
        }
    }

    #[test]
    fn test_cse_then_dce_pipeline() {
        let mut arena = Arena::new();
        let _garbage = arena.alloc(Node::Scalar(123.0));
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let sum1 = arena.alloc(Node::Add(x, y));
        let sum2 = arena.alloc(Node::Add(x, y));
        let _more_garbage = arena.alloc(Node::Scalar(456.0));
        let product = arena.alloc(Node::Mul(sum1, sum2));

        // Original: 7 nodes (2 garbage + 2 duplicate sums + x + y + mul)
        assert_eq!(arena.len(), 7);

        // CSE first (works on reachable nodes from root)
        let (cse_arena, cse_root) = cse(&arena, product);

        // After CSE: x, y, sum (deduplicated), mul = 4 nodes
        // Note: CSE only processes reachable nodes, so garbage is already gone
        assert_eq!(cse_arena.len(), 4);

        // DCE (should be a no-op after CSE since CSE only copies reachable)
        let (final_arena, _final_root) = dce(&cse_arena, cse_root);
        assert_eq!(final_arena.len(), 4);
    }

    #[test]
    fn test_simplify_reduce_arg_select_zero_basis_folds() {
        // basis all-zero -> selecting any element is 0 -> Scalar(0.0)
        let mut arena = Arena::new();
        let picker = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let basis = arena.alloc(Node::ZeroVector { len: 3 });
        let node = arena.alloc(Node::ReduceArgSelect {
            basis,
            picker,
            is_max: true,
        });
        let simplified = simplify(&mut arena, node);
        match arena.get(simplified) {
            Node::Scalar(v) => assert!(v.abs() < f64::EPSILON),
            other => panic!("expected Scalar(0.0), got {other:?}"),
        }
    }

    #[test]
    fn test_simplify_reduce_arg_select_scalar_zero_basis_folds() {
        // width-1 case: basis is Scalar(0.0) -> fold to Scalar(0.0)
        let mut arena = Arena::new();
        let picker = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let basis = arena.alloc(Node::Scalar(0.0));
        let node = arena.alloc(Node::ReduceArgSelect {
            basis,
            picker,
            is_max: true,
        });
        let simplified = simplify(&mut arena, node);
        match arena.get(simplified) {
            Node::Scalar(v) => assert!(v.abs() < f64::EPSILON),
            other => panic!("expected Scalar(0.0), got {other:?}"),
        }
    }
}
