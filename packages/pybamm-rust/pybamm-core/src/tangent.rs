//! Symbolic differentiation for forward-mode automatic differentiation.
//!
//! Rewrites a primal expression into one that evaluates a Jacobian-vector
//! product: `TangentStateVector` and `TangentParameter` nodes stand in for the
//! differentiated variable and read their seed at evaluation time, so one
//! compiled tape serves every direction.

use std::collections::HashSet;
use std::collections::hash_map::RandomState;

use crate::arena::{Arena, NodeId, NodeMap};
use crate::ir::infer_sizes;
use crate::node::{InterpolantData, Node};

/// Allocate a zero matching a primal subtree's structural width.
///
/// A zero derivative must keep the width of the node it replaces. A bare
/// `Scalar(0.0)` collapses a vector derivative to length 1, shifting every
/// downstream `Concat` offset and truncating the output; `len == 0` (a
/// pure-algebraic model's empty `concatenated_rhs`) would widen to 1 and shift
/// the same offsets the other way, so it stays a `ZeroVector`.
fn zero_of_width(arena: &mut Arena, len: usize) -> NodeId {
    if len == 1 {
        arena.alloc(Node::Scalar(0.0))
    } else {
        arena.alloc(Node::ZeroVector { len })
    }
}

/// Output width of every node in `arena`, indexed by `NodeId::index()`.
///
/// Computed before differentiation mutates the arena. The recursion only ever
/// queries primal nodes, so appending tangent nodes cannot invalidate it.
fn primal_widths(arena: &Arena, root: NodeId) -> Vec<usize> {
    let order = arena.topological_order(root);
    infer_sizes(arena, &order)
}

/// Which variables carry a tangent; every other leaf differentiates to zero.
///
/// Also names the axis a Jacobian differentiates along, so `jacobian` re-exports
/// it rather than defining a second enum over the same two cases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiffTarget {
    /// `d/dy`: seed `StateVector` nodes from `TangentInputs::dy`; Jacobian
    /// columns are states.
    States,
    /// `d/dp`: seed `InputParameter` nodes from `TangentInputs::dp`; Jacobian
    /// columns are parameters.
    Params,
}

/// Build the tangent expression for `(d expr/dy) @ dy`.
///
/// `StateVector` nodes become `TangentStateVector` nodes that read `dy` at
/// evaluation time; parameters and time are constants.
#[must_use]
pub fn tangent_wrt_states(arena: &mut Arena, expr: NodeId) -> NodeId {
    let mut memo: NodeMap<NodeId> = NodeMap::new(arena.len());
    let no_filter: Option<&HashSet<usize, RandomState>> = None;
    let widths = primal_widths(arena, expr);
    differentiate(
        arena,
        expr,
        DiffTarget::States,
        &mut memo,
        no_filter,
        &widths,
    )
}

/// Build the tangent expression for `(d expr/dp) @ dp`.
///
/// `InputParameter` nodes become `TangentParameter` nodes that read `dp` at
/// evaluation time; states and time are constants.
#[must_use]
pub fn tangent_wrt_params(arena: &mut Arena, expr: NodeId) -> NodeId {
    let mut memo: NodeMap<NodeId> = NodeMap::new(arena.len());
    let no_filter: Option<&HashSet<usize, RandomState>> = None;
    let widths = primal_widths(arena, expr);
    differentiate(
        arena,
        expr,
        DiffTarget::Params,
        &mut memo,
        no_filter,
        &widths,
    )
}

/// Build the tangent expression for `(d expr/dy) @ dy`, restricted to
/// `StateVector` nodes whose index range overlaps `active_indices`.
///
/// State vectors outside the active set are constants. Overlap is
/// all-or-nothing: a node straddling the boundary is seeded over its whole
/// range. Used for the algebraic Jacobian sub-block.
#[must_use]
pub fn tangent_wrt_subset<S: ::std::hash::BuildHasher>(
    arena: &mut Arena,
    expr: NodeId,
    active_indices: &HashSet<usize, S>,
) -> NodeId {
    let mut memo: NodeMap<NodeId> = NodeMap::new(arena.len());
    let widths = primal_widths(arena, expr);
    differentiate(
        arena,
        expr,
        DiffTarget::States,
        &mut memo,
        Some(active_indices),
        &widths,
    )
}

/// The literal value of `id`, or `None` for any other node — nothing is folded,
/// so a constant-valued subtree still reads as non-scalar here.
fn get_scalar(arena: &Arena, id: NodeId) -> Option<f64> {
    match arena.get(id) {
        Node::Scalar(v) => Some(*v),
        _ => None,
    }
}

/// Whether `v` is a whole number, to an absolute tolerance of `f64::EPSILON`.
fn is_integer(v: f64) -> bool {
    (v.round() - v).abs() < f64::EPSILON
}

/// Differentiate `id`, memoising one derivative per primal node so a shared
/// subtree is differentiated once.
///
/// `widths` carries the primal output widths that zero derivatives must match.
#[allow(clippy::match_same_arms, clippy::branches_sharing_code)]
fn differentiate<S: ::std::hash::BuildHasher>(
    arena: &mut Arena,
    id: NodeId,
    mode: DiffTarget,
    memo: &mut NodeMap<NodeId>,
    state_filter: Option<&HashSet<usize, S>>,
    widths: &[usize],
) -> NodeId {
    if let Some(&cached) = memo.get(id) {
        return cached;
    }

    let width = widths.get(id.index()).copied().unwrap_or(1);

    let result = match arena.get(id).clone() {
        // Constants - derivative is 0
        Node::Scalar(_)
        | Node::Array(_)
        | Node::ZeroVector { .. }
        | Node::SparseMatrix(_)
        | Node::Time => zero_of_width(arena, width),

        // d(y')/dy is the mass matrix, which the solver supplies separately.
        Node::StateVectorDot { .. } => zero_of_width(arena, width),

        // Only first order: a tangent node has no tangent of its own.
        Node::TangentStateVector { .. } | Node::TangentParameter { .. } => {
            zero_of_width(arena, width)
        },

        // Active variables
        Node::StateVector { start, end } => {
            if mode == DiffTarget::States {
                let active =
                    state_filter.is_none_or(|filter| (start..end).any(|i| filter.contains(&i)));
                if active {
                    arena.alloc(Node::TangentStateVector { start, end })
                } else {
                    zero_of_width(arena, width)
                }
            } else {
                // dy/dp = 0
                zero_of_width(arena, width)
            }
        },

        Node::InputParameter {
            index,
            width: param_width,
            ..
        } => {
            if mode == DiffTarget::Params {
                // dp/dp = tangent_p, replicated to the packed width so a width-1
                // tangent under an Index slice can't read outside its buffer slot.
                let tangent = arena.alloc(Node::TangentParameter { index });
                if param_width > 1 {
                    arena.alloc(Node::Concat(vec![tangent; param_width]))
                } else {
                    tangent
                }
            } else {
                // dp/dy = 0
                zero_of_width(arena, width)
            }
        },

        // Binary operations
        Node::Add(a, b) => {
            // d(a + b) = da + db
            let da = differentiate(arena, a, mode, memo, state_filter, widths);
            let db = differentiate(arena, b, mode, memo, state_filter, widths);
            arena.alloc(Node::Add(da, db))
        },

        Node::Sub(a, b) => {
            // d(a - b) = da - db
            let da = differentiate(arena, a, mode, memo, state_filter, widths);
            let db = differentiate(arena, b, mode, memo, state_filter, widths);
            arena.alloc(Node::Sub(da, db))
        },

        Node::Mul(a, b) => {
            // d(a * b) = a * db + da * b (product rule)
            let da = differentiate(arena, a, mode, memo, state_filter, widths);
            let db = differentiate(arena, b, mode, memo, state_filter, widths);
            let a_db = arena.alloc(Node::Mul(a, db));
            let da_b = arena.alloc(Node::Mul(da, b));
            arena.alloc(Node::Add(a_db, da_b))
        },

        Node::Div(a, b) => {
            // d(a / b) = (da * b - a * db) / b^2 (quotient rule)
            let da = differentiate(arena, a, mode, memo, state_filter, widths);
            let db = differentiate(arena, b, mode, memo, state_filter, widths);
            let da_b = arena.alloc(Node::Mul(da, b));
            let a_db = arena.alloc(Node::Mul(a, db));
            let numer = arena.alloc(Node::Sub(da_b, a_db));
            let two = arena.alloc(Node::Scalar(2.0));
            let b_sq = arena.alloc(Node::Pow(b, two));
            arena.alloc(Node::Div(numer, b_sq))
        },

        Node::Pow(base, exp) => {
            // A constant exponent takes the power rule, which holds for a
            // negative base; a varying one needs log(base).
            if let Some(n) = get_scalar(arena, exp) {
                // The integer test only changes which node carries the exponent.
                if is_integer(n) {
                    // Integer power: d(a^n) = n * a^(n-1) * da
                    let da = differentiate(arena, base, mode, memo, state_filter, widths);
                    let n_scalar = arena.alloc(Node::Scalar(n));
                    let n_minus_1 = arena.alloc(Node::Scalar(n - 1.0));
                    let base_pow_nm1 = arena.alloc(Node::Pow(base, n_minus_1));
                    let n_times_pow = arena.alloc(Node::Mul(n_scalar, base_pow_nm1));
                    arena.alloc(Node::Mul(n_times_pow, da))
                } else {
                    // Non-integer constant exponent: d(a^b) = b * a^(b-1) * da
                    let da = differentiate(arena, base, mode, memo, state_filter, widths);
                    let b_minus_1 = arena.alloc(Node::Scalar(n - 1.0));
                    let base_pow_bm1 = arena.alloc(Node::Pow(base, b_minus_1));
                    let b_times_pow = arena.alloc(Node::Mul(exp, base_pow_bm1));
                    arena.alloc(Node::Mul(b_times_pow, da))
                }
            } else {
                // General case: d(a^b) = a^b * (b * da/a + log(a) * db)
                let da = differentiate(arena, base, mode, memo, state_filter, widths);
                let db = differentiate(arena, exp, mode, memo, state_filter, widths);
                let a_pow_b = arena.alloc(Node::Pow(base, exp));
                let log_a = arena.alloc(Node::Log(base));
                let da_over_a = arena.alloc(Node::Div(da, base));
                let b_da_over_a = arena.alloc(Node::Mul(exp, da_over_a));
                let log_a_db = arena.alloc(Node::Mul(log_a, db));
                let inner = arena.alloc(Node::Add(b_da_over_a, log_a_db));
                arena.alloc(Node::Mul(a_pow_b, inner))
            }
        },

        // Unary operations
        Node::Neg(a) => {
            let da = differentiate(arena, a, mode, memo, state_filter, widths);
            arena.alloc(Node::Neg(da))
        },

        Node::Abs(a) => {
            // d(|a|) = sign(a) * da
            let da = differentiate(arena, a, mode, memo, state_filter, widths);
            let sign_a = arena.alloc(Node::Sign(a));
            arena.alloc(Node::Mul(sign_a, da))
        },

        Node::Sqrt(a) => {
            // d(sqrt(a)) = da / (2 * sqrt(a))
            let da = differentiate(arena, a, mode, memo, state_filter, widths);
            let two = arena.alloc(Node::Scalar(2.0));
            let sqrt_a = arena.alloc(Node::Sqrt(a));
            let two_sqrt_a = arena.alloc(Node::Mul(two, sqrt_a));
            arena.alloc(Node::Div(da, two_sqrt_a))
        },

        Node::Exp(a) => {
            // d(exp(a)) = exp(a) * da
            let da = differentiate(arena, a, mode, memo, state_filter, widths);
            let exp_a = arena.alloc(Node::Exp(a));
            arena.alloc(Node::Mul(exp_a, da))
        },

        Node::Log(a) => {
            // d(log(a)) = da / a
            let da = differentiate(arena, a, mode, memo, state_filter, widths);
            arena.alloc(Node::Div(da, a))
        },

        Node::Sin(a) => {
            // d(sin(a)) = cos(a) * da
            let da = differentiate(arena, a, mode, memo, state_filter, widths);
            let cos_a = arena.alloc(Node::Cos(a));
            arena.alloc(Node::Mul(cos_a, da))
        },

        Node::Cos(a) => {
            // d(cos(a)) = -sin(a) * da
            let da = differentiate(arena, a, mode, memo, state_filter, widths);
            let sin_a = arena.alloc(Node::Sin(a));
            let neg_sin_a = arena.alloc(Node::Neg(sin_a));
            arena.alloc(Node::Mul(neg_sin_a, da))
        },

        Node::Tanh(a) => {
            // d(tanh(a)) = (1 - tanh(a)^2) * da
            let da = differentiate(arena, a, mode, memo, state_filter, widths);
            let tanh_a = arena.alloc(Node::Tanh(a));
            let one = arena.alloc(Node::Scalar(1.0));
            let two = arena.alloc(Node::Scalar(2.0));
            let tanh_sq = arena.alloc(Node::Pow(tanh_a, two));
            let one_minus_tanh_sq = arena.alloc(Node::Sub(one, tanh_sq));
            arena.alloc(Node::Mul(one_minus_tanh_sq, da))
        },

        Node::Sinh(a) => {
            // d(sinh(a)) = cosh(a) * da
            let da = differentiate(arena, a, mode, memo, state_filter, widths);
            let cosh_a = arena.alloc(Node::Cosh(a));
            arena.alloc(Node::Mul(cosh_a, da))
        },

        Node::Cosh(a) => {
            // d(cosh(a)) = sinh(a) * da
            let da = differentiate(arena, a, mode, memo, state_filter, widths);
            let sinh_a = arena.alloc(Node::Sinh(a));
            arena.alloc(Node::Mul(sinh_a, da))
        },

        Node::Arcsinh(a) => {
            // d(arcsinh(a)) = da / sqrt(1 + a^2)
            let da = differentiate(arena, a, mode, memo, state_filter, widths);
            let one = arena.alloc(Node::Scalar(1.0));
            let two = arena.alloc(Node::Scalar(2.0));
            let a_sq = arena.alloc(Node::Pow(a, two));
            let one_plus_a_sq = arena.alloc(Node::Add(one, a_sq));
            let sqrt_denom = arena.alloc(Node::Sqrt(one_plus_a_sq));
            arena.alloc(Node::Div(da, sqrt_denom))
        },

        Node::Arctan(a) => {
            // d(arctan(a)) = da / (1 + a^2)
            let da = differentiate(arena, a, mode, memo, state_filter, widths);
            let one = arena.alloc(Node::Scalar(1.0));
            let two = arena.alloc(Node::Scalar(2.0));
            let a_sq = arena.alloc(Node::Pow(a, two));
            let one_plus_a_sq = arena.alloc(Node::Add(one, a_sq));
            arena.alloc(Node::Div(da, one_plus_a_sq))
        },

        Node::Erf(a) => {
            // d(erf(a)) = 2/sqrt(pi) * exp(-a^2) * da
            let da = differentiate(arena, a, mode, memo, state_filter, widths);
            let two_over_sqrt_pi = arena.alloc(Node::Scalar(2.0 / std::f64::consts::PI.sqrt()));
            let neg_one = arena.alloc(Node::Scalar(-1.0));
            let two = arena.alloc(Node::Scalar(2.0));
            let a_sq = arena.alloc(Node::Pow(a, two));
            let neg_a_sq = arena.alloc(Node::Mul(neg_one, a_sq));
            let exp_neg_a_sq = arena.alloc(Node::Exp(neg_a_sq));
            let coeff = arena.alloc(Node::Mul(two_over_sqrt_pi, exp_neg_a_sq));
            arena.alloc(Node::Mul(coeff, da))
        },

        // Step functions: zero away from the jumps, which are not modelled.
        Node::Sign(_) | Node::Floor(_) | Node::Ceiling(_) => zero_of_width(arena, width),
        Node::EqualHeaviside(_, _) | Node::NotEqualHeaviside(_, _) | Node::Equality(_, _) => {
            zero_of_width(arena, width)
        },

        Node::Minimum(a, b) => {
            // Subgradient: the (a <= b) indicator picks da, its complement db.
            let da = differentiate(arena, a, mode, memo, state_filter, widths);
            let db = differentiate(arena, b, mode, memo, state_filter, widths);
            let selector = arena.alloc(Node::EqualHeaviside(a, b));
            let one = arena.alloc(Node::Scalar(1.0));
            let one_minus_sel = arena.alloc(Node::Sub(one, selector));
            let sel_da = arena.alloc(Node::Mul(selector, da));
            let not_sel_db = arena.alloc(Node::Mul(one_minus_sel, db));
            arena.alloc(Node::Add(sel_da, not_sel_db))
        },

        Node::Maximum(a, b) => {
            // Same, with the operands swapped: EqualHeaviside(b, a) is (a >= b).
            let da = differentiate(arena, a, mode, memo, state_filter, widths);
            let db = differentiate(arena, b, mode, memo, state_filter, widths);
            let selector = arena.alloc(Node::EqualHeaviside(b, a));
            let one = arena.alloc(Node::Scalar(1.0));
            let one_minus_sel = arena.alloc(Node::Sub(one, selector));
            let sel_da = arena.alloc(Node::Mul(selector, da));
            let not_sel_db = arena.alloc(Node::Mul(one_minus_sel, db));
            arena.alloc(Node::Add(sel_da, not_sel_db))
        },

        Node::Modulo(a, _b) => {
            // d(a % b) = da (ignoring discontinuities at integer multiples of b)
            differentiate(arena, a, mode, memo, state_filter, widths)
        },

        Node::Hypot(a, b) => {
            // d(hypot(a, b)) = (a*da + b*db) / hypot(a, b)
            let da = differentiate(arena, a, mode, memo, state_filter, widths);
            let db = differentiate(arena, b, mode, memo, state_filter, widths);
            let a_da = arena.alloc(Node::Mul(a, da));
            let b_db = arena.alloc(Node::Mul(b, db));
            let numer = arena.alloc(Node::Add(a_da, b_db));
            let hypot_ab = arena.alloc(Node::Hypot(a, b));
            arena.alloc(Node::Div(numer, hypot_ab))
        },

        Node::MatMul(a, b) => {
            // The matrix is always constant, so it contributes no term of its own.
            let db = differentiate(arena, b, mode, memo, state_filter, widths);
            arena.alloc(Node::MatMul(a, db))
        },

        Node::Index { child, start, end } => {
            // d(v[start:end]) = dv[start:end]
            let d_child = differentiate(arena, child, mode, memo, state_filter, widths);
            arena.alloc(Node::Index {
                child: d_child,
                start,
                end,
            })
        },

        Node::Concat(children) => {
            // d(concat(a, b, ...)) = concat(da, db, ...)
            let d_children: Vec<NodeId> = children
                .iter()
                .map(|&c| differentiate(arena, c, mode, memo, state_filter, widths))
                .collect();
            arena.alloc(Node::Concat(d_children))
        },

        Node::Interpolant1DLinear { data, child } => {
            let d_child = differentiate(arena, child, mode, memo, state_filter, widths);

            // Slopes are baked into the node here, so evaluation only has to
            // pick a segment.
            let slopes = compute_interpolant_slopes(&data);

            let deriv_interp = arena.alloc(Node::Interpolant1DLinearDeriv {
                slopes: slopes.into_boxed_slice(),
                x_data: data.x_data.clone().into_boxed_slice(),
                child,
            });

            arena.alloc(Node::Mul(deriv_interp, d_child))
        },

        // Only first order: the solver Jacobian never needs an interpolant's
        // second derivative.
        Node::Interpolant1DLinearDeriv { .. } => zero_of_width(arena, width),

        // Cubic/pchip interpolation: d(interp(x)) = interp'(x) * dx.
        Node::Interpolant1DCubic { data, child } => {
            let d_child = differentiate(arena, child, mode, memo, state_filter, widths);
            let deriv_interp = arena.alloc(Node::Interpolant1DCubicDeriv { data, child });
            arena.alloc(Node::Mul(deriv_interp, d_child))
        },
        // Only first order, as for the linear interpolant above.
        Node::Interpolant1DCubicDeriv { .. } => zero_of_width(arena, width),

        // Multivariate chain rule: sum over axes of ∂interp/∂x_a · dg_a.
        Node::InterpolantNd { data, children } => {
            let mut sum: Option<NodeId> = None;
            for (axis, &child) in children.iter().enumerate() {
                let d_child = differentiate(arena, child, mode, memo, state_filter, widths);
                let partial = arena.alloc(Node::InterpolantNdPartial {
                    data: data.clone(),
                    children: children.clone(),
                    axis: u32::try_from(axis).expect("axis index fits in u32"),
                });
                let term = arena.alloc(Node::Mul(partial, d_child));
                sum = Some(sum.map_or(term, |acc| arena.alloc(Node::Add(acc, term))));
            }
            sum.expect("InterpolantNd has at least one child")
        },
        // Only first order, matching the 1D interpolant-derivative treatment.
        Node::InterpolantNdPartial { .. } => zero_of_width(arena, width),

        Node::Conditional { selector, branches } => {
            // The selector only switches, so it carries no derivative.
            let d_branches: Vec<NodeId> = branches
                .iter()
                .map(|&b| differentiate(arena, b, mode, memo, state_filter, widths))
                .collect();
            arena.alloc(Node::Conditional {
                selector,
                branches: d_branches,
            })
        },

        // Picks the tangent component at the argmax, ties going to the first
        // occurrence, as `ReduceArgSelect` evaluation does for the primal.
        Node::MaxReduce(a) => {
            let da = differentiate(arena, a, mode, memo, state_filter, widths);
            arena.alloc(Node::ReduceArgSelect {
                basis: da,
                picker: a,
                is_max: true,
            })
        },
        Node::MinReduce(a) => {
            let da = differentiate(arena, a, mode, memo, state_filter, widths);
            arena.alloc(Node::ReduceArgSelect {
                basis: da,
                picker: a,
                is_max: false,
            })
        },

        // Only first order: this node exists only inside a derivative tape,
        // which is never differentiated again.
        Node::ReduceArgSelect { .. } => zero_of_width(arena, width),
    };

    memo.insert(id, result);
    result
}

/// Segment slopes of a piecewise linear interpolant: `n - 1` values for `n`
/// breakpoints, and zero across a segment whose breakpoints coincide.
pub(crate) fn compute_interpolant_slopes(data: &InterpolantData) -> Vec<f64> {
    let n = data.x_data.len();
    if n < 2 {
        return vec![];
    }

    let mut slopes = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        let dx = data.x_data[i + 1] - data.x_data[i];
        let dy = data.y_data[i + 1] - data.y_data[i];
        slopes.push(if dx.abs() > f64::EPSILON {
            dy / dx
        } else {
            0.0
        });
    }
    slopes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{CompiledExpr, TangentInputs};
    use crate::ir::TypedIr;
    use crate::simplify::{SimplifyMode, cse, dce, simplify, simplify_with_mode};
    use crate::zero_propagate::zero_propagate;

    /// Run the binding/jacobian tangent pipeline end to end and return the
    /// compiled tape. Mirrors `function.rs::tangent_expr` /
    /// `jacobian.rs::finish` exactly so width-collapse regressions surface here.
    fn compile_tangent_pipeline(arena: &Arena, root: NodeId, wrt_params: bool) -> CompiledExpr {
        let mut a = arena.clone();
        let root = if wrt_params {
            tangent_wrt_params(&mut a, root)
        } else {
            tangent_wrt_states(&mut a, root)
        };
        let root = simplify_with_mode(&mut a, root, SimplifyMode::Aggressive);
        let (za, root) = zero_propagate(&a, root);
        let (ca, root) = cse(&za, root);
        let (da, root) = dce(&ca, root);
        CompiledExpr::from_ir(TypedIr::from_arena(&da, root))
    }

    #[test]
    fn test_jvp_width_vector_plus_scalar_param() {
        // f = y[0:3] + p0, so df/dp @ [1] must be [1, 1, 1]: the tangent_p tape
        // must not collapse to length 1 and truncate the JVP to [1, 0, 0].
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let p = arena.alloc(Node::InputParameter {
            name: "a".into(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let root = arena.alloc(Node::Add(y, p));

        let tp = compile_tangent_pipeline(&arena, root, true);
        assert_eq!(
            tp.output_len(),
            3,
            "tangent_p tape collapsed below primal width"
        );
        let mut s = vec![0.0; tp.scratch_len()];
        let tangent = TangentInputs {
            dy: None,
            dp: Some(&[1.0]),
        };
        let r = tp.eval_with_tangent(&mut s, 0.0, &[0.0, 0.0, 0.0], &[], &[5.0], &tangent);
        assert_eq!(r, &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_jvp_width_wide_param_index_slice() {
        // f = b[1]*y0, b width-2: a width-1 tangent under Index{1..2} would read
        // outside its buffer slot. Seeded direction gives y0; zero direction gives 0.
        let mut arena = Arena::new();
        let b = arena.alloc(Node::InputParameter {
            name: "b".into(),
            index: 0,
            offset: 0,
            width: 2,
        });
        let b1 = arena.alloc(Node::Index {
            child: b,
            start: 1,
            end: 2,
        });
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let root = arena.alloc(Node::Mul(b1, y0));

        let tp = compile_tangent_pipeline(&arena, root, true);
        let mut s = vec![0.0; tp.scratch_len()];
        let seeded = TangentInputs {
            dy: None,
            dp: Some(&[1.0]),
        };
        let r = tp.eval_with_tangent(&mut s, 0.0, &[3.0], &[], &[10.0, 20.0], &seeded);
        assert_eq!(r, &[3.0]);

        let mut s2 = vec![0.0; tp.scratch_len()];
        let unseeded = TangentInputs {
            dy: None,
            dp: Some(&[0.0]),
        };
        let r0 = tp.eval_with_tangent(&mut s2, 0.0, &[3.0], &[], &[10.0, 20.0], &unseeded);
        assert_eq!(r0, &[0.0]);
    }

    #[test]
    fn test_jvp_width_zero_dfdy_keeps_length() {
        // f = const_vector([1,2,3]) + p0; df/dy ≡ 0. Repro 2.
        // The tangent_y tape must keep length 3 (all zeros), not collapse to [0].
        use crate::node::ArrayData;
        let mut arena = Arena::new();
        let cv = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![1.0, 2.0, 3.0],
            shape: crate::node::Shape::vector(3),
        })));
        let p = arena.alloc(Node::InputParameter {
            name: "a".into(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let root = arena.alloc(Node::Add(cv, p));

        let ty = compile_tangent_pipeline(&arena, root, false);
        assert_eq!(
            ty.output_len(),
            3,
            "tangent_y tape collapsed below primal width"
        );
        let mut s = vec![0.0; ty.scratch_len()];
        let tangent = TangentInputs {
            dy: Some(&[0.0]),
            dp: None,
        };
        let r = ty.eval_with_tangent(&mut s, 0.0, &[], &[], &[5.0], &tangent);
        assert_eq!(r, &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_jvp_width_concat_collapsed_child_offset() {
        // f = concat(const_vec[1,2], y0); df/dy must place y0's derivative at
        // row 2, not shift it to row 0 when the const-vector child collapses.
        use crate::node::ArrayData;
        let mut arena = Arena::new();
        let cv = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![1.0, 2.0],
            shape: crate::node::Shape::vector(2),
        })));
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let root = arena.alloc(Node::Concat(vec![cv, y0]));

        let ty = compile_tangent_pipeline(&arena, root, false);
        assert_eq!(ty.output_len(), 3, "concat tangent width must match primal");
        let mut s = vec![0.0; ty.scratch_len()];
        let tangent = TangentInputs {
            dy: Some(&[1.0]),
            dp: None,
        };
        let r = ty.eval_with_tangent(&mut s, 0.0, &[7.0], &[], &[], &tangent);
        // rows 0,1 come from the constant vector (df/dy = 0); row 2 = d(y0)/dy = 1.
        assert_eq!(r, &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_jvp_width_concat_empty_child_offset() {
        // f = concat(empty_vec[], y0 + 1); mirrors a pure-algebraic PyBaMM model
        // (concatenated_rhs is a length-0 Vector). df/dy0 must land at row 0,
        // not be swallowed by a phantom length-1 zero for the length-0 child.
        use crate::node::ArrayData;
        let mut arena = Arena::new();
        let empty = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![],
            shape: crate::node::Shape::vector(0),
        })));
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let one = arena.alloc(Node::Scalar(1.0));
        let expr = arena.alloc(Node::Add(y0, one));
        let root = arena.alloc(Node::Concat(vec![empty, expr]));

        let ty = compile_tangent_pipeline(&arena, root, false);
        assert_eq!(
            ty.output_len(),
            1,
            "concat tangent width must match primal (empty child contributes 0)"
        );
        let mut s = vec![0.0; ty.scratch_len()];
        let tangent = TangentInputs {
            dy: Some(&[1.0]),
            dp: None,
        };
        let r = ty.eval_with_tangent(&mut s, 0.0, &[7.0], &[], &[], &tangent);
        assert_eq!(
            r,
            &[1.0],
            "d(y0 + 1)/dy0 must be 1, not swallowed by the empty child"
        );
    }

    #[test]
    fn test_product_rule() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let product = arena.alloc(Node::Mul(x, y));

        let jac = tangent_wrt_states(&mut arena, product);
        let jac = simplify(&mut arena, jac);

        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        // d(x*y) @ [1, 0] = y = 4
        let tangent = TangentInputs {
            dy: Some(&[1.0, 0.0]),
            dp: None,
        };
        let result =
            compiled.eval_with_tangent(&mut s_compiled, 0.0, &[2.0, 4.0], &[], &[], &tangent);
        assert!(
            (result[0] - 4.0).abs() < 1e-14,
            "Expected 4.0, got {}",
            result[0]
        );

        // d(x*y) @ [0, 1] = x = 2
        let tangent = TangentInputs {
            dy: Some(&[0.0, 1.0]),
            dp: None,
        };
        let result =
            compiled.eval_with_tangent(&mut s_compiled, 0.0, &[2.0, 4.0], &[], &[], &tangent);
        assert!(
            (result[0] - 2.0).abs() < 1e-14,
            "Expected 2.0, got {}",
            result[0]
        );
    }

    #[test]
    fn test_diff_max_reduce_selects_argmax_tangent() {
        // d(max(y)) @ seed = seed[argmax(y)]; y=[0.3,0.9,0.1] -> k=1 -> seed[1]=20
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let expr = arena.alloc(Node::MaxReduce(y));
        let jac = tangent_wrt_states(&mut arena, expr);
        let jac = simplify(&mut arena, jac);
        let compiled = CompiledExpr::new(&arena, jac);
        let mut s = vec![0.0; compiled.scratch_len()];
        let tangent = TangentInputs {
            dy: Some(&[10.0, 20.0, 30.0]),
            dp: None,
        };
        let result = compiled.eval_with_tangent(&mut s, 0.0, &[0.3, 0.9, 0.1], &[], &[], &tangent);
        assert!(
            (result[0] - 20.0).abs() < 1e-14,
            "expected 20.0, got {}",
            result[0]
        );
    }

    #[test]
    fn test_diff_min_reduce_selects_argmin_tangent() {
        // d(min(y)) @ seed = seed[argmin(y)]; y=[0.3,0.9,0.1] -> k=2 -> seed[2]=30
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let expr = arena.alloc(Node::MinReduce(y));
        let jac = tangent_wrt_states(&mut arena, expr);
        let jac = simplify(&mut arena, jac);
        let compiled = CompiledExpr::new(&arena, jac);
        let mut s = vec![0.0; compiled.scratch_len()];
        let tangent = TangentInputs {
            dy: Some(&[10.0, 20.0, 30.0]),
            dp: None,
        };
        let result = compiled.eval_with_tangent(&mut s, 0.0, &[0.3, 0.9, 0.1], &[], &[], &tangent);
        assert!(
            (result[0] - 30.0).abs() < 1e-14,
            "expected 30.0, got {}",
            result[0]
        );
    }

    #[test]
    fn test_chain_rule() {
        // d(exp(2*x)) / dx = 2 * exp(2*x)
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let two = arena.alloc(Node::Scalar(2.0));
        let two_x = arena.alloc(Node::Mul(two, x));
        let expr = arena.alloc(Node::Exp(two_x));

        let jac = tangent_wrt_states(&mut arena, expr);
        let jac = simplify(&mut arena, jac);

        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        let x_val: f64 = 1.0;
        let expected = 2.0 * (2.0 * x_val).exp();
        let tangent = TangentInputs {
            dy: Some(&[1.0]),
            dp: None,
        };
        let result = compiled.eval_with_tangent(&mut s_compiled, 0.0, &[x_val], &[], &[], &tangent);
        assert!(
            (result[0] - expected).abs() < 1e-12,
            "Expected {}, got {}",
            expected,
            result[0]
        );
    }

    #[test]
    fn test_quotient_rule() {
        // d(x / y) = (y - x) / y^2 when dx=1, dy=1
        // At x=1, y=2: (2 - 1) / 4 = 0.25
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let quotient = arena.alloc(Node::Div(x, y));

        let jac = tangent_wrt_states(&mut arena, quotient);
        let jac = simplify(&mut arena, jac);

        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        // d(x/y) @ [1, 0] = 1/y = 0.5
        let tangent = TangentInputs {
            dy: Some(&[1.0, 0.0]),
            dp: None,
        };
        let result =
            compiled.eval_with_tangent(&mut s_compiled, 0.0, &[1.0, 2.0], &[], &[], &tangent);
        assert!(
            (result[0] - 0.5).abs() < 1e-14,
            "Expected 0.5, got {}",
            result[0]
        );

        // d(x/y) @ [0, 1] = -x/y^2 = -1/4 = -0.25
        let tangent = TangentInputs {
            dy: Some(&[0.0, 1.0]),
            dp: None,
        };
        let result =
            compiled.eval_with_tangent(&mut s_compiled, 0.0, &[1.0, 2.0], &[], &[], &tangent);
        assert!(
            (result[0] + 0.25).abs() < 1e-14,
            "Expected -0.25, got {}",
            result[0]
        );
    }

    #[test]
    fn test_power_rule_integer() {
        // d(x^3) / dx = 3x^2
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let three = arena.alloc(Node::Scalar(3.0));
        let x_cubed = arena.alloc(Node::Pow(x, three));

        let jac = tangent_wrt_states(&mut arena, x_cubed);
        let jac = simplify(&mut arena, jac);

        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        let x_val = 2.0;
        let expected = 3.0 * x_val * x_val; // 12.0
        let tangent = TangentInputs {
            dy: Some(&[1.0]),
            dp: None,
        };
        let result = compiled.eval_with_tangent(&mut s_compiled, 0.0, &[x_val], &[], &[], &tangent);
        assert!(
            (result[0] - expected).abs() < 1e-12,
            "Expected {}, got {}",
            expected,
            result[0]
        );
    }

    #[test]
    fn test_power_rule_negative_base() {
        // d(x^2) / dx = 2x (works for negative x)
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let two = arena.alloc(Node::Scalar(2.0));
        let x_sq = arena.alloc(Node::Pow(x, two));

        let jac = tangent_wrt_states(&mut arena, x_sq);
        let jac = simplify(&mut arena, jac);

        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        let x_val = -3.0;
        let expected = 2.0 * x_val; // -6.0
        let tangent = TangentInputs {
            dy: Some(&[1.0]),
            dp: None,
        };
        let result = compiled.eval_with_tangent(&mut s_compiled, 0.0, &[x_val], &[], &[], &tangent);
        assert!(
            (result[0] - expected).abs() < 1e-12,
            "Expected {}, got {}",
            expected,
            result[0]
        );
    }

    #[test]
    fn test_sqrt_derivative() {
        // d(sqrt(x)) / dx = 1 / (2*sqrt(x))
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let sqrt_x = arena.alloc(Node::Sqrt(x));

        let jac = tangent_wrt_states(&mut arena, sqrt_x);
        let jac = simplify(&mut arena, jac);

        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        let x_val: f64 = 4.0;
        let expected = 0.5 / x_val.sqrt(); // 0.25
        let tangent = TangentInputs {
            dy: Some(&[1.0]),
            dp: None,
        };
        let result = compiled.eval_with_tangent(&mut s_compiled, 0.0, &[x_val], &[], &[], &tangent);
        assert!(
            (result[0] - expected).abs() < 1e-12,
            "Expected {}, got {}",
            expected,
            result[0]
        );
    }

    #[test]
    fn test_log_derivative() {
        // d(log(x)) / dx = 1/x
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let log_x = arena.alloc(Node::Log(x));

        let jac = tangent_wrt_states(&mut arena, log_x);
        let jac = simplify(&mut arena, jac);

        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        let x_val = 2.0;
        let expected = 1.0 / x_val;
        let tangent = TangentInputs {
            dy: Some(&[1.0]),
            dp: None,
        };
        let result = compiled.eval_with_tangent(&mut s_compiled, 0.0, &[x_val], &[], &[], &tangent);
        assert!(
            (result[0] - expected).abs() < 1e-12,
            "Expected {}, got {}",
            expected,
            result[0]
        );
    }

    #[test]
    fn test_sin_derivative() {
        // d(sin(x)) / dx = cos(x)
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let sin_x = arena.alloc(Node::Sin(x));

        let jac = tangent_wrt_states(&mut arena, sin_x);
        let jac = simplify(&mut arena, jac);

        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        let x_val = std::f64::consts::PI / 4.0;
        let expected = x_val.cos();
        let tangent = TangentInputs {
            dy: Some(&[1.0]),
            dp: None,
        };
        let result = compiled.eval_with_tangent(&mut s_compiled, 0.0, &[x_val], &[], &[], &tangent);
        assert!(
            (result[0] - expected).abs() < 1e-12,
            "Expected {}, got {}",
            expected,
            result[0]
        );
    }

    #[test]
    fn test_cos_derivative() {
        // d(cos(x)) / dx = -sin(x)
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let cos_x = arena.alloc(Node::Cos(x));

        let jac = tangent_wrt_states(&mut arena, cos_x);
        let jac = simplify(&mut arena, jac);

        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        let x_val = std::f64::consts::PI / 4.0;
        let expected = -x_val.sin();
        let tangent = TangentInputs {
            dy: Some(&[1.0]),
            dp: None,
        };
        let result = compiled.eval_with_tangent(&mut s_compiled, 0.0, &[x_val], &[], &[], &tangent);
        assert!(
            (result[0] - expected).abs() < 1e-12,
            "Expected {}, got {}",
            expected,
            result[0]
        );
    }

    #[test]
    fn test_tanh_derivative() {
        // d(tanh(x)) / dx = 1 - tanh(x)^2
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let tanh_x = arena.alloc(Node::Tanh(x));

        let jac = tangent_wrt_states(&mut arena, tanh_x);
        let jac = simplify(&mut arena, jac);

        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        let x_val: f64 = 1.0;
        let tanh_val = x_val.tanh();
        let expected = tanh_val.mul_add(-tanh_val, 1.0);
        let tangent = TangentInputs {
            dy: Some(&[1.0]),
            dp: None,
        };
        let result = compiled.eval_with_tangent(&mut s_compiled, 0.0, &[x_val], &[], &[], &tangent);
        assert!(
            (result[0] - expected).abs() < 1e-12,
            "Expected {}, got {}",
            expected,
            result[0]
        );
    }

    #[test]
    fn test_sinh_derivative() {
        // d(sinh(x)) / dx = cosh(x)
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let sinh_x = arena.alloc(Node::Sinh(x));

        let jac = tangent_wrt_states(&mut arena, sinh_x);
        let jac = simplify(&mut arena, jac);

        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        let x_val: f64 = 1.5;
        let expected = x_val.cosh();
        let tangent = TangentInputs {
            dy: Some(&[1.0]),
            dp: None,
        };
        let result = compiled.eval_with_tangent(&mut s_compiled, 0.0, &[x_val], &[], &[], &tangent);
        assert!(
            (result[0] - expected).abs() < 1e-12,
            "Expected {}, got {}",
            expected,
            result[0]
        );
    }

    #[test]
    fn test_cosh_derivative() {
        // d(cosh(x)) / dx = sinh(x)
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let cosh_x = arena.alloc(Node::Cosh(x));

        let jac = tangent_wrt_states(&mut arena, cosh_x);
        let jac = simplify(&mut arena, jac);

        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        let x_val: f64 = 1.5;
        let expected = x_val.sinh();
        let tangent = TangentInputs {
            dy: Some(&[1.0]),
            dp: None,
        };
        let result = compiled.eval_with_tangent(&mut s_compiled, 0.0, &[x_val], &[], &[], &tangent);
        assert!(
            (result[0] - expected).abs() < 1e-12,
            "Expected {}, got {}",
            expected,
            result[0]
        );
    }

    #[test]
    fn test_arcsinh_derivative() {
        // d(arcsinh(x)) / dx = 1 / sqrt(1 + x^2)
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let arcsinh_x = arena.alloc(Node::Arcsinh(x));

        let jac = tangent_wrt_states(&mut arena, arcsinh_x);
        let jac = simplify(&mut arena, jac);

        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        let x_val: f64 = 2.0;
        let expected = 1.0 / x_val.mul_add(x_val, 1.0).sqrt();
        let tangent = TangentInputs {
            dy: Some(&[1.0]),
            dp: None,
        };
        let result = compiled.eval_with_tangent(&mut s_compiled, 0.0, &[x_val], &[], &[], &tangent);
        assert!(
            (result[0] - expected).abs() < 1e-12,
            "Expected {}, got {}",
            expected,
            result[0]
        );
    }

    #[test]
    fn test_arctan_derivative() {
        // d(arctan(x)) / dx = 1 / (1 + x^2)
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let arctan_x = arena.alloc(Node::Arctan(x));

        let jac = tangent_wrt_states(&mut arena, arctan_x);
        let jac = simplify(&mut arena, jac);

        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        let x_val: f64 = 1.0;
        let expected = 1.0 / x_val.mul_add(x_val, 1.0);
        let tangent = TangentInputs {
            dy: Some(&[1.0]),
            dp: None,
        };
        let result = compiled.eval_with_tangent(&mut s_compiled, 0.0, &[x_val], &[], &[], &tangent);
        assert!(
            (result[0] - expected).abs() < 1e-12,
            "Expected {}, got {}",
            expected,
            result[0]
        );
    }

    #[test]
    fn test_neg_derivative() {
        // d(-x) / dx = -1
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let neg_x = arena.alloc(Node::Neg(x));

        let jac = tangent_wrt_states(&mut arena, neg_x);
        let jac = simplify(&mut arena, jac);

        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        let tangent = TangentInputs {
            dy: Some(&[1.0]),
            dp: None,
        };
        let result = compiled.eval_with_tangent(&mut s_compiled, 0.0, &[5.0], &[], &[], &tangent);
        assert!(
            (result[0] + 1.0).abs() < 1e-14,
            "Expected -1.0, got {}",
            result[0]
        );
    }

    #[test]
    fn test_abs_derivative() {
        // d(|x|) / dx = sign(x)
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let abs_x = arena.alloc(Node::Abs(x));

        let jac = tangent_wrt_states(&mut arena, abs_x);
        let jac = simplify(&mut arena, jac);

        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        let tangent = TangentInputs {
            dy: Some(&[1.0]),
            dp: None,
        };
        let result = compiled.eval_with_tangent(&mut s_compiled, 0.0, &[5.0], &[], &[], &tangent);
        assert!(
            (result[0] - 1.0).abs() < 1e-14,
            "Expected 1.0, got {}",
            result[0]
        );

        let result = compiled.eval_with_tangent(&mut s_compiled, 0.0, &[-5.0], &[], &[], &tangent);
        assert!(
            (result[0] + 1.0).abs() < 1e-14,
            "Expected -1.0, got {}",
            result[0]
        );
    }

    #[test]
    fn test_index_derivative() {
        // d(x[1:3]) / dx = dx[1:3]
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 5 });
        let slice = arena.alloc(Node::Index {
            child: x,
            start: 1,
            end: 3,
        });

        let jac = tangent_wrt_states(&mut arena, slice);
        let jac = simplify(&mut arena, jac);

        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        let tangent = TangentInputs {
            dy: Some(&[0.0, 1.0, 2.0, 0.0, 0.0]),
            dp: None,
        };
        let result = compiled.eval_with_tangent(
            &mut s_compiled,
            0.0,
            &[0.0, 1.0, 2.0, 3.0, 4.0],
            &[],
            &[],
            &tangent,
        );
        assert_eq!(result.len(), 2);
        assert!(
            (result[0] - 1.0).abs() < 1e-14,
            "Expected 1.0, got {}",
            result[0]
        );
        assert!(
            (result[1] - 2.0).abs() < 1e-14,
            "Expected 2.0, got {}",
            result[1]
        );
    }

    #[test]
    fn test_concat_derivative() {
        // d(concat(x, y)) = concat(dx, dy)
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let y = arena.alloc(Node::StateVector { start: 2, end: 4 });
        let concat = arena.alloc(Node::Concat(vec![x, y]));

        let jac = tangent_wrt_states(&mut arena, concat);
        let jac = simplify(&mut arena, jac);

        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        let tangent = TangentInputs {
            dy: Some(&[1.0, 2.0, 3.0, 4.0]),
            dp: None,
        };
        let result = compiled.eval_with_tangent(
            &mut s_compiled,
            0.0,
            &[0.0, 0.0, 0.0, 0.0],
            &[],
            &[],
            &tangent,
        );
        assert_eq!(result.len(), 4);
        assert!((result[0] - 1.0).abs() < 1e-14);
        assert!((result[1] - 2.0).abs() < 1e-14);
        assert!((result[2] - 3.0).abs() < 1e-14);
        assert!((result[3] - 4.0).abs() < 1e-14);
    }

    #[test]
    fn test_parameter_differentiation() {
        // d(p * x) / dp = x
        let mut arena = Arena::new();
        let p = arena.alloc(Node::InputParameter {
            name: "k".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let expr = arena.alloc(Node::Mul(p, x));

        let jac = tangent_wrt_params(&mut arena, expr);
        let jac = simplify(&mut arena, jac);

        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        let tangent = TangentInputs {
            dy: None,
            dp: Some(&[1.0]),
        };
        let x_val = 3.0;
        let result =
            compiled.eval_with_tangent(&mut s_compiled, 0.0, &[x_val], &[], &[2.0], &tangent);
        assert!(
            (result[0] - x_val).abs() < 1e-14,
            "Expected {}, got {}",
            x_val,
            result[0]
        );
    }

    #[test]
    fn test_constant_derivative() {
        let mut arena = Arena::new();
        let c = arena.alloc(Node::Scalar(42.0));

        let jac = tangent_wrt_states(&mut arena, c);
        let jac = simplify(&mut arena, jac);

        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        let tangent = TangentInputs {
            dy: Some(&[]),
            dp: None,
        };
        let result = compiled.eval_with_tangent(&mut s_compiled, 0.0, &[], &[], &[], &tangent);
        assert!((result[0]).abs() < 1e-14, "Expected 0.0, got {}", result[0]);
    }

    #[test]
    fn test_time_derivative() {
        // d(t) / dy = 0
        let mut arena = Arena::new();
        let t = arena.alloc(Node::Time);

        let jac = tangent_wrt_states(&mut arena, t);
        let jac = simplify(&mut arena, jac);

        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        let tangent = TangentInputs {
            dy: Some(&[]),
            dp: None,
        };
        let result = compiled.eval_with_tangent(&mut s_compiled, 5.0, &[], &[], &[], &tangent);
        assert!((result[0]).abs() < 1e-14, "Expected 0.0, got {}", result[0]);
    }

    #[test]
    fn test_minimum_derivative() {
        // d(min(x, y)) = dx if x <= y, else dy
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let min_xy = arena.alloc(Node::Minimum(x, y));

        let jac = tangent_wrt_states(&mut arena, min_xy);
        let jac = simplify(&mut arena, jac);

        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        // x < y, so derivative is dx
        let tangent = TangentInputs {
            dy: Some(&[1.0, 0.0]),
            dp: None,
        };
        let result =
            compiled.eval_with_tangent(&mut s_compiled, 0.0, &[1.0, 3.0], &[], &[], &tangent);
        assert!(
            (result[0] - 1.0).abs() < 1e-14,
            "Expected 1.0, got {}",
            result[0]
        );

        // x > y, so derivative is dy
        let tangent = TangentInputs {
            dy: Some(&[0.0, 1.0]),
            dp: None,
        };
        let result =
            compiled.eval_with_tangent(&mut s_compiled, 0.0, &[5.0, 2.0], &[], &[], &tangent);
        assert!(
            (result[0] - 1.0).abs() < 1e-14,
            "Expected 1.0, got {}",
            result[0]
        );
    }

    #[test]
    fn test_maximum_derivative() {
        // d(max(x, y)) = dx if x >= y, else dy
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let max_xy = arena.alloc(Node::Maximum(x, y));

        let jac = tangent_wrt_states(&mut arena, max_xy);
        let jac = simplify(&mut arena, jac);

        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        // x > y, so derivative is dx
        let tangent = TangentInputs {
            dy: Some(&[1.0, 0.0]),
            dp: None,
        };
        let result =
            compiled.eval_with_tangent(&mut s_compiled, 0.0, &[5.0, 2.0], &[], &[], &tangent);
        assert!(
            (result[0] - 1.0).abs() < 1e-14,
            "Expected 1.0, got {}",
            result[0]
        );

        // x < y, so derivative is dy
        let tangent = TangentInputs {
            dy: Some(&[0.0, 1.0]),
            dp: None,
        };
        let result =
            compiled.eval_with_tangent(&mut s_compiled, 0.0, &[1.0, 3.0], &[], &[], &tangent);
        assert!(
            (result[0] - 1.0).abs() < 1e-14,
            "Expected 1.0, got {}",
            result[0]
        );
    }

    #[test]
    fn test_hypot_derivative() {
        // d(hypot(x, y)) = (x*dx + y*dy) / hypot(x, y)
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let hypot_xy = arena.alloc(Node::Hypot(x, y));

        let jac = tangent_wrt_states(&mut arena, hypot_xy);
        let jac = simplify(&mut arena, jac);

        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        // At (3, 4), hypot = 5
        // d(hypot)/dx = 3/5 = 0.6
        let tangent = TangentInputs {
            dy: Some(&[1.0, 0.0]),
            dp: None,
        };
        let result =
            compiled.eval_with_tangent(&mut s_compiled, 0.0, &[3.0, 4.0], &[], &[], &tangent);
        assert!(
            (result[0] - 0.6).abs() < 1e-12,
            "Expected 0.6, got {}",
            result[0]
        );

        // d(hypot)/dy = 4/5 = 0.8
        let tangent = TangentInputs {
            dy: Some(&[0.0, 1.0]),
            dp: None,
        };
        let result =
            compiled.eval_with_tangent(&mut s_compiled, 0.0, &[3.0, 4.0], &[], &[], &tangent);
        assert!(
            (result[0] - 0.8).abs() < 1e-12,
            "Expected 0.8, got {}",
            result[0]
        );
    }

    #[test]
    fn test_interpolation_derivative() {
        // d(interp(x)) / dx = slope at x
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let interp = arena.alloc(Node::Interpolant1DLinear {
            data: Box::new(InterpolantData {
                x_data: vec![0.0, 1.0, 2.0],
                y_data: vec![0.0, 10.0, 30.0], // slopes: 10, 20
            }),
            child: x,
        });

        let jac = tangent_wrt_states(&mut arena, interp);
        let jac = simplify(&mut arena, jac);

        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        // x=0.5 lands in segment [0, 1], slope 10
        let tangent = TangentInputs {
            dy: Some(&[1.0]),
            dp: None,
        };
        let result = compiled.eval_with_tangent(&mut s_compiled, 0.0, &[0.5], &[], &[], &tangent);
        assert!(
            (result[0] - 10.0).abs() < 1e-12,
            "Expected 10.0, got {}",
            result[0]
        );

        // x=1.5 lands in segment [1, 2], slope 20
        let result = compiled.eval_with_tangent(&mut s_compiled, 0.0, &[1.5], &[], &[], &tangent);
        assert!(
            (result[0] - 20.0).abs() < 1e-12,
            "Expected 20.0, got {}",
            result[0]
        );

        // Out-of-domain: extend with boundary-segment slope (not 0)
        // Below x=0.0: first segment slope = 10
        let result = compiled.eval_with_tangent(&mut s_compiled, 0.0, &[-0.5], &[], &[], &tangent);
        assert!(
            (result[0] - 10.0).abs() < 1e-12,
            "Expected 10.0 (first-segment slope) below domain, got {}",
            result[0]
        );

        // Above x=2.0: last segment slope = 20
        let result = compiled.eval_with_tangent(&mut s_compiled, 0.0, &[2.5], &[], &[], &tangent);
        assert!(
            (result[0] - 20.0).abs() < 1e-12,
            "Expected 20.0 (last-segment slope) above domain, got {}",
            result[0]
        );
    }

    #[test]
    fn test_cubic_interpolation_derivative() {
        use crate::node::CubicInterpolantData;
        // p(dx) = 1 + 2*dx + 3*dx^2 + 4*dx^3 on [0, 5]; p'(dx) = 2 + 6*dx + 12*dx^2.
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let interp = arena.alloc(Node::Interpolant1DCubic {
            data: Box::new(CubicInterpolantData {
                breakpoints: vec![0.0, 5.0],
                coeffs: vec![[1.0, 2.0, 3.0, 4.0]],
            }),
            child: x,
        });
        let jac = tangent_wrt_states(&mut arena, interp);
        let jac = simplify(&mut arena, jac);
        let compiled = CompiledExpr::new(&arena, jac);
        let mut s = vec![0.0; compiled.scratch_len()];
        let tangent = TangentInputs {
            dy: Some(&[1.0]),
            dp: None,
        };

        // At x=1 (dx=1): p' = 2 + 6 + 12 = 20
        let r = compiled.eval_with_tangent(&mut s, 0.0, &[1.0], &[], &[], &tangent);
        assert!((r[0] - 20.0).abs() < 1e-12, "expected 20, got {}", r[0]);
        // At x=0 (dx=0): p' = 2
        let r = compiled.eval_with_tangent(&mut s, 0.0, &[0.0], &[], &[], &tangent);
        assert!((r[0] - 2.0).abs() < 1e-12, "expected 2, got {}", r[0]);
        // Below domain x=-1 (clamps to interval 0, dx=-1): p' = 2 - 6 + 12 = 8
        let r = compiled.eval_with_tangent(&mut s, 0.0, &[-1.0], &[], &[], &tangent);
        assert!((r[0] - 8.0).abs() < 1e-12, "expected 8, got {}", r[0]);
        // Above domain x=6 (clamps to interval 0, dx=6): p' = 2 + 36 + 432 = 470
        let r = compiled.eval_with_tangent(&mut s, 0.0, &[6.0], &[], &[], &tangent);
        assert!((r[0] - 470.0).abs() < 1e-12, "expected 470, got {}", r[0]);
    }

    #[test]
    fn test_nd_interpolation_partial_derivatives() {
        use crate::node::NdInterpolantData;
        // p = 3*dx0^2 + 5*dx1^3 + dx0*dx1 on one cell [0,4]x[0,4].
        // ∂p/∂x0 = 6*dx0 + dx1; ∂p/∂x1 = 15*dx1^2 + dx0.
        let mut arena = Arena::new();
        let x0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let x1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let mut coeffs = vec![0.0; 16];
        coeffs[2 * 4] = 3.0; // dx0^2
        coeffs[3] = 5.0; // dx1^3
        coeffs[4 + 1] = 1.0; // dx0*dx1
        let interp = arena.alloc(Node::InterpolantNd {
            data: Box::new(NdInterpolantData {
                breakpoints: vec![vec![0.0, 4.0], vec![0.0, 4.0]],
                coeffs,
                order: 4,
            }),
            children: vec![x0, x1],
        });
        let jac = tangent_wrt_states(&mut arena, interp);
        let jac = simplify(&mut arena, jac);
        let compiled = CompiledExpr::new(&arena, jac);
        let mut s = vec![0.0; compiled.scratch_len()];

        // At (2,1): ∂p/∂x0 = 12 + 1 = 13; ∂p/∂x1 = 15 + 2 = 17 (seeded
        // independently so each partial is verified on its own).
        let t0 = TangentInputs {
            dy: Some(&[1.0, 0.0]),
            dp: None,
        };
        let r = compiled.eval_with_tangent(&mut s, 0.0, &[2.0, 1.0], &[], &[], &t0);
        assert!((r[0] - 13.0).abs() < 1e-12, "expected 13, got {}", r[0]);
        let t1 = TangentInputs {
            dy: Some(&[0.0, 1.0]),
            dp: None,
        };
        let r = compiled.eval_with_tangent(&mut s, 0.0, &[2.0, 1.0], &[], &[], &t1);
        assert!((r[0] - 17.0).abs() < 1e-12, "expected 17, got {}", r[0]);
        // Directional derivative sums both partials: 13 + 17 = 30.
        let tb = TangentInputs {
            dy: Some(&[1.0, 1.0]),
            dp: None,
        };
        let r = compiled.eval_with_tangent(&mut s, 0.0, &[2.0, 1.0], &[], &[], &tb);
        assert!((r[0] - 30.0).abs() < 1e-12, "expected 30, got {}", r[0]);
    }

    #[test]
    fn test_nd_interpolation_bilinear_partials() {
        use crate::node::NdInterpolantData;
        // p = 2 + 3*dx0 + 4*dx1 + 5*dx0*dx1 on one cell [0,2]x[0,2].
        // ∂p/∂x0 = 3 + 5*dx1; ∂p/∂x1 = 4 + 5*dx0.
        let mut arena = Arena::new();
        let x0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let x1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let interp = arena.alloc(Node::InterpolantNd {
            data: Box::new(NdInterpolantData {
                breakpoints: vec![vec![0.0, 2.0], vec![0.0, 2.0]],
                coeffs: vec![2.0, 4.0, 3.0, 5.0],
                order: 2,
            }),
            children: vec![x0, x1],
        });
        let jac = tangent_wrt_states(&mut arena, interp);
        let jac = simplify(&mut arena, jac);
        let compiled = CompiledExpr::new(&arena, jac);
        let mut s = vec![0.0; compiled.scratch_len()];

        // At (1,2): ∂p/∂x0 = 3 + 10 = 13; ∂p/∂x1 = 4 + 5 = 9.
        let t0 = TangentInputs {
            dy: Some(&[1.0, 0.0]),
            dp: None,
        };
        let r = compiled.eval_with_tangent(&mut s, 0.0, &[1.0, 2.0], &[], &[], &t0);
        assert!((r[0] - 13.0).abs() < 1e-12, "expected 13, got {}", r[0]);
        let t1 = TangentInputs {
            dy: Some(&[0.0, 1.0]),
            dp: None,
        };
        let r = compiled.eval_with_tangent(&mut s, 0.0, &[1.0, 2.0], &[], &[], &t1);
        assert!((r[0] - 9.0).abs() < 1e-12, "expected 9, got {}", r[0]);
    }

    #[test]
    fn test_memoization() {
        // x * x reaches the same child twice, so the memo is what keeps one
        // tangent of x rather than two.
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let x_sq = arena.alloc(Node::Mul(x, x));

        let jac = tangent_wrt_states(&mut arena, x_sq);
        let jac = simplify(&mut arena, jac);

        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        // d(x^2) @ [1] = 2x at x=3 => 6
        let tangent = TangentInputs {
            dy: Some(&[1.0]),
            dp: None,
        };
        let result = compiled.eval_with_tangent(&mut s_compiled, 0.0, &[3.0], &[], &[], &tangent);
        assert!(
            (result[0] - 6.0).abs() < 1e-12,
            "Expected 6.0, got {}",
            result[0]
        );
    }

    #[test]
    fn test_conditional_derivative() {
        // d(cond(sel, [b1, b2])) = cond(sel, [db1, db2])
        let mut arena = Arena::new();
        let selector = arena.alloc(Node::Scalar(1.0)); // Select first branch
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let two = arena.alloc(Node::Scalar(2.0));
        let branch1 = arena.alloc(Node::Mul(two, x)); // 2x
        let three = arena.alloc(Node::Scalar(3.0));
        let branch2 = arena.alloc(Node::Mul(three, x)); // 3x
        let cond = arena.alloc(Node::Conditional {
            selector,
            branches: vec![branch1, branch2],
        });

        let jac = tangent_wrt_states(&mut arena, cond);
        let jac = simplify(&mut arena, jac);

        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        // With selector=1, should select d(2x) = 2
        let tangent = TangentInputs {
            dy: Some(&[1.0]),
            dp: None,
        };
        let result = compiled.eval_with_tangent(&mut s_compiled, 0.0, &[5.0], &[], &[], &tangent);
        assert!(
            (result[0] - 2.0).abs() < 1e-12,
            "Expected 2.0, got {}",
            result[0]
        );
    }

    #[test]
    fn test_matmul_derivative() {
        // d(A @ v) = A @ dv
        use crate::node::{CsrData, Shape};

        let mut arena = Arena::new();
        // 2x3, one entry per row: A @ dv = [2*dv[0], 3*dv[1]]
        let sparse = arena.alloc(Node::SparseMatrix(Box::new(CsrData {
            indptr: vec![0, 1, 2],
            indices: vec![0, 1],
            data: vec![2.0, 3.0], // Diagonal with 2 and 3
            shape: Shape::matrix(2, 3),
        })));
        let v = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let matmul = arena.alloc(Node::MatMul(sparse, v));

        let jac = tangent_wrt_states(&mut arena, matmul);
        let jac = simplify(&mut arena, jac);

        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        // A @ [1, 0, 0] = [2, 0]
        let tangent = TangentInputs {
            dy: Some(&[1.0, 0.0, 0.0]),
            dp: None,
        };
        let result =
            compiled.eval_with_tangent(&mut s_compiled, 0.0, &[0.0, 0.0, 0.0], &[], &[], &tangent);
        assert_eq!(result.len(), 2);
        assert!(
            (result[0] - 2.0).abs() < 1e-14,
            "Expected 2.0, got {}",
            result[0]
        );
        assert!((result[1]).abs() < 1e-14, "Expected 0.0, got {}", result[1]);
    }

    #[test]
    fn test_erf_derivative() {
        // d(erf(x))/dx = 2/sqrt(pi) * exp(-x^2)
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let erf_x = arena.alloc(Node::Erf(x));
        let jac = tangent_wrt_states(&mut arena, erf_x);
        let jac = simplify(&mut arena, jac);
        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        let x_val: f64 = 1.0;
        let two_over_sqrt_pi = 2.0 / std::f64::consts::PI.sqrt();
        let expected = two_over_sqrt_pi * (-x_val * x_val).exp();
        let tangent = TangentInputs {
            dy: Some(&[1.0]),
            dp: None,
        };
        let result = compiled.eval_with_tangent(&mut s_compiled, 0.0, &[x_val], &[], &[], &tangent);
        assert!(
            (result[0] - expected).abs() < 1e-12,
            "Expected {}, got {}",
            expected,
            result[0]
        );
    }

    #[test]
    fn test_power_rule_fractional() {
        // d(x^1.5) / dx = 1.5 * x^0.5
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let exp = arena.alloc(Node::Scalar(1.5));
        let x_pow = arena.alloc(Node::Pow(x, exp));

        let jac = tangent_wrt_states(&mut arena, x_pow);
        let jac = simplify(&mut arena, jac);
        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        let x_val: f64 = 4.0;
        let expected = 1.5 * x_val.sqrt(); // 1.5 * 2.0 = 3.0
        let tangent = TangentInputs {
            dy: Some(&[1.0]),
            dp: None,
        };
        let result = compiled.eval_with_tangent(&mut s_compiled, 0.0, &[x_val], &[], &[], &tangent);
        assert!(
            (result[0] - expected).abs() < 1e-12,
            "Expected {}, got {}",
            expected,
            result[0]
        );

        let mut arena2 = Arena::new();
        let x2 = arena2.alloc(Node::StateVector { start: 0, end: 1 });
        let half = arena2.alloc(Node::Scalar(0.5));
        let x_sqrt = arena2.alloc(Node::Pow(x2, half));
        let jac2 = tangent_wrt_states(&mut arena2, x_sqrt);
        let jac2 = simplify(&mut arena2, jac2);
        let compiled2 = CompiledExpr::new(&arena2, jac2);
        let mut s_compiled2 = vec![0.0; compiled2.scratch_len()];

        // d(x^0.5) / dx = 0.5 * x^(-0.5) = 0.5 / sqrt(x)
        let x_val2: f64 = 9.0;
        let expected2 = 0.5 / x_val2.sqrt(); // 0.5 / 3.0 ≈ 0.1667
        let result2 =
            compiled2.eval_with_tangent(&mut s_compiled2, 0.0, &[x_val2], &[], &[], &tangent);
        assert!(
            (result2[0] - expected2).abs() < 1e-12,
            "Expected {}, got {}",
            expected2,
            result2[0]
        );
    }

    #[test]
    fn test_power_rule_variable_exponent() {
        // d(x^y)/dx @ [1, 0] = y * x^(y-1), d(x^y)/dy @ [0, 1] = x^y * log(x)
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let x_pow_y = arena.alloc(Node::Pow(x, y));

        let jac = tangent_wrt_states(&mut arena, x_pow_y);
        let jac = simplify(&mut arena, jac);
        let compiled = CompiledExpr::new(&arena, jac);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        let x_val: f64 = 2.0;
        let y_val: f64 = 3.0;

        // d(x^y)/dx: tangent [1, 0] -> y * x^(y-1) = 3 * 2^2 = 12
        let tangent_dx = TangentInputs {
            dy: Some(&[1.0, 0.0]),
            dp: None,
        };
        let result_dx = compiled.eval_with_tangent(
            &mut s_compiled,
            0.0,
            &[x_val, y_val],
            &[],
            &[],
            &tangent_dx,
        );
        let expected_dx = y_val * x_val.powf(y_val - 1.0); // 3 * 4 = 12
        assert!(
            (result_dx[0] - expected_dx).abs() < 1e-10,
            "d(x^y)/dx: Expected {}, got {}",
            expected_dx,
            result_dx[0]
        );

        // d(x^y)/dy: tangent [0, 1] -> x^y * log(x) = 8 * ln(2) ≈ 5.545
        let tangent_dy = TangentInputs {
            dy: Some(&[0.0, 1.0]),
            dp: None,
        };
        let result_dy = compiled.eval_with_tangent(
            &mut s_compiled,
            0.0,
            &[x_val, y_val],
            &[],
            &[],
            &tangent_dy,
        );
        let expected_dy = x_val.powf(y_val) * x_val.ln();
        assert!(
            (result_dy[0] - expected_dy).abs() < 1e-10,
            "d(x^y)/dy: Expected {}, got {}",
            expected_dy,
            result_dy[0]
        );
    }

    #[test]
    fn test_tangent_wrt_subset() {
        // f = x0 * x1, differentiate only w.r.t. x1
        // df/dx1 = x0 (x0 treated as constant)
        let mut arena = Arena::new();
        let x0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let x1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let expr = arena.alloc(Node::Mul(x0, x1));

        let subset = HashSet::from([1usize]);
        let deriv = tangent_wrt_subset(&mut arena, expr, &subset);
        let deriv = simplify(&mut arena, deriv);

        let topo = arena.topological_order(deriv);
        for &nid in &topo {
            if let Node::TangentStateVector { start, .. } = *arena.get(nid) {
                assert_ne!(start, 0, "Should not differentiate w.r.t. x0");
            }
        }
        assert!(
            topo.iter().any(|&nid| matches!(
                arena.get(nid),
                Node::TangentStateVector { start: 1, end: 2 }
            )),
            "Should contain TangentStateVector for x1"
        );
    }

    #[test]
    fn test_tangent_wrt_subset_numerical() {
        // f = x0^2 + 3*x1, differentiate only w.r.t. x1
        // df/dx1 = 3 (x0 treated as constant)
        let mut arena = Arena::new();
        let x0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let x1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let x0_sq = arena.alloc(Node::Mul(x0, x0));
        let three = arena.alloc(Node::Scalar(3.0));
        let three_x1 = arena.alloc(Node::Mul(three, x1));
        let expr = arena.alloc(Node::Add(x0_sq, three_x1));

        let subset = HashSet::from([1usize]);
        let deriv = tangent_wrt_subset(&mut arena, expr, &subset);
        let deriv = simplify(&mut arena, deriv);
        let compiled = CompiledExpr::new(&arena, deriv);
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

        // With tangent [0, 1]: df/dx1 = 3
        let tangent = TangentInputs {
            dy: Some(&[0.0, 1.0]),
            dp: None,
        };
        let result =
            compiled.eval_with_tangent(&mut s_compiled, 0.0, &[5.0, 2.0], &[], &[], &tangent);
        assert!(
            (result[0] - 3.0).abs() < 1e-12,
            "Expected 3.0, got {}",
            result[0]
        );

        // With tangent [1, 0]: should be 0 (x0 not in active set)
        let tangent_x0 = TangentInputs {
            dy: Some(&[1.0, 0.0]),
            dp: None,
        };
        let result_x0 =
            compiled.eval_with_tangent(&mut s_compiled, 0.0, &[5.0, 2.0], &[], &[], &tangent_x0);
        assert!(
            result_x0[0].abs() < 1e-12,
            "Expected 0.0, got {}",
            result_x0[0]
        );
    }

    #[test]
    fn test_tangent_wrt_subset_all_active_matches_full() {
        let mut arena = Arena::new();
        let x0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let x1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let sum = arena.alloc(Node::Add(x0, x1));
        let expr = arena.alloc(Node::Mul(sum, x0)); // f = (x0 + x1) * x0

        let all_active = HashSet::from([0usize, 1]);

        let mut arena_full = arena.clone();
        let full = tangent_wrt_states(&mut arena_full, expr);
        let full = simplify(&mut arena_full, full);
        let compiled_full = CompiledExpr::new(&arena_full, full);
        let mut s_compiled_full = vec![0.0; compiled_full.scratch_len()];

        let subset = tangent_wrt_subset(&mut arena, expr, &all_active);
        let subset = simplify(&mut arena, subset);
        let compiled_subset = CompiledExpr::new(&arena, subset);
        let mut s_compiled_subset = vec![0.0; compiled_subset.scratch_len()];

        for dy in &[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]] {
            let tangent = TangentInputs {
                dy: Some(dy),
                dp: None,
            };
            let r_full = compiled_full.eval_with_tangent(
                &mut s_compiled_full,
                0.0,
                &[3.0, 7.0],
                &[],
                &[],
                &tangent,
            );
            let r_sub = compiled_subset.eval_with_tangent(
                &mut s_compiled_subset,
                0.0,
                &[3.0, 7.0],
                &[],
                &[],
                &tangent,
            );
            assert!(
                (r_full[0] - r_sub[0]).abs() < 1e-12,
                "Mismatch for dy={dy:?}: full={}, subset={}",
                r_full[0],
                r_sub[0]
            );
        }
    }
}
