//! Symbolic forward-mode differentiation into Jacobian-vector product expressions.
//!
//! Tangent nodes read their seed at evaluation time, so one compiled tape serves every direction.

use crate::arena::{Arena, NodeId, NodeMap};
use crate::ir::infer_sizes;
use crate::node::Node;

/// Which variables carry a tangent; every other leaf differentiates to zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiffTarget {
    /// `d/dy`, seeded from `TangentInputs::dy`.
    States,
    /// `d/dp`, seeded from `TangentInputs::dp`.
    Params,
}

/// Build the tangent expression for `(d expr/dy) @ dy`.
#[must_use]
pub fn tangent_wrt_states(arena: &mut Arena, expr: NodeId) -> NodeId {
    differentiate_root(arena, expr, DiffTarget::States, None)
}

/// Build the tangent expression for `(d expr/dp) @ dp`.
#[must_use]
pub fn tangent_wrt_params(arena: &mut Arena, expr: NodeId) -> NodeId {
    differentiate_root(arena, expr, DiffTarget::Params, None)
}

/// [`tangent_wrt_states`] seeding only `StateVector` nodes that overlap `active_indices`.
///
/// `active_indices` must ascend. Overlap is all-or-nothing: a node straddling the
/// boundary is seeded over its whole range.
#[must_use]
pub fn tangent_wrt_subset(arena: &mut Arena, expr: NodeId, active_indices: &[usize]) -> NodeId {
    assert!(
        active_indices.is_sorted(),
        "active state indices must ascend"
    );
    differentiate_root(arena, expr, DiffTarget::States, Some(active_indices))
}

/// Differentiate `expr`, seeding only states in `state_filter` when one is given.
fn differentiate_root(
    arena: &mut Arena,
    expr: NodeId,
    mode: DiffTarget,
    state_filter: Option<&[usize]>,
) -> NodeId {
    let mut pending = arena.topological_order(expr);
    let widths = infer_sizes(arena, &pending);
    // The order's buffer, one slot per reachable node, becomes the work stack.
    pending.clear();
    pending.push(expr);
    let mut walk = Differentiator {
        mode,
        state_filter,
        widths,
        memo: NodeMap::new(arena.len()),
        pending,
        parked_sums: Vec::new(),
    };
    while let Some(&id) = walk.pending.last() {
        if walk.memo.contains_key(id) {
            walk.pending.pop();
        } else if let Some(derivative) = walk.differentiate(arena, id) {
            walk.pending.pop();
            walk.memo.insert(id, derivative);
        }
    }
    *walk
        .memo
        .get(expr)
        .expect("the walk ends with `expr` differentiated")
}

/// Forward-mode rules, walked with an explicit stack so graph depth is bounded by the heap.
///
/// A rule missing a child's derivative queues the child and is retried after it, so nodes
/// are allocated in exactly the order a depth-first recursion would allocate them.
struct Differentiator<'a> {
    mode: DiffTarget,
    state_filter: Option<&'a [usize]>,
    /// Primal output widths.
    widths: Vec<usize>,
    /// One derivative per primal node.
    memo: NodeMap<NodeId>,
    /// Nodes left to differentiate; each waits on those pushed above it.
    pending: Vec<NodeId>,
    /// `(node, next axis, sum so far)` of each `InterpolantNd` waiting on a child, innermost
    /// last: a node's children finish before it, so the one retried is always on top.
    parked_sums: Vec<(NodeId, usize, Option<NodeId>)>,
}

impl Differentiator<'_> {
    /// Queue the not-yet-differentiated `children`, first on top; `true` if there were none.
    fn queue_missing(&mut self, children: &[NodeId]) -> bool {
        let n_pending = self.pending.len();
        for &child in children.iter().rev() {
            if !self.memo.contains_key(child) {
                self.pending.push(child);
            }
        }
        self.pending.len() == n_pending
    }

    fn derivative(&self, id: NodeId) -> NodeId {
        *self
            .memo
            .get(id)
            .expect("a rule reads only children it has seen differentiated")
    }

    /// The derivatives of `children`, or `None` once the missing ones are queued.
    fn ready<const N: usize>(&mut self, children: [NodeId; N]) -> Option<[NodeId; N]> {
        self.queue_missing(&children)
            .then(|| children.map(|child| self.derivative(child)))
    }

    /// [`ready`](Self::ready) for a variadic node.
    fn ready_all(&mut self, children: &[NodeId]) -> Option<Vec<NodeId>> {
        self.queue_missing(children).then(|| {
            children
                .iter()
                .map(|&child| self.derivative(child))
                .collect()
        })
    }

    /// Where `id`'s axis-by-axis sum stopped, or its start if it never parked.
    fn unpark_sum(&mut self, id: NodeId) -> (usize, Option<NodeId>) {
        match self.parked_sums.last() {
            Some(&(parked, axis, sum)) if parked == id => {
                self.parked_sums.pop();
                (axis, sum)
            },
            _ => (0, None),
        }
    }
}

/// Whether the ascending `indices` hold any state in `start..end`.
fn overlaps(indices: &[usize], start: usize, end: usize) -> bool {
    let first = indices.partition_point(|&index| index < start);
    indices.get(first).is_some_and(|&index| index < end)
}

/// Whether `id` is a literal, so its derivative is identically zero.
fn is_literal(arena: &Arena, id: NodeId) -> bool {
    matches!(
        &arena[id],
        Node::Scalar(_) | Node::Array(_) | Node::ZeroVector { .. }
    )
}

/// Whether `id` is a zero [`Node::zero`] builds, so a product with it can be skipped.
fn is_literal_zero(arena: &Arena, id: NodeId) -> bool {
    match &arena[id] {
        Node::Scalar(v) => v.to_bits() == 0.0_f64.to_bits(),
        Node::ZeroVector { .. } => true,
        _ => false,
    }
}

/// The smallest positive subnormal, which the rules below substitute for an exact zero.
pub(crate) const SMALLEST_SUBNORMAL: f64 = f64::from_bits(1);

/// `d(a^b)/db` given `pow`, the node `a^b`: `a^b * ln(a)` where `a > 0` and 0 elsewhere.
///
/// `ln` reads `max(a, smallest subnormal)`, so the gated-off product is a finite zero, not
/// `0 * -inf`; for `a > 0` the gate and the `max` are exact, so the value is `a^b * ln(a)`.
fn exponent_partial(arena: &mut Arena, base: NodeId, pow: NodeId) -> NodeId {
    let zero = arena.alloc(Node::Scalar(0.0));
    let positive = arena.alloc(Node::NotEqualHeaviside(zero, base));
    let gated = arena.alloc(Node::Mul(positive, pow));
    let tiny = arena.alloc(Node::Scalar(SMALLEST_SUBNORMAL));
    let log_argument = arena.alloc(Node::Maximum(base, tiny));
    let log_a = arena.alloc(Node::Log(log_argument));
    arena.alloc(Node::Mul(gated, log_a))
}

/// `d` with an exact zero raised to the smallest subnormal, as a denominator.
///
/// A zero numerator over it stays zero where `d` is zero, instead of `0 / 0`, and one above
/// 1e-15 in magnitude still overflows to ±inf. Elsewhere the sum is exact, NaN included.
fn nonzero_denominator(arena: &mut Arena, d: NodeId) -> NodeId {
    let zero = arena.alloc(Node::Scalar(0.0));
    let is_zero = arena.alloc(Node::Equality(d, zero));
    let tiny = arena.alloc(Node::Scalar(SMALLEST_SUBNORMAL));
    let lift = arena.alloc(Node::Mul(is_zero, tiny));
    arena.alloc(Node::Add(d, lift))
}

/// `exp * base^(exp - 1) * da`, the base's share of `d(base^exp)`.
///
/// An exponent below 1 makes the partial infinite at a zero base, where the product is
/// `0 * inf` for a zero `da`; such exponents, and any that is not a literal, take
/// `exp * da / base^(1 - exp)` over a [`nonzero_denominator`] instead.
fn base_tangent(arena: &mut Arena, base: NodeId, exp: NodeId, da: NodeId) -> NodeId {
    let literal = arena[exp].as_scalar();
    if let Some(e) = literal.filter(|&e| e >= 1.0) {
        let exp_minus_1 = arena.alloc(Node::Scalar(e - 1.0));
        let pow_nm1 = arena.alloc(Node::Pow(base, exp_minus_1));
        let d_base = arena.alloc(Node::Mul(exp, pow_nm1));
        return arena.alloc(Node::Mul(d_base, da));
    }
    let one_minus_exp = if let Some(e) = literal {
        arena.alloc(Node::Scalar(1.0 - e))
    } else {
        let one = arena.alloc(Node::Scalar(1.0));
        arena.alloc(Node::Sub(one, exp))
    };
    let pow = arena.alloc(Node::Pow(base, one_minus_exp));
    let denominator = nonzero_denominator(arena, pow);
    // On the numerator: dividing the guarded denominator by `|exp| >= 2` rounds it back to 0.
    let numerator = arena.alloc(Node::Mul(exp, da));
    arena.alloc(Node::Div(numerator, denominator))
}

/// `da` where `selector` is 1, else `db`: the subgradient `Minimum` and `Maximum` share.
///
/// Masked, so an infinite or NaN tangent of the operand not taken is dropped, not `0 * inf`.
fn blend_tangents(arena: &mut Arena, selector: NodeId, da: NodeId, db: NodeId) -> NodeId {
    let one = arena.alloc(Node::Scalar(1.0));
    let one_minus_sel = arena.alloc(Node::Sub(one, selector));
    let sel_da = arena.alloc(Node::MaskedMul(selector, da));
    let not_sel_db = arena.alloc(Node::MaskedMul(one_minus_sel, db));
    arena.alloc(Node::Add(sel_da, not_sel_db))
}

impl Differentiator<'_> {
    /// `id`'s derivative, or `None` once it has queued the work it waits on.
    ///
    /// Allocates nothing before returning `None`, so a retry starts clean; an `InterpolantNd`
    /// parks what it allocated and resumes from there.
    fn differentiate(&mut self, arena: &mut Arena, id: NodeId) -> Option<NodeId> {
        let (mode, state_filter) = (self.mode, self.state_filter);
        let width = self.widths[id.index()];

        let result = match arena[id] {
            Node::StateVector { start, end }
                if mode == DiffTarget::States
                    && state_filter.is_none_or(|filter| overlaps(filter, start, end)) =>
            {
                arena.alloc(Node::TangentStateVector { start, end })
            },

            Node::InputParameter {
                index,
                width: param_width,
                ..
            } if mode == DiffTarget::Params => {
                // Replicated to the packed width so an `Index` slice stays inside the buffer slot.
                let tangent = arena.alloc(Node::TangentParameter { index });
                if param_width == 1 {
                    tangent
                } else {
                    arena.alloc(Node::Concat(vec![tangent; param_width]))
                }
            },

            // Leaves that carry no tangent in this mode. d(y')/dy is the mass matrix, which the
            // solver supplies separately.
            Node::Scalar(_)
            | Node::Array(_)
            | Node::ZeroVector { .. }
            | Node::SparseMatrix(_)
            | Node::Time
            | Node::StateVector { .. }
            | Node::StateVectorDot { .. }
            | Node::InputParameter { .. }
            | Node::TangentStateVector { .. }
            | Node::TangentParameter { .. }
            // Step functions: zero away from the jumps, which are not modelled.
            | Node::Sign(_)
            | Node::Floor(_)
            | Node::Ceiling(_)
            | Node::EqualHeaviside(..)
            | Node::NotEqualHeaviside(..)
            | Node::Equality(..)
            // Derivative-only nodes: the solver Jacobian never needs a second derivative.
            | Node::Interpolant1DLinearDeriv { .. }
            | Node::Interpolant1DCubicDeriv { .. }
            | Node::InterpolantNdPartial { .. }
            | Node::ReduceArgSelect { .. } => arena.alloc(Node::zero(width)),

            Node::Add(a, b) => {
                let [da, db] = self.ready([a, b])?;
                arena.alloc(Node::Add(da, db))
            },

            Node::Sub(a, b) => {
                let [da, db] = self.ready([a, b])?;
                arena.alloc(Node::Sub(da, db))
            },

            Node::Mul(a, b) => {
                let [da, db] = self.ready([a, b])?;
                let a_db = arena.alloc(Node::Mul(a, db));
                let da_b = arena.alloc(Node::Mul(da, b));
                arena.alloc(Node::Add(a_db, da_b))
            },

            // `(da - (a / b) db) / b` overflows only where `a / b` does, unlike squaring `b`;
            // masked, so a zero `db` drops an overflowed quotient rather than meeting `0 * inf`.
            Node::Div(a, b) => {
                let [da, db] = self.ready([a, b])?;
                let quotient_db = arena.alloc(Node::MaskedMul(db, id));
                let numer = arena.alloc(Node::Sub(da, quotient_db));
                arena.alloc(Node::Div(numer, b))
            },

            // Masked on both terms, so a zero mask still drops the tangent of whatever it masks.
            Node::MaskedMul(mask, value) => {
                let [d_mask, d_value] = self.ready([mask, value])?;
                let mask_term = arena.alloc(Node::MaskedMul(d_mask, value));
                let value_term = arena.alloc(Node::MaskedMul(mask, d_value));
                arena.alloc(Node::Add(mask_term, value_term))
            },

            // `x^0` is constant, so its tangent is zero even where `0 * x^-1` is not.
            Node::Pow(_, exp) if arena[exp].as_scalar() == Some(0.0) => {
                arena.alloc(Node::zero(width))
            },

            // Partials times tangents; a constant exponent skips the `ln` term, which a
            // zero base would turn into `0 * -inf` for every seed.
            Node::Pow(base, exp) => {
                let [da] = self.ready([base])?;
                let db = if is_literal(arena, exp) {
                    None
                } else {
                    Some(self.ready([exp])?[0])
                };
                let base_term = base_tangent(arena, base, exp, da);
                db.filter(|&db| !is_literal_zero(arena, db))
                    .map_or(base_term, |db| {
                        let d_exp = exponent_partial(arena, base, id);
                        let exp_term = arena.alloc(Node::Mul(d_exp, db));
                        arena.alloc(Node::Add(base_term, exp_term))
                    })
            },

            Node::Neg(a) => {
                let [da] = self.ready([a])?;
                arena.alloc(Node::Neg(da))
            },

            Node::Abs(a) => {
                let [da] = self.ready([a])?;
                let sign_a = arena.alloc(Node::Sign(a));
                arena.alloc(Node::Mul(sign_a, da))
            },

            Node::Sqrt(a) => {
                let [da] = self.ready([a])?;
                let two = arena.alloc(Node::Scalar(2.0));
                let two_sqrt_a = arena.alloc(Node::Mul(two, id));
                let denominator = nonzero_denominator(arena, two_sqrt_a);
                arena.alloc(Node::Div(da, denominator))
            },

            Node::Exp(a) => {
                let [da] = self.ready([a])?;
                arena.alloc(Node::Mul(id, da))
            },

            Node::Log(a) => {
                let [da] = self.ready([a])?;
                arena.alloc(Node::Div(da, a))
            },

            Node::Sin(a) => {
                let [da] = self.ready([a])?;
                let cos_a = arena.alloc(Node::Cos(a));
                arena.alloc(Node::Mul(cos_a, da))
            },

            Node::Cos(a) => {
                let [da] = self.ready([a])?;
                let sin_a = arena.alloc(Node::Sin(a));
                let neg_sin_a = arena.alloc(Node::Neg(sin_a));
                arena.alloc(Node::Mul(neg_sin_a, da))
            },

            Node::Tanh(a) => {
                let [da] = self.ready([a])?;
                let one = arena.alloc(Node::Scalar(1.0));
                let two = arena.alloc(Node::Scalar(2.0));
                let tanh_sq = arena.alloc(Node::Pow(id, two));
                let one_minus_tanh_sq = arena.alloc(Node::Sub(one, tanh_sq));
                arena.alloc(Node::Mul(one_minus_tanh_sq, da))
            },

            Node::Sinh(a) => {
                let [da] = self.ready([a])?;
                let cosh_a = arena.alloc(Node::Cosh(a));
                arena.alloc(Node::Mul(cosh_a, da))
            },

            Node::Cosh(a) => {
                let [da] = self.ready([a])?;
                let sinh_a = arena.alloc(Node::Sinh(a));
                arena.alloc(Node::Mul(sinh_a, da))
            },

            // `hypot(1, a)` is `sqrt(1 + a^2)` without squaring `a`, which overflows past 1e154.
            Node::Arcsinh(a) => {
                let [da] = self.ready([a])?;
                let one = arena.alloc(Node::Scalar(1.0));
                let denominator = arena.alloc(Node::Hypot(one, a));
                arena.alloc(Node::Div(da, denominator))
            },

            Node::Arctan(a) => {
                let [da] = self.ready([a])?;
                let one = arena.alloc(Node::Scalar(1.0));
                let two = arena.alloc(Node::Scalar(2.0));
                let a_sq = arena.alloc(Node::Pow(a, two));
                let one_plus_a_sq = arena.alloc(Node::Add(one, a_sq));
                arena.alloc(Node::Div(da, one_plus_a_sq))
            },

            Node::Erf(a) => {
                let [da] = self.ready([a])?;
                let two_over_sqrt_pi = arena.alloc(Node::Scalar(std::f64::consts::FRAC_2_SQRT_PI));
                let neg_one = arena.alloc(Node::Scalar(-1.0));
                let two = arena.alloc(Node::Scalar(2.0));
                let a_sq = arena.alloc(Node::Pow(a, two));
                let neg_a_sq = arena.alloc(Node::Mul(neg_one, a_sq));
                let exp_neg_a_sq = arena.alloc(Node::Exp(neg_a_sq));
                let coeff = arena.alloc(Node::Mul(two_over_sqrt_pi, exp_neg_a_sq));
                arena.alloc(Node::Mul(coeff, da))
            },

            // Subgradient: `da` wherever `a` is the result, ties included; a NaN operand is
            // never the result, as `minimumNumber` and `maximumNumber` return the other one.
            Node::Minimum(a, b) | Node::Maximum(a, b) => {
                let [da, db] = self.ready([a, b])?;
                let selector = arena.alloc(Node::Equality(a, id));
                blend_tangents(arena, selector, da, db)
            },

            Node::Modulo(a, b) => {
                // Ignores the discontinuities at integer multiples of b.
                let [da] = self.ready([a])?;
                let db = if is_literal(arena, b) {
                    None
                } else {
                    Some(self.ready([b])?[0])
                };
                // A scalar `a` broadcasts against a vector `b`; replicate rather than add a zero,
                // which would flip a -0.0 tangent.
                let da = if self.widths[a.index()] == 1 && width > 1 {
                    arena.alloc(Node::Concat(vec![da; width]))
                } else {
                    da
                };
                db.map_or(da, |db| {
                    // d/db is -trunc(a / b); reading it off the remainder rounds as `%` does.
                    let shift = arena.alloc(Node::Sub(id, a));
                    let neg_quotient = arena.alloc(Node::Div(shift, b));
                    let db_term = arena.alloc(Node::Mul(neg_quotient, db));
                    arena.alloc(Node::Add(da, db_term))
                })
            },

            // Each leg over the hypotenuse is at most 1, so no product overflows, and is 0 at the
            // origin, so the tangent is 0 there, as `abs`'s is.
            Node::Hypot(a, b) => {
                let [da, db] = self.ready([a, b])?;
                let denominator = nonzero_denominator(arena, id);
                let a_share = arena.alloc(Node::Div(a, denominator));
                let b_share = arena.alloc(Node::Div(b, denominator));
                let a_term = arena.alloc(Node::Mul(a_share, da));
                let b_term = arena.alloc(Node::Mul(b_share, db));
                arena.alloc(Node::Add(a_term, b_term))
            },

            Node::MatMul(a, b) => {
                // The matrix is always constant, so it contributes no term of its own.
                let [db] = self.ready([b])?;
                arena.alloc(Node::MatMul(a, db))
            },

            Node::Index { child, start, end } => {
                let [d_child] = self.ready([child])?;
                arena.alloc(Node::Index {
                    child: d_child,
                    start,
                    end,
                })
            },

            Node::Concat(ref children) => {
                let d_children = self.ready_all(children)?;
                arena.alloc(Node::Concat(d_children))
            },

            Node::Interpolant1DLinear { ref data, child } => {
                let [d_child] = self.ready([child])?;
                let data = data.clone();

                let deriv_interp = arena.alloc(Node::Interpolant1DLinearDeriv { data, child });

                arena.alloc(Node::Mul(deriv_interp, d_child))
            },

            Node::Interpolant1DCubic { ref data, child } => {
                let [d_child] = self.ready([child])?;
                let data = data.clone();
                let deriv_interp = arena.alloc(Node::Interpolant1DCubicDeriv { data, child });
                arena.alloc(Node::Mul(deriv_interp, d_child))
            },

            // Each axis's term is allocated before the next child is differentiated, so a
            // missing child parks the sum so far rather than discarding it.
            Node::InterpolantNd { .. } => {
                let (mut axis, mut sum) = self.unpark_sum(id);
                loop {
                    let Node::InterpolantNd { data, children } = &arena[id] else {
                        unreachable!("matched as an `InterpolantNd` above")
                    };
                    let Some(&child) = children.get(axis) else {
                        break sum.expect("InterpolantNd has at least one child");
                    };
                    let Some([d_child]) = self.ready([child]) else {
                        self.parked_sums.push((id, axis, sum));
                        return None;
                    };
                    let partial = Node::InterpolantNdPartial {
                        data: data.clone(),
                        children: children.clone(),
                        axis: u32::try_from(axis).expect("axis index fits in u32"),
                    };
                    let partial = arena.alloc(partial);
                    let term = arena.alloc(Node::Mul(partial, d_child));
                    sum = Some(sum.map_or(term, |acc| arena.alloc(Node::Add(acc, term))));
                    axis += 1;
                }
            },

            Node::Conditional {
                selector,
                ref branches,
            } => {
                let d_branches = self.ready_all(branches)?;
                arena.alloc(Node::Conditional {
                    selector,
                    branches: d_branches,
                })
            },

            // Ties go to the first occurrence, matching the primal.
            Node::MaxReduce(a) => {
                let [da] = self.ready([a])?;
                arena.alloc(Node::ReduceArgSelect {
                    basis: da,
                    picker: a,
                    is_max: true,
                })
            },
            Node::MinReduce(a) => {
                let [da] = self.ready([a])?;
                arena.alloc(Node::ReduceArgSelect {
                    basis: da,
                    picker: a,
                    is_max: false,
                })
            },
        };

        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_2_SQRT_PI, FRAC_PI_4};

    use super::*;
    use crate::arena::DagRef;
    use crate::eval::{CompiledExpr, TangentInputs};
    use crate::node::{
        ArrayData, CsrData, CubicInterpolantData, InterpolantData, NdInterpolantData, Shape,
    };
    use crate::simplify::{cse, dce, simplify};
    use crate::zero_propagate::zero_propagate;

    /// Run the tangent pipeline end to end and return the compiled tape.
    fn compile_tangent_pipeline(arena: &Arena, root: NodeId, wrt_params: bool) -> CompiledExpr {
        let mut a = arena.clone();
        let root = if wrt_params {
            tangent_wrt_params(&mut a, root)
        } else {
            tangent_wrt_states(&mut a, root)
        };
        let root = simplify(&mut a, root);
        let propagated = zero_propagate(DagRef::new(&a, root));
        let shared = cse(propagated.as_ref());
        let pruned = dce(shared.as_ref());
        CompiledExpr::new(pruned.as_ref())
    }

    /// `(d root/dy) @ dy` through the simplified tangent tape.
    fn jvp(arena: &Arena, root: NodeId, y: &[f64], inputs: &[f64], dy: &[f64]) -> Vec<f64> {
        let mut arena = arena.clone();
        let tangent = tangent_wrt_states(&mut arena, root);
        let tangent = simplify(&mut arena, tangent);
        let compiled = CompiledExpr::new(DagRef::new(&arena, tangent));
        let mut scratch = vec![0.0; compiled.scratch_len()];
        let seed = TangentInputs {
            dy: Some(dy),
            dp: None,
        };
        compiled
            .eval_with_tangent(&mut scratch, 0.0, y, &[], inputs, &seed)
            .to_vec()
    }

    /// The gradient of a scalar `root`, one forward sweep per state.
    fn forward_gradient(arena: &Arena, root: NodeId, y: &[f64], inputs: &[f64]) -> Vec<f64> {
        (0..y.len())
            .map(|state| {
                let mut dy = vec![0.0; y.len()];
                dy[state] = 1.0;
                jvp(arena, root, y, inputs, &dy)[0]
            })
            .collect()
    }

    #[test]
    fn power_rule_with_an_input_exponent_holds_at_a_zero_or_negative_base() {
        // d(y0^p)/dy0 = p * y0^(p - 1); a 0 * (1 / 0) form made it NaN at the origin.
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let p = arena.alloc(Node::InputParameter {
            name: "p".into(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let root = arena.alloc(Node::Pow(y0, p));
        for y in [0.0, -1.0, 2.0] {
            let closed_form = [2.0 * y];
            assert_eq!(forward_gradient(&arena, root, &[y], &[2.0]), closed_form);
        }
    }

    #[test]
    fn power_rule_with_a_state_exponent_is_finite_at_a_zero_base() {
        // At (0, 2): d/dy0 = 2 * 0 and d/dy1 is 0 by the `a > 0` convention.
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let root = arena.alloc(Node::Pow(y0, y1));
        let y = [0.0, 2.0];
        assert_eq!(forward_gradient(&arena, root, &y, &[]), [0.0, 0.0]);
    }

    #[test]
    fn a_zero_literal_exponent_has_a_zero_tangent_before_simplification() {
        // `0 * x^-1` is NaN at a zero base, so the rule must not form it.
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let zero = arena.alloc(Node::Scalar(0.0));
        let root = arena.alloc(Node::Pow(y, zero));
        let tangent = tangent_wrt_states(&mut arena, root);
        let compiled = CompiledExpr::new(DagRef::new(&arena, tangent));
        let mut scratch = vec![0.0; compiled.scratch_len()];
        let seed = TangentInputs {
            dy: Some(&[1.0, 1.0]),
            dp: None,
        };
        let got = compiled.eval_with_tangent(&mut scratch, 0.0, &[0.0, 2.0], &[], &[], &seed);
        assert_eq!(got, [0.0, 0.0]);
    }

    #[test]
    fn a_zero_tangent_stays_zero_where_a_partial_is_infinite() {
        // `f + y1` at y0 = y2 = 0, as CasADi's `jacobian` gives it; each was NaN in every column.
        enum F {
            Sqrt,
            Hypot,
            Literal(f64),
            Input(f64),
        }
        let cases = [
            (F::Sqrt, [f64::INFINITY, 1.0, 0.0]),
            // Its own columns are 0 at the origin, as `abs`'s are; CasADi gives NaN.
            (F::Hypot, [0.0, 1.0, 0.0]),
            (F::Literal(0.5), [f64::INFINITY, 1.0, 0.0]),
            (F::Literal(2.0), [0.0, 1.0, 0.0]),
            (F::Literal(-1.0), [f64::NEG_INFINITY, 1.0, 0.0]),
            (F::Input(0.5), [f64::INFINITY, 1.0, 0.0]),
            (F::Input(0.0), [0.0, 1.0, 0.0]),
            (F::Input(-1.0), [f64::NEG_INFINITY, 1.0, 0.0]),
            (F::Input(-2.0), [f64::NEG_INFINITY, 1.0, 0.0]),
        ];
        for (f, expected) in cases {
            let mut arena = Arena::new();
            let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
            let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
            let y2 = arena.alloc(Node::StateVector { start: 2, end: 3 });
            let (singular, inputs) = match f {
                F::Sqrt => (arena.alloc(Node::Sqrt(y0)), vec![]),
                F::Hypot => (arena.alloc(Node::Hypot(y0, y2)), vec![]),
                F::Literal(e) => {
                    let e = arena.alloc(Node::Scalar(e));
                    (arena.alloc(Node::Pow(y0, e)), vec![])
                },
                F::Input(p) => {
                    let e = arena.alloc(Node::InputParameter {
                        name: "p".into(),
                        index: 0,
                        offset: 0,
                        width: 1,
                    });
                    (arena.alloc(Node::Pow(y0, e)), vec![p])
                },
            };
            let root = arena.alloc(Node::Add(singular, y1));
            let got = forward_gradient(&arena, root, &[0.0, 1.0, 0.0], &inputs);
            assert_eq!(got, expected, "{:?}", arena[singular]);
        }
    }

    #[test]
    fn a_negative_square_root_still_has_a_nan_tangent() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let root = arena.alloc(Node::Sqrt(y));
        assert!(forward_gradient(&arena, root, &[-1.0], &[])[0].is_nan());
    }

    #[test]
    fn min_and_max_differentiate_the_operand_they_return_beside_a_nan() {
        type Build = fn(NodeId, NodeId) -> Node;
        let ops: [(&str, Build); 2] = [("minimum", Node::Minimum), ("maximum", Node::Maximum)];
        for (label, op) in ops {
            for state_first in [true, false] {
                let mut arena = Arena::new();
                let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
                let nan = arena.alloc(Node::Scalar(f64::NAN));
                let node = if state_first { op(y, nan) } else { op(nan, y) };
                let root = arena.alloc(node);
                let got = forward_gradient(&arena, root, &[0.5], &[]);
                assert_eq!(got, [1.0], "{label}, state first: {state_first}");
            }
        }

        // Element by element: the NaN entry returns the state, the 0 entry the literal.
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let literal = ArrayData::try_new(vec![f64::NAN, 0.0], Shape::vector(2));
        let literal = arena.alloc(Node::Array(Box::new(literal.expect("valid array"))));
        let root = arena.alloc(Node::Minimum(y, literal));
        assert_eq!(jvp(&arena, root, &[0.5, 0.5], &[], &[1.0, 1.0]), [1.0, 0.0]);
    }

    #[test]
    fn min_and_max_drop_a_non_finite_tangent_of_the_operand_they_do_not_return() {
        // `op(y0, sqrt(y1))`: the square root's tangent is inf at y1 = 0 and NaN below it.
        type Build = fn(NodeId, NodeId) -> Node;
        let cases: [(&str, Build, [f64; 2], [f64; 2]); 4] = [
            ("maximum", Node::Maximum, [1.0, 0.0], [1.0, 0.0]),
            ("maximum", Node::Maximum, [1.0, -1.0], [1.0, 0.0]),
            ("minimum", Node::Minimum, [-1.0, 0.0], [1.0, 0.0]),
            // Returned, the infinite tangent still flows.
            ("maximum", Node::Maximum, [-1.0, 0.0], [0.0, f64::INFINITY]),
        ];
        for (label, op, y, expected) in cases {
            let mut arena = Arena::new();
            let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
            let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
            let sqrt_y1 = arena.alloc(Node::Sqrt(y1));
            let root = arena.alloc(op(y0, sqrt_y1));
            let got = forward_gradient(&arena, root, &y, &[]);
            assert_eq!(got, expected, "{label} at {y:?}");
        }
    }

    #[test]
    fn masked_rules_differentiate_again_to_the_hessian() {
        // `dy^T H dy` for dy = [1, 1], from differentiating the first tangent once more.
        type Build = fn(&mut Arena, NodeId, NodeId) -> NodeId;
        let cases: [(&str, Build, [f64; 2], f64); 2] = [
            // H = [[0, -1/y1^2], [-1/y1^2, 2 y0/y1^3]].
            (
                "quotient",
                |a, x, y| a.alloc(Node::Div(x, y)),
                [1.0, 2.0],
                -0.25,
            ),
            // Returns y0^2, whose H is [[2, 0], [0, 0]].
            (
                "maximum",
                |a, x, y| {
                    let square = a.alloc(Node::Mul(x, x));
                    a.alloc(Node::Maximum(square, y))
                },
                [2.0, 1.0],
                2.0,
            ),
        ];
        for (label, build, y, expected) in cases {
            let mut arena = Arena::new();
            let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
            let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
            let root = build(&mut arena, y0, y1);
            let first = tangent_wrt_states(&mut arena, root);
            let second = tangent_wrt_states(&mut arena, first);
            let compiled = CompiledExpr::new(DagRef::new(&arena, second));
            let mut scratch = vec![0.0; compiled.scratch_len()];
            let seed = TangentInputs {
                dy: Some(&[1.0, 1.0]),
                dp: None,
            };
            let got = compiled.eval_with_tangent(&mut scratch, 0.0, &y, &[], &[], &seed);
            assert_eq!(got, [expected], "{label} at {y:?}");
        }
    }

    #[test]
    fn test_jvp_width_vector_plus_scalar_param() {
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
        // A width-1 tangent under `Index{1..2}` would read outside its buffer slot.
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
        let mut arena = Arena::new();
        let cv = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![1.0, 2.0, 3.0],
            shape: Shape::vector(3),
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
        let mut arena = Arena::new();
        let cv = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![1.0, 2.0],
            shape: Shape::vector(2),
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
        assert_eq!(r, &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_jvp_width_concat_empty_child_offset() {
        let mut arena = Arena::new();
        let empty = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![],
            shape: Shape::vector(0),
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

    /// Deeper than a recursive walk survives on a 2 MiB test-thread stack.
    const DEEP: usize = 200_000;

    #[test]
    fn differentiates_a_deep_chain() {
        // Cycles a one-child rule, a two-child rule and the axis-by-axis N-D rule.
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let table = NdInterpolantData::try_new(vec![vec![0.0, 1.0]; 2], vec![1.0; 4], 2)
            .expect("valid table");
        let mut id = y;
        for level in 0..DEEP {
            id = arena.alloc(match level % 3 {
                0 => Node::Sin(id),
                1 => Node::Mul(id, y),
                _ => Node::InterpolantNd {
                    data: Box::new(table.clone()),
                    children: vec![id, y],
                },
            });
        }

        let mut tangent = tangent_wrt_states(&mut arena, id);
        for level in (0..DEEP).rev() {
            tangent = match (level % 3, &arena[tangent]) {
                (0, &Node::Mul(_, inner)) => inner,
                (1, &Node::Add(_, product)) => match arena[product] {
                    Node::Mul(inner, _) => inner,
                    ref other => panic!("level {level}: expected da * b, got {other:?}"),
                },
                (2, &Node::Add(first_axis, _)) => match arena[first_axis] {
                    Node::Mul(_, inner) => inner,
                    ref other => panic!("level {level}: expected a partial term, got {other:?}"),
                },
                (_, other) => panic!("level {level}: unexpected {other:?}"),
            };
        }
        assert!(matches!(
            &arena[tangent],
            Node::TangentStateVector { start: 0, end: 1 }
        ));
    }

    #[test]
    fn single_state_rules_match_their_closed_forms() {
        type Build = fn(&mut Arena, NodeId) -> NodeId;
        fn pow(arena: &mut Arena, x: NodeId, exponent: f64) -> NodeId {
            let exponent = arena.alloc(Node::Scalar(exponent));
            arena.alloc(Node::Pow(x, exponent))
        }
        fn linear(arena: &mut Arena, x: NodeId) -> NodeId {
            let data = InterpolantData::try_new(vec![0.0, 1.0, 2.0], vec![0.0, 10.0, 30.0]);
            arena.alloc(Node::Interpolant1DLinear {
                data: Box::new(data.expect("valid interpolant")),
                child: x,
            })
        }
        // p(dx) = 1 + 2*dx + 3*dx^2 + 4*dx^3 on [0, 5], extrapolated outside it.
        fn cubic(arena: &mut Arena, x: NodeId) -> NodeId {
            arena.alloc(Node::Interpolant1DCubic {
                data: Box::new(CubicInterpolantData {
                    breakpoints: vec![0.0, 5.0],
                    coeffs: vec![[1.0, 2.0, 3.0, 4.0]],
                }),
                child: x,
            })
        }
        let cases: [(&str, Build, f64, f64); 30] = [
            (
                "sin",
                |a, x| a.alloc(Node::Sin(x)),
                FRAC_PI_4,
                FRAC_PI_4.cos(),
            ),
            (
                "cos",
                |a, x| a.alloc(Node::Cos(x)),
                FRAC_PI_4,
                -FRAC_PI_4.sin(),
            ),
            (
                "tanh",
                |a, x| a.alloc(Node::Tanh(x)),
                1.0,
                1.0 - 1.0_f64.tanh().powi(2),
            ),
            ("sinh", |a, x| a.alloc(Node::Sinh(x)), 1.5, 1.5_f64.cosh()),
            ("cosh", |a, x| a.alloc(Node::Cosh(x)), 1.5, 1.5_f64.sinh()),
            (
                "arcsinh",
                |a, x| a.alloc(Node::Arcsinh(x)),
                2.0,
                1.0 / 5.0_f64.sqrt(),
            ),
            ("arctan", |a, x| a.alloc(Node::Arctan(x)), 1.0, 0.5),
            (
                "erf",
                |a, x| a.alloc(Node::Erf(x)),
                1.0,
                FRAC_2_SQRT_PI * (-1.0_f64).exp(),
            ),
            ("neg", |a, x| a.alloc(Node::Neg(x)), 5.0, -1.0),
            ("abs", |a, x| a.alloc(Node::Abs(x)), 5.0, 1.0),
            ("abs", |a, x| a.alloc(Node::Abs(x)), -5.0, -1.0),
            ("sqrt", |a, x| a.alloc(Node::Sqrt(x)), 4.0, 0.25),
            ("log", |a, x| a.alloc(Node::Log(x)), 2.0, 0.5),
            (
                "exp chain",
                |a, x| {
                    let two = a.alloc(Node::Scalar(2.0));
                    let two_x = a.alloc(Node::Mul(two, x));
                    a.alloc(Node::Exp(two_x))
                },
                1.0,
                2.0 * 2.0_f64.exp(),
            ),
            ("x^3", |a, x| pow(a, x, 3.0), 2.0, 12.0),
            ("x^2, negative base", |a, x| pow(a, x, 2.0), -3.0, -6.0),
            ("x^1.5", |a, x| pow(a, x, 1.5), 4.0, 3.0),
            ("x^0.5", |a, x| pow(a, x, 0.5), 9.0, 0.5 / 3.0),
            ("x * x, memoised", |a, x| a.alloc(Node::Mul(x, x)), 3.0, 6.0),
            ("linear interpolant", linear, 0.5, 10.0),
            ("linear interpolant", linear, 1.5, 20.0),
            ("linear interpolant, below", linear, -0.5, 10.0),
            ("linear interpolant, above", linear, 2.5, 20.0),
            ("cubic interpolant", cubic, 1.0, 20.0),
            ("cubic interpolant", cubic, 0.0, 2.0),
            ("cubic interpolant, below", cubic, -1.0, 8.0),
            ("cubic interpolant, above", cubic, 6.0, 470.0),
            (
                "conditional",
                |a, x| {
                    let selector = a.alloc(Node::Scalar(1.0));
                    let two = a.alloc(Node::Scalar(2.0));
                    let three = a.alloc(Node::Scalar(3.0));
                    let first = a.alloc(Node::Mul(two, x));
                    let second = a.alloc(Node::Mul(three, x));
                    a.alloc(Node::Conditional {
                        selector,
                        branches: vec![first, second],
                    })
                },
                5.0,
                2.0,
            ),
            ("constant", |a, _| a.alloc(Node::Scalar(42.0)), 1.0, 0.0),
            ("time", |a, _| a.alloc(Node::Time), 1.0, 0.0),
        ];
        for (label, build, x, expected) in cases {
            let mut arena = Arena::new();
            let state = arena.alloc(Node::StateVector { start: 0, end: 1 });
            let root = build(&mut arena, state);
            let got = jvp(&arena, root, &[x], &[], &[1.0])[0];
            assert!(
                (got - expected).abs() < 1e-12,
                "{label} at {x}: expected {expected}, got {got}"
            );
        }
    }

    #[test]
    fn two_state_rules_match_their_closed_forms() {
        type Build = fn(&mut Arena, NodeId, NodeId) -> NodeId;
        fn interpolant(arena: &mut Arena, x: NodeId, y: NodeId, data: NdInterpolantData) -> NodeId {
            arena.alloc(Node::InterpolantNd {
                data: Box::new(data),
                children: vec![x, y],
            })
        }
        let cases: [(&str, Build, [f64; 2], [f64; 2]); 10] = [
            (
                "product",
                |a, x, y| a.alloc(Node::Mul(x, y)),
                [2.0, 4.0],
                [4.0, 2.0],
            ),
            (
                "quotient",
                |a, x, y| a.alloc(Node::Div(x, y)),
                [1.0, 2.0],
                [0.5, -0.25],
            ),
            (
                "minimum",
                |a, x, y| a.alloc(Node::Minimum(x, y)),
                [1.0, 3.0],
                [1.0, 0.0],
            ),
            (
                "minimum",
                |a, x, y| a.alloc(Node::Minimum(x, y)),
                [5.0, 2.0],
                [0.0, 1.0],
            ),
            (
                "maximum",
                |a, x, y| a.alloc(Node::Maximum(x, y)),
                [5.0, 2.0],
                [1.0, 0.0],
            ),
            (
                "maximum",
                |a, x, y| a.alloc(Node::Maximum(x, y)),
                [1.0, 3.0],
                [0.0, 1.0],
            ),
            (
                "hypot",
                |a, x, y| a.alloc(Node::Hypot(x, y)),
                [3.0, 4.0],
                [0.6, 0.8],
            ),
            (
                "power",
                |a, x, y| a.alloc(Node::Pow(x, y)),
                [2.0, 3.0],
                [12.0, 8.0 * std::f64::consts::LN_2],
            ),
            // p = 3*dx0^2 + 5*dx1^3 + dx0*dx1 on one cell [0,4]x[0,4].
            (
                "N-D interpolant",
                |a, x, y| {
                    let mut coeffs = vec![0.0; 16];
                    coeffs[2 * 4] = 3.0;
                    coeffs[3] = 5.0;
                    coeffs[4 + 1] = 1.0;
                    let breakpoints = vec![vec![0.0, 4.0], vec![0.0, 4.0]];
                    let data = NdInterpolantData::try_new(breakpoints, coeffs, 4);
                    interpolant(a, x, y, data.expect("valid table"))
                },
                [2.0, 1.0],
                [13.0, 17.0],
            ),
            // p = 2 + 3*dx0 + 4*dx1 + 5*dx0*dx1 on one cell [0,2]x[0,2].
            (
                "bilinear interpolant",
                |a, x, y| {
                    let breakpoints = vec![vec![0.0, 2.0], vec![0.0, 2.0]];
                    let data = NdInterpolantData::try_new(breakpoints, vec![2.0, 4.0, 3.0, 5.0], 2);
                    interpolant(a, x, y, data.expect("valid table"))
                },
                [1.0, 2.0],
                [13.0, 9.0],
            ),
        ];
        for (label, build, y, expected) in cases {
            let mut arena = Arena::new();
            let x0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
            let x1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
            let root = build(&mut arena, x0, x1);
            let got = forward_gradient(&arena, root, &y, &[]);
            for (got, expected) in got.iter().zip(expected) {
                assert!(
                    (got - expected).abs() < 1e-10,
                    "{label} at {y:?}: expected {expected}, got {got}"
                );
            }
        }
    }

    #[test]
    fn rules_keep_the_range_of_a_primal_that_does_not_overflow() {
        // Each tangent is representable; squaring an operand took the first three to 0 or inf.
        type Build = fn(&mut Arena, NodeId, NodeId) -> NodeId;
        /// Label, root, states, seed and the closed-form tangent.
        type Case = (&'static str, Build, [f64; 2], [f64; 2], f64);
        let cases: [Case; 4] = [
            (
                "arcsinh",
                |a, x, _| a.alloc(Node::Arcsinh(x)),
                [1e200, 0.0],
                [1.0, 0.0],
                1e-200,
            ),
            (
                "hypot",
                |a, x, y| a.alloc(Node::Hypot(x, y)),
                [1e308, 1e308],
                [1.0, 1.0],
                std::f64::consts::SQRT_2,
            ),
            (
                "quotient",
                |a, x, y| a.alloc(Node::Div(x, y)),
                [1.0, 1e200],
                [1.0, 0.0],
                1e-200,
            ),
            // The quotient overflows, but a fixed divisor leaves `da / b` finite.
            (
                "quotient past overflow",
                |a, x, y| a.alloc(Node::Div(x, y)),
                [1e300, 1e-10],
                [1.0, 0.0],
                1e10,
            ),
        ];
        for (label, build, y, dy, expected) in cases {
            let mut arena = Arena::new();
            let x0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
            let x1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
            let root = build(&mut arena, x0, x1);
            let got = jvp(&arena, root, &y, &[], &dy)[0];
            assert!(
                (got - expected).abs() <= 1e-15 * expected,
                "{label} at {y:?}: expected {expected}, got {got}"
            );
        }
    }

    #[test]
    fn vector_rules_route_each_element() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let max = arena.alloc(Node::MaxReduce(y));
        let min = arena.alloc(Node::MinReduce(y));
        let (at, dy) = ([0.3, 0.9, 0.1], [10.0, 20.0, 30.0]);
        assert_eq!(jvp(&arena, max, &at, &[], &dy), [20.0], "argmax's tangent");
        assert_eq!(jvp(&arena, min, &at, &[], &dy), [30.0], "argmin's tangent");

        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 5 });
        let slice = arena.alloc(Node::Index {
            child: y,
            start: 1,
            end: 3,
        });
        let dy = [0.0, 1.0, 2.0, 0.0, 0.0];
        assert_eq!(jvp(&arena, slice, &[0.0; 5], &[], &dy), [1.0, 2.0]);

        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let y = arena.alloc(Node::StateVector { start: 2, end: 4 });
        let joined = arena.alloc(Node::Concat(vec![x, y]));
        let dy = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(jvp(&arena, joined, &[0.0; 4], &[], &dy), dy);

        let mut arena = Arena::new();
        let sparse = arena.alloc(Node::SparseMatrix(Box::new(CsrData {
            indptr: vec![0, 1, 2],
            indices: vec![0, 1],
            data: vec![2.0, 3.0],
            shape: Shape::matrix(2, 3),
        })));
        let v = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let product = arena.alloc(Node::MatMul(sparse, v));
        let dy = [1.0, 0.0, 0.0];
        assert_eq!(jvp(&arena, product, &[0.0; 3], &[], &dy), [2.0, 0.0]);
    }

    #[test]
    fn test_parameter_differentiation() {
        let mut arena = Arena::new();
        let p = arena.alloc(Node::InputParameter {
            name: "k".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let expr = arena.alloc(Node::Mul(p, x));
        let compiled = compile_tangent_pipeline(&arena, expr, true);
        let mut scratch = vec![0.0; compiled.scratch_len()];
        let seed = TangentInputs {
            dy: None,
            dp: Some(&[1.0]),
        };
        let result = compiled.eval_with_tangent(&mut scratch, 0.0, &[3.0], &[], &[2.0], &seed);
        assert_eq!(result, [3.0]);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn a_zero_width_input_has_a_zero_width_tangent() {
        // A one-row tangent made `d[p, y0]/dp` two rows wide, its top row a spurious 1.
        let mut arena = Arena::new();
        let p = arena.alloc(Node::InputParameter {
            name: "p".to_string(),
            index: 0,
            offset: 0,
            width: 0,
        });
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let root = arena.alloc(Node::Concat(vec![p, y0]));
        let compiled = compile_tangent_pipeline(&arena, root, true);
        let mut scratch = vec![0.0; compiled.scratch_len()];
        let seed = TangentInputs {
            dy: None,
            dp: Some(&[1.0]),
        };
        let result = compiled.eval_with_tangent(&mut scratch, 0.0, &[3.0], &[], &[], &seed);
        assert_eq!(result, [0.0]);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn modulo_tangent_keeps_a_scalar_dividends_broadcast_width() {
        // y0 mod [2, 3, 5] is three wide; summing it through a matrix needs all three rows.
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let divisors = arena.alloc(Node::Array(Box::new(
            ArrayData::try_new(vec![2.0, 3.0, 5.0], Shape::vector(3)).expect("valid array"),
        )));
        let remainders = arena.alloc(Node::Modulo(y0, divisors));
        let ones = arena.alloc(Node::SparseMatrix(Box::new(
            CsrData::try_new(vec![0, 3], vec![0, 1, 2], vec![1.0; 3], Shape::matrix(1, 3))
                .expect("valid row matrix"),
        )));
        let root = arena.alloc(Node::MatMul(ones, remainders));

        let jac = tangent_wrt_states(&mut arena, root);
        let jac = simplify(&mut arena, jac);
        let compiled = CompiledExpr::new(DagRef::new(&arena, jac));
        let mut scratch = vec![0.0; compiled.scratch_len()];
        let tangent = TangentInputs {
            dy: Some(&[1.0]),
            dp: None,
        };
        let result = compiled.eval_with_tangent(&mut scratch, 0.0, &[7.5], &[], &[], &tangent);
        assert_eq!(result, &[3.0]);
    }

    #[test]
    fn modulo_tangent_differentiates_an_input_divisor() {
        // d(y0 mod p)/dp is -trunc(y0 / p), a positive step for a negative dividend.
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let p = arena.alloc(Node::InputParameter {
            name: "p".into(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let root = arena.alloc(Node::Modulo(y0, p));

        let tp = compile_tangent_pipeline(&arena, root, true);
        let tangent = TangentInputs {
            dy: None,
            dp: Some(&[1.0]),
        };
        for (y, expected) in [(2.0, -1.0), (3.0, -2.0), (0.5, 0.0), (-2.0, 1.0)] {
            let mut s = vec![0.0; tp.scratch_len()];
            let r = tp.eval_with_tangent(&mut s, 0.0, &[y], &[], &[1.3], &tangent);
            assert!(
                (r[0] - expected).abs() < 1e-12,
                "y0 = {y}: {} != {expected}",
                r[0]
            );
        }
    }

    #[test]
    fn test_tangent_wrt_subset() {
        let mut arena = Arena::new();
        let x0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let x1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let expr = arena.alloc(Node::Mul(x0, x1));

        let deriv = tangent_wrt_subset(&mut arena, expr, &[1]);
        let deriv = simplify(&mut arena, deriv);

        let topo = arena.topological_order(deriv);
        for &nid in &topo {
            if let Node::TangentStateVector { start, .. } = arena[nid] {
                assert_ne!(start, 0, "Should not differentiate w.r.t. x0");
            }
        }
        assert!(
            topo.iter()
                .any(|&nid| matches!(&arena[nid], Node::TangentStateVector { start: 1, end: 2 })),
            "Should contain TangentStateVector for x1"
        );
    }

    #[test]
    fn overlap_needs_one_active_index_inside_the_range() {
        let active = [2, 5, 9];
        assert!(overlaps(&active, 0, 3), "2 lies in 0..3");
        assert!(overlaps(&active, 5, 6));
        assert!(!overlaps(&active, 3, 5), "end is exclusive");
        assert!(!overlaps(&active, 10, 12));
        assert!(!overlaps(&active, 5, 5), "an empty range holds nothing");
        assert!(!overlaps(&[], 0, 4));
    }

    #[test]
    #[should_panic(expected = "active state indices must ascend")]
    fn tangent_wrt_subset_rejects_unsorted_indices() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let _ = tangent_wrt_subset(&mut arena, x, &[1, 0]);
    }

    #[test]
    fn test_tangent_wrt_subset_numerical() {
        let mut arena = Arena::new();
        let x0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let x1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let x0_sq = arena.alloc(Node::Mul(x0, x0));
        let three = arena.alloc(Node::Scalar(3.0));
        let three_x1 = arena.alloc(Node::Mul(three, x1));
        let expr = arena.alloc(Node::Add(x0_sq, three_x1));

        let deriv = tangent_wrt_subset(&mut arena, expr, &[1]);
        let deriv = simplify(&mut arena, deriv);
        let compiled = CompiledExpr::new(DagRef::new(&arena, deriv));
        let mut s_compiled = vec![0.0; compiled.scratch_len()];

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
        let expr = arena.alloc(Node::Mul(sum, x0));

        let mut arena_full = arena.clone();
        let full = tangent_wrt_states(&mut arena_full, expr);
        let full = simplify(&mut arena_full, full);
        let compiled_full = CompiledExpr::new(DagRef::new(&arena_full, full));
        let mut s_compiled_full = vec![0.0; compiled_full.scratch_len()];

        let subset = tangent_wrt_subset(&mut arena, expr, &[0, 1]);
        let subset = simplify(&mut arena, subset);
        let compiled_subset = CompiledExpr::new(DagRef::new(&arena, subset));
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
