//! Expression simplification, CSE and DCE, chiefly to keep derivative expressions small.
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

use crate::arena::{Arena, Dag, DagRef, NodeId, NodeMap};
use crate::ir::{BinaryOp, UnaryOp};
use crate::node::{ArrayData, Node, Shape};

/// Constant-fold and apply algebraic identities, as `CasADi`'s SX simplifies on the fly.
///
/// Like `CasADi`, it folds `0 * x`, `0 / x` and `x - x` to zero and `x / x` to one, so the
/// result matches the input only where every value is finite, up to the sign of zero.
/// Constant powers lower only where the rewrite takes `pow`'s value bitwise.
#[must_use]
pub fn simplify(arena: &mut Arena, root: NodeId) -> NodeId {
    let mut simplified: NodeMap<NodeId> = NodeMap::new(arena.len());
    let mut lens: NodeMap<usize> = NodeMap::new(arena.len());
    for id in arena.topological_order(root) {
        let result = simplify_node(arena, id, &simplified, &mut lens);
        simplified.insert(id, result);
    }
    simplified.rebuilt(root)
}

/// Exact literal match, so tiny nonzero coefficients (e.g. 1e-17) are never treated as 0 or 1.
fn is_scalar(arena: &Arena, id: NodeId, value: f64) -> bool {
    arena.get(id).as_scalar() == Some(value)
}

fn is_zero(arena: &Arena, id: NodeId) -> bool {
    is_scalar(arena, id, 0.0)
}

fn is_one(arena: &Arena, id: NodeId) -> bool {
    is_scalar(arena, id, 1.0)
}

fn is_neg_one(arena: &Arena, id: NodeId) -> bool {
    is_scalar(arena, id, -1.0)
}

/// [`node_width`](crate::ir::node_width) of a node, memoized via `lens`.
///
/// Walks with an explicit stack, so graph depth is bounded by the heap.
pub fn node_len(arena: &Arena, id: NodeId, lens: &mut NodeMap<usize>) -> usize {
    if let Some(&len) = lens.get(id) {
        return len;
    }
    let mut pending = vec![id];
    while let Some(&top) = pending.last() {
        if lens.contains_key(top) {
            pending.pop();
            continue;
        }
        let n_pending = pending.len();
        let len = crate::ir::node_width(arena, arena.get(top), |child| {
            lens.get(child).copied().unwrap_or_else(|| {
                pending.push(child);
                0
            })
        });
        // Valid only if every child it read was already measured.
        if pending.len() == n_pending {
            pending.pop();
            lens.insert(top, len);
        }
    }
    *lens.get(id).expect("the walk ends with `id` measured")
}

/// Simplify `id`, whose children are already simplified in `simplified`.
///
/// A node no rule rewrites is reused, or copied over its simplified children.
fn simplify_node(
    arena: &mut Arena,
    id: NodeId,
    simplified: &NodeMap<NodeId>,
    lens: &mut NodeMap<usize>,
) -> NodeId {
    let s = |child: NodeId| simplified.rebuilt(child);
    let rewritten = match *arena.get(id) {
        Node::Add(lhs, rhs) => simplify_add(arena, s(lhs), s(rhs)),
        Node::Sub(lhs, rhs) => simplify_sub(arena, s(lhs), s(rhs), lens),
        Node::Mul(lhs, rhs) => simplify_mul(arena, s(lhs), s(rhs), lens),
        Node::Div(lhs, rhs) => simplify_div(arena, s(lhs), s(rhs), lens),
        Node::Pow(base, exp) => simplify_pow(arena, s(base), s(exp), lens),
        Node::Minimum(lhs, rhs) => fold_binary(arena, s(lhs), s(rhs), BinaryOp::Minimum),
        Node::Maximum(lhs, rhs) => fold_binary(arena, s(lhs), s(rhs), BinaryOp::Maximum),
        Node::Modulo(lhs, rhs) => fold_binary(arena, s(lhs), s(rhs), BinaryOp::Modulo),
        Node::Hypot(lhs, rhs) => fold_binary(arena, s(lhs), s(rhs), BinaryOp::Hypot),

        Node::Neg(child) => simplify_neg(arena, s(child)),
        Node::Abs(child) => fold_unary(arena, s(child), UnaryOp::Abs),
        Node::Sqrt(child) => fold_unary(arena, s(child), UnaryOp::Sqrt),
        Node::Exp(child) => fold_unary(arena, s(child), UnaryOp::Exp),
        Node::Log(child) => fold_unary(arena, s(child), UnaryOp::Log),
        Node::Sin(child) => fold_unary(arena, s(child), UnaryOp::Sin),
        Node::Cos(child) => fold_unary(arena, s(child), UnaryOp::Cos),
        Node::Tanh(child) => fold_unary(arena, s(child), UnaryOp::Tanh),
        Node::Sinh(child) => fold_unary(arena, s(child), UnaryOp::Sinh),
        Node::Cosh(child) => fold_unary(arena, s(child), UnaryOp::Cosh),
        Node::Arcsinh(child) => fold_unary(arena, s(child), UnaryOp::Arcsinh),
        Node::Arctan(child) => fold_unary(arena, s(child), UnaryOp::Arctan),
        Node::Erf(child) => fold_unary(arena, s(child), UnaryOp::Erf),
        Node::Sign(child) => fold_unary(arena, s(child), UnaryOp::Sign),
        Node::Floor(child) => fold_unary(arena, s(child), UnaryOp::Floor),
        Node::Ceiling(child) => fold_unary(arena, s(child), UnaryOp::Ceiling),

        Node::ReduceArgSelect { basis, .. }
            if is_zero(arena, s(basis))
                || matches!(arena.get(s(basis)), Node::ZeroVector { .. }) =>
        {
            Some(arena.alloc(Node::Scalar(0.0)))
        },

        Node::Scalar(_)
        | Node::Array(_)
        | Node::ZeroVector { .. }
        | Node::SparseMatrix(_)
        | Node::StateVector { .. }
        | Node::StateVectorDot { .. }
        | Node::InputParameter { .. }
        | Node::Time
        | Node::TangentStateVector { .. }
        | Node::TangentParameter { .. }
        | Node::MatMul(..)
        | Node::EqualHeaviside(..)
        | Node::NotEqualHeaviside(..)
        | Node::Equality(..)
        | Node::MaxReduce(_)
        | Node::MinReduce(_)
        | Node::ReduceArgSelect { .. }
        | Node::Index { .. }
        | Node::Concat(_)
        | Node::Interpolant1DLinear { .. }
        | Node::Interpolant1DLinearDeriv { .. }
        | Node::Interpolant1DCubic { .. }
        | Node::Interpolant1DCubicDeriv { .. }
        | Node::InterpolantNd { .. }
        | Node::InterpolantNdPartial { .. }
        | Node::Conditional { .. } => None,
    };
    rewritten.unwrap_or_else(|| reuse_or_rebuild(arena, id, simplified))
}

/// `id` itself when no child changed, else a copy of it over the simplified children.
fn reuse_or_rebuild(arena: &mut Arena, id: NodeId, simplified: &NodeMap<NodeId>) -> NodeId {
    let node = arena.get(id);
    let mut changed = false;
    node.for_each_child(|child| changed |= simplified.rebuilt(child) != child);
    if !changed {
        return id;
    }
    let copy = node.map_children(|child| simplified.rebuilt(child));
    arena.alloc(copy)
}

/// `const op const -> const`, valued exactly as evaluation would.
fn fold_binary(arena: &mut Arena, lhs: NodeId, rhs: NodeId, op: BinaryOp) -> Option<NodeId> {
    let (a, b) = (arena.get(lhs).as_scalar()?, arena.get(rhs).as_scalar()?);
    Some(arena.alloc(Node::Scalar(op.apply(a, b))))
}

/// `op(const) -> const`, valued exactly as evaluation would.
fn fold_unary(arena: &mut Arena, child: NodeId, op: UnaryOp) -> Option<NodeId> {
    let v = arena.get(child).as_scalar()?;
    Some(arena.alloc(Node::Scalar(op.apply(v))))
}

/// `x + 0 -> x`, `0 + x -> x`, `const + const -> const`.
fn simplify_add(arena: &mut Arena, lhs: NodeId, rhs: NodeId) -> Option<NodeId> {
    if let Some(sum) = fold_binary(arena, lhs, rhs, BinaryOp::Add) {
        return Some(sum);
    }

    if is_zero(arena, lhs) {
        return Some(rhs);
    }

    if is_zero(arena, rhs) {
        return Some(lhs);
    }

    None
}

/// `x - 0 -> x`, `0 - x -> -x`, `x - x -> 0`, `const - const -> const`.
fn simplify_sub(
    arena: &mut Arena,
    lhs: NodeId,
    rhs: NodeId,
    lens: &mut NodeMap<usize>,
) -> Option<NodeId> {
    if let Some(difference) = fold_binary(arena, lhs, rhs, BinaryOp::Sub) {
        return Some(difference);
    }

    if is_zero(arena, rhs) {
        return Some(lhs);
    }

    if is_zero(arena, lhs) {
        return Some(arena.alloc(Node::Neg(rhs)));
    }

    if lhs == rhs {
        let len = node_len(arena, lhs, lens);
        return Some(arena.alloc(Node::zero(len)));
    }

    None
}

/// `1 * x -> x`, `x * 1 -> x`, `-1 * x -> -x`, `0 * x -> 0`, `const * const -> const`.
fn simplify_mul(
    arena: &mut Arena,
    lhs: NodeId,
    rhs: NodeId,
    lens: &mut NodeMap<usize>,
) -> Option<NodeId> {
    if let Some(product) = fold_binary(arena, lhs, rhs, BinaryOp::Mul) {
        return Some(product);
    }

    if is_one(arena, lhs) {
        return Some(rhs);
    }

    if is_one(arena, rhs) {
        return Some(lhs);
    }

    if is_neg_one(arena, lhs) {
        return Some(arena.alloc(Node::Neg(rhs)));
    }

    if is_neg_one(arena, rhs) {
        return Some(arena.alloc(Node::Neg(lhs)));
    }

    if is_zero(arena, lhs) || is_zero(arena, rhs) {
        let len = node_len(arena, lhs, lens).max(node_len(arena, rhs, lens));
        return Some(arena.alloc(Node::zero(len)));
    }

    None
}

/// `x / 1 -> x`, `0 / x -> 0`, `x / x -> 1`, `const / const -> const`.
fn simplify_div(
    arena: &mut Arena,
    lhs: NodeId,
    rhs: NodeId,
    lens: &mut NodeMap<usize>,
) -> Option<NodeId> {
    if let Some(quotient) = fold_binary(arena, lhs, rhs, BinaryOp::Div) {
        return Some(quotient);
    }

    if is_one(arena, rhs) {
        return Some(lhs);
    }

    if is_zero(arena, lhs) {
        let len = node_len(arena, rhs, lens);
        return Some(arena.alloc(Node::zero(len)));
    }

    // Scalar only: a vector x / x is a ones vector.
    if lhs == rhs && node_len(arena, lhs, lens) == 1 {
        return Some(arena.alloc(Node::Scalar(1.0)));
    }

    None
}

/// `const^const -> const`, `1^x -> 1`, and `x^const` as [`lower_constant_pow`] rewrites it.
fn simplify_pow(
    arena: &mut Arena,
    base: NodeId,
    exp: NodeId,
    lens: &mut NodeMap<usize>,
) -> Option<NodeId> {
    if let Some(power) = fold_binary(arena, base, exp, BinaryOp::Pow) {
        return Some(power);
    }

    // Scalar only: 1^x broadcasts to the exponent's length.
    if is_one(arena, base) && node_len(arena, exp, lens) == 1 {
        return Some(base);
    }

    let e = arena.get(exp).as_scalar()?;
    let len = node_len(arena, base, lens);
    lower_constant_pow(arena, base, e, len)
}

/// Rewrite `base^e`, `base` of length `len`, where the rewrite takes `pow`'s value bitwise.
///
/// Only `e` in {0, 1, 2, -1} qualify: `1`, `x`, `x*x` and `1/x` round once from exact
/// operands, so equal a correctly rounded `pow` for every base, special values included, and
/// the evaluator's `pow` takes the same forms. A chain such as `x*(x*x)` rounds twice, and
/// `sqrt(x)` differs at `-0` and `-inf`.
#[allow(clippy::float_cmp)]
fn lower_constant_pow(arena: &mut Arena, base: NodeId, e: f64, len: usize) -> Option<NodeId> {
    // A zero-width base against the scalar exponent is invalid; `x * x` would make it valid.
    if len == 0 {
        return None;
    }
    if e == 0.0 {
        let ones = if len == 1 {
            Node::Scalar(1.0)
        } else {
            let ones = ArrayData::try_new(vec![1.0; len], Shape::vector(len))
                .expect("a vector's shape matches its length");
            Node::Array(Box::new(ones))
        };
        return Some(arena.alloc(ones));
    }
    if e == 1.0 {
        return Some(base);
    }
    if e == 2.0 {
        return Some(arena.alloc(Node::Mul(base, base)));
    }
    if e == -1.0 {
        let one = arena.alloc(Node::Scalar(1.0));
        return Some(arena.alloc(Node::Div(one, base)));
    }
    None
}

/// `-const -> const`, `--x -> x`.
fn simplify_neg(arena: &mut Arena, child: NodeId) -> Option<NodeId> {
    if let Some(negated) = fold_unary(arena, child, UnaryOp::Neg) {
        return Some(negated);
    }

    match *arena.get(child) {
        Node::Neg(inner) => Some(inner),
        _ => None,
    }
}

/// Common subexpression elimination into a new arena of unique reachable nodes.
///
/// The output is laid out in [`dce`]'s order, so a `dce` after it is the identity.
#[must_use]
pub fn cse(dag: DagRef<'_>) -> Dag {
    use rustc_hash::FxHashMap;
    use rustc_hash::FxHasher;
    use std::hash::Hasher;

    use crate::node::{structural_hash, structurally_identical};

    let (arena, root) = (dag.arena(), dag.root());
    let mut new_arena = Arena::new();
    let mut old_to_new: NodeMap<NodeId> = NodeMap::new(arena.len());
    // Newest node per hash; each node links to the one it displaced, by new id.
    let mut newest_with_hash: FxHashMap<u64, NodeId> = FxHashMap::default();
    let mut older_with_hash: Vec<Option<NodeId>> = Vec::new();

    for old_id in arena.topological_order(root) {
        let remapped = arena.get(old_id).map_children(|c| old_to_new.rebuilt(c));
        let mut hasher = FxHasher::default();
        structural_hash(&remapped, &mut hasher);
        let hash = hasher.finish();

        let canonical = std::iter::successors(newest_with_hash.get(&hash).copied(), |id| {
            older_with_hash[id.index()]
        })
        .find(|&existing| structurally_identical(new_arena.get(existing), &remapped));
        let new_id = canonical.unwrap_or_else(|| {
            let new_id = new_arena.alloc(remapped);
            older_with_hash.push(newest_with_hash.insert(hash, new_id));
            new_id
        });
        old_to_new.insert(old_id, new_id);
    }

    Dag::new(new_arena, old_to_new.rebuilt(root))
}

/// Compile-prep pipeline: simplify -> zero propagation -> CSE.
///
/// The result is dead-code free: CSE keeps only what the root reaches.
#[must_use]
pub fn simplify_pipeline(mut dag: Dag) -> Dag {
    use crate::zero_propagate::zero_propagate;

    dag.rewrite_in_place(simplify);
    let propagated = zero_propagate(dag.as_ref());
    // The largest of the three arenas, so free it before `cse` builds its own.
    drop(dag);
    cse(propagated.as_ref())
}

/// Dead code elimination into a new arena of nodes reachable from the root.
#[must_use]
pub fn dce(dag: DagRef<'_>) -> Dag {
    let (arena, root) = (dag.arena(), dag.root());
    let mut new_arena = Arena::new();
    let mut old_to_new: NodeMap<NodeId> = NodeMap::new(arena.len());

    for old_id in arena.topological_order(root) {
        let copy = arena.get(old_id).map_children(|c| old_to_new.rebuilt(c));
        old_to_new.insert(old_id, new_arena.alloc(copy));
    }

    Dag::new(new_arena, old_to_new.rebuilt(root))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::Node;

    #[test]
    fn simplify_pipeline_matches_manual_pass_order() {
        use crate::zero_propagate::zero_propagate;

        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let s = arena.alloc(Node::Sin(y0));
        let one = arena.alloc(Node::Scalar(1.0));
        let zero = arena.alloc(Node::Scalar(0.0));
        let m = arena.alloc(Node::Mul(s, one));
        let decorated = arena.alloc(Node::Add(m, zero));

        let simplified = simplify_pipeline(Dag::new(arena.clone(), decorated));

        let mut manual = arena;
        let r = simplify(&mut manual, decorated);
        let propagated = zero_propagate(DagRef::new(&manual, r));
        let shared = cse(propagated.as_ref());
        let pruned = dce(shared.as_ref());

        assert_eq!(simplified.arena().len(), pruned.arena().len());
        assert_eq!(simplified.root(), pruned.root());
        assert_eq!(
            simplified.arena().get(simplified.root()),
            pruned.arena().get(pruned.root())
        );
    }

    #[test]
    fn cse_output_is_already_dead_code_free() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let y_twin = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let two = arena.alloc(Node::Scalar(2.0));
        let two_twin = arena.alloc(Node::Scalar(2.0));
        let sin_a = arena.alloc(Node::Sin(y));
        let sin_b = arena.alloc(Node::Sin(y_twin));
        let exp_a = arena.alloc(Node::Exp(sin_a));
        let exp_b = arena.alloc(Node::Exp(sin_b));
        let _dead = arena.alloc(Node::Cos(y));
        let scaled_a = arena.alloc(Node::Mul(two, exp_a));
        let scaled_b = arena.alloc(Node::Mul(two_twin, exp_b));
        // The same operands under two ops, which must stay apart.
        let sum = arena.alloc(Node::Add(scaled_a, sin_a));
        let difference = arena.alloc(Node::Sub(scaled_b, sin_b));
        let root = arena.alloc(Node::Concat(vec![sum, difference, sin_b]));

        let shared = cse(DagRef::new(&arena, root));
        let pruned = dce(shared.as_ref());

        assert_eq!(
            shared.arena().len(),
            8,
            "duplicates merged, dead node dropped"
        );
        let out = shared.arena();
        let Node::Concat(parts) = out.get(shared.root()) else {
            panic!("the concat must survive")
        };
        let (&Node::Add(scaled, sin), &Node::Sub(scaled_twin, sin_twin)) =
            (out.get(parts[0]), out.get(parts[1]))
        else {
            panic!("the Add and the Sub must stay apart")
        };
        assert_eq!((scaled, sin), (scaled_twin, sin_twin), "nested twins merge");
        assert_eq!(parts[2], sin);
        assert_eq!(shared.arena().nodes(), pruned.arena().nodes());
        assert_eq!(shared.root(), pruned.root());
    }

    /// Deeper than a recursive walk survives on a 2 MiB test-thread stack.
    const DEEP: usize = 200_000;

    #[test]
    fn node_len_handles_a_deep_chain() {
        let mut arena = Arena::new();
        let mut id = arena.alloc(Node::StateVector { start: 0, end: 3 });
        for _ in 0..DEEP {
            id = arena.alloc(Node::Sin(id));
        }
        let mut lens = NodeMap::new(arena.len());
        assert_eq!(node_len(&arena, id, &mut lens), 3);
    }

    #[test]
    fn a_zero_width_conditional_measures_zero_as_lowering_does() {
        let mut arena = Arena::new();
        let selector = arena.alloc(Node::Scalar(1.0));
        let empty = arena.alloc(Node::StateVector { start: 0, end: 0 });
        let cond = arena.alloc(Node::Conditional {
            selector,
            branches: vec![empty, empty],
        });
        let ratio = arena.alloc(Node::Div(cond, cond));

        let lowered = crate::ir::infer_sizes(&arena, &arena.topological_order(ratio));
        let mut lens = NodeMap::new(arena.len());
        assert_eq!(node_len(&arena, cond, &mut lens), lowered[cond.index()]);

        let simplified = simplify(&mut arena, ratio);
        assert!(
            matches!(arena.get(simplified), Node::Div(..)),
            "a zero-width x / x is not the scalar 1"
        );
    }

    #[test]
    fn simplify_handles_a_deep_chain() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let one = arena.alloc(Node::Scalar(1.0));
        let mut id = y;
        for _ in 0..DEEP {
            let sin = arena.alloc(Node::Sin(id));
            id = arena.alloc(Node::Mul(sin, one));
        }

        let mut root = simplify(&mut arena, id);
        for _ in 0..DEEP {
            let Node::Sin(child) = *arena.get(root) else {
                panic!("every `* 1` must fold away, leaving the sines")
            };
            root = child;
        }
        assert_eq!(root, y);
    }

    /// `x^e` over a scalar state `x`, simplified and compiled once.
    fn simplified_pow_of_state(e: f64) -> impl FnMut(f64) -> f64 {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let exp = arena.alloc(Node::Scalar(e));
        let expr = arena.alloc(Node::Pow(x, exp));
        let simplified = simplify(&mut arena, expr);
        let compiled = crate::eval::CompiledExpr::new(DagRef::new(&arena, simplified));
        let mut scratch = vec![0.0; compiled.scratch_len()];
        move |y| compiled.eval(&mut scratch, 0.0, &[y], &[], &[])[0]
    }

    /// `id` rendered infix, with the leaf `x` as `x`.
    fn render(arena: &Arena, id: NodeId, x: NodeId) -> String {
        let r = |child: NodeId| render(arena, child, x);
        match arena.get(id) {
            _ if id == x => "x".to_string(),
            Node::Scalar(v) => v.to_string(),
            Node::Array(ones) => format!("{:?}", ones.data),
            Node::ZeroVector { len } => format!("zeros({len})"),
            Node::Neg(a) => format!("-{}", r(*a)),
            Node::Sqrt(a) => format!("sqrt({})", r(*a)),
            Node::Add(a, b) => format!("({}+{})", r(*a), r(*b)),
            Node::Sub(a, b) => format!("({}-{})", r(*a), r(*b)),
            Node::Mul(a, b) => format!("({}*{})", r(*a), r(*b)),
            Node::Div(a, b) => format!("({}/{})", r(*a), r(*b)),
            Node::Pow(a, b) => format!("pow({},{})", r(*a), r(*b)),
            other => panic!("unexpected {other:?}"),
        }
    }

    /// The node `build` makes over a state `x` of length `len`, simplified and rendered.
    fn simplified(len: usize, build: impl FnOnce(&mut Arena, NodeId) -> Node) -> String {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: len });
        let node = build(&mut arena, x);
        let expr = arena.alloc(node);
        let root = simplify(&mut arena, expr);
        render(&arena, root, x)
    }

    /// `x` raised to a constant, simplified and rendered infix over the leaf `x`.
    fn lowered_pow(len: usize, e: f64) -> String {
        simplified(len, |arena, x| Node::Pow(x, arena.alloc(Node::Scalar(e))))
    }

    #[test]
    fn identities_fold_as_casadi_folds_them() {
        type Build = fn(&mut Arena, NodeId) -> Node;
        fn lit(arena: &mut Arena, value: f64) -> NodeId {
            arena.alloc(Node::Scalar(value))
        }
        let cases: [(usize, Build, &str); 26] = [
            (1, |a, x| Node::Mul(x, lit(a, 1.0)), "x"),
            (1, |a, x| Node::Mul(lit(a, 1.0), x), "x"),
            (1, |a, x| Node::Add(x, lit(a, 0.0)), "x"),
            (1, |a, x| Node::Add(lit(a, 0.0), x), "x"),
            (1, |a, x| Node::Sub(x, lit(a, 0.0)), "x"),
            (1, |a, x| Node::Sub(x, lit(a, -0.0)), "x"),
            (1, |a, x| Node::Sub(lit(a, 0.0), x), "-x"),
            (1, |a, x| Node::Sub(lit(a, -0.0), x), "-x"),
            (1, |a, x| Node::Div(x, lit(a, 1.0)), "x"),
            (1, |a, x| Node::Mul(lit(a, -1.0), x), "-x"),
            (1, |a, x| Node::Mul(x, lit(a, -1.0)), "-x"),
            (1, |a, x| Node::Pow(lit(a, 1.0), x), "1"),
            (1, |a, x| Node::Neg(a.alloc(Node::Neg(x))), "x"),
            (1, |a, x| Node::Mul(lit(a, 0.0), x), "0"),
            (1, |a, x| Node::Mul(x, lit(a, 0.0)), "0"),
            (1, |a, x| Node::Div(lit(a, 0.0), x), "0"),
            (1, |_, x| Node::Sub(x, x), "0"),
            (1, |_, x| Node::Div(x, x), "1"),
            // An exact match only: a tiny coefficient is not zero.
            (
                1,
                |a, x| Node::Mul(lit(a, 1e-17), x),
                "(0.00000000000000001*x)",
            ),
            (3, |a, x| Node::Mul(lit(a, 0.0), x), "zeros(3)"),
            (3, |a, x| Node::Mul(x, lit(a, 0.0)), "zeros(3)"),
            (3, |_, x| Node::Sub(x, x), "zeros(3)"),
            (3, |a, x| Node::Div(lit(a, 0.0), x), "zeros(3)"),
            // A vector `x / x` is a ones vector and `1^x` broadcasts, so neither folds.
            (3, |_, x| Node::Div(x, x), "(x/x)"),
            (3, |a, x| Node::Pow(lit(a, 1.0), x), "pow(1,x)"),
            (3, |a, x| Node::Sub(x, lit(a, -0.0)), "x"),
        ];
        for (len, build, expected) in cases {
            let mut input = Arena::new();
            let x = input.alloc(Node::StateVector { start: 0, end: len });
            let node = build(&mut input, x);
            let expr = input.alloc(node);
            let label = render(&input, expr, x);
            assert_eq!(
                simplified(len, build),
                expected,
                "{label}, x of length {len}"
            );
        }
    }

    #[test]
    fn constant_powers_lower_only_where_exact() {
        let cases = [
            (1, 0.0, "1"),
            (1, 1.0, "x"),
            (1, 2.0, "(x*x)"),
            (1, -1.0, "(1/x)"),
            (1, 3.0, "pow(x,3)"),
            (1, -2.0, "pow(x,-2)"),
            (1, 0.5, "pow(x,0.5)"),
            (1, -0.5, "pow(x,-0.5)"),
            (1, 0.25, "pow(x,0.25)"),
            (3, 0.0, "[1.0, 1.0, 1.0]"),
            (3, 1.0, "x"),
            (3, 2.0, "(x*x)"),
            (3, -1.0, "(1/x)"),
            (3, 4.0, "pow(x,4)"),
            (3, 0.5, "pow(x,0.5)"),
            (0, 2.0, "pow(x,2)"),
        ];
        for (len, e, expected) in cases {
            assert_eq!(lowered_pow(len, e), expected, "x^{e}, x of length {len}");
        }
    }

    /// Whatever `simplify` makes of a constant power takes `pow`'s value bit for bit, at
    /// special values and across the binade range; a NaN need only meet a NaN.
    #[test]
    fn lowered_powers_take_pow_values_bitwise() {
        let special = [
            0.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            5e-324,
            -5e-324,
            f64::MAX,
            -f64::MAX,
            1.0,
            -1.0,
            1.5e154,
            3e-162,
        ];
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let swept = std::iter::repeat_with(|| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            f64::from_bits(state)
        });
        let bases: Vec<f64> = special.into_iter().chain(swept.take(20_000)).collect();
        let exponents = [
            0.0, -0.0, 1.0, 2.0, -1.0, 3.0, 4.0, -2.0, 0.5, -0.5, 0.25, -0.25, 1.5, 100.0, -100.0,
        ];
        for e in exponents {
            let mut lowered = simplified_pow_of_state(e);
            for &y in &bases {
                let (got, want) = (lowered(y), BinaryOp::Pow.apply(y, e));
                assert!(
                    got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
                    "y={y:e}, e={e}: lowered {got:e} != pow {want:e}"
                );
            }
        }
    }

    #[test]
    fn constant_operands_fold_to_their_value() {
        type Binary = fn(NodeId, NodeId) -> Node;
        type Unary = fn(NodeId) -> Node;

        let binary: [(Binary, f64, f64, f64); 5] = [
            (Node::Add, 2.0, 3.0, 5.0),
            (Node::Sub, 5.0, 3.0, 2.0),
            (Node::Mul, 3.0, 4.0, 12.0),
            (Node::Div, 12.0, 4.0, 3.0),
            (Node::Pow, 2.0, 3.0, 8.0),
        ];
        for (op, a, b, expected) in binary {
            let mut arena = Arena::new();
            let (lhs, rhs) = (arena.alloc(Node::Scalar(a)), arena.alloc(Node::Scalar(b)));
            let expr = arena.alloc(op(lhs, rhs));
            let simplified = simplify(&mut arena, expr);
            assert_eq!(
                arena.get(simplified),
                &Node::Scalar(expected),
                "{:?}",
                arena.get(expr)
            );
        }

        // Bitwise against the interpreter's op too: `sign(0)` is 0, not `f64::signum`'s 1.
        let unary: [(Unary, UnaryOp, f64, f64); 8] = [
            (Node::Neg, UnaryOp::Neg, 5.0, -5.0),
            (Node::Sin, UnaryOp::Sin, 0.0, 0.0),
            (Node::Cos, UnaryOp::Cos, 0.0, 1.0),
            (Node::Exp, UnaryOp::Exp, 0.0, 1.0),
            (Node::Sqrt, UnaryOp::Sqrt, 4.0, 2.0),
            (Node::Erf, UnaryOp::Erf, 0.0, 0.0),
            (Node::Erf, UnaryOp::Erf, 0.5, 0.520_499_877_813_046_5),
            (Node::Sign, UnaryOp::Sign, 0.0, 0.0),
        ];
        for (node, op, x, expected) in unary {
            let mut arena = Arena::new();
            let operand = arena.alloc(Node::Scalar(x));
            let expr = arena.alloc(node(operand));
            let simplified = simplify(&mut arena, expr);
            let label = format!("{:?}", arena.get(expr));
            let &Node::Scalar(folded) = arena.get(simplified) else {
                panic!("{label} must fold to a scalar")
            };
            assert_eq!(
                folded.to_bits(),
                op.apply(x).to_bits(),
                "{label}: {folded:?}"
            );
            assert_eq!(folded.to_bits(), expected.to_bits(), "{label}: {folded:?}");
        }
    }

    #[test]
    fn test_nested_simplification() {
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
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let zero = arena.alloc(Node::Scalar(0.0));
        let one = arena.alloc(Node::Scalar(1.0));

        let x_times_one = arena.alloc(Node::Mul(x, one));
        let lhs = arena.alloc(Node::Add(x_times_one, zero));
        let y_times_zero = arena.alloc(Node::Mul(y, zero));
        let expr = arena.alloc(Node::Sub(lhs, y_times_zero));

        assert_eq!(simplify(&mut arena, expr), x);
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
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let zero = arena.alloc(Node::Scalar(0.0));
        let x_plus_zero = arena.alloc(Node::Add(x, zero));
        let expr = arena.alloc(Node::Add(x_plus_zero, x_plus_zero));

        let simplified = simplify(&mut arena, expr);

        match arena.get(simplified) {
            Node::Add(l, r) => {
                assert_eq!(*l, x);
                assert_eq!(*r, x);
            },
            _ => panic!("Expected Add node"),
        }
    }

    #[test]
    fn an_already_simple_graph_is_reused_not_copied() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let two = arena.alloc(Node::Scalar(2.0));
        let sin = arena.alloc(Node::Sin(y));
        let scaled = arena.alloc(Node::Mul(two, sin));
        let root = arena.alloc(Node::Sub(scaled, y));
        let len = arena.len();

        assert_eq!(simplify(&mut arena, root), root);
        assert_eq!(arena.len(), len, "nothing to rewrite, so nothing allocated");
    }

    #[test]
    fn a_shared_binary_fold_is_memoised() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let three = arena.alloc(Node::Scalar(3.0));
        let four = arena.alloc(Node::Scalar(4.0));
        let hypot = arena.alloc(Node::Hypot(three, four));
        let left = arena.alloc(Node::Mul(y, hypot));
        let right = arena.alloc(Node::Mul(hypot, y));
        let root = arena.alloc(Node::Concat(vec![left, right]));

        let simplified = simplify(&mut arena, root);

        let Node::Concat(children) = arena.get(simplified) else {
            panic!("the concat must survive")
        };
        let (&Node::Mul(_, folded_left), &Node::Mul(folded_right, _)) =
            (arena.get(children[0]), arena.get(children[1]))
        else {
            panic!("both products must survive")
        };
        assert_eq!(arena.get(folded_left), &Node::Scalar(5.0));
        assert_eq!(folded_left, folded_right, "one fold, shared by both users");
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
    fn cse_distinguishes_state_vectors_with_different_ranges() {
        let mut arena = Arena::new();
        let y1 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y2 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let root = arena.alloc(Node::Add(y1, y2));

        let shared = cse(DagRef::new(&arena, root));
        assert_eq!(
            shared.arena().len(),
            3,
            "StateVectors with different ranges must remain distinct"
        );
    }

    /// Regression: `FxHash` cancels paired sign flips, and `==` equates `0.0` with `-0.0`.
    #[test]
    fn cse_keeps_arrays_that_differ_only_in_signed_zeros() {
        use crate::eval::CompiledExpr;
        use crate::node::{ArrayData, Shape};

        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let one = arena.alloc(Node::Scalar(1.0));
        let mut reciprocal = |data: Vec<f64>| {
            let array = ArrayData::try_new(data, Shape::vector(3)).expect("a valid array");
            let coeff = arena.alloc(Node::Array(Box::new(array)));
            let scaled = arena.alloc(Node::Mul(coeff, y));
            arena.alloc(Node::Div(one, scaled))
        };
        let a = reciprocal(vec![0.0, -0.0, 3.0]);
        let b = reciprocal(vec![-0.0, 0.0, 3.0]);
        let root = arena.alloc(Node::Concat(vec![a, b]));

        let shared = cse(DagRef::new(&arena, root));
        assert_eq!(
            shared.arena().len(),
            arena.len(),
            "no two nodes are identical"
        );

        let eval = |dag: DagRef<'_>| {
            let compiled = CompiledExpr::new(dag);
            let mut scratch = vec![0.0; compiled.scratch_len()];
            let values = compiled.eval(&mut scratch, 0.0, &[1.0; 3], &[], &[]);
            values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        };
        assert_eq!(eval(shared.as_ref()), eval(DagRef::new(&arena, root)));
    }

    #[test]
    fn dce_keeps_exactly_the_reachable_nodes_and_their_sharing() {
        let mut arena = Arena::new();
        let _unreachable = arena.alloc(Node::Scalar(999.0));
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let _unreachable_too = arena.alloc(Node::Scalar(888.0));
        let one = arena.alloc(Node::Scalar(1.0));
        let sum = arena.alloc(Node::Add(x, one));
        let diamond = arena.alloc(Node::Mul(sum, sum));
        let negated = arena.alloc(Node::Neg(diamond));
        let root = arena.alloc(Node::Abs(negated));

        let pruned = dce(DagRef::new(&arena, root));

        let out = pruned.arena();
        assert_eq!(out.len(), arena.len() - 2, "only the unreachable nodes go");
        let &Node::Abs(abs_child) = out.get(pruned.root()) else {
            panic!("expected Abs")
        };
        let &Node::Neg(neg_child) = out.get(abs_child) else {
            panic!("expected Neg")
        };
        let &Node::Mul(l, r) = out.get(neg_child) else {
            panic!("expected Mul")
        };
        assert_eq!(l, r, "both operands reference the same Add node");
        let &Node::Add(state, constant) = out.get(l) else {
            panic!("expected Add")
        };
        assert_eq!(out.get(state), &Node::StateVector { start: 0, end: 1 });
        assert_eq!(out.get(constant), &Node::Scalar(1.0));
    }

    #[test]
    fn a_reduce_arg_select_over_a_zero_basis_folds_to_zero() {
        for (zero_basis, width) in [(Node::ZeroVector { len: 3 }, 3), (Node::Scalar(0.0), 1)] {
            let mut arena = Arena::new();
            let picker = arena.alloc(Node::StateVector {
                start: 0,
                end: width,
            });
            let basis = arena.alloc(zero_basis);
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
}
