//! Shape-aware zero propagation: folds every subtree it proves identically zero.
//!
//! The rules are `CasADi`'s on-the-fly ones (`0 * x`, `0 / x`, `0 @ v` and `M @ 0` are zero),
//! so the result matches the input up to the sign of zero wherever every intermediate of the
//! input is finite. At a `0 * inf` or `0 / 0` it yields zero where the input yields NaN.

use crate::arena::{Arena, Dag, DagRef, NodeId, NodeMap};
use crate::ir::node_width;
use crate::node::Node;

/// A node's output length, and whether its value is proved identically zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ShapeInfo {
    len: usize,
    all_zero: bool,
}

fn infer_shape(arena: &Arena, id: NodeId, shapes: &[ShapeInfo]) -> ShapeInfo {
    let node = arena.get(id);
    let is_zero = |child: &NodeId| shapes[child.index()].all_zero;
    let all_zero = match node {
        Node::Scalar(v) => *v == 0.0,
        Node::ZeroVector { .. } => true,
        Node::Array(arr) => arr.data.iter().all(|&v| v == 0.0),

        Node::Add(a, b) | Node::Sub(a, b) => is_zero(a) && is_zero(b),
        Node::Mul(a, b) | Node::MatMul(a, b) => is_zero(a) || is_zero(b),
        Node::Div(a, _) => is_zero(a),

        Node::Neg(c)
        | Node::Abs(c)
        | Node::Index { child: c, .. }
        | Node::Sqrt(c)
        | Node::Sin(c)
        | Node::Tanh(c)
        | Node::Sinh(c)
        | Node::Arcsinh(c)
        | Node::Arctan(c)
        | Node::Erf(c)
        | Node::Sign(c)
        | Node::Floor(c)
        | Node::Ceiling(c)
        | Node::MaxReduce(c)
        | Node::MinReduce(c)
        | Node::ReduceArgSelect { basis: c, .. } => is_zero(c),
        // A conditional yields one of its branches or zeros, whatever the selector.
        Node::Concat(children)
        | Node::Conditional {
            branches: children, ..
        } => children.iter().all(is_zero),

        // log(0) = -inf, and exp, cos and cosh of 0 are 1.
        Node::Log(_)
        | Node::Exp(_)
        | Node::Cos(_)
        | Node::Cosh(_)
        | Node::SparseMatrix(_)
        | Node::StateVector { .. }
        | Node::StateVectorDot { .. }
        | Node::TangentStateVector { .. }
        | Node::InputParameter { .. }
        | Node::TangentParameter { .. }
        | Node::Time
        | Node::Pow(..)
        | Node::Minimum(..)
        | Node::Maximum(..)
        | Node::Hypot(..)
        | Node::Modulo(..)
        | Node::EqualHeaviside(..)
        | Node::NotEqualHeaviside(..)
        | Node::Equality(..)
        | Node::Interpolant1DLinear { .. }
        | Node::Interpolant1DLinearDeriv { .. }
        | Node::Interpolant1DCubic { .. }
        | Node::Interpolant1DCubicDeriv { .. }
        | Node::InterpolantNd { .. }
        | Node::InterpolantNdPartial { .. } => false,
    };
    let len = node_width(arena, node, |child| shapes[child.index()].len);
    ShapeInfo { len, all_zero }
}

/// Shape of every node in `order`, indexed by `NodeId`.
fn analyze_shapes(arena: &Arena, order: &[NodeId]) -> Vec<ShapeInfo> {
    let unvisited = ShapeInfo {
        len: 0,
        all_zero: false,
    };
    let mut shapes = vec![unvisited; arena.len()];
    for &id in order {
        shapes[id.index()] = infer_shape(arena, id, &shapes);
    }
    shapes
}

/// Rebuild `dag` with every subtree proved identically zero folded to a zero literal.
///
/// Exact up to the sign of zero wherever every intermediate of `dag` is finite; see the module
/// docs for what happens elsewhere.
pub fn zero_propagate(dag: DagRef<'_>) -> Dag {
    let (arena, root) = (dag.arena(), dag.root());
    let order = arena.topological_order(root);
    let shapes = analyze_shapes(arena, &order);
    // `zero + other` folds to `other` only when no broadcast is lost.
    let absorbed = |zero: NodeId, other: NodeId| {
        shapes[zero.index()].all_zero && shapes[zero.index()].len == shapes[other.index()].len
    };

    let mut new_arena = Arena::new();
    let mut old_to_new: NodeMap<NodeId> = NodeMap::new(arena.len());

    for old_id in order {
        let shape = shapes[old_id.index()];
        let node = arena.get(old_id);
        let new_id = match *node {
            _ if shape.all_zero => new_arena.alloc(Node::zero(shape.len)),
            Node::Add(a, b) if absorbed(a, b) => old_to_new.rebuilt(b),
            Node::Add(a, b) | Node::Sub(a, b) if absorbed(b, a) => old_to_new.rebuilt(a),
            Node::Sub(a, b) if absorbed(a, b) => new_arena.alloc(Node::Neg(old_to_new.rebuilt(b))),
            _ => new_arena.alloc(node.map_children(|c| old_to_new.rebuilt(c))),
        };
        old_to_new.insert(old_id, new_id);
    }

    Dag::new(new_arena, old_to_new.rebuilt(root))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::CompiledExpr;
    use crate::node::{ArrayData, CsrData, Shape};

    fn array(data: Vec<f64>, shape: Shape) -> Node {
        Node::Array(Box::new(
            ArrayData::try_new(data, shape).expect("a valid array"),
        ))
    }

    /// `root`'s shape as the pass infers it.
    fn shape_of(arena: &Arena, root: NodeId) -> ShapeInfo {
        analyze_shapes(arena, &arena.topological_order(root))[root.index()]
    }

    fn eval(dag: DagRef<'_>, y: &[f64]) -> Vec<f64> {
        let compiled = CompiledExpr::new(dag);
        let mut scratch = vec![0.0; compiled.scratch_len()];
        compiled.eval(&mut scratch, 0.0, y, &[], &[]).to_vec()
    }

    #[test]
    fn a_literal_is_zero_only_when_every_entry_is() {
        let cases = [
            (Node::Scalar(0.0), 1, true),
            (Node::Scalar(-0.0), 1, true),
            (Node::Scalar(5.0), 1, false),
            (Node::ZeroVector { len: 100 }, 100, true),
            (Node::StateVector { start: 0, end: 10 }, 10, false),
            (array(vec![0.0, -0.0, 0.0], Shape::vector(3)), 3, true),
            // Tiny nonzero coefficients must not be erased as zeros.
            (array(vec![1.0e-17, -2.0e-17], Shape::vector(2)), 2, false),
        ];
        for (node, len, all_zero) in cases {
            let mut arena = Arena::new();
            let id = arena.alloc(node);
            let expected = ShapeInfo { len, all_zero };
            assert_eq!(shape_of(&arena, id), expected, "{:?}", arena.get(id));
        }
    }

    #[test]
    fn operators_fold_by_casadis_on_the_fly_rules() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let selector = arena.alloc(Node::StateVector { start: 2, end: 3 });
        let zero = arena.alloc(Node::ZeroVector { len: 2 });
        let matrix = arena.alloc(array(vec![1.0, 2.0, 3.0, 4.0], Shape::matrix(2, 2)));
        let zero_matrix = arena.alloc(array(vec![0.0; 4], Shape::matrix(2, 2)));
        let conditional = |branches| Node::Conditional { selector, branches };
        let cases = [
            (Node::Mul(zero, x), true),
            (Node::Mul(x, zero), true),
            (Node::Add(zero, x), false),
            (Node::Add(zero, zero), true),
            (Node::Sub(zero, zero), true),
            (Node::Div(zero, x), true),
            (Node::Div(x, zero), false),
            (Node::MatMul(matrix, zero), true),
            (Node::MatMul(zero_matrix, x), true),
            (Node::MatMul(matrix, x), false),
            (Node::Sin(zero), true),
            (Node::Exp(zero), false),
            (Node::Cos(zero), false),
            (Node::Log(zero), false),
            (Node::Concat(vec![zero, zero]), true),
            (Node::Concat(vec![zero, x]), false),
            (conditional(vec![zero, zero]), true),
            (conditional(vec![zero, x]), false),
        ];
        for (node, all_zero) in cases {
            let id = arena.alloc(node);
            assert_eq!(
                shape_of(&arena, id).all_zero,
                all_zero,
                "{:?}",
                arena.get(id)
            );
        }
    }

    #[test]
    fn the_rebuild_drops_proven_zeros_and_keeps_the_rest() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 5 });
        let y = arena.alloc(Node::StateVector { start: 5, end: 10 });
        let zero = arena.alloc(Node::Scalar(0.0));
        let zero_times_x = arena.alloc(Node::Mul(zero, x));
        let csr = CsrData::try_new(
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, 2.0, 3.0, 4.0],
            Shape::matrix(2, 2),
        )
        .expect("a valid matrix");
        let matrix = arena.alloc(Node::SparseMatrix(Box::new(csr)));
        let zero_vector = arena.alloc(Node::ZeroVector { len: 2 });
        let rebuilt = |arena: &mut Arena, node| {
            let root = arena.alloc(node);
            let folded = zero_propagate(DagRef::new(arena, root));
            folded.arena().get(folded.root()).clone()
        };

        let y_alone = rebuilt(&mut arena, Node::Add(zero_times_x, y));
        assert_eq!(y_alone, Node::StateVector { start: 5, end: 10 });
        let product = rebuilt(&mut arena, Node::MatMul(matrix, zero_vector));
        assert_eq!(product, Node::ZeroVector { len: 2 });
        let kept = rebuilt(&mut arena, Node::Add(x, y));
        assert!(
            matches!(kept, Node::Add(..)),
            "a nonzero sum is kept, got {kept:?}"
        );
    }

    #[test]
    fn test_zero_propagate_preserves_tiny_nonzero_derivative_coefficients() {
        use crate::eval::TangentInputs;
        use crate::tangent::tangent_wrt_states;

        let mut arena = Arena::new();
        let coeff = arena.alloc(array(vec![1.0e-17, -2.0e-17], Shape::vector(2)));
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let expr = arena.alloc(Node::Mul(coeff, y));

        let tangent = tangent_wrt_states(&mut arena, expr);
        let folded = zero_propagate(DagRef::new(&arena, tangent));
        let compiled = CompiledExpr::new(folded.as_ref());
        let mut s_compiled = vec![0.0; compiled.scratch_len()];
        let tangent_inputs = TangentInputs {
            dy: Some(&[1.0, 1.0]),
            dp: None,
        };

        let result = compiled.eval_with_tangent(
            &mut s_compiled,
            0.0,
            &[3.0, 4.0],
            &[],
            &[],
            &tangent_inputs,
        );
        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0e-17).abs() < f64::EPSILON);
        assert!((result[1] - (-2.0e-17)).abs() < f64::EPSILON);
    }

    /// Its own length is never read, so a huge declared shape must not overflow.
    #[test]
    fn a_sparse_matrix_literal_measures_zero_as_lowering_does() {
        let mut arena = Arena::new();
        let matrix = arena.alloc(Node::SparseMatrix(Box::new(
            CsrData::try_new(vec![0, 0, 0], vec![], vec![], Shape::matrix(2, usize::MAX))
                .expect("an empty CSR matrix is valid"),
        )));
        assert_eq!(shape_of(&arena, matrix).len, 0);
    }

    /// Regression: an all-zero dense matrix used to fold to a `ZeroVector` and stay a
    /// `MatMul` operand, which lowering rejects; now the product folds with it.
    #[test]
    fn a_zero_matrix_literal_folds_its_product() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let sin_y = arena.alloc(Node::Sin(y));
        let matrix = arena.alloc(array(vec![0.0; 4], Shape::matrix(2, 2)));
        let product = arena.alloc(Node::MatMul(matrix, sin_y));
        let root = arena.alloc(Node::Add(product, y));

        let folded = zero_propagate(DagRef::new(&arena, root));
        assert_eq!(folded.arena().get(folded.root()), arena.get(y));
        assert_eq!(eval(folded.as_ref(), &[0.5, -1.5]), [0.5, -1.5]);
    }

    /// The motivating case: a parameter tangent of an input-free quotient is `0 / x`.
    #[test]
    fn a_parameter_tangent_of_an_input_free_quotient_folds_away() {
        use crate::simplify::simplify_pipeline;
        use crate::tangent::tangent_wrt_params;

        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let log = arena.alloc(Node::Log(y));
        let root = arena.alloc(Node::Div(log, y));

        let mut tangent = Dag::new(arena, root);
        tangent.rewrite_in_place(tangent_wrt_params);
        let folded = simplify_pipeline(tangent);
        assert_eq!(folded.arena().nodes(), [Node::ZeroVector { len: 3 }]);
    }

    #[test]
    fn test_pow_shape_uses_node_width() {
        let mut arena = Arena::new();
        let base = arena.alloc(Node::Scalar(2.0));
        let exp = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let pow = arena.alloc(Node::Pow(base, exp));

        assert_eq!(shape_of(&arena, pow).len, 3);
    }

    #[test]
    fn test_zero_times_pow_vector_exponent_preserves_shape() {
        let mut arena = Arena::new();
        let zero = arena.alloc(Node::Scalar(0.0));
        let base = arena.alloc(Node::Scalar(2.0));
        let exp = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let pow = arena.alloc(Node::Pow(base, exp));
        let expr = arena.alloc(Node::Mul(zero, pow));

        let folded = zero_propagate(DagRef::new(&arena, expr));

        assert_eq!(
            folded.arena().get(folded.root()),
            &Node::ZeroVector { len: 3 }
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn zero_propagate_may_normalise_the_sign_of_zero() {
        // `-y + 0` and `0 - y` fold to `-y`; at y = 0 the unfolded sum and difference are +0.0.
        let builds: [fn(&mut Arena, NodeId, NodeId) -> Node; 2] = [
            |arena, y, zero| Node::Add(arena.alloc(Node::Neg(y)), zero),
            |_, y, zero| Node::Sub(zero, y),
        ];
        for build in builds {
            let mut arena = Arena::new();
            let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
            let zero = arena.alloc(Node::Scalar(0.0));
            let node = build(&mut arena, y, zero);
            let root = arena.alloc(node);

            let a = eval(DagRef::new(&arena, root), &[0.0])[0];
            let b = eval(zero_propagate(DagRef::new(&arena, root)).as_ref(), &[0.0])[0];

            let label = format!("{:?}", arena.get(root));
            assert_eq!(a, b, "{label}: values must be equal");
            assert!(a == 0.0 && b == 0.0, "{label}: both must be zero");
            // The sign is not guaranteed; pin it so a contract change fails here.
            assert!(a.is_sign_positive(), "{label}: unfolded is +0.0");
            assert!(b.is_sign_negative(), "{label}: the folded form keeps -0.0");
        }
    }
}
