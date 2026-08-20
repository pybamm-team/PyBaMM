//! Shape-aware zero propagation for dead branch elimination.
//!
//! Uses a three-valued lattice to prevent unsafe folds:
//! - `0 / Unknown` must NOT fold to 0 (divisor could be 0 → NaN)
//! - `log(AllZero)` must NOT fold to 0 (log(0) = -∞)

use crate::arena::NodeMap;
use crate::arena::{Arena, NodeId};
use crate::node::{ArrayData, Node};

/// Zero-status lattice for safe constant folding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZeroStatus {
    /// Provably all zeros (from `Scalar(0.0)`, `ZeroVector`, or `Array{[0,0,...]}`)
    AllZero,
    /// Provably no zeros (from non-zero constants like `Scalar(5.0)`)
    DefinitelyNonZero,
    /// Could be anything at runtime (state vectors, parameters, etc.)
    Unknown,
}

/// Shape and zero-status for a node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapeInfo {
    /// Length: 1 for scalar, n for vector
    pub len: usize,
    /// Zero status from the lattice
    pub zero_status: ZeroStatus,
}

impl ShapeInfo {
    /// A node of `len` elements with an explicit lattice status.
    pub const fn new(len: usize, zero_status: ZeroStatus) -> Self {
        Self { len, zero_status }
    }

    /// Every element is provably zero, which is what licenses a fold.
    pub const fn all_zero(len: usize) -> Self {
        Self {
            len,
            zero_status: ZeroStatus::AllZero,
        }
    }

    /// No element can be zero, which is what makes a divisor safe to fold under.
    pub const fn definitely_nonzero(len: usize) -> Self {
        Self {
            len,
            zero_status: ZeroStatus::DefinitelyNonZero,
        }
    }

    /// Nothing is known, the conservative default that blocks folding.
    pub const fn unknown(len: usize) -> Self {
        Self {
            len,
            zero_status: ZeroStatus::Unknown,
        }
    }

    /// Whether the value is provably all zeros.
    pub const fn is_all_zero(&self) -> bool {
        matches!(self.zero_status, ZeroStatus::AllZero)
    }

    /// Whether the value is provably free of zeros.
    pub const fn is_definitely_nonzero(&self) -> bool {
        matches!(self.zero_status, ZeroStatus::DefinitelyNonZero)
    }
}

/// Infer shape for a single node given shapes of its children.
fn infer_shape(arena: &Arena, id: NodeId, shapes: &[ShapeInfo]) -> ShapeInfo {
    match arena.get(id) {
        // Scalar literals
        Node::Scalar(v) => {
            if *v == 0.0 {
                ShapeInfo::all_zero(1)
            } else {
                ShapeInfo::definitely_nonzero(1)
            }
        },

        // Zero vector (first-class)
        Node::ZeroVector { len } => ShapeInfo::all_zero(*len),

        // Arrays - check if all zeros
        Node::Array(arr) => {
            let len = arr.shape.rows * arr.shape.cols;
            if arr.data.iter().all(|&v| v == 0.0) {
                ShapeInfo::all_zero(len)
            } else if arr.data.iter().all(|&v| v != 0.0) {
                ShapeInfo::definitely_nonzero(len)
            } else {
                ShapeInfo::unknown(len)
            }
        },

        // State vectors - Unknown (could be anything at runtime)
        Node::StateVector { start, end }
        | Node::StateVectorDot { start, end }
        | Node::TangentStateVector { start, end } => ShapeInfo::unknown(end - start),

        // Runtime values (time, parameters); InputParameter keeps its packed width
        Node::InputParameter { width, .. } => ShapeInfo::unknown(*width),
        Node::TangentParameter { .. } | Node::Time => ShapeInfo::unknown(1),

        // Sparse matrix (matrix, Unknown)
        Node::SparseMatrix(csr) => {
            let len = csr.shape.rows * csr.shape.cols;
            ShapeInfo::unknown(len)
        },

        // Binary operations
        Node::Add(a, b) | Node::Sub(a, b) => {
            let sa = &shapes[a.index()];
            let sb = &shapes[b.index()];
            let len = sa.len.max(sb.len);
            let status = if sa.is_all_zero() && sb.is_all_zero() {
                ZeroStatus::AllZero
            } else {
                ZeroStatus::Unknown
            };
            ShapeInfo::new(len, status)
        },

        Node::Mul(a, b) => {
            let sa = &shapes[a.index()];
            let sb = &shapes[b.index()];
            let len = sa.len.max(sb.len);
            let status = if sa.is_all_zero() || sb.is_all_zero() {
                ZeroStatus::AllZero
            } else {
                ZeroStatus::Unknown
            };
            ShapeInfo::new(len, status)
        },

        Node::Div(a, b) => {
            let sa = &shapes[a.index()];
            let sb = &shapes[b.index()];
            let len = sa.len.max(sb.len);
            // CRITICAL: Only fold 0/x when x is DefinitelyNonZero
            let status = if sa.is_all_zero() && sb.is_definitely_nonzero() {
                ZeroStatus::AllZero
            } else {
                ZeroStatus::Unknown
            };
            ShapeInfo::new(len, status)
        },

        Node::Pow(base, exp) => {
            let sb = &shapes[base.index()];
            let se = &shapes[exp.index()];
            ShapeInfo::unknown(sb.len.max(se.len))
        },

        Node::Minimum(a, b)
        | Node::Maximum(a, b)
        | Node::Hypot(a, b)
        | Node::Modulo(a, b)
        | Node::EqualHeaviside(a, b)
        | Node::NotEqualHeaviside(a, b)
        | Node::Equality(a, b) => {
            let sa = &shapes[a.index()];
            let sb = &shapes[b.index()];
            let len = sa.len.max(sb.len);
            ShapeInfo::unknown(len)
        },

        Node::MatMul(a, b) => {
            let sb = &shapes[b.index()];
            let result_len = match arena.get(*a) {
                Node::SparseMatrix(csr) => csr.shape.rows,
                Node::Array(arr) => arr.shape.rows,
                _ => sb.len,
            };
            let status = if sb.is_all_zero() {
                ZeroStatus::AllZero
            } else {
                ZeroStatus::Unknown
            };
            ShapeInfo::new(result_len, status)
        },

        // Unary operations - preserve ZeroStatus for Neg/Abs
        Node::Neg(c) | Node::Abs(c) => {
            let sc = &shapes[c.index()];
            ShapeInfo::new(sc.len, sc.zero_status)
        },

        // Nonzero by range: exp(x) > 0 and cosh(x) >= 1 for every x.
        Node::Exp(c) | Node::Cosh(c) => {
            let sc = &shapes[c.index()];
            ShapeInfo::definitely_nonzero(sc.len)
        },

        // log(0) = -∞, NOT 0
        Node::Log(c) => {
            let sc = &shapes[c.index()];
            ShapeInfo::unknown(sc.len)
        },

        // f(0) = 0 for these functions
        Node::Sqrt(c)
        | Node::Sin(c)
        | Node::Tanh(c)
        | Node::Sinh(c)
        | Node::Arcsinh(c)
        | Node::Arctan(c)
        | Node::Erf(c)
        | Node::Sign(c)
        | Node::Floor(c)
        | Node::Ceiling(c) => {
            let sc = &shapes[c.index()];
            let status = if sc.is_all_zero() {
                ZeroStatus::AllZero
            } else {
                ZeroStatus::Unknown
            };
            ShapeInfo::new(sc.len, status)
        },

        // cos(0) = 1, so NOT zero
        Node::Cos(c) => {
            let sc = &shapes[c.index()];
            if sc.is_all_zero() {
                ShapeInfo::definitely_nonzero(sc.len)
            } else {
                ShapeInfo::unknown(sc.len)
            }
        },

        Node::MaxReduce(c) | Node::MinReduce(c) => {
            let sc = &shapes[c.index()];
            let status = if sc.is_all_zero() {
                ZeroStatus::AllZero
            } else {
                ZeroStatus::Unknown
            };
            ShapeInfo::new(1, status)
        },

        Node::ReduceArgSelect { basis, .. } => {
            let status = if shapes[basis.index()].is_all_zero() {
                ZeroStatus::AllZero
            } else {
                ZeroStatus::Unknown
            };
            ShapeInfo::new(1, status)
        },

        // Structural nodes
        Node::Index { child, start, end } => {
            let sc = &shapes[child.index()];
            ShapeInfo::new(end - start, sc.zero_status)
        },

        Node::Concat(children) => {
            let total_len: usize = children.iter().map(|c| shapes[c.index()].len).sum();
            let all_zero = children.iter().all(|c| shapes[c.index()].is_all_zero());
            let status = if all_zero {
                ZeroStatus::AllZero
            } else {
                ZeroStatus::Unknown
            };
            ShapeInfo::new(total_len, status)
        },

        // Interpolation - Unknown
        Node::Interpolant1DLinear { child, .. }
        | Node::Interpolant1DLinearDeriv { child, .. }
        | Node::Interpolant1DCubic { child, .. }
        | Node::Interpolant1DCubicDeriv { child, .. } => {
            let sc = &shapes[child.index()];
            ShapeInfo::unknown(sc.len)
        },

        // N-D interpolation - Unknown, element-wise over children
        Node::InterpolantNd { children, .. } | Node::InterpolantNdPartial { children, .. } => {
            let len = children
                .iter()
                .map(|c| shapes[c.index()].len)
                .max()
                .unwrap_or(1);
            ShapeInfo::unknown(len)
        },

        // Conditional - conservatively Unknown; length mirrors `ir::infer_sizes`
        Node::Conditional { branches, .. } => {
            let len = branches
                .iter()
                .map(|b| shapes[b.index()].len)
                .max()
                .unwrap_or(1);
            ShapeInfo::unknown(len)
        },
    }
}

/// Analyze shapes for all nodes reachable from root.
pub fn analyze_shapes(arena: &Arena, root: NodeId) -> Vec<ShapeInfo> {
    let mut shapes = vec![
        ShapeInfo {
            len: 0,
            zero_status: ZeroStatus::Unknown
        };
        arena.len()
    ];
    let order = arena.topological_order(root);

    for id in order {
        let info = infer_shape(arena, id, &shapes);
        shapes[id.index()] = info;
    }

    shapes
}

/// Create a zero node of the appropriate size.
fn make_zero(arena: &mut Arena, len: usize) -> NodeId {
    if len == 1 {
        arena.alloc(Node::Scalar(0.0))
    } else {
        arena.alloc(Node::ZeroVector { len })
    }
}

/// Propagate zero information and eliminate dead branches.
///
/// Value-preserving, not sign-of-zero-preserving, matching the relaxed
/// guarantee documented on
/// [`SimplifyMode::Conservative`](crate::simplify::SimplifyMode::Conservative).
/// The folds `0 + b -> b`, `a + 0 -> a`, `a - 0 -> a` and `0 - b -> -b` are
/// guarded on an operand this pass *proved* all-zero, not on a literal `+0.0`,
/// so they reach computed zeros as well and a folded zero result may carry the
/// opposite sign from an unfolded one.
///
/// `a - 0 -> a` and `0 - b -> -b` therefore go further than `Conservative`
/// `simplify`, which restricts them to the zero-sign that is exact. This pass has
/// no mode switch and always runs in `simplify_pipeline`, so the relaxed
/// guarantee above is what licenses the difference.
pub fn zero_propagate(arena: &Arena, root: NodeId) -> (Arena, NodeId) {
    let shapes = analyze_shapes(arena, root);

    let mut new_arena = Arena::new();
    let mut old_to_new: NodeMap<NodeId> = NodeMap::new(arena.len());

    let order = arena.topological_order(root);

    for old_id in order {
        let shape = &shapes[old_id.index()];

        // If this node is provably AllZero, replace with zero literal
        if shape.is_all_zero() {
            let new_id = make_zero(&mut new_arena, shape.len);
            old_to_new.insert(old_id, new_id);
            continue;
        }

        // Otherwise, rebuild the node with potential simplifications
        let new_id = match arena.get(old_id) {
            // Leaf nodes - copy directly
            Node::Scalar(v) => new_arena.alloc(Node::Scalar(*v)),
            Node::ZeroVector { len } => new_arena.alloc(Node::ZeroVector { len: *len }),
            Node::Array(arr) => new_arena.alloc(Node::Array(Box::new(ArrayData {
                data: arr.data.clone(),
                shape: arr.shape,
            }))),
            Node::SparseMatrix(csr) => new_arena.alloc(Node::SparseMatrix(csr.clone())),
            Node::StateVector { start, end } => new_arena.alloc(Node::StateVector {
                start: *start,
                end: *end,
            }),
            Node::StateVectorDot { start, end } => new_arena.alloc(Node::StateVectorDot {
                start: *start,
                end: *end,
            }),
            Node::TangentStateVector { start, end } => new_arena.alloc(Node::TangentStateVector {
                start: *start,
                end: *end,
            }),
            Node::InputParameter {
                name,
                index,
                offset,
                width,
            } => new_arena.alloc(Node::InputParameter {
                name: name.clone(),
                index: *index,
                offset: *offset,
                width: *width,
            }),
            Node::TangentParameter { index } => {
                new_arena.alloc(Node::TangentParameter { index: *index })
            },
            Node::Time => new_arena.alloc(Node::Time),

            // Binary ops with zero elimination
            Node::Add(a, b) => {
                let na = old_to_new
                    .get(*a)
                    .copied()
                    .expect("child must be processed");
                let nb = old_to_new
                    .get(*b)
                    .copied()
                    .expect("child must be processed");
                let same_shape = shapes[a.index()].len == shapes[b.index()].len;
                if same_shape && shapes[a.index()].is_all_zero() {
                    nb // 0 + b = b
                } else if same_shape && shapes[b.index()].is_all_zero() {
                    na // a + 0 = a
                } else {
                    new_arena.alloc(Node::Add(na, nb))
                }
            },

            Node::Sub(a, b) => {
                let na = old_to_new
                    .get(*a)
                    .copied()
                    .expect("child must be processed");
                let nb = old_to_new
                    .get(*b)
                    .copied()
                    .expect("child must be processed");
                let same_shape = shapes[a.index()].len == shapes[b.index()].len;
                if same_shape && shapes[b.index()].is_all_zero() {
                    na // a - 0 = a
                } else if same_shape && shapes[a.index()].is_all_zero() {
                    new_arena.alloc(Node::Neg(nb)) // 0 - b = -b
                } else {
                    new_arena.alloc(Node::Sub(na, nb))
                }
            },

            Node::Mul(a, b) => {
                let na = old_to_new
                    .get(*a)
                    .copied()
                    .expect("child must be processed");
                let nb = old_to_new
                    .get(*b)
                    .copied()
                    .expect("child must be processed");
                new_arena.alloc(Node::Mul(na, nb))
            },

            Node::Div(a, b) => {
                let na = old_to_new
                    .get(*a)
                    .copied()
                    .expect("child must be processed");
                let nb = old_to_new
                    .get(*b)
                    .copied()
                    .expect("child must be processed");
                new_arena.alloc(Node::Div(na, nb))
            },

            Node::Pow(a, b) => {
                let na = old_to_new
                    .get(*a)
                    .copied()
                    .expect("child must be processed");
                let nb = old_to_new
                    .get(*b)
                    .copied()
                    .expect("child must be processed");
                new_arena.alloc(Node::Pow(na, nb))
            },

            Node::MatMul(a, b) => {
                let na = old_to_new
                    .get(*a)
                    .copied()
                    .expect("child must be processed");
                let nb = old_to_new
                    .get(*b)
                    .copied()
                    .expect("child must be processed");
                new_arena.alloc(Node::MatMul(na, nb))
            },

            Node::Minimum(a, b) => {
                let na = old_to_new
                    .get(*a)
                    .copied()
                    .expect("child must be processed");
                let nb = old_to_new
                    .get(*b)
                    .copied()
                    .expect("child must be processed");
                new_arena.alloc(Node::Minimum(na, nb))
            },

            Node::Maximum(a, b) => {
                let na = old_to_new
                    .get(*a)
                    .copied()
                    .expect("child must be processed");
                let nb = old_to_new
                    .get(*b)
                    .copied()
                    .expect("child must be processed");
                new_arena.alloc(Node::Maximum(na, nb))
            },

            Node::Modulo(a, b) => {
                let na = old_to_new
                    .get(*a)
                    .copied()
                    .expect("child must be processed");
                let nb = old_to_new
                    .get(*b)
                    .copied()
                    .expect("child must be processed");
                new_arena.alloc(Node::Modulo(na, nb))
            },

            Node::Hypot(a, b) => {
                let na = old_to_new
                    .get(*a)
                    .copied()
                    .expect("child must be processed");
                let nb = old_to_new
                    .get(*b)
                    .copied()
                    .expect("child must be processed");
                new_arena.alloc(Node::Hypot(na, nb))
            },

            Node::EqualHeaviside(a, b) => {
                let na = old_to_new
                    .get(*a)
                    .copied()
                    .expect("child must be processed");
                let nb = old_to_new
                    .get(*b)
                    .copied()
                    .expect("child must be processed");
                new_arena.alloc(Node::EqualHeaviside(na, nb))
            },

            Node::NotEqualHeaviside(a, b) => {
                let na = old_to_new
                    .get(*a)
                    .copied()
                    .expect("child must be processed");
                let nb = old_to_new
                    .get(*b)
                    .copied()
                    .expect("child must be processed");
                new_arena.alloc(Node::NotEqualHeaviside(na, nb))
            },

            Node::Equality(a, b) => {
                let na = old_to_new
                    .get(*a)
                    .copied()
                    .expect("child must be processed");
                let nb = old_to_new
                    .get(*b)
                    .copied()
                    .expect("child must be processed");
                new_arena.alloc(Node::Equality(na, nb))
            },

            // Unary ops
            Node::Neg(c) => new_arena.alloc(Node::Neg(
                old_to_new
                    .get(*c)
                    .copied()
                    .expect("child must be processed"),
            )),
            Node::Abs(c) => new_arena.alloc(Node::Abs(
                old_to_new
                    .get(*c)
                    .copied()
                    .expect("child must be processed"),
            )),
            Node::Sqrt(c) => new_arena.alloc(Node::Sqrt(
                old_to_new
                    .get(*c)
                    .copied()
                    .expect("child must be processed"),
            )),
            Node::Exp(c) => new_arena.alloc(Node::Exp(
                old_to_new
                    .get(*c)
                    .copied()
                    .expect("child must be processed"),
            )),
            Node::Log(c) => new_arena.alloc(Node::Log(
                old_to_new
                    .get(*c)
                    .copied()
                    .expect("child must be processed"),
            )),
            Node::Sin(c) => new_arena.alloc(Node::Sin(
                old_to_new
                    .get(*c)
                    .copied()
                    .expect("child must be processed"),
            )),
            Node::Cos(c) => new_arena.alloc(Node::Cos(
                old_to_new
                    .get(*c)
                    .copied()
                    .expect("child must be processed"),
            )),
            Node::Tanh(c) => new_arena.alloc(Node::Tanh(
                old_to_new
                    .get(*c)
                    .copied()
                    .expect("child must be processed"),
            )),
            Node::Sinh(c) => new_arena.alloc(Node::Sinh(
                old_to_new
                    .get(*c)
                    .copied()
                    .expect("child must be processed"),
            )),
            Node::Cosh(c) => new_arena.alloc(Node::Cosh(
                old_to_new
                    .get(*c)
                    .copied()
                    .expect("child must be processed"),
            )),
            Node::Arcsinh(c) => new_arena.alloc(Node::Arcsinh(
                old_to_new
                    .get(*c)
                    .copied()
                    .expect("child must be processed"),
            )),
            Node::Arctan(c) => new_arena.alloc(Node::Arctan(
                old_to_new
                    .get(*c)
                    .copied()
                    .expect("child must be processed"),
            )),
            Node::Erf(c) => new_arena.alloc(Node::Erf(
                old_to_new
                    .get(*c)
                    .copied()
                    .expect("child must be processed"),
            )),
            Node::Sign(c) => new_arena.alloc(Node::Sign(
                old_to_new
                    .get(*c)
                    .copied()
                    .expect("child must be processed"),
            )),
            Node::Floor(c) => new_arena.alloc(Node::Floor(
                old_to_new
                    .get(*c)
                    .copied()
                    .expect("child must be processed"),
            )),
            Node::Ceiling(c) => new_arena.alloc(Node::Ceiling(
                old_to_new
                    .get(*c)
                    .copied()
                    .expect("child must be processed"),
            )),
            Node::MaxReduce(c) => new_arena.alloc(Node::MaxReduce(
                old_to_new
                    .get(*c)
                    .copied()
                    .expect("child must be processed"),
            )),
            Node::MinReduce(c) => new_arena.alloc(Node::MinReduce(
                old_to_new
                    .get(*c)
                    .copied()
                    .expect("child must be processed"),
            )),
            Node::ReduceArgSelect {
                basis,
                picker,
                is_max,
            } => new_arena.alloc(Node::ReduceArgSelect {
                basis: old_to_new
                    .get(*basis)
                    .copied()
                    .expect("child must be processed"),
                picker: old_to_new
                    .get(*picker)
                    .copied()
                    .expect("child must be processed"),
                is_max: *is_max,
            }),

            // Structural nodes
            Node::Index { child, start, end } => new_arena.alloc(Node::Index {
                child: old_to_new
                    .get(*child)
                    .copied()
                    .expect("child must be processed"),
                start: *start,
                end: *end,
            }),

            Node::Concat(children) => {
                let new_children: Vec<NodeId> = children
                    .iter()
                    .map(|c| {
                        old_to_new
                            .get(*c)
                            .copied()
                            .expect("child must be processed")
                    })
                    .collect();
                new_arena.alloc(Node::Concat(new_children))
            },

            // Interpolation nodes
            Node::Interpolant1DLinear { data, child } => {
                new_arena.alloc(Node::Interpolant1DLinear {
                    data: data.clone(),
                    child: old_to_new
                        .get(*child)
                        .copied()
                        .expect("child must be processed"),
                })
            },

            Node::Interpolant1DLinearDeriv {
                slopes,
                x_data,
                child,
            } => new_arena.alloc(Node::Interpolant1DLinearDeriv {
                slopes: slopes.clone(),
                x_data: x_data.clone(),
                child: old_to_new
                    .get(*child)
                    .copied()
                    .expect("child must be processed"),
            }),
            Node::Interpolant1DCubic { data, child } => new_arena.alloc(Node::Interpolant1DCubic {
                data: data.clone(),
                child: old_to_new
                    .get(*child)
                    .copied()
                    .expect("child must be processed"),
            }),
            Node::Interpolant1DCubicDeriv { data, child } => {
                new_arena.alloc(Node::Interpolant1DCubicDeriv {
                    data: data.clone(),
                    child: old_to_new
                        .get(*child)
                        .copied()
                        .expect("child must be processed"),
                })
            },
            Node::InterpolantNd { data, children } => {
                let new_children: Vec<NodeId> = children
                    .iter()
                    .map(|c| {
                        old_to_new
                            .get(*c)
                            .copied()
                            .expect("child must be processed")
                    })
                    .collect();
                new_arena.alloc(Node::InterpolantNd {
                    data: data.clone(),
                    children: new_children,
                })
            },
            Node::InterpolantNdPartial {
                data,
                children,
                axis,
            } => {
                let new_children: Vec<NodeId> = children
                    .iter()
                    .map(|c| {
                        old_to_new
                            .get(*c)
                            .copied()
                            .expect("child must be processed")
                    })
                    .collect();
                new_arena.alloc(Node::InterpolantNdPartial {
                    data: data.clone(),
                    children: new_children,
                    axis: *axis,
                })
            },

            Node::Conditional { selector, branches } => {
                let new_branches: Vec<NodeId> = branches
                    .iter()
                    .map(|b| {
                        old_to_new
                            .get(*b)
                            .copied()
                            .expect("child must be processed")
                    })
                    .collect();
                new_arena.alloc(Node::Conditional {
                    selector: old_to_new
                        .get(*selector)
                        .copied()
                        .expect("child must be processed"),
                    branches: new_branches,
                })
            },
        };

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
    use crate::node::Shape;

    #[test]
    fn test_zero_status_lattice() {
        let zero = ShapeInfo::all_zero(10);
        assert!(zero.is_all_zero());
        assert!(!zero.is_definitely_nonzero());

        let nonzero = ShapeInfo::definitely_nonzero(1);
        assert!(!nonzero.is_all_zero());
        assert!(nonzero.is_definitely_nonzero());

        let unknown = ShapeInfo::unknown(5);
        assert!(!unknown.is_all_zero());
        assert!(!unknown.is_definitely_nonzero());
    }

    #[test]
    fn test_infer_shape_scalar_zero() {
        let mut arena = Arena::new();
        let scalar = arena.alloc(Node::Scalar(0.0));
        let shapes = vec![
            ShapeInfo {
                len: 0,
                zero_status: ZeroStatus::Unknown
            };
            arena.len()
        ];
        let info = infer_shape(&arena, scalar, &shapes);
        assert_eq!(info.len, 1);
        assert!(info.is_all_zero());
    }

    #[test]
    fn test_infer_shape_scalar_nonzero() {
        let mut arena = Arena::new();
        let scalar = arena.alloc(Node::Scalar(5.0));
        let shapes = vec![
            ShapeInfo {
                len: 0,
                zero_status: ZeroStatus::Unknown
            };
            arena.len()
        ];
        let info = infer_shape(&arena, scalar, &shapes);
        assert_eq!(info.len, 1);
        assert!(info.is_definitely_nonzero());
    }

    #[test]
    fn test_infer_shape_state_vector() {
        let mut arena = Arena::new();
        let sv = arena.alloc(Node::StateVector { start: 0, end: 10 });
        let shapes = vec![
            ShapeInfo {
                len: 0,
                zero_status: ZeroStatus::Unknown
            };
            arena.len()
        ];
        let info = infer_shape(&arena, sv, &shapes);
        assert_eq!(info.len, 10);
        assert!(!info.is_all_zero());
        assert!(!info.is_definitely_nonzero());
    }

    #[test]
    fn test_infer_shape_zero_vector() {
        let mut arena = Arena::new();
        let zv = arena.alloc(Node::ZeroVector { len: 100 });
        let shapes = vec![
            ShapeInfo {
                len: 0,
                zero_status: ZeroStatus::Unknown
            };
            arena.len()
        ];
        let info = infer_shape(&arena, zv, &shapes);
        assert_eq!(info.len, 100);
        assert!(info.is_all_zero());
    }

    #[test]
    fn test_infer_shape_array_all_zero() {
        let mut arena = Arena::new();
        let arr = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![0.0, 0.0, 0.0],
            shape: Shape::vector(3),
        })));
        let shapes = vec![
            ShapeInfo {
                len: 0,
                zero_status: ZeroStatus::Unknown
            };
            arena.len()
        ];
        let info = infer_shape(&arena, arr, &shapes);
        assert_eq!(info.len, 3);
        assert!(info.is_all_zero());
    }

    #[test]
    fn test_infer_shape_array_tiny_nonzero_not_zero() {
        let mut arena = Arena::new();
        let arr = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![1.0e-17, -2.0e-17],
            shape: Shape::vector(2),
        })));
        let shapes = vec![
            ShapeInfo {
                len: 0,
                zero_status: ZeroStatus::Unknown
            };
            arena.len()
        ];
        let info = infer_shape(&arena, arr, &shapes);
        assert_eq!(info.len, 2);
        assert!(
            !info.is_all_zero(),
            "tiny non-zero coefficients must not be erased as zeros"
        );
    }

    #[test]
    fn test_infer_shape_mul_zero_times_unknown() {
        let mut arena = Arena::new();
        let zero = arena.alloc(Node::Scalar(0.0));
        let x = arena.alloc(Node::StateVector { start: 0, end: 5 });
        let mul = arena.alloc(Node::Mul(zero, x));

        let mut shapes = vec![
            ShapeInfo {
                len: 0,
                zero_status: ZeroStatus::Unknown
            };
            arena.len()
        ];
        shapes[zero.index()] = ShapeInfo::all_zero(1);
        shapes[x.index()] = ShapeInfo::unknown(5);

        let info = infer_shape(&arena, mul, &shapes);
        assert_eq!(info.len, 5);
        assert!(info.is_all_zero(), "AllZero * Unknown should be AllZero");
    }

    #[test]
    fn test_infer_shape_div_zero_by_nonzero() {
        let mut arena = Arena::new();
        let zero = arena.alloc(Node::Scalar(0.0));
        let five = arena.alloc(Node::Scalar(5.0));
        let div = arena.alloc(Node::Div(zero, five));

        let mut shapes = vec![
            ShapeInfo {
                len: 0,
                zero_status: ZeroStatus::Unknown
            };
            arena.len()
        ];
        shapes[zero.index()] = ShapeInfo::all_zero(1);
        shapes[five.index()] = ShapeInfo::definitely_nonzero(1);

        let info = infer_shape(&arena, div, &shapes);
        assert!(
            info.is_all_zero(),
            "AllZero / DefinitelyNonZero should be AllZero"
        );
    }

    #[test]
    fn test_infer_shape_div_zero_by_unknown() {
        // CRITICAL: 0 / Unknown must NOT be AllZero (divisor could be 0 → NaN)
        let mut arena = Arena::new();
        let zero = arena.alloc(Node::Scalar(0.0));
        let x = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let div = arena.alloc(Node::Div(zero, x));

        let mut shapes = vec![
            ShapeInfo {
                len: 0,
                zero_status: ZeroStatus::Unknown
            };
            arena.len()
        ];
        shapes[zero.index()] = ShapeInfo::all_zero(1);
        shapes[x.index()] = ShapeInfo::unknown(1);

        let info = infer_shape(&arena, div, &shapes);
        assert!(!info.is_all_zero(), "AllZero / Unknown must NOT be AllZero");
    }

    #[test]
    fn test_infer_shape_exp() {
        // exp(x) > 0 for all x, so always DefinitelyNonZero
        let mut arena = Arena::new();
        let zero = arena.alloc(Node::Scalar(0.0));
        let exp = arena.alloc(Node::Exp(zero));

        let mut shapes = vec![
            ShapeInfo {
                len: 0,
                zero_status: ZeroStatus::Unknown
            };
            arena.len()
        ];
        shapes[zero.index()] = ShapeInfo::all_zero(1);

        let info = infer_shape(&arena, exp, &shapes);
        assert!(
            info.is_definitely_nonzero(),
            "exp(0) = 1, which is DefinitelyNonZero"
        );
    }

    #[test]
    fn test_infer_shape_log_zero() {
        // log(0) = -∞, NOT 0, so must be Unknown
        let mut arena = Arena::new();
        let zero = arena.alloc(Node::Scalar(0.0));
        let log = arena.alloc(Node::Log(zero));

        let mut shapes = vec![
            ShapeInfo {
                len: 0,
                zero_status: ZeroStatus::Unknown
            };
            arena.len()
        ];
        shapes[zero.index()] = ShapeInfo::all_zero(1);

        let info = infer_shape(&arena, log, &shapes);
        assert!(
            !info.is_all_zero(),
            "log(AllZero) must NOT be AllZero (log(0) = -∞)"
        );
    }

    #[test]
    fn test_analyze_shapes_simple() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 5 });
        let zero = arena.alloc(Node::Scalar(0.0));
        let mul = arena.alloc(Node::Mul(zero, x));

        let shapes = analyze_shapes(&arena, mul);

        assert!(shapes[zero.index()].is_all_zero());
        assert!(!shapes[x.index()].is_all_zero());
        assert!(
            shapes[mul.index()].is_all_zero(),
            "AllZero * Unknown should be AllZero"
        );
    }

    #[test]
    fn test_analyze_shapes_nested() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 5 });
        let zero = arena.alloc(Node::Scalar(0.0));
        let mul = arena.alloc(Node::Mul(zero, x));
        let y = arena.alloc(Node::StateVector { start: 5, end: 10 });
        let add = arena.alloc(Node::Add(mul, y));

        let shapes = analyze_shapes(&arena, add);

        assert!(shapes[mul.index()].is_all_zero());
        assert!(!shapes[y.index()].is_all_zero());
        assert!(
            !shapes[add.index()].is_all_zero(),
            "AllZero + Unknown is Unknown"
        );
    }

    #[test]
    fn test_zero_propagate_eliminates_mul_zero() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 5 });
        let zero = arena.alloc(Node::Scalar(0.0));
        let mul = arena.alloc(Node::Mul(zero, x));
        let y = arena.alloc(Node::StateVector { start: 5, end: 10 });
        let add = arena.alloc(Node::Add(mul, y));

        let (new_arena, new_root) = zero_propagate(&arena, add);

        // Result should be equivalent to just y (Add(0, y) => y)
        match new_arena.get(new_root) {
            Node::StateVector { start: 5, end: 10 } => {},
            Node::Add(_, _) => {
                panic!("Add should have been simplified to just y");
            },
            other => panic!("Unexpected result: {other:?}"),
        }
    }

    #[test]
    fn test_zero_propagate_preserves_nonzero() {
        let mut arena = Arena::new();
        let x = arena.alloc(Node::StateVector { start: 0, end: 5 });
        let y = arena.alloc(Node::StateVector { start: 5, end: 10 });
        let add = arena.alloc(Node::Add(x, y));

        let (new_arena, new_root) = zero_propagate(&arena, add);

        // Should remain an Add
        match new_arena.get(new_root) {
            Node::Add(_, _) => {},
            _ => panic!("Expected Add to be preserved"),
        }
    }

    #[test]
    fn test_zero_propagate_preserves_tiny_nonzero_derivative_coefficients() {
        use crate::eval::{CompiledExpr, TangentInputs};
        use crate::tangent::tangent_wrt_states;

        let mut arena = Arena::new();
        let coeff = arena.alloc(Node::Array(Box::new(ArrayData {
            data: vec![1.0e-17, -2.0e-17],
            shape: Shape::vector(2),
        })));
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let expr = arena.alloc(Node::Mul(coeff, y));

        let tangent = tangent_wrt_states(&mut arena, expr);
        let (new_arena, new_root) = zero_propagate(&arena, tangent);
        let compiled = CompiledExpr::new(&new_arena, new_root);
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

    #[test]
    fn test_zero_propagate_matmul_zero() {
        use crate::node::CsrData;

        let mut arena = Arena::new();
        let matrix = arena.alloc(Node::SparseMatrix(Box::new(CsrData {
            indptr: vec![0, 2, 4],
            indices: vec![0, 1, 0, 1],
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: Shape::matrix(2, 2),
        })));
        let zero_vec = arena.alloc(Node::ZeroVector { len: 2 });
        let matmul = arena.alloc(Node::MatMul(matrix, zero_vec));

        let (new_arena, new_root) = zero_propagate(&arena, matmul);

        // Result should be a ZeroVector (or Scalar for len=1)
        match new_arena.get(new_root) {
            Node::ZeroVector { len } => {
                assert_eq!(*len, 2);
            },
            Node::Scalar(v) if v.abs() < f64::EPSILON => {
                // Also acceptable for scalar case
            },
            other => panic!("Expected ZeroVector or zero Scalar, got {other:?}"),
        }
    }

    #[test]
    fn test_analyze_shapes_dense_returns_per_node_info() {
        let mut arena = Arena::new();
        let a = arena.alloc(Node::Scalar(0.0));
        let b = arena.alloc(Node::Scalar(3.0));
        let c = arena.alloc(Node::Add(a, b));

        let shapes = analyze_shapes(&arena, c);
        assert!(shapes[a.index()].is_all_zero());
        assert!(shapes[b.index()].is_definitely_nonzero());
    }

    #[test]
    fn test_pow_shape_uses_broadcast_len() {
        // Pow broadcasts like the other binaries (`ir::infer_sizes` uses
        // max(base, exp)); a scalar^vector result has the exponent's length.
        let mut arena = Arena::new();
        let base = arena.alloc(Node::Scalar(2.0));
        let exp = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let pow = arena.alloc(Node::Pow(base, exp));

        let shapes = analyze_shapes(&arena, pow);
        assert_eq!(shapes[pow.index()].len, 3);
    }

    #[test]
    fn test_zero_times_pow_vector_exponent_preserves_shape() {
        // 0 * 2^y with y of length 3 is a zero *vector*; the materialized
        // zero must keep the broadcast length.
        let mut arena = Arena::new();
        let zero = arena.alloc(Node::Scalar(0.0));
        let base = arena.alloc(Node::Scalar(2.0));
        let exp = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let pow = arena.alloc(Node::Pow(base, exp));
        let expr = arena.alloc(Node::Mul(zero, pow));

        let (new_arena, new_root) = zero_propagate(&arena, expr);

        assert_eq!(new_arena.get(new_root), &Node::ZeroVector { len: 3 });
    }

    /// Documents `zero_propagate`'s contract (see its doc comment): value-exact,
    /// but a zero result's sign may be normalised. `-0.0 + 0.0` is `+0.0`;
    /// folding the `+ 0` away yields `-0.0`. Both are zero, so this is permitted.
    #[test]
    #[allow(clippy::float_cmp)] // exact equality is the point: pins value-exactness
    fn zero_propagate_may_normalise_the_sign_of_zero() {
        use crate::eval::CompiledExpr;

        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let neg_y = arena.alloc(Node::Neg(y));
        let zero = arena.alloc(Node::Scalar(0.0));
        let root = arena.alloc(Node::Add(neg_y, zero));

        let before = CompiledExpr::new(&arena, root);
        let (folded_arena, folded_root) = zero_propagate(&arena, root);
        let after = CompiledExpr::new(&folded_arena, folded_root);

        let mut s1 = vec![0.0; before.scratch_len()];
        let mut s2 = vec![0.0; after.scratch_len()];
        let a = before.eval(&mut s1, 0.0, &[0.0], &[], &[])[0];
        let b = after.eval(&mut s2, 0.0, &[0.0], &[], &[])[0];

        assert_eq!(a, b, "values must be equal");
        assert!(a == 0.0 && b == 0.0, "both must be zero");
        // The sign is explicitly NOT guaranteed; assert only that we know
        // which way it went, so a contract change fails loudly here.
        assert!(a.is_sign_positive(), "unfolded -0.0 + 0.0 is +0.0");
        assert!(b.is_sign_negative(), "folded form keeps -0.0");
    }

    /// Sibling of the Add case above, for the Sub-family fold named in
    /// `zero_propagate`'s doc comment: `0 - b -> -b` is sign-exact only from
    /// `-0.0`, but this pass folds it for any provably-all-zero left operand.
    /// `0.0 - 0.0` is `+0.0`; folding to `Neg(y)` yields `-0.0`.
    #[test]
    #[allow(clippy::float_cmp)] // exact equality is the point: pins value-exactness
    fn zero_propagate_sub_fold_may_normalise_the_sign_of_zero() {
        use crate::eval::CompiledExpr;

        let mut arena = Arena::new();
        let zero = arena.alloc(Node::Scalar(0.0));
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let root = arena.alloc(Node::Sub(zero, y));

        let before = CompiledExpr::new(&arena, root);
        let (folded_arena, folded_root) = zero_propagate(&arena, root);
        let after = CompiledExpr::new(&folded_arena, folded_root);

        let mut s1 = vec![0.0; before.scratch_len()];
        let mut s2 = vec![0.0; after.scratch_len()];
        let a = before.eval(&mut s1, 0.0, &[0.0], &[], &[])[0];
        let b = after.eval(&mut s2, 0.0, &[0.0], &[], &[])[0];

        assert_eq!(a, b, "values must be equal");
        assert!(a == 0.0 && b == 0.0, "both must be zero");
        assert!(a.is_sign_positive(), "0.0 - 0.0 is +0.0");
        assert!(b.is_sign_negative(), "the fold to Neg(y) keeps -0.0");
    }
}
