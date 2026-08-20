//! Random-DAG property test: assert `detect_sparsity_per_output` produces
//! output bit-identical to a vendored copy of the pre-refactor recursive
//! implementation.
//!
//! The oracle code below is a verbatim copy of `sparsity.rs` at the
//! commit that starts this refactor. It is frozen — do NOT update it
//! when `sparsity.rs` changes. That is the whole point of the oracle.

use pybamm_core::arena::{Arena, NodeId};
use pybamm_core::node::Node;
use pybamm_core::{SparsityPattern, detect_sparsity_per_output};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::HashSet;

// Frozen copy of `sparsity.rs::analyze_output_dependencies` and its helpers,
// renamed with an `Oracle` suffix, kept as an independent implementation.

#[derive(Clone, Debug)]
enum ElementDepsOracle {
    Scalar(HashSet<usize>),
    Vector(Vec<HashSet<usize>>),
}

impl ElementDepsOracle {
    fn scalar_empty() -> Self {
        Self::Scalar(HashSet::new())
    }
    const fn len(&self) -> usize {
        match self {
            Self::Scalar(_) => 1,
            Self::Vector(v) => v.len(),
        }
    }
    fn get(&self, idx: usize) -> &HashSet<usize> {
        match self {
            Self::Scalar(deps) => deps,
            Self::Vector(v) => &v[idx],
        }
    }
    fn union_all(&self) -> HashSet<usize> {
        match self {
            Self::Scalar(deps) => deps.clone(),
            Self::Vector(v) => {
                let mut result = HashSet::new();
                for deps in v {
                    result.extend(deps);
                }
                result
            },
        }
    }
    fn to_vector(&self, len: usize) -> Vec<HashSet<usize>> {
        match self {
            Self::Scalar(deps) => vec![deps.clone(); len],
            Self::Vector(v) => {
                if v.len() == len {
                    v.clone()
                } else if v.len() == 1 {
                    vec![v[0].clone(); len]
                } else {
                    let union = self.union_all();
                    vec![union; len]
                }
            },
        }
    }
}

fn analyze_oracle(arena: &Arena, node_id: NodeId) -> ElementDepsOracle {
    match arena.get(node_id) {
        // StateVector: each element depends only on its corresponding state index
        Node::StateVector { start, end } => {
            let deps: Vec<HashSet<usize>> = (*start..*end)
                .map(|i| {
                    let mut set = HashSet::new();
                    set.insert(i);
                    set
                })
                .collect();
            if deps.len() == 1 {
                ElementDepsOracle::Scalar(deps.into_iter().next().unwrap())
            } else {
                ElementDepsOracle::Vector(deps)
            }
        },

        // No state dependencies for these types
        Node::StateVectorDot { start, end } | Node::TangentStateVector { start, end } => {
            let len = end - start;
            if len == 1 {
                ElementDepsOracle::scalar_empty()
            } else {
                ElementDepsOracle::Vector(vec![HashSet::new(); len])
            }
        },
        Node::TangentParameter { .. }
        | Node::Scalar(_)
        | Node::Time
        | Node::InputParameter { .. } => ElementDepsOracle::scalar_empty(),
        Node::ZeroVector { len } => {
            if *len == 1 {
                ElementDepsOracle::scalar_empty()
            } else {
                ElementDepsOracle::Vector(vec![HashSet::new(); *len])
            }
        },
        Node::Array(arr) => {
            if arr.data().len() == 1 {
                ElementDepsOracle::scalar_empty()
            } else {
                ElementDepsOracle::Vector(vec![HashSet::new(); arr.data().len()])
            }
        },
        Node::SparseMatrix(csr) => {
            // Sparse matrix is constant, no state dependencies
            // Output is rows x cols, but typically used as matrix not vector
            let total = csr.shape().rows * csr.shape().cols;
            if total == 1 {
                ElementDepsOracle::scalar_empty()
            } else {
                ElementDepsOracle::Vector(vec![HashSet::new(); csr.shape().rows])
            }
        },

        // Unary operations: preserve per-element structure
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
        | Node::Ceiling(a) => analyze_oracle(arena, *a),

        // Reductions: output is scalar with union of all input dependencies
        Node::MaxReduce(a) | Node::MinReduce(a) => {
            let child_deps = analyze_oracle(arena, *a);
            ElementDepsOracle::Scalar(child_deps.union_all())
        },

        // Internal-only node, never produced by `random_dag` below; no
        // pre-refactor oracle behavior exists to freeze for it.
        Node::ReduceArgSelect { .. } => unreachable!("not generated by random_dag"),

        // Binary operations: combine with broadcast semantics
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
            let deps_a = analyze_oracle(arena, *a);
            let deps_b = analyze_oracle(arena, *b);
            combine_binary_oracle(&deps_a, &deps_b)
        },

        // MatMul: row i depends on columns with non-zeros
        // For dense case: row i of output depends on all elements of vector
        Node::MatMul(mat_id, vec_id) => {
            let vec_deps = analyze_oracle(arena, *vec_id);

            // Check if matrix is sparse
            match arena.get(*mat_id) {
                Node::SparseMatrix(csr) => {
                    // For sparse matrix: row i depends on columns where mat[i,:] is non-zero
                    let n_rows = csr.shape().rows;
                    let mut result = Vec::with_capacity(n_rows);

                    for row in 0..n_rows {
                        let row_start = csr.indptr()[row];
                        let row_end = csr.indptr()[row + 1];
                        let mut row_deps = HashSet::new();

                        // Row i depends on vec[col] for each non-zero mat[i, col]
                        for &col in &csr.indices()[row_start..row_end] {
                            row_deps.extend(vec_deps.get(col));
                        }
                        result.push(row_deps);
                    }

                    if result.len() == 1 {
                        ElementDepsOracle::Scalar(result.into_iter().next().unwrap())
                    } else {
                        ElementDepsOracle::Vector(result)
                    }
                },
                Node::Array(arr) => {
                    // Dense matrix: each output row depends on all vector elements
                    let union = vec_deps.union_all();
                    let n_rows = arr.shape().rows;
                    if n_rows == 1 {
                        ElementDepsOracle::Scalar(union)
                    } else {
                        ElementDepsOracle::Vector(vec![union; n_rows])
                    }
                },
                _ => {
                    // For computed matrices, assume dense (conservative)
                    let mat_deps = analyze_oracle(arena, *mat_id);
                    let union_mat = mat_deps.union_all();
                    let union_vec = vec_deps.union_all();
                    let mut combined = union_mat;
                    combined.extend(union_vec);
                    ElementDepsOracle::Scalar(combined)
                },
            }
        },

        // Index: select subset of child dependencies
        Node::Index { child, start, end } => {
            let child_deps = analyze_oracle(arena, *child);
            let len = end - start;

            match &child_deps {
                ElementDepsOracle::Scalar(deps) => {
                    // Indexing a scalar (should be 0..1)
                    if len == 1 {
                        ElementDepsOracle::Scalar(deps.clone())
                    } else {
                        ElementDepsOracle::Vector(vec![deps.clone(); len])
                    }
                },
                ElementDepsOracle::Vector(v) => {
                    let subset: Vec<_> = v[*start..*end].to_vec();
                    if subset.len() == 1 {
                        ElementDepsOracle::Scalar(subset.into_iter().next().unwrap())
                    } else {
                        ElementDepsOracle::Vector(subset)
                    }
                },
            }
        },

        // Concat: concatenate child dependencies
        Node::Concat(children) => {
            let mut result = Vec::new();
            for child in children {
                let child_deps = analyze_oracle(arena, *child);
                match child_deps {
                    ElementDepsOracle::Scalar(deps) => result.push(deps),
                    ElementDepsOracle::Vector(v) => result.extend(v),
                }
            }
            if result.len() == 1 {
                ElementDepsOracle::Scalar(result.into_iter().next().unwrap())
            } else {
                ElementDepsOracle::Vector(result)
            }
        },

        // Interpolant: output depends on child (the x value being interpolated)
        Node::Interpolant1DLinear { child, .. }
        | Node::Interpolant1DLinearDeriv { child, .. }
        | Node::Interpolant1DCubic { child, .. }
        | Node::Interpolant1DCubicDeriv { child, .. } => analyze_oracle(arena, *child),

        // N-D interpolant: element-wise, union of all children's deps
        Node::InterpolantNd { children, .. } | Node::InterpolantNdPartial { children, .. } => {
            let mut acc = analyze_oracle(arena, children[0]);
            for &c in &children[1..] {
                let next = analyze_oracle(arena, c);
                acc = combine_binary_oracle(&acc, &next);
            }
            acc
        },

        // Conditional: union of selector and all branches
        Node::Conditional { selector, branches } => {
            let selector_deps = analyze_oracle(arena, *selector);

            // Get dependencies from all branches
            let branch_deps: Vec<_> = branches.iter().map(|b| analyze_oracle(arena, *b)).collect();

            // Determine output length from branches
            let output_len = branch_deps
                .iter()
                .map(ElementDepsOracle::len)
                .max()
                .unwrap_or(1);

            // Combine: each output element depends on selector + corresponding branch elements
            let selector_union = selector_deps.union_all();
            let mut result = Vec::with_capacity(output_len);

            for i in 0..output_len {
                let mut elem_deps = selector_union.clone();
                for bd in &branch_deps {
                    if let Some(last_idx) = bd.len().checked_sub(1) {
                        elem_deps.extend(bd.get(i.min(last_idx)));
                    }
                }
                result.push(elem_deps);
            }

            if result.len() == 1 {
                ElementDepsOracle::Scalar(result.into_iter().next().unwrap())
            } else {
                ElementDepsOracle::Vector(result)
            }
        },
    }
}

fn combine_binary_oracle(a: &ElementDepsOracle, b: &ElementDepsOracle) -> ElementDepsOracle {
    match (a, b) {
        // Both scalars: union
        (ElementDepsOracle::Scalar(da), ElementDepsOracle::Scalar(db)) => {
            let mut combined = da.clone();
            combined.extend(db);
            ElementDepsOracle::Scalar(combined)
        },
        // Scalar + Vector: broadcast scalar to each element
        (ElementDepsOracle::Scalar(scalar_deps), ElementDepsOracle::Vector(vec_deps)) => {
            let result: Vec<_> = vec_deps
                .iter()
                .map(|vd| {
                    let mut combined = scalar_deps.clone();
                    combined.extend(vd);
                    combined
                })
                .collect();
            ElementDepsOracle::Vector(result)
        },
        // Vector + Scalar: broadcast scalar to each element
        (ElementDepsOracle::Vector(vec_deps), ElementDepsOracle::Scalar(scalar_deps)) => {
            let result: Vec<_> = vec_deps
                .iter()
                .map(|vd| {
                    let mut combined = vd.clone();
                    combined.extend(scalar_deps);
                    combined
                })
                .collect();
            ElementDepsOracle::Vector(result)
        },
        // Vector + Vector: element-wise union
        (ElementDepsOracle::Vector(va), ElementDepsOracle::Vector(vb)) => {
            let len = va.len().max(vb.len());
            let va_expanded = a.to_vector(len);
            let vb_expanded = b.to_vector(len);

            let result: Vec<_> = va_expanded
                .iter()
                .zip(vb_expanded.iter())
                .map(|(da, db)| {
                    let mut combined = da.clone();
                    combined.extend(db);
                    combined
                })
                .collect();
            ElementDepsOracle::Vector(result)
        },
    }
}

fn detect_oracle(
    arena: &Arena,
    root: NodeId,
    n_outputs: usize,
    n_states: usize,
) -> SparsityPattern {
    let output_deps = analyze_oracle(arena, root);
    let mut pattern = SparsityPattern::new(n_outputs, n_states);
    let deps_vec = output_deps.to_vector(n_outputs);
    for (row, deps) in deps_vec.iter().enumerate() {
        pattern.indptr[row] = pattern.indices.len();
        let mut sorted: Vec<usize> = deps.iter().copied().collect();
        sorted.sort_unstable();
        pattern.indices.extend(sorted);
    }
    pattern.indptr[n_outputs] = pattern.indices.len();
    pattern
}

// Random DAG generator

fn random_dag(rng: &mut StdRng, n_states: usize, target_nodes: usize) -> (Arena, NodeId, usize) {
    let mut arena = Arena::new();
    let y = arena.alloc(Node::StateVector {
        start: 0,
        end: n_states,
    });
    let two = arena.alloc(Node::Scalar(2.0));
    let mut pool: Vec<NodeId> = vec![y, two];

    // Add some single-element state vectors
    for i in 0..n_states {
        pool.push(arena.alloc(Node::StateVector {
            start: i,
            end: i + 1,
        }));
    }

    while arena.len() < target_nodes {
        let op = rng.random_range(0..7);
        let id = match op {
            0 => {
                let a = pool[rng.random_range(0..pool.len())];
                arena.alloc(Node::Sin(a))
            },
            1 => {
                let a = pool[rng.random_range(0..pool.len())];
                arena.alloc(Node::Exp(a))
            },
            2 => {
                let a = pool[rng.random_range(0..pool.len())];
                let b = pool[rng.random_range(0..pool.len())];
                arena.alloc(Node::Add(a, b))
            },
            3 => {
                let a = pool[rng.random_range(0..pool.len())];
                let b = pool[rng.random_range(0..pool.len())];
                arena.alloc(Node::Mul(a, b))
            },
            4 => {
                let a = pool[rng.random_range(0..pool.len())];
                arena.alloc(Node::Neg(a))
            },
            5 => {
                // Concat of 2–4 pool elements
                let k = rng.random_range(2..=4);
                let children: Vec<NodeId> = (0..k)
                    .map(|_| pool[rng.random_range(0..pool.len())])
                    .collect();
                arena.alloc(Node::Concat(children))
            },
            _ => {
                let a = pool[rng.random_range(0..pool.len())];
                arena.alloc(Node::Sub(a, a))
            },
        };
        pool.push(id);
    }

    // Wrap with a final Concat to fix the output shape
    let n_out = n_states;
    let final_children: Vec<NodeId> = (0..n_out)
        .map(|_| pool[rng.random_range(0..pool.len())])
        .collect();
    let root = arena.alloc(Node::Concat(final_children));
    (arena, root, n_out)
}

// The property test

#[test]
fn property_sparsity_matches_oracle() {
    let mut rng = StdRng::seed_from_u64(0x00C0_FFEE);
    for trial in 0..200 {
        let n_states = rng.random_range(2..30);
        let target = rng.random_range(20..80);
        let (arena, root, n_outputs) = random_dag(&mut rng, n_states, target);

        let actual = detect_sparsity_per_output(&arena, root, n_outputs, n_states);
        let oracle = detect_oracle(&arena, root, n_outputs, n_states);

        assert_eq!(
            (actual.nrows, actual.ncols, &actual.indptr, &actual.indices),
            (oracle.nrows, oracle.ncols, &oracle.indptr, &oracle.indices),
            "trial {trial}: mismatch on n_states={n_states} target_nodes={target}",
        );
    }
}
