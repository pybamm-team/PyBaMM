//! Per-output sparsity of `d(root)/dy`, read off the DAG without evaluating it.
//!
//! Each node is annotated with the set of state indices its value can depend on,
//! propagated bottom-up as bitsets; the sets reaching each element of the root
//! become one row of a CSR [`SparsityPattern`]. Tracking outputs individually
//! rather than unioning them is what makes coloring pay: a tridiagonal Jacobian
//! keeps its three colors instead of collapsing to a dense union.
//!
//! The result is structural and conservative: an entry may be present and
//! evaluate to zero, but a missing entry is zero for every `y`, which is the
//! direction coloring and assembly depend on.

use crate::arena::{Arena, NodeId};
use crate::node::Node;

/// Packed bitset over n state indices. Backed by `Vec<u64>`.
///
/// `iter()` yields set indices in ascending order, which matches the
/// CSR column-sorted requirement of `SparsityPattern`, callers do not
/// need to sort after collecting from a `BitSet`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct BitSet {
    bits: Vec<u64>,
    n: usize,
}

impl BitSet {
    fn zeros(n: usize) -> Self {
        Self {
            bits: vec![0u64; n.div_ceil(64)],
            n,
        }
    }

    fn insert(&mut self, i: usize) {
        debug_assert!(i < self.n, "BitSet::insert({i}) out of range n={}", self.n);
        self.bits[i / 64] |= 1u64 << (i % 64);
    }

    fn union_with(&mut self, other: &Self) {
        debug_assert_eq!(self.n, other.n, "BitSet::union_with shape mismatch");
        for (a, &b) in self.bits.iter_mut().zip(other.bits.iter()) {
            *a |= b;
        }
    }

    fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.bits
            .iter()
            .enumerate()
            .flat_map(|(word_idx, &word)| BitsIter {
                word,
                base: word_idx * 64,
            })
    }
}

struct BitsIter {
    word: u64,
    base: usize,
}

impl Iterator for BitsIter {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        if self.word == 0 {
            return None;
        }
        let tz = self.word.trailing_zeros() as usize;
        self.word &= self.word - 1; // clear lowest set bit
        Some(self.base + tz)
    }
}

/// CSR structure of a Jacobian: which entries can be non-zero, without values.
///
/// Column indices are ascending within each row, which [`merge_with`] relies on:
/// it is a two-way sorted merge, so unsorted input silently yields a wrong union
/// rather than an error.
///
/// [`merge_with`]: Self::merge_with
#[derive(Debug, Clone)]
pub struct SparsityPattern {
    /// Rows, one per output element.
    pub nrows: usize,
    /// Columns, one per state (or per parameter for a `df/dp` pattern).
    pub ncols: usize,
    /// Row start offsets into `indices`, length `nrows + 1`.
    pub indptr: Vec<usize>,
    /// Column index of every structural entry, row by row.
    pub indices: Vec<usize>,
}

impl SparsityPattern {
    /// An all-zero pattern of the given shape.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            indptr: vec![0; nrows + 1],
            indices: Vec::new(),
        }
    }

    /// Structural non-zeros, which is the value-buffer length assembly needs.
    pub const fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Row of every CSR entry, i.e. the inverse of `indptr`.
    pub fn entry_rows(&self) -> Vec<usize> {
        let mut rows = vec![0usize; self.nnz()];
        for row in 0..self.nrows {
            rows[self.indptr[row]..self.indptr[row + 1]].fill(row);
        }
        rows
    }

    /// Entries of each row, as the widths `detect_dense_rows` measures.
    pub fn row_widths(&self) -> Vec<usize> {
        (0..self.nrows)
            .map(|row| self.indptr[row + 1] - self.indptr[row])
            .collect()
    }

    /// Merge another sparsity pattern into this one (union of nonzero positions).
    ///
    /// Combines df/dy sparsity with mass-matrix sparsity for the full
    /// `J = df/dy - cj*M` pattern KLU needs.
    pub fn merge_with(&mut self, other: &Self) {
        assert_eq!(self.nrows, other.nrows, "Row count mismatch");
        assert_eq!(self.ncols, other.ncols, "Column count mismatch");

        let mut new_indices = Vec::with_capacity(self.indices.len() + other.indices.len());
        let mut new_indptr = vec![0usize; self.nrows + 1];

        for row in 0..self.nrows {
            let self_start = self.indptr[row];
            let self_end = self.indptr[row + 1];
            let other_start = other.indptr[row];
            let other_end = other.indptr[row + 1];

            // Merge sorted column indices
            let mut i = self_start;
            let mut j = other_start;
            while i < self_end && j < other_end {
                match self.indices[i].cmp(&other.indices[j]) {
                    std::cmp::Ordering::Less => {
                        new_indices.push(self.indices[i]);
                        i += 1;
                    },
                    std::cmp::Ordering::Greater => {
                        new_indices.push(other.indices[j]);
                        j += 1;
                    },
                    std::cmp::Ordering::Equal => {
                        new_indices.push(self.indices[i]);
                        i += 1;
                        j += 1;
                    },
                }
            }
            while i < self_end {
                new_indices.push(self.indices[i]);
                i += 1;
            }
            while j < other_end {
                new_indices.push(other.indices[j]);
                j += 1;
            }
            new_indptr[row + 1] = new_indices.len();
        }

        self.indices = new_indices;
        self.indptr = new_indptr;
    }

    /// Fully dense pattern: every (row, col) entry present.
    ///
    /// Used for df/dp jacobians where parameter sparsity is not detected;
    /// coloring a dense pattern yields one color per column (= one JVP
    /// sweep per parameter, matching the per-parameter cost of unit seeds).
    pub fn dense(nrows: usize, ncols: usize) -> Self {
        let indptr = (0..=nrows).map(|r| r * ncols).collect();
        let indices = (0..nrows).flat_map(|_| 0..ncols).collect();
        Self {
            nrows,
            ncols,
            indptr,
            indices,
        }
    }

    /// Create sparsity pattern from CSR data.
    pub fn from_csr_data(csr: &crate::node::CsrData) -> Self {
        Self {
            nrows: csr.shape.rows,
            ncols: csr.shape.cols,
            indptr: csr.indptr.clone(),
            indices: csr.indices.clone(),
        }
    }
}

/// Per-element dependency information
/// Each element in the output may depend on different state variables
#[derive(Debug, Clone)]
enum ElementDeps {
    /// Scalar with specific state dependencies
    Scalar(BitSet),
    /// Vector where each element has its own dependencies
    Vector(Vec<BitSet>),
}

impl ElementDeps {
    /// Create scalar dependencies with no state variables
    fn scalar_empty(n_states: usize) -> Self {
        Self::Scalar(BitSet::zeros(n_states))
    }

    /// Get the number of elements
    const fn len(&self) -> usize {
        match self {
            Self::Scalar(_) => 1,
            Self::Vector(v) => v.len(),
        }
    }

    /// Get dependencies for a specific element, broadcasting scalars
    fn get(&self, idx: usize) -> &BitSet {
        match self {
            Self::Scalar(deps) => deps,
            Self::Vector(v) => &v[idx],
        }
    }

    /// Get union of all dependencies
    fn union_all(&self, n_states: usize) -> BitSet {
        match self {
            // Scalar already carries its own size; n_states only sizes the
            // accumulator in the Vector arm.
            Self::Scalar(deps) => deps.clone(),
            Self::Vector(v) => {
                let mut result = BitSet::zeros(n_states);
                for deps in v {
                    result.union_with(deps);
                }
                result
            },
        }
    }

    /// Convert to vector form with given length (broadcasts scalar)
    fn to_vector(&self, len: usize, n_states: usize) -> Vec<BitSet> {
        match self {
            Self::Scalar(deps) => vec![deps.clone(); len],
            Self::Vector(v) => {
                if v.len() == len {
                    v.clone()
                } else if v.len() == 1 {
                    vec![v[0].clone(); len]
                } else {
                    // Unexpected shape mismatch - return union for safety
                    let union = self.union_all(n_states);
                    vec![union; len]
                }
            },
        }
    }
}

/// Bottom-up memoized analysis. For each reachable node, compute its
/// `ElementDeps` exactly once. Children are guaranteed in `deps` before
/// the parent is processed because `topological_order` visits them
/// first.
fn analyze_output_dependencies(arena: &Arena, root: NodeId, n_states: usize) -> ElementDeps {
    let order = arena.topological_order(root);
    let mut deps: Vec<Option<ElementDeps>> = (0..arena.len()).map(|_| None).collect();
    for &id in &order {
        let computed = compute_node_deps(arena, id, n_states, &deps);
        deps[id.index()] = Some(computed);
    }
    deps[root.index()]
        .take()
        .expect("root must be in topological order")
}

fn compute_node_deps(
    arena: &Arena,
    id: NodeId,
    n_states: usize,
    deps: &[Option<ElementDeps>],
) -> ElementDeps {
    match arena.get(id) {
        Node::StateVector { start, end } => {
            let len = end - start;
            if len == 1 {
                let mut bs = BitSet::zeros(n_states);
                bs.insert(*start);
                ElementDeps::Scalar(bs)
            } else {
                let v: Vec<BitSet> = (*start..*end)
                    .map(|i| {
                        let mut bs = BitSet::zeros(n_states);
                        bs.insert(i);
                        bs
                    })
                    .collect();
                ElementDeps::Vector(v)
            }
        },
        Node::StateVectorDot { start, end } | Node::TangentStateVector { start, end } => {
            let len = end - start;
            if len == 1 {
                ElementDeps::scalar_empty(n_states)
            } else {
                ElementDeps::Vector(vec![BitSet::zeros(n_states); len])
            }
        },
        Node::TangentParameter { .. }
        | Node::Scalar(_)
        | Node::Time
        | Node::InputParameter { .. } => ElementDeps::scalar_empty(n_states),
        Node::ZeroVector { len } => {
            if *len == 1 {
                ElementDeps::scalar_empty(n_states)
            } else {
                ElementDeps::Vector(vec![BitSet::zeros(n_states); *len])
            }
        },
        Node::Array(arr) => {
            if arr.data.len() == 1 {
                ElementDeps::scalar_empty(n_states)
            } else {
                ElementDeps::Vector(vec![BitSet::zeros(n_states); arr.data.len()])
            }
        },
        Node::SparseMatrix(csr) => {
            let total = csr.shape.rows * csr.shape.cols;
            if total == 1 {
                ElementDeps::scalar_empty(n_states)
            } else {
                ElementDeps::Vector(vec![BitSet::zeros(n_states); csr.shape.rows])
            }
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
        | Node::Ceiling(a) => child_deps(deps, *a).clone(),
        Node::MaxReduce(a) | Node::MinReduce(a) => {
            ElementDeps::Scalar(child_deps(deps, *a).union_all(n_states))
        },
        Node::ReduceArgSelect { basis, picker, .. } => {
            // Argmax/argmin is runtime-dependent, so the static pattern must
            // union deps over the whole reduced vector (deps(basis) subset of deps(picker)).
            let mut combined = child_deps(deps, *basis).union_all(n_states);
            combined.union_with(&child_deps(deps, *picker).union_all(n_states));
            ElementDeps::Scalar(combined)
        },
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
            combine_binary_deps(child_deps(deps, *a), child_deps(deps, *b), n_states)
        },
        Node::MatMul(mat_id, vec_id) => {
            let vec_d = child_deps(deps, *vec_id);
            match arena.get(*mat_id) {
                Node::SparseMatrix(csr) => {
                    let n_rows = csr.shape.rows;
                    let mut result = Vec::with_capacity(n_rows);
                    for row in 0..n_rows {
                        let row_start = csr.indptr[row];
                        let row_end = csr.indptr[row + 1];
                        let mut row_deps = BitSet::zeros(n_states);
                        for &col in &csr.indices[row_start..row_end] {
                            row_deps.union_with(vec_d.get(col));
                        }
                        result.push(row_deps);
                    }
                    if result.len() == 1 {
                        ElementDeps::Scalar(result.into_iter().next().unwrap())
                    } else {
                        ElementDeps::Vector(result)
                    }
                },
                Node::Array(arr) => {
                    let u = vec_d.union_all(n_states);
                    let n_rows = arr.shape.rows;
                    if n_rows == 1 {
                        ElementDeps::Scalar(u)
                    } else {
                        ElementDeps::Vector(vec![u; n_rows])
                    }
                },
                _ => {
                    let mat_d = child_deps(deps, *mat_id);
                    let mut combined = mat_d.union_all(n_states);
                    combined.union_with(&vec_d.union_all(n_states));
                    ElementDeps::Scalar(combined)
                },
            }
        },
        Node::Index { child, start, end } => {
            let cd = child_deps(deps, *child);
            let len = end - start;
            match cd {
                ElementDeps::Scalar(b) => {
                    if len == 1 {
                        ElementDeps::Scalar(b.clone())
                    } else {
                        ElementDeps::Vector(vec![b.clone(); len])
                    }
                },
                ElementDeps::Vector(v) => {
                    let subset: Vec<_> = v[*start..*end].to_vec();
                    if subset.len() == 1 {
                        ElementDeps::Scalar(subset.into_iter().next().unwrap())
                    } else {
                        ElementDeps::Vector(subset)
                    }
                },
            }
        },
        Node::Concat(children) => {
            let mut result = Vec::new();
            for c in children {
                match child_deps(deps, *c) {
                    ElementDeps::Scalar(b) => result.push(b.clone()),
                    ElementDeps::Vector(v) => result.extend(v.iter().cloned()),
                }
            }
            if result.len() == 1 {
                ElementDeps::Scalar(result.into_iter().next().unwrap())
            } else {
                ElementDeps::Vector(result)
            }
        },
        Node::Interpolant1DLinear { child, .. }
        | Node::Interpolant1DLinearDeriv { child, .. }
        | Node::Interpolant1DCubic { child, .. }
        | Node::Interpolant1DCubicDeriv { child, .. } => child_deps(deps, *child).clone(),
        Node::InterpolantNd { children, .. } | Node::InterpolantNdPartial { children, .. } => {
            let mut acc = child_deps(deps, children[0]).clone();
            for &c in &children[1..] {
                acc = combine_binary_deps(&acc, child_deps(deps, c), n_states);
            }
            acc
        },
        Node::Conditional { selector, branches } => {
            let sel = child_deps(deps, *selector);
            let bd: Vec<&ElementDeps> = branches.iter().map(|b| child_deps(deps, *b)).collect();
            let output_len = bd.iter().map(|d| d.len()).max().unwrap_or(1);
            let sel_union = sel.union_all(n_states);
            let mut result = Vec::with_capacity(output_len);
            for i in 0..output_len {
                let mut e = sel_union.clone();
                for bdj in &bd {
                    if let Some(last_idx) = bdj.len().checked_sub(1) {
                        e.union_with(bdj.get(i.min(last_idx)));
                    }
                }
                result.push(e);
            }
            if result.len() == 1 {
                ElementDeps::Scalar(result.into_iter().next().unwrap())
            } else {
                ElementDeps::Vector(result)
            }
        },
    }
}

fn child_deps(deps: &[Option<ElementDeps>], id: NodeId) -> &ElementDeps {
    deps[id.index()]
        .as_ref()
        .expect("child must be processed before parent in topological order")
}

/// Combine dependencies for binary operations with broadcast semantics
fn combine_binary_deps(a: &ElementDeps, b: &ElementDeps, n_states: usize) -> ElementDeps {
    match (a, b) {
        // Both scalars: union
        (ElementDeps::Scalar(da), ElementDeps::Scalar(db)) => {
            let mut c = da.clone();
            c.union_with(db);
            ElementDeps::Scalar(c)
        },
        // Scalar + Vector: broadcast scalar to each element
        (ElementDeps::Scalar(scalar_deps), ElementDeps::Vector(vec_deps)) => {
            let result: Vec<_> = vec_deps
                .iter()
                .map(|vd| {
                    let mut c = scalar_deps.clone();
                    c.union_with(vd);
                    c
                })
                .collect();
            ElementDeps::Vector(result)
        },
        // Vector + Scalar: broadcast scalar to each element
        (ElementDeps::Vector(vec_deps), ElementDeps::Scalar(scalar_deps)) => {
            let result: Vec<_> = vec_deps
                .iter()
                .map(|vd| {
                    let mut c = vd.clone();
                    c.union_with(scalar_deps);
                    c
                })
                .collect();
            ElementDeps::Vector(result)
        },
        // Vector + Vector: element-wise union
        (ElementDeps::Vector(va), ElementDeps::Vector(vb)) => {
            let len = va.len().max(vb.len());
            let va_expanded = a.to_vector(len, n_states);
            let vb_expanded = b.to_vector(len, n_states);

            let result: Vec<_> = va_expanded
                .iter()
                .zip(vb_expanded.iter())
                .map(|(da, db)| {
                    let mut c = da.clone();
                    c.union_with(db);
                    c
                })
                .collect();
            ElementDeps::Vector(result)
        },
    }
}

/// Detect sparsity pattern with per-output dependency tracking
///
/// Tracks which state variables each individual output element depends on,
/// rather than a conservative union across all outputs. This enables efficient
/// coloring for banded Jacobians (e.g., tridiagonal patterns get O(3) colors
/// instead of O(n)).
pub fn detect_sparsity_per_output(
    arena: &Arena,
    root: NodeId,
    n_outputs: usize,
    n_states: usize,
) -> SparsityPattern {
    let output_deps = analyze_output_dependencies(arena, root, n_states);

    // Build CSR pattern
    let mut pattern = SparsityPattern::new(n_outputs, n_states);

    // Convert to vector form to iterate over rows
    let deps_vec = output_deps.to_vector(n_outputs, n_states);

    for (row, bs) in deps_vec.iter().enumerate() {
        pattern.indptr[row] = pattern.indices.len();
        pattern.indices.extend(bs.iter());
    }
    pattern.indptr[n_outputs] = pattern.indices.len();

    pattern
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coloring::color_columns;
    use crate::node::{CsrData, Shape};

    /// Collect every state index that any reachable `StateVector` node uses.
    /// One topological walk, `O(arena.len())` + total `StateVector` slot count.
    fn collect_state_deps(arena: &Arena, root: NodeId, n_states: usize) -> BitSet {
        let mut acc = BitSet::zeros(n_states);
        for id in arena.topological_order(root) {
            if let Node::StateVector { start, end } = arena.get(id) {
                for i in *start..*end {
                    acc.insert(i);
                }
            }
        }
        acc
    }

    /// Conservative sparsity oracle: every output row gets the union of all
    /// state dependencies. A superset of `detect_sparsity_per_output`, which it
    /// cross-checks in tests.
    fn detect_sparsity_simple(
        arena: &Arena,
        root: NodeId,
        n_outputs: usize,
        n_states: usize,
    ) -> SparsityPattern {
        let deps = collect_state_deps(arena, root, n_states);
        let mut pattern = SparsityPattern::new(n_outputs, n_states);
        let sorted_deps: Vec<usize> = deps.iter().collect();
        for row in 0..n_outputs {
            pattern.indptr[row] = pattern.indices.len();
            pattern.indices.extend_from_slice(&sorted_deps);
        }
        pattern.indptr[n_outputs] = pattern.indices.len();
        pattern
    }

    #[test]
    fn test_sparsity_pattern_new() {
        let pattern = SparsityPattern::new(3, 4);
        assert_eq!(pattern.nrows, 3);
        assert_eq!(pattern.ncols, 4);
        assert_eq!(pattern.indptr.len(), 4); // nrows + 1
        assert_eq!(pattern.nnz(), 0);
    }

    #[test]
    fn test_detect_sparsity_scalar() {
        let mut arena = Arena::new();
        let sv = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let two = arena.alloc(Node::Scalar(2.0));
        let expr = arena.alloc(Node::Mul(two, sv));

        let pattern = detect_sparsity_simple(&arena, expr, 2, 4);

        // Output depends on states 0 and 1
        assert_eq!(pattern.nrows, 2);
        assert_eq!(pattern.ncols, 4);
        // Each row should have indices [0, 1]
        assert_eq!(
            &pattern.indices[pattern.indptr[0]..pattern.indptr[1]],
            &[0, 1]
        );
    }

    #[test]
    fn test_detect_sparsity_disjoint() {
        let mut arena = Arena::new();
        let sv0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let sv1 = arena.alloc(Node::StateVector { start: 2, end: 3 });
        let expr = arena.alloc(Node::Add(sv0, sv1));

        let pattern = detect_sparsity_simple(&arena, expr, 1, 4);

        // Output depends on states 0 and 2
        let row_indices = &pattern.indices[pattern.indptr[0]..pattern.indptr[1]];
        assert!(row_indices.contains(&0));
        assert!(row_indices.contains(&2));
        assert!(!row_indices.contains(&1));
    }

    #[test]
    fn test_per_output_sparsity_diagonal() {
        // f(y) = [y0, y1, y2] - each output depends only on its corresponding input
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let y2 = arena.alloc(Node::StateVector { start: 2, end: 3 });
        let expr = arena.alloc(Node::Concat(vec![y0, y1, y2]));

        let pattern = detect_sparsity_per_output(&arena, expr, 3, 3);

        // Row 0 should only depend on column 0
        let row0 = &pattern.indices[pattern.indptr[0]..pattern.indptr[1]];
        assert_eq!(row0, &[0], "Row 0 should only depend on state 0");

        // Row 1 should only depend on column 1
        let row1 = &pattern.indices[pattern.indptr[1]..pattern.indptr[2]];
        assert_eq!(row1, &[1], "Row 1 should only depend on state 1");

        // Row 2 should only depend on column 2
        let row2 = &pattern.indices[pattern.indptr[2]..pattern.indptr[3]];
        assert_eq!(row2, &[2], "Row 2 should only depend on state 2");

        // Diagonal matrix should need only 1 color
        let coloring = color_columns(&pattern);
        assert_eq!(coloring.n_colors, 1, "Diagonal should need only 1 color");
    }

    #[test]
    fn test_per_output_sparsity_identity_passthrough() {
        // f(y) = y (identity function on vector)
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 4 });

        let pattern = detect_sparsity_per_output(&arena, y, 4, 4);

        // Each row i should depend only on column i
        for i in 0..4 {
            let row = &pattern.indices[pattern.indptr[i]..pattern.indptr[i + 1]];
            assert_eq!(row, &[i], "Row {i} should only depend on state {i}");
        }

        // Diagonal needs 1 color
        let coloring = color_columns(&pattern);
        assert_eq!(coloring.n_colors, 1);
    }

    #[test]
    fn test_sparsity_tridiagonal() {
        // f(y) = [y0+y1, y0+y1+y2, y1+y2+y3, y2+y3]
        // Should have tridiagonal-like sparsity
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let y2 = arena.alloc(Node::StateVector { start: 2, end: 3 });
        let y3 = arena.alloc(Node::StateVector { start: 3, end: 4 });

        let f0 = arena.alloc(Node::Add(y0, y1));
        let f1_partial = arena.alloc(Node::Add(y0, y1));
        let f1 = arena.alloc(Node::Add(f1_partial, y2));
        let f2_partial = arena.alloc(Node::Add(y1, y2));
        let f2 = arena.alloc(Node::Add(f2_partial, y3));
        let f3 = arena.alloc(Node::Add(y2, y3));

        let expr = arena.alloc(Node::Concat(vec![f0, f1, f2, f3]));

        let pattern = detect_sparsity_per_output(&arena, expr, 4, 4);

        // Check row dependencies
        let row0 = &pattern.indices[pattern.indptr[0]..pattern.indptr[1]];
        assert_eq!(row0, &[0, 1], "Row 0 should depend on states 0, 1");

        let row1 = &pattern.indices[pattern.indptr[1]..pattern.indptr[2]];
        assert_eq!(row1, &[0, 1, 2], "Row 1 should depend on states 0, 1, 2");

        let row2 = &pattern.indices[pattern.indptr[2]..pattern.indptr[3]];
        assert_eq!(row2, &[1, 2, 3], "Row 2 should depend on states 1, 2, 3");

        let row3 = &pattern.indices[pattern.indptr[3]..pattern.indptr[4]];
        assert_eq!(row3, &[2, 3], "Row 3 should depend on states 2, 3");

        // Check that coloring can exploit sparsity
        let coloring = color_columns(&pattern);
        // Tridiagonal needs at most 3 colors
        assert!(
            coloring.n_colors <= 3,
            "Expected <= 3 colors, got {}",
            coloring.n_colors
        );
    }

    #[test]
    fn test_per_output_sparsity_scalar_broadcast() {
        // f(y) = [y0 * c, y1 * c, y2 * c] where c is a scalar
        // Each output should depend only on its corresponding input
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let y2 = arena.alloc(Node::StateVector { start: 2, end: 3 });
        let c = arena.alloc(Node::Scalar(2.0));

        let f0 = arena.alloc(Node::Mul(y0, c));
        let f1 = arena.alloc(Node::Mul(y1, c));
        let f2 = arena.alloc(Node::Mul(y2, c));
        let expr = arena.alloc(Node::Concat(vec![f0, f1, f2]));

        let pattern = detect_sparsity_per_output(&arena, expr, 3, 3);

        // Each row should depend only on its corresponding column
        for i in 0..3 {
            let row = &pattern.indices[pattern.indptr[i]..pattern.indptr[i + 1]];
            assert_eq!(row, &[i], "Row {i} should only depend on state {i}");
        }
    }

    #[test]
    fn test_per_output_sparsity_unary_preserves() {
        // f(y) = [exp(y0), sin(y1), log(y2)]
        // Unary ops should preserve per-element structure
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let y2 = arena.alloc(Node::StateVector { start: 2, end: 3 });

        let f0 = arena.alloc(Node::Exp(y0));
        let f1 = arena.alloc(Node::Sin(y1));
        let f2 = arena.alloc(Node::Log(y2));
        let expr = arena.alloc(Node::Concat(vec![f0, f1, f2]));

        let pattern = detect_sparsity_per_output(&arena, expr, 3, 3);

        // Each row should depend only on its corresponding column
        for i in 0..3 {
            let row = &pattern.indices[pattern.indptr[i]..pattern.indptr[i + 1]];
            assert_eq!(row, &[i], "Row {i} should only depend on state {i}");
        }
    }

    #[test]
    fn test_per_output_sparsity_index() {
        // f(y) = y[1:3] where y has 4 elements
        // Output[0] should depend on state 1, output[1] on state 2
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 4 });
        let expr = arena.alloc(Node::Index {
            child: y,
            start: 1,
            end: 3,
        });

        let pattern = detect_sparsity_per_output(&arena, expr, 2, 4);

        let row0 = &pattern.indices[pattern.indptr[0]..pattern.indptr[1]];
        assert_eq!(row0, &[1], "Row 0 should only depend on state 1");

        let row1 = &pattern.indices[pattern.indptr[1]..pattern.indptr[2]];
        assert_eq!(row1, &[2], "Row 1 should only depend on state 2");
    }

    #[test]
    fn test_per_output_sparsity_reduction() {
        // f(y) = max(y) where y = [y0, y1, y2]
        // Scalar output should depend on all inputs
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let expr = arena.alloc(Node::MaxReduce(y));

        let pattern = detect_sparsity_per_output(&arena, expr, 1, 3);

        let row0 = &pattern.indices[pattern.indptr[0]..pattern.indptr[1]];
        assert_eq!(row0, &[0, 1, 2], "Reduction should depend on all inputs");
    }

    #[test]
    fn test_per_output_sparsity_sparse_matmul() {
        // Sparse matrix M @ y where M is tridiagonal
        // Each output row i depends only on states i-1, i, i+1
        let mut arena = Arena::new();

        // Tridiagonal 3x3: [1 1 0; 1 1 1; 0 1 1]
        let csr = CsrData {
            indptr: vec![0, 2, 5, 7],
            indices: vec![0, 1, 0, 1, 2, 1, 2],
            data: vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            shape: Shape::matrix(3, 3),
        };
        let mat = arena.alloc(Node::SparseMatrix(Box::new(csr)));
        let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let expr = arena.alloc(Node::MatMul(mat, y));

        let pattern = detect_sparsity_per_output(&arena, expr, 3, 3);

        let row0 = &pattern.indices[pattern.indptr[0]..pattern.indptr[1]];
        assert_eq!(row0, &[0, 1], "Row 0 should depend on states 0, 1");

        let row1 = &pattern.indices[pattern.indptr[1]..pattern.indptr[2]];
        assert_eq!(row1, &[0, 1, 2], "Row 1 should depend on states 0, 1, 2");

        let row2 = &pattern.indices[pattern.indptr[2]..pattern.indptr[3]];
        assert_eq!(row2, &[1, 2], "Row 2 should depend on states 1, 2");

        // Should need only 3 colors
        let coloring = color_columns(&pattern);
        assert!(coloring.n_colors <= 3);
    }

    #[test]
    fn test_per_output_vs_simple_diagonal() {
        // Compare per-output vs simple detection on diagonal case
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let y2 = arena.alloc(Node::StateVector { start: 2, end: 3 });
        let expr = arena.alloc(Node::Concat(vec![y0, y1, y2]));

        let pattern_simple = detect_sparsity_simple(&arena, expr, 3, 3);
        let pattern_per_output = detect_sparsity_per_output(&arena, expr, 3, 3);

        // Simple: all rows have same dependencies [0, 1, 2]
        let coloring_simple = color_columns(&pattern_simple);
        assert_eq!(coloring_simple.n_colors, 3, "Simple should need 3 colors");

        // Per-output: diagonal, only needs 1 color
        let coloring_per_output = color_columns(&pattern_per_output);
        assert_eq!(
            coloring_per_output.n_colors, 1,
            "Per-output should need only 1 color"
        );
    }

    #[test]
    fn test_per_output_sparsity_vector_add() {
        // f(y) = y + y (element-wise add)
        // Each output i should depend only on input i
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let expr = arena.alloc(Node::Add(y, y));

        let pattern = detect_sparsity_per_output(&arena, expr, 3, 3);

        for i in 0..3 {
            let row = &pattern.indices[pattern.indptr[i]..pattern.indptr[i + 1]];
            assert_eq!(row, &[i], "Row {i} should only depend on state {i}");
        }
    }

    #[test]
    fn test_per_output_sparsity_cross_dependency() {
        // f(y) = [y0 + y1, y1 + y2]
        // Creates off-diagonal dependencies
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let y2 = arena.alloc(Node::StateVector { start: 2, end: 3 });

        let f0 = arena.alloc(Node::Add(y0, y1));
        let f1 = arena.alloc(Node::Add(y1, y2));
        let expr = arena.alloc(Node::Concat(vec![f0, f1]));

        let pattern = detect_sparsity_per_output(&arena, expr, 2, 3);

        let row0 = &pattern.indices[pattern.indptr[0]..pattern.indptr[1]];
        assert_eq!(row0, &[0, 1], "Row 0 should depend on states 0, 1");

        let row1 = &pattern.indices[pattern.indptr[1]..pattern.indptr[2]];
        assert_eq!(row1, &[1, 2], "Row 1 should depend on states 1, 2");

        // Should need 2 colors (columns 0 and 2 can share, column 1 needs its own)
        let coloring = color_columns(&pattern);
        assert!(coloring.n_colors <= 2);
    }

    #[test]
    fn test_per_output_empty_deps() {
        // f(y) = [1.0, 2.0, 3.0] - constants with no state deps
        let mut arena = Arena::new();
        let c1 = arena.alloc(Node::Scalar(1.0));
        let c2 = arena.alloc(Node::Scalar(2.0));
        let c3 = arena.alloc(Node::Scalar(3.0));
        let expr = arena.alloc(Node::Concat(vec![c1, c2, c3]));

        let pattern = detect_sparsity_per_output(&arena, expr, 3, 3);

        // All rows should have no dependencies
        for i in 0..3 {
            let row = &pattern.indices[pattern.indptr[i]..pattern.indptr[i + 1]];
            assert!(row.is_empty(), "Row {i} should have no dependencies");
        }

        // Zero nonzeros
        assert_eq!(pattern.nnz(), 0);
    }

    #[test]
    fn test_large_tridiagonal_coloring_efficiency() {
        // Large tridiagonal system to verify O(3) coloring vs O(n)
        let n = 100;
        let mut arena = Arena::new();

        // Create y[i] for i in 0..n
        let y_nodes: Vec<NodeId> = (0..n)
            .map(|i| {
                arena.alloc(Node::StateVector {
                    start: i,
                    end: i + 1,
                })
            })
            .collect();

        // f[i] = y[i-1] + y[i] + y[i+1] (with boundary handling)
        let f_nodes: Vec<NodeId> = (0..n)
            .map(|i| {
                if i == 0 {
                    arena.alloc(Node::Add(y_nodes[0], y_nodes[1]))
                } else if i == n - 1 {
                    arena.alloc(Node::Add(y_nodes[n - 2], y_nodes[n - 1]))
                } else {
                    let partial = arena.alloc(Node::Add(y_nodes[i - 1], y_nodes[i]));
                    arena.alloc(Node::Add(partial, y_nodes[i + 1]))
                }
            })
            .collect();

        let expr = arena.alloc(Node::Concat(f_nodes));

        // Compare simple vs per-output
        let pattern_simple = detect_sparsity_simple(&arena, expr, n, n);
        let pattern_per_output = detect_sparsity_per_output(&arena, expr, n, n);

        let coloring_simple = color_columns(&pattern_simple);
        let coloring_per_output = color_columns(&pattern_per_output);

        // Simple should need n colors (dense)
        assert_eq!(
            coloring_simple.n_colors, n,
            "Simple detection should need {n} colors"
        );

        // Per-output should need only 3 colors (tridiagonal)
        assert!(
            coloring_per_output.n_colors <= 3,
            "Per-output should need <= 3 colors, got {}",
            coloring_per_output.n_colors
        );
    }

    #[test]
    fn test_bitset_zeros_empty() {
        let bs = BitSet::zeros(100);
        assert_eq!(bs.iter().count(), 0);
    }

    #[test]
    fn test_bitset_insert_and_iter_ascending() {
        let mut bs = BitSet::zeros(200);
        bs.insert(150);
        bs.insert(3);
        bs.insert(64);
        bs.insert(199);
        let v: Vec<usize> = bs.iter().collect();
        assert_eq!(v, vec![3, 64, 150, 199]);
    }

    #[test]
    fn test_bitset_insert_idempotent() {
        let mut bs = BitSet::zeros(10);
        bs.insert(5);
        bs.insert(5);
        let v: Vec<usize> = bs.iter().collect();
        assert_eq!(v, vec![5]);
    }

    #[test]
    fn test_bitset_union_with() {
        let mut a = BitSet::zeros(128);
        a.insert(1);
        a.insert(70);
        let mut b = BitSet::zeros(128);
        b.insert(70);
        b.insert(127);
        a.union_with(&b);
        let v: Vec<usize> = a.iter().collect();
        assert_eq!(v, vec![1, 70, 127]);
    }

    #[test]
    fn test_bitset_clone_eq() {
        let mut bs = BitSet::zeros(64);
        bs.insert(0);
        bs.insert(63);
        let clone = bs.clone();
        assert_eq!(bs, clone);
    }

    #[test]
    fn test_bitset_word_boundaries() {
        // Indices at u64-word boundaries
        let mut bs = BitSet::zeros(200);
        for i in [0, 63, 64, 127, 128, 191, 192, 199] {
            bs.insert(i);
        }
        let v: Vec<usize> = bs.iter().collect();
        assert_eq!(v, vec![0, 63, 64, 127, 128, 191, 192, 199]);
    }

    #[test]
    fn test_deep_chain_does_not_stack_overflow() {
        // Locks in stack safety for deep chains. The iterative analyzer
        // over Arena::topological_order makes this trivially bounded by
        // heap, not stack.
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let mut current = y;
        for _ in 0..5000 {
            current = arena.alloc(Node::Sin(current));
            current = arena.alloc(Node::Neg(current));
        }
        let pattern = detect_sparsity_per_output(&arena, current, 1, 1);
        let row = &pattern.indices[pattern.indptr[0]..pattern.indptr[1]];
        assert_eq!(row, &[0], "Deep chain output depends only on state 0");
    }

    #[test]
    fn dense_pattern_has_all_entries() {
        let p = SparsityPattern::dense(2, 3);
        assert_eq!(p.nrows, 2);
        assert_eq!(p.ncols, 3);
        assert_eq!(p.indptr, vec![0, 3, 6]);
        assert_eq!(p.indices, vec![0, 1, 2, 0, 1, 2]);
        assert_eq!(p.nnz(), 6);
    }

    #[test]
    fn dense_pattern_zero_cols_is_empty() {
        let p = SparsityPattern::dense(2, 0);
        assert_eq!(p.indptr, vec![0, 0, 0]);
        assert_eq!(p.nnz(), 0);
    }

    /// 3x3 CSR fixture: row 0 has 1 entry, row 1 has 3 entries, row 2 has 1 entry.
    fn make_row_filter_fixture() -> SparsityPattern {
        let mut pattern = SparsityPattern::new(3, 3);
        pattern.indptr = vec![0, 1, 4, 5];
        pattern.indices = vec![0, 0, 1, 2, 2];
        pattern
    }

    #[test]
    fn entry_rows_and_row_widths_invert_indptr() {
        let pattern = make_row_filter_fixture();
        assert_eq!(pattern.entry_rows(), vec![0, 1, 1, 1, 2]);
        assert_eq!(pattern.row_widths(), vec![1, 3, 1]);
    }
}
