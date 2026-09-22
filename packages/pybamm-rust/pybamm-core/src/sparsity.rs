//! Per-output sparsity of `d(root)/dy`, read off the DAG without evaluating it.
//!
//! Conservative: a missing entry is zero for every `y` as both AD modes differentiate,
//! which coloring relies on. Step functions, derivative-only nodes and `Conditional`
//! selectors differentiate to literal zeros, so they add no entries.

use std::ops::Range;
use std::rc::Rc;

use crate::arena::{Arena, NodeId, NodeMap};
use crate::error::CoreError;
use crate::ir::infer_sizes;
use crate::node::{CsrData, Node};

/// Each element's state dependencies, stored CSR-style.
///
/// Rows are ascending and deduplicated, so CSR rows built from them need no sort.
#[derive(Debug)]
struct DepRows {
    indptr: Vec<usize>,
    indices: Vec<usize>,
}

impl DepRows {
    fn with_capacity(len: usize) -> Self {
        let mut indptr = Vec::with_capacity(len + 1);
        indptr.push(0);
        Self {
            indptr,
            indices: Vec::new(),
        }
    }

    fn empty(len: usize) -> Self {
        Self {
            indptr: vec![0; len + 1],
            indices: Vec::new(),
        }
    }

    const fn len(&self) -> usize {
        self.indptr.len() - 1
    }

    fn row(&self, i: usize) -> &[usize] {
        &self.indices[self.indptr[i]..self.indptr[i + 1]]
    }

    fn push_row(&mut self, deps: &[usize]) {
        self.indices.extend_from_slice(deps);
        self.indptr.push(self.indices.len());
    }

    /// Push the union of two ascending rows.
    fn push_union(&mut self, a: &[usize], b: &[usize]) {
        extend_with_union(&mut self.indices, a, b);
        self.indptr.push(self.indices.len());
    }
}

/// Append the ascending union of two ascending, deduplicated slices to `out`.
fn extend_with_union(out: &mut Vec<usize>, a: &[usize], b: &[usize]) {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            },
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            },
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            },
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
}

/// The ascending, deduplicated union of every index in `rows`.
fn sorted_union<'a>(rows: impl IntoIterator<Item = &'a [usize]>) -> Vec<usize> {
    let mut union: Vec<usize> = rows.into_iter().flatten().copied().collect();
    union.sort_unstable();
    union.dedup();
    union
}

/// Transpose a compressed pattern (CSR to CSC or back) whose `indices` lie in `0..n_minor`.
///
/// Returns the transposed pointer array and, minor by minor, each entry's `(major, position)`
/// with majors ascending; `position` indexes the input `indices`.
pub fn transpose(
    n_minor: usize,
    ptr: &[usize],
    indices: &[usize],
) -> (Vec<usize>, Vec<(usize, usize)>) {
    let mut transposed_ptr = vec![0usize; n_minor + 1];
    for &minor in indices {
        transposed_ptr[minor + 1] += 1;
    }
    for minor in 0..n_minor {
        transposed_ptr[minor + 1] += transposed_ptr[minor];
    }
    let mut entries = vec![(0usize, 0usize); indices.len()];
    let mut fill = transposed_ptr[..n_minor].to_vec();
    for (major, span) in ptr.windows(2).enumerate() {
        let (start, end) = (span[0], span[1]);
        for (position, &minor) in (start..end).zip(&indices[start..end]) {
            entries[fill[minor]] = (major, position);
            fill[minor] += 1;
        }
    }
    (transposed_ptr, entries)
}

/// CSR structure of a Jacobian: which entries can be non-zero, without values.
///
/// `indptr` runs from 0 to `nnz` without decreasing, and each row's columns strictly
/// ascend below `ncols`. Every constructor upholds this, so merges, transposes and
/// slot lookups need not re-check it.
#[derive(Debug, Clone)]
pub struct SparsityPattern {
    nrows: usize,
    ncols: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
}

impl SparsityPattern {
    /// Build a pattern from CSR parts, validating every structural invariant.
    pub(crate) fn try_new(
        nrows: usize,
        ncols: usize,
        indptr: Vec<usize>,
        indices: Vec<usize>,
    ) -> Result<Self, CoreError> {
        check_csr_parts(nrows, ncols, &indptr, &indices)?;
        Ok(Self {
            nrows,
            ncols,
            indptr,
            indices,
        })
    }

    /// Wrap CSR parts an internal builder already guarantees valid.
    ///
    /// Only debug builds re-check them, as [`try_new`](Self::try_new) would.
    pub(crate) fn new_unchecked(
        nrows: usize,
        ncols: usize,
        indptr: Vec<usize>,
        indices: Vec<usize>,
    ) -> Self {
        if cfg!(debug_assertions)
            && let Err(err) = check_csr_parts(nrows, ncols, &indptr, &indices)
        {
            panic!("an internal sparsity pattern broke its invariants: {err}");
        }
        Self {
            nrows,
            ncols,
            indptr,
            indices,
        }
    }

    /// Rows, one per output element.
    #[inline]
    pub const fn nrows(&self) -> usize {
        self.nrows
    }

    /// Columns, one per state (or per parameter for a `df/dp` pattern).
    #[inline]
    pub const fn ncols(&self) -> usize {
        self.ncols
    }

    /// Row start offsets into [`indices`](Self::indices), length `nrows + 1`.
    #[inline]
    pub const fn indptr(&self) -> &[usize] {
        self.indptr.as_slice()
    }

    /// Column index of every structural entry, row by row.
    #[inline]
    pub const fn indices(&self) -> &[usize] {
        self.indices.as_slice()
    }

    /// Positions of `row`'s entries in [`indices`](Self::indices).
    #[inline]
    pub fn row_span(&self, row: usize) -> Range<usize> {
        self.indptr[row]..self.indptr[row + 1]
    }

    /// Number of structural non-zeros.
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

    /// Number of entries in each row.
    #[cfg(test)]
    pub fn row_widths(&self) -> Vec<usize> {
        (0..self.nrows)
            .map(|row| self.indptr[row + 1] - self.indptr[row])
            .collect()
    }

    /// Union another pattern's entries into this one.
    pub(crate) fn merge_with(&mut self, other: &Self) {
        assert_eq!(self.nrows, other.nrows, "Row count mismatch");
        assert_eq!(self.ncols, other.ncols, "Column count mismatch");

        let mut new_indices = Vec::with_capacity(self.indices.len() + other.indices.len());
        let mut new_indptr = vec![0usize; self.nrows + 1];

        for row in 0..self.nrows {
            extend_with_union(
                &mut new_indices,
                &self.indices[self.indptr[row]..self.indptr[row + 1]],
                &other.indices[other.indptr[row]..other.indptr[row + 1]],
            );
            new_indptr[row + 1] = new_indices.len();
        }

        self.indices = new_indices;
        self.indptr = new_indptr;
    }

    /// Fully dense pattern: every (row, col) entry present.
    pub(crate) fn dense(nrows: usize, ncols: usize) -> Self {
        let indptr = (0..=nrows).map(|r| r * ncols).collect();
        let indices = (0..nrows).flat_map(|_| 0..ncols).collect();
        Self {
            nrows,
            ncols,
            indptr,
            indices,
        }
    }
}

/// The structure of a caller-supplied matrix, refused if a row repeats or reorders columns.
impl TryFrom<&CsrData> for SparsityPattern {
    type Error = CoreError;

    fn try_from(csr: &CsrData) -> Result<Self, CoreError> {
        let shape = csr.shape();
        Self::try_new(
            shape.rows,
            shape.cols,
            csr.indptr().to_vec(),
            csr.indices().to_vec(),
        )
    }
}

/// Check the [`SparsityPattern`] invariants on raw CSR parts.
fn check_csr_parts(
    nrows: usize,
    ncols: usize,
    indptr: &[usize],
    indices: &[usize],
) -> Result<(), CoreError> {
    // Not `nrows + 1`, which wraps for `nrows == usize::MAX` and admits an empty indptr.
    if indptr.len().checked_sub(1) != Some(nrows) {
        return Err(CoreError::Csr(format!(
            "indptr length {} must equal rows + 1 for {nrows} rows",
            indptr.len(),
        )));
    }
    if indptr[0] != 0 {
        return Err(CoreError::Csr(format!(
            "indptr[0] must be 0, got {}",
            indptr[0]
        )));
    }
    if indptr.windows(2).any(|span| span[1] < span[0]) {
        return Err(CoreError::Csr("indptr must be non-decreasing".to_string()));
    }
    if indptr[nrows] != indices.len() {
        return Err(CoreError::Csr(format!(
            "indptr tail {} must equal nnz {}",
            indptr[nrows],
            indices.len()
        )));
    }
    for (row, span) in indptr.windows(2).enumerate() {
        let columns = &indices[span[0]..span[1]];
        if let Some(&[previous, column]) = columns.windows(2).find(|pair| pair[0] >= pair[1]) {
            return Err(CoreError::Csr(format!(
                "row {row} lists column {column} after column {previous}; columns must strictly ascend"
            )));
        }
        if let Some(&last) = columns.last()
            && last >= ncols
        {
            return Err(CoreError::Csr(format!(
                "column index {last} out of range for {ncols} columns"
            )));
        }
    }
    Ok(())
}

/// Per-element state dependencies of one node; `Scalar` broadcasts.
///
/// Shared, so a node that keeps its child's dependencies clones in O(1).
#[derive(Debug, Clone)]
enum ElementDeps {
    Scalar(Rc<[usize]>),
    Vector(Rc<DepRows>),
}

impl ElementDeps {
    /// `len` elements with no dependencies, a scalar when `len` is 1.
    fn empty(len: usize) -> Self {
        if len == 1 {
            Self::Scalar(Rc::from([]))
        } else {
            Self::Vector(Rc::new(DepRows::empty(len)))
        }
    }

    /// Wrap `rows`, collapsing a single element to a scalar.
    fn collapsed(rows: DepRows) -> Self {
        if rows.len() == 1 {
            Self::Scalar(Rc::from(rows.indices))
        } else {
            Self::Vector(Rc::new(rows))
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Scalar(_) => 1,
            Self::Vector(v) => v.len(),
        }
    }

    /// Whether no element depends on any state.
    fn is_state_free(&self) -> bool {
        match self {
            Self::Scalar(deps) => deps.is_empty(),
            Self::Vector(v) => v.indices.is_empty(),
        }
    }

    /// Dependencies of element `idx`, broadcasting scalars.
    fn get(&self, idx: usize) -> &[usize] {
        match self {
            Self::Scalar(deps) => deps,
            Self::Vector(v) => v.row(idx),
        }
    }

    fn union_all(&self) -> Vec<usize> {
        match self {
            Self::Scalar(deps) => deps.to_vec(),
            Self::Vector(v) => sorted_union((0..v.len()).map(|i| v.row(i))),
        }
    }
}

/// `id`'s dependencies, one row per element of its `width`, from its children's.
fn compute_node_deps(
    arena: &Arena,
    id: NodeId,
    width: usize,
    deps: &NodeMap<ElementDeps>,
) -> ElementDeps {
    match arena.get(id) {
        Node::StateVector { start, end } => {
            let mut rows = DepRows::with_capacity(end - start);
            for i in *start..*end {
                rows.push_row(&[i]);
            }
            ElementDeps::collapsed(rows)
        },
        // Both AD modes differentiate these to a literal zero: no state leaf, a step function,
        // or a derivative node whose own derivative is not taken.
        Node::StateVectorDot { .. }
        | Node::TangentStateVector { .. }
        | Node::TangentParameter { .. }
        | Node::Scalar(_)
        | Node::Time
        | Node::InputParameter { .. }
        | Node::ZeroVector { .. }
        | Node::Array(_)
        | Node::SparseMatrix(_)
        | Node::EqualHeaviside(..)
        | Node::NotEqualHeaviside(..)
        | Node::Equality(..)
        | Node::Sign(_)
        | Node::Floor(_)
        | Node::Ceiling(_)
        | Node::Interpolant1DLinearDeriv { .. }
        | Node::Interpolant1DCubicDeriv { .. }
        | Node::InterpolantNdPartial { .. }
        | Node::ReduceArgSelect { .. } => ElementDeps::empty(width),
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
        | Node::Interpolant1DLinear { child: a, .. }
        | Node::Interpolant1DCubic { child: a, .. } => child_deps(deps, *a).clone(),
        Node::MaxReduce(a) | Node::MinReduce(a) => {
            ElementDeps::Scalar(child_deps(deps, *a).union_all().into())
        },
        Node::Add(a, b)
        | Node::Sub(a, b)
        | Node::Mul(a, b)
        | Node::Div(a, b)
        | Node::Pow(a, b)
        | Node::Minimum(a, b)
        | Node::Maximum(a, b)
        | Node::Modulo(a, b)
        | Node::Hypot(a, b) => combine_binary_deps(child_deps(deps, *a), child_deps(deps, *b)),
        Node::MatMul(mat_id, vec_id) => {
            let vec_d = child_deps(deps, *vec_id);
            match arena.get(*mat_id) {
                Node::SparseMatrix(csr) => {
                    let n_rows = csr.shape.rows;
                    let mut rows = DepRows::with_capacity(n_rows);
                    let mut scratch = Vec::new();
                    for row in 0..n_rows {
                        scratch.clear();
                        for &col in &csr.indices[csr.indptr[row]..csr.indptr[row + 1]] {
                            scratch.extend_from_slice(vec_d.get(col));
                        }
                        scratch.sort_unstable();
                        scratch.dedup();
                        rows.push_row(&scratch);
                    }
                    ElementDeps::collapsed(rows)
                },
                Node::Array(arr) => {
                    let u = vec_d.union_all();
                    let n_rows = arr.shape.rows;
                    if n_rows == 1 {
                        ElementDeps::Scalar(u.into())
                    } else {
                        let mut rows = DepRows::with_capacity(n_rows);
                        for _ in 0..n_rows {
                            rows.push_row(&u);
                        }
                        ElementDeps::Vector(Rc::new(rows))
                    }
                },
                other => unreachable!("infer_sizes admits only constant matrices, got {other:?}"),
            }
        },
        Node::Index { child, start, end } => {
            let child = child_deps(deps, *child);
            let mut rows = DepRows::with_capacity(width);
            for i in *start..*end {
                rows.push_row(child.get(i));
            }
            ElementDeps::collapsed(rows)
        },
        Node::Concat(children) => {
            let mut rows = DepRows::with_capacity(width);
            for &child in children {
                let child = child_deps(deps, child);
                for i in 0..child.len() {
                    rows.push_row(child.get(i));
                }
            }
            ElementDeps::collapsed(rows)
        },
        Node::InterpolantNd { children, .. } => {
            let mut acc = child_deps(deps, children[0]).clone();
            for &c in &children[1..] {
                acc = combine_binary_deps(&acc, child_deps(deps, c));
            }
            acc
        },
        // Only the branches are differentiated; the selector's tangent is never taken.
        Node::Conditional { branches, .. } => {
            let branches: Vec<&ElementDeps> =
                branches.iter().map(|&b| child_deps(deps, b)).collect();
            let mut rows = DepRows::with_capacity(width);
            let mut scratch = Vec::new();
            for i in 0..width {
                scratch.clear();
                for branch in &branches {
                    scratch.extend_from_slice(branch.get(i));
                }
                scratch.sort_unstable();
                scratch.dedup();
                rows.push_row(&scratch);
            }
            ElementDeps::collapsed(rows)
        },
    }
}

fn child_deps(deps: &NodeMap<ElementDeps>, id: NodeId) -> &ElementDeps {
    deps.get(id)
        .expect("child must be processed before parent in topological order")
}

/// Combine dependencies for binary operations with broadcast semantics.
fn combine_binary_deps(a: &ElementDeps, b: &ElementDeps) -> ElementDeps {
    if let (ElementDeps::Scalar(da), ElementDeps::Scalar(db)) = (a, b) {
        let mut union = Vec::with_capacity(da.len() + db.len());
        extend_with_union(&mut union, da, db);
        return ElementDeps::Scalar(union.into());
    }
    // A state-free operand adds nothing, so the other's rows pass through shared.
    for (constant, other) in [(a, b), (b, a)] {
        if matches!(other, ElementDeps::Vector(_)) && constant.is_state_free() {
            return other.clone();
        }
    }
    let len = a.len().max(b.len());
    let mut rows = DepRows::with_capacity(len);
    for i in 0..len {
        rows.push_union(a.get(i), b.get(i));
    }
    ElementDeps::Vector(Rc::new(rows))
}

/// Detect the sparsity pattern, tracking each output element's dependencies separately.
///
/// Panics, as lowering would, if `root`'s graph breaks a width rule or is not
/// `n_outputs` wide. `n_states` must cover every state `root` reads; only debug builds
/// check it.
pub fn detect_sparsity_per_output(
    arena: &Arena,
    root: NodeId,
    n_outputs: usize,
    n_states: usize,
) -> SparsityPattern {
    let order = arena.topological_order(root);
    let widths = infer_sizes(arena, &order);
    assert_eq!(
        widths[root.index()],
        n_outputs,
        "the root is {} wide, not {n_outputs}",
        widths[root.index()]
    );
    let output_deps = arena.fold_frontier(&order, |id, deps| {
        let width = widths[id.index()];
        let node_deps = compute_node_deps(arena, id, width, deps);
        // In release too: a row count off its width shifts every later row's pattern.
        assert_eq!(
            node_deps.len(),
            width,
            "node {id:?} has {width} elements but {} dependency rows",
            node_deps.len()
        );
        node_deps
    });
    // `DepRows` keeps every row ascending and deduplicated, so its CSR is the pattern's.
    let (indptr, indices) = match output_deps {
        ElementDeps::Scalar(deps) => (vec![0, deps.len()], deps.to_vec()),
        ElementDeps::Vector(rows) => Rc::try_unwrap(rows).map_or_else(
            |rows| (rows.indptr.clone(), rows.indices.clone()),
            |rows| (rows.indptr, rows.indices),
        ),
    };
    SparsityPattern::new_unchecked(n_outputs, n_states, indptr, indices)
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::coloring::color_columns;
    use crate::node::{InterpolantData, Shape};

    /// A pattern whose row `r` holds the columns `rows[r]`, which must ascend.
    pub fn pattern_from_rows(ncols: usize, rows: &[Vec<usize>]) -> SparsityPattern {
        let mut indptr = vec![0];
        let mut indices = Vec::new();
        for row in rows {
            indices.extend_from_slice(row);
            indptr.push(indices.len());
        }
        SparsityPattern::try_new(rows.len(), ncols, indptr, indices).expect("a valid pattern")
    }

    /// Each row's columns.
    fn rows(pattern: &SparsityPattern) -> Vec<&[usize]> {
        (0..pattern.nrows)
            .map(|row| &pattern.indices[pattern.row_span(row)])
            .collect()
    }

    /// `n` single-element state leaves.
    fn scalar_states(arena: &mut Arena, n: usize) -> Vec<NodeId> {
        (0..n)
            .map(|i| {
                arena.alloc(Node::StateVector {
                    start: i,
                    end: i + 1,
                })
            })
            .collect()
    }

    #[test]
    fn test_per_output_sparsity_diagonal() {
        let mut arena = Arena::new();
        let y = scalar_states(&mut arena, 3);
        let expr = arena.alloc(Node::Concat(y));
        let pattern = detect_sparsity_per_output(&arena, expr, 3, 3);
        assert_eq!(rows(&pattern), [[0].as_slice(), &[1], &[2]]);
        assert_eq!(color_columns(&pattern).n_colors(), 1);
    }

    #[test]
    fn test_per_output_sparsity_identity_passthrough() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 4 });
        let pattern = detect_sparsity_per_output(&arena, y, 4, 4);
        assert_eq!(rows(&pattern), [[0].as_slice(), &[1], &[2], &[3]]);
        assert_eq!(color_columns(&pattern).n_colors(), 1);
    }

    #[test]
    fn test_sparsity_tridiagonal() {
        let n = 100;
        let mut arena = Arena::new();
        let y = scalar_states(&mut arena, n);
        let f: Vec<NodeId> = (0..n)
            .map(|i| {
                let left = arena.alloc(Node::Add(y[i.saturating_sub(1)], y[i]));
                if i + 1 < n {
                    arena.alloc(Node::Add(left, y[i + 1]))
                } else {
                    left
                }
            })
            .collect();
        let expr = arena.alloc(Node::Concat(f));
        let pattern = detect_sparsity_per_output(&arena, expr, n, n);
        let rows = rows(&pattern);
        assert_eq!(rows[..3], [[0, 1].as_slice(), &[0, 1, 2], &[1, 2, 3]]);
        assert_eq!(rows[n - 1], [n - 2, n - 1]);
        let colors = color_columns(&pattern).n_colors();
        assert!(colors <= 3, "expected <= 3 colours, got {colors}");
    }

    #[test]
    fn test_per_output_sparsity_scalar_broadcast() {
        let mut arena = Arena::new();
        let c = arena.alloc(Node::Scalar(2.0));
        let f: Vec<NodeId> = scalar_states(&mut arena, 3)
            .into_iter()
            .map(|yi| arena.alloc(Node::Mul(yi, c)))
            .collect();
        let expr = arena.alloc(Node::Concat(f));
        let pattern = detect_sparsity_per_output(&arena, expr, 3, 3);
        assert_eq!(rows(&pattern), [[0].as_slice(), &[1], &[2]]);
    }

    #[test]
    fn test_per_output_sparsity_unary_preserves() {
        let mut arena = Arena::new();
        let y = scalar_states(&mut arena, 3);
        let f0 = arena.alloc(Node::Exp(y[0]));
        let f1 = arena.alloc(Node::Sin(y[1]));
        let f2 = arena.alloc(Node::Log(y[2]));
        let expr = arena.alloc(Node::Concat(vec![f0, f1, f2]));
        let pattern = detect_sparsity_per_output(&arena, expr, 3, 3);
        assert_eq!(rows(&pattern), [[0].as_slice(), &[1], &[2]]);
    }

    #[test]
    fn test_per_output_sparsity_index() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 4 });
        let expr = arena.alloc(Node::Index {
            child: y,
            start: 1,
            end: 3,
        });
        let pattern = detect_sparsity_per_output(&arena, expr, 2, 4);
        assert_eq!(rows(&pattern), [[1].as_slice(), &[2]]);
    }

    #[test]
    fn test_per_output_sparsity_reduction() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let expr = arena.alloc(Node::MaxReduce(y));
        let pattern = detect_sparsity_per_output(&arena, expr, 1, 3);
        assert_eq!(rows(&pattern), [[0, 1, 2].as_slice()]);
    }

    #[test]
    fn test_per_output_sparsity_sparse_matmul() {
        let mut arena = Arena::new();
        let csr = CsrData {
            indptr: vec![0, 2, 5, 7],
            indices: vec![0, 1, 0, 1, 2, 1, 2],
            data: vec![1.0; 7],
            shape: Shape::matrix(3, 3),
        };
        let mat = arena.alloc(Node::SparseMatrix(Box::new(csr)));
        let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let expr = arena.alloc(Node::MatMul(mat, y));
        let pattern = detect_sparsity_per_output(&arena, expr, 3, 3);
        assert_eq!(rows(&pattern), [[0, 1].as_slice(), &[0, 1, 2], &[1, 2]]);
        assert!(color_columns(&pattern).n_colors() <= 3);
    }

    #[test]
    fn test_per_output_sparsity_vector_add() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let expr = arena.alloc(Node::Add(y, y));
        let pattern = detect_sparsity_per_output(&arena, expr, 3, 3);
        assert_eq!(rows(&pattern), [[0].as_slice(), &[1], &[2]]);
    }

    #[test]
    fn test_per_output_sparsity_cross_dependency() {
        let mut arena = Arena::new();
        let y = scalar_states(&mut arena, 3);
        let f0 = arena.alloc(Node::Add(y[0], y[1]));
        let f1 = arena.alloc(Node::Add(y[1], y[2]));
        let expr = arena.alloc(Node::Concat(vec![f0, f1]));
        let pattern = detect_sparsity_per_output(&arena, expr, 2, 3);
        assert_eq!(rows(&pattern), [[0, 1].as_slice(), &[1, 2]]);
        assert!(color_columns(&pattern).n_colors() <= 2);
    }

    #[test]
    fn test_per_output_empty_deps() {
        let mut arena = Arena::new();
        let c: Vec<NodeId> = [1.0, 2.0, 3.0]
            .map(|v| arena.alloc(Node::Scalar(v)))
            .to_vec();
        let expr = arena.alloc(Node::Concat(c));
        let pattern = detect_sparsity_per_output(&arena, expr, 3, 3);
        assert_eq!(pattern.indptr, [0, 0, 0, 0]);
    }

    #[test]
    fn push_union_merges_ascending_rows_without_duplicates() {
        let mut rows = DepRows::with_capacity(2);
        rows.push_union(&[1, 70, 199], &[0, 70, 127]);
        rows.push_union(&[], &[5]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows.row(0), &[0, 1, 70, 127, 199]);
        assert_eq!(rows.row(1), &[5]);
    }

    #[test]
    fn a_state_free_operand_shares_the_other_operands_rows() {
        let mut rows = DepRows::with_capacity(3);
        for state in [4, 1, 7] {
            rows.push_row(&[state]);
        }
        let varying = ElementDeps::Vector(Rc::new(rows));
        let ElementDeps::Vector(input) = &varying else {
            unreachable!("built as a vector")
        };
        for constant in [ElementDeps::empty(1), ElementDeps::empty(3)] {
            for combined in [
                combine_binary_deps(&constant, &varying),
                combine_binary_deps(&varying, &constant),
            ] {
                let ElementDeps::Vector(output) = combined else {
                    panic!("a vector operand keeps the result a vector")
                };
                assert!(Rc::ptr_eq(&output, input), "rows must pass through shared");
            }
        }
    }

    #[test]
    fn sorted_union_sorts_and_deduplicates() {
        let union = sorted_union([[9, 3].as_slice(), &[3, 1], &[]]);
        assert_eq!(union, vec![1, 3, 9]);
    }

    #[test]
    fn shared_subexpression_deps_survive_until_their_last_parent() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let shared = arena.alloc(Node::Exp(y));
        let first = arena.alloc(Node::Sin(shared));
        let second = arena.alloc(Node::Add(shared, first));
        let expr = arena.alloc(Node::Add(first, second));
        let pattern = detect_sparsity_per_output(&arena, expr, 3, 3);
        assert_eq!(pattern.indptr, vec![0, 1, 2, 3]);
        assert_eq!(pattern.indices, vec![0, 1, 2]);
    }

    /// `Concat([p * y0, y1])` for a width-3 input `p`: four elements, three on `y0`.
    fn wide_parameter_concat(arena: &mut Arena) -> NodeId {
        let p = arena.alloc(Node::InputParameter {
            name: "p".into(),
            index: 0,
            offset: 0,
            width: 3,
        });
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let scaled = arena.alloc(Node::Mul(p, y0));
        arena.alloc(Node::Concat(vec![scaled, y1]))
    }

    #[test]
    fn a_wide_input_parameter_keeps_concat_rows_aligned() {
        let mut arena = Arena::new();
        let concat = wide_parameter_concat(&mut arena);
        let middle = arena.alloc(Node::Index {
            child: concat,
            start: 1,
            end: 2,
        });
        let pattern = detect_sparsity_per_output(&arena, middle, 1, 2);
        assert_eq!(rows(&pattern), [[0].as_slice()], "p[1] * y0 depends on y0");
    }

    #[test]
    fn a_wide_input_parameter_slices_past_its_first_element() {
        let mut arena = Arena::new();
        let concat = wide_parameter_concat(&mut arena);
        let tail = arena.alloc(Node::Index {
            child: concat,
            start: 2,
            end: 4,
        });
        let pattern = detect_sparsity_per_output(&arena, tail, 2, 2);
        assert_eq!(rows(&pattern), [[0].as_slice(), &[1]]);
    }

    #[test]
    fn a_conditional_selector_adds_no_entries() {
        // cond(H(sum(y) <= 0), [sin(y), cos(y)]) is diagonal, not 2500 entries and 50 colours.
        let n = 50;
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: n });
        let ones = CsrData::try_new(
            vec![0, n],
            (0..n).collect(),
            vec![1.0; n],
            Shape::matrix(1, n),
        )
        .expect("valid row matrix");
        let ones = arena.alloc(Node::SparseMatrix(Box::new(ones)));
        let sum = arena.alloc(Node::MatMul(ones, y));
        let zero = arena.alloc(Node::Scalar(0.0));
        let selector = arena.alloc(Node::EqualHeaviside(sum, zero));
        let sin = arena.alloc(Node::Sin(y));
        let cos = arena.alloc(Node::Cos(y));
        let root = arena.alloc(Node::Conditional {
            selector,
            branches: vec![sin, cos],
        });
        let pattern = detect_sparsity_per_output(&arena, root, n, n);
        assert_eq!(pattern.nnz(), n);
        assert_eq!(color_columns(&pattern).n_colors(), 1);
    }

    #[test]
    fn steps_and_derivative_interpolants_add_no_entries() {
        // sin(y) * g(y0), for g a step function or an interpolant slope, stays diagonal.
        let n = 50;
        let table = InterpolantData::try_new(vec![0.0, 1.0], vec![0.0, 2.0]).expect("valid table");
        let factors: [fn(NodeId, NodeId, &InterpolantData) -> Node; 7] = [
            |y0, half, _| Node::EqualHeaviside(y0, half),
            |y0, half, _| Node::NotEqualHeaviside(y0, half),
            |y0, half, _| Node::Equality(y0, half),
            |y0, _, _| Node::Sign(y0),
            |y0, _, _| Node::Floor(y0),
            |y0, _, _| Node::Ceiling(y0),
            |y0, _, table| Node::Interpolant1DLinearDeriv {
                data: Box::new(table.clone()),
                child: y0,
            },
        ];
        for factor in factors {
            let mut arena = Arena::new();
            let y = arena.alloc(Node::StateVector { start: 0, end: n });
            let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
            let half = arena.alloc(Node::Scalar(0.5));
            let sin = arena.alloc(Node::Sin(y));
            let step = arena.alloc(factor(y0, half, &table));
            let root = arena.alloc(Node::Mul(sin, step));
            let pattern = detect_sparsity_per_output(&arena, root, n, n);
            assert_eq!(pattern.nnz(), n, "{:?}", arena.get(step));
        }
    }

    #[test]
    #[should_panic(expected = "the root is 2 wide, not 3")]
    fn a_misdeclared_output_width_is_refused() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let _ = detect_sparsity_per_output(&arena, y, 3, 2);
    }

    #[test]
    #[should_panic(expected = "invalid expression graph")]
    fn a_width_invalid_graph_is_refused() {
        // `Add(Concat([y, y0]), y)` mixes widths 3 and 2, which no broadcast admits.
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let concat = arena.alloc(Node::Concat(vec![y, y0]));
        let root = arena.alloc(Node::Add(concat, y));
        let _ = detect_sparsity_per_output(&arena, root, 3, 2);
    }

    #[test]
    fn test_deep_chain_does_not_stack_overflow() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let mut current = y;
        for _ in 0..5000 {
            current = arena.alloc(Node::Sin(current));
            current = arena.alloc(Node::Neg(current));
        }
        let pattern = detect_sparsity_per_output(&arena, current, 1, 1);
        assert_eq!(rows(&pattern), [[0].as_slice()]);
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

    fn make_row_filter_fixture() -> SparsityPattern {
        pattern_from_rows(3, &[vec![0], vec![0, 1, 2], vec![2]])
    }

    #[test]
    fn transpose_lists_each_column_in_row_order_and_round_trips() {
        let pattern = make_row_filter_fixture();
        let (colptr, entries) = transpose(pattern.ncols, &pattern.indptr, &pattern.indices);
        assert_eq!(colptr, vec![0, 2, 3, 5]);
        assert_eq!(entries, vec![(0, 0), (1, 1), (1, 2), (1, 3), (2, 4)]);

        let rowind: Vec<usize> = entries.iter().map(|&(row, _)| row).collect();
        let (indptr, back) = transpose(pattern.nrows, &colptr, &rowind);
        assert_eq!(indptr, pattern.indptr);
        let indices: Vec<usize> = back.iter().map(|&(col, _)| col).collect();
        assert_eq!(indices, pattern.indices);
    }

    #[test]
    fn entry_rows_and_row_widths_invert_indptr() {
        let pattern = make_row_filter_fixture();
        assert_eq!(pattern.entry_rows(), vec![0, 1, 1, 1, 2]);
        assert_eq!(pattern.row_widths(), vec![1, 3, 1]);
    }

    fn csr_error(result: Result<SparsityPattern, CoreError>) -> String {
        match result {
            Err(CoreError::Csr(message)) => message,
            Err(other) => panic!("expected a CSR error, got {other}"),
            Ok(_) => panic!("malformed parts were accepted"),
        }
    }

    #[test]
    fn try_new_refuses_each_broken_invariant() {
        let cases: [(&str, usize, Vec<usize>, Vec<usize>); 7] = [
            ("indptr length", 2, vec![0, 1], vec![0]),
            ("indptr[0]", 1, vec![1, 1], vec![0]),
            ("non-decreasing", 2, vec![0, 2, 1], vec![0, 1]),
            ("indptr tail", 1, vec![0, 1], vec![0, 1]),
            (
                "row 0 lists column 0 after column 1",
                1,
                vec![0, 2],
                vec![1, 0],
            ),
            (
                "row 1 lists column 2 after column 2",
                2,
                vec![0, 1, 3],
                vec![0, 2, 2],
            ),
            ("column index 3 out of range", 1, vec![0, 2], vec![0, 3]),
        ];
        for (expected, nrows, indptr, indices) in cases {
            let message = csr_error(SparsityPattern::try_new(nrows, 3, indptr, indices));
            assert!(
                message.contains(expected),
                "{expected:?} not in {message:?}"
            );
        }
        let pattern = SparsityPattern::try_new(2, 3, vec![0, 2, 2], vec![0, 2]).expect("valid");
        assert_eq!(pattern.row_span(0), 0..2);
    }

    #[test]
    fn a_csr_matrix_must_list_each_rows_columns_ascending() {
        let csr = |indices: Vec<usize>| {
            CsrData::try_new(vec![0, 2, 3], indices, vec![1.0; 3], Shape::matrix(2, 2))
                .expect("structurally valid CSR")
        };
        let message = csr_error(SparsityPattern::try_from(&csr(vec![1, 0, 1])));
        assert!(message.contains("row 0"), "{message}");
        let pattern = SparsityPattern::try_from(&csr(vec![0, 1, 1])).expect("ascending rows");
        assert_eq!(
            (pattern.indptr(), pattern.indices()),
            (&[0, 2, 3][..], &[0, 1, 1][..])
        );
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "broke its invariants")]
    fn new_unchecked_rechecks_in_debug_builds() {
        let _ = SparsityPattern::new_unchecked(1, 2, vec![0, 2], vec![1, 1]);
    }
}
