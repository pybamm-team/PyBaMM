//! Lift wide output rows of an expression out as standalone scalar expressions.
//!
//! An element index is pushed down the DAG; a node with no cheap indexed form is copied whole,
//! once per block, and sliced.

use rustc_hash::FxHashMap;

use crate::arena::{Arena, Dag, DagRef, NodeId, NodeMap};
use crate::node::{CsrData, Node, Shape};
use crate::simplify::{dce, node_len};

/// A matrix row reading at least `1 / VECTOR_ROW_SHARE` of its vector stays one vector product:
/// the whole vector costs less than a scalar term per column.
const VECTOR_ROW_SHARE: usize = 4;

/// A group of output rows lifted out as one expression owning its own arena.
#[derive(Debug)]
pub struct ScalarRowBlock {
    /// Parent output rows this block holds, in element order.
    pub rows: Vec<usize>,
    /// `Concat` of the extracted rows, one element each.
    pub dag: Dag,
}

/// Extract `rows` of `root` as one block: a `Concat` whose element `i` is `rows[i]`.
///
/// Every row below the root's width extracts.
pub fn extract_scalar_rows(dag: DagRef<'_>, rows: &[usize]) -> ScalarRowBlock {
    let mut extractor = Extractor::new(dag.arena());
    let width = extractor.width(dag.root());
    let extracted = rows
        .iter()
        .map(|&row| {
            assert!(row < width, "row {row} is outside the root's {width} rows");
            extractor.push(dag.root(), row)
        })
        .collect();
    let concat = extractor.work.alloc(Node::Concat(extracted));
    // Renumber children-first, so the block's layout follows its structure alone.
    let dag = dce(DagRef::new(&extractor.work, concat));
    ScalarRowBlock {
        rows: rows.to_vec(),
        dag,
    }
}

/// Index-push walker over one primal DAG.
struct Extractor<'a> {
    source: &'a Arena,
    /// Every synthesised scalar node, plus copies of the source nodes they reference.
    work: Arena,
    /// Where each referenced source node was copied into `work`.
    imported: NodeMap<NodeId>,
    /// `node_len` memo over `source`.
    widths: NodeMap<usize>,
    /// Element `index` of node `id`, shared across every requested row.
    cache: FxHashMap<(NodeId, usize), NodeId>,
    /// Synthesised `Scalar` nodes, interned by bit pattern.
    scalars: FxHashMap<u64, NodeId>,
    /// [`push`](Self::push)'s work stack, kept to reuse its capacity across rows.
    frames: Vec<Frame>,
    /// Elements of finished pushes, each awaiting the frame below it.
    elements: Vec<NodeId>,
}

impl<'a> Extractor<'a> {
    fn new(source: &'a Arena) -> Self {
        Self {
            source,
            work: Arena::new(),
            imported: NodeMap::new(source.len()),
            widths: NodeMap::new(source.len()),
            cache: FxHashMap::default(),
            scalars: FxHashMap::default(),
            frames: Vec::new(),
            elements: Vec::new(),
        }
    }

    fn alloc(&mut self, node: Node) -> NodeId {
        self.work.alloc(node)
    }

    fn width(&mut self, id: NodeId) -> usize {
        node_len(self.source, id, &mut self.widths)
    }

    /// Source node `id` as a node of `work`, copying its cone on first use.
    fn import(&mut self, id: NodeId) -> NodeId {
        if let Some(&copied) = self.imported.get(id) {
            return copied;
        }
        let source = self.source;
        let mut pending = vec![id];
        while let Some(&top) = pending.last() {
            if self.imported.contains_key(top) {
                pending.pop();
                continue;
            }
            let node = source.get(top);
            let n_pending = pending.len();
            node.for_each_child(|child| {
                if !self.imported.contains_key(child) {
                    pending.push(child);
                }
            });
            if pending.len() == n_pending {
                pending.pop();
                let copy = node.map_children(|child| self.imported.rebuilt(child));
                let copied = self.work.alloc(copy);
                self.imported.insert(top, copied);
            }
        }
        self.imported.rebuilt(id)
    }

    /// A `Scalar` node for `value`, interned so equal coefficients share one.
    fn scalar(&mut self, value: f64) -> NodeId {
        if let Some(&hit) = self.scalars.get(&value.to_bits()) {
            return hit;
        }
        let id = self.alloc(Node::Scalar(value));
        self.scalars.insert(value.to_bits(), id);
        id
    }

    /// Element `index` of `id` as a width-1 node in `work`.
    ///
    /// Walks with explicit stacks, so graph depth is bounded by the heap.
    fn push(&mut self, id: NodeId, index: usize) -> NodeId {
        self.frames.clear();
        self.elements.clear();
        self.frames.push(Frame::Push(id, index));
        while let Some(frame) = self.frames.pop() {
            match frame {
                Frame::Push(id, index) => self.descend(id, index),
                Frame::Cache(id, index) => {
                    let &element = self.elements.last().expect("the push left its element");
                    self.cache.insert((id, index), element);
                },
                Frame::Rebuild {
                    id,
                    index,
                    operands,
                } => {
                    let mut elements = self.elements[operands..].iter();
                    let indexed = self
                        .source
                        .get(id)
                        .map_children(|_| *elements.next().expect("each child left one element"));
                    self.elements.truncate(operands);
                    let element = self.alloc(indexed);
                    self.finish(id, index, element);
                },
                Frame::Term { coefficient, first } => {
                    let element = self.elements.pop().expect("the vector element is on top");
                    let term = self.scaled(coefficient, element);
                    // Left to right as the vector matmul sums, but without its `+0.0` start or
                    // zero `Array` terms, so a signed zero or a `0 * inf` can differ.
                    let sum = if first {
                        term
                    } else {
                        let accumulated = self.elements.pop().expect("the row sum is below it");
                        self.alloc(Node::Add(accumulated, term))
                    };
                    self.elements.push(sum);
                },
            }
        }
        let [element] = self.elements[..] else {
            unreachable!("the walk leaves exactly the requested element")
        };
        element
    }

    /// Leave `element` as element `index` of `id`, cached for later rows.
    fn finish(&mut self, id: NodeId, index: usize, element: NodeId) {
        self.cache.insert((id, index), element);
        self.elements.push(element);
    }

    /// Leave element `index` of `id` as a slice of `id` copied whole.
    fn slice_whole(&mut self, id: NodeId, index: usize) {
        let whole = self.import(id);
        let element = self.alloc(Node::Index {
            child: whole,
            start: index,
            end: index + 1,
        });
        self.finish(id, index, element);
    }

    /// Run a [`Frame::Push`]: leave its element, or queue the frames that build it.
    fn descend(&mut self, id: NodeId, index: usize) {
        let width = self.width(id);
        debug_assert!(index < width, "element {index} of a node {width} wide");
        if width == 1 {
            let whole = self.import(id);
            self.elements.push(whole);
            return;
        }
        if let Some(&hit) = self.cache.get(&(id, index)) {
            self.elements.push(hit);
            return;
        }
        // Copied out so the match borrows the source graph, not `self`.
        let source = self.source;
        match source.get(id) {
            Node::Concat(children) => {
                let mut offset = 0;
                let mut located = None;
                for &child in children {
                    let len = self.width(child);
                    if index < offset + len {
                        located = Some(Frame::Push(child, index - offset));
                        break;
                    }
                    offset += len;
                }
                let located = located.expect("the children's widths sum to the concat's");
                self.frames.extend([Frame::Cache(id, index), located]);
            },
            Node::Index { child, start, .. } => {
                let at = start + index;
                self.frames
                    .extend([Frame::Cache(id, index), Frame::Push(*child, at)]);
            },
            Node::StateVector { start, .. } => {
                let at = start + index;
                let element = self.alloc(Node::StateVector {
                    start: at,
                    end: at + 1,
                });
                self.finish(id, index, element);
            },
            Node::StateVectorDot { start, .. } => {
                let at = start + index;
                let element = self.alloc(Node::StateVectorDot {
                    start: at,
                    end: at + 1,
                });
                self.finish(id, index, element);
            },
            Node::Array(array) => {
                let element = self.scalar(array.data()[index]);
                self.finish(id, index, element);
            },
            Node::ZeroVector { .. } => {
                let element = self.scalar(0.0);
                self.finish(id, index, element);
            },
            Node::MatMul(matrix, vector) => match matrix_row(source.get(*matrix), index) {
                None => self.slice_whole(id, index),
                Some(terms) if terms.is_empty() => {
                    let element = self.scalar(0.0);
                    self.finish(id, index, element);
                },
                Some(terms) if terms.len() * VECTOR_ROW_SHARE >= self.width(*vector) => {
                    let cols = self.width(*vector);
                    let row = row_matrix(terms, cols);
                    let (row, vector) = (self.alloc(row), self.import(*vector));
                    let element = self.alloc(Node::MatMul(row, vector));
                    self.finish(id, index, element);
                },
                Some(terms) => {
                    self.frames.push(Frame::Cache(id, index));
                    for (k, &(coefficient, col)) in terms.iter().enumerate().rev() {
                        let first = k == 0;
                        self.frames.extend([
                            Frame::Term { coefficient, first },
                            Frame::Push(*vector, col),
                        ]);
                    }
                },
            },
            // Indexes each operand, keeping width-1 ones whole.
            node if is_elementwise(node) => {
                self.frames.push(Frame::Rebuild {
                    id,
                    index,
                    operands: self.elements.len(),
                });
                let first_child = self.frames.len();
                node.for_each_child(|child| {
                    let at = if self.width(child) == 1 { 0 } else { index };
                    self.frames.push(Frame::Push(child, at));
                });
                self.frames[first_child..].reverse();
            },
            // Input leaves, interpolants and the like: one copy serves every element.
            _ => self.slice_whole(id, index),
        }
    }

    /// `coefficient * element`, folding the exact `±1` cases.
    fn scaled(&mut self, coefficient: f64, element: NodeId) -> NodeId {
        if coefficient.to_bits() == 1.0_f64.to_bits() {
            return element;
        }
        if coefficient.to_bits() == (-1.0_f64).to_bits() {
            return self.alloc(Node::Neg(element));
        }
        let scale = self.scalar(coefficient);
        self.alloc(Node::Mul(scale, element))
    }
}

/// Work left in [`Extractor::push`]; each frame waits on those pushed above it.
#[derive(Clone, Copy)]
enum Frame {
    /// Leave element `index` of `id` on the element stack.
    Push(NodeId, usize),
    /// Record the element on top as element `index` of `id`.
    Cache(NodeId, usize),
    /// Rebuild elementwise `id` over its children's elements, `operands` onwards, as its
    /// element `index`.
    Rebuild {
        id: NodeId,
        index: usize,
        operands: usize,
    },
    /// Fold `coefficient` times the element on top into the matmul row sum below it,
    /// or start the sum when `first`.
    Term { coefficient: f64, first: bool },
}

/// The `(coefficient, column)` terms of row `row` of a constant matrix: a sparse row's stored
/// entries, or a dense row's non-zeros.
fn matrix_row(matrix: &Node, row: usize) -> Option<Vec<(f64, usize)>> {
    match matrix {
        Node::SparseMatrix(csr) => {
            let (start, end) = (csr.indptr()[row], csr.indptr()[row + 1]);
            Some(
                csr.data()[start..end]
                    .iter()
                    .copied()
                    .zip(csr.indices()[start..end].iter().copied())
                    .collect(),
            )
        },
        Node::Array(array) => {
            let cols = array.shape().cols;
            Some(
                array.data()[row * cols..(row + 1) * cols]
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|&(_, value)| !is_zero(value))
                    .map(|(col, value)| (value, col))
                    .collect(),
            )
        },
        _ => None,
    }
}

/// `terms`, in column order, as a one-row sparse matrix over `cols` columns.
fn row_matrix(terms: Vec<(f64, usize)>, cols: usize) -> Node {
    let n_terms = terms.len();
    let (data, indices) = terms.into_iter().unzip();
    let row = CsrData::try_new(vec![0, n_terms], indices, data, Shape::matrix(1, cols))
        .expect("a matrix row's columns lie inside the matrix");
    Node::SparseMatrix(Box::new(row))
}

/// Whether indexing this node is just indexing each of its children.
///
/// Exhaustive so every new [`Node`] variant must be classified here.
const fn is_elementwise(node: &Node) -> bool {
    match node {
        Node::Add(..)
        | Node::Sub(..)
        | Node::Mul(..)
        | Node::Div(..)
        | Node::Pow(..)
        | Node::Minimum(..)
        | Node::Maximum(..)
        | Node::Modulo(..)
        | Node::Hypot(..)
        | Node::EqualHeaviside(..)
        | Node::NotEqualHeaviside(..)
        | Node::Equality(..)
        | Node::Neg(_)
        | Node::Abs(_)
        | Node::Sqrt(_)
        | Node::Exp(_)
        | Node::Log(_)
        | Node::Sin(_)
        | Node::Cos(_)
        | Node::Tanh(_)
        | Node::Sinh(_)
        | Node::Cosh(_)
        | Node::Arcsinh(_)
        | Node::Arctan(_)
        | Node::Erf(_)
        | Node::Sign(_)
        | Node::Floor(_)
        | Node::Ceiling(_)
        | Node::Conditional { .. } => true,
        // Handled by `descend`, or not indexable.
        Node::Scalar(_)
        | Node::Array(_)
        | Node::ZeroVector { .. }
        | Node::SparseMatrix(_)
        | Node::StateVector { .. }
        | Node::StateVectorDot { .. }
        | Node::InputParameter { .. }
        | Node::Time
        | Node::MatMul(..)
        | Node::Index { .. }
        | Node::Concat(_)
        | Node::Interpolant1DLinear { .. }
        | Node::Interpolant1DCubic { .. }
        | Node::InterpolantNd { .. }
        | Node::InterpolantNdPartial { .. }
        | Node::MaxReduce(_)
        | Node::MinReduce(_)
        | Node::ReduceArgSelect { .. }
        | Node::TangentStateVector { .. }
        | Node::TangentParameter { .. }
        | Node::Interpolant1DLinearDeriv { .. }
        | Node::Interpolant1DCubicDeriv { .. } => false,
    }
}

/// Both signed zeros, compared by bit pattern.
const fn is_zero(value: f64) -> bool {
    value.to_bits() == 0.0_f64.to_bits() || value.to_bits() == (-0.0_f64).to_bits()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::CompiledExpr;
    use crate::node::ArrayData;

    fn extracted_value(arena: &Arena, root: NodeId, row: usize, y: &[f64], p: &[f64]) -> f64 {
        let block = extract_scalar_rows(DagRef::new(arena, root), &[row]);
        let expr = CompiledExpr::new(block.dag.as_ref());
        let mut scratch = vec![0.0; expr.scratch_len()];
        expr.eval(&mut scratch, 0.5, y, &[], p)[0]
    }

    fn reference_value(arena: &Arena, root: NodeId, row: usize, y: &[f64], p: &[f64]) -> f64 {
        let expr = CompiledExpr::new(DagRef::new(arena, root));
        let mut scratch = vec![0.0; expr.scratch_len()];
        expr.eval(&mut scratch, 0.5, y, &[], p)[row]
    }

    fn assert_extracts_elementwise(arena: &Arena, root: NodeId, width: usize, y: &[f64]) {
        for row in 0..width {
            let extracted = extracted_value(arena, root, row, y, &[]);
            let reference = reference_value(arena, root, row, y, &[]);
            assert_eq!(
                extracted.to_bits(),
                reference.to_bits(),
                "row {row}: extracted {extracted} != reference {reference}"
            );
        }
    }

    fn states(arena: &mut Arena, n: usize) -> NodeId {
        arena.alloc(Node::StateVector { start: 0, end: n })
    }

    fn sample(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (i as f64).mul_add(0.37, 1.1).sin() + 1.5)
            .collect()
    }

    #[test]
    fn state_vector_rows_narrow_to_one_entry() {
        let mut arena = Arena::new();
        let y = states(&mut arena, 5);
        assert_extracts_elementwise(&arena, y, 5, &sample(5));
    }

    #[test]
    fn elementwise_rows_index_both_operands() {
        let mut arena = Arena::new();
        let y = states(&mut arena, 6);
        let scalar = arena.alloc(Node::Scalar(2.5));
        let product = arena.alloc(Node::Mul(y, y));
        let shifted = arena.alloc(Node::Add(product, scalar));
        let root = arena.alloc(Node::Tanh(shifted));
        assert_extracts_elementwise(&arena, root, 6, &sample(6));
    }

    #[test]
    fn broadcast_operand_is_not_indexed() {
        let mut arena = Arena::new();
        let y = states(&mut arena, 4);
        let numerator = arena.alloc(Node::Scalar(3.0));
        let root = arena.alloc(Node::Div(numerator, y));
        assert_extracts_elementwise(&arena, root, 4, &sample(4));
    }

    #[test]
    fn index_and_concat_rows_resolve_through_structure() {
        let mut arena = Arena::new();
        let y = states(&mut arena, 8);
        let tail = arena.alloc(Node::Index {
            child: y,
            start: 5,
            end: 8,
        });
        let head = arena.alloc(Node::Sin(y));
        let root = arena.alloc(Node::Concat(vec![head, tail]));
        assert_extracts_elementwise(&arena, root, 11, &sample(8));
    }

    #[test]
    fn literal_rows_extract_their_own_element() {
        let mut arena = Arena::new();
        let y = states(&mut arena, 3);
        let array = arena.alloc(Node::Array(Box::new(
            ArrayData::try_new(vec![1.5, -2.5, 4.0], Shape::vector(3)).expect("valid array"),
        )));
        let zeros = arena.alloc(Node::ZeroVector { len: 3 });
        let sum = arena.alloc(Node::Add(array, zeros));
        let root = arena.alloc(Node::Mul(sum, y));
        assert_extracts_elementwise(&arena, root, 3, &sample(3));
    }

    #[test]
    fn conditional_rows_index_branches_and_keep_the_selector() {
        let mut arena = Arena::new();
        let y = states(&mut arena, 4);
        let selector = arena.alloc(Node::Scalar(2.0));
        let first = arena.alloc(Node::Sin(y));
        let second = arena.alloc(Node::Cos(y));
        let root = arena.alloc(Node::Conditional {
            selector,
            branches: vec![first, second],
        });
        assert_extracts_elementwise(&arena, root, 4, &sample(4));
    }

    #[test]
    fn sparse_matmul_rows_expand_to_their_stored_entries() {
        let mut arena = Arena::new();
        let n = 6;
        let y = states(&mut arena, n);
        let squared = arena.alloc(Node::Mul(y, y));
        // Rows of 2, 1 and 0 entries.
        let matrix = arena.alloc(Node::SparseMatrix(Box::new(
            CsrData::try_new(
                vec![0, 2, 3, 3],
                vec![0, 3, 5],
                vec![2.0, -1.0, 1.0],
                Shape::matrix(3, n),
            )
            .expect("valid matrix"),
        )));
        let root = arena.alloc(Node::MatMul(matrix, squared));
        assert_extracts_elementwise(&arena, root, 3, &sample(n));
    }

    #[test]
    fn dense_matmul_rows_keep_only_their_non_zeros() {
        let mut arena = Arena::new();
        let n = 4;
        let y = states(&mut arena, n);
        let matrix = arena.alloc(Node::Array(Box::new(
            ArrayData::try_new(
                vec![
                    1.0, 0.0, 0.5, 0.0, //
                    0.0, 0.0, 0.0, 0.0, //
                    -1.0, 2.0, 0.0, 3.0,
                ],
                Shape::matrix(3, n),
            )
            .expect("valid matrix"),
        )));
        let root = arena.alloc(Node::MatMul(matrix, y));
        assert_extracts_elementwise(&arena, root, 3, &sample(n));
    }

    #[test]
    fn vector_input_parameters_index_by_element() {
        let mut arena = Arena::new();
        let y = states(&mut arena, 3);
        let parameter = arena.alloc(Node::InputParameter {
            name: "k".into(),
            index: 0,
            offset: 0,
            width: 3,
        });
        let root = arena.alloc(Node::Mul(parameter, y));
        let y_values = sample(3);
        let p = [0.25, -1.5, 2.0];
        for row in 0..3 {
            assert_eq!(
                extracted_value(&arena, root, row, &y_values, &p).to_bits(),
                reference_value(&arena, root, row, &y_values, &p).to_bits()
            );
        }
    }

    #[test]
    fn rows_of_an_interpolant_slice_one_shared_copy() {
        // Interpolants once declined, so their rows fell back to the colour sweep.
        let mut arena = Arena::new();
        let y = states(&mut arena, 4);
        let interpolated = arena.alloc(Node::Interpolant1DLinear {
            data: Box::new(
                crate::node::InterpolantData::try_new(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 4.0])
                    .expect("valid table"),
            ),
            child: y,
        });
        let root = arena.alloc(Node::Exp(interpolated));
        assert_extracts_elementwise(&arena, root, 4, &sample(4));

        let block = extract_scalar_rows(DagRef::new(&arena, root), &[3, 0, 2]);
        let is_table = |node: &&Node| matches!(node, Node::Interpolant1DLinear { .. });
        let copies = block.dag.arena().nodes().iter().filter(is_table).count();
        assert_eq!(copies, 1, "every row must slice the same copy of the table");
    }

    #[test]
    #[should_panic(expected = "row 3 is outside the root's 3 rows")]
    fn out_of_range_rows_are_rejected() {
        let mut arena = Arena::new();
        let y = states(&mut arena, 3);
        let _ = extract_scalar_rows(DagRef::new(&arena, y), &[3]);
    }

    #[test]
    fn a_dense_matmul_row_stays_one_vector_product() {
        // A scalar term per column made each row's backward pass walk the group's whole tape.
        let (n, width) = (16, 2);
        let mut arena = Arena::new();
        let y = states(&mut arena, n);
        let squared = arena.alloc(Node::Mul(y, y));
        let matrix = arena.alloc(Node::SparseMatrix(Box::new(
            CsrData::try_new(
                vec![0, n, 2 * n],
                (0..n).chain(0..n).collect(),
                (0..2 * n)
                    .map(|k| 0.25f64.mul_add(k as f64, -1.5))
                    .collect(),
                Shape::matrix(width, n),
            )
            .expect("valid matrix"),
        )));
        let root = arena.alloc(Node::MatMul(matrix, squared));
        assert_extracts_elementwise(&arena, root, width, &sample(n));

        // Two row matrices and their products over one shared `y * y`, then the `Concat`.
        let block = extract_scalar_rows(DagRef::new(&arena, root), &[0, 1]);
        assert_eq!(block.dag.arena().len(), 7);
    }

    #[test]
    fn scalar_blocks_extract_without_synthesis() {
        let mut arena = Arena::new();
        let n = 8;
        let y = states(&mut arena, n);
        let matrix = arena.alloc(Node::SparseMatrix(Box::new(
            CsrData::try_new(
                vec![0, n],
                (0..n).collect(),
                vec![1.0; n],
                Shape::matrix(1, n),
            )
            .expect("valid matrix"),
        )));
        let root = arena.alloc(Node::MatMul(matrix, y));
        // Matrix, state read, matmul and the wrapping `Concat`: nothing synthesised.
        let block = extract_scalar_rows(DagRef::new(&arena, root), &[0]);
        assert_eq!(block.dag.arena().len(), 4);
        assert_extracts_elementwise(&arena, root, 1, &sample(n));
    }

    #[test]
    fn rows_share_the_upstream_they_have_in_common() {
        let mut arena = Arena::new();
        let n = 64;
        let y = states(&mut arena, n);
        let squared = arena.alloc(Node::Mul(y, y));
        let matrix = arena.alloc(Node::SparseMatrix(Box::new(
            CsrData::try_new(
                vec![0, n, 2 * n],
                (0..n).chain(0..n).collect(),
                vec![1.0; 2 * n],
                Shape::matrix(2, n),
            )
            .expect("valid matrix"),
        )));
        let root = arena.alloc(Node::MatMul(matrix, squared));
        let one = extract_scalar_rows(DagRef::new(&arena, root), &[0]);
        let both = extract_scalar_rows(DagRef::new(&arena, root), &[0, 1]);
        assert!(
            both.dag.arena().len() < 2 * one.dag.arena().len(),
            "two rows took {} nodes against {} for one",
            both.dag.arena().len(),
            one.dag.arena().len()
        );
    }

    /// Deeper than a recursive walk survives on a 2 MiB test-thread stack.
    const DEEP: usize = 200_000;

    #[test]
    fn extracts_a_row_through_a_deep_chain() {
        // Cycles every way a push descends: elementwise, `Index`, `Concat` and a short matmul row.
        let n = 8;
        let mut arena = Arena::new();
        let y = states(&mut arena, n);
        let identity = arena.alloc(Node::SparseMatrix(Box::new(
            CsrData::try_new(
                (0..=n).collect(),
                (0..n).collect(),
                vec![1.0; n],
                Shape::matrix(n, n),
            )
            .expect("valid matrix"),
        )));
        let mut id = y;
        for level in 0..DEEP {
            id = arena.alloc(match level % 4 {
                0 => Node::Sin(id),
                1 => Node::Index {
                    child: id,
                    start: 0,
                    end: n,
                },
                2 => Node::Concat(vec![id]),
                _ => Node::MatMul(identity, id),
            });
        }

        let block = extract_scalar_rows(DagRef::new(&arena, id), &[1]);
        let extracted = block.dag.arena();
        let Node::Concat(ref elements) = *extracted.get(block.dag.root()) else {
            panic!("a block is a `Concat` of its rows")
        };
        let mut element = elements[0];
        for _ in 0..DEEP / 4 {
            let Node::Sin(child) = *extracted.get(element) else {
                panic!("only the sines survive the identity levels")
            };
            element = child;
        }
        assert!(matches!(
            extracted.get(element),
            Node::StateVector { start: 1, end: 2 }
        ));
    }

    #[test]
    fn block_elements_name_the_rows_they_hold() {
        let mut arena = Arena::new();
        let n = 6;
        let y = states(&mut arena, n);
        let root = arena.alloc(Node::Sin(y));
        let wanted = [4usize, 1, 3];
        let block = extract_scalar_rows(DagRef::new(&arena, root), &wanted);
        let y_values = sample(n);
        assert_eq!(
            block.rows, wanted,
            "the block must keep the requested order"
        );
        let expr = CompiledExpr::new(block.dag.as_ref());
        let mut scratch = vec![0.0; expr.scratch_len()];
        let got = expr.eval(&mut scratch, 0.5, &y_values, &[], &[]).to_vec();
        for (&value, &row) in got.iter().zip(&block.rows) {
            assert_eq!(
                value.to_bits(),
                reference_value(&arena, root, row, &y_values, &[]).to_bits()
            );
        }
    }
}
