//! Lift wide output rows of an expression out as standalone scalar expressions.
//!
//! A row wide enough to dominate the Jacobian colouring is filled by a reverse
//! pass over its own sub-expression instead ([`crate::adjoint`]). That needs the
//! row as a width-1 root, and a discretised model rarely offers one: its rhs is
//! a `Concat` of *blocks*, and a 2-D current collector puts one dense row per
//! collector node inside a single vector-valued block.
//!
//! So the row is synthesised rather than found. Extraction pushes an element
//! index down the DAG — `Concat` picks a child, `StateVector` narrows to one
//! entry, elementwise nodes index their operands, and `MatMul` against a
//! constant expands row `r` into its dot product. Nodes with no cheap indexed
//! form (interpolants, whose tables would be copied per element) stop the walk,
//! and the caller falls back to colouring the row like any other.
//!
//! Rows are extracted in groups, because they overlap almost entirely: a dense
//! operator makes every row of a block read every lane of the same upstream
//! expression. On a 12x12 pouch cell one row alone is 50k instructions of which
//! 49k is that shared upstream, so a tape per row costs 7.8M instructions where
//! four rows per tape costs 2.4M and one tape for all 144 costs 283k.
//!
//! Grouping is not free in the other direction: every row walks its group's
//! whole tape backwards, so one assembly costs 66 ms at one row per tape, 58 ms
//! at four, and 215 ms at all 144 -- worse than not splitting at all.
//! [`crate::jacobian::ROWS_PER_TAPE`] picks the knee.

use rustc_hash::FxHashMap;

use crate::arena::{Arena, NodeId, NodeMap};
use crate::node::Node;
use crate::simplify::{dce, node_len};

/// A group of output rows lifted out as one expression owning its own arena.
///
/// Element `i` of [`root`](Self::root) is element `i` of [`rows`](Self::rows).
#[derive(Debug)]
pub struct ScalarRowBlock {
    /// Parent output rows this block holds, in element order.
    pub rows: Vec<usize>,
    /// Arena holding only the nodes those rows reach.
    pub arena: Arena,
    /// `Concat` of the extracted rows, one element each.
    pub root: NodeId,
}

/// Extract `rows` of `root` as one block: a `Concat` whose element `i` is
/// `rows[i]`.
///
/// The rows share one tape and one forward pass. Each row's backward pass walks
/// only the shared upstream plus its own cone, so holding every row on one tape
/// costs nothing per row and carries the shared expression exactly once.
///
/// `None` means the caller must colour every row as before: extraction is
/// all-or-nothing, since the colouring that adopted `rows` assumed all of them
/// leave the sweep.
pub fn extract_scalar_rows(arena: &Arena, root: NodeId, rows: &[usize]) -> Option<ScalarRowBlock> {
    debug_assert!(!rows.is_empty(), "the caller declines before extracting");
    let mut extractor = Extractor::new(arena);
    let mut extracted = Vec::with_capacity(rows.len());
    for &row in rows {
        extracted.push(extractor.push(root, row)?);
    }
    let concat = extractor.work.alloc(Node::Concat(extracted));
    // Compacted because tape construction allocates per node of the arena it is
    // handed, and the extraction arena still carries the whole model.
    let (arena, root) = dce(&extractor.work, concat);
    Some(ScalarRowBlock {
        rows: rows.to_vec(),
        arena,
        root,
    })
}

/// Index-push walker over one primal DAG.
struct Extractor<'a> {
    /// The graph being indexed. Its ids stay valid in `work`, which starts as
    /// its clone, so widths and children can be read straight from here.
    source: &'a Arena,
    /// `source` plus every synthesised scalar node.
    work: Arena,
    /// `node_len` memo over `source`; synthesised nodes are all width 1 and are
    /// never queried.
    widths: NodeMap<usize>,
    /// Element `index` of node `id`, shared across every requested row.
    cache: FxHashMap<(NodeId, usize), NodeId>,
    /// Synthesised `Scalar` nodes by bit pattern. A discretisation matrix has a
    /// handful of distinct coefficients across hundreds of thousands of entries.
    scalars: FxHashMap<u64, NodeId>,
}

impl<'a> Extractor<'a> {
    fn new(source: &'a Arena) -> Self {
        Self {
            source,
            work: source.clone(),
            widths: NodeMap::new(source.len()),
            cache: FxHashMap::default(),
            scalars: FxHashMap::default(),
        }
    }

    fn alloc(&mut self, node: Node) -> NodeId {
        self.work.alloc(node)
    }

    fn width(&mut self, id: NodeId) -> usize {
        node_len(self.source, id, &mut self.widths)
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
    fn push(&mut self, id: NodeId, index: usize) -> Option<NodeId> {
        let width = self.width(id);
        if index >= width {
            return None;
        }
        // Reused whole: a width-1 node already is its own element 0.
        if width == 1 {
            return Some(id);
        }
        if let Some(&hit) = self.cache.get(&(id, index)) {
            return Some(hit);
        }
        let pushed = self.push_uncached(id, index)?;
        self.cache.insert((id, index), pushed);
        Some(pushed)
    }

    fn push_uncached(&mut self, id: NodeId, index: usize) -> Option<NodeId> {
        // Copied out so the match borrows the source graph, not `self`.
        let source = self.source;
        match source.get(id) {
            Node::Concat(children) => {
                let mut offset = 0;
                for &child in children {
                    let len = self.width(child);
                    if index < offset + len {
                        return self.push(child, index - offset);
                    }
                    offset += len;
                }
                None
            },
            Node::Index { child, start, .. } => self.push(*child, start + index),
            Node::StateVector { start, .. } => {
                let at = start + index;
                Some(self.alloc(Node::StateVector {
                    start: at,
                    end: at + 1,
                }))
            },
            Node::StateVectorDot { start, .. } => {
                let at = start + index;
                Some(self.alloc(Node::StateVectorDot {
                    start: at,
                    end: at + 1,
                }))
            },
            Node::Array(array) => {
                let value = array.data()[index];
                Some(self.scalar(value))
            },
            Node::ZeroVector { .. } => Some(self.scalar(0.0)),
            // A vector input has no narrower load, but slicing a leaf costs
            // nothing: the tape reads the packed values either way.
            Node::InputParameter { .. } => Some(self.alloc(Node::Index {
                child: id,
                start: index,
                end: index + 1,
            })),
            Node::MatMul(matrix, vector) => self.push_matmul_row(*matrix, *vector, index),
            node if is_elementwise(node) => self.push_elementwise(node, index),
            // Interpolants are the notable omission: `map_children` clones their
            // table, so indexing one would copy it per element and per tape.
            _ => None,
        }
    }

    /// Rebuild an elementwise node on indexed operands, broadcasting the
    /// width-1 ones unchanged.
    fn push_elementwise(&mut self, node: &Node, index: usize) -> Option<NodeId> {
        let mut children = Vec::new();
        node.for_each_child(|child| children.push(child));
        let mut pushed = Vec::with_capacity(children.len());
        for &child in &children {
            let at = if self.width(child) == 1 { 0 } else { index };
            pushed.push(self.push(child, at)?);
        }
        // `map_children` visits children in `for_each_child` order, so the
        // mapped ids line up one for one.
        let mut mapped = pushed.into_iter();
        let indexed = node.map_children(|_| mapped.next().expect("one push per child"));
        Some(self.alloc(indexed))
    }

    /// Row `row` of a constant matrix contracted against `vector`.
    ///
    /// Structural zeros are skipped, and so are stored zeros of a dense
    /// operand: this is how a 256-wide block stops being one term per column.
    fn push_matmul_row(&mut self, matrix: NodeId, vector: NodeId, row: usize) -> Option<NodeId> {
        let terms: Vec<(f64, usize)> = match self.source.get(matrix) {
            Node::SparseMatrix(csr) => {
                let (start, end) = (csr.indptr()[row], csr.indptr()[row + 1]);
                csr.data()[start..end]
                    .iter()
                    .copied()
                    .zip(csr.indices()[start..end].iter().copied())
                    .collect()
            },
            Node::Array(array) => {
                let cols = array.shape().cols;
                array.data()[row * cols..(row + 1) * cols]
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|&(_, value)| !is_zero(value))
                    .map(|(col, value)| (value, col))
                    .collect()
            },
            // Only a constant left operand has rows to index; anything else is
            // rejected before lowering anyway.
            _ => return None,
        };

        let mut sum: Option<NodeId> = None;
        for (coefficient, col) in terms {
            let element = self.push(vector, col)?;
            let term = self.scaled(coefficient, element);
            // Left-to-right, matching the accumulation order the vector matmul
            // evaluates in.
            sum = Some(sum.map_or(term, |accumulated| self.alloc(Node::Add(accumulated, term))));
        }
        Some(sum.unwrap_or_else(|| self.scalar(0.0)))
    }

    /// `coefficient * element`, folding the unit coefficients a discretisation
    /// matrix is mostly made of. Both foldings are exact.
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

/// Whether indexing this node is just indexing each of its children.
///
/// `Conditional` qualifies because its selector is width 1 and so indexes to
/// itself, leaving only the branches to narrow.
///
/// Exhaustive on purpose: a new [`Node`] variant must be classified here, or it
/// would silently stop extraction and decline the split with nothing to show.
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
        // Handled by `push_uncached` ahead of this check, or genuinely not
        // indexable: reductions collapse width, interpolants would copy their
        // table per element, and a matmul needs its row expanded.
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

/// Both signed zeros. Spelled through the bit pattern because a dropped term
/// must be one the matmul could not have contributed to.
const fn is_zero(value: f64) -> bool {
    value.to_bits() == 0.0_f64.to_bits() || value.to_bits() == (-0.0_f64).to_bits()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::CompiledExpr;
    use crate::node::{ArrayData, CsrData, Shape};

    /// Element `row` of `root`, evaluated through an extracted scalar tape.
    fn extracted_value(arena: &Arena, root: NodeId, row: usize, y: &[f64], p: &[f64]) -> f64 {
        let block = extract_scalar_rows(arena, root, &[row])
            .unwrap_or_else(|| panic!("row {row} must extract"));
        let expr = CompiledExpr::new(&block.arena, block.root);
        let mut scratch = vec![0.0; expr.scratch_len()];
        expr.eval(&mut scratch, 0.5, y, &[], p)[0]
    }

    /// Element `row` of `root` evaluated the ordinary way, as the reference.
    fn reference_value(arena: &Arena, root: NodeId, row: usize, y: &[f64], p: &[f64]) -> f64 {
        let expr = CompiledExpr::new(arena, root);
        let mut scratch = vec![0.0; expr.scratch_len()];
        expr.eval(&mut scratch, 0.5, y, &[], p)[row]
    }

    /// Every element of `root` extracts to the value the whole expression puts
    /// at that position.
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
        // A vector op, a broadcast scalar operand, and a unary on top.
        let product = arena.alloc(Node::Mul(y, y));
        let shifted = arena.alloc(Node::Add(product, scalar));
        let root = arena.alloc(Node::Tanh(shifted));
        assert_extracts_elementwise(&arena, root, 6, &sample(6));
    }

    #[test]
    fn broadcast_operand_is_not_indexed() {
        // `Div` with the vector on the right: the width-1 side must stay whole
        // rather than be indexed alongside it.
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
        // Rows of 2, 1 and 0 entries, so the empty-row and single-term paths
        // are both exercised alongside the ordinary sum.
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
    fn dense_matmul_rows_expand_row_major_and_skip_zeros() {
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
    fn unsupported_nodes_stop_the_walk() {
        // An interpolant is deliberately not indexable: its table would be
        // copied per element.
        let mut arena = Arena::new();
        let y = states(&mut arena, 4);
        let interpolated = arena.alloc(Node::Interpolant1DLinear {
            data: Box::new(
                crate::node::InterpolantData::try_new(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 4.0])
                    .expect("valid table"),
            ),
            child: y,
        });
        assert!(extract_scalar_rows(&arena, interpolated, &[1]).is_none());
    }

    #[test]
    fn out_of_range_rows_are_rejected() {
        let mut arena = Arena::new();
        let y = states(&mut arena, 3);
        assert!(extract_scalar_rows(&arena, y, &[3]).is_none());
    }

    /// A row that is already its own width-1 block is returned as it stands,
    /// so the pre-existing scalar path costs no synthesis.
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
        // Matrix, state read, matmul, and the one-element Concat wrapping them:
        // the row as it already stood, nothing synthesised.
        let block = extract_scalar_rows(&arena, root, &[0]).expect("no synthesis needed");
        assert_eq!(block.arena.len(), 4);
        assert_extracts_elementwise(&arena, root, 1, &sample(n));
    }

    /// Rows of one block read the same upstream, and the shared tape must hold
    /// it once. Two rows over the same `n` squares cost one set of squares.
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
        let one = extract_scalar_rows(&arena, root, &[0]).expect("one row");
        let both = extract_scalar_rows(&arena, root, &[0, 1]).expect("both rows");
        // Only the second row's own sum tree is added, not another copy of the
        // n squares and n state reads.
        assert!(
            both.arena.len() < 2 * one.arena.len(),
            "two rows took {} nodes against {} for one",
            both.arena.len(),
            one.arena.len()
        );
    }

    /// Every block element is the row its `rows` entry names, whatever order
    /// the rows were requested in.
    #[test]
    fn block_elements_name_the_rows_they_hold() {
        let mut arena = Arena::new();
        let n = 6;
        let y = states(&mut arena, n);
        let root = arena.alloc(Node::Sin(y));
        let wanted = [4usize, 1, 3];
        let block = extract_scalar_rows(&arena, root, &wanted).expect("rows");
        let y_values = sample(n);
        assert_eq!(
            block.rows, wanted,
            "the block must keep the requested order"
        );
        let expr = CompiledExpr::new(&block.arena, block.root);
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
