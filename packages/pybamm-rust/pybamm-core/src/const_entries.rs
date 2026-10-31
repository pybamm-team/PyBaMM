//! Jacobian entries a compile pass proves independent of `(t, y, y_dot, inputs)`.
//!
//! Folds follow the tangent tape's operator order, so values match the sweep up to a zero's sign
//! wherever the sweep is finite; a non-finite literal factor can fold where the sweep gives NaN.

use std::cmp::Ordering;

use crate::accumulate::mul_acc;
use crate::arena::{Arena, NodeId, NodeMap};
use crate::ir::infer_sizes;
use crate::node::Node;
use crate::sparsity::SparsityPattern;

/// Coefficients per element beyond which the element is abandoned and swept.
///
/// Bounds compile time and memory on a near-dense linear model.
const MAX_CONSTANT_DEGREE: usize = 64;

/// One seed's coefficient in one element of the tangent expression.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Coeff {
    /// Seed index, which is the Jacobian column.
    seed: usize,
    /// Folded value, or `None` when the coefficient depends on the state.
    value: Option<f64>,
}

impl Coeff {
    const fn unknown(seed: usize) -> Self {
        Self { seed, value: None }
    }

    fn map(self, f: impl Fn(f64) -> f64) -> Self {
        Self {
            seed: self.seed,
            value: self.value.map(f),
        }
    }

    fn negated(self) -> Self {
        self.map(|v| -v)
    }
}

/// One element's resolved coefficients, ascending by seed.
#[derive(Clone, Copy, Debug)]
struct Element<'a> {
    coeffs: &'a [Coeff],
    /// False once an unmodelled node is reached; an absent seed then proves nothing.
    complete: bool,
}

impl Element<'_> {
    const SEED_FREE: Self = Self {
        coeffs: &[],
        complete: true,
    };
}

/// Per-element seed coefficients of one node, flattened CSR-style.
///
/// A single element broadcasts; an incomplete element carries no entries.
#[derive(Clone, Debug)]
struct Coeffs {
    /// Element start offsets into `entries`, length `n_elems + 1`.
    offsets: Vec<usize>,
    /// Coefficients of every element, ascending by seed within an element.
    entries: Vec<Coeff>,
    complete: Vec<bool>,
}

impl Coeffs {
    /// Room for `n_elems` elements holding `n_entries` coefficients between them.
    fn with_capacity(n_elems: usize, n_entries: usize) -> Self {
        let mut offsets = Vec::with_capacity(n_elems + 1);
        offsets.push(0);
        Self {
            offsets,
            entries: Vec::with_capacity(n_entries),
            complete: Vec::with_capacity(n_elems),
        }
    }

    fn unresolved(n: usize) -> Self {
        Self {
            offsets: vec![0; n + 1],
            entries: Vec::new(),
            complete: vec![false; n],
        }
    }

    /// Close one element, recording it incomplete when over the degree cap.
    fn push(&mut self, coeffs: &[Coeff], complete: bool) {
        self.entries.extend_from_slice(coeffs);
        self.close(complete);
    }

    /// Close the element appended to `entries` since the last one, as [`Self::push`] does.
    fn close(&mut self, complete: bool) {
        let start = self.offsets[self.offsets.len() - 1];
        let keep = complete && self.entries.len() - start <= MAX_CONSTANT_DEGREE;
        if !keep {
            self.entries.truncate(start);
        }
        self.complete.push(keep);
        self.offsets.push(self.entries.len());
    }

    fn push_element(&mut self, element: Element<'_>) {
        self.push(element.coeffs, element.complete);
    }

    const fn n_elems(&self) -> usize {
        self.complete.len()
    }

    fn elem(&self, i: usize) -> Element<'_> {
        let i = i.min(self.n_elems() - 1);
        Element {
            coeffs: &self.entries[self.offsets[i]..self.offsets[i + 1]],
            complete: self.complete[i],
        }
    }
}

/// A node's value where the pass folded it to literals; `Scalar` broadcasts.
#[derive(Clone, Debug)]
enum Literal {
    Scalar(f64),
    Vector(Vec<f64>),
}

impl Literal {
    fn at(&self, i: usize) -> f64 {
        match self {
            Self::Scalar(v) => *v,
            Self::Vector(v) => v[i.min(v.len() - 1)],
        }
    }

    fn negated(&self) -> Self {
        match self {
            Self::Scalar(v) => Self::Scalar(-v),
            Self::Vector(v) => Self::Vector(v.iter().map(|x| -x).collect()),
        }
    }
}

/// What the pass knows about one node's value.
#[derive(Debug)]
enum NodeInfo {
    /// Seed-free and not folded.
    Primal,
    /// Folds to a literal, which is what lets a `Mul`/`Div` scale a coefficient.
    Literal(Literal),
    Seeded(Coeffs),
}

impl NodeInfo {
    fn elem(&self, i: usize) -> Element<'_> {
        match self {
            Self::Seeded(coeffs) => coeffs.elem(i),
            Self::Primal | Self::Literal(_) => Element::SEED_FREE,
        }
    }

    const fn n_elems(&self) -> usize {
        match self {
            Self::Primal | Self::Literal(Literal::Scalar(_)) => 1,
            Self::Literal(Literal::Vector(v)) => v.len(),
            Self::Seeded(coeffs) => coeffs.n_elems(),
        }
    }

    const fn literal(&self) -> Option<&Literal> {
        match self {
            Self::Literal(literal) => Some(literal),
            Self::Primal | Self::Seeded(_) => None,
        }
    }

    const fn is_seeded(&self) -> bool {
        matches!(self, Self::Seeded(_))
    }

    /// Coefficients this node supplies across `n_elems` elements, broadcasting a single one.
    const fn entries_over(&self, n_elems: usize) -> usize {
        match self {
            Self::Seeded(coeffs) if coeffs.n_elems() == 1 => coeffs.entries.len() * n_elems,
            Self::Seeded(coeffs) => coeffs.entries.len(),
            Self::Primal | Self::Literal(_) => 0,
        }
    }
}

/// Merge buffers reused across nodes.
#[derive(Debug, Default)]
struct Scratch {
    out: Vec<Coeff>,
    acc: Vec<Coeff>,
    children: Vec<NodeId>,
}

/// Append `map_a(a) + map_b(b)` to `out`, a sorted merge over seed indices.
///
/// Each map used is exact (`x - y == x + (-y)`), so the fold still matches the tape.
fn merge(
    a: &[Coeff],
    b: &[Coeff],
    out: &mut Vec<Coeff>,
    map_a: impl Fn(Coeff) -> Coeff,
    map_b: impl Fn(Coeff) -> Coeff,
) {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].seed.cmp(&b[j].seed) {
            Ordering::Less => {
                out.push(map_a(a[i]));
                i += 1;
            },
            Ordering::Greater => {
                out.push(map_b(b[j]));
                j += 1;
            },
            Ordering::Equal => {
                let (left, right) = (map_a(a[i]), map_b(b[j]));
                out.push(Coeff {
                    seed: left.seed,
                    value: left.value.zip(right.value).map(|(x, y)| x + y),
                });
                i += 1;
                j += 1;
            },
        }
    }
    out.extend(a[i..].iter().copied().map(&map_a));
    out.extend(b[j..].iter().copied().map(&map_b));
}

/// Append `a + b`, or `a - b` when `subtract`, to `out`.
fn merge_sum(a: &[Coeff], b: &[Coeff], subtract: bool, out: &mut Vec<Coeff>) {
    merge(a, b, out, |c| c, |c| if subtract { c.negated() } else { c });
}

/// Append `acc + factor * coeffs`, the accumulation one `MatMul` row performs, to `out`.
///
/// A seed already in `acc` fuses its product as the tape's `mul_acc` does; a new one rounds alone.
fn merge_scaled(acc: &[Coeff], coeffs: &[Coeff], factor: f64, out: &mut Vec<Coeff>) {
    let (mut i, mut j) = (0, 0);
    while i < acc.len() && j < coeffs.len() {
        match acc[i].seed.cmp(&coeffs[j].seed) {
            Ordering::Less => {
                out.push(acc[i]);
                i += 1;
            },
            Ordering::Greater => {
                out.push(coeffs[j].map(|v| v * factor));
                j += 1;
            },
            Ordering::Equal => {
                out.push(Coeff {
                    seed: acc[i].seed,
                    value: acc[i]
                        .value
                        .zip(coeffs[j].value)
                        .map(|(sum, v)| mul_acc(sum, factor, v)),
                });
                i += 1;
                j += 1;
            },
        }
    }
    out.extend_from_slice(&acc[i..]);
    out.extend(coeffs[j..].iter().map(|c| c.map(|v| v * factor)));
}

/// Append the seeds of `a` and `b`, every coefficient unresolved, to `out`.
fn merge_unknown(a: &[Coeff], b: &[Coeff], out: &mut Vec<Coeff>) {
    let unknown = |c: Coeff| Coeff::unknown(c.seed);
    merge(a, b, out, unknown, unknown);
}

/// Union the seeds of `elements` into `acc`, every coefficient unresolved.
///
/// Returns false at the first incomplete element or once past the degree cap.
fn union_seeds<'a>(
    elements: impl Iterator<Item = Element<'a>>,
    acc: &mut Vec<Coeff>,
    out: &mut Vec<Coeff>,
) -> bool {
    acc.clear();
    for element in elements {
        if !element.complete {
            return false;
        }
        if element.coeffs.is_empty() {
            continue;
        }
        out.clear();
        merge_unknown(acc, element.coeffs, out);
        std::mem::swap(acc, out);
        if acc.len() > MAX_CONSTANT_DEGREE {
            return false;
        }
    }
    true
}

/// Classify every entry of `pattern` against a tangent expression.
///
/// Returns a per-CSR-entry mask of the ones a sweep must still produce, and
/// `(csr_idx, value)` for the rest. `tangent_root` must be the *simplified*
/// tangent, so a folded value follows the tape's operator order.
#[must_use]
pub fn classify_constant_entries(
    arena: &Arena,
    tangent_root: NodeId,
    pattern: &SparsityPattern,
) -> (Vec<bool>, Vec<(usize, f64)>) {
    let rows = classify_rows(arena, tangent_root, pattern.nrows());

    let mut varying = vec![false; pattern.nnz()];
    let mut constants = Vec::new();
    for row in 0..pattern.nrows() {
        let (row_start, row_end) = (pattern.indptr()[row], pattern.indptr()[row + 1]);
        let element = rows.elem(row);
        if !element.complete {
            varying[row_start..row_end].fill(true);
            continue;
        }
        let mut pos = 0;
        for (offset, &col) in pattern.indices()[row_start..row_end].iter().enumerate() {
            while pos < element.coeffs.len() && element.coeffs[pos].seed < col {
                pos += 1;
            }
            let csr_idx = row_start + offset;
            match element.coeffs.get(pos) {
                Some(coeff) if coeff.seed == col => match coeff.value {
                    Some(value) => constants.push((csr_idx, value)),
                    None => varying[csr_idx] = true,
                },
                // Absent from a complete element: exactly zero.
                _ => constants.push((csr_idx, 0.0)),
            }
        }
    }
    (varying, constants)
}

/// Seed coefficients of each of `n_rows` output rows, never broadcast.
fn classify_rows(arena: &Arena, tangent_root: NodeId, n_rows: usize) -> Coeffs {
    let order = arena.topological_order(tangent_root);
    let sizes = infer_sizes(arena, &order);
    let mut scratch = Scratch::default();
    let root = arena.fold_frontier(&order, |id, info| {
        classify_node(arena, id, &sizes, info, &mut scratch)
    });
    match root {
        NodeInfo::Seeded(coeffs) if coeffs.n_elems() == n_rows => coeffs,
        other => {
            let mut rows = Coeffs::with_capacity(n_rows, other.entries_over(n_rows));
            for row in 0..n_rows {
                rows.push_element(other.elem(row));
            }
            rows
        },
    }
}

fn child_info(info: &NodeMap<NodeInfo>, id: NodeId) -> &NodeInfo {
    info.get(id)
        .expect("children are classified before their parents")
}

const fn combined_elems(a: &NodeInfo, b: &NodeInfo, len: usize) -> usize {
    if a.n_elems() == 1 && b.n_elems() == 1 {
        1
    } else {
        len
    }
}

/// Fold an element-wise binary operator over two literal operands, else `Primal`.
fn fold_literals(a: &NodeInfo, b: &NodeInfo, len: usize, op: impl Fn(f64, f64) -> f64) -> NodeInfo {
    let (Some(left), Some(right)) = (a.literal(), b.literal()) else {
        return NodeInfo::Primal;
    };
    if let (Literal::Scalar(x), Literal::Scalar(y)) = (left, right) {
        return NodeInfo::Literal(Literal::Scalar(op(*x, *y)));
    }
    NodeInfo::Literal(Literal::Vector(
        (0..len).map(|i| op(left.at(i), right.at(i))).collect(),
    ))
}

/// Coefficients of `seeded` scaled by `factor` element-wise, or divided by it.
fn scale_seeded(seeded: &NodeInfo, factor: &NodeInfo, len: usize, divide: bool) -> NodeInfo {
    let n_elems = combined_elems(seeded, factor, len);
    match factor.literal() {
        Some(literal) if divide => map_coeffs(seeded, n_elems, |c, i| c.map(|v| v / literal.at(i))),
        Some(literal) => map_coeffs(seeded, n_elems, |c, i| c.map(|v| v * literal.at(i))),
        None => map_coeffs(seeded, n_elems, |c, _| Coeff::unknown(c.seed)),
    }
}

/// Every element's coefficients mapped through `f(coeff, element_index)`.
fn map_coeffs(seeded: &NodeInfo, n_elems: usize, f: impl Fn(Coeff, usize) -> Coeff) -> NodeInfo {
    let mut built = Coeffs::with_capacity(n_elems, seeded.entries_over(n_elems));
    for i in 0..n_elems {
        let element = seeded.elem(i);
        built
            .entries
            .extend(element.coeffs.iter().map(|&c| f(c, i)));
        built.close(element.complete);
    }
    NodeInfo::Seeded(built)
}

/// Coefficients a seed-wise union of `a` and `b` holds at least, so usually all it needs.
fn union_hint(a: &NodeInfo, b: &NodeInfo, n_elems: usize) -> usize {
    a.entries_over(n_elems).max(b.entries_over(n_elems))
}

/// Seeds of both operands, every coefficient unresolved.
fn unknown_pair(a: &NodeInfo, b: &NodeInfo, len: usize) -> NodeInfo {
    let n_elems = combined_elems(a, b, len);
    let mut built = Coeffs::with_capacity(n_elems, union_hint(a, b, n_elems));
    for i in 0..n_elems {
        let (left, right) = (a.elem(i), b.elem(i));
        merge_unknown(left.coeffs, right.coeffs, &mut built.entries);
        built.close(left.complete && right.complete);
    }
    NodeInfo::Seeded(built)
}

/// Whether any of `children` carries a seed.
fn any_seeded(children: &[NodeId], info: &NodeMap<NodeInfo>) -> bool {
    children.iter().any(|&c| child_info(info, c).is_seeded())
}

/// One scalar element carrying every seed any child element touches.
fn reduce_all(children: &[NodeId], info: &NodeMap<NodeInfo>, scratch: &mut Scratch) -> NodeInfo {
    if !any_seeded(children, info) {
        return NodeInfo::Primal;
    }
    let Scratch { out, acc, .. } = scratch;
    let complete = union_seeds(
        children
            .iter()
            .map(|&c| child_info(info, c))
            .flat_map(|child| (0..child.n_elems()).map(move |i| child.elem(i))),
        acc,
        out,
    );
    let mut built = Coeffs::with_capacity(1, acc.len());
    built.push(acc, complete);
    NodeInfo::Seeded(built)
}

/// Fallback for unmodelled node kinds: the union of the children's seeds, unresolved.
fn unknown_elementwise(
    node: &Node,
    sizes: &[usize],
    info: &NodeMap<NodeInfo>,
    len: usize,
    scratch: &mut Scratch,
) -> NodeInfo {
    scratch.children.clear();
    node.for_each_child(|child| scratch.children.push(child));
    let Scratch { out, acc, children } = scratch;
    if !any_seeded(children, info) {
        return NodeInfo::Primal;
    }
    let broadcast = children
        .iter()
        .all(|&c| child_info(info, c).n_elems() == 1 && sizes[c.index()] <= 1);
    let n_elems = if broadcast { 1 } else { len };
    let widest = children
        .iter()
        .map(|&c| child_info(info, c).entries_over(n_elems))
        .max();
    let mut built = Coeffs::with_capacity(n_elems, widest.unwrap_or(0));
    for i in 0..n_elems {
        let elements = children.iter().map(|&c| child_info(info, c).elem(i));
        let complete = union_seeds(elements, acc, out);
        built.push(acc, complete);
    }
    NodeInfo::Seeded(built)
}

/// Accumulate one matrix row in the tape's own column order, so folds match bit for bit.
fn matmul_row(
    columns: impl Iterator<Item = (usize, f64)>,
    vector: &NodeInfo,
    scratch: &mut Scratch,
) -> bool {
    let Scratch { out, acc, .. } = scratch;
    acc.clear();
    for (col, value) in columns {
        let element = vector.elem(col);
        // A non-finite entry makes the tape's `value * 0.0` NaN where the fold drops it.
        if !element.complete || !value.is_finite() {
            return false;
        }
        if element.coeffs.is_empty() {
            continue;
        }
        out.clear();
        merge_scaled(acc, element.coeffs, value, out);
        std::mem::swap(acc, out);
        if acc.len() > MAX_CONSTANT_DEGREE {
            return false;
        }
    }
    true
}

/// `Add`/`Sub`: coefficients merge seed-wise; two seed-free operands fold.
fn classify_add_sub(left: &NodeInfo, right: &NodeInfo, subtract: bool, len: usize) -> NodeInfo {
    if !left.is_seeded() && !right.is_seeded() {
        return fold_literals(
            left,
            right,
            len,
            |x, y| if subtract { x - y } else { x + y },
        );
    }
    let n_elems = combined_elems(left, right, len);
    let mut built = Coeffs::with_capacity(n_elems, union_hint(left, right, n_elems));
    for i in 0..n_elems {
        let (le, re) = (left.elem(i), right.elem(i));
        merge_sum(le.coeffs, re.coeffs, subtract, &mut built.entries);
        built.close(le.complete && re.complete);
    }
    NodeInfo::Seeded(built)
}

/// `MatMul` against a literal matrix; a non-literal matrix resolves nothing.
fn classify_matmul(
    arena: &Arena,
    matrix: NodeId,
    vector: &NodeInfo,
    len: usize,
    scratch: &mut Scratch,
) -> NodeInfo {
    if !vector.is_seeded() {
        return NodeInfo::Primal;
    }
    match arena.get(matrix) {
        Node::SparseMatrix(csr) => {
            let mut built = Coeffs::with_capacity(csr.shape.rows, csr.indices.len());
            for row in 0..csr.shape.rows {
                let span = csr.indptr[row]..csr.indptr[row + 1];
                let columns = csr.indices[span.clone()]
                    .iter()
                    .copied()
                    .zip(csr.data[span].iter().copied());
                let complete = matmul_row(columns, vector, scratch);
                built.push(&scratch.acc, complete);
            }
            NodeInfo::Seeded(built)
        },
        Node::Array(array) => {
            let (rows, cols) = (array.shape.rows, array.shape.cols);
            let mut built = Coeffs::with_capacity(rows, rows);
            for row in 0..rows {
                let columns = array.data[row * cols..(row + 1) * cols]
                    .iter()
                    .copied()
                    .enumerate();
                let complete = matmul_row(columns, vector, scratch);
                built.push(&scratch.acc, complete);
            }
            NodeInfo::Seeded(built)
        },
        _ => NodeInfo::Seeded(Coeffs::unresolved(len)),
    }
}

/// `Concat`: children's elements laid end to end, folding when all are literal.
fn classify_concat(
    children: &[NodeId],
    sizes: &[usize],
    info: &NodeMap<NodeInfo>,
    len: usize,
) -> NodeInfo {
    let infos = children.iter().map(|&c| child_info(info, c));
    let widths = children.iter().map(|&c| sizes[c.index()]);

    if any_seeded(children, info) {
        let n_entries = infos
            .clone()
            .zip(widths.clone())
            .map(|(child, width)| child.entries_over(width))
            .sum();
        let mut built = Coeffs::with_capacity(len, n_entries);
        for (child, width) in infos.zip(widths) {
            for i in 0..width {
                built.push_element(child.elem(i));
            }
        }
        return NodeInfo::Seeded(built);
    }
    let Some(literals) = infos.map(NodeInfo::literal).collect::<Option<Vec<_>>>() else {
        return NodeInfo::Primal;
    };
    let mut values = Vec::with_capacity(len);
    for (literal, width) in literals.iter().zip(widths) {
        values.extend((0..width).map(|i| literal.at(i)));
    }
    NodeInfo::Literal(Literal::Vector(values))
}

/// `Index`: a window into the child's elements, or the child itself when it broadcasts.
fn classify_index(child: &NodeInfo, start: usize, end: usize) -> NodeInfo {
    match child {
        NodeInfo::Primal => NodeInfo::Primal,
        NodeInfo::Literal(Literal::Scalar(value)) => NodeInfo::Literal(Literal::Scalar(*value)),
        NodeInfo::Literal(Literal::Vector(values)) => {
            NodeInfo::Literal(Literal::Vector(values[start..end].to_vec()))
        },
        NodeInfo::Seeded(coeffs) if coeffs.n_elems() == 1 => NodeInfo::Seeded(coeffs.clone()),
        NodeInfo::Seeded(coeffs) => {
            let n_entries = coeffs.offsets[end] - coeffs.offsets[start];
            let mut built = Coeffs::with_capacity(end - start, n_entries);
            for i in start..end {
                built.push_element(coeffs.elem(i));
            }
            NodeInfo::Seeded(built)
        },
    }
}

fn classify_node(
    arena: &Arena,
    id: NodeId,
    sizes: &[usize],
    info: &NodeMap<NodeInfo>,
    scratch: &mut Scratch,
) -> NodeInfo {
    let len = sizes[id.index()];
    if len == 0 {
        return NodeInfo::Primal;
    }
    match arena.get(id) {
        Node::Scalar(value) => NodeInfo::Literal(Literal::Scalar(*value)),
        Node::ZeroVector { .. } => NodeInfo::Literal(Literal::Scalar(0.0)),
        Node::Array(array) => NodeInfo::Literal(Literal::Vector(array.data.clone())),
        Node::SparseMatrix(_)
        | Node::Time
        | Node::InputParameter { .. }
        | Node::StateVector { .. }
        | Node::StateVectorDot { .. } => NodeInfo::Primal,

        Node::TangentStateVector { start, end } => {
            let mut built = Coeffs::with_capacity(end - start, end - start);
            for seed in *start..*end {
                built.push(
                    &[Coeff {
                        seed,
                        value: Some(1.0),
                    }],
                    true,
                );
            }
            NodeInfo::Seeded(built)
        },

        // A parameter tangent seeds outside the state index space, so resolves nothing.
        Node::TangentParameter { .. } => NodeInfo::Seeded(Coeffs::unresolved(1)),

        Node::Neg(a) => match child_info(info, *a) {
            NodeInfo::Primal => NodeInfo::Primal,
            NodeInfo::Literal(literal) => NodeInfo::Literal(literal.negated()),
            seeded @ NodeInfo::Seeded(_) => {
                map_coeffs(seeded, seeded.n_elems(), |c, _| c.negated())
            },
        },

        Node::Add(a, b) => classify_add_sub(child_info(info, *a), child_info(info, *b), false, len),
        Node::Sub(a, b) => classify_add_sub(child_info(info, *a), child_info(info, *b), true, len),

        Node::Mul(a, b) => {
            let (left, right) = (child_info(info, *a), child_info(info, *b));
            // IEEE multiplication commutes, so one rule serves both operand orders.
            match (left.is_seeded(), right.is_seeded()) {
                (false, false) => fold_literals(left, right, len, |x, y| x * y),
                (false, true) => scale_seeded(right, left, len, false),
                (true, false) => scale_seeded(left, right, len, false),
                (true, true) => unknown_pair(left, right, len),
            }
        },

        Node::Div(a, b) => {
            let (left, right) = (child_info(info, *a), child_info(info, *b));
            if right.is_seeded() {
                unknown_pair(left, right, len)
            } else if left.is_seeded() {
                scale_seeded(left, right, len, true)
            } else {
                fold_literals(left, right, len, |x, y| x / y)
            }
        },

        Node::MatMul(matrix, vector) => {
            classify_matmul(arena, *matrix, child_info(info, *vector), len, scratch)
        },

        Node::Index { child, start, end } => classify_index(child_info(info, *child), *start, *end),

        Node::Concat(children) => classify_concat(children, sizes, info, len),

        Node::MaxReduce(a) | Node::MinReduce(a) => reduce_all(&[*a], info, scratch),
        Node::ReduceArgSelect { basis, picker, .. } => {
            reduce_all(&[*basis, *picker], info, scratch)
        },

        other => unknown_elementwise(other, sizes, info, len, scratch),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Dag;
    use crate::eval::{SplitCompiledExpr, TangentInputs};
    use crate::node::{ArrayData, CsrData, Shape};
    use crate::simplify::simplify_pipeline;
    use crate::sparsity::detect_sparsity_per_output;
    use crate::tangent::tangent_wrt_states;

    fn classify_states(
        arena: &Arena,
        root: NodeId,
        n_rows: usize,
        n_states: usize,
    ) -> (SparsityPattern, Vec<bool>, Vec<(usize, f64)>) {
        let mut tangent = Dag::new(arena.clone(), root);
        tangent.rewrite_in_place(tangent_wrt_states);
        let tangent = simplify_pipeline(tangent);
        let pattern = detect_sparsity_per_output(arena, root, n_rows, n_states);
        let (varying, entries) =
            classify_constant_entries(tangent.arena(), tangent.root(), &pattern);
        (pattern, varying, entries)
    }

    fn constant_map(
        pattern: &SparsityPattern,
        entries: &[(usize, f64)],
    ) -> std::collections::HashMap<(usize, usize), f64> {
        let rows = pattern.entry_rows();
        entries
            .iter()
            .map(|&(csr_idx, value)| ((rows[csr_idx], pattern.indices()[csr_idx]), value))
            .collect()
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn linear_row_folds_to_exact_coefficients() {
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let two = arena.alloc(Node::Scalar(2.0));
        let scaled = arena.alloc(Node::Mul(two, y0));
        let row0 = arena.alloc(Node::Sub(scaled, y1));
        let root = arena.alloc(Node::Concat(vec![row0, y1]));

        let (pattern, varying, entries) = classify_states(&arena, root, 2, 2);
        assert!(!varying.iter().any(|&v| v), "a linear model sweeps nothing");
        let constants = constant_map(&pattern, &entries);
        assert_eq!(constants[&(0, 0)], 2.0);
        assert_eq!(constants[&(0, 1)], -1.0);
        assert_eq!(constants[&(1, 1)], 1.0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn array_factor_scales_each_element() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let a = arena.alloc(Node::Array(Box::new(
            ArrayData::try_new(vec![1.5, -2.5, 4.0], Shape::vector(3)).expect("valid array"),
        )));
        let root = arena.alloc(Node::Mul(a, y));

        let (pattern, varying, entries) = classify_states(&arena, root, 3, 3);
        assert!(!varying.iter().any(|&v| v));
        let constants = constant_map(&pattern, &entries);
        for (row, expected) in [1.5, -2.5, 4.0].into_iter().enumerate() {
            assert_eq!(constants[&(row, row)], expected);
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn sparse_matmul_folds_matrix_entries() {
        let n = 4;
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: n });
        let (mut indptr, mut indices, mut data) = (vec![0usize], Vec::new(), Vec::new());
        for row in 0..n {
            for col in row.saturating_sub(1)..(row + 2).min(n) {
                indices.push(col);
                data.push((row * n + col) as f64 * 0.25);
            }
            indptr.push(indices.len());
        }
        let mut expected: Vec<(usize, usize, f64)> = Vec::new();
        for row in 0..n {
            expected.extend((indptr[row]..indptr[row + 1]).map(|k| (row, indices[k], data[k])));
        }
        let matrix = arena.alloc(Node::SparseMatrix(Box::new(
            CsrData::try_new(indptr, indices, data, Shape::matrix(n, n)).expect("valid matrix"),
        )));
        let root = arena.alloc(Node::MatMul(matrix, y));

        let (pattern, varying, entries) = classify_states(&arena, root, n, n);
        assert!(!varying.iter().any(|&v| v));
        let constants = constant_map(&pattern, &entries);
        for (row, col, value) in expected {
            assert_eq!(constants[&(row, col)], value, "entry ({row}, {col})");
        }
    }

    #[test]
    fn state_dependent_factor_stays_unproven() {
        // exp(y) is seed-free but not literal, so it cannot fold.
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let factor = arena.alloc(Node::Exp(y));
        let root = arena.alloc(Node::Mul(factor, y));

        let (_, varying, entries) = classify_states(&arena, root, 2, 2);
        assert!(varying.iter().all(|&v| v), "every entry must be swept");
        assert_eq!(entries, []);
    }

    #[test]
    fn unmodelled_node_marks_its_row_incomplete() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let reduced = arena.alloc(Node::MaxReduce(y));
        let root = arena.alloc(Node::Concat(vec![reduced, y]));

        let (pattern, varying, _) = classify_states(&arena, root, 4, 3);
        let row0 = pattern.row_span(0);
        assert!(
            varying[row0.clone()].iter().all(|&v| v),
            "the reduced row must sweep"
        );
        assert!(
            !varying[row0.end..].iter().any(|&v| v),
            "the linear rows must not"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn structurally_present_but_absent_entries_classify_as_zero() {
        // `0 * y0` keeps `y0` in the pattern but differentiates to nothing.
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let zero = arena.alloc(Node::Scalar(0.0));
        let vanishing = arena.alloc(Node::Mul(zero, y0));
        let root = arena.alloc(Node::Concat(vec![vanishing, y1]));

        let (pattern, varying, entries) = classify_states(&arena, root, 2, 2);
        assert_eq!(pattern.indices()[pattern.row_span(0)], [0]);
        assert!(!varying.iter().any(|&v| v));
        let constants = constant_map(&pattern, &entries);
        assert_eq!(constants[&(0, 0)], 0.0);
        assert_eq!(constants[&(1, 1)], 1.0);
    }

    /// Assert every folded entry of `root`'s Jacobian equals the tangent sweep bit for bit.
    fn assert_folds_match_the_tape(arena: &Arena, root: NodeId, y_values: &[f64]) {
        let n = y_values.len();
        let mut tangent = Dag::new(arena.clone(), root);
        tangent.rewrite_in_place(tangent_wrt_states);
        let tangent = simplify_pipeline(tangent);
        let pattern = detect_sparsity_per_output(arena, root, n, n);
        let (_, entries) = classify_constant_entries(tangent.arena(), tangent.root(), &pattern);

        let expr = SplitCompiledExpr::compile(tangent.as_ref());
        let mut scratch = vec![0.0; expr.scratch_len()];
        let table = constant_map(&pattern, &entries);
        assert!(!table.is_empty(), "nothing folded, so nothing was compared");
        let mut cache = expr.eval_primal(&mut scratch, 0.0, y_values, &[], &[]);
        for col in 0..n {
            let mut seed = vec![0.0; n];
            seed[col] = 1.0;
            let swept = cache
                .eval_tangent(&TangentInputs {
                    dy: Some(&seed),
                    dp: None,
                })
                .to_vec();
            for (row, &tape) in swept.iter().enumerate() {
                if let Some(&value) = table.get(&(row, col)) {
                    assert_eq!(
                        value.to_bits(),
                        tape.to_bits(),
                        "entry ({row}, {col}) folded to {value}, tape gave {tape}"
                    );
                }
            }
        }
    }

    #[test]
    fn folded_values_match_the_tape() {
        let n = 3;
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: n });
        let matrix = arena.alloc(Node::SparseMatrix(Box::new(
            CsrData::try_new(
                vec![0, 2, 3, 5],
                vec![0, 1, 1, 1, 2],
                vec![0.25, -1.75, 3.5, 0.125, -0.5],
                Shape::matrix(n, n),
            )
            .expect("valid matrix"),
        )));
        let linear = arena.alloc(Node::MatMul(matrix, y));
        let nonlinear = arena.alloc(Node::Sin(y));
        let root = arena.alloc(Node::Add(linear, nonlinear));
        assert_folds_match_the_tape(&arena, root, &[0.3, -1.2, 2.4]);
    }

    #[test]
    fn a_cancelling_row_folds_as_the_tape_accumulates_it() {
        // Row of `outer @ inner` is `-(1 + 2^-29) + (1 + 2^-30)^2`: 2^-60 fused, 0 unfused.
        let (a, b) = (1.0 + 2f64.powi(-29), 1.0 + 2f64.powi(-30));
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let inner = arena.alloc(Node::SparseMatrix(Box::new(
            CsrData::try_new(vec![0, 1, 2], vec![0, 0], vec![a, b], Shape::matrix(2, 1))
                .expect("valid matrix"),
        )));
        let outer = arena.alloc(Node::SparseMatrix(Box::new(
            CsrData::try_new(vec![0, 2], vec![0, 1], vec![-1.0, b], Shape::matrix(1, 2))
                .expect("valid matrix"),
        )));
        let stacked = arena.alloc(Node::MatMul(inner, y));
        let root = arena.alloc(Node::MatMul(outer, stacked));
        assert_folds_match_the_tape(&arena, root, &[0.7]);
    }

    #[test]
    fn wide_linear_rows_hit_the_degree_cap() {
        let n = MAX_CONSTANT_DEGREE + 8;
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: n });
        let ones = arena.alloc(Node::SparseMatrix(Box::new(
            CsrData::try_new(
                vec![0, n],
                (0..n).collect(),
                vec![1.0; n],
                Shape::matrix(1, n),
            )
            .expect("valid row matrix"),
        )));
        let root = arena.alloc(Node::MatMul(ones, y));

        let (_, varying, entries) = classify_states(&arena, root, 1, n);
        assert!(varying.iter().all(|&v| v));
        assert_eq!(entries, []);
    }
}
