//! Jacobian entries a compile pass proves independent of `(t, y, y_dot, inputs)`.
//!
//! The tangent expression `T(y, s) = J(y) · s` is linear in the seed `s`, so the
//! coefficient of every seed in every output element falls out of one bottom-up
//! pass over the tape. A coefficient that folds to a literal is a Jacobian entry
//! no sweep has to produce: assembly writes it from a table, and the column
//! coloring only has to keep the remaining entries exact.
//!
//! The pass is conservative in one direction only. A coefficient it cannot
//! resolve is reported as varying, never as constant, and a node kind it does
//! not model marks the elements it reaches *incomplete*, after which a seed's
//! absence proves nothing and every entry of the affected row is swept.
//!
//! Classifying here — on the simplified tangent tape rather than on the primal
//! graph — is what makes a folded value bit-identical to the one the sweep
//! would have produced: the fold walks the same operators in the same order, so
//! a `MatMul` accumulates over its columns in the same sequence. The single
//! difference it allows itself is the sign of a zero, which dropping an exactly
//! zero term can flip and which `simplify` already documents as acceptable.
//!
//! One boundary sits outside that guarantee. Where the tape overflows, a term
//! the fold drops as exactly zero evaluates to `inf * 0.0 = NaN` and poisons
//! the sweep, so a folded entry can be finite where the swept one is not. The
//! rest of that row still comes out `NaN` and the step is rejected either way,
//! which is why nothing here tries to reproduce the poisoning.

use std::cmp::Ordering;

use crate::arena::{Arena, NodeId};
use crate::ir::infer_sizes;
use crate::node::Node;
use crate::sparsity::SparsityPattern;

/// Coefficients per element beyond which an element is abandoned rather than
/// grown. Bounds compile time and memory on a near-dense, fully linear model,
/// where the propagation would otherwise compute the whole Jacobian
/// symbolically; a row wider than this is swept, exactly as it is today.
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

    /// Same seed, value transformed wherever it is resolved.
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
    /// False once a node kind the pass cannot model has been reached, after
    /// which an absent seed proves nothing.
    complete: bool,
}

impl Element<'_> {
    /// An element no seed reaches: every coefficient is exactly zero.
    const SEED_FREE: Self = Self {
        coeffs: &[],
        complete: true,
    };
}

/// Per-element seed coefficients of one node, flattened CSR-style.
///
/// A single element means "broadcast": every element of the node reads it. An
/// element flagged incomplete carries no entries, since they could no longer
/// prove anything about the seeds they omit.
#[derive(Clone, Debug)]
struct Coeffs {
    /// Element start offsets into `entries`, length `n_elems + 1`.
    offsets: Vec<usize>,
    /// Coefficients of every element, ascending by seed within an element.
    entries: Vec<Coeff>,
    /// Per element, parallel to the `offsets` windows.
    complete: Vec<bool>,
}

impl Coeffs {
    /// Empty, sized for `n_elems` pushes. At least one coefficient per element
    /// is the common shape, so `entries` is reserved for that.
    fn with_capacity(n_elems: usize) -> Self {
        let mut offsets = Vec::with_capacity(n_elems + 1);
        offsets.push(0);
        Self {
            offsets,
            entries: Vec::with_capacity(n_elems),
            complete: Vec::with_capacity(n_elems),
        }
    }

    /// `n` elements the pass could not model at all.
    fn unresolved(n: usize) -> Self {
        Self {
            offsets: vec![0; n + 1],
            entries: Vec::new(),
            complete: vec![false; n],
        }
    }

    /// Close one element. Over the cap it is recorded incomplete and its
    /// entries dropped, which is what bounds the pass on a dense linear model.
    fn push(&mut self, coeffs: &[Coeff], complete: bool) {
        let keep = complete && coeffs.len() <= MAX_CONSTANT_DEGREE;
        if keep {
            self.entries.extend_from_slice(coeffs);
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
    /// Free of every seed and not folded: an ordinary primal subexpression.
    Primal,
    /// Folds to a literal, which is what lets a `Mul`/`Div` scale a coefficient.
    Literal(Literal),
    /// Carries seeds; the general case.
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
}

/// Merge buffers reused across nodes, so the pass allocates per node rather
/// than per element.
#[derive(Debug, Default)]
struct Scratch {
    out: Vec<Coeff>,
    acc: Vec<Coeff>,
    children: Vec<NodeId>,
}

/// `out = map_a(a) + map_b(b)`, a sorted merge over seed indices. Either side
/// unresolved leaves that seed's sum unresolved.
///
/// The three uses -- sum, difference, scaled accumulation -- differ only in
/// those two maps, and each is exact: `x - y == x + (-y)` and `factor * y ==
/// y * factor` bit for bit, so the fold still matches the tape.
fn merge(
    a: &[Coeff],
    b: &[Coeff],
    out: &mut Vec<Coeff>,
    map_a: impl Fn(Coeff) -> Coeff,
    map_b: impl Fn(Coeff) -> Coeff,
) {
    out.clear();
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

/// `out = a + b`, or `a - b` when `subtract`.
fn merge_sum(a: &[Coeff], b: &[Coeff], subtract: bool, out: &mut Vec<Coeff>) {
    merge(a, b, out, |c| c, |c| if subtract { c.negated() } else { c });
}

/// `out = acc + factor * coeffs`, the accumulation one `MatMul` row performs.
fn merge_scaled(acc: &[Coeff], coeffs: &[Coeff], factor: f64, out: &mut Vec<Coeff>) {
    merge(acc, coeffs, out, |c| c, |c| c.map(|v| v * factor));
}

/// `out` = the seeds of `a` and `b`, every coefficient unresolved: all a node
/// kind the pass cannot model can still say is which columns it touches.
fn merge_unknown(a: &[Coeff], b: &[Coeff], out: &mut Vec<Coeff>) {
    let unknown = |c: Coeff| Coeff::unknown(c.seed);
    merge(a, b, out, unknown, unknown);
}

/// Union the seeds of `elements` into `acc`, every coefficient unresolved.
///
/// Returns false at the first incomplete element or once the degree cap is
/// passed, after which the accumulated seeds prove nothing about the ones they
/// omit. Seed-free elements contribute nothing and are skipped.
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
        merge_unknown(acc, element.coeffs, out);
        std::mem::swap(acc, out);
        if acc.len() > MAX_CONSTANT_DEGREE {
            return false;
        }
    }
    true
}

/// Classify every entry of `pattern` against a tangent expression: a per-CSR
/// -entry mask of the ones a sweep must still produce, and `(csr_idx, value)`
/// for the rest.
///
/// The tangent expression's linearity in the seed makes the coefficient of each
/// seed recoverable in one bottom-up pass; a coefficient that folds to a literal
/// is an entry no sweep has to produce. `tangent_root` must be the *simplified*
/// tangent expression, so that a folded value follows the same operator order
/// the tape executes.
///
/// A pattern entry the tangent proves absent is a constant zero rather than an
/// omission, so the two halves together cover every entry of `pattern`.
#[must_use]
pub fn classify_constant_entries(
    arena: &Arena,
    tangent_root: NodeId,
    pattern: &SparsityPattern,
) -> (Vec<bool>, Vec<(usize, f64)>) {
    let rows = classify_rows(arena, tangent_root, pattern.nrows);

    let mut varying = vec![false; pattern.nnz()];
    let mut constants = Vec::new();
    for row in 0..pattern.nrows {
        let (row_start, row_end) = (pattern.indptr[row], pattern.indptr[row + 1]);
        let element = rows.elem(row);
        if !element.complete {
            varying[row_start..row_end].fill(true);
            continue;
        }
        let mut pos = 0;
        for (offset, &col) in pattern.indices[row_start..row_end].iter().enumerate() {
            while pos < element.coeffs.len() && element.coeffs[pos].seed < col {
                pos += 1;
            }
            let csr_idx = row_start + offset;
            match element.coeffs.get(pos) {
                Some(coeff) if coeff.seed == col => match coeff.value {
                    Some(value) => constants.push((csr_idx, value)),
                    None => varying[csr_idx] = true,
                },
                // Absent from a complete element: exactly zero, forever.
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

    // Freeing at the last read keeps the live set a frontier over the
    // tape rather than the whole graph.
    let mut remaining = vec![0u32; arena.len()];
    for &id in &order {
        arena.get(id).for_each_child(|child| {
            remaining[child.index()] += 1;
        });
    }

    let mut info: Vec<Option<NodeInfo>> = (0..arena.len()).map(|_| None).collect();
    let mut scratch = Scratch::default();
    for &id in &order {
        let computed = classify_node(arena, id, &sizes, &info, &mut scratch);
        arena.get(id).for_each_child(|child| {
            remaining[child.index()] -= 1;
            if remaining[child.index()] == 0 {
                info[child.index()] = None;
            }
        });
        info[id.index()] = Some(computed);
    }

    let root = info[tangent_root.index()]
        .take()
        .expect("root is classified last");
    match root {
        // Already one entry per row, and every element in it has been through
        // the same cap check, so re-pushing it would only duplicate the table.
        NodeInfo::Seeded(coeffs) if coeffs.n_elems() == n_rows => coeffs,
        other => {
            let mut rows = Coeffs::with_capacity(n_rows);
            for row in 0..n_rows {
                rows.push_element(other.elem(row));
            }
            rows
        },
    }
}

fn child_info(info: &[Option<NodeInfo>], id: NodeId) -> &NodeInfo {
    info[id.index()]
        .as_ref()
        .expect("children are classified before their parents")
}

/// Elements to materialise: one when every operand broadcasts, else the node's
/// full width.
const fn combined_elems(a: &NodeInfo, b: &NodeInfo, len: usize) -> usize {
    if a.n_elems() == 1 && b.n_elems() == 1 {
        1
    } else {
        len
    }
}

/// Fold an element-wise binary operator over two literal operands. Anything
/// else is seed-free but unfoldable, which still blocks a `Mul` from scaling.
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
/// A non-literal factor leaves every coefficient unresolved: the tangent still
/// reaches those columns, but their values move with the state.
fn scale_seeded(
    seeded: &NodeInfo,
    factor: &NodeInfo,
    len: usize,
    divide: bool,
    scratch: &mut Scratch,
) -> NodeInfo {
    let n_elems = combined_elems(seeded, factor, len);
    match factor.literal() {
        Some(literal) if divide => map_coeffs(seeded, n_elems, scratch, |c, i| {
            c.map(|v| v / literal.at(i))
        }),
        Some(literal) => map_coeffs(seeded, n_elems, scratch, |c, i| {
            c.map(|v| v * literal.at(i))
        }),
        None => map_coeffs(seeded, n_elems, scratch, |c, _| Coeff::unknown(c.seed)),
    }
}

/// Every element's coefficients mapped through `f`, completeness preserved.
/// `f` is given the element index, which is what lets a per-element literal
/// factor scale each row by its own value.
fn map_coeffs(
    seeded: &NodeInfo,
    n_elems: usize,
    scratch: &mut Scratch,
    f: impl Fn(Coeff, usize) -> Coeff,
) -> NodeInfo {
    let mut built = Coeffs::with_capacity(n_elems);
    for i in 0..n_elems {
        let element = seeded.elem(i);
        scratch.out.clear();
        scratch.out.extend(element.coeffs.iter().map(|&c| f(c, i)));
        built.push(&scratch.out, element.complete);
    }
    NodeInfo::Seeded(built)
}

/// Seeds of both operands, every coefficient unresolved. The shape a product
/// or quotient of two seeded operands takes; it cannot arise from a tangent
/// transform, which is linear in the seed, but costs nothing to survive.
fn unknown_pair(a: &NodeInfo, b: &NodeInfo, len: usize, scratch: &mut Scratch) -> NodeInfo {
    let n_elems = combined_elems(a, b, len);
    let mut built = Coeffs::with_capacity(n_elems);
    for i in 0..n_elems {
        let (left, right) = (a.elem(i), b.elem(i));
        merge_unknown(left.coeffs, right.coeffs, &mut scratch.out);
        built.push(&scratch.out, left.complete && right.complete);
    }
    NodeInfo::Seeded(built)
}

/// One scalar element carrying every seed any child element touches: what a
/// reduction over an unknown position can promise.
fn reduce_all(children: &[NodeId], info: &[Option<NodeInfo>], scratch: &mut Scratch) -> NodeInfo {
    let infos: Vec<&NodeInfo> = children.iter().map(|&c| child_info(info, c)).collect();
    if !infos.iter().any(|child| child.is_seeded()) {
        return NodeInfo::Primal;
    }
    let Scratch { out, acc, .. } = scratch;
    let complete = union_seeds(
        infos
            .iter()
            .flat_map(|child| (0..child.n_elems()).map(move |i| child.elem(i))),
        acc,
        out,
    );
    let mut built = Coeffs::with_capacity(1);
    built.push(acc, complete);
    NodeInfo::Seeded(built)
}

/// Element-wise fallback for every node kind the pass does not model: the union
/// of the children's seeds, all unresolved. Sound because a node's value can
/// only depend on seeds its children depend on.
fn unknown_elementwise(
    node: &Node,
    sizes: &[usize],
    info: &[Option<NodeInfo>],
    len: usize,
    scratch: &mut Scratch,
) -> NodeInfo {
    scratch.children.clear();
    node.for_each_child(|child| scratch.children.push(child));
    let Scratch { out, acc, children } = scratch;
    // Resolved once per node, not once per element.
    let infos: Vec<(&NodeInfo, usize)> = children
        .iter()
        .map(|&c| (child_info(info, c), sizes[c.index()]))
        .collect();
    if !infos.iter().any(|(child, _)| child.is_seeded()) {
        return NodeInfo::Primal;
    }
    let broadcast = infos
        .iter()
        .all(|&(child, size)| child.n_elems() == 1 && size <= 1);
    let n_elems = if broadcast { 1 } else { len };
    let mut built = Coeffs::with_capacity(n_elems);
    for i in 0..n_elems {
        let complete = union_seeds(infos.iter().map(|(child, _)| child.elem(i)), acc, out);
        built.push(acc, complete);
    }
    NodeInfo::Seeded(built)
}

/// Accumulate one matrix row's contributions in the tape's own column order,
/// so a folded value matches the sweep bit for bit.
fn matmul_row(
    columns: impl Iterator<Item = (usize, f64)>,
    vector: &NodeInfo,
    scratch: &mut Scratch,
) -> bool {
    let Scratch { out, acc, .. } = scratch;
    acc.clear();
    for (col, value) in columns {
        let element = vector.elem(col);
        // A non-finite matrix entry makes the tape's `value * 0.0` term NaN
        // where the fold drops it, so this row promises nothing.
        if !element.complete || !value.is_finite() {
            return false;
        }
        if element.coeffs.is_empty() {
            continue;
        }
        merge_scaled(acc, element.coeffs, value, out);
        std::mem::swap(acc, out);
        if acc.len() > MAX_CONSTANT_DEGREE {
            return false;
        }
    }
    true
}

/// `Add`/`Sub`: coefficients merge seed-wise; two seed-free operands fold.
fn classify_add_sub(
    left: &NodeInfo,
    right: &NodeInfo,
    subtract: bool,
    len: usize,
    scratch: &mut Scratch,
) -> NodeInfo {
    if !left.is_seeded() && !right.is_seeded() {
        return fold_literals(
            left,
            right,
            len,
            |x, y| if subtract { x - y } else { x + y },
        );
    }
    let n_elems = combined_elems(left, right, len);
    let mut built = Coeffs::with_capacity(n_elems);
    for i in 0..n_elems {
        let (le, re) = (left.elem(i), right.elem(i));
        merge_sum(le.coeffs, re.coeffs, subtract, &mut scratch.out);
        built.push(&scratch.out, le.complete && re.complete);
    }
    NodeInfo::Seeded(built)
}

/// `MatMul` against a literal matrix: each row accumulates in the tape's own
/// column order. A non-literal matrix resolves nothing.
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
            let mut built = Coeffs::with_capacity(csr.shape.rows);
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
            let mut built = Coeffs::with_capacity(rows);
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
    info: &[Option<NodeInfo>],
    len: usize,
) -> NodeInfo {
    let infos: Vec<&NodeInfo> = children.iter().map(|&c| child_info(info, c)).collect();
    let widths = children.iter().map(|&c| sizes[c.index()]);

    if infos.iter().any(|child| child.is_seeded()) {
        let mut built = Coeffs::with_capacity(len);
        for (child, width) in infos.iter().zip(widths) {
            for i in 0..width {
                built.push_element(child.elem(i));
            }
        }
        return NodeInfo::Seeded(built);
    }
    let Some(literals) = infos
        .iter()
        .map(|child| child.literal())
        .collect::<Option<Vec<_>>>()
    else {
        return NodeInfo::Primal;
    };
    let mut values = Vec::with_capacity(len);
    for (literal, width) in literals.iter().zip(widths) {
        values.extend((0..width).map(|i| literal.at(i)));
    }
    NodeInfo::Literal(Literal::Vector(values))
}

/// `Index`: a window into the child's elements, or the child itself when it
/// broadcasts.
fn classify_index(child: &NodeInfo, start: usize, end: usize) -> NodeInfo {
    match child {
        NodeInfo::Primal => NodeInfo::Primal,
        NodeInfo::Literal(Literal::Scalar(value)) => NodeInfo::Literal(Literal::Scalar(*value)),
        NodeInfo::Literal(Literal::Vector(values)) => {
            NodeInfo::Literal(Literal::Vector(values[start..end].to_vec()))
        },
        NodeInfo::Seeded(coeffs) if coeffs.n_elems() == 1 => NodeInfo::Seeded(coeffs.clone()),
        NodeInfo::Seeded(coeffs) => {
            let mut built = Coeffs::with_capacity(end - start);
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
    info: &[Option<NodeInfo>],
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
            let mut built = Coeffs::with_capacity(end - start);
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

        // A parameter tangent is a seed outside this index space, so nothing
        // downstream of it can be resolved against the state seeds.
        Node::TangentParameter { .. } => NodeInfo::Seeded(Coeffs::unresolved(1)),

        Node::Neg(a) => match child_info(info, *a) {
            NodeInfo::Primal => NodeInfo::Primal,
            NodeInfo::Literal(literal) => NodeInfo::Literal(literal.negated()),
            seeded @ NodeInfo::Seeded(_) => {
                map_coeffs(seeded, seeded.n_elems(), scratch, |c, _| c.negated())
            },
        },

        Node::Add(a, b) => classify_add_sub(
            child_info(info, *a),
            child_info(info, *b),
            false,
            len,
            scratch,
        ),
        Node::Sub(a, b) => classify_add_sub(
            child_info(info, *a),
            child_info(info, *b),
            true,
            len,
            scratch,
        ),

        Node::Mul(a, b) => {
            let (left, right) = (child_info(info, *a), child_info(info, *b));
            // Scaling is order-free in IEEE arithmetic, so one rule serves
            // both `literal * tangent` and `tangent * literal`.
            match (left.is_seeded(), right.is_seeded()) {
                (false, false) => fold_literals(left, right, len, |x, y| x * y),
                (false, true) => scale_seeded(right, left, len, false, scratch),
                (true, false) => scale_seeded(left, right, len, false, scratch),
                (true, true) => unknown_pair(left, right, len, scratch),
            }
        },

        Node::Div(a, b) => {
            let (left, right) = (child_info(info, *a), child_info(info, *b));
            if right.is_seeded() {
                unknown_pair(left, right, len, scratch)
            } else if left.is_seeded() {
                scale_seeded(left, right, len, true, scratch)
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
    use crate::eval::{CompiledExpr, TangentInputs};
    use crate::ir::TypedIr;
    use crate::node::{ArrayData, CsrData, Shape};
    use crate::simplify::simplify_pipeline;
    use crate::sparsity::detect_sparsity_per_output;
    use crate::tangent::tangent_wrt_states;

    /// Classify `root`'s state Jacobian the way `JacobianData` does, and return
    /// the pattern alongside the split.
    fn classify_states(
        arena: &Arena,
        root: NodeId,
        n_rows: usize,
        n_states: usize,
    ) -> (SparsityPattern, Vec<bool>, Vec<(usize, f64)>) {
        let mut diff_arena = arena.clone();
        let tangent_root = tangent_wrt_states(&mut diff_arena, root);
        let (diff_arena, tangent_root) = simplify_pipeline(diff_arena, tangent_root);
        let pattern = detect_sparsity_per_output(arena, root, n_rows, n_states);
        let (varying, entries) = classify_constant_entries(&diff_arena, tangent_root, &pattern);
        (pattern, varying, entries)
    }

    /// Dense `(row, col) -> value` view of a split's constant table.
    fn constant_map(
        pattern: &SparsityPattern,
        entries: &[(usize, f64)],
    ) -> std::collections::HashMap<(usize, usize), f64> {
        let rows = pattern.entry_rows();
        entries
            .iter()
            .map(|&(csr_idx, value)| ((rows[csr_idx], pattern.indices[csr_idx]), value))
            .collect()
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact equality is the point: pins the fold
    fn linear_row_folds_to_exact_coefficients() {
        // f(y) = [2*y0 - y1, y1] over 2 states: row 0 is fully constant.
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
    #[allow(clippy::float_cmp)] // exact equality is the point: pins the fold
    fn array_factor_scales_each_element() {
        // f(y) = a * y with a literal vector a, so df/dy is diag(a).
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
    #[allow(clippy::float_cmp)] // exact equality is the point: pins the fold
    fn sparse_matmul_folds_matrix_entries() {
        // f(y) = A @ y for a literal tridiagonal A: df/dy is A itself.
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
        // f(y) = exp(y) * y: the tangent carries exp(y) as a factor, which is
        // seed-free but not literal, so the entry has to be swept.
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let factor = arena.alloc(Node::Exp(y));
        let root = arena.alloc(Node::Mul(factor, y));

        let (_, varying, entries) = classify_states(&arena, root, 2, 2);
        assert!(varying.iter().all(|&v| v), "every entry must be swept");
        assert!(entries.is_empty());
    }

    #[test]
    fn unmodelled_node_marks_its_row_incomplete() {
        // A reduction over the whole state is not modelled element-wise, so its
        // row is swept even though the summed rows are linear.
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let reduced = arena.alloc(Node::MaxReduce(y));
        let root = arena.alloc(Node::Concat(vec![reduced, y]));

        let (pattern, varying, _) = classify_states(&arena, root, 4, 3);
        let row0 = pattern.indptr[0]..pattern.indptr[1];
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
    #[allow(clippy::float_cmp)] // exact equality is the point: pins the fold
    fn structurally_present_but_absent_entries_classify_as_zero() {
        // `sign` keeps its argument in the pattern but differentiates to
        // nothing, so entry (0, 0) is a constant zero, not an omission.
        let mut arena = Arena::new();
        let y0 = arena.alloc(Node::StateVector { start: 0, end: 1 });
        let y1 = arena.alloc(Node::StateVector { start: 1, end: 2 });
        let step = arena.alloc(Node::Sign(y0));
        let root = arena.alloc(Node::Concat(vec![step, y1]));

        let (pattern, varying, entries) = classify_states(&arena, root, 2, 2);
        assert_eq!(pattern.indices[pattern.indptr[0]..pattern.indptr[1]], [0]);
        assert!(!varying.iter().any(|&v| v));
        let constants = constant_map(&pattern, &entries);
        assert_eq!(constants[&(0, 0)], 0.0);
        assert_eq!(constants[&(1, 1)], 1.0);
    }

    #[test]
    fn folded_values_match_the_tape() {
        // The soundness property in miniature: every folded entry equals what
        // the tangent tape produces for that column, bit for bit.
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

        let mut diff_arena = arena.clone();
        let tangent_root = tangent_wrt_states(&mut diff_arena, root);
        let (diff_arena, tangent_root) = simplify_pipeline(diff_arena, tangent_root);
        let pattern = detect_sparsity_per_output(&arena, root, n, n);
        let (_, entries) = classify_constant_entries(&diff_arena, tangent_root, &pattern);

        let expr = CompiledExpr::from_ir(TypedIr::from_arena_split_eval(&diff_arena, tangent_root));
        let mut scratch = vec![0.0; expr.scratch_len()];
        let y_values = [0.3, -1.2, 2.4];
        let table = constant_map(&pattern, &entries);
        let mut cache = expr.eval_primal(&mut scratch, 0.0, &y_values, &[], &[]);
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
    fn wide_linear_rows_hit_the_degree_cap() {
        // Past MAX_CONSTANT_DEGREE columns an element is abandoned rather than
        // grown, so the row falls back to being swept.
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
        assert!(entries.is_empty());
    }
}
