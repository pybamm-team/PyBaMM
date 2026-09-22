//! [`Node`], the expression DAG vocabulary handed over from Python.
//!
//! Nodes are not shape-checked on construction; [`validate`](crate::validate)
//! reports what lowering will reject.

use crate::arena::NodeId;
use crate::error::CoreError;

/// Declared `rows × cols` of a literal; only matmul reads it, elsewhere the flat length counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Shape {
    pub rows: usize,
    pub cols: usize,
}

/// Knots, values and per-segment slopes of a 1-D linear interpolant.
///
/// `slopes` is derived only in [`try_new`](Self::try_new), which deserialising also uses.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serialize",
    derive(serde::Serialize, serde::Deserialize),
    serde(try_from = "PersistedInterpolant")
)]
pub struct InterpolantData {
    pub(crate) x_data: Vec<f64>,
    pub(crate) y_data: Vec<f64>,
    #[cfg_attr(feature = "serialize", serde(skip_serializing))]
    pub(crate) slopes: Vec<f64>,
}

/// Breakpoints and per-interval coefficients of a 1-D cubic interpolant.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serialize",
    derive(serde::Serialize, serde::Deserialize),
    serde(try_from = "PersistedCubicInterpolant")
)]
pub struct CubicInterpolantData {
    pub(crate) breakpoints: Vec<f64>,
    /// Per-interval `[c0, c1, c2, c3]` of `c0 + c1*dx + c2*dx^2 + c3*dx^3`,
    /// `dx = x - breakpoints[i]`.
    pub(crate) coeffs: Vec<[f64; 4]>,
}

/// Breakpoints and coefficients of a 2-D or 3-D tensor-product interpolant.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serialize",
    derive(serde::Serialize, serde::Deserialize),
    serde(try_from = "PersistedNdInterpolant")
)]
pub struct NdInterpolantData {
    pub(crate) breakpoints: Vec<Vec<f64>>,
    /// Cell-major (axis 0 slowest) power-basis tensors, `order^ndim` per cell with
    /// axis-0 power slowest: `p(dx) = Σ c[a0,..] · Π dx_i^a_i`.
    pub(crate) coeffs: Vec<f64>,
    /// Per-axis polynomial order: 2 = multilinear, 4 = tensor cubic.
    pub(crate) order: u32,
}

impl Shape {
    /// `len × 1`, the column-vector orientation `PyBaMM` discretises into.
    pub const fn vector(len: usize) -> Self {
        Self { rows: len, cols: 1 }
    }

    pub const fn matrix(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }
}

/// A constant sparse matrix in CSR form.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serialize",
    derive(serde::Serialize, serde::Deserialize),
    serde(try_from = "PersistedCsr")
)]
pub struct CsrData {
    pub(crate) indptr: Vec<usize>,
    pub(crate) indices: Vec<usize>,
    pub(crate) data: Vec<f64>,
    pub(crate) shape: Shape,
}

/// A constant dense array, stored row-major and flat.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serialize",
    derive(serde::Serialize, serde::Deserialize),
    serde(try_from = "PersistedArray")
)]
pub struct ArrayData {
    pub(crate) data: Vec<f64>,
    pub(crate) shape: Shape,
}

/// Define `$persisted`, the on-disk form of `$data`, which loads through the validating `$new`.
///
/// Bincode is positional, so the fields must follow `$data`'s declaration order.
#[cfg(feature = "serialize")]
macro_rules! persisted_via_constructor {
    ($persisted:ident => $data:ident::$new:ident($($field:ident: $ty:ty),+ $(,)?)) => {
        #[derive(serde::Deserialize)]
        struct $persisted {
            $($field: $ty),+
        }

        impl TryFrom<$persisted> for $data {
            type Error = CoreError;

            fn try_from(persisted: $persisted) -> Result<Self, CoreError> {
                Self::$new($(persisted.$field),+)
            }
        }
    };
}

#[cfg(feature = "serialize")]
persisted_via_constructor!(PersistedCsr => CsrData::try_new(
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f64>,
    shape: Shape,
));
#[cfg(feature = "serialize")]
persisted_via_constructor!(PersistedArray => ArrayData::try_new(data: Vec<f64>, shape: Shape));
#[cfg(feature = "serialize")]
persisted_via_constructor!(PersistedInterpolant => InterpolantData::try_new(
    x_data: Vec<f64>,
    y_data: Vec<f64>,
));
#[cfg(feature = "serialize")]
persisted_via_constructor!(PersistedCubicInterpolant => CubicInterpolantData::try_new(
    breakpoints: Vec<f64>,
    coeffs: Vec<[f64; 4]>,
));
#[cfg(feature = "serialize")]
persisted_via_constructor!(PersistedNdInterpolant => NdInterpolantData::try_new(
    breakpoints: Vec<Vec<f64>>,
    coeffs: Vec<f64>,
    order: u32,
));

impl CsrData {
    /// Build a CSR matrix, validating its structural invariants.
    pub fn try_new(
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<f64>,
        shape: Shape,
    ) -> Result<Self, CoreError> {
        let (rows, cols) = (shape.rows, shape.cols);
        // Not `rows + 1`, which wraps for `rows == usize::MAX` and admits an empty indptr.
        if indptr.len().checked_sub(1) != Some(rows) {
            return Err(CoreError::Csr(format!(
                "indptr length {} must equal rows + 1 for {rows} rows",
                indptr.len(),
            )));
        }
        if indptr[0] != 0 {
            return Err(CoreError::Csr(format!(
                "indptr[0] must be 0, got {}",
                indptr[0]
            )));
        }
        if indptr.windows(2).any(|w| w[1] < w[0]) {
            return Err(CoreError::Csr("indptr must be non-decreasing".to_string()));
        }
        if indices.len() != data.len() {
            return Err(CoreError::Csr(format!(
                "indices length {} must equal data length {}",
                indices.len(),
                data.len()
            )));
        }
        let nnz = indptr[rows];
        if nnz != data.len() {
            return Err(CoreError::Csr(format!(
                "indptr tail {nnz} must equal nnz {}",
                data.len()
            )));
        }
        if let Some(&max_col) = indices.iter().max()
            && max_col >= cols
        {
            return Err(CoreError::Csr(format!(
                "column index {max_col} out of range for {cols} columns"
            )));
        }
        Ok(Self {
            indptr,
            indices,
            data,
            shape,
        })
    }

    /// Row-offset array, length `shape.rows + 1`.
    pub fn indptr(&self) -> &[usize] {
        &self.indptr
    }

    /// Column index for each stored entry.
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Stored values, parallel to [`indices`](Self::indices).
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Matrix shape.
    pub const fn shape(&self) -> Shape {
        self.shape
    }
}

impl ArrayData {
    /// Build a dense array, validating that `data.len()` equals `rows * cols`.
    pub fn try_new(data: Vec<f64>, shape: Shape) -> Result<Self, CoreError> {
        let expected = shape.rows.checked_mul(shape.cols).ok_or_else(|| {
            CoreError::Array(format!(
                "shape {}x{} overflows usize",
                shape.rows, shape.cols
            ))
        })?;
        if data.len() != expected {
            return Err(CoreError::Array(format!(
                "data length {} must equal rows*cols = {expected}",
                data.len()
            )));
        }
        Ok(Self { data, shape })
    }

    /// Row-major dense values.
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Array shape.
    pub const fn shape(&self) -> Shape {
        self.shape
    }
}

impl InterpolantData {
    /// Build a 1D linear table from finite, non-decreasing knots, deriving the slopes.
    ///
    /// A repeated interior knot is a step, right-continuous as `CasADi`'s; the end segments
    /// extrapolate, so they must have nonzero width.
    pub fn try_new(x_data: Vec<f64>, y_data: Vec<f64>) -> Result<Self, CoreError> {
        if x_data.is_empty() {
            return Err(CoreError::Interpolant(
                "x_data must be non-empty".to_string(),
            ));
        }
        if x_data.len() != y_data.len() {
            return Err(CoreError::Interpolant(format!(
                "x_data length {} must equal y_data length {}",
                x_data.len(),
                y_data.len()
            )));
        }
        if !x_data.iter().all(|v| v.is_finite()) {
            return Err(CoreError::Interpolant("x_data must be finite".to_string()));
        }
        if x_data.windows(2).any(|w| w[1] < w[0]) {
            return Err(CoreError::Interpolant(
                "x_data must be non-decreasing".to_string(),
            ));
        }
        let n = x_data.len();
        if n > 1 && (x_data[1] <= x_data[0] || x_data[n - 1] <= x_data[n - 2]) {
            return Err(CoreError::Interpolant(
                "x_data must not repeat its first or last knot".to_string(),
            ));
        }
        // The knot search never picks a zero-width segment, so its slope is never read.
        let slopes = x_data
            .windows(2)
            .zip(y_data.windows(2))
            .map(|(x, y)| {
                if x[1] > x[0] {
                    (y[1] - y[0]) / (x[1] - x[0])
                } else {
                    0.0
                }
            })
            .collect();
        Ok(Self {
            x_data,
            y_data,
            slopes,
        })
    }
}

impl CubicInterpolantData {
    /// Build a 1D cubic table from flat row-major coefficients `[c0,c1,c2,c3, c0,...]`.
    pub fn try_from_flat(breakpoints: Vec<f64>, coeffs: &[f64]) -> Result<Self, CoreError> {
        let (tuples, rest) = coeffs.as_chunks::<4>();
        if !rest.is_empty() {
            return Err(CoreError::Interpolant(format!(
                "cubic interpolant coefficients must be a whole number of 4-tuples, got {}",
                coeffs.len()
            )));
        }
        Self::try_new(breakpoints, tuples.to_vec())
    }

    /// Build a 1D cubic table, validating breakpoints and one coefficient tuple per segment.
    pub fn try_new(breakpoints: Vec<f64>, coeffs: Vec<[f64; 4]>) -> Result<Self, CoreError> {
        if breakpoints.len() < 2 {
            return Err(CoreError::Interpolant(format!(
                "cubic interpolant needs at least 2 breakpoints, got {}",
                breakpoints.len()
            )));
        }
        let nseg = breakpoints.len() - 1;
        if coeffs.len() != nseg {
            return Err(CoreError::Interpolant(format!(
                "cubic interpolant needs {nseg} coefficient tuples (breakpoints - 1), got {}",
                coeffs.len()
            )));
        }
        if !breakpoints.iter().all(|v| v.is_finite()) {
            return Err(CoreError::Interpolant(
                "cubic interpolant breakpoints must be finite".to_string(),
            ));
        }
        if breakpoints.windows(2).any(|w| w[1] <= w[0]) {
            return Err(CoreError::Interpolant(
                "cubic interpolant breakpoints must be strictly increasing".to_string(),
            ));
        }
        Ok(Self {
            breakpoints,
            coeffs,
        })
    }
}

impl NdInterpolantData {
    /// Build an N-D tensor table, validating axes, order, knots and coefficient count.
    pub fn try_new(
        breakpoints: Vec<Vec<f64>>,
        coeffs: Vec<f64>,
        order: u32,
    ) -> Result<Self, CoreError> {
        let ndim = breakpoints.len();
        if !(2..=3).contains(&ndim) {
            return Err(CoreError::Interpolant(format!(
                "N-D interpolant supports 2 or 3 axes, got {ndim}"
            )));
        }
        if order != 2 && order != 4 {
            return Err(CoreError::Interpolant(format!(
                "N-D interpolant order must be 2 or 4, got {order}"
            )));
        }
        // Checked: a table deserialised from caller bytes must not wrap to a small count.
        let overflow =
            || CoreError::Interpolant("N-D interpolant cell count overflows usize".to_string());
        let mut ncells = 1usize;
        for (axis, knots) in breakpoints.iter().enumerate() {
            if knots.len() < 2 {
                return Err(CoreError::Interpolant(format!(
                    "N-D interpolant axis {axis} needs at least 2 breakpoints, got {}",
                    knots.len()
                )));
            }
            if !knots.iter().all(|v| v.is_finite()) {
                return Err(CoreError::Interpolant(format!(
                    "N-D interpolant axis {axis} breakpoints must be finite"
                )));
            }
            if knots.windows(2).any(|w| w[1] <= w[0]) {
                return Err(CoreError::Interpolant(format!(
                    "N-D interpolant axis {axis} breakpoints must be strictly increasing"
                )));
            }
            ncells = ncells.checked_mul(knots.len() - 1).ok_or_else(overflow)?;
        }
        let order_usize = order as usize;
        let per_cell = match ndim {
            2 => order_usize * order_usize,
            3 => order_usize * order_usize * order_usize,
            _ => unreachable!("ndim validated to 2..=3"),
        };
        let expected = ncells.checked_mul(per_cell).ok_or_else(overflow)?;
        if coeffs.len() != expected {
            return Err(CoreError::Interpolant(format!(
                "N-D interpolant needs {expected} coefficients (ncells * order^ndim), got {}",
                coeffs.len()
            )));
        }
        Ok(Self {
            breakpoints,
            coeffs,
            order,
        })
    }
}

/// One node of the expression DAG.
///
/// Children are [`NodeId`]s into the owning arena. Arithmetic and unary variants are
/// element-wise, with a scalar operand broadcasting against a vector.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum Node {
    Scalar(f64),
    Array(Box<ArrayData>),
    ZeroVector {
        len: usize,
    },
    SparseMatrix(Box<CsrData>),
    /// Reads `y[start..end]` of the solver's global state vector.
    StateVector {
        start: usize,
        end: usize,
    },
    /// Reads `y'[start..end]`, in the same index space as [`StateVector`](Self::StateVector).
    StateVectorDot {
        start: usize,
        end: usize,
    },
    InputParameter {
        name: String,
        /// Registration order among distinct names; used for sensitivity indexing.
        index: usize,
        /// Offset into the packed `p` values array.
        offset: usize,
        /// Number of packed values this parameter occupies (>1 for vector inputs).
        width: usize,
    },
    Time,

    Add(NodeId, NodeId),
    Sub(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Div(NodeId, NodeId),
    Pow(NodeId, NodeId),
    /// Matrix-vector product; the left child must be a constant
    /// [`SparseMatrix`](Self::SparseMatrix) or [`Array`](Self::Array).
    MatMul(NodeId, NodeId),
    Minimum(NodeId, NodeId),
    Maximum(NodeId, NodeId),
    Modulo(NodeId, NodeId),
    Hypot(NodeId, NodeId),
    EqualHeaviside(NodeId, NodeId),
    NotEqualHeaviside(NodeId, NodeId),
    Equality(NodeId, NodeId),

    /// Half-open slice `child[start..end]` of the child's own value.
    Index {
        child: NodeId,
        start: usize,
        end: usize,
    },
    /// Children joined end to end; for a right-hand side this is the solver's equation order.
    Concat(Vec<NodeId>),

    Interpolant1DLinear {
        data: Box<InterpolantData>,
        child: NodeId,
    },
    /// 1D cubic/pchip interpolation (piecewise cubic, power-basis coeffs).
    Interpolant1DCubic {
        data: Box<CubicInterpolantData>,
        child: NodeId,
    },
    /// N-D tensor-product interpolation, one equal-length child per axis.
    InterpolantNd {
        data: Box<NdInterpolantData>,
        children: Vec<NodeId>,
    },

    Neg(NodeId),
    Abs(NodeId),
    Sqrt(NodeId),
    Exp(NodeId),
    Log(NodeId),
    Sin(NodeId),
    Cos(NodeId),
    Tanh(NodeId),
    Sinh(NodeId),
    Cosh(NodeId),
    Arcsinh(NodeId),
    Arctan(NodeId),
    Erf(NodeId),
    Sign(NodeId),
    Floor(NodeId),
    Ceiling(NodeId),
    MaxReduce(NodeId),
    MinReduce(NodeId),

    /// Internal: `basis[k]`, `k` the first-occurrence argmax (or argmin) of `picker`;
    /// the subgradient of `MaxReduce`/`MinReduce`.
    ReduceArgSelect {
        basis: NodeId,
        picker: NodeId,
        is_max: bool,
    },

    /// Selects `branches[i]` when `selector` is in the open window `(i + 0.5, i + 1.5)`;
    /// an exact half-way value matches nothing and evaluates to zeros.
    Conditional {
        selector: NodeId,
        branches: Vec<NodeId>,
    },

    /// Reads a slice of the tangent state vector.
    TangentStateVector {
        start: usize,
        end: usize,
    },
    /// Reads a single tangent parameter.
    TangentParameter {
        index: usize,
    },
    /// Derivative of linear interpolation, from the table's pre-computed slopes.
    Interpolant1DLinearDeriv {
        data: Box<InterpolantData>,
        child: NodeId,
    },
    /// Derivative of 1D cubic/pchip interpolation.
    Interpolant1DCubicDeriv {
        data: Box<CubicInterpolantData>,
        child: NodeId,
    },
    /// Partial derivative of N-D interpolation along `axis`.
    InterpolantNdPartial {
        data: Box<NdInterpolantData>,
        children: Vec<NodeId>,
        axis: u32,
    },
}

/// Hand each child of `$node` to `$visit`, in declaration order.
///
/// One match serves the shared and mutable visitors, so their orders cannot drift apart.
macro_rules! visit_children {
    ($node:expr, $visit:ident) => {
        match $node {
            Node::Scalar(_)
            | Node::Array(_)
            | Node::ZeroVector { .. }
            | Node::SparseMatrix(_)
            | Node::StateVector { .. }
            | Node::StateVectorDot { .. }
            | Node::InputParameter { .. }
            | Node::Time
            | Node::TangentStateVector { .. }
            | Node::TangentParameter { .. } => {},

            Node::Add(l, r)
            | Node::Sub(l, r)
            | Node::Mul(l, r)
            | Node::Div(l, r)
            | Node::Pow(l, r)
            | Node::MatMul(l, r)
            | Node::Minimum(l, r)
            | Node::Maximum(l, r)
            | Node::Modulo(l, r)
            | Node::Hypot(l, r)
            | Node::EqualHeaviside(l, r)
            | Node::NotEqualHeaviside(l, r)
            | Node::Equality(l, r) => {
                $visit(l);
                $visit(r);
            },

            Node::Neg(c)
            | Node::Abs(c)
            | Node::Sqrt(c)
            | Node::Exp(c)
            | Node::Log(c)
            | Node::Sin(c)
            | Node::Cos(c)
            | Node::Tanh(c)
            | Node::Sinh(c)
            | Node::Cosh(c)
            | Node::Arcsinh(c)
            | Node::Arctan(c)
            | Node::Erf(c)
            | Node::Sign(c)
            | Node::Floor(c)
            | Node::Ceiling(c)
            | Node::MaxReduce(c)
            | Node::MinReduce(c) => $visit(c),

            Node::Index { child, .. }
            | Node::Interpolant1DLinear { child, .. }
            | Node::Interpolant1DLinearDeriv { child, .. }
            | Node::Interpolant1DCubic { child, .. }
            | Node::Interpolant1DCubicDeriv { child, .. } => $visit(child),

            Node::Concat(children)
            | Node::InterpolantNd { children, .. }
            | Node::InterpolantNdPartial { children, .. } => {
                for c in children {
                    $visit(c);
                }
            },

            Node::Conditional { selector, branches } => {
                $visit(selector);
                for b in branches {
                    $visit(b);
                }
            },

            Node::ReduceArgSelect { basis, picker, .. } => {
                $visit(basis);
                $visit(picker);
            },
        }
    };
}

/// Require `start <= end`; an inverted pair would wrap the width in release.
const fn check_extent(node: &'static str, start: usize, end: usize) -> Result<(), CoreError> {
    if start > end {
        return Err(CoreError::InvertedExtent { node, start, end });
    }
    Ok(())
}

impl Node {
    /// A zero of broadcast length `len`: a `Scalar` for 1, else a `ZeroVector` (0 included).
    pub(crate) const fn zero(len: usize) -> Self {
        if len == 1 {
            Self::Scalar(0.0)
        } else {
            Self::ZeroVector { len }
        }
    }

    /// The value of a `Scalar` literal; constant-valued subtrees are not folded.
    pub(crate) const fn as_scalar(&self) -> Option<f64> {
        match self {
            Self::Scalar(value) => Some(*value),
            _ => None,
        }
    }

    /// The state slice `y[start..end]`.
    pub fn state_vector(start: usize, end: usize) -> Result<Self, CoreError> {
        check_extent("StateVector", start, end)?;
        Ok(Self::StateVector { start, end })
    }

    /// The time-derivative slice `y_dot[start..end]`.
    pub fn state_vector_dot(start: usize, end: usize) -> Result<Self, CoreError> {
        check_extent("StateVectorDot", start, end)?;
        Ok(Self::StateVectorDot { start, end })
    }

    /// `child[start..end]`; [`validate`](crate::validate) checks it fits the child.
    pub fn index(child: NodeId, start: usize, end: usize) -> Result<Self, CoreError> {
        check_extent("Index", start, end)?;
        Ok(Self::Index { child, start, end })
    }

    /// A branch selected by `selector`, 1-based.
    pub fn conditional(selector: NodeId, branches: Vec<NodeId>) -> Result<Self, CoreError> {
        if branches.is_empty() {
            return Err(CoreError::EmptyConditional);
        }
        Ok(Self::Conditional { selector, branches })
    }

    /// A tensor-product interpolant evaluated at one coordinate per axis.
    pub fn interpolant_nd(
        data: NdInterpolantData,
        children: Vec<NodeId>,
    ) -> Result<Self, CoreError> {
        let axes = data.breakpoints.len();
        if children.len() != axes {
            return Err(CoreError::InterpolantNdArity {
                axes,
                children: children.len(),
            });
        }
        Ok(Self::InterpolantNd {
            data: Box::new(data),
            children,
        })
    }

    /// Visit each child `NodeId` of this node, allocation-free.
    pub(crate) fn for_each_child<F: FnMut(NodeId)>(&self, mut f: F) {
        let mut visit = |child: &NodeId| f(*child);
        visit_children!(self, visit);
    }

    /// Visit each child `NodeId` in place, in [`for_each_child`](Self::for_each_child) order.
    pub(crate) fn for_each_child_mut<F: FnMut(&mut NodeId)>(&mut self, mut f: F) {
        visit_children!(self, f);
    }

    /// A copy of this node with `f` applied to each child `NodeId`, in
    /// [`for_each_child`](Self::for_each_child) order.
    #[must_use]
    pub(crate) fn map_children<F: FnMut(NodeId) -> NodeId>(&self, mut f: F) -> Self {
        let mut mapped = self.clone();
        mapped.for_each_child_mut(|child| *child = f(*child));
        mapped
    }
}

fn hash_nd_interpolant_data<H: std::hash::Hasher>(data: &NdInterpolantData, hasher: &mut H) {
    use std::hash::Hash;
    data.order.hash(hasher);
    for knots in &data.breakpoints {
        knots.len().hash(hasher);
        for v in knots {
            v.to_bits().hash(hasher);
        }
    }
    for v in &data.coeffs {
        v.to_bits().hash(hasher);
    }
}

/// Hash a node by structure for CSE, floats by bit pattern.
///
/// Pairs with [`structurally_identical`]: identical nodes hash equally, but distinct ones
/// may collide, e.g. payloads differing in an even number of zero signs.
pub(crate) fn structural_hash<H: std::hash::Hasher>(node: &Node, hasher: &mut H) {
    use std::hash::Hash;
    std::mem::discriminant(node).hash(hasher);
    match node {
        Node::Scalar(v) => v.to_bits().hash(hasher),
        Node::Array(arr) => {
            for v in &arr.data {
                v.to_bits().hash(hasher);
            }
            arr.shape.rows.hash(hasher);
            arr.shape.cols.hash(hasher);
        },
        Node::ZeroVector { len } => len.hash(hasher),
        Node::SparseMatrix(csr) => {
            csr.indptr.hash(hasher);
            csr.indices.hash(hasher);
            for v in &csr.data {
                v.to_bits().hash(hasher);
            }
            csr.shape.rows.hash(hasher);
            csr.shape.cols.hash(hasher);
        },
        Node::StateVector { start, end }
        | Node::StateVectorDot { start, end }
        | Node::TangentStateVector { start, end }
        | Node::Index { start, end, .. } => {
            start.hash(hasher);
            end.hash(hasher);
        },
        Node::InputParameter {
            name,
            index,
            offset,
            width,
        } => {
            name.hash(hasher);
            index.hash(hasher);
            offset.hash(hasher);
            width.hash(hasher);
        },
        Node::TangentParameter { index } => index.hash(hasher),
        Node::Concat(children) => children.len().hash(hasher),
        Node::Conditional { branches, .. } => branches.len().hash(hasher),
        // Slopes derive from knots and values, so need no hashing.
        Node::Interpolant1DLinear { data, .. } | Node::Interpolant1DLinearDeriv { data, .. } => {
            for v in &data.x_data {
                v.to_bits().hash(hasher);
            }
            for v in &data.y_data {
                v.to_bits().hash(hasher);
            }
        },
        Node::Interpolant1DCubic { data, .. } | Node::Interpolant1DCubicDeriv { data, .. } => {
            for v in &data.breakpoints {
                v.to_bits().hash(hasher);
            }
            for c in &data.coeffs {
                for v in c {
                    v.to_bits().hash(hasher);
                }
            }
        },
        Node::InterpolantNd { data, .. } => hash_nd_interpolant_data(data, hasher),
        Node::InterpolantNdPartial { data, axis, .. } => {
            hash_nd_interpolant_data(data, hasher);
            axis.hash(hasher);
        },
        Node::Time
        | Node::Add(_, _)
        | Node::Sub(_, _)
        | Node::Mul(_, _)
        | Node::Div(_, _)
        | Node::Pow(_, _)
        | Node::MatMul(_, _)
        | Node::Minimum(_, _)
        | Node::Maximum(_, _)
        | Node::Modulo(_, _)
        | Node::Hypot(_, _)
        | Node::EqualHeaviside(_, _)
        | Node::NotEqualHeaviside(_, _)
        | Node::Equality(_, _)
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
        | Node::MaxReduce(_)
        | Node::MinReduce(_) => {},
        Node::ReduceArgSelect { is_max, .. } => is_max.hash(hasher),
    }
    node.for_each_child(|c| c.hash(hasher));
}

/// Whether `a` and `b` are the same node, with floats compared by bit pattern.
///
/// The equality [`structural_hash`] pairs with: derived `==` would merge `0.0` and `-0.0`.
pub(crate) fn structurally_identical(a: &Node, b: &Node) -> bool {
    fn same_bits(x: &[f64], y: &[f64]) -> bool {
        x.iter()
            .map(|v| v.to_bits())
            .eq(y.iter().map(|v| v.to_bits()))
    }
    // `==` has already matched the variant and every non-float field.
    a == b
        && match (a, b) {
            (Node::Scalar(x), Node::Scalar(y)) => x.to_bits() == y.to_bits(),
            (Node::Array(x), Node::Array(y)) => same_bits(&x.data, &y.data),
            (Node::SparseMatrix(x), Node::SparseMatrix(y)) => same_bits(&x.data, &y.data),
            (
                Node::Interpolant1DLinear { data: x, .. }
                | Node::Interpolant1DLinearDeriv { data: x, .. },
                Node::Interpolant1DLinear { data: y, .. }
                | Node::Interpolant1DLinearDeriv { data: y, .. },
            ) => same_bits(&x.x_data, &y.x_data) && same_bits(&x.y_data, &y.y_data),
            (
                Node::Interpolant1DCubic { data: x, .. }
                | Node::Interpolant1DCubicDeriv { data: x, .. },
                Node::Interpolant1DCubic { data: y, .. }
                | Node::Interpolant1DCubicDeriv { data: y, .. },
            ) => {
                same_bits(&x.breakpoints, &y.breakpoints)
                    && same_bits(x.coeffs.as_flattened(), y.coeffs.as_flattened())
            },
            (
                Node::InterpolantNd { data: x, .. } | Node::InterpolantNdPartial { data: x, .. },
                Node::InterpolantNd { data: y, .. } | Node::InterpolantNdPartial { data: y, .. },
            ) => {
                x.breakpoints
                    .iter()
                    .zip(&y.breakpoints)
                    .all(|(p, q)| same_bits(p, q))
                    && same_bits(&x.coeffs, &y.coeffs)
            },
            _ => true,
        }
}

const _: () = {
    assert!(size_of::<Node>() <= 48);
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;

    /// `result` is `Ok` exactly when `valid`, and otherwise an error `kind` accepts.
    #[track_caller]
    fn assert_validated<T>(
        result: Result<T, CoreError>,
        valid: bool,
        kind: fn(&CoreError) -> bool,
        label: &str,
    ) {
        match result {
            Ok(_) => assert!(valid, "{label} must be refused"),
            Err(err) => assert!(!valid && kind(&err), "{label}: {err}"),
        }
    }

    fn is_interpolant(err: &CoreError) -> bool {
        matches!(err, CoreError::Interpolant(_))
    }

    #[test]
    fn csr_try_new_accepts_a_consistent_matrix_and_refuses_the_rest() {
        let (square, tall) = (Shape::matrix(2, 2), Shape::matrix(usize::MAX, 1));
        let cases = [
            (vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0], square, true),
            (vec![0, 2], vec![0, 1], vec![1.0, 2.0], square, false),
            (vec![], vec![], vec![], tall, false),
            (vec![0, 2, 1], vec![0, 1], vec![1.0, 2.0], square, false),
            (vec![0, 1, 2], vec![0, 5], vec![1.0, 2.0], square, false),
            (vec![0, 1, 2], vec![0, 1], vec![1.0], square, false),
            (vec![0, 1, 2], vec![0], vec![1.0], square, false),
        ];
        for (indptr, indices, data, shape, valid) in cases {
            let label = format!("{indptr:?} {indices:?} {data:?} {shape:?}");
            let result = CsrData::try_new(indptr, indices, data, shape);
            assert_validated(result, valid, |e| matches!(e, CoreError::Csr(_)), &label);
        }
    }

    #[test]
    fn array_try_new_accepts_matching_length() {
        assert!(ArrayData::try_new(vec![1.0, 2.0, 3.0, 4.0], Shape::matrix(2, 2)).is_ok());
    }

    #[test]
    fn array_try_new_rejects_length_mismatch() {
        assert!(ArrayData::try_new(vec![1.0, 2.0, 3.0], Shape::matrix(2, 2)).is_err());
    }

    #[test]
    fn interpolant_try_new_differences_a_valid_table_into_slopes_and_refuses_the_rest() {
        /// Knots, values, and the slopes of a valid table or `None` for a refused one.
        type Case = (&'static [f64], &'static [f64], Option<&'static [f64]>);
        let cases: [Case; 10] = [
            (&[0.0, 1.0, 3.0], &[0.0, 10.0, 40.0], Some(&[10.0, 15.0])),
            (&[1.0], &[2.0], Some(&[])),
            // A repeated interior knot is a step, whose zero-width segment has no slope.
            (
                &[0.0, 1.0, 1.0, 2.0],
                &[0.0, 1.0, 3.0, 5.0],
                Some(&[1.0, 0.0, 2.0]),
            ),
            (&[], &[], None),
            (&[0.0, 1.0], &[10.0], None),
            (&[0.0, 2.0, 1.0], &[1.0, 2.0, 3.0], None),
            (&[0.0, f64::NAN], &[1.0, 2.0], None),
            // The end segments extrapolate, so neither may have zero width.
            (&[0.0, 0.0, 1.0], &[1.0, 2.0, 3.0], None),
            (&[0.0, 1.0, 1.0], &[1.0, 2.0, 3.0], None),
            (&[1.0, 1.0], &[1.0, 2.0], None),
        ];
        for (x_data, y_data, slopes) in cases {
            let label = format!("{x_data:?} -> {y_data:?}");
            let result = InterpolantData::try_new(x_data.to_vec(), y_data.to_vec());
            if let (Ok(data), Some(slopes)) = (&result, slopes) {
                assert_eq!(data.slopes, slopes, "{label}");
            }
            assert_validated(result, slopes.is_some(), is_interpolant, &label);
        }
    }

    #[test]
    fn cubic_try_new_accepts_one_tuple_per_segment_and_refuses_the_rest() {
        let segment = [1.0, 2.0, 3.0, 4.0];
        let cases = [
            (vec![0.0, 1.0], vec![segment], true),
            (vec![0.0], vec![], false),
            (vec![0.0, 1.0], vec![segment, segment], false),
            (vec![1.0, 0.0], vec![segment], false),
        ];
        for (breakpoints, coeffs, valid) in cases {
            let label = format!("{breakpoints:?} with {} segment(s)", coeffs.len());
            let result = CubicInterpolantData::try_new(breakpoints, coeffs);
            assert_validated(result, valid, is_interpolant, &label);
        }
    }

    #[test]
    fn nd_try_new_accepts_a_consistent_table_and_refuses_the_rest() {
        let square = || vec![vec![0.0, 1.0], vec![0.0, 1.0]];
        let cases = [
            (square(), vec![1.0, 2.0, 3.0, 4.0], 2, true),
            (square(), vec![1.0, 2.0, 3.0, 4.0], 3, false),
            (square(), vec![1.0, 2.0, 3.0], 2, false),
            (vec![vec![0.0], vec![0.0, 1.0]], vec![1.0, 2.0], 2, false),
        ];
        for (breakpoints, coeffs, order, valid) in cases {
            let label = format!("{breakpoints:?}, {} coeff(s), order {order}", coeffs.len());
            let result = NdInterpolantData::try_new(breakpoints, coeffs, order);
            assert_validated(result, valid, is_interpolant, &label);
        }
    }

    #[test]
    fn test_for_each_child_collects_all_children() {
        let mut arena = Arena::new();
        let a = arena.alloc(Node::Scalar(1.0));
        let b = arena.alloc(Node::Scalar(2.0));
        let add = Node::Add(a, b);

        let mut collected = Vec::new();
        add.for_each_child(|c| collected.push(c));
        assert_eq!(collected, vec![a, b]);

        let leaf = Node::Scalar(2.72);
        let mut leaf_children = Vec::new();
        leaf.for_each_child(|c| leaf_children.push(c));
        assert_eq!(leaf_children, []);

        let concat = Node::Concat(vec![a, b, a]);
        let mut concat_children = Vec::new();
        concat.for_each_child(|c| concat_children.push(c));
        assert_eq!(concat_children, vec![a, b, a]);
    }

    #[test]
    fn map_children_hands_out_children_in_for_each_child_order() {
        let ids: Vec<NodeId> = (0..4u32).map(NodeId::from).collect();
        let nodes = [
            Node::Conditional {
                selector: ids[3],
                branches: vec![ids[0], ids[2]],
            },
            Node::ReduceArgSelect {
                basis: ids[1],
                picker: ids[0],
                is_max: true,
            },
            Node::Sub(ids[2], ids[1]),
            Node::Concat(vec![ids[0], ids[1], ids[2]]),
            Node::Scalar(3.15),
            Node::ZeroVector { len: 4 },
        ];
        for node in nodes {
            assert_eq!(
                node.map_children(|c| c),
                node,
                "the identity map keeps payloads"
            );
            let mut children = Vec::new();
            node.for_each_child(|c| children.push(c));
            let mut handed = Vec::new();
            let mut calls = 0u32;
            let mapped = node.map_children(|c| {
                handed.push(c);
                calls += 1;
                NodeId::from(100 + calls)
            });
            assert_eq!(handed, children, "{node:?}");
            let mut visited = Vec::new();
            mapped.for_each_child(|c| visited.push(c.raw()));
            assert_eq!(visited, (101..=100 + calls).collect::<Vec<_>>());
        }
    }

    #[test]
    fn test_structural_hash_distinguishes_payload_only_differences() {
        use rustc_hash::FxHasher;
        use std::hash::Hasher;

        let s1 = Node::StateVector { start: 0, end: 5 };
        let s2 = Node::StateVector { start: 1, end: 6 };

        let mut h1 = FxHasher::default();
        let mut h2 = FxHasher::default();
        structural_hash(&s1, &mut h1);
        structural_hash(&s2, &mut h2);
        assert_ne!(h1.finish(), h2.finish());

        let zv1 = Node::ZeroVector { len: 3 };
        let zv2 = Node::ZeroVector { len: 7 };
        let mut h3 = FxHasher::default();
        let mut h4 = FxHasher::default();
        structural_hash(&zv1, &mut h3);
        structural_hash(&zv2, &mut h4);
        assert_ne!(h3.finish(), h4.finish());

        let ip1 = Node::InputParameter {
            name: "p1".into(),
            index: 0,
            offset: 0,
            width: 1,
        };
        let ip2 = Node::InputParameter {
            name: "p1".into(),
            index: 1,
            offset: 1,
            width: 1,
        };
        let mut h5 = FxHasher::default();
        let mut h6 = FxHasher::default();
        structural_hash(&ip1, &mut h5);
        structural_hash(&ip2, &mut h6);
        assert_ne!(h5.finish(), h6.finish());
    }

    #[test]
    fn structural_identity_tells_signed_zeros_apart_in_every_payload() {
        let child = NodeId::from(0u32);
        let payloads = |zero: f64| {
            [
                Node::Scalar(zero),
                Node::Array(Box::new(
                    ArrayData::try_new(vec![zero], Shape::vector(1)).expect("valid array"),
                )),
                Node::SparseMatrix(Box::new(
                    CsrData::try_new(vec![0, 1], vec![0], vec![zero], Shape::matrix(1, 1))
                        .expect("valid CSR"),
                )),
                Node::Interpolant1DLinearDeriv {
                    data: Box::new(
                        InterpolantData::try_new(vec![0.0, 1.0], vec![zero, 1.0])
                            .expect("valid table"),
                    ),
                    child,
                },
                Node::Interpolant1DCubic {
                    data: Box::new(
                        CubicInterpolantData::try_new(vec![0.0, 1.0], vec![[zero, 1.0, 0.0, 0.0]])
                            .expect("valid cubic"),
                    ),
                    child,
                },
                Node::InterpolantNdPartial {
                    data: Box::new(
                        NdInterpolantData::try_new(
                            vec![vec![0.0, 1.0], vec![0.0, 1.0]],
                            vec![zero, 1.0, 1.0, 1.0],
                            2,
                        )
                        .expect("valid N-D table"),
                    ),
                    children: vec![child, child],
                    axis: 1,
                },
            ]
        };
        for (positive, negative) in payloads(0.0).iter().zip(&payloads(-0.0)) {
            assert!(structurally_identical(positive, &positive.clone()));
            assert_eq!(positive, negative, "derived `==` equates the two zeros");
            assert!(!structurally_identical(positive, negative), "{positive:?}");
        }
    }

    #[test]
    fn a_sliced_node_refuses_an_inverted_extent() {
        let child = NodeId::from(0u32);
        assert!(matches!(
            Node::state_vector(2, 1),
            Err(CoreError::InvertedExtent {
                node: "StateVector",
                start: 2,
                end: 1
            })
        ));
        assert!(matches!(
            Node::state_vector_dot(2, 1),
            Err(CoreError::InvertedExtent { .. })
        ));
        assert!(matches!(
            Node::index(child, 3, 1),
            Err(CoreError::InvertedExtent { .. })
        ));

        assert!(Node::state_vector(1, 1).is_ok());
        assert!(Node::index(child, 1, 3).is_ok());
    }

    #[test]
    fn a_conditional_needs_a_branch_to_select() {
        let selector = NodeId::from(0u32);
        assert!(matches!(
            Node::conditional(selector, vec![]),
            Err(CoreError::EmptyConditional)
        ));
        assert!(Node::conditional(selector, vec![NodeId::from(1u32)]).is_ok());
    }

    #[test]
    fn an_nd_interpolant_needs_one_child_per_axis() {
        let data =
            NdInterpolantData::try_new(vec![vec![0.0, 1.0], vec![0.0, 1.0]], vec![0.0; 4], 2)
                .expect("a 2-axis multilinear table");
        assert!(matches!(
            Node::interpolant_nd(data.clone(), vec![NodeId::from(0u32)]),
            Err(CoreError::InterpolantNdArity {
                axes: 2,
                children: 1
            })
        ));
        assert!(Node::interpolant_nd(data, vec![NodeId::from(0u32), NodeId::from(1u32)]).is_ok());
    }

    #[test]
    fn flat_cubic_coefficients_must_be_whole_tuples() {
        let breakpoints = vec![0.0, 1.0, 2.0];
        assert!(matches!(
            CubicInterpolantData::try_from_flat(breakpoints.clone(), &[1.0; 7]),
            Err(CoreError::Interpolant(_))
        ));
        let table = CubicInterpolantData::try_from_flat(breakpoints, &[1.0; 8])
            .expect("two segments of four");
        assert_eq!(table.coeffs.len(), 2);
    }
}

#[cfg(all(test, feature = "serialize"))]
mod persisted_tests {
    use super::*;

    fn table() -> InterpolantData {
        InterpolantData::try_new(vec![0.0, 1.0, 3.0], vec![1.0, 2.0, 0.0]).expect("valid table")
    }

    #[test]
    fn a_linear_interpolant_round_trips_with_its_slopes_rederived() {
        let data = table();
        let bytes = bincode::serialize(&data).expect("serialise");
        let back: InterpolantData = bincode::deserialize(&bytes).expect("deserialise");
        assert_eq!(back, data);
        assert_eq!(back.slopes, data.slopes);
    }

    /// Snapshots written before slopes were stored must still load.
    #[test]
    fn the_persisted_form_is_knots_and_values_only() {
        let data = table();
        let bytes = bincode::serialize(&data).expect("serialise");
        let bare = bincode::serialize(&(&data.x_data, &data.y_data)).expect("serialise");
        assert_eq!(bytes, bare);
    }

    #[test]
    fn an_invalid_persisted_interpolant_is_refused() {
        let repeated_last_knot = (vec![0.0, 1.0, 1.0], vec![1.0, 2.0, 3.0]);
        let bytes = bincode::serialize(&repeated_last_knot).expect("serialise");
        assert!(bincode::deserialize::<InterpolantData>(&bytes).is_err());
    }

    fn round_trip<T>(value: &T)
    where
        T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
    {
        let bytes = bincode::serialize(value).expect("serialise");
        assert_eq!(
            &bincode::deserialize::<T>(&bytes).expect("deserialise"),
            value
        );
    }

    #[test]
    fn validated_tables_round_trip_field_for_field() {
        round_trip(
            &CsrData::try_new(vec![0, 1, 1], vec![2], vec![5.0], Shape::matrix(2, 3))
                .expect("valid CSR"),
        );
        round_trip(&ArrayData::try_new(vec![1.0, 2.0], Shape::vector(2)).expect("valid array"));
        round_trip(
            &InterpolantData::try_new(vec![0.0, 1.0, 1.0, 2.0], vec![0.0, 1.0, 3.0, 5.0])
                .expect("valid step"),
        );
        round_trip(
            &CubicInterpolantData::try_new(vec![0.0, 1.0], vec![[1.0, 2.0, 3.0, 4.0]])
                .expect("valid cubic"),
        );
        round_trip(
            &NdInterpolantData::try_new(vec![vec![0.0, 1.0], vec![0.0, 2.0]], vec![0.5; 4], 2)
                .expect("valid N-D table"),
        );
    }

    #[test]
    fn invalid_persisted_tables_are_refused() {
        let csr_tail_past_data = (
            vec![0usize, 2],
            vec![0usize],
            vec![1.0],
            Shape::matrix(1, 1),
        );
        let bytes = bincode::serialize(&csr_tail_past_data).expect("serialise");
        assert!(bincode::deserialize::<CsrData>(&bytes).is_err());

        let array_short = (vec![1.0], Shape::matrix(2, 2));
        let bytes = bincode::serialize(&array_short).expect("serialise");
        assert!(bincode::deserialize::<ArrayData>(&bytes).is_err());

        let cubic_short = (vec![0.0, 1.0, 2.0], vec![[0.0; 4]]);
        let bytes = bincode::serialize(&cubic_short).expect("serialise");
        assert!(bincode::deserialize::<CubicInterpolantData>(&bytes).is_err());

        let nd_order_three = (vec![vec![0.0, 1.0], vec![0.0, 1.0]], vec![0.0; 9], 3u32);
        let bytes = bincode::serialize(&nd_order_three).expect("serialise");
        assert!(bincode::deserialize::<NdInterpolantData>(&bytes).is_err());
    }
}
