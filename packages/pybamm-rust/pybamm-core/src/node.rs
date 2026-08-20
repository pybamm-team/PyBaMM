//! The expression vocabulary Python hands over.
//!
//! [`Node`] is one DAG node: an operator referencing its children by
//! [`NodeId`], a leaf reading state or parameters, or a literal
//! carrying its own data (dense arrays, CSR matrices, 1-D and N-D interpolant
//! tables). Everything downstream matches on this enum, from simplification
//! through differentiation to lowering, so support for a new `PyBaMM` operator
//! starts with a variant here.
//!
//! Nodes are shape-carrying but not shape-checked on construction;
//! [`first_invalid`](crate::first_invalid) and
//! [`first_unsupported`](crate::first_unsupported) report what lowering will
//! reject.

use crate::arena::NodeId;
use crate::error::CoreError;

/// Declared `rows × cols` of a literal.
///
/// Carried for matmul dimension checks and for reporting. Elsewhere evaluation
/// works on the flat element count, so a `1 × n` and an `n × 1` array are alike.
/// As a [`MatMul`](Node::MatMul) left operand they are not, since `DenseMatMul`
/// reads `rows`/`cols` at evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Shape {
    pub rows: usize,
    pub cols: usize,
}

/// Knots and values of a 1-D linear interpolant.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct InterpolantData {
    pub(crate) x_data: Vec<f64>,
    pub(crate) y_data: Vec<f64>,
}

/// Breakpoints and per-interval coefficients of a 1-D cubic interpolant.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CubicInterpolantData {
    /// Interval breakpoints, length nseg + 1.
    pub(crate) breakpoints: Vec<f64>,
    /// Per-interval power-basis coeffs `[c0, c1, c2, c3]`, length nseg.
    /// `p(dx) = c0 + c1*dx + c2*dx^2 + c3*dx^3`, `dx = x - breakpoints[i]`.
    pub(crate) coeffs: Vec<[f64; 4]>,
}

/// Per-axis breakpoints and per-cell coefficients of a 2-D or 3-D
/// tensor-product interpolant.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct NdInterpolantData {
    /// Per-axis knot vectors (2 or 3 axes), each of length `nseg_a + 1`.
    pub(crate) breakpoints: Vec<Vec<f64>>,
    /// Flat per-cell power-basis tensors: cell-major (axis-0 segment
    /// slowest), `order^ndim` coeffs per cell (axis-0 power slowest,
    /// ascending powers): `p(dx) = Σ c[a0,..] · Π dx_i^a_i`.
    pub(crate) coeffs: Vec<f64>,
    /// Per-axis polynomial order: 2 = multilinear, 4 = tensor cubic.
    pub(crate) order: u32,
}

impl Shape {
    /// `1 × 1`.
    pub const fn scalar() -> Self {
        Self { rows: 1, cols: 1 }
    }

    /// `len × 1`, the column-vector orientation `PyBaMM` discretises into.
    pub const fn vector(len: usize) -> Self {
        Self { rows: len, cols: 1 }
    }

    pub const fn matrix(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }
}

/// A constant sparse matrix, in the CSR form `PyBaMM`'s discretisation produces.
///
/// Spatial operators arrive as matrices, so most of a discretised model's constant
/// data is one of these. CSR is kept as given and converted at the boundaries that
/// need column-major, namely the solver's Jacobian and mass sparsity.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CsrData {
    pub(crate) indptr: Vec<usize>,
    pub(crate) indices: Vec<usize>,
    pub(crate) data: Vec<f64>,
    pub(crate) shape: Shape,
}

/// A constant dense array, stored row-major and flat.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ArrayData {
    pub(crate) data: Vec<f64>,
    pub(crate) shape: Shape,
}

impl CsrData {
    /// Build a CSR matrix, validating its structural invariants: `indptr`
    /// length (`rows + 1`), leading zero, non-decreasing offsets whose tail
    /// equals the nnz, matching `indices`/`data` lengths, and in-range column
    /// indices. Internal transform passes that already hold the invariant
    /// still construct via the (crate-visible) struct literal.
    pub fn try_new(
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<f64>,
        shape: Shape,
    ) -> Result<Self, CoreError> {
        let (rows, cols) = (shape.rows, shape.cols);
        if indptr.len() != rows + 1 {
            return Err(CoreError::Csr(format!(
                "indptr length {} must equal rows + 1 = {}",
                indptr.len(),
                rows + 1
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
        let nnz = indptr[rows]; // in bounds: len == rows + 1
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
    /// Build a 1D linear interpolation table, validating that the grid is
    /// non-empty, `x`/`y` lengths match, and the knots are finite and strictly
    /// increasing (the segment lookup and slope division rely on both).
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
        if x_data.windows(2).any(|w| w[1] <= w[0]) {
            return Err(CoreError::Interpolant(
                "x_data must be strictly increasing".to_string(),
            ));
        }
        Ok(Self { x_data, y_data })
    }
}

impl CubicInterpolantData {
    /// Build a 1D cubic interpolation table, validating at least two finite,
    /// strictly increasing breakpoints and one coefficient tuple per segment.
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
    /// Build an N-D (2 or 3 axis) tensor interpolation table, validating the
    /// axis count, order (2 or 4), per-axis finite strictly-increasing knots
    /// (at least two each), and the `ncells * order^ndim` coefficient count.
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
            ncells *= knots.len() - 1;
        }
        let order_usize = order as usize;
        let per_cell = match ndim {
            2 => order_usize * order_usize,
            3 => order_usize * order_usize * order_usize,
            _ => unreachable!("ndim validated to 2..=3"),
        };
        let expected = ncells * per_cell;
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
/// Children are [`NodeId`]s into the arena that owns this node, so a node is
/// meaningless on its own and cheap to share. Values are `f64` vectors: the
/// arithmetic and unary variants are element-wise with a scalar operand
/// broadcasting against a vector, matching [`BinaryOp`](crate::BinaryOp) and
/// [`UnaryOp`](crate::UnaryOp), which they lower to one-for-one.
///
/// Variants marked internal are produced by differentiation rather than by
/// Python, and a few carry `Box`ed payloads to keep the enum small enough that a
/// DAG of them stays compact.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum Node {
    // Leaf nodes
    Scalar(f64),
    Array(Box<ArrayData>),
    /// Zero vector of specified length (first-class, not dense Array)
    ZeroVector {
        len: usize,
    },
    SparseMatrix(Box<CsrData>),
    /// Reads `y[start..end]`, a half-open range in the solver's global state
    /// vector rather than any per-equation numbering.
    StateVector {
        start: usize,
        end: usize,
    },
    /// Reads `y'[start..end]` from the state derivative, in the same index space
    /// as [`StateVector`](Self::StateVector). Only a residual formulation
    /// supplies it.
    StateVectorDot {
        start: usize,
        end: usize,
    },
    InputParameter {
        name: String,
        /// Registration order among distinct names (0-based); used by
        /// `TangentParameter`/sensitivity indexing, unaffected by width.
        index: usize,
        /// Cumulative offset into the packed `p` values array.
        offset: usize,
        /// Number of packed values this parameter occupies (>1 for vector inputs).
        width: usize,
    },
    Time,

    // Binary operations
    Add(NodeId, NodeId),
    Sub(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Div(NodeId, NodeId),
    Pow(NodeId, NodeId),
    /// Matrix-vector product. The left child must be a constant
    /// [`SparseMatrix`](Self::SparseMatrix) or [`Array`](Self::Array), since there
    /// is no general matrix-matrix product, and its column count must equal the
    /// right child's width.
    MatMul(NodeId, NodeId),
    Minimum(NodeId, NodeId),
    Maximum(NodeId, NodeId),
    Modulo(NodeId, NodeId),
    Hypot(NodeId, NodeId),
    EqualHeaviside(NodeId, NodeId),
    NotEqualHeaviside(NodeId, NodeId),
    Equality(NodeId, NodeId),

    // Structural nodes
    /// Half-open slice `child[start..end]` of the child's own value, unrelated to
    /// state-vector indices.
    Index {
        child: NodeId,
        start: usize,
        end: usize,
    },
    /// Children joined end to end; the order here is the layout of the result, and
    /// for a model's right-hand side it is the equation ordering the solver sees.
    Concat(Vec<NodeId>),

    // Interpolation (boxed to reduce enum size)
    Interpolant1DLinear {
        data: Box<InterpolantData>,
        child: NodeId,
    },
    /// 1D cubic/pchip interpolation (piecewise cubic, power-basis coeffs).
    Interpolant1DCubic {
        data: Box<CubicInterpolantData>,
        child: NodeId,
    },
    /// N-D (2D/3D) tensor-product interpolation: one child per axis,
    /// evaluated element-wise over equal-length children.
    InterpolantNd {
        data: Box<NdInterpolantData>,
        children: Vec<NodeId>,
    },

    // Unary operations
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

    /// Internal: `basis[k]` where `k` is the first-occurrence argmax
    /// (`is_max = true`) or argmin of `picker`. Created by `differentiate`
    /// as the subgradient of `MaxReduce`/`MinReduce`; never built from Python.
    ReduceArgSelect {
        basis: NodeId,
        picker: NodeId,
        is_max: bool,
    },

    // Conditional branching
    /// Selects `branches[i]` when `selector` falls in the open window
    /// `(i + 0.5, i + 1.5)`, so 2.0 and 1.6 both pick `branches[1]` but an exact
    /// half-way value matches nothing and evaluates to zeros. Branches are ordinary
    /// subexpressions, and the compiler arranges for only the selected one to be
    /// evaluated where it can prove ownership.
    Conditional {
        selector: NodeId,
        branches: Vec<NodeId>,
    },

    // Tangent nodes for forward-mode AD
    /// Reads a slice of the tangent state vector (corresponds to `LoadTangentState` in `TypedIr`)
    TangentStateVector {
        start: usize,
        end: usize,
    },
    /// Reads a single tangent parameter (corresponds to `LoadTangentParameter` in `TypedIr`)
    TangentParameter {
        index: usize,
    },
    /// Derivative of linear interpolation (used during symbolic differentiation)
    Interpolant1DLinearDeriv {
        slopes: Box<[f64]>,
        x_data: Box<[f64]>,
        child: NodeId,
    },
    /// Derivative of 1D cubic/pchip interpolation (used during AD).
    Interpolant1DCubicDeriv {
        data: Box<CubicInterpolantData>,
        child: NodeId,
    },
    /// Partial derivative of N-D interpolation along `axis` (used during AD).
    InterpolantNdPartial {
        data: Box<NdInterpolantData>,
        children: Vec<NodeId>,
        axis: u32,
    },
}

impl Node {
    /// Visit each child `NodeId` of this node, allocation-free.
    pub fn for_each_child<F: FnMut(NodeId)>(&self, mut f: F) {
        match self {
            // Leaves have no children
            Self::Scalar(_)
            | Self::Array(_)
            | Self::ZeroVector { .. }
            | Self::SparseMatrix(_)
            | Self::StateVector { .. }
            | Self::StateVectorDot { .. }
            | Self::InputParameter { .. }
            | Self::Time
            | Self::TangentStateVector { .. }
            | Self::TangentParameter { .. } => {},

            // Binary operations
            Self::Add(l, r)
            | Self::Sub(l, r)
            | Self::Mul(l, r)
            | Self::Div(l, r)
            | Self::Pow(l, r)
            | Self::MatMul(l, r)
            | Self::Minimum(l, r)
            | Self::Maximum(l, r)
            | Self::Modulo(l, r)
            | Self::Hypot(l, r)
            | Self::EqualHeaviside(l, r)
            | Self::NotEqualHeaviside(l, r)
            | Self::Equality(l, r) => {
                f(*l);
                f(*r);
            },

            // Unary operations
            Self::Neg(c)
            | Self::Abs(c)
            | Self::Sqrt(c)
            | Self::Exp(c)
            | Self::Log(c)
            | Self::Sin(c)
            | Self::Cos(c)
            | Self::Tanh(c)
            | Self::Sinh(c)
            | Self::Cosh(c)
            | Self::Arcsinh(c)
            | Self::Arctan(c)
            | Self::Erf(c)
            | Self::Sign(c)
            | Self::Floor(c)
            | Self::Ceiling(c)
            | Self::MaxReduce(c)
            | Self::MinReduce(c) => {
                f(*c);
            },

            // Structural / Interpolation with single child
            Self::Index { child, .. }
            | Self::Interpolant1DLinear { child, .. }
            | Self::Interpolant1DLinearDeriv { child, .. }
            | Self::Interpolant1DCubic { child, .. }
            | Self::Interpolant1DCubicDeriv { child, .. } => {
                f(*child);
            },
            Self::Concat(children)
            | Self::InterpolantNd { children, .. }
            | Self::InterpolantNdPartial { children, .. } => {
                for &c in children {
                    f(c);
                }
            },

            Self::Conditional { selector, branches } => {
                f(*selector);
                for &b in branches {
                    f(b);
                }
            },

            Self::ReduceArgSelect { basis, picker, .. } => {
                f(*basis);
                f(*picker);
            },
        }
    }

    /// Returns a new `Node` with a transformation applied to each child `NodeId`.
    ///
    /// Leaves (scalars, arrays, time, etc.) are cloned unchanged. For nodes with
    /// children (binary ops, unary ops, structural, interpolants, conditionals),
    /// the closure `f` is called on every child `NodeId`, and a new node is
    /// constructed with the transformed children.
    ///
    /// Common uses:
    /// - Renumbering node IDs after deduplication
    /// - Substituting nodes (e.g. replacing `InputParameter` with constants)
    /// - Pruning unused branches (replacing with a zero node)
    #[must_use]
    pub fn map_children<F: FnMut(NodeId) -> NodeId>(&self, mut f: F) -> Self {
        match self {
            // Leaves: clone unchanged.
            Self::Scalar(_)
            | Self::Array(_)
            | Self::ZeroVector { .. }
            | Self::SparseMatrix(_)
            | Self::StateVector { .. }
            | Self::StateVectorDot { .. }
            | Self::InputParameter { .. }
            | Self::Time
            | Self::TangentStateVector { .. }
            | Self::TangentParameter { .. } => self.clone(),

            // Binary operations
            Self::Add(l, r) => Self::Add(f(*l), f(*r)),
            Self::Sub(l, r) => Self::Sub(f(*l), f(*r)),
            Self::Mul(l, r) => Self::Mul(f(*l), f(*r)),
            Self::Div(l, r) => Self::Div(f(*l), f(*r)),
            Self::Pow(l, r) => Self::Pow(f(*l), f(*r)),
            Self::MatMul(l, r) => Self::MatMul(f(*l), f(*r)),
            Self::Minimum(l, r) => Self::Minimum(f(*l), f(*r)),
            Self::Maximum(l, r) => Self::Maximum(f(*l), f(*r)),
            Self::Modulo(l, r) => Self::Modulo(f(*l), f(*r)),
            Self::Hypot(l, r) => Self::Hypot(f(*l), f(*r)),
            Self::EqualHeaviside(l, r) => Self::EqualHeaviside(f(*l), f(*r)),
            Self::NotEqualHeaviside(l, r) => Self::NotEqualHeaviside(f(*l), f(*r)),
            Self::Equality(l, r) => Self::Equality(f(*l), f(*r)),

            // Unary operations
            Self::Neg(c) => Self::Neg(f(*c)),
            Self::Abs(c) => Self::Abs(f(*c)),
            Self::Sqrt(c) => Self::Sqrt(f(*c)),
            Self::Exp(c) => Self::Exp(f(*c)),
            Self::Log(c) => Self::Log(f(*c)),
            Self::Sin(c) => Self::Sin(f(*c)),
            Self::Cos(c) => Self::Cos(f(*c)),
            Self::Tanh(c) => Self::Tanh(f(*c)),
            Self::Sinh(c) => Self::Sinh(f(*c)),
            Self::Cosh(c) => Self::Cosh(f(*c)),
            Self::Arcsinh(c) => Self::Arcsinh(f(*c)),
            Self::Arctan(c) => Self::Arctan(f(*c)),
            Self::Erf(c) => Self::Erf(f(*c)),
            Self::Sign(c) => Self::Sign(f(*c)),
            Self::Floor(c) => Self::Floor(f(*c)),
            Self::Ceiling(c) => Self::Ceiling(f(*c)),
            Self::MaxReduce(c) => Self::MaxReduce(f(*c)),
            Self::MinReduce(c) => Self::MinReduce(f(*c)),

            // Structural nodes
            Self::Index { child, start, end } => Self::Index {
                child: f(*child),
                start: *start,
                end: *end,
            },
            Self::Concat(children) => Self::Concat(children.iter().map(|&c| f(c)).collect()),

            // Interpolation nodes
            Self::Interpolant1DLinear { data, child } => Self::Interpolant1DLinear {
                data: data.clone(),
                child: f(*child),
            },
            Self::Interpolant1DLinearDeriv {
                slopes,
                x_data,
                child,
            } => Self::Interpolant1DLinearDeriv {
                slopes: slopes.clone(),
                x_data: x_data.clone(),
                child: f(*child),
            },
            Self::Interpolant1DCubic { data, child } => Self::Interpolant1DCubic {
                data: data.clone(),
                child: f(*child),
            },
            Self::Interpolant1DCubicDeriv { data, child } => Self::Interpolant1DCubicDeriv {
                data: data.clone(),
                child: f(*child),
            },
            Self::InterpolantNd { data, children } => Self::InterpolantNd {
                data: data.clone(),
                children: children.iter().map(|&c| f(c)).collect(),
            },
            Self::InterpolantNdPartial {
                data,
                children,
                axis,
            } => Self::InterpolantNdPartial {
                data: data.clone(),
                children: children.iter().map(|&c| f(c)).collect(),
                axis: *axis,
            },

            Self::Conditional { selector, branches } => Self::Conditional {
                selector: f(*selector),
                branches: branches.iter().map(|&b| f(b)).collect(),
            },

            Self::ReduceArgSelect {
                basis,
                picker,
                is_max,
            } => Self::ReduceArgSelect {
                basis: f(*basis),
                picker: f(*picker),
                is_max: *is_max,
            },
        }
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

/// Hash a node by structure, for the CSE key that finds duplicate subexpressions.
///
/// Children are hashed through `child_remap`, so two nodes collide only when their
/// operands are already known to be the same value, which is what lets CSE work
/// bottom-up in one pass. Floats are hashed by bit pattern, which keeps `0.0` and
/// `-0.0` distinct rather than merging expressions that differ in sign of zero.
pub fn structural_hash<H: std::hash::Hasher>(
    node: &Node,
    hasher: &mut H,
    mut child_remap: impl FnMut(NodeId) -> NodeId,
) {
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
        Node::Interpolant1DLinear { data, .. } => {
            for v in &data.x_data {
                v.to_bits().hash(hasher);
            }
            for v in &data.y_data {
                v.to_bits().hash(hasher);
            }
        },
        Node::Interpolant1DLinearDeriv { slopes, x_data, .. } => {
            for v in slopes {
                v.to_bits().hash(hasher);
            }
            for v in x_data {
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
    node.for_each_child(|c| {
        use std::hash::Hash;
        child_remap(c).hash(hasher);
    });
}

const _: () = {
    assert!(size_of::<Node>() <= 48);
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;

    // ---- F3: validated data-structure constructors ----

    #[test]
    fn csr_try_new_accepts_valid_matrix() {
        let m = CsrData::try_new(
            vec![0, 1, 2],
            vec![0, 1],
            vec![1.0, 2.0],
            Shape::matrix(2, 2),
        );
        assert!(m.is_ok());
    }

    #[test]
    fn csr_try_new_rejects_wrong_indptr_length() {
        let m = CsrData::try_new(vec![0, 2], vec![0, 1], vec![1.0, 2.0], Shape::matrix(2, 2));
        assert!(m.is_err());
    }

    #[test]
    fn csr_try_new_rejects_non_monotonic_indptr() {
        let m = CsrData::try_new(
            vec![0, 2, 1],
            vec![0, 1],
            vec![1.0, 2.0],
            Shape::matrix(2, 2),
        );
        assert!(m.is_err());
    }

    #[test]
    fn csr_try_new_rejects_out_of_range_column() {
        let m = CsrData::try_new(
            vec![0, 1, 2],
            vec![0, 5],
            vec![1.0, 2.0],
            Shape::matrix(2, 2),
        );
        assert!(m.is_err());
    }

    #[test]
    fn csr_try_new_rejects_indices_data_length_mismatch() {
        let m = CsrData::try_new(vec![0, 1, 2], vec![0, 1], vec![1.0], Shape::matrix(2, 2));
        assert!(m.is_err());
    }

    #[test]
    fn csr_try_new_rejects_indptr_tail_mismatch() {
        let m = CsrData::try_new(vec![0, 1, 2], vec![0], vec![1.0], Shape::matrix(2, 2));
        assert!(m.is_err());
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
    fn interpolant_try_new_accepts_increasing_grid() {
        assert!(InterpolantData::try_new(vec![0.0, 1.0, 2.0], vec![10.0, 20.0, 30.0]).is_ok());
    }

    #[test]
    fn interpolant_try_new_rejects_empty_grid() {
        assert!(InterpolantData::try_new(vec![], vec![]).is_err());
    }

    #[test]
    fn interpolant_try_new_rejects_length_mismatch() {
        assert!(InterpolantData::try_new(vec![0.0, 1.0], vec![10.0]).is_err());
    }

    #[test]
    fn interpolant_try_new_rejects_non_increasing_knots() {
        assert!(InterpolantData::try_new(vec![0.0, 2.0, 1.0], vec![1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn interpolant_try_new_rejects_non_finite_knot() {
        assert!(InterpolantData::try_new(vec![0.0, f64::NAN], vec![1.0, 2.0]).is_err());
    }

    #[test]
    fn cubic_try_new_accepts_valid() {
        assert!(CubicInterpolantData::try_new(vec![0.0, 1.0], vec![[1.0, 2.0, 3.0, 4.0]]).is_ok());
    }

    #[test]
    fn cubic_try_new_rejects_too_few_breakpoints() {
        assert!(CubicInterpolantData::try_new(vec![0.0], vec![]).is_err());
    }

    #[test]
    fn cubic_try_new_rejects_coeff_count_mismatch() {
        let c = CubicInterpolantData::try_new(
            vec![0.0, 1.0],
            vec![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        );
        assert!(c.is_err());
    }

    #[test]
    fn cubic_try_new_rejects_non_increasing_breakpoints() {
        assert!(CubicInterpolantData::try_new(vec![1.0, 0.0], vec![[1.0, 2.0, 3.0, 4.0]]).is_err());
    }

    #[test]
    fn nd_try_new_accepts_valid_2d_linear() {
        let nd = NdInterpolantData::try_new(
            vec![vec![0.0, 1.0], vec![0.0, 1.0]],
            vec![1.0, 2.0, 3.0, 4.0],
            2,
        );
        assert!(nd.is_ok());
    }

    #[test]
    fn nd_try_new_rejects_bad_order() {
        let nd = NdInterpolantData::try_new(
            vec![vec![0.0, 1.0], vec![0.0, 1.0]],
            vec![1.0, 2.0, 3.0, 4.0],
            3,
        );
        assert!(nd.is_err());
    }

    #[test]
    fn nd_try_new_rejects_coeff_count_mismatch() {
        let nd = NdInterpolantData::try_new(
            vec![vec![0.0, 1.0], vec![0.0, 1.0]],
            vec![1.0, 2.0, 3.0],
            2,
        );
        assert!(nd.is_err());
    }

    #[test]
    fn nd_try_new_rejects_axis_with_too_few_knots() {
        let nd = NdInterpolantData::try_new(vec![vec![0.0], vec![0.0, 1.0]], vec![1.0, 2.0], 2);
        assert!(nd.is_err());
    }

    #[test]
    fn test_tangent_state_vector_node() {
        let node = Node::TangentStateVector { start: 0, end: 3 };
        match node {
            Node::TangentStateVector { start, end } => {
                assert_eq!(start, 0);
                assert_eq!(end, 3);
            },
            _ => panic!("Expected TangentStateVector"),
        }
    }

    #[test]
    fn test_tangent_parameter_node() {
        let node = Node::TangentParameter { index: 2 };
        match node {
            Node::TangentParameter { index } => {
                assert_eq!(index, 2);
            },
            _ => panic!("Expected TangentParameter"),
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
        assert!(leaf_children.is_empty());

        let concat = Node::Concat(vec![a, b, a]);
        let mut concat_children = Vec::new();
        concat.for_each_child(|c| concat_children.push(c));
        assert_eq!(concat_children, vec![a, b, a]);
    }

    #[test]
    fn test_map_children_remaps_binary() {
        let id_a: NodeId = 0u32.into();
        let id_b: NodeId = 1u32.into();
        let id_x: NodeId = 10u32.into();

        let add = Node::Add(id_a, id_b);
        let mapped = add.map_children(|c| if c == id_a { id_x } else { c });
        match mapped {
            Node::Add(l, r) => {
                assert_eq!(l, id_x);
                assert_eq!(r, id_b);
            },
            _ => panic!("expected Add"),
        }
    }

    #[test]
    fn test_map_children_preserves_leaves() {
        let s = Node::Scalar(3.15);
        let mapped = s.map_children(|c| c);
        assert_eq!(s, mapped);

        let zv = Node::ZeroVector { len: 4 };
        let mapped_zv = zv.map_children(|c| c);
        assert_eq!(zv, mapped_zv);
    }

    #[test]
    fn test_map_children_preserves_concat_arity_and_order() {
        let a: NodeId = 0u32.into();
        let b: NodeId = 1u32.into();
        let c: NodeId = 2u32.into();
        let concat = Node::Concat(vec![a, b, c]);
        let mapped = concat.map_children(|x| NodeId::from(x.raw() + 100));
        match mapped {
            Node::Concat(children) => {
                assert_eq!(children, vec![100u32.into(), 101u32.into(), 102u32.into()]);
            },
            _ => panic!("expected Concat"),
        }
    }

    #[test]
    fn test_interpolant1d_linear_deriv_node() {
        let node = Node::Interpolant1DLinearDeriv {
            slopes: vec![10.0, 10.0].into_boxed_slice(),
            x_data: vec![0.0, 1.0, 2.0].into_boxed_slice(),
            child: {
                let mut arena = Arena::new();
                arena.alloc(Node::Scalar(1.5))
            },
        };
        match node {
            Node::Interpolant1DLinearDeriv { slopes, x_data, .. } => {
                assert_eq!(slopes.len(), 2);
                assert_eq!(x_data.len(), 3);
            },
            _ => panic!("Expected Interpolant1DLinearDeriv"),
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
        structural_hash(&s1, &mut h1, |c| c);
        structural_hash(&s2, &mut h2, |c| c);
        assert_ne!(h1.finish(), h2.finish());

        let zv1 = Node::ZeroVector { len: 3 };
        let zv2 = Node::ZeroVector { len: 7 };
        let mut h3 = FxHasher::default();
        let mut h4 = FxHasher::default();
        structural_hash(&zv1, &mut h3, |c| c);
        structural_hash(&zv2, &mut h4, |c| c);
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
        structural_hash(&ip1, &mut h5, |c| c);
        structural_hash(&ip2, &mut h6, |c| c);
        assert_ne!(h5.finish(), h6.finish());
    }
}
