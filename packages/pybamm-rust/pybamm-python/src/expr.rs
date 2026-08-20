// PyO3 bindings require specific argument types that clippy flags incorrectly
#![allow(clippy::needless_pass_by_value)]

use std::collections::HashMap;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use pybamm_core::{
    Arena, ArrayData, CompiledExpr, CubicInterpolantData, InterpolantData, NdInterpolantData, Node,
    NodeId, Shape,
};

use crate::errors::core_err_to_py;

/// Validate a half-open `[start, end)` extent, returning a `ValueError` when
/// inverted so the `end - start` size computation cannot wrap in release.
fn check_range(start: usize, end: usize, what: &str) -> PyResult<()> {
    if start > end {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{what}: start ({start}) must not exceed end ({end})"
        )));
    }
    Ok(())
}

// `module` is required so pickle can locate the class as `pybamm.rust.ExprGraph`
// instead of the pyo3 default `builtins.ExprGraph`, which pickle cannot import.
#[pyclass(module = "pybamm.rust")]
#[derive(Debug)]
pub struct ExprGraph {
    arena: Arena,
    input_map: HashMap<String, usize>,
    /// Packed width per registered input, indexed by registration order
    /// (parallel to `input_map`'s values).
    input_widths: Vec<usize>,
}

/// Serialized form of every `ExprGraph` field, used by the pickle protocol.
#[cfg(feature = "serialize")]
#[derive(serde::Serialize, serde::Deserialize)]
struct ExprGraphState {
    arena: Arena,
    input_map: HashMap<String, usize>,
    input_widths: Vec<usize>,
}

#[pyclass(frozen, module = "pybamm.rust")]
pub struct Expr {
    node_id: NodeId,
    graph: Py<ExprGraph>,
}

impl std::fmt::Debug for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Expr")
            .field("node_id", &self.node_id)
            .finish_non_exhaustive()
    }
}

impl Expr {
    /// Node id, validated to belong to `graph`. Guards against using an `Expr`
    /// from one `ExprGraph` in another, which would index the wrong arena.
    pub(crate) fn node_id_in(&self, graph: &Py<ExprGraph>) -> PyResult<NodeId> {
        if self.graph.as_ptr() == graph.as_ptr() {
            Ok(self.node_id)
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "Expr belongs to a different ExprGraph than the graph being built into",
            ))
        }
    }
}

impl ExprGraph {
    /// Get a reference to the arena (crate-internal).
    pub(crate) const fn arena(&self) -> &Arena {
        &self.arena
    }

    /// Mutable arena reference for build-time node allocation (crate-internal).
    pub(crate) const fn arena_mut(&mut self) -> &mut Arena {
        &mut self.arena
    }

    /// Input names ordered by registration index (crate-internal).
    pub(crate) fn input_names(&self) -> Vec<String> {
        let mut names = vec![String::new(); self.input_map.len()];
        for (name, &idx) in &self.input_map {
            names[idx].clone_from(name);
        }
        names
    }

    /// Packed width per input, ordered by registration index (crate-internal).
    pub(crate) fn input_widths(&self) -> Vec<usize> {
        self.input_widths.clone()
    }
}

/// Validate that the graph rooted at `root` can be lowered safely.
///
/// Called at every binding entry point so unsupported nodes and invalid shape
/// relationships surface as catchable Python errors instead of FFI panics.
pub fn check_supported(arena: &Arena, root: NodeId) -> PyResult<()> {
    if let Some(msg) = pybamm_core::first_unsupported(arena, root) {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "Rust conversion does not support this expression: {msg}. Use a \
             CasADi-backed model (set `model.convert_to_format = 'casadi'`)."
        )));
    }
    if let Some(msg) = pybamm_core::first_invalid(arena, root) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid Rust expression graph: {msg}"
        )));
    }
    Ok(())
}

#[pymethods]
impl ExprGraph {
    #[new]
    fn new() -> Self {
        Self {
            arena: Arena::new(),
            input_map: HashMap::new(),
            input_widths: Vec::new(),
        }
    }

    fn scalar(slf: Py<Self>, py: Python<'_>, value: f64) -> Expr {
        let id = slf.borrow_mut(py).arena.alloc(Node::Scalar(value));
        Expr {
            node_id: id,
            graph: slf,
        }
    }

    fn time(slf: Py<Self>, py: Python<'_>) -> Expr {
        let id = slf.borrow_mut(py).arena.alloc(Node::Time);
        Expr {
            node_id: id,
            graph: slf,
        }
    }

    fn state_vector(slf: Py<Self>, py: Python<'_>, start: usize, end: usize) -> PyResult<Expr> {
        check_range(start, end, "state_vector")?;
        let id = slf
            .borrow_mut(py)
            .arena
            .alloc(Node::StateVector { start, end });
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn state_vector_dot(slf: Py<Self>, py: Python<'_>, start: usize, end: usize) -> PyResult<Expr> {
        check_range(start, end, "state_vector_dot")?;
        let id = slf
            .borrow_mut(py)
            .arena
            .alloc(Node::StateVectorDot { start, end });
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    /// Total packed width of every input parameter registered in this graph
    /// (sum of `input_widths`, not the count of distinct names), this is
    /// the length the FFI/solver boundary expects the stacked `p` array to
    /// have, and what `n_inputs=` on `CompiledModel.from_expr` must receive.
    fn n_inputs(&self) -> usize {
        self.input_widths.iter().sum()
    }

    /// Number of nodes in the expression arena (introspection).
    #[getter]
    const fn n_nodes(&self) -> usize {
        self.arena.len()
    }

    /// Register (or re-look-up) a named input parameter.
    ///
    /// `width` is the number of packed values the parameter occupies (>1 for
    /// vector-valued inputs). Re-registering an existing name must repeat the
    /// same width, mismatches would silently corrupt every other parameter's
    /// offset into the packed `p` array.
    #[pyo3(signature = (name, width = 1))]
    fn input_parameter(slf: Py<Self>, py: Python<'_>, name: &str, width: usize) -> PyResult<Expr> {
        let mut graph = slf.borrow_mut(py);
        let index = if let Some(&idx) = graph.input_map.get(name) {
            let existing_width = graph.input_widths[idx];
            if existing_width != width {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "input parameter '{name}' re-registered with width {width}, \
                     previously registered with width {existing_width}"
                )));
            }
            idx
        } else {
            let idx = graph.input_map.len();
            graph.input_map.insert(name.to_string(), idx);
            graph.input_widths.push(width);
            idx
        };
        let offset: usize = graph.input_widths[..index].iter().sum();
        let id = graph.arena.alloc(Node::InputParameter {
            name: name.to_string(),
            index,
            offset,
            width,
        });
        drop(graph);
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn array(slf: Py<Self>, py: Python<'_>, data: PyReadonlyArray1<'_, f64>) -> PyResult<Expr> {
        let vec = data.as_slice()?.to_vec();
        let len = vec.len();
        let array = ArrayData::try_new(vec, Shape::vector(len)).map_err(core_err_to_py)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Array(Box::new(array)));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    /// Dense matrix constant from row-major data (rows × cols).
    fn dense_matrix(
        slf: Py<Self>,
        py: Python<'_>,
        data: PyReadonlyArray1<'_, f64>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Expr> {
        let vec = data.as_slice()?.to_vec();
        let array = ArrayData::try_new(vec, Shape::matrix(rows, cols)).map_err(core_err_to_py)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Array(Box::new(array)));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn add(slf: Py<Self>, py: Python<'_>, a: &Expr, b: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let b_id = b.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Add(a_id, b_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn sub(slf: Py<Self>, py: Python<'_>, a: &Expr, b: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let b_id = b.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Sub(a_id, b_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn mul(slf: Py<Self>, py: Python<'_>, a: &Expr, b: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let b_id = b.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Mul(a_id, b_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn div(slf: Py<Self>, py: Python<'_>, a: &Expr, b: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let b_id = b.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Div(a_id, b_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn neg(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Neg(a_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn abs(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Abs(a_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn pow(slf: Py<Self>, py: Python<'_>, a: &Expr, b: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let b_id = b.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Pow(a_id, b_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn sqrt(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Sqrt(a_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn exp(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Exp(a_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn log(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Log(a_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn sin(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Sin(a_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn cos(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Cos(a_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn tanh(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Tanh(a_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn sinh(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Sinh(a_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn cosh(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Cosh(a_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn arcsinh(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Arcsinh(a_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn arctan(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Arctan(a_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn erf(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Erf(a_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn sign(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Sign(a_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn floor(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Floor(a_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn ceiling(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Ceiling(a_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn max_reduce(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::MaxReduce(a_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn min_reduce(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::MinReduce(a_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn interpolant_1d_linear(
        slf: Py<Self>,
        py: Python<'_>,
        x_data: Vec<f64>,
        y_data: Vec<f64>,
        child: &Expr,
    ) -> PyResult<Expr> {
        let child_id = child.node_id_in(&slf)?;
        let data = InterpolantData::try_new(x_data, y_data).map_err(core_err_to_py)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Interpolant1DLinear {
            data: Box::new(data),
            child: child_id,
        });
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    /// Build a 1D cubic/pchip interpolant. `coeffs` is flat row-major
    /// `[c0,c1,c2,c3, c0,c1,c2,c3, ...]`, length `4 * (breakpoints.len() - 1)`.
    /// `breakpoints` must have at least 2 entries (one segment).
    fn interpolant_1d_cubic(
        slf: Py<Self>,
        py: Python<'_>,
        breakpoints: Vec<f64>,
        coeffs: Vec<f64>,
        child: &Expr,
    ) -> PyResult<Expr> {
        // Flat coeffs are per-segment power-basis groups of 4; guard the grouping
        // before chunking (which would silently drop a non-multiple-of-4 tail).
        if !coeffs.len().is_multiple_of(4) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "interpolant_1d_cubic: coeffs length must be a multiple of 4, got {}",
                coeffs.len()
            )));
        }
        let child_id = child.node_id_in(&slf)?;
        let coeffs: Vec<[f64; 4]> = coeffs
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect();
        let data = CubicInterpolantData::try_new(breakpoints, coeffs).map_err(core_err_to_py)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Interpolant1DCubic {
            data: Box::new(data),
            child: child_id,
        });
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    /// Build an N-D (2 or 3 axis) tensor-product interpolant. `breakpoints`
    /// holds the per-axis knot vectors. `coeffs` is flat: cell-major (axis-0
    /// segment slowest), then `order^ndim` power coeffs per cell (axis-0
    /// power slowest, ascending powers). `order` is 2 (multilinear) or 4
    /// (tensor cubic). One child per axis, evaluated element-wise.
    fn interpolant_nd(
        slf: Py<Self>,
        py: Python<'_>,
        breakpoints: Vec<Vec<f64>>,
        coeffs: Vec<f64>,
        order: usize,
        children: Vec<Bound<'_, Expr>>,
    ) -> PyResult<Expr> {
        // Child count is a graph-construction concern, one child per axis; the
        // table's own invariants are checked by `NdInterpolantData::try_new`.
        let ndim = breakpoints.len();
        if children.len() != ndim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "interpolant_nd: expected {ndim} children (one per axis), got {}",
                children.len()
            )));
        }
        let order_u32 = u32::try_from(order).map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "interpolant_nd: order {order} is too large"
            ))
        })?;
        let data =
            NdInterpolantData::try_new(breakpoints, coeffs, order_u32).map_err(core_err_to_py)?;
        let child_ids: Vec<NodeId> = children
            .iter()
            .map(|c| c.borrow().node_id_in(&slf))
            .collect::<PyResult<_>>()?;
        let id = slf.borrow_mut(py).arena.alloc(Node::InterpolantNd {
            data: Box::new(data),
            children: child_ids,
        });
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn index(
        slf: Py<Self>,
        py: Python<'_>,
        child: &Expr,
        start: usize,
        end: usize,
    ) -> PyResult<Expr> {
        check_range(start, end, "index")?;
        let child_id = child.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Index {
            child: child_id,
            start,
            end,
        });
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn concat(slf: Py<Self>, py: Python<'_>, children: Vec<Bound<'_, Expr>>) -> PyResult<Expr> {
        let child_ids: Vec<NodeId> = children
            .iter()
            .map(|c| c.borrow().node_id_in(&slf))
            .collect::<PyResult<_>>()?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Concat(child_ids));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn matmul(slf: Py<Self>, py: Python<'_>, a: &Expr, b: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let b_id = b.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::MatMul(a_id, b_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn minimum(slf: Py<Self>, py: Python<'_>, a: &Expr, b: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let b_id = b.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Minimum(a_id, b_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn maximum(slf: Py<Self>, py: Python<'_>, a: &Expr, b: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let b_id = b.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Maximum(a_id, b_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn modulo(slf: Py<Self>, py: Python<'_>, a: &Expr, b: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let b_id = b.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Modulo(a_id, b_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn hypot(slf: Py<Self>, py: Python<'_>, a: &Expr, b: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let b_id = b.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Hypot(a_id, b_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn equal_heaviside(slf: Py<Self>, py: Python<'_>, a: &Expr, b: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let b_id = b.node_id_in(&slf)?;
        let id = slf
            .borrow_mut(py)
            .arena
            .alloc(Node::EqualHeaviside(a_id, b_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn not_equal_heaviside(slf: Py<Self>, py: Python<'_>, a: &Expr, b: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let b_id = b.node_id_in(&slf)?;
        let id = slf
            .borrow_mut(py)
            .arena
            .alloc(Node::NotEqualHeaviside(a_id, b_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn equality(slf: Py<Self>, py: Python<'_>, a: &Expr, b: &Expr) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let b_id = b.node_id_in(&slf)?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Equality(a_id, b_id));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    /// Create a conditional expression with selector and branches.
    /// Branch i is active when i - 0.5 < selector < i + 0.5 (1-based indexing).
    fn conditional(
        slf: Py<Self>,
        py: Python<'_>,
        selector: &Expr,
        branches: Vec<Bound<'_, Expr>>,
    ) -> PyResult<Expr> {
        if branches.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Conditional requires at least one branch",
            ));
        }

        let selector_id = selector.node_id_in(&slf)?;
        let branch_ids: Vec<NodeId> = branches
            .iter()
            .map(|b| b.borrow().node_id_in(&slf))
            .collect::<PyResult<_>>()?;
        let id = slf.borrow_mut(py).arena.alloc(Node::Conditional {
            selector: selector_id,
            branches: branch_ids,
        });

        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn sparse_matrix(
        slf: Py<Self>,
        py: Python<'_>,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: PyReadonlyArray1<'_, f64>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Expr> {
        let data = data.as_slice()?.to_vec();
        let csr = pybamm_core::CsrData::try_new(indptr, indices, data, Shape::matrix(rows, cols))
            .map_err(core_err_to_py)?;
        let id = slf
            .borrow_mut(py)
            .arena
            .alloc(Node::SparseMatrix(Box::new(csr)));
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    fn eval_to_float(
        &self,
        expr: &Expr,
        t: f64,
        y: Vec<f64>,
        y_dot: Vec<f64>,
        inputs: Vec<f64>,
    ) -> PyResult<f64> {
        check_supported(&self.arena, expr.node_id)?;
        let compiled = CompiledExpr::new(&self.arena, expr.node_id);
        let mut scratch = vec![0.0; compiled.scratch_len()];
        let result = compiled.eval(&mut scratch, t, &y, &y_dot, &inputs);
        result.first().copied().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "eval_to_float: expression evaluates to an empty array",
            )
        })
    }

    fn eval_to_array<'py>(
        &self,
        py: Python<'py>,
        expr: &Expr,
        t: f64,
        y: PyReadonlyArray1<'_, f64>,
        y_dot: PyReadonlyArray1<'_, f64>,
        inputs: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        check_supported(&self.arena, expr.node_id)?;
        let y_slice = y.as_slice()?;
        let y_dot_slice = y_dot.as_slice()?;
        let compiled = CompiledExpr::new(&self.arena, expr.node_id);
        let mut scratch = vec![0.0; compiled.scratch_len()];
        let result = compiled.eval(&mut scratch, t, y_slice, y_dot_slice, &inputs);
        Ok(PyArray1::from_slice(py, result))
    }

    /// Compile an expression into an immutable, shareable `CompiledFunction`.
    ///
    /// `n_states` overrides the scanned state extent so partial-group
    /// expressions can carry the full system width.
    #[pyo3(signature = (expr, name = None, n_states = None))]
    fn compile(
        slf: Py<Self>,
        py: Python<'_>,
        expr: &Expr,
        name: Option<String>,
        n_states: Option<usize>,
    ) -> PyResult<crate::function::CompiledFunction> {
        let node_id = expr.node_id_in(&slf)?;
        check_supported(slf.borrow(py).arena(), node_id)?;
        crate::function::CompiledFunction::build(py, slf, node_id, name, n_states)
    }

    /// Compile a named set of outputs into ONE tape with cross-output
    /// sharing: synthetic concat root + recorded slice offsets.
    #[pyo3(signature = (outputs, name = None, n_states = None))]
    fn compile_group(
        slf: Py<Self>,
        py: Python<'_>,
        outputs: &Bound<'_, pyo3::types::PyDict>,
        name: Option<String>,
        n_states: Option<usize>,
    ) -> PyResult<crate::group::CompiledFunctionGroup> {
        let mut names = Vec::with_capacity(outputs.len());
        let mut ids = Vec::with_capacity(outputs.len());
        for (key, value) in outputs.iter() {
            names.push(key.extract::<String>()?);
            let expr = value.extract::<PyRef<'_, Expr>>()?;
            ids.push(expr.node_id_in(&slf)?);
        }
        if ids.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "compile_group requires at least one output",
            ));
        }
        {
            let g = slf.borrow(py);
            for &id in &ids {
                check_supported(g.arena(), id)?;
            }
        }
        crate::group::CompiledFunctionGroup::build(py, slf, names, ids, name, n_states)
    }

    /// Get the number of colors needed for Jacobian computation
    #[cfg(feature = "serialize")]
    fn dump_dag(
        &self,
        expr: &Expr,
        path: &str,
        model_name: &str,
        n_states: usize,
        n_params: usize,
    ) -> PyResult<()> {
        let snapshot = pybamm_core::DagSnapshot {
            arena: self.arena.clone(),
            root: expr.node_id,
            n_states,
            n_params,
            mass_matrix: None,
            model_name: model_name.to_string(),
        };
        let bytes = snapshot.to_bytes();
        std::fs::write(path, bytes)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Serialize the arena and registered inputs to bytes (pickle protocol).
    #[cfg(feature = "serialize")]
    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let state = ExprGraphState {
            arena: self.arena.clone(),
            input_map: self.input_map.clone(),
            input_widths: self.input_widths.clone(),
        };
        let bytes = bincode::serialize(&state).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("ExprGraph serialize failed: {e}"))
        })?;
        Ok(pyo3::types::PyBytes::new(py, &bytes))
    }

    /// Restore the arena and registered inputs from bytes (pickle protocol).
    #[cfg(feature = "serialize")]
    fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
        let state: ExprGraphState = bincode::deserialize(state).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("ExprGraph deserialize failed: {e}"))
        })?;
        self.arena = state.arena;
        self.input_map = state.input_map;
        self.input_widths = state.input_widths;
        Ok(())
    }

    /// Empty args tuple for pickle (zero-arg `#[new]`); explicit `PyTuple`
    /// because pyo3 maps a `()` return to Python `None`, which pickle rejects.
    #[cfg(feature = "serialize")]
    #[allow(clippy::unused_self)]
    fn __getnewargs__<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyTuple> {
        pyo3::types::PyTuple::empty(py)
    }
}

#[pymethods]
impl Expr {
    #[getter]
    const fn id(&self) -> u32 {
        self.node_id.raw()
    }

    fn __add__(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        let other_id = other.node_id_in(&self.graph)?;
        let id = self
            .graph
            .borrow_mut(py)
            .arena
            .alloc(Node::Add(self.node_id, other_id));
        Ok(Self {
            node_id: id,
            graph: self.graph.clone_ref(py),
        })
    }

    fn __sub__(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        let other_id = other.node_id_in(&self.graph)?;
        let id = self
            .graph
            .borrow_mut(py)
            .arena
            .alloc(Node::Sub(self.node_id, other_id));
        Ok(Self {
            node_id: id,
            graph: self.graph.clone_ref(py),
        })
    }

    fn __mul__(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        let other_id = other.node_id_in(&self.graph)?;
        let id = self
            .graph
            .borrow_mut(py)
            .arena
            .alloc(Node::Mul(self.node_id, other_id));
        Ok(Self {
            node_id: id,
            graph: self.graph.clone_ref(py),
        })
    }

    fn __truediv__(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        let other_id = other.node_id_in(&self.graph)?;
        let id = self
            .graph
            .borrow_mut(py)
            .arena
            .alloc(Node::Div(self.node_id, other_id));
        Ok(Self {
            node_id: id,
            graph: self.graph.clone_ref(py),
        })
    }

    fn __pow__(
        &self,
        py: Python<'_>,
        other: &Self,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let other_id = other.node_id_in(&self.graph)?;
        let id = self
            .graph
            .borrow_mut(py)
            .arena
            .alloc(Node::Pow(self.node_id, other_id));
        Ok(Self {
            node_id: id,
            graph: self.graph.clone_ref(py),
        })
    }

    fn __neg__(&self, py: Python<'_>) -> PyResult<Self> {
        let id = self
            .graph
            .borrow_mut(py)
            .arena
            .alloc(Node::Neg(self.node_id));
        Ok(Self {
            node_id: id,
            graph: self.graph.clone_ref(py),
        })
    }
}
