// PyO3 bindings require specific argument types that clippy flags incorrectly
#![allow(clippy::needless_pass_by_value)]

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use pybamm_core::{
    Arena, ArrayData, CompiledExpr, CubicInterpolantData, DagRef, GraphError, InterpolantData,
    NdInterpolantData, Node, NodeId, ParamRegistry, ReadExtents, Shape,
};

use crate::errors::core_err_to_py;

/// Wrapper type for the rust-core [`Arena`] struct
#[pyclass(module = "pybamm.rust")]
#[derive(Debug)]
pub struct ExprGraph {
    arena: Arena,
    /// Names, registration indices and packed slots of the inputs in `arena`.
    params: ParamRegistry,
}

/// Wrapper type for the rust-core [`NodeId`] struct
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
    /// Node id, validated to belong to `graph` so it cannot index the wrong arena.
    pub(crate) fn node_id_in(&self, graph: &Py<ExprGraph>) -> PyResult<NodeId> {
        if self.graph.is(graph) {
            Ok(self.node_id)
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "Expr belongs to a different ExprGraph than the graph being built into",
            ))
        }
    }
}

impl ExprGraph {
    pub(crate) const fn arena(&self) -> &Arena {
        &self.arena
    }

    pub(crate) const fn arena_mut(&mut self) -> &mut Arena {
        &mut self.arena
    }

    /// `raw` as a node of this graph; `_rebuild` takes ids from Python unchecked.
    pub(crate) fn checked_root(&self, raw: u32) -> PyResult<NodeId> {
        let root = NodeId::from(raw);
        let n_nodes = self.arena.len();
        if root.index() >= n_nodes {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "_rebuild: root {raw} is out of range for a graph of {n_nodes} nodes"
            )));
        }
        Ok(root)
    }

    /// Shared access to `slf`, as a `ValueError` instead of a panic while it is borrowed mutably.
    fn borrow_graph<'py>(slf: &'py Py<Self>, py: Python<'py>) -> PyResult<PyRef<'py, Self>> {
        slf.try_borrow(py).map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("ExprGraph: graph borrow conflict")
        })
    }

    /// Exclusive access to `slf`, as a `ValueError` instead of a panic while it is borrowed.
    fn borrow_graph_mut<'py>(slf: &'py Py<Self>, py: Python<'py>) -> PyResult<PyRefMut<'py, Self>> {
        slf.try_borrow_mut(py).map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("ExprGraph: graph borrow conflict")
        })
    }

    /// Allocate `node` in `slf` and hand back its `Expr`.
    fn alloc(slf: Py<Self>, py: Python<'_>, node: Node) -> PyResult<Expr> {
        let id = Self::borrow_graph_mut(&slf, py)?.arena.alloc(node);
        Ok(Expr {
            node_id: id,
            graph: slf,
        })
    }

    /// Allocate `make(child)` after checking `a` belongs to `slf`.
    fn alloc_unary(
        slf: Py<Self>,
        py: Python<'_>,
        a: &Expr,
        make: impl FnOnce(NodeId) -> Node,
    ) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        Self::alloc(slf, py, make(a_id))
    }

    /// Allocate `make(left, right)` after checking both operands belong to `slf`.
    fn alloc_binary(
        slf: Py<Self>,
        py: Python<'_>,
        a: &Expr,
        b: &Expr,
        make: impl FnOnce(NodeId, NodeId) -> Node,
    ) -> PyResult<Expr> {
        let a_id = a.node_id_in(&slf)?;
        let b_id = b.node_id_in(&slf)?;
        Self::alloc(slf, py, make(a_id, b_id))
    }

    /// Node ids of `exprs`, each checked to belong to `slf`.
    fn node_ids_in(slf: &Py<Self>, exprs: &[Bound<'_, Expr>]) -> PyResult<Vec<NodeId>> {
        exprs.iter().map(|e| e.get().node_id_in(slf)).collect()
    }

    pub(crate) const fn params(&self) -> &ParamRegistry {
        &self.params
    }
}

/// Reject a buffer shorter than `root` reads, which would otherwise panic.
fn check_read_extents(
    arena: &Arena,
    root: NodeId,
    method: &str,
    y: usize,
    y_dot: usize,
    inputs: usize,
) -> PyResult<()> {
    let extents = ReadExtents::scan(arena, root);
    for (name, got, needed) in [
        ("y", y, extents.y),
        ("y_dot", y_dot, extents.y_dot.unwrap_or(0)),
        ("inputs", inputs, extents.inputs),
    ] {
        if got < needed {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{method}: the expression reads {needed} value(s) of {name}, got {got}"
            )));
        }
    }
    Ok(())
}

/// Validate that the graph rooted at `root` can be lowered safely.
pub fn check_supported(arena: &Arena, root: NodeId) -> PyResult<()> {
    pybamm_core::validate(arena, root).map_err(|err| match err {
        GraphError::Unsupported(msg) => pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "Rust conversion does not support this expression: {msg}. Use a \
             CasADi-backed model (set `model.convert_to_format = 'casadi'`)."
        )),
        GraphError::Invalid(msg) => {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid Rust expression graph: {msg}"))
        },
    })
}

#[pymethods]
impl ExprGraph {
    #[new]
    fn new() -> Self {
        Self {
            arena: Arena::new(),
            params: ParamRegistry::new(),
        }
    }

    fn scalar(slf: Py<Self>, py: Python<'_>, value: f64) -> PyResult<Expr> {
        Self::alloc(slf, py, Node::Scalar(value))
    }

    fn time(slf: Py<Self>, py: Python<'_>) -> PyResult<Expr> {
        Self::alloc(slf, py, Node::Time)
    }

    fn state_vector(slf: Py<Self>, py: Python<'_>, start: usize, end: usize) -> PyResult<Expr> {
        let node = Node::state_vector(start, end).map_err(core_err_to_py)?;
        Self::alloc(slf, py, node)
    }

    fn state_vector_dot(slf: Py<Self>, py: Python<'_>, start: usize, end: usize) -> PyResult<Expr> {
        let node = Node::state_vector_dot(start, end).map_err(core_err_to_py)?;
        Self::alloc(slf, py, node)
    }

    /// Total packed width of the registered inputs (the stacked `p` length).
    const fn n_inputs(&self) -> usize {
        self.params.packed_width().get()
    }

    /// Registration index (not packed offset) of `name`, or `None` if unregistered.
    fn input_index(&self, name: &str) -> Option<usize> {
        self.params.index_of(name).map(pybamm_core::ParamIdx::index)
    }

    /// Number of nodes in the expression arena.
    #[getter]
    const fn n_nodes(&self) -> usize {
        self.arena.len()
    }

    /// Register (or re-look-up) a named input parameter of `width` packed values.
    #[pyo3(signature = (name, width = 1))]
    fn input_parameter(slf: Py<Self>, py: Python<'_>, name: &str, width: usize) -> PyResult<Expr> {
        let mut graph = Self::borrow_graph_mut(&slf, py)?;
        let (index, slot) = graph.params.register(name, width).map_err(core_err_to_py)?;
        let id = graph.arena.alloc(Node::InputParameter {
            name: name.to_string(),
            index: index.index(),
            offset: slot.offset.index(),
            width: slot.width.index(),
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
        Self::alloc(slf, py, Node::Array(Box::new(array)))
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
        Self::alloc(slf, py, Node::Array(Box::new(array)))
    }

    fn add(slf: Py<Self>, py: Python<'_>, a: &Expr, b: &Expr) -> PyResult<Expr> {
        Self::alloc_binary(slf, py, a, b, Node::Add)
    }

    fn sub(slf: Py<Self>, py: Python<'_>, a: &Expr, b: &Expr) -> PyResult<Expr> {
        Self::alloc_binary(slf, py, a, b, Node::Sub)
    }

    fn mul(slf: Py<Self>, py: Python<'_>, a: &Expr, b: &Expr) -> PyResult<Expr> {
        Self::alloc_binary(slf, py, a, b, Node::Mul)
    }

    fn div(slf: Py<Self>, py: Python<'_>, a: &Expr, b: &Expr) -> PyResult<Expr> {
        Self::alloc_binary(slf, py, a, b, Node::Div)
    }

    fn neg(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        Self::alloc_unary(slf, py, a, Node::Neg)
    }

    fn abs(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        Self::alloc_unary(slf, py, a, Node::Abs)
    }

    fn pow(slf: Py<Self>, py: Python<'_>, a: &Expr, b: &Expr) -> PyResult<Expr> {
        Self::alloc_binary(slf, py, a, b, Node::Pow)
    }

    fn sqrt(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        Self::alloc_unary(slf, py, a, Node::Sqrt)
    }

    fn exp(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        Self::alloc_unary(slf, py, a, Node::Exp)
    }

    fn log(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        Self::alloc_unary(slf, py, a, Node::Log)
    }

    fn sin(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        Self::alloc_unary(slf, py, a, Node::Sin)
    }

    fn cos(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        Self::alloc_unary(slf, py, a, Node::Cos)
    }

    fn tanh(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        Self::alloc_unary(slf, py, a, Node::Tanh)
    }

    fn sinh(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        Self::alloc_unary(slf, py, a, Node::Sinh)
    }

    fn cosh(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        Self::alloc_unary(slf, py, a, Node::Cosh)
    }

    fn arcsinh(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        Self::alloc_unary(slf, py, a, Node::Arcsinh)
    }

    fn arctan(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        Self::alloc_unary(slf, py, a, Node::Arctan)
    }

    fn erf(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        Self::alloc_unary(slf, py, a, Node::Erf)
    }

    fn sign(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        Self::alloc_unary(slf, py, a, Node::Sign)
    }

    fn floor(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        Self::alloc_unary(slf, py, a, Node::Floor)
    }

    fn ceiling(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        Self::alloc_unary(slf, py, a, Node::Ceiling)
    }

    fn max_reduce(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        Self::alloc_unary(slf, py, a, Node::MaxReduce)
    }

    fn min_reduce(slf: Py<Self>, py: Python<'_>, a: &Expr) -> PyResult<Expr> {
        Self::alloc_unary(slf, py, a, Node::MinReduce)
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
        let node = Node::Interpolant1DLinear {
            data: Box::new(data),
            child: child_id,
        };
        Self::alloc(slf, py, node)
    }

    /// Build a 1D cubic/pchip interpolant.
    ///
    /// `coeffs` is flat row-major `[c0,c1,c2,c3, c0,c1,c2,c3, ...]`, four per segment.
    fn interpolant_1d_cubic(
        slf: Py<Self>,
        py: Python<'_>,
        breakpoints: Vec<f64>,
        coeffs: Vec<f64>,
        child: &Expr,
    ) -> PyResult<Expr> {
        let child_id = child.node_id_in(&slf)?;
        let data =
            CubicInterpolantData::try_from_flat(breakpoints, &coeffs).map_err(core_err_to_py)?;
        let node = Node::Interpolant1DCubic {
            data: Box::new(data),
            child: child_id,
        };
        Self::alloc(slf, py, node)
    }

    /// Build an N-D (2 or 3 axis) tensor-product interpolant, one child per axis.
    ///
    /// `coeffs` is cell-major (axis-0 slowest), then `order^ndim` ascending power
    /// coeffs per cell; `order` is 2 (multilinear) or 4 (tensor cubic).
    fn interpolant_nd(
        slf: Py<Self>,
        py: Python<'_>,
        breakpoints: Vec<Vec<f64>>,
        coeffs: Vec<f64>,
        order: usize,
        children: Vec<Bound<'_, Expr>>,
    ) -> PyResult<Expr> {
        let order_u32 = u32::try_from(order).map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "interpolant_nd: order {order} is too large"
            ))
        })?;
        let data =
            NdInterpolantData::try_new(breakpoints, coeffs, order_u32).map_err(core_err_to_py)?;
        let child_ids = Self::node_ids_in(&slf, &children)?;
        let node = Node::interpolant_nd(data, child_ids).map_err(core_err_to_py)?;
        Self::alloc(slf, py, node)
    }

    fn index(
        slf: Py<Self>,
        py: Python<'_>,
        child: &Expr,
        start: usize,
        end: usize,
    ) -> PyResult<Expr> {
        let child_id = child.node_id_in(&slf)?;
        let node = Node::index(child_id, start, end).map_err(core_err_to_py)?;
        Self::alloc(slf, py, node)
    }

    fn concat(slf: Py<Self>, py: Python<'_>, children: Vec<Bound<'_, Expr>>) -> PyResult<Expr> {
        let child_ids = Self::node_ids_in(&slf, &children)?;
        Self::alloc(slf, py, Node::Concat(child_ids))
    }

    fn matmul(slf: Py<Self>, py: Python<'_>, a: &Expr, b: &Expr) -> PyResult<Expr> {
        Self::alloc_binary(slf, py, a, b, Node::MatMul)
    }

    fn minimum(slf: Py<Self>, py: Python<'_>, a: &Expr, b: &Expr) -> PyResult<Expr> {
        Self::alloc_binary(slf, py, a, b, Node::Minimum)
    }

    fn maximum(slf: Py<Self>, py: Python<'_>, a: &Expr, b: &Expr) -> PyResult<Expr> {
        Self::alloc_binary(slf, py, a, b, Node::Maximum)
    }

    fn modulo(slf: Py<Self>, py: Python<'_>, a: &Expr, b: &Expr) -> PyResult<Expr> {
        Self::alloc_binary(slf, py, a, b, Node::Modulo)
    }

    fn hypot(slf: Py<Self>, py: Python<'_>, a: &Expr, b: &Expr) -> PyResult<Expr> {
        Self::alloc_binary(slf, py, a, b, Node::Hypot)
    }

    fn equal_heaviside(slf: Py<Self>, py: Python<'_>, a: &Expr, b: &Expr) -> PyResult<Expr> {
        Self::alloc_binary(slf, py, a, b, Node::EqualHeaviside)
    }

    fn not_equal_heaviside(slf: Py<Self>, py: Python<'_>, a: &Expr, b: &Expr) -> PyResult<Expr> {
        Self::alloc_binary(slf, py, a, b, Node::NotEqualHeaviside)
    }

    fn equality(slf: Py<Self>, py: Python<'_>, a: &Expr, b: &Expr) -> PyResult<Expr> {
        Self::alloc_binary(slf, py, a, b, Node::Equality)
    }

    /// Conditional expression; branch i (1-based) is active when i - 0.5 < selector < i + 0.5.
    fn conditional(
        slf: Py<Self>,
        py: Python<'_>,
        selector: &Expr,
        branches: Vec<Bound<'_, Expr>>,
    ) -> PyResult<Expr> {
        let selector_id = selector.node_id_in(&slf)?;
        let branch_ids = Self::node_ids_in(&slf, &branches)?;
        let node = Node::conditional(selector_id, branch_ids).map_err(core_err_to_py)?;
        Self::alloc(slf, py, node)
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
        Self::alloc(slf, py, Node::SparseMatrix(Box::new(csr)))
    }

    fn eval_to_float(
        slf: Py<Self>,
        py: Python<'_>,
        expr: &Expr,
        t: f64,
        y: Vec<f64>,
        y_dot: Vec<f64>,
        inputs: Vec<f64>,
    ) -> PyResult<f64> {
        let node_id = expr.node_id_in(&slf)?;
        let g = Self::borrow_graph(&slf, py)?;
        check_supported(&g.arena, node_id)?;
        check_read_extents(
            &g.arena,
            node_id,
            "eval_to_float",
            y.len(),
            y_dot.len(),
            inputs.len(),
        )?;
        let compiled = CompiledExpr::new(DagRef::new(&g.arena, node_id));
        if compiled.output_len() != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "eval_to_float: expression evaluates to {} values, not one; use eval_to_array",
                compiled.output_len()
            )));
        }
        let mut scratch = vec![0.0; compiled.scratch_len()];
        Ok(compiled.eval(&mut scratch, t, &y, &y_dot, &inputs)[0])
    }

    fn eval_to_array<'py>(
        slf: Py<Self>,
        py: Python<'py>,
        expr: &Expr,
        t: f64,
        y: PyReadonlyArray1<'_, f64>,
        y_dot: PyReadonlyArray1<'_, f64>,
        inputs: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let node_id = expr.node_id_in(&slf)?;
        let g = Self::borrow_graph(&slf, py)?;
        check_supported(&g.arena, node_id)?;
        let y_slice = y.as_slice()?;
        let y_dot_slice = y_dot.as_slice()?;
        check_read_extents(
            &g.arena,
            node_id,
            "eval_to_array",
            y_slice.len(),
            y_dot_slice.len(),
            inputs.len(),
        )?;
        let compiled = CompiledExpr::new(DagRef::new(&g.arena, node_id));
        let mut scratch = vec![0.0; compiled.scratch_len()];
        let result = compiled.eval(&mut scratch, t, y_slice, y_dot_slice, &inputs);
        Ok(PyArray1::from_slice(py, result))
    }

    /// Compile an expression into an immutable, shareable `CompiledFunction`.
    ///
    /// `n_states` overrides the scanned state extent, e.g. to the full system width.
    #[pyo3(signature = (expr, name = None, n_states = None))]
    fn compile(
        slf: Py<Self>,
        py: Python<'_>,
        expr: &Expr,
        name: Option<String>,
        n_states: Option<usize>,
    ) -> PyResult<crate::compiled_function::CompiledFunction> {
        let node_id = expr.node_id_in(&slf)?;
        check_supported(Self::borrow_graph(&slf, py)?.arena(), node_id)?;
        crate::compiled_function::CompiledFunction::build(py, slf, node_id, name, n_states)
    }

    /// Compile a named set of outputs into one shared tape as a `CompiledFunction`.
    #[pyo3(signature = (outputs, name = None, n_states = None))]
    fn compile_outputs(
        slf: Py<Self>,
        py: Python<'_>,
        outputs: &Bound<'_, pyo3::types::PyDict>,
        name: Option<String>,
        n_states: Option<usize>,
    ) -> PyResult<crate::compiled_function::CompiledFunction> {
        let (names, ids): (Vec<String>, Vec<NodeId>) = outputs
            .iter()
            .map(|(key, value)| {
                let name = key.extract::<String>()?;
                let id = value.extract::<PyRef<'_, Expr>>()?.node_id_in(&slf)?;
                Ok((name, id))
            })
            .collect::<PyResult<_>>()?;
        if ids.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "compile_outputs requires at least one output",
            ));
        }
        {
            let g = Self::borrow_graph(&slf, py)?;
            for &id in &ids {
                check_supported(g.arena(), id)?;
            }
        }
        crate::compiled_function::CompiledFunction::build_outputs(
            py, slf, names, ids, name, n_states,
        )
    }

    /// Write a serialized `DagSnapshot` of `expr` to `path`.
    #[cfg(feature = "serialize")]
    fn dump_dag(
        slf: Py<Self>,
        py: Python<'_>,
        expr: &Expr,
        path: &str,
        model_name: &str,
        n_states: usize,
        n_params: usize,
    ) -> PyResult<()> {
        let node_id = expr.node_id_in(&slf)?;
        let snapshot = pybamm_core::DagSnapshot {
            dag: pybamm_core::Dag::new(Self::borrow_graph(&slf, py)?.arena.clone(), node_id),
            n_states,
            packed_input_width: pybamm_core::PackedWidth::new(n_params),
            mass_matrix: None,
            model_name: model_name.to_string(),
        };
        let bytes = snapshot.to_bytes();
        std::fs::write(path, bytes)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Serialize the graph to bytes (pickle protocol).
    ///
    /// The arena alone: the registry is rebuilt from its `InputParameter` nodes, so the two
    /// cannot disagree. The bytes match the one-field wrapper older pickles were written with.
    #[cfg(feature = "serialize")]
    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let bytes = bincode::serialize(&self.arena).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("ExprGraph serialize failed: {e}"))
        })?;
        Ok(pyo3::types::PyBytes::new(py, &bytes))
    }

    /// Restore the graph from bytes (pickle protocol), into a fresh graph only.
    #[cfg(feature = "serialize")]
    fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
        // Live `Expr`s and compiled artifacts hold ids into this arena; a swap would repoint them.
        if !self.arena.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "ExprGraph.__setstate__ restores into an empty graph only; this one has {} nodes",
                self.arena.len()
            )));
        }
        let arena: Arena = bincode::deserialize(state).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("ExprGraph deserialize failed: {e}"))
        })?;
        self.params = ParamRegistry::from_arena(&arena).map_err(core_err_to_py)?;
        self.arena = arena;
        Ok(())
    }

    /// Empty args tuple for pickle; a `()` return would map to `None`, which pickle rejects.
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
        ExprGraph::add(self.graph.clone_ref(py), py, self, other)
    }

    fn __sub__(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        ExprGraph::sub(self.graph.clone_ref(py), py, self, other)
    }

    fn __mul__(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        ExprGraph::mul(self.graph.clone_ref(py), py, self, other)
    }

    fn __truediv__(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        ExprGraph::div(self.graph.clone_ref(py), py, self, other)
    }

    fn __pow__(
        &self,
        py: Python<'_>,
        other: &Self,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        ExprGraph::pow(self.graph.clone_ref(py), py, self, other)
    }

    fn __neg__(&self, py: Python<'_>) -> PyResult<Self> {
        ExprGraph::neg(self.graph.clone_ref(py), py, self)
    }
}

#[cfg(all(test, feature = "serialize"))]
mod tests {
    use super::*;

    #[test]
    fn setstate_restores_only_into_an_empty_graph() {
        let mut source = ExprGraph::new();
        let _ = source.arena.alloc(Node::Time);
        let state = bincode::serialize(&source.arena).expect("an arena serializes");

        let mut fresh = ExprGraph::new();
        assert!(fresh.__setstate__(&state).is_ok());
        assert_eq!(fresh.n_nodes(), 1);
        assert!(
            fresh.__setstate__(&state).is_err(),
            "a populated graph must refuse a new arena"
        );
        assert_eq!(fresh.n_nodes(), 1);
    }
}
