// PyO3 bindings require specific argument types that clippy flags incorrectly
#![allow(clippy::needless_pass_by_value)]

use std::sync::Arc;

use numpy::{PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyDict;

use pybamm_core::model as core_model;
use pybamm_core::{CsrData, NodeId, ObservableKind, Shape};

use crate::errors::core_err_to_py;
use crate::evaluator_pool::EvaluatorPool;
use crate::expr::{Expr, ExprGraph};
use crate::function::CompiledFunction;
use crate::jacobian::CompiledJacobian;
use crate::signature::FunctionSignature;

/// `(graph, rhs_root, output_roots, event_roots, algebraic_root, algebraic_variable_indices,
/// mass_data, mass_indptr, mass_indices, n_inputs, sens_param_indices)`, the
/// `__reduce__`/`_rebuild` argument tuple for `CompiledModel`'s pickle protocol.
type RebuildArgs = (
    Py<ExprGraph>,
    u32,
    Vec<u32>,
    Vec<u32>,
    Option<u32>,
    Vec<usize>,
    Vec<f64>,
    Vec<i64>,
    Vec<i64>,
    usize,
    Vec<usize>,
);

/// The Python face of [`core_model::CompiledModel`]: the same immutable
/// artifact behind the same `Arc`, plus the retained derivation graph and the
/// bundle accessors (`rhs`, `jacobian`, `outputs`, `events`,
/// `algebraic_residual`, `algebraic_jacobian`) composed over the same tapes.
///
/// Holding the artifact rather than an evaluator is what keeps it shareable;
/// solvers take their per-solve state from `evaluator_pool`.
// `module` is required so pickle can locate the class as `pybamm.rust.CompiledModel`
// instead of the pyo3 default `builtins.CompiledModel`, which pickle cannot import.
#[pyclass(module = "pybamm.rust")]
pub struct CompiledModel {
    pub(crate) compiled: Arc<core_model::CompiledModel>,
    /// Scratch for the direct `eval_*`/`assemble_*` helpers, bound on first
    /// use, so a model that is only lowered and handed on never allocates one.
    scratch: Option<core_model::Workspace>,
    /// Retained derivation source for the bundle views (jacobian / jvp / the
    /// algebraic-subset jacobian all re-run the AD pipeline on this arena).
    graph: Py<ExprGraph>,
    rhs_root: NodeId,
    output_roots: Vec<NodeId>,
    event_roots: Vec<NodeId>,
    algebraic_root: Option<NodeId>,
    /// Strictly-ascending global state indices of the algebraic block.
    algebraic_variable_indices: Vec<usize>,
    // Cached views: every accessor hands back the SAME prepared artifact, so pools,
    // lazy tangent tapes and sparsity/coloring prep amortise across accesses.
    rhs_view: PyOnceLock<Py<CompiledFunction>>,
    jac_view: PyOnceLock<Py<CompiledJacobian>>,
    outputs_view: PyOnceLock<Vec<Py<CompiledFunction>>>,
    events_view: PyOnceLock<Vec<Py<CompiledFunction>>>,
    algebraic_residual_view: PyOnceLock<Option<Py<CompiledFunction>>>,
    algebraic_jacobian_view: PyOnceLock<Option<Py<CompiledJacobian>>>,
}

impl std::fmt::Debug for CompiledModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledModel")
            .field("n_states", &self.compiled.n_states())
            .field("n_inputs", &self.compiled.n_params())
            .field("output_len", &self.compiled.output_len())
            .field("n_colors", &self.compiled.coloring().n_colors)
            .finish_non_exhaustive()
    }
}

impl CompiledModel {
    /// Run `body` against the artifact and a [`core_model::Workspace`] bound on
    /// first use. Only the direct `eval_*`/`assemble_*` helpers need one.
    fn with_scratch<R>(
        &mut self,
        body: impl FnOnce(&core_model::CompiledModel, &mut core_model::Workspace) -> R,
    ) -> R {
        let Self {
            compiled, scratch, ..
        } = self;
        body(
            compiled,
            scratch.get_or_insert_with(|| compiled.create_workspace()),
        )
    }

    /// One `CompiledFunction` view per observable of `kind`, named `label[i]`.
    ///
    /// `roots` are this family's retained arena roots, in the order they were
    /// compiled, so a view carries the node it came from as well as its tape.
    fn observable_views(
        &self,
        py: Python<'_>,
        kind: ObservableKind,
        roots: &[NodeId],
        label: &str,
    ) -> PyResult<Vec<Py<CompiledFunction>>> {
        let set = self.compiled.observables(kind);
        (0..set.count())
            .map(|i| {
                Py::new(
                    py,
                    CompiledFunction::from_shared(
                        py,
                        set.expr_arc(i),
                        self.graph.clone_ref(py),
                        roots[i],
                        self.compiled.n_states(),
                        Some(format!("{label}[{i}]")),
                    )?,
                )
            })
            .collect()
    }

    /// Build a [`FunctionSignature`] for a bundle view over the retained graph.
    ///
    /// Bundle views always carry the full system width and never use `y_dot`
    /// (the residual's `cj`/mass coupling stays solver-side).
    fn view_signature(
        &self,
        py: Python<'_>,
        output_len: usize,
        name: Option<String>,
    ) -> PyResult<FunctionSignature> {
        let g = self.graph.try_borrow(py).map_err(|_| {
            PyValueError::new_err("CompiledModel: graph borrow conflict building view signature")
        })?;
        Ok(FunctionSignature {
            input_names: g.input_names(),
            input_widths: g.input_widths(),
            n_states: self.compiled.n_states(),
            uses_y_dot: false,
            output_len,
            name,
        })
    }

    /// Shared builder over already-resolved roots, the raw CSR mass matrix and
    /// the optional artifacts.
    ///
    /// Both `from_expr` (roots from validated `Expr`s) and `_rebuild` (roots
    /// bounds-checked against the arena) funnel through here, so the
    /// `check_supported` gate below runs exactly once per public entry point.
    #[allow(clippy::too_many_arguments)]
    fn build_from_parts(
        py: Python<'_>,
        graph: Py<ExprGraph>,
        rhs_root: NodeId,
        output_roots: Vec<NodeId>,
        event_roots: Vec<NodeId>,
        algebraic_root: Option<NodeId>,
        algebraic_variable_indices: Vec<usize>,
        mass_data: Vec<f64>,
        mass_indptr: Vec<i64>,
        mass_indices: Vec<i64>,
        n_inputs: usize,
        sens_param_indices: Vec<usize>,
    ) -> PyResult<Self> {
        // Infer n_states from indptr length
        if mass_indptr.is_empty() {
            return Err(PyValueError::new_err(
                "mass_indptr must have at least 1 element",
            ));
        }
        let n = mass_indptr.len() - 1;

        // scipy CSR arrays arrive as i64; convert with a bounds check so a
        // negative entry becomes a clear error rather than a wrapped `usize`.
        let indptr: Vec<usize> = mass_indptr
            .iter()
            .map(|&x| usize::try_from(x))
            .collect::<Result<_, _>>()
            .map_err(|_| PyValueError::new_err("mass_indptr entries must be non-negative"))?;
        let indices: Vec<usize> = mass_indices
            .iter()
            .map(|&x| usize::try_from(x))
            .collect::<Result<_, _>>()
            .map_err(|_| PyValueError::new_err("mass_indices entries must be non-negative"))?;

        let mass = CsrData::try_new(indptr, indices, mass_data, Shape::matrix(n, n))
            .map_err(core_err_to_py)?;

        // `new_wrt_state_subset` asserts a strictly-ascending subset; callers pass
        // a contiguous range, but normalise so the invariant holds regardless.
        let mut algebraic_variable_indices = algebraic_variable_indices;
        algebraic_variable_indices.sort_unstable();
        algebraic_variable_indices.dedup();

        let g = graph.try_borrow(py).map_err(|_| {
            PyValueError::new_err("CompiledModel.build_from_parts: graph borrow conflict")
        })?;
        let arena = g.arena();

        // Validate every root before lowering so unsupported nodes or invalid
        // shape relationships surface as catchable Python errors.
        crate::expr::check_supported(arena, rhs_root)?;
        for &root in &output_roots {
            crate::expr::check_supported(arena, root)?;
        }
        for &root in &event_roots {
            crate::expr::check_supported(arena, root)?;
        }
        if let Some(root) = algebraic_root {
            crate::expr::check_supported(arena, root)?;
        }

        let options = algebraic_root.map_or_else(
            || core_model::CompiledModelOptions::new().with_sensitivities(&sens_param_indices),
            |algebraic| {
                core_model::CompiledModelOptions::new()
                    .with_sensitivities(&sens_param_indices)
                    .with_algebraic(core_model::CompiledModelAlgebraicBlock::new(
                        algebraic,
                        &algebraic_variable_indices,
                    ))
            },
        );

        // Composed before the `Arc`, so appending an output or event is a plain
        // `&mut self` call rather than a copy-on-write plus workspace rebuild.
        let mut compiled = core_model::CompiledModel::new_with_options(
            arena, rhs_root, mass, n, n_inputs, options,
        );

        // Output-variable nodes must come from the same arena as the rhs, which
        // holds because `PyBaMM` builds both through one `ExprGraph`.
        for &root in &output_roots {
            compiled.add_output(arena, root);
        }

        // Compile each event expression for root-finding during integration.
        for &root in &event_roots {
            compiled.add_event(arena, root);
        }
        drop(g);

        // Fuse the events into one tape so both hot loops evaluate shared event
        // subgraphs once. Needs a mutable arena borrow to alloc the `Concat` root.
        if event_roots.len() >= 2 {
            let mut g = graph.try_borrow_mut(py).map_err(|_| {
                PyValueError::new_err("CompiledModel.build_from_parts: graph borrow conflict")
            })?;
            compiled.fuse_events(g.arena_mut(), &event_roots);
        }

        Ok(Self {
            compiled: Arc::new(compiled),
            scratch: None,
            graph,
            rhs_root,
            output_roots,
            event_roots,
            algebraic_root,
            algebraic_variable_indices,
            rhs_view: PyOnceLock::new(),
            jac_view: PyOnceLock::new(),
            outputs_view: PyOnceLock::new(),
            events_view: PyOnceLock::new(),
            algebraic_residual_view: PyOnceLock::new(),
            algebraic_jacobian_view: PyOnceLock::new(),
        })
    }
}

#[pymethods]
impl CompiledModel {
    /// Create a compiled model from an expression graph.
    ///
    /// # Arguments
    ///
    /// * `graph` - The expression graph containing the RHS expression
    /// * `expr` - The root expression node (f(t, y))
    /// * `mass_data` - Non-zero values of mass matrix (CSR data array)
    /// * `mass_indptr` - CSR indptr array (length `n_states` + 1)
    /// * `mass_indices` - CSR column indices array
    /// * `n_inputs` - Number of input parameters (default 0)
    ///
    /// # Returns
    ///
    /// A new `CompiledModel` ready for evaluation
    #[staticmethod]
    #[pyo3(signature = (
        graph,
        expr,
        mass_data,
        mass_indptr,
        mass_indices,
        n_inputs = 0,
        sens_param_indices = vec![],
        output_exprs = vec![],
        algebraic_expr = None,
        algebraic_variable_indices = vec![],
        event_exprs = vec![],
    ))]
    #[allow(clippy::too_many_arguments)] // PyO3 keyword args, all distinct concerns
    fn from_expr(
        py: Python<'_>,
        graph: Py<ExprGraph>,
        expr: &Expr,
        mass_data: PyReadonlyArray1<'_, f64>,
        mass_indptr: PyReadonlyArray1<'_, i64>,
        mass_indices: PyReadonlyArray1<'_, i64>,
        n_inputs: usize,
        sens_param_indices: Vec<usize>,
        output_exprs: Vec<PyRef<'_, Expr>>,
        algebraic_expr: Option<&Expr>,
        algebraic_variable_indices: Vec<usize>,
        event_exprs: Vec<PyRef<'_, Expr>>,
    ) -> PyResult<Self> {
        // numpy -> Vec conversions; `build_from_parts` owns the CSR assembly
        // so it is shared with the `_rebuild` pickle path.
        let mass_data: Vec<f64> = mass_data.as_slice()?.to_vec();
        let mass_indptr: Vec<i64> = mass_indptr.as_slice()?.to_vec();
        let mass_indices: Vec<i64> = mass_indices.as_slice()?.to_vec();

        // Expr -> NodeId conversions: the bundle retains the roots so the view
        // accessors can compose prepared artifacts over the shared tapes.
        let rhs_root = expr.node_id_in(&graph)?;
        let output_roots: Vec<NodeId> = output_exprs
            .iter()
            .map(|e| e.node_id_in(&graph))
            .collect::<PyResult<_>>()?;
        let event_roots: Vec<NodeId> = event_exprs
            .iter()
            .map(|e| e.node_id_in(&graph))
            .collect::<PyResult<_>>()?;
        let algebraic_root = algebraic_expr
            .as_ref()
            .map(|e| e.node_id_in(&graph))
            .transpose()?;

        Self::build_from_parts(
            py,
            graph,
            rhs_root,
            output_roots,
            event_roots,
            algebraic_root,
            algebraic_variable_indices,
            mass_data,
            mass_indptr,
            mass_indices,
            n_inputs,
            sens_param_indices,
        )
    }

    /// Rebuild from the retained `(graph, roots, mass CSR, options)`
    /// derivation source (pickle protocol). Recompiles every tape and resets
    /// the lazily-derived bundle-view caches.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn _rebuild(
        py: Python<'_>,
        graph: Py<ExprGraph>,
        rhs_root: u32,
        output_roots: Vec<u32>,
        event_roots: Vec<u32>,
        algebraic_root: Option<u32>,
        algebraic_variable_indices: Vec<usize>,
        mass_data: Vec<f64>,
        mass_indptr: Vec<i64>,
        mass_indices: Vec<i64>,
        n_inputs: usize,
        sens_param_indices: Vec<usize>,
    ) -> PyResult<Self> {
        let rhs_root = NodeId::from(rhs_root);
        let output_roots: Vec<NodeId> = output_roots.into_iter().map(NodeId::from).collect();
        let event_roots: Vec<NodeId> = event_roots.into_iter().map(NodeId::from).collect();
        let algebraic_root = algebraic_root.map(NodeId::from);

        {
            // Reachable from Python with arbitrary ids: bounds-check every
            // root before `build_from_parts` indexes the arena with it.
            let g = graph.borrow(py);
            let n_nodes = g.arena().len();
            let check_bounds = |root: NodeId| -> PyResult<()> {
                if root.index() >= n_nodes {
                    return Err(PyValueError::new_err(format!(
                        "_rebuild: root {} is out of range for a graph of {n_nodes} nodes",
                        root.raw(),
                    )));
                }
                Ok(())
            };
            check_bounds(rhs_root)?;
            for &root in &output_roots {
                check_bounds(root)?;
            }
            for &root in &event_roots {
                check_bounds(root)?;
            }
            if let Some(root) = algebraic_root {
                check_bounds(root)?;
            }
        }

        Self::build_from_parts(
            py,
            graph,
            rhs_root,
            output_roots,
            event_roots,
            algebraic_root,
            algebraic_variable_indices,
            mass_data,
            mass_indptr,
            mass_indices,
            n_inputs,
            sens_param_indices,
        )
    }

    /// `(callable, args)` pair for the pickle protocol: rebuild from the
    /// retained graph, roots, and mass CSR rather than serializing derived
    /// evaluator/coloring state.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, RebuildArgs)> {
        // `CompiledModel` is not frozen (it has `&mut self` eval methods),
        // so `Bound::get` is unavailable; borrow the wrapped value instead.
        let this = slf.borrow();
        let rebuild = slf.get_type().getattr("_rebuild")?;
        let mass = this.compiled.mass_matrix();
        // Mirrors `mass_indptr`/`mass_indices` back to the i64 width they
        // arrived in from scipy via `from_expr`.
        #[allow(clippy::cast_possible_wrap)]
        let mass_indptr: Vec<i64> = mass.indptr().iter().map(|&x| x as i64).collect();
        #[allow(clippy::cast_possible_wrap)]
        let mass_indices: Vec<i64> = mass.indices().iter().map(|&x| x as i64).collect();
        Ok((
            rebuild,
            (
                this.graph.clone_ref(py),
                this.rhs_root.raw(),
                this.output_roots.iter().map(|r| r.raw()).collect(),
                this.event_roots.iter().map(|r| r.raw()).collect(),
                this.algebraic_root.map(NodeId::raw),
                this.algebraic_variable_indices.clone(),
                mass.data().to_vec(),
                mass_indptr,
                mass_indices,
                this.compiled.n_params(),
                this.compiled.sens_param_indices().to_vec(),
            ),
        ))
    }

    // Each view is built ONCE from the bundle's shared `Arc`s and cached, so
    // repeated access is a `clone_ref` and the prepared state amortises.

    /// Primal f(t, y, p) as a shareable `CompiledFunction` view.
    #[getter]
    fn rhs(&self, py: Python<'_>) -> PyResult<Py<CompiledFunction>> {
        self.rhs_view
            .get_or_try_init(py, || {
                Py::new(
                    py,
                    CompiledFunction::from_shared(
                        py,
                        self.compiled.primal_expr_arc(),
                        self.graph.clone_ref(py),
                        self.rhs_root,
                        self.compiled.n_states(),
                        Some("rhs".to_string()),
                    )?,
                )
            })
            .map(|f| f.clone_ref(py))
    }

    /// The retained derivation arena. Observation lowers a new root into it
    /// (`symbol.to_rust(graph)` + `graph.compile`) so the observed expression
    /// shares the solve's input-parameter and state indices by construction.
    #[getter]
    fn graph(&self, py: Python<'_>) -> Py<ExprGraph> {
        self.graph.clone_ref(py)
    }

    /// Pure df/dy (cj = 0, no mass): the bundle's composed `JacobianData`.
    ///
    /// Note: `model.rhs.jacobian()` re-derives its own artifact from the
    /// graph; prefer this composed one (zero extra prep).
    #[getter]
    fn jacobian(&self, py: Python<'_>) -> PyResult<Py<CompiledJacobian>> {
        self.jac_view
            .get_or_try_init(py, || {
                let sig =
                    self.view_signature(py, self.compiled.output_len(), Some("rhs".to_string()))?;
                Py::new(
                    py,
                    CompiledJacobian::build(py, self.compiled.jacobian_data(), sig)?,
                )
            })
            .map(|j| j.clone_ref(py))
    }

    /// Output-variable expressions as shareable `CompiledFunction` views.
    #[getter]
    fn outputs(&self, py: Python<'_>) -> PyResult<Vec<Py<CompiledFunction>>> {
        let views = self.outputs_view.get_or_try_init(py, || {
            self.observable_views(py, ObservableKind::Outputs, &self.output_roots, "output")
        })?;
        Ok(views.iter().map(|f| f.clone_ref(py)).collect())
    }

    /// Event expressions as shareable `CompiledFunction` views.
    #[getter]
    fn events(&self, py: Python<'_>) -> PyResult<Vec<Py<CompiledFunction>>> {
        let views = self.events_view.get_or_try_init(py, || {
            self.observable_views(py, ObservableKind::Events, &self.event_roots, "event")
        })?;
        Ok(views.iter().map(|f| f.clone_ref(py)).collect())
    }

    /// Algebraic residual g(t, y, p) as a shareable view, or `None` for ODEs.
    #[getter]
    fn algebraic_residual(&self, py: Python<'_>) -> PyResult<Option<Py<CompiledFunction>>> {
        let cached = self.algebraic_residual_view.get_or_try_init(py, || {
            let Some(expr) = self.compiled.algebraic_expr_arc() else {
                return PyResult::Ok(None);
            };
            let root = self.algebraic_root.ok_or_else(|| {
                PyValueError::new_err("algebraic_expr implies algebraic_root: internal invariant")
            })?;
            Ok(Some(Py::new(
                py,
                CompiledFunction::from_shared(
                    py,
                    expr,
                    self.graph.clone_ref(py),
                    root,
                    self.compiled.n_states(),
                    Some("algebraic_residual".to_string()),
                )?,
            )?))
        })?;
        Ok(cached.as_ref().map(|f| f.clone_ref(py)))
    }

    /// `dg/dy_alg` as a standalone prepared jacobian (`n_algebraic` x `n_algebraic`), or
    /// `None` for ODEs.
    ///
    /// A view onto the artifact the model already compiled, as `algebraic_residual`
    /// is of the residual, so it costs a shared handle rather than a second
    /// tangent transform, sparsity detection and colouring of the same expression.
    #[getter]
    fn algebraic_jacobian(&self, py: Python<'_>) -> PyResult<Option<Py<CompiledJacobian>>> {
        let cached = self.algebraic_jacobian_view.get_or_try_init(py, || {
            let Some(data) = self.compiled.algebraic_jacobian_data() else {
                return PyResult::Ok(None);
            };
            let sig =
                self.view_signature(py, data.n_rows(), Some("algebraic_residual".to_string()))?;
            Ok(Some(Py::new(py, CompiledJacobian::build(py, data, sig)?)?))
        })?;
        Ok(cached.as_ref().map(|j| j.clone_ref(py)))
    }

    /// Compute residual r = M*y' - f(t,y) and return as a new array.
    ///
    /// This is the DAE residual function used by IDAKLU and other DAE solvers.
    /// For performance-critical code, use `residual_into` to avoid allocation.
    ///
    /// # Arguments
    ///
    /// * `t` - Time value
    /// * `y` - State vector
    /// * `yp` - Time derivative of state vector y'
    /// * `inputs` - Input parameters (can be empty array)
    ///
    /// # Returns
    ///
    /// The residual as a numpy array
    fn eval_residual<'py>(
        &mut self,
        py: Python<'py>,
        t: f64,
        y: PyReadonlyArray1<'_, f64>,
        yp: PyReadonlyArray1<'_, f64>,
        inputs: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let y_slice = y.as_slice()?;
        let yp_slice = yp.as_slice()?;
        let inputs_slice = inputs.as_slice()?;
        let mut output = vec![0.0; self.compiled.output_len()];

        self.with_scratch(|compiled, ws| {
            compiled.eval_residual(ws, t, y_slice, yp_slice, inputs_slice, &mut output);
        });

        Ok(PyArray1::from_vec(py, output))
    }

    /// Compute residual r = M*y' - f(t,y) into a pre-allocated output array.
    ///
    /// This avoids allocation overhead and is preferred for solver hot paths.
    ///
    /// # Arguments
    ///
    /// * `t` - Time value
    /// * `y` - State vector
    /// * `yp` - Time derivative of state vector y'
    /// * `inputs` - Input parameters (can be empty array)
    /// * `output` - Pre-allocated output array (length `n_states`)
    fn eval_residual_into(
        &mut self,
        t: f64,
        y: PyReadonlyArray1<'_, f64>,
        yp: PyReadonlyArray1<'_, f64>,
        inputs: PyReadonlyArray1<'_, f64>,
        mut output: PyReadwriteArray1<'_, f64>,
    ) -> PyResult<()> {
        let y_slice = y.as_slice()?;
        let yp_slice = yp.as_slice()?;
        let inputs_slice = inputs.as_slice()?;
        let output_slice = output.as_slice_mut()?;

        if output_slice.len() < self.compiled.n_states() {
            return Err(PyValueError::new_err(format!(
                "output array too small: need {} elements, got {}",
                self.compiled.n_states(),
                output_slice.len()
            )));
        }

        self.with_scratch(|compiled, ws| {
            compiled.eval_residual(ws, t, y_slice, yp_slice, inputs_slice, output_slice);
        });
        Ok(())
    }

    /// Get the number of states in the model.
    #[getter]
    fn n_states(&self) -> usize {
        self.compiled.n_states()
    }

    /// Get the number of input parameters in the model.
    ///
    /// Core calls these `n_params`; we expose them as `n_inputs` at the Python
    /// boundary to match the C ABI naming.
    #[getter]
    fn n_inputs(&self) -> usize {
        self.compiled.n_params()
    }

    /// Get algebraic-state IDs as a `(n_states,)` numpy array using IDA's
    /// convention: `1.0` for differential states, `0.0` for algebraic states.
    fn algebraic_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let n = self.compiled.n_states();
        let mut buf = vec![0.0; n];
        self.compiled.algebraic_ids_f64(&mut buf);
        PyArray1::from_vec(py, buf)
    }

    /// Whether the model has algebraic sub-block expressions.
    #[getter]
    fn has_algebraic(&self) -> bool {
        self.compiled.has_algebraic()
    }

    /// Number of algebraic states in the compiled sub-block.
    #[getter]
    fn n_algebraic(&self) -> usize {
        self.compiled.n_algebraic()
    }

    /// Number of non-zeros in the assembled algebraic Jacobian.
    #[getter]
    fn algebraic_jacobian_nnz(&self) -> usize {
        self.compiled.algebraic_jacobian_nnz()
    }

    /// Get the algebraic Jacobian sparsity pattern as COO `(rows, cols)`.
    #[allow(clippy::type_complexity)]
    fn algebraic_jacobian_sparsity_pattern<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<usize>>)> {
        if !self.compiled.has_algebraic() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Model has no algebraic block",
            ));
        }
        Ok((
            PyArray1::from_vec(py, self.compiled.algebraic_jacobian_row_indices().to_vec()),
            PyArray1::from_vec(py, self.compiled.algebraic_jacobian_col_indices().to_vec()),
        ))
    }

    /// Number of forward-sensitivity parameters configured.
    #[getter]
    fn n_sens_params(&self) -> usize {
        self.compiled.n_sens_params()
    }

    /// Number of compiled output-variable expressions.
    #[getter]
    fn n_outputs(&self) -> usize {
        self.compiled.n_outputs()
    }

    /// Number of compiled event expressions.
    #[getter]
    fn n_events(&self) -> usize {
        self.compiled.n_events()
    }

    /// Get the output length of f(t, y).
    #[getter]
    fn output_len(&self) -> usize {
        self.compiled.output_len()
    }

    /// Get the sparsity pattern of df/dy as (indptr, indices).
    ///
    /// Returns the sparsity pattern in CSR format for use with
    /// sparse matrix construction.
    #[allow(clippy::type_complexity)]
    fn sparsity_pattern<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<usize>>)> {
        let pattern = self.compiled.sparsity();
        let indptr = PyArray1::from_vec(py, pattern.indptr.clone());
        let indices = PyArray1::from_vec(py, pattern.indices.clone());
        Ok((indptr, indices))
    }

    /// Number of colors in the Jacobian sparsity pattern's graph coloring, each
    /// filling one or more columns in a single sweep. This is the reduced coloring
    /// when a dense-row split was adopted, so it need not cover every row.
    #[getter]
    fn n_colors(&self) -> usize {
        self.compiled.coloring().n_colors
    }

    /// Assemble the Jacobian into a pre-allocated CSC data buffer.
    ///
    /// This is the zero-allocation version for performance-critical code.
    /// The caller must pre-allocate `jac_data` with length `nnz`.
    ///
    /// # Arguments
    ///
    /// * `t` - Time value
    /// * `y` - State vector
    /// * `cj` - Jacobian coefficient from solver (for J = df/dy - cj*M)
    /// * `inputs` - Input parameters (can be empty array)
    /// * `jac_data` - Pre-allocated output buffer in CSC order (length `nnz`)
    fn assemble_jacobian_csc_into(
        &mut self,
        t: f64,
        y: PyReadonlyArray1<'_, f64>,
        cj: f64,
        inputs: PyReadonlyArray1<'_, f64>,
        mut jac_data: PyReadwriteArray1<'_, f64>,
    ) -> PyResult<()> {
        let y_slice = y.as_slice()?;
        let inputs_slice = inputs.as_slice()?;
        let jac_slice = jac_data.as_slice_mut()?;

        let nnz = self.compiled.nnz();
        if jac_slice.len() < nnz {
            return Err(PyValueError::new_err(format!(
                "jac_data buffer too small: need {nnz} elements, got {}",
                jac_slice.len()
            )));
        }

        self.with_scratch(|compiled, ws| {
            ws.set_cj(cj);
            compiled.assemble_jacobian_csc_into(ws, t, y_slice, inputs_slice, jac_slice);
        });
        Ok(())
    }

    /// Get the number of non-zeros in the Jacobian.
    #[getter]
    #[allow(clippy::missing_const_for_fn)] // PyO3 getters can't be const
    fn nnz(&self) -> usize {
        self.compiled.nnz()
    }

    /// Jacobian assembly strategy name this model compiled to.
    #[getter]
    fn jacobian_strategy(&self) -> &'static str {
        self.compiled.jacobian_strategy().as_str()
    }

    /// Jacobian assembly stats for benchmarking and debug attribution.
    fn jacobian_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = self.compiled.jacobian_stats();
        let dict = PyDict::new(py);
        dict.set_item("strategy", stats.strategy.as_str())?;
        dict.set_item("n_colors", stats.n_colors)?;
        dict.set_item("nnz", stats.nnz)?;
        dict.set_item("n_dense_rows", stats.n_dense_rows)?;
        dict.set_item("n_dense_row_candidates", stats.n_dense_row_candidates)?;
        dict.set_item("n_constant_entries", stats.n_constant_entries)?;
        dict.set_item("n_swept_columns", stats.n_swept_columns)?;
        dict.set_item("jac_lane_width", stats.jac_lane_width)?;
        dict.set_item("dense_row_entries", stats.dense_row_entries)?;
        dict.set_item(
            "dense_row_tape_instructions",
            stats.dense_row_tape_instructions,
        )?;
        // primal_end is an index into the raw tape, not an instruction count,
        // so it is left as a raw quantity here.
        dict.set_item(
            "split_eval_primal_instructions",
            stats.split_eval_primal_instructions,
        )?;
        dict.set_item(
            "split_eval_total_instructions",
            stats.split_eval_total_instructions,
        )?;
        dict.set_item(
            "split_eval_raw_instructions",
            stats.split_eval_raw_instructions,
        )?;
        dict.set_item("split_eval_dispatch_count", stats.split_eval_dispatch_count)?;
        dict.set_item(
            "branch_block_lens",
            pyo3::types::PyTuple::new(py, &stats.branch_block_lens)?,
        )?;
        Ok(dict)
    }

    /// `(csc_idx, value)` for every entry proved constant at compile time.
    ///
    /// Indices address the same buffer `assemble_jacobian_csc_into` fills, so
    /// these are the slots no colour sweep writes.
    #[allow(clippy::type_complexity)]
    fn constant_jacobian_entries<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<f64>>) {
        let entries = self.compiled.constant_jacobian_entries();
        let (indices, values): (Vec<usize>, Vec<f64>) = entries.iter().copied().unzip();
        (
            PyArray1::from_vec(py, indices),
            PyArray1::from_vec(py, values),
        )
    }

    /// Get the CSC sparsity pattern for KLU compatibility.
    ///
    /// Returns `(colptr, rowind)` where:
    /// - `colptr` has length `n_states + 1`
    /// - `rowind` has length `nnz`
    #[allow(clippy::type_complexity)]
    fn csc_sparsity_pattern<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<usize>>)> {
        let csc = self.compiled.csc_sparsity();
        let colptr = PyArray1::from_vec(py, csc.colptr.clone());
        let rowind = PyArray1::from_vec(py, csc.rowind.clone());
        Ok((colptr, rowind))
    }

    /// `n` independent evaluators over this model's tape, one per parallel solver.
    ///
    /// Rejects `n == 0` rather than handing back an empty pool: the caller would
    /// then build a solver group with no solvers and divide by its size.
    fn evaluator_pool(&self, n: usize) -> PyResult<EvaluatorPool> {
        if n == 0 {
            return Err(PyValueError::new_err(
                "evaluator_pool needs at least one evaluator, got 0",
            ));
        }
        Ok(EvaluatorPool::from_compiled(&self.compiled, n))
    }
}
