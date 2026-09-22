// PyO3 bindings require specific argument types that clippy flags incorrectly
#![allow(clippy::needless_pass_by_value)]

use std::sync::Arc;

use numpy::{PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyDict;

use pybamm_core::model as core_model;
use pybamm_core::{
    CsrData, NodeId, ObservableKind, PackedWidth, ParamIdx, Shape, resolve_input_width,
};

use crate::compiled_function::CompiledFunction;
use crate::errors::core_err_to_py;
use crate::evaluator_pool::EvaluatorPool;
use crate::expr::{Expr, ExprGraph};
use crate::function_signature::FunctionSignature;
use crate::jacobian::CompiledJacobian;

/// The `_rebuild` argument tuple, in its parameter order.
type RebuildArgs = (
    Py<ExprGraph>,
    u32,
    Vec<u32>,
    Vec<u32>,
    Option<u32>,
    Vec<usize>,
    Vec<f64>,
    Vec<usize>,
    Vec<usize>,
    usize,
    Vec<usize>,
);

/// CSR index entries as `usize`, refusing the negatives an `int64` array admits.
fn csr_indices(values: &[i64], name: &str) -> PyResult<Vec<usize>> {
    values
        .iter()
        .map(|&x| usize::try_from(x))
        .collect::<Result<_, _>>()
        .map_err(|_| PyValueError::new_err(format!("{name} entries must be non-negative")))
}

/// Pair an algebraic root with its variable indices, refusing indices that have no root.
fn algebraic_block(
    root: Option<NodeId>,
    variable_indices: Vec<usize>,
) -> PyResult<Option<(NodeId, Vec<usize>)>> {
    match root {
        Some(root) => Ok(Some((root, variable_indices))),
        None if variable_indices.is_empty() => Ok(None),
        None => Err(PyValueError::new_err(
            "algebraic_variable_indices need an algebraic_expr to index",
        )),
    }
}

/// The Python face of [`core_model::CompiledModel`], with bundle views over its tapes.
#[pyclass(module = "pybamm.rust")]
pub struct CompiledModel {
    pub(crate) compiled: Arc<core_model::CompiledModel>,
    /// Scratch for the direct `eval_*`/`assemble_*` helpers, bound on first use.
    scratch: Option<core_model::Workspace>,
    /// Retained derivation source for the bundle views and `PreparedSolver`.
    graph: Py<ExprGraph>,
    rhs_root: NodeId,
    output_roots: Vec<NodeId>,
    event_roots: Vec<NodeId>,
    /// The algebraic block's root and its strictly-ascending global state indices.
    algebraic: Option<(NodeId, Vec<usize>)>,
    // Cached so every accessor hands back the same prepared artifact.
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
            .field("n_inputs", &self.compiled.input_width().get())
            .field("output_len", &self.compiled.output_len())
            .field("n_colors", &self.compiled.n_colors())
            .finish_non_exhaustive()
    }
}

/// Run `body` against `compiled` and the workspace parked in `slot`, made on first use.
///
/// Taken out for the call and parked again only on return, so no scratch state from an
/// interrupted call survives into the next.
fn with_parked_workspace<R>(
    slot: &mut Option<core_model::Workspace>,
    compiled: &core_model::CompiledModel,
    body: impl FnOnce(&core_model::CompiledModel, &mut core_model::Workspace) -> R,
) -> R {
    let mut workspace = slot.take().unwrap_or_else(|| compiled.create_workspace());
    let result = body(compiled, &mut workspace);
    *slot = Some(workspace);
    result
}

impl CompiledModel {
    /// Run `body` against the artifact and a [`core_model::Workspace`] bound on first use.
    fn with_scratch<R>(
        &mut self,
        body: impl FnOnce(&core_model::CompiledModel, &mut core_model::Workspace) -> R,
    ) -> R {
        with_parked_workspace(&mut self.scratch, &self.compiled, body)
    }

    /// Reject `y`, `yp` or packed `inputs` of another length, which would panic in evaluation.
    fn check_eval_args(
        &self,
        method: &str,
        y: usize,
        yp: Option<usize>,
        inputs: usize,
    ) -> PyResult<()> {
        let n_states = self.compiled.n_states();
        let checks = [
            ("y", Some(y), n_states),
            ("yp", yp, n_states),
            ("inputs", Some(inputs), self.compiled.input_width().get()),
        ];
        for (name, got, expected) in checks {
            if let Some(got) = got
                && got != expected
            {
                return Err(PyValueError::new_err(format!(
                    "CompiledModel.{method}: expected {name} of length {expected}, got {got}"
                )));
            }
        }
        Ok(())
    }

    /// One `CompiledFunction` view per observable of `kind`, named `label[i]`.
    fn observable_views(
        &self,
        py: Python<'_>,
        kind: ObservableKind,
        roots: &[NodeId],
        label: &str,
    ) -> PyResult<Vec<Py<CompiledFunction>>> {
        let exprs = self.compiled.observable_exprs(kind);
        (exprs.zip(roots).enumerate())
            .map(|(i, (expr, &root))| {
                Py::new(
                    py,
                    CompiledFunction::from_shared(
                        py,
                        Arc::clone(expr),
                        self.graph.clone_ref(py),
                        root,
                        self.compiled.n_states(),
                        Some(format!("{label}[{i}]")),
                    )?,
                )
            })
            .collect()
    }

    /// Build a [`FunctionSignature`] for a bundle view (full system width, no `y_dot`).
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
            params: g.params().clone(),
            n_states: self.compiled.n_states(),
            uses_y_dot: false,
            output_len,
            name,
        })
    }

    /// Build from already-resolved roots, the raw CSR mass matrix and the options.
    #[allow(clippy::too_many_arguments)]
    fn build_from_parts(
        py: Python<'_>,
        graph: Py<ExprGraph>,
        rhs_root: NodeId,
        output_roots: Vec<NodeId>,
        event_roots: Vec<NodeId>,
        algebraic: Option<(NodeId, Vec<usize>)>,
        mass_data: Vec<f64>,
        mass_indptr: &[i64],
        mass_indices: &[i64],
        n_inputs: Option<usize>,
        sens_param_indices: Vec<usize>,
    ) -> PyResult<Self> {
        let sens_param_indices = sens_param_indices
            .into_iter()
            .map(|index| {
                u32::try_from(index).map(ParamIdx::new).map_err(|_| {
                    PyValueError::new_err(format!(
                        "sensitivity index {index} names no input parameter"
                    ))
                })
            })
            .collect::<PyResult<Vec<ParamIdx>>>()?;
        // Infer n_states from indptr length
        if mass_indptr.is_empty() {
            return Err(PyValueError::new_err(
                "mass_indptr must have at least 1 element",
            ));
        }
        let n = mass_indptr.len() - 1;

        let indptr = csr_indices(mass_indptr, "mass_indptr")?;
        let indices = csr_indices(mass_indices, "mass_indices")?;
        let mass = CsrData::try_new(indptr, indices, mass_data, Shape::matrix(n, n))
            .map_err(core_err_to_py)?;

        // Mutable throughout: fusing two or more observables allocates a `Concat` in the arena.
        let mut g = graph.try_borrow_mut(py).map_err(|_| {
            PyValueError::new_err("CompiledModel.build_from_parts: graph borrow conflict")
        })?;
        let input_width = {
            let arena = g.arena();
            let roots = std::iter::once(rhs_root)
                .chain(output_roots.iter().copied())
                .chain(event_roots.iter().copied())
                .chain(algebraic.as_ref().map(|&(root, _)| root));
            for root in roots {
                crate::expr::check_supported(arena, root)?;
            }
            resolve_input_width(n_inputs.map(PackedWidth::new), g.params().packed_width())
                .map_err(core_err_to_py)?
        };

        let options = core_model::CompiledModelOptions {
            sensitivities: &sens_param_indices,
            algebraic: algebraic.as_ref().map(|(residual, variable_indices)| {
                core_model::AlgebraicBlockSpec {
                    residual: *residual,
                    variable_indices,
                }
            }),
        };
        let mut compiled =
            core_model::CompiledModel::new(g.arena(), rhs_root, mass, n, input_width, options)
                .map_err(core_err_to_py)?;
        let families = [
            (ObservableKind::Outputs, &output_roots),
            (ObservableKind::Events, &event_roots),
        ];
        for (kind, roots) in families {
            compiled
                .set_observables(kind, g.arena_mut(), roots)
                .map_err(core_err_to_py)?;
        }
        drop(g);

        Ok(Self {
            compiled: Arc::new(compiled),
            scratch: None,
            graph,
            rhs_root,
            output_roots,
            event_roots,
            algebraic,
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
    /// Create a compiled model from an expression graph and a CSR mass matrix.
    ///
    /// `n_inputs` defaults to the graph's registered packed width; a smaller one is refused.
    #[staticmethod]
    #[pyo3(signature = (
        graph,
        expr,
        mass_data,
        mass_indptr,
        mass_indices,
        n_inputs = None,
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
        n_inputs: Option<usize>,
        sens_param_indices: Vec<usize>,
        output_exprs: Vec<PyRef<'_, Expr>>,
        algebraic_expr: Option<&Expr>,
        algebraic_variable_indices: Vec<usize>,
        event_exprs: Vec<PyRef<'_, Expr>>,
    ) -> PyResult<Self> {
        let mass_data: Vec<f64> = mass_data.as_slice()?.to_vec();
        let mass_indptr = mass_indptr.as_slice()?;
        let mass_indices = mass_indices.as_slice()?;

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
        let algebraic = algebraic_block(algebraic_root, algebraic_variable_indices)?;

        Self::build_from_parts(
            py,
            graph,
            rhs_root,
            output_roots,
            event_roots,
            algebraic,
            mass_data,
            mass_indptr,
            mass_indices,
            n_inputs,
            sens_param_indices,
        )
    }

    /// Rebuild from the retained graph, roots and mass CSR (pickle protocol).
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
        let (rhs_root, output_roots, event_roots, algebraic_root) = {
            let g = graph.try_borrow(py).map_err(|_| {
                PyValueError::new_err("CompiledModel._rebuild: graph borrow conflict")
            })?;
            let checked = |roots: Vec<u32>| -> PyResult<Vec<NodeId>> {
                roots.into_iter().map(|raw| g.checked_root(raw)).collect()
            };
            (
                g.checked_root(rhs_root)?,
                checked(output_roots)?,
                checked(event_roots)?,
                algebraic_root.map(|raw| g.checked_root(raw)).transpose()?,
            )
        };
        let algebraic = algebraic_block(algebraic_root, algebraic_variable_indices)?;

        Self::build_from_parts(
            py,
            graph,
            rhs_root,
            output_roots,
            event_roots,
            algebraic,
            mass_data,
            &mass_indptr,
            &mass_indices,
            Some(n_inputs),
            sens_param_indices,
        )
    }

    /// `(callable, args)` pickle pair that rebuilds from the retained graph and roots.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, RebuildArgs)> {
        let this = slf.borrow();
        let rebuild = slf.get_type().getattr("_rebuild")?;
        let mass = this.compiled.mass_matrix();
        let (algebraic_root, algebraic_variable_indices) = this
            .algebraic
            .as_ref()
            .map_or((None, Vec::new()), |(root, indices)| {
                (Some(root.raw()), indices.clone())
            });
        Ok((
            rebuild,
            (
                this.graph.clone_ref(py),
                this.rhs_root.raw(),
                this.output_roots.iter().map(|r| r.raw()).collect(),
                this.event_roots.iter().map(|r| r.raw()).collect(),
                algebraic_root,
                algebraic_variable_indices,
                mass.data().to_vec(),
                mass.indptr().to_vec(),
                mass.indices().to_vec(),
                this.compiled.input_width().get(),
                this.compiled
                    .sens_param_indices()
                    .iter()
                    .copied()
                    .map(ParamIdx::index)
                    .collect::<Vec<usize>>(),
            ),
        ))
    }

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

    /// The retained derivation graph; roots lowered into it share the solve's indices.
    #[getter]
    fn graph(&self, py: Python<'_>) -> Py<ExprGraph> {
        self.graph.clone_ref(py)
    }

    /// Pure df/dy (cj = 0, no mass): the bundle's composed `JacobianData`.
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
            let Some((block, &(root, _))) = self.compiled.algebraic().zip(self.algebraic.as_ref())
            else {
                return PyResult::Ok(None);
            };
            Ok(Some(Py::new(
                py,
                CompiledFunction::from_shared(
                    py,
                    Arc::clone(block.residual_expr()),
                    self.graph.clone_ref(py),
                    root,
                    self.compiled.n_states(),
                    Some("algebraic_residual".to_string()),
                )?,
            )?))
        })?;
        Ok(cached.as_ref().map(|f| f.clone_ref(py)))
    }

    /// `dg/dy_alg` as a prepared jacobian (`n_algebraic` x `n_algebraic`), or `None` for ODEs.
    #[getter]
    fn algebraic_jacobian(&self, py: Python<'_>) -> PyResult<Option<Py<CompiledJacobian>>> {
        let cached = self.algebraic_jacobian_view.get_or_try_init(py, || {
            let Some(block) = self.compiled.algebraic() else {
                return PyResult::Ok(None);
            };
            let data = Arc::clone(block.jacobian());
            let sig =
                self.view_signature(py, data.n_rows(), Some("algebraic_residual".to_string()))?;
            Ok(Some(Py::new(py, CompiledJacobian::build(py, data, sig)?)?))
        })?;
        Ok(cached.as_ref().map(|j| j.clone_ref(py)))
    }

    /// Compute residual r = M*y' - f(t,y) and return as a new array.
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
        self.check_eval_args(
            "eval_residual",
            y_slice.len(),
            Some(yp_slice.len()),
            inputs_slice.len(),
        )?;
        let mut output = vec![0.0; self.compiled.output_len()];

        self.with_scratch(|compiled, ws| {
            compiled.eval_residual(ws, t, y_slice, yp_slice, inputs_slice, &mut output);
        });

        Ok(PyArray1::from_vec(py, output))
    }

    /// Compute residual r = M*y' - f(t,y) into a pre-allocated output array.
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
        self.check_eval_args(
            "eval_residual_into",
            y_slice.len(),
            Some(yp_slice.len()),
            inputs_slice.len(),
        )?;
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

    /// Get the packed width of the model's input parameters.
    #[getter]
    fn n_inputs(&self) -> usize {
        self.compiled.input_width().get()
    }

    /// Algebraic-state IDs in IDA's convention: `1.0` differential, `0.0` algebraic.
    fn algebraic_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let n = self.compiled.n_states();
        let mut buf = vec![0.0; n];
        self.compiled.state_kinds().write_ida_id(&mut buf);
        PyArray1::from_vec(py, buf)
    }

    /// Whether the model has algebraic sub-block expressions.
    #[getter]
    fn has_algebraic(&self) -> bool {
        self.compiled.algebraic().is_some()
    }

    /// Number of algebraic states in the compiled sub-block, or 0 without one.
    #[getter]
    fn n_algebraic(&self) -> usize {
        self.compiled
            .algebraic()
            .map_or(0, core_model::AlgebraicBlock::n_variables)
    }

    /// Number of non-zeros in the assembled algebraic Jacobian, or 0 without a block.
    #[getter]
    fn algebraic_jacobian_nnz(&self) -> usize {
        self.compiled
            .algebraic()
            .map_or(0, |block| block.jacobian().nnz())
    }

    /// Get the algebraic Jacobian sparsity pattern as COO `(rows, cols)`.
    #[allow(clippy::type_complexity)]
    fn algebraic_jacobian_sparsity_pattern<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<usize>>)> {
        let Some(block) = self.compiled.algebraic() else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Model has no algebraic block",
            ));
        };
        let (rows, cols) = block.jacobian_coo();
        Ok((
            PyArray1::from_vec(py, rows.to_vec()),
            PyArray1::from_vec(py, cols.to_vec()),
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

    /// CSR `(indptr, indices)` of `df/dy` merged with the mass pattern.
    #[allow(clippy::type_complexity)]
    fn sparsity_pattern<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<usize>>)> {
        let pattern = self.compiled.sparsity();
        let indptr = PyArray1::from_vec(py, pattern.indptr().to_vec());
        let indices = PyArray1::from_vec(py, pattern.indices().to_vec());
        Ok((indptr, indices))
    }

    /// Number of colors in the Jacobian coloring, reduced when dense rows are split out.
    #[getter]
    fn n_colors(&self) -> usize {
        self.compiled.n_colors()
    }

    /// Assemble J = df/dy - cj*M into a pre-allocated CSC data buffer of length `nnz`.
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
        self.check_eval_args(
            "assemble_jacobian_csc_into",
            y_slice.len(),
            None,
            inputs_slice.len(),
        )?;
        let jac_slice = jac_data.as_slice_mut()?;

        let nnz = self.compiled.nnz();
        if jac_slice.len() < nnz {
            return Err(PyValueError::new_err(format!(
                "jac_data buffer too small: need {nnz} elements, got {}",
                jac_slice.len()
            )));
        }

        self.with_scratch(|compiled, ws| {
            compiled.assemble_jacobian_csc_into(ws, t, y_slice, inputs_slice, cj, jac_slice);
        });
        Ok(())
    }

    /// Get the number of non-zeros in the Jacobian.
    #[getter]
    #[allow(clippy::missing_const_for_fn)] // PyO3 getters can't be const
    fn nnz(&self) -> usize {
        self.compiled.nnz()
    }

    /// Jacobian assembly stats for benchmarking and debug attribution.
    fn jacobian_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = self.compiled.jacobian_stats();
        let dict = PyDict::new(py);
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

    /// Get the CSC sparsity pattern as `(colptr, rowind)` for KLU.
    #[allow(clippy::type_complexity)]
    fn csc_sparsity_pattern<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<usize>>)> {
        let csc = self.compiled.csc_sparsity();
        let colptr = PyArray1::from_vec(py, csc.colptr().to_vec());
        let rowind = PyArray1::from_vec(py, csc.rowind().to_vec());
        Ok((colptr, rowind))
    }

    /// `n` independent evaluators over this model's tape, one per parallel solver.
    fn evaluator_pool(&self, n: usize) -> PyResult<EvaluatorPool> {
        if n == 0 {
            return Err(PyValueError::new_err(
                "evaluator_pool needs at least one evaluator, got 0",
            ));
        }
        Ok(EvaluatorPool::from_compiled(&self.compiled, n))
    }
}
