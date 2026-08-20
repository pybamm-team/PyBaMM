//! `CompiledFunctionGroup`: shared-tape multi-output artifact.

// PyO3 bindings require specific argument types that clippy flags incorrectly
#![allow(clippy::needless_pass_by_value)]

use std::sync::Arc;

use numpy::ndarray::ShapeBuilder;
use numpy::{
    AllowTypeChange, PyArray1, PyArray2, PyArrayLike1, PyReadonlyArray1, PyReadonlyArray2,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use pybamm_core::{CompiledExpr, Node, NodeId, TypedIr, scan_state_usage, simplify_pipeline};

use crate::expr::ExprGraph;
use crate::scratch::{Buffer, ScratchPool};
use crate::signature::FunctionSignature;

/// A named set of outputs compiled into ONE shared tape. Cross-output
/// common subexpressions are evaluated once; per-output results are recovered via
/// recorded `(offset, len)` slices into the synthetic concat root.
#[pyclass(frozen, module = "pybamm.rust")]
pub struct CompiledFunctionGroup {
    expr: Arc<CompiledExpr>,
    sig: FunctionSignature,
    pool: ScratchPool<Buffer>,
    out_names: Vec<String>,
    /// Per-output (offset, len) into the concat result.
    slices: Vec<(usize, usize)>,
}

impl std::fmt::Debug for CompiledFunctionGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledFunctionGroup")
            .field("sig", &self.sig)
            .field("out_names", &self.out_names)
            .finish_non_exhaustive()
    }
}

impl CompiledFunctionGroup {
    pub(crate) fn build(
        py: Python<'_>,
        graph: Py<ExprGraph>,
        names: Vec<String>,
        ids: Vec<NodeId>,
        name: Option<String>,
        n_states_override: Option<usize>,
    ) -> PyResult<Self> {
        let g = graph
            .try_borrow(py)
            .map_err(|_| PyValueError::new_err("compile_group: graph borrow conflict"))?;
        // Per-output lengths from per-child IRs (build-time only).
        let lens: Vec<usize> = ids
            .iter()
            .map(|&id| TypedIr::from_arena(g.arena(), id).output_len())
            .collect();
        // The synthetic concat root lives on a clone: compiling a group
        // must not grow the caller's graph.
        let mut work = g.arena().clone();
        let input_names = g.input_names();
        let input_widths = g.input_widths();
        drop(g);
        let root = work.alloc(Node::Concat(ids));
        // Scan the ORIGINAL root (pre-simplify): the signature is the
        // user-declared contract.
        let usage = scan_state_usage(&work, root);
        // Running the pipeline over the combined reachable set gives cross-output
        // CSE for structurally-identical nodes, not just shared NodeIds.
        let (work, root) = simplify_pipeline(work, root);
        let expr = Arc::new(CompiledExpr::from_ir(TypedIr::from_arena(&work, root)));

        let mut slices = Vec::with_capacity(lens.len());
        let mut offset = 0;
        for &len in &lens {
            slices.push((offset, len));
            offset += len;
        }
        if offset != expr.output_len() {
            return Err(PyValueError::new_err(format!(
                "compile_group: internal error: output slices tile {} values \
                 but the compiled tape produces {}",
                offset,
                expr.output_len()
            )));
        }
        let sig = FunctionSignature {
            input_names,
            input_widths,
            n_states: crate::function::resolve_n_states(
                usage.n_states,
                n_states_override,
                name.as_deref(),
            )?,
            uses_y_dot: usage.uses_y_dot,
            output_len: expr.output_len(),
            name,
        };
        let pool = ScratchPool::new(Buffer(expr.scratch_len()));
        Ok(Self {
            expr,
            sig,
            pool,
            out_names: names,
            slices,
        })
    }

    fn eval_columns(
        &self,
        py: Python<'_>,
        ts: &[f64],
        y_cols: &[f64],
        n_rows: usize,
        packed: &[f64],
    ) -> Vec<f64> {
        let n_t = ts.len();
        let total = self.sig.output_len;
        let mut out = vec![0.0; total * n_t];
        let expr = &self.expr;
        let mut scratch = self.pool.acquire();
        py.detach(|| {
            for j in 0..n_t {
                let y = &y_cols[j * n_rows..(j + 1) * n_rows];
                let dst = &mut out[j * total..(j + 1) * total];
                expr.eval_into(&mut scratch, ts[j], y, &[], packed, dst);
            }
        });
        self.pool.release(scratch);
        out
    }
}

#[pymethods]
impl CompiledFunctionGroup {
    /// Evaluate all outputs once over the shared tape; returns a list of
    /// arrays in declared order.
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        y: PyReadonlyArray1<'_, f64>,
        p: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<Bound<'py, PyArray1<f64>>>> {
        // The eval paths slice an empty y_dot, which the tape would index and panic
        // on. Build stays allowed so signature introspection still works.
        self.sig.reject_y_dot("group eval")?;
        let packed = self.sig.extract_p(p)?;
        let y_slice = y.as_slice()?;
        self.sig.check_y(y_slice.len())?;

        let mut full = vec![0.0; self.sig.output_len];
        let mut scratch = self.pool.acquire();
        self.expr
            .eval_into(&mut scratch, t, y_slice, &[], &packed, &mut full);
        self.pool.release(scratch);

        Ok(self
            .slices
            .iter()
            .map(|&(off, len)| PyArray1::from_slice(py, &full[off..off + len]))
            .collect())
    }

    /// Trajectory sweep over the shared tape: one crossing, one tape eval
    /// per column regardless of output count.
    fn eval_trajectory<'py>(
        &self,
        py: Python<'py>,
        ts: PyArrayLike1<'_, f64, AllowTypeChange>,
        y_traj: PyReadonlyArray2<'_, f64>,
        p: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
        self.sig.reject_y_dot("group eval")?;
        let packed = self.sig.extract_p(p)?;
        let ts_view = ts.as_array();
        let ts_slice = crate::function::contiguous_slice_1d(&ts_view);
        let view = y_traj.as_array();
        let (n_rows, n_t) = (view.shape()[0], view.shape()[1]);
        if n_rows != self.sig.n_states {
            return Err(PyValueError::new_err(format!(
                "{}: Y.shape[0] must equal n_states ({}), got {}",
                self.sig.display_name(),
                self.sig.n_states,
                n_rows
            )));
        }
        if ts_slice.len() != n_t {
            return Err(PyValueError::new_err(format!(
                "{}: len(ts) ({}) must equal Y.shape[1] ({})",
                self.sig.display_name(),
                ts_slice.len(),
                n_t
            )));
        }
        let y_cols = crate::function::columns_slice(&view);

        let flat = self.eval_columns(py, &ts_slice, &y_cols, n_rows, &packed);
        let total = self.sig.output_len;
        self.slices
            .iter()
            .map(|&(off, len)| {
                let mut data = Vec::with_capacity(len * n_t);
                for j in 0..n_t {
                    data.extend_from_slice(&flat[j * total + off..j * total + off + len]);
                }
                let arr =
                    numpy::ndarray::Array2::from_shape_vec((len, n_t).f(), data).map_err(|e| {
                        PyValueError::new_err(format!("eval_trajectory shape error: {e}"))
                    })?;
                Ok(PyArray2::from_owned_array(py, arr))
            })
            .collect()
    }

    /// Cubic-Hermite reconstruct the state at each `t_query` from the solver
    /// knots, then evaluate the shared tape once per query and slice per
    /// output. API parity with `CompiledFunction::eval_trajectory_hermite`;
    /// full-state observation routes through per-variable `CompiledFunction`,
    /// not this group method.
    #[pyo3(signature = (t_query, ts, ys, yps, p))]
    #[allow(clippy::suboptimal_flops)] // mixed-sign cubic-Hermite basis, mirrors function.rs
    fn eval_trajectory_hermite<'py>(
        &self,
        py: Python<'py>,
        t_query: PyArrayLike1<'_, f64, AllowTypeChange>,
        ts: PyArrayLike1<'_, f64, AllowTypeChange>,
        ys: PyReadonlyArray2<'_, f64>,
        yps: PyReadonlyArray2<'_, f64>,
        p: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
        self.sig.reject_y_dot("group eval_trajectory_hermite")?;
        let packed = self.sig.extract_p(p)?;
        let tq_view = t_query.as_array();
        let tq = crate::function::contiguous_slice_1d(&tq_view);
        let ts_view = ts.as_array();
        let ts_slice = crate::function::contiguous_slice_1d(&ts_view);
        let y_view = ys.as_array();
        let yp_view = yps.as_array();
        let (n_rows, n_knots) = (y_view.shape()[0], y_view.shape()[1]);
        if n_rows != self.sig.n_states {
            return Err(PyValueError::new_err(format!(
                "{}: Y.shape[0] must equal n_states ({}), got {}",
                self.sig.display_name(),
                self.sig.n_states,
                n_rows
            )));
        }
        if ts_slice.len() != n_knots {
            return Err(PyValueError::new_err(format!(
                "{}: len(ts) ({}) must equal Y.shape[1] ({})",
                self.sig.display_name(),
                ts_slice.len(),
                n_knots
            )));
        }
        if yp_view.shape()[0] != n_rows || yp_view.shape()[1] != n_knots {
            return Err(PyValueError::new_err(format!(
                "{}: yps shape must equal ys shape ({}, {}), got ({}, {})",
                self.sig.display_name(),
                n_rows,
                n_knots,
                yp_view.shape()[0],
                yp_view.shape()[1]
            )));
        }
        if n_knots < 2 {
            return Err(PyValueError::new_err(format!(
                "{}: need >= 2 knots for Hermite interpolation, got {}",
                self.sig.display_name(),
                n_knots
            )));
        }

        // Zero-copy when Y/YP are already F-contiguous; one gathering copy otherwise.
        let y_cols = crate::function::columns_slice(&y_view);
        let yp_cols = crate::function::columns_slice(&yp_view);

        let total = self.sig.output_len;
        let n_query = tq.len();
        let mut flat = vec![0.0_f64; total * n_query];
        let expr = &self.expr;
        let mut scratch = self.pool.acquire();
        let mut y_interp = vec![0.0_f64; n_rows];
        py.detach(|| {
            for q in 0..n_query {
                let i = crate::function::locate_interval(&ts_slice, tq[q]);
                let h = ts_slice[i + 1] - ts_slice[i];
                let s = if h > 0.0 {
                    (tq[q] - ts_slice[i]) / h
                } else {
                    0.0
                };
                let (s2, s3) = (s * s, s * s * s);
                // cubic-Hermite basis; derivative terms scaled by the step h
                let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
                let h10 = s3 - 2.0 * s2 + s;
                let h01 = -2.0 * s3 + 3.0 * s2;
                let h11 = s3 - s2;
                let yi = &y_cols[i * n_rows..(i + 1) * n_rows];
                let yi1 = &y_cols[(i + 1) * n_rows..(i + 2) * n_rows];
                let ypi = &yp_cols[i * n_rows..(i + 1) * n_rows];
                let ypi1 = &yp_cols[(i + 1) * n_rows..(i + 2) * n_rows];
                for m in 0..n_rows {
                    y_interp[m] = h00 * yi[m] + h10 * h * ypi[m] + h01 * yi1[m] + h11 * h * ypi1[m];
                }
                let dst = &mut flat[q * total..(q + 1) * total];
                expr.eval_into(&mut scratch, tq[q], &y_interp, &[], &packed, dst);
            }
        });
        self.pool.release(scratch);

        self.slices
            .iter()
            .map(|&(off, len)| {
                let mut data = Vec::with_capacity(len * n_query);
                for q in 0..n_query {
                    data.extend_from_slice(&flat[q * total + off..q * total + off + len]);
                }
                let arr = numpy::ndarray::Array2::from_shape_vec((len, n_query).f(), data)
                    .map_err(|e| {
                        PyValueError::new_err(format!("eval_trajectory_hermite shape error: {e}"))
                    })?;
                Ok(PyArray2::from_owned_array(py, arr))
            })
            .collect()
    }

    #[getter]
    fn names<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        pyo3::types::PyTuple::new(py, &self.out_names)
    }
    #[getter]
    fn output_lens(&self) -> Vec<usize> {
        self.slices.iter().map(|&(_, len)| len).collect()
    }
    /// Pack a {name: value} mapping into the stacked input layout
    /// (same signature surface as `CompiledFunction`).
    fn pack<'py>(
        &self,
        py: Python<'py>,
        mapping: &Bound<'_, pyo3::types::PyDict>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(PyArray1::from_vec(py, self.sig.pack(mapping)?))
    }

    #[getter]
    fn input_names<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        pyo3::types::PyTuple::new(py, &self.sig.input_names)
    }
    /// Registered-name count (the `vp`/parameter-tangent seed length), NOT
    /// the packed input width, unlike `ExprGraph::n_inputs`.
    #[getter]
    const fn n_inputs(&self) -> usize {
        self.sig.input_names.len()
    }
    #[getter]
    const fn n_states(&self) -> usize {
        self.sig.n_states
    }
    #[getter]
    const fn output_len(&self) -> usize {
        self.sig.output_len
    }
    #[getter]
    const fn uses_y_dot(&self) -> bool {
        self.sig.uses_y_dot
    }
    /// Instruction count excluding conditional branch blocks: the common tape
    /// plus one dispatch per conditional. Makes cross-output CSE observable and
    /// is directly comparable to `casadi.Function.n_instructions()`.
    #[getter]
    fn n_instructions(&self) -> usize {
        self.expr.ir().common_instruction_count()
    }
    /// Raw tape length, branch blocks included.
    #[getter]
    fn n_instructions_total(&self) -> usize {
        self.expr.ir().instructions().len()
    }
    /// Per-branch block lengths, in tape order.
    #[getter]
    fn branch_block_lens<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        pyo3::types::PyTuple::new(py, self.expr.ir().branch_block_lens())
    }
    #[getter]
    fn name(&self) -> Option<String> {
        self.sig.name.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "CompiledFunctionGroup(name={:?}, outputs={:?}, n_states={})",
            self.sig.display_name(),
            self.out_names,
            self.sig.n_states
        )
    }
}
