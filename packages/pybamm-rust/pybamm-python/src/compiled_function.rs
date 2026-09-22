//! `CompiledFunction`: the central prepared artifact.

// PyO3 bindings require specific argument types that clippy flags incorrectly
#![allow(clippy::needless_pass_by_value)]

use std::sync::{Arc, OnceLock};

use numpy::ndarray::ShapeBuilder;
use numpy::{
    AllowTypeChange, PyArray1, PyArray2, PyArrayLike1, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArray2,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyDict;

use pybamm_core::{
    CompiledExpr, DagRef, Node, NodeId, ReadExtents, TangentInputs, dce, infer_sizes,
    simplify_pipeline, tangent_wrt_params, tangent_wrt_states,
};

use crate::expr::ExprGraph;
use crate::function_metadata::FunctionMetadata;
use crate::jacobian::CompiledJacobian;
use crate::scratch_pool::{ScratchPool, buffer};

/// Target working-set size (~1 MiB) for a lane-batched trajectory tile.
const TARGET_BYTES: usize = 1 << 20;

/// Memoised tangent tape plus its scratch pool (one per seed class).
struct TangentEntry {
    expr: CompiledExpr,
    pool: ScratchPool<Box<[f64]>>,
}

/// The `_rebuild` argument tuple, in its parameter order.
type RebuildArgs = (
    Py<ExprGraph>,
    u32,
    Option<String>,
    Option<usize>,
    Option<Vec<String>>,
    Option<Vec<usize>>,
);

/// Which rows of a multi-output tape belong to which declared name.
#[derive(Debug, Clone)]
struct OutputLayout {
    names: Vec<String>,
    lens: Vec<usize>,
}

impl OutputLayout {
    fn new(names: Vec<String>, lens: Vec<usize>, output_len: usize) -> PyResult<Self> {
        if names.len() != lens.len() {
            return Err(PyValueError::new_err(format!(
                "output layout: {} name(s) but {} length(s)",
                names.len(),
                lens.len()
            )));
        }
        // Checked, so pickled lengths cannot wrap around to the tape's length.
        let Some(total) = lens
            .iter()
            .try_fold(0_usize, |sum, &len| sum.checked_add(len))
        else {
            return Err(PyValueError::new_err(
                "output layout: the declared lengths overflow",
            ));
        };
        if total != output_len {
            return Err(PyValueError::new_err(format!(
                "output layout: the declared lengths tile {total} value(s) but the \
                 compiled tape produces {output_len}"
            )));
        }
        Ok(Self { names, lens })
    }

    /// `(offset, len)` per output, in declared order.
    fn spans(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.lens.iter().scan(0, |offset, &len| {
            let start = *offset;
            *offset += len;
            Some((start, len))
        })
    }
}

/// One expression compiled into one tape: `f(t, y, p)`, plus `y_dot` when read.
///
/// `p` is a name-keyed dict or an already-packed array; derivatives are memoised.
#[pyclass(frozen, module = "pybamm.rust")]
pub struct CompiledFunction {
    expr: Arc<CompiledExpr>,
    metadata: FunctionMetadata,
    pool: ScratchPool<Box<[f64]>>,
    /// Retained for lazy derivation (jacobian/jvp).
    graph: Py<ExprGraph>,
    root: NodeId,
    /// Row layout when compiled from named outputs.
    layout: Option<OutputLayout>,
    /// Tangent tapes, derived on first use: `OnceLock` as this frozen class fills them via `&self`.
    tangent_y: OnceLock<TangentEntry>,
    tangent_p: OnceLock<TangentEntry>,
    jac_y: PyOnceLock<Py<CompiledJacobian>>,
    jac_p: PyOnceLock<Py<CompiledJacobian>>,
}

impl std::fmt::Debug for CompiledFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledFunction")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

impl CompiledFunction {
    pub(crate) fn build(
        py: Python<'_>,
        graph: Py<ExprGraph>,
        root: NodeId,
        name: Option<String>,
        n_states_override: Option<usize>,
    ) -> PyResult<Self> {
        let g = graph.try_borrow(py).map_err(|_| {
            PyValueError::new_err(format!(
                "{}: graph borrow conflict while compiling",
                name.as_deref().unwrap_or("<anonymous>")
            ))
        })?;
        // Scan before simplifying, which can shrink the reachable state extent.
        let extents = ReadExtents::scan(g.arena(), root);
        let n_states = resolve_n_states(extents.n_states(), n_states_override, name.as_deref())?;
        let params = g.params().clone();
        // A reachable copy, so the retained graph stays the unsimplified source for jvp/jacobian.
        let dag = dce(DagRef::new(g.arena(), root));
        drop(g);
        let expr = Arc::new(py.detach(|| CompiledExpr::new(simplify_pipeline(dag).as_ref())));
        let metadata = FunctionMetadata {
            params,
            n_states,
            uses_y_dot: extents.y_dot.is_some(),
            output_len: expr.output_len(),
            name,
        };
        Ok(Self {
            expr,
            metadata,
            pool: ScratchPool::default(),
            graph,
            root,
            layout: None,
            tangent_y: OnceLock::new(),
            tangent_p: OnceLock::new(),
            jac_y: PyOnceLock::new(),
            jac_p: PyOnceLock::new(),
        })
    }

    /// Compile named outputs into one tape via a `Concat`, so shared subexpressions run once.
    pub(crate) fn build_outputs(
        py: Python<'_>,
        graph: Py<ExprGraph>,
        names: Vec<String>,
        ids: Vec<NodeId>,
        name: Option<String>,
        n_states_override: Option<usize>,
    ) -> PyResult<Self> {
        let (root, lens) = {
            let mut g = graph
                .try_borrow_mut(py)
                .map_err(|_| PyValueError::new_err("compile_outputs: graph borrow conflict"))?;
            let root = g.arena_mut().alloc(Node::Concat(ids.clone()));
            // The concat tape only reports the total length; one size pass covers every output.
            let sizes = infer_sizes(g.arena(), &g.arena().topological_order(root));
            (root, ids.iter().map(|id| sizes[id.index()]).collect())
        };
        let mut this = Self::build(py, graph, root, name, n_states_override)?;
        this.layout = Some(OutputLayout::new(names, lens, this.metadata.output_len)?);
        Ok(this)
    }

    /// Bundle-view constructor: shares an existing tape (no recompilation).
    pub(crate) fn from_shared(
        py: Python<'_>,
        expr: Arc<CompiledExpr>,
        graph: Py<ExprGraph>,
        root: NodeId,
        n_states: usize,
        name: Option<String>,
    ) -> PyResult<Self> {
        let g = graph.try_borrow(py).map_err(|_| {
            PyValueError::new_err("from_shared: graph borrow conflict while preparing bundle view")
        })?;
        let metadata = FunctionMetadata {
            params: g.params().clone(),
            n_states, // bundle views always carry the full system width
            uses_y_dot: ReadExtents::scan(g.arena(), root).y_dot.is_some(),
            output_len: expr.output_len(),
            name,
        };
        drop(g);
        Ok(Self {
            expr,
            metadata,
            pool: ScratchPool::default(),
            graph,
            root,
            layout: None,
            tangent_y: OnceLock::new(),
            tangent_p: OnceLock::new(),
            jac_y: PyOnceLock::new(),
            jac_p: PyOnceLock::new(),
        })
    }

    fn eval_inner(
        &self,
        t: f64,
        y: &[f64],
        p: &[f64],
        y_dot: Option<&[f64]>,
        out: &mut [f64],
    ) -> PyResult<()> {
        self.metadata
            .check_len("y", y.len(), self.metadata.n_states)?;
        if self.metadata.uses_y_dot {
            match y_dot {
                Some(yd) => self
                    .metadata
                    .check_len("y_dot", yd.len(), self.metadata.n_states)?,
                None => {
                    return Err(PyValueError::new_err(format!(
                        "{}: expression uses y_dot; pass y_dot=",
                        self.metadata.display_name()
                    )));
                },
            }
        }
        let mut scratch = self.pool.acquire(|| buffer(self.expr.scratch_len()));
        self.expr
            .eval_into(&mut scratch, t, y, y_dot.unwrap_or(&[]), p, out);
        self.pool.release(scratch);
        Ok(())
    }

    /// Lazily derive and cache the tangent tape + scratch pool for one seed class.
    fn tangent_entry(&self, py: Python<'_>, wrt_params: bool) -> PyResult<&TangentEntry> {
        let cell = if wrt_params {
            &self.tangent_p
        } else {
            &self.tangent_y
        };
        if let Some(cached) = cell.get() {
            return Ok(cached);
        }
        let g = self.graph.try_borrow(py).map_err(|_| {
            PyValueError::new_err(format!(
                "{}: graph borrow conflict during tangent derivation",
                self.metadata.display_name()
            ))
        })?;
        let mut tangent = dce(DagRef::new(g.arena(), self.root));
        drop(g);
        let expr = py.detach(|| {
            tangent.rewrite_in_place(if wrt_params {
                tangent_wrt_params
            } else {
                tangent_wrt_states
            });
            CompiledExpr::new(simplify_pipeline(tangent).as_ref())
        });
        Ok(cell.get_or_init(|| TangentEntry {
            expr,
            pool: ScratchPool::default(),
        }))
    }
}

#[pymethods]
impl CompiledFunction {
    #[pyo3(signature = (t, y, p, y_dot = None))]
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        y: PyReadonlyArray1<'_, f64>,
        p: &Bound<'_, PyAny>,
        y_dot: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let packed = self.metadata.extract_p(p)?;
        let mut out = vec![0.0; self.metadata.output_len];
        let yd = y_dot.as_ref().map(|a| a.as_slice()).transpose()?;
        self.eval_inner(t, y.as_slice()?, &packed, yd, &mut out)?;
        Ok(PyArray1::from_vec(py, out))
    }

    /// Alias for `__call__`.
    #[pyo3(signature = (t, y, p, y_dot = None))]
    fn eval<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        y: PyReadonlyArray1<'_, f64>,
        p: &Bound<'_, PyAny>,
        y_dot: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.__call__(py, t, y, p, y_dot)
    }

    /// Evaluate into a pre-allocated output array (no intermediate allocation).
    #[pyo3(signature = (t, y, p, out, y_dot = None))]
    fn eval_into(
        &self,
        t: f64,
        y: PyReadonlyArray1<'_, f64>,
        p: &Bound<'_, PyAny>,
        out: &Bound<'_, PyArray1<f64>>,
        y_dot: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<()> {
        // Fails when `out` is read-only or overlaps `y` or `y_dot`, which are already borrowed
        let mut out = out.try_readwrite().map_err(|err| {
            PyValueError::new_err(format!(
                "{}: out must be a writeable array that shares no memory with y, p or \
                 y_dot ({err})",
                self.metadata.display_name()
            ))
        })?;
        let packed = self.metadata.extract_p(p)?;
        let out_slice = out.as_slice_mut()?;
        self.metadata
            .check_len("out", out_slice.len(), self.metadata.output_len)?;
        let yd = y_dot.as_ref().map(|a| a.as_slice()).transpose()?;
        self.eval_inner(t, y.as_slice()?, &packed, yd, out_slice)
    }

    /// Pack a {name: value} mapping into the stacked input layout.
    fn pack<'py>(
        &self,
        py: Python<'py>,
        mapping: &Bound<'_, PyDict>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(PyArray1::from_vec(py, self.metadata.pack(mapping)?))
    }

    #[getter]
    fn input_names<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        pyo3::types::PyTuple::new(py, self.metadata.params.names())
    }
    /// Registration-space extent (the `vp` seed length), not the packed input width.
    #[getter]
    const fn n_inputs(&self) -> usize {
        self.metadata.params.index_extent()
    }
    #[getter]
    const fn n_states(&self) -> usize {
        self.metadata.n_states
    }
    #[getter]
    const fn output_len(&self) -> usize {
        self.metadata.output_len
    }
    /// Declared output names, or `None` for a single-expression function.
    #[getter]
    fn output_names<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, pyo3::types::PyTuple>>> {
        self.layout
            .as_ref()
            .map(|layout| pyo3::types::PyTuple::new(py, &layout.names))
            .transpose()
    }
    /// Rows each declared output owns, ordered as `output_names`.
    #[getter]
    fn output_lens<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, pyo3::types::PyTuple>>> {
        self.layout
            .as_ref()
            .map(|layout| pyo3::types::PyTuple::new(py, &layout.lens))
            .transpose()
    }
    /// One view (not a copy) per declared output, sliced along the leading axis.
    fn split<'py>(&self, values: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let Some(layout) = self.layout.as_ref() else {
            return Err(PyValueError::new_err(format!(
                "{}: split needs named outputs; this function was compiled from a \
                 single expression",
                self.metadata.display_name()
            )));
        };
        let n_rows: usize = values
            .getattr("shape")
            .and_then(|shape| shape.get_item(0))
            .and_then(|rows| rows.extract())
            .map_err(|_| {
                PyValueError::new_err(format!(
                    "{}: split expects an array with a leading axis of {} row(s)",
                    self.metadata.display_name(),
                    self.metadata.output_len
                ))
            })?;
        if n_rows != self.metadata.output_len {
            return Err(PyValueError::new_err(format!(
                "{}: split expects {} row(s), got {}",
                self.metadata.display_name(),
                self.metadata.output_len,
                n_rows
            )));
        }
        let py = values.py();
        layout
            .spans()
            .map(|(offset, len)| {
                let rows = pyo3::types::PySlice::new(
                    py,
                    isize::try_from(offset).unwrap_or(isize::MAX),
                    isize::try_from(offset + len).unwrap_or(isize::MAX),
                    1,
                );
                values.get_item(rows)
            })
            .collect()
    }
    #[getter]
    const fn uses_y_dot(&self) -> bool {
        self.metadata.uses_y_dot
    }
    #[getter]
    fn name(&self) -> Option<String> {
        self.metadata.name.clone()
    }
    /// Instruction count excluding conditional branch blocks (one dispatch each).
    #[getter]
    fn n_instructions(&self) -> usize {
        self.expr.ir().common_instruction_count()
    }
    /// Raw tape length, branch blocks included.
    #[getter]
    fn n_instructions_total(&self) -> usize {
        self.expr.ir().instructions().len()
    }
    /// How many dispatches `n_instructions` includes, one per conditional.
    #[getter]
    fn n_dispatches(&self) -> usize {
        self.expr.ir().dispatch_count()
    }
    /// Per-branch block lengths, in tape order.
    #[getter]
    fn branch_block_lens<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        pyo3::types::PyTuple::new(py, self.expr.ir().branch_block_lens())
    }

    /// Forward-mode JVP: df/dy @ vy (+ df/dp @ vp when given).
    #[pyo3(signature = (t, y, p, vy, vp = None))]
    fn jvp<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        y: PyReadonlyArray1<'_, f64>,
        p: &Bound<'_, PyAny>,
        vy: PyReadonlyArray1<'_, f64>,
        vp: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.metadata.reject_y_dot("jvp")?;
        let packed = self.metadata.extract_p(p)?;
        let y_slice = y.as_slice()?;
        self.metadata
            .check_len("y", y_slice.len(), self.metadata.n_states)?;
        let vy_slice = vy.as_slice()?;
        self.metadata
            .check_len("vy", vy_slice.len(), self.metadata.n_states)?;

        let ty = self.tangent_entry(py, false)?;
        let mut scratch = ty.pool.acquire(|| buffer(ty.expr.scratch_len()));
        let tangent = TangentInputs {
            dy: Some(vy_slice),
            dp: None,
        };
        let mut out: Vec<f64> = ty
            .expr
            .eval_with_tangent(&mut scratch, t, y_slice, &[], &packed, &tangent)
            .to_vec();
        ty.pool.release(scratch);

        if let Some(vp) = vp {
            let vp_slice = vp.as_slice()?;
            self.metadata.check_vp(vp_slice.len())?;
            let tp = self.tangent_entry(py, true)?;
            let mut scratch_p = tp.pool.acquire(|| buffer(tp.expr.scratch_len()));
            let tangent_p = TangentInputs {
                dy: None,
                dp: Some(vp_slice),
            };
            let contrib =
                tp.expr
                    .eval_with_tangent(&mut scratch_p, t, y_slice, &[], &packed, &tangent_p);
            for (o, c) in out.iter_mut().zip(contrib) {
                *o += *c;
            }
            tp.pool.release(scratch_p);
        }
        Ok(PyArray1::from_vec(py, out))
    }

    /// Lazy, cached-per-wrt prepared jacobian.
    #[pyo3(signature = (wrt = "y"))]
    fn jacobian(&self, py: Python<'_>, wrt: &str) -> PyResult<Py<CompiledJacobian>> {
        self.metadata.reject_y_dot("jacobian")?;
        let cell = match wrt {
            "y" => &self.jac_y,
            "p" => &self.jac_p,
            other => {
                return Err(PyValueError::new_err(format!(
                    "{}: wrt must be 'y' or 'p', got {:?}",
                    self.metadata.display_name(),
                    other
                )));
            },
        };
        cell.get_or_try_init(py, || {
            let g = self.graph.try_borrow(py).map_err(|_| {
                PyValueError::new_err(format!(
                    "{}: graph borrow conflict during jacobian derivation",
                    self.metadata.display_name()
                ))
            })?;
            let data = match wrt {
                "y" => pybamm_core::JacobianData::new_wrt_states(
                    g.arena(),
                    self.root,
                    self.metadata.output_len,
                    self.metadata.n_states,
                ),
                _ => pybamm_core::JacobianData::new_wrt_params(
                    g.arena(),
                    self.root,
                    self.metadata.output_len,
                    self.metadata.params.index_extent(),
                ),
            };
            drop(g);
            Py::new(
                py,
                CompiledJacobian::build(py, Arc::new(data), self.metadata.clone())?,
            )
        })
        .map(|j| j.clone_ref(py))
    }

    /// Evaluate along a trajectory, returning `(output_len, n_t)` F-contiguous.
    fn eval_trajectory<'py>(
        &self,
        py: Python<'py>,
        ts: PyArrayLike1<'_, f64, AllowTypeChange>,
        y_traj: PyReadonlyArray2<'_, f64>,
        p: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let packed = self.metadata.extract_p(p)?;
        let ts_view = ts.as_array();
        let ts_slice = contiguous_slice_1d(&ts_view);
        let view = y_traj.as_array();
        let (n_rows, n_t) = (view.shape()[0], view.shape()[1]);
        self.metadata
            .check_trajectory(n_rows, n_t, ts_slice.len())?;
        self.metadata.reject_y_dot("eval_trajectory")?;

        let y_cols = columns_slice(&view);

        let out_len = self.metadata.output_len;
        let mut out = vec![0.0_f64; out_len * n_t];
        let expr = Arc::clone(&self.expr);
        let scratch_len = expr.scratch_len();
        let k_max = lane_tile(scratch_len);
        let k_alloc = k_max.min(n_t.max(1));
        let mut scratch = vec![0.0_f64; scratch_len * k_alloc];
        let batch_result = py.detach(|| -> Result<(), pybamm_core::BatchEvalError> {
            let mut j0 = 0;
            while j0 < n_t {
                let k = (n_t - j0).min(k_max);
                let ts_tile = &ts_slice[j0..j0 + k];
                let y_tile = &y_cols[j0 * n_rows..(j0 + k) * n_rows];
                let root = expr.eval_batch(&mut scratch, k, ts_tile, y_tile, n_rows, &packed)?;
                scatter_lanes(root, k, out_len, &mut out[j0 * out_len..(j0 + k) * out_len]);
                j0 += k;
            }
            Ok(())
        });
        batch_result.map_err(|e| PyValueError::new_err(e.to_string()))?;

        let arr = numpy::ndarray::Array2::from_shape_vec((out_len, n_t).f(), out)
            .map_err(|e| PyValueError::new_err(format!("eval_trajectory shape error: {e}")))?;
        Ok(PyArray2::from_owned_array(py, arr))
    }

    /// Forward-mode JVP per time column: `df/dy @ vy_j (+ df/dp @ vp when given)`.
    ///
    /// `vy_traj` is `(n_states, n_t)`; `vp` has `n_inputs` entries, not the packed
    /// width. Returns `(output_len, n_t)` F-contiguous.
    #[pyo3(signature = (ts, y_traj, p, vy_traj, vp = None))]
    fn jvp_trajectory<'py>(
        &self,
        py: Python<'py>,
        ts: PyArrayLike1<'_, f64, AllowTypeChange>,
        y_traj: PyReadonlyArray2<'_, f64>,
        p: &Bound<'_, PyAny>,
        vy_traj: PyReadonlyArray2<'_, f64>,
        vp: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.metadata.reject_y_dot("jvp_trajectory")?;
        let packed = self.metadata.extract_p(p)?;
        let ts_view = ts.as_array();
        let ts_slice = contiguous_slice_1d(&ts_view);
        let view = y_traj.as_array();
        let (n_rows, n_t) = (view.shape()[0], view.shape()[1]);
        self.metadata
            .check_trajectory(n_rows, n_t, ts_slice.len())?;
        let vy_view = vy_traj.as_array();
        if vy_view.shape()[0] != self.metadata.n_states || vy_view.shape()[1] != n_t {
            return Err(PyValueError::new_err(format!(
                "{}: vy_traj shape must be (n_states, n_t) = ({}, {}), got ({}, {})",
                self.metadata.display_name(),
                self.metadata.n_states,
                n_t,
                vy_view.shape()[0],
                vy_view.shape()[1]
            )));
        }

        let y_cols = columns_slice(&view);
        let vy_cols = columns_slice(&vy_view);

        let ty = self.tangent_entry(py, false)?;
        let vp_slice: Option<&[f64]> = match &vp {
            Some(arr) => {
                let s = arr.as_slice()?;
                self.metadata.check_vp(s.len())?;
                Some(s)
            },
            None => None,
        };
        let tp = match vp_slice {
            Some(_) => Some(self.tangent_entry(py, true)?),
            None => None,
        };

        let out_len = self.metadata.output_len;
        let mut out = vec![0.0_f64; out_len * n_t];
        let mut scratch_y = ty.pool.acquire(|| buffer(ty.expr.scratch_len()));
        let mut scratch_p = tp.map(|t| t.pool.acquire(|| buffer(t.expr.scratch_len())));
        py.detach(|| {
            for j in 0..n_t {
                let y = &y_cols[j * n_rows..(j + 1) * n_rows];
                let vy = &vy_cols[j * n_rows..(j + 1) * n_rows];
                let dst = &mut out[j * out_len..(j + 1) * out_len];

                let tangent_y = TangentInputs {
                    dy: Some(vy),
                    dp: None,
                };
                let res_y = ty.expr.eval_with_tangent(
                    &mut scratch_y,
                    ts_slice[j],
                    y,
                    &[],
                    &packed,
                    &tangent_y,
                );
                dst.copy_from_slice(res_y);

                if let (Some(tp), Some(scratch_p), Some(vp)) = (tp, scratch_p.as_mut(), vp_slice) {
                    let tangent_p = TangentInputs {
                        dy: None,
                        dp: Some(vp),
                    };
                    let res_p = tp.expr.eval_with_tangent(
                        scratch_p,
                        ts_slice[j],
                        y,
                        &[],
                        &packed,
                        &tangent_p,
                    );
                    for (o, c) in dst.iter_mut().zip(res_p) {
                        *o += *c;
                    }
                }
            }
        });
        ty.pool.release(scratch_y);
        if let (Some(tp), Some(scratch_p)) = (tp, scratch_p) {
            tp.pool.release(scratch_p);
        }

        let arr = numpy::ndarray::Array2::from_shape_vec((out_len, n_t).f(), out)
            .map_err(|e| PyValueError::new_err(format!("jvp_trajectory shape error: {e}")))?;
        Ok(PyArray2::from_owned_array(py, arr))
    }

    /// Cubic-Hermite reconstruct the state at each `t_query` from the solver
    /// knots (`ts`, `ys`, `yps`), then evaluate; returns `(output_len, n_query)`.
    #[pyo3(signature = (t_query, ts, ys, yps, p))]
    #[allow(clippy::suboptimal_flops)] // mul_add would obscure the cubic-Hermite basis
    fn eval_trajectory_hermite<'py>(
        &self,
        py: Python<'py>,
        t_query: PyArrayLike1<'_, f64, AllowTypeChange>,
        ts: PyArrayLike1<'_, f64, AllowTypeChange>,
        ys: PyReadonlyArray2<'_, f64>,
        yps: PyReadonlyArray2<'_, f64>,
        p: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.metadata.reject_y_dot("eval_trajectory_hermite")?;
        let packed = self.metadata.extract_p(p)?;
        let tq_view = t_query.as_array();
        let tq = contiguous_slice_1d(&tq_view);
        let ts_view = ts.as_array();
        let ts_slice = contiguous_slice_1d(&ts_view);
        let y_view = ys.as_array();
        let yp_view = yps.as_array();
        let (n_rows, n_knots) = (y_view.shape()[0], y_view.shape()[1]);
        self.metadata
            .check_trajectory(n_rows, n_knots, ts_slice.len())?;
        if yp_view.shape()[0] != n_rows || yp_view.shape()[1] != n_knots {
            return Err(PyValueError::new_err(format!(
                "{}: yps shape must equal ys shape ({}, {}), got ({}, {})",
                self.metadata.display_name(),
                n_rows,
                n_knots,
                yp_view.shape()[0],
                yp_view.shape()[1]
            )));
        }
        self.metadata.check_knots(n_knots)?;

        let y_cols = columns_slice(&y_view);
        let yp_cols = columns_slice(&yp_view);

        let out_len = self.metadata.output_len;
        let n_query = tq.len();
        let mut out = vec![0.0_f64; out_len * n_query];
        let expr = Arc::clone(&self.expr);
        let scratch_len = expr.scratch_len();
        let k_max = lane_tile(scratch_len);
        let k_alloc = k_max.min(n_query.max(1));
        let mut scratch = vec![0.0_f64; scratch_len * k_alloc];
        // Reconstructed state columns for one tile, (n_states, k) F-contiguous.
        let mut y_tile = vec![0.0_f64; n_rows * k_alloc];
        let batch_result = py.detach(|| -> Result<(), pybamm_core::BatchEvalError> {
            let mut q0 = 0;
            while q0 < n_query {
                let k = (n_query - q0).min(k_max);
                for l in 0..k {
                    let q = q0 + l;
                    let i = locate_interval(&ts_slice, tq[q]);
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
                    let col = &mut y_tile[l * n_rows..(l + 1) * n_rows];
                    for m in 0..n_rows {
                        col[m] = h00 * yi[m] + h10 * h * ypi[m] + h01 * yi1[m] + h11 * h * ypi1[m];
                    }
                }
                let root = expr.eval_batch(
                    &mut scratch,
                    k,
                    &tq[q0..q0 + k],
                    &y_tile[..k * n_rows],
                    n_rows,
                    &packed,
                )?;
                scatter_lanes(root, k, out_len, &mut out[q0 * out_len..(q0 + k) * out_len]);
                q0 += k;
            }
            Ok(())
        });
        batch_result.map_err(|e| PyValueError::new_err(e.to_string()))?;

        let arr =
            numpy::ndarray::Array2::from_shape_vec((out_len, n_query).f(), out).map_err(|e| {
                PyValueError::new_err(format!("eval_trajectory_hermite shape error: {e}"))
            })?;
        Ok(PyArray2::from_owned_array(py, arr))
    }

    fn __repr__(&self) -> String {
        let outputs = self.layout.as_ref().map_or_else(String::new, |layout| {
            format!(", outputs={:?}", layout.names)
        });
        format!(
            "CompiledFunction(name={:?}, inputs={:?}, n_states={}, output_len={}{}, \
             uses_y_dot={})",
            self.metadata.display_name(),
            self.metadata.params.names(),
            self.metadata.n_states,
            self.metadata.output_len,
            outputs,
            self.metadata.uses_y_dot
        )
    }

    /// Rebuild from the retained `(graph, root)` (pickle protocol).
    ///
    /// `n_states` is re-pinned exactly so a user-widened system round-trips.
    #[staticmethod]
    #[pyo3(signature = (graph, root, name, n_states, output_names = None, output_lens = None))]
    fn _rebuild(
        py: Python<'_>,
        graph: Py<ExprGraph>,
        root: u32,
        name: Option<String>,
        n_states: Option<usize>,
        output_names: Option<Vec<String>>,
        output_lens: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let root = {
            let g = graph
                .try_borrow(py)
                .map_err(|_| PyValueError::new_err("_rebuild: graph borrow conflict"))?;
            let root = g.checked_root(root)?;
            crate::expr::check_supported(g.arena(), root)?;
            root
        };
        let mut this = Self::build(py, graph, root, name, n_states)?;
        this.layout = match (output_names, output_lens) {
            (Some(names), Some(lens)) => {
                Some(OutputLayout::new(names, lens, this.metadata.output_len)?)
            },
            (None, None) => None,
            _ => {
                return Err(PyValueError::new_err(
                    "_rebuild: output_names and output_lens must be given together",
                ));
            },
        };
        Ok(this)
    }

    /// `(callable, args)` pickle pair that rebuilds from the retained graph and root.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, RebuildArgs)> {
        let this = slf.get();
        let rebuild = slf.get_type().getattr("_rebuild")?;
        let (output_names, output_lens) = this.layout.as_ref().map_or((None, None), |layout| {
            (Some(layout.names.clone()), Some(layout.lens.clone()))
        });
        Ok((
            rebuild,
            (
                this.graph.clone_ref(py),
                this.root.raw(),
                this.name(),
                Some(this.n_states()),
                output_names,
                output_lens,
            ),
        ))
    }
}

/// Lanes per batched trajectory tile, so one tile's scratch stays near [`TARGET_BYTES`].
fn lane_tile(scratch_len: usize) -> usize {
    (TARGET_BYTES / (8 * scratch_len.max(1))).clamp(8, 64)
}

/// Scatter lane-minor `root[e * k + l]` (element `e`, lane `l`) into point-major `dst`.
fn scatter_lanes(root: &[f64], k: usize, out_len: usize, dst: &mut [f64]) {
    for l in 0..k {
        for (e, o) in dst[l * out_len..(l + 1) * out_len].iter_mut().enumerate() {
            *o = root[e * k + l];
        }
    }
}

/// Column-major slice of Y: zero-copy when F-contiguous, else one gathering copy.
fn columns_slice<'a>(view: &'a numpy::ndarray::ArrayView2<'a, f64>) -> std::borrow::Cow<'a, [f64]> {
    match view.as_slice_memory_order() {
        Some(s) if view.strides()[0] == 1 && view.strides().iter().all(|&st| st >= 0) => {
            std::borrow::Cow::Borrowed(s)
        },
        _ => {
            let (n_rows, n_t) = (view.shape()[0], view.shape()[1]);
            let mut owned = Vec::with_capacity(n_rows * n_t);
            for j in 0..n_t {
                owned.extend(view.column(j).iter());
            }
            std::borrow::Cow::Owned(owned)
        },
    }
}

/// Contiguous view of a 1-D time array: zero-copy when contiguous, else one copy.
///
/// `AllowTypeChange` passes a strided f64 ndarray through, which `.as_slice()` rejects.
fn contiguous_slice_1d<'a>(
    view: &'a numpy::ndarray::ArrayView1<'a, f64>,
) -> std::borrow::Cow<'a, [f64]> {
    view.as_slice().map_or_else(
        || std::borrow::Cow::Owned(view.to_vec()),
        std::borrow::Cow::Borrowed,
    )
}

/// Bracketing interval of `x` in ascending `ts` (len >= 2), clamped to `[0, n-2]`.
fn locate_interval(ts: &[f64], x: f64) -> usize {
    let nseg = ts.len() - 1;
    if x <= ts[0] {
        return 0;
    }
    if x >= ts[nseg] {
        return nseg - 1;
    }
    let (mut lo, mut hi) = (0usize, nseg);
    while hi - lo > 1 {
        let mid = usize::midpoint(lo, hi);
        if ts[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Resolve an `n_states` override against the scanned extent; a smaller one would
/// admit a too-short `y` and panic inside the tape.
fn resolve_n_states(
    scanned: usize,
    requested: Option<usize>,
    name: Option<&str>,
) -> PyResult<usize> {
    pybamm_core::resolve_state_count(requested, scanned)
        .map_err(|err| PyValueError::new_err(format!("{}: {err}", name.unwrap_or("<anonymous>"))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_layout_rejects_lengths_that_wrap_to_the_tape_length() {
        let names = vec!["a".to_string(), "b".to_string()];
        // usize::MAX + 3 wraps to 2, the tape length.
        assert!(OutputLayout::new(names.clone(), vec![usize::MAX, 3], 2).is_err());
        let layout = OutputLayout::new(names, vec![1, 1], 2).ok();
        let spans: Option<Vec<_>> = layout.map(|layout| layout.spans().collect());
        assert_eq!(spans, Some(vec![(0, 1), (1, 1)]));
    }
}
