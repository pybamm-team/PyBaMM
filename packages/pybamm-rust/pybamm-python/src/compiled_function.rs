//! `CompiledFunction`: the central prepared artifact.

// PyO3 bindings require specific argument types that clippy flags incorrectly
#![allow(clippy::needless_pass_by_value)]

use std::ops::Range;
use std::sync::Arc;

use numpy::ndarray::{Array2, ShapeBuilder};
use numpy::{
    AllowTypeChange, PyArray1, PyArray2, PyArrayLike1, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArray2,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyMapping, PySlice, PyTuple};

use pybamm_core::{
    Arena, BatchEvalError, CompiledExpr, Dag, DagRef, DiffTarget, JacobianData, NodeId,
    ReadExtents, TangentInputs, dce, infer_sizes, simplify_pipeline, tangent_wrt_params,
    tangent_wrt_states,
};

use crate::arrays;
use crate::expr::ExprGraph;
use crate::function_metadata::FunctionMetadata;
use crate::jacobian::CompiledJacobian;
use crate::scratch_pool::ScratchPool;

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
    usize,
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
    /// Pair each output name with its length, refusing lengths that don't sum to `output_len`.
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
/// `p` is a name-keyed mapping or an already-packed array; derivatives are memoised.
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
    /// Derivatives, each built once on first use, however many threads ask for it at once.
    tangent_y: PyOnceLock<TangentEntry>,
    tangent_p: PyOnceLock<TangentEntry>,
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
    /// Compile `root` of `graph`, checking it reads no more states than `n_states_override`.
    ///
    /// The function takes the inputs registered below `vp_len` indices, or every input the
    /// graph holds for `None`.
    pub(crate) fn build(
        py: Python<'_>,
        graph: Py<ExprGraph>,
        root: NodeId,
        name: Option<String>,
        n_states_override: Option<usize>,
        vp_len: Option<usize>,
    ) -> PyResult<Self> {
        let g = graph.try_borrow(py).map_err(|_| {
            PyValueError::new_err(format!(
                "{}: graph borrow conflict while compiling",
                name.as_deref().unwrap_or("<anonymous>")
            ))
        })?;
        // Scan before simplifying, which can shrink the reachable state extent.
        let reads = ReadExtents::scan(g.arena(), root);
        let n_states = resolve_n_states(reads.n_states(), n_states_override, name.as_deref())?;
        let params = g.registry_at(vp_len, &reads)?;
        // A reachable copy, so simplifying leaves the retained graph untouched.
        let dag = dce(DagRef::new(g.arena(), root));
        drop(g);
        let expr = py.detach(|| CompiledExpr::new(simplify_pipeline(dag).as_ref()));
        let metadata = FunctionMetadata {
            params,
            n_states,
            uses_y_dot: reads.y_dot.is_some(),
            output_len: expr.output_len(),
            name,
        };
        Ok(Self {
            expr: Arc::new(expr),
            metadata,
            pool: ScratchPool::default(),
            graph,
            root,
            layout: None,
            tangent_y: PyOnceLock::new(),
            tangent_p: PyOnceLock::new(),
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
            // Checked before the concat exists, so a refused call leaves the graph as it was.
            let reads = ReadExtents::scan_all(g.arena(), ids.iter().copied());
            resolve_n_states(reads.n_states(), n_states_override, name.as_deref())?;
            // Kept in the graph, which `_rebuild` and the derivatives compile from.
            let root = g.concat_of(&ids);
            // The concat tape only reports the total length; one size pass covers every output.
            let sizes = infer_sizes(g.arena(), &g.arena().topological_order(root));
            (root, ids.iter().map(|id| sizes[id.index()]).collect())
        };
        let mut this = Self::build(py, graph, root, name, n_states_override, None)?;
        this.layout = Some(OutputLayout::new(names, lens, this.metadata.output_len)?);
        Ok(this)
    }

    /// A view sharing an already compiled `expr`, whose extent must fit in `n_states`.
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
        let extents = ReadExtents::scan(g.arena(), root);
        // Bundle views carry the full system width, which must cover what the root reads.
        let n_states = resolve_n_states(extents.n_states(), Some(n_states), name.as_deref())?;
        let metadata = FunctionMetadata {
            params: g.params().clone(),
            n_states,
            uses_y_dot: extents.y_dot.is_some(),
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
            tangent_y: PyOnceLock::new(),
            tangent_p: PyOnceLock::new(),
            jac_y: PyOnceLock::new(),
            jac_p: PyOnceLock::new(),
        })
    }

    /// Evaluate at one point into `out`, after checking every argument's length.
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
        let mut scratch = self.pool.acquire_buffer(self.expr.scratch_len());
        self.expr
            .eval_into(&mut scratch, t, y, y_dot.unwrap_or(&[]), p, out);
        self.pool.release(scratch);
        Ok(())
    }

    /// The tangent tape and scratch pool for `wrt`, derived on first use.
    fn tangent_entry(&self, py: Python<'_>, wrt: DiffTarget) -> PyResult<&TangentEntry> {
        let cell = match wrt {
            DiffTarget::States => &self.tangent_y,
            DiffTarget::Params => &self.tangent_p,
        };
        cell.get_or_try_init(py, || {
            let source = self.source(py, "tangent derivation")?;
            let rewrite: fn(&mut Arena, NodeId) -> NodeId = match wrt {
                DiffTarget::States => tangent_wrt_states,
                DiffTarget::Params => tangent_wrt_params,
            };
            let expr = py.detach(|| {
                let mut tangent = simplify_pipeline(source);
                tangent.rewrite_in_place(rewrite);
                CompiledExpr::new(simplify_pipeline(tangent).as_ref())
            });
            Ok(TangentEntry {
                expr,
                pool: ScratchPool::default(),
            })
        })
    }

    /// A reachable copy of the retained root, for a derivative to simplify then differentiate.
    ///
    /// Re-simplifying it, as the primal tape was, keeps each derivative in step with the
    /// primal. That repeats work once per derivative, rather than holding the simplified
    /// graph for the function's lifetime.
    fn source(&self, py: Python<'_>, during: &str) -> PyResult<Dag> {
        let g = self.graph.try_borrow(py).map_err(|_| {
            PyValueError::new_err(format!(
                "{}: graph borrow conflict during {during}",
                self.metadata.display_name()
            ))
        })?;
        Ok(dce(DagRef::new(g.arena(), self.root)))
    }

    /// `wrt` as a differentiation target: `"y"` or `"p"`.
    fn diff_target(&self, wrt: &str) -> PyResult<DiffTarget> {
        match wrt {
            "y" => Ok(DiffTarget::States),
            "p" => Ok(DiffTarget::Params),
            other => Err(PyValueError::new_err(format!(
                "{}: wrt must be 'y' or 'p', got {other:?}",
                self.metadata.display_name()
            ))),
        }
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
        let y_dot = y_dot.as_ref().map(arrays::values);
        self.eval_inner(t, &arrays::values(&y), &packed, y_dot.as_deref(), &mut out)?;
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

    /// Evaluate into `out`, a contiguous, writeable array of `output_len` values.
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
        let out_slice = out.as_slice_mut().map_err(|_| {
            PyValueError::new_err(format!(
                "{}: out must be contiguous",
                self.metadata.display_name()
            ))
        })?;
        self.metadata
            .check_len("out", out_slice.len(), self.metadata.output_len)?;
        let y_dot = y_dot.as_ref().map(arrays::values);
        self.eval_inner(t, &arrays::values(&y), &packed, y_dot.as_deref(), out_slice)
    }

    /// Pack a {name: value} mapping into the stacked input layout.
    fn pack<'py>(
        &self,
        py: Python<'py>,
        mapping: &Bound<'_, PyMapping>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(PyArray1::from_vec(py, self.metadata.pack(mapping)?))
    }

    #[getter]
    fn input_names<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.metadata.params.names())
    }
    /// Length of a `vp` direction: one entry per input registration index, not the packed
    /// input width.
    #[getter]
    const fn vp_len(&self) -> usize {
        self.metadata.vp_len()
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
    fn output_names<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        self.layout
            .as_ref()
            .map(|layout| PyTuple::new(py, &layout.names))
            .transpose()
    }
    /// Rows each declared output owns, ordered as `output_names`.
    #[getter]
    fn output_lens<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        self.layout
            .as_ref()
            .map(|layout| PyTuple::new(py, &layout.lens))
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
                let rows = PySlice::new(
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
    fn branch_block_lens<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.expr.ir().branch_block_lens())
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
        let (y, vy) = (arrays::values(&y), arrays::values(&vy));
        self.metadata
            .check_len("y", y.len(), self.metadata.n_states)?;
        self.metadata
            .check_len("vy", vy.len(), self.metadata.n_states)?;
        let vp = vp.as_ref().map(arrays::values);
        let vp = vp.as_deref();
        if let Some(vp) = vp {
            self.metadata.check_vp(vp.len())?;
        }

        let ty = self.tangent_entry(py, DiffTarget::States)?;
        let mut scratch = ty.pool.acquire_buffer(ty.expr.scratch_len());
        let tangent = TangentInputs {
            dy: Some(&*vy),
            dp: None,
        };
        let mut out: Vec<f64> = ty
            .expr
            .eval_with_tangent(&mut scratch, t, &y, &[], &packed, &tangent)
            .to_vec();
        ty.pool.release(scratch);

        if let Some(vp) = vp {
            let tp = self.tangent_entry(py, DiffTarget::Params)?;
            let mut scratch_p = tp.pool.acquire_buffer(tp.expr.scratch_len());
            let tangent_p = TangentInputs {
                dy: None,
                dp: Some(vp),
            };
            let contrib =
                tp.expr
                    .eval_with_tangent(&mut scratch_p, t, &y, &[], &packed, &tangent_p);
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
        let wrt = self.diff_target(wrt)?;
        let cell = match wrt {
            DiffTarget::States => &self.jac_y,
            DiffTarget::Params => &self.jac_p,
        };
        cell.get_or_try_init(py, || {
            let source = self.source(py, "jacobian derivation")?;
            let n_rows = self.metadata.output_len;
            let data = py.detach(|| {
                let simplified = simplify_pipeline(source);
                match wrt {
                    DiffTarget::States => JacobianData::new_wrt_states(
                        simplified.as_ref(),
                        n_rows,
                        self.metadata.n_states,
                    ),
                    DiffTarget::Params => JacobianData::new_wrt_params(
                        simplified.as_ref(),
                        n_rows,
                        self.metadata.vp_len(),
                    ),
                }
            });
            Py::new(
                py,
                CompiledJacobian::build(Arc::new(data), self.metadata.clone())?,
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
        self.metadata.reject_y_dot("eval_trajectory")?;
        let packed = self.metadata.extract_p(p)?;
        let ts = arrays::values(&ts);
        let y_view = y_traj.as_array();
        let (n_rows, n_t) = y_view.dim();
        self.metadata
            .check_trajectory("y_traj", n_rows, n_t, ts.len())?;
        let y_cols = arrays::columns(&y_view);

        let expr = &*self.expr;
        let out_len = self.metadata.output_len;
        let out = py
            .detach(|| {
                eval_tiled(expr.scratch_len(), out_len, n_t, |scratch, lanes| {
                    let states = &y_cols[lanes.start * n_rows..lanes.end * n_rows];
                    expr.eval_batch(scratch, lanes.len(), &ts[lanes], states, n_rows, &packed)
                })
            })
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        column_major(py, out, (out_len, n_t))
    }

    /// Forward-mode JVP per time column: `df/dy @ vy_j (+ df/dp @ vp when given)`.
    ///
    /// `vy_traj` is `(n_states, n_t)`; `vp` has `vp_len` entries, not the packed
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
        let vp = vp.as_ref().map(arrays::values);
        let vp = vp.as_deref();
        if let Some(vp) = vp {
            self.metadata.check_vp(vp.len())?;
        }
        let ts = arrays::values(&ts);
        let (y_view, vy_view) = (y_traj.as_array(), vy_traj.as_array());
        let (n_rows, n_t) = y_view.dim();
        self.metadata
            .check_trajectory("y_traj", n_rows, n_t, ts.len())?;
        if vy_view.dim() != (n_rows, n_t) {
            return Err(PyValueError::new_err(format!(
                "{}: vy_traj shape must be (n_states, n_t) = ({n_rows}, {n_t}), got {:?}",
                self.metadata.display_name(),
                vy_view.dim()
            )));
        }
        let (y_cols, vy_cols) = (arrays::columns(&y_view), arrays::columns(&vy_view));

        let ty = self.tangent_entry(py, DiffTarget::States)?;
        let tp = vp
            .map(|_| self.tangent_entry(py, DiffTarget::Params))
            .transpose()?;

        let out_len = self.metadata.output_len;
        let mut out = vec![0.0_f64; out_len * n_t];
        let mut scratch_y = ty.pool.acquire_buffer(ty.expr.scratch_len());
        let mut scratch_p = tp.map(|t| t.pool.acquire_buffer(t.expr.scratch_len()));
        py.detach(|| {
            for (j, &t) in ts.iter().enumerate() {
                let y = column(&y_cols, n_rows, j);
                let dst = &mut out[j * out_len..(j + 1) * out_len];

                let tangent_y = TangentInputs {
                    dy: Some(column(&vy_cols, n_rows, j)),
                    dp: None,
                };
                let res_y =
                    ty.expr
                        .eval_with_tangent(&mut scratch_y, t, y, &[], &packed, &tangent_y);
                dst.copy_from_slice(res_y);

                if let (Some(tp), Some(scratch_p), Some(vp)) = (tp, scratch_p.as_mut(), vp) {
                    let tangent_p = TangentInputs {
                        dy: None,
                        dp: Some(vp),
                    };
                    let res_p =
                        tp.expr
                            .eval_with_tangent(scratch_p, t, y, &[], &packed, &tangent_p);
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
        column_major(py, out, (out_len, n_t))
    }

    /// Cubic-Hermite reconstruct the state at each `t_query` from the solver
    /// knots (`ts`, `ys`, `yps`), then evaluate; returns `(output_len, n_query)`.
    ///
    /// `ts` must be finite and strictly increasing.
    #[pyo3(signature = (t_query, ts, ys, yps, p))]
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
        let (t_query, ts) = (arrays::values(&t_query), arrays::values(&ts));
        let (y_view, yp_view) = (ys.as_array(), yps.as_array());
        let (n_rows, n_knots) = y_view.dim();
        self.metadata
            .check_trajectory("ys", n_rows, n_knots, ts.len())?;
        if yp_view.dim() != (n_rows, n_knots) {
            return Err(PyValueError::new_err(format!(
                "{}: yps shape must equal ys shape ({n_rows}, {n_knots}), got {:?}",
                self.metadata.display_name(),
                yp_view.dim()
            )));
        }
        self.metadata.check_knots(&ts)?;
        let (y_cols, yp_cols) = (arrays::columns(&y_view), arrays::columns(&yp_view));

        let expr = &*self.expr;
        let out_len = self.metadata.output_len;
        let n_query = t_query.len();
        // Reconstructed state columns for one tile, (n_states, k) F-contiguous.
        let mut states = Vec::with_capacity(n_rows * lane_tile(expr.scratch_len()).min(n_query));
        let out = py
            .detach(|| {
                eval_tiled(expr.scratch_len(), out_len, n_query, |scratch, lanes| {
                    states.clear();
                    for &t in &t_query[lanes.clone()] {
                        let i = locate_interval(&ts, t);
                        let h = ts[i + 1] - ts[i];
                        let s = (t - ts[i]) / h;
                        let (s2, s3) = (s * s, s * s * s);
                        // cubic-Hermite basis; derivative terms scaled by the step h
                        let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
                        let h10 = s3 - 2.0 * s2 + s;
                        let h01 = -2.0 * s3 + 3.0 * s2;
                        let h11 = s3 - s2;
                        let (y0, y1) = (column(&y_cols, n_rows, i), column(&y_cols, n_rows, i + 1));
                        let (yp0, yp1) =
                            (column(&yp_cols, n_rows, i), column(&yp_cols, n_rows, i + 1));
                        states.extend((0..n_rows).map(|m| {
                            h00 * y0[m] + h10 * h * yp0[m] + h01 * y1[m] + h11 * h * yp1[m]
                        }));
                    }
                    let k = lanes.len();
                    expr.eval_batch(scratch, k, &t_query[lanes], &states, n_rows, &packed)
                })
            })
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        column_major(py, out, (out_len, n_query))
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
    /// `n_states` is re-pinned exactly so a user-widened system round-trips, and the inputs
    /// are cut back to the `vp_len` registration indices the function was compiled with.
    #[staticmethod]
    #[pyo3(
        name = "_rebuild",
        signature = (graph, root, name, n_states, vp_len, output_names = None, output_lens = None)
    )]
    #[allow(clippy::too_many_arguments)]
    fn rebuild(
        py: Python<'_>,
        graph: Py<ExprGraph>,
        root: u32,
        name: Option<String>,
        n_states: Option<usize>,
        vp_len: usize,
        output_names: Option<Vec<String>>,
        output_lens: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let root = {
            let g = graph
                .try_borrow(py)
                .map_err(|_| PyValueError::new_err("_rebuild: graph borrow conflict"))?;
            let root = g.checked_root(root)?;
            g.check_supported([root])?;
            root
        };
        let mut this = Self::build(py, graph, root, name, n_states, Some(vp_len))?;
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
                this.vp_len(),
                output_names,
                output_lens,
            ),
        ))
    }
}

/// Lanes per batched trajectory tile: as many as keep one tile's scratch near
/// [`TARGET_BYTES`], clamped to `8..=64`.
fn lane_tile(scratch_len: usize) -> usize {
    (TARGET_BYTES / (8 * scratch_len.max(1))).clamp(8, 64)
}

/// Evaluate `n_points` points of a tape with `scratch_len` scratch in lane tiles, gathering
/// each point's `output_len` values into an `(output_len, n_points)` column-major vector.
///
/// `eval_tile(scratch, lanes)` evaluates the points `lanes` in `scratch`, returning the
/// tape's root lane-minor.
fn eval_tiled(
    scratch_len: usize,
    output_len: usize,
    n_points: usize,
    mut eval_tile: impl FnMut(&mut [f64], Range<usize>) -> Result<&[f64], BatchEvalError>,
) -> Result<Vec<f64>, BatchEvalError> {
    let lanes_per_tile = lane_tile(scratch_len);
    let mut scratch = vec![0.0; scratch_len * lanes_per_tile.min(n_points.max(1))];
    let mut out = vec![0.0; output_len * n_points];
    for start in (0..n_points).step_by(lanes_per_tile) {
        let lanes = start..(start + lanes_per_tile).min(n_points);
        let k = lanes.len();
        let root = eval_tile(&mut scratch, lanes.clone())?;
        let dst = &mut out[lanes.start * output_len..lanes.end * output_len];
        // Lane-minor `root[e * k + l]` (element `e`, lane `l`) into point-major `dst`.
        for l in 0..k {
            for (e, o) in dst[l * output_len..(l + 1) * output_len]
                .iter_mut()
                .enumerate()
            {
                *o = root[e * k + l];
            }
        }
    }
    Ok(out)
}

/// `values` as an `(n_rows, n_cols)` F-contiguous array.
fn column_major(
    py: Python<'_>,
    values: Vec<f64>,
    (n_rows, n_cols): (usize, usize),
) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let array = Array2::from_shape_vec((n_rows, n_cols).f(), values)
        .map_err(|err| PyValueError::new_err(format!("trajectory shape error: {err}")))?;
    Ok(PyArray2::from_owned_array(py, array))
}

/// Column `j` of the column-major `columns`, each `n_rows` long.
fn column(columns: &[f64], n_rows: usize, j: usize) -> &[f64] {
    &columns[j * n_rows..(j + 1) * n_rows]
}

/// Bracketing interval of `x` in strictly increasing `ts` (len >= 2), clamped to `[0, n-2]`.
fn locate_interval(ts: &[f64], x: f64) -> usize {
    if x <= ts[0] {
        return 0;
    }
    ts[1..ts.len() - 1].partition_point(|&t| t <= x)
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

    #[test]
    fn tiles_cover_every_point_once_in_order() {
        // Outputs (point, -point) per point, so a misplaced lane or element shows.
        for n_points in [0, 1, 63, 64, 65, 200] {
            // Two scratch slots per lane, so tiles hold 64 lanes.
            let out = eval_tiled(2, 2, n_points, |scratch, lanes| {
                let k = lanes.len();
                for (lane, point) in lanes.enumerate() {
                    scratch[lane] = point as f64;
                    scratch[k + lane] = -(point as f64);
                }
                Ok(&scratch[..2 * k])
            });
            let expected: Vec<f64> = (0..n_points)
                .flat_map(|point| [point as f64, -(point as f64)])
                .collect();
            assert_eq!(out.ok(), Some(expected), "n_points = {n_points}");
        }
    }
}
