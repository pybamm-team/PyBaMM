//! `CompiledFunction`: the central prepared artifact.

// PyO3 bindings require specific argument types that clippy flags incorrectly
#![allow(clippy::needless_pass_by_value)]

use std::sync::{Arc, OnceLock};

use numpy::ndarray::ShapeBuilder;
use numpy::{
    AllowTypeChange, PyArray1, PyArray2, PyArrayLike1, PyReadonlyArray1, PyReadonlyArray2,
    PyReadwriteArray1,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyDict;

use pybamm_core::{
    Arena, CompiledExpr, NodeId, TangentInputs, TypedIr, scan_state_usage, simplify_pipeline,
    tangent_wrt_params, tangent_wrt_states,
};

use crate::expr::ExprGraph;
use crate::jacobian::CompiledJacobian;
use crate::scratch::{Buffer, ScratchPool};
use crate::signature::FunctionSignature;

/// Target working-set size for a lane-batched trajectory tile (~1 MiB). The
/// tile lane count is `clamp(TARGET_BYTES / (8 * scratch_len), 8, 64)`.
const TARGET_BYTES: usize = 1 << 20;

/// Memoised tangent tape plus its scratch pool (one per seed class).
struct TangentEntry {
    expr: CompiledExpr,
    pool: ScratchPool<Buffer>,
}

/// `(graph, root, name, n_states)`, the `__reduce__`/`_rebuild` argument
/// tuple for `CompiledFunction`'s pickle protocol.
type RebuildArgs = (Py<ExprGraph>, u32, Option<String>, Option<usize>);

// `module` is required so pickle can locate the class as `pybamm.rust.CompiledFunction`
// instead of the pyo3 default `builtins.CompiledFunction`, which pickle cannot import.
#[pyclass(frozen, module = "pybamm.rust")]
pub struct CompiledFunction {
    pub(crate) expr: Arc<CompiledExpr>,
    pub(crate) sig: FunctionSignature,
    pub(crate) pool: ScratchPool<Buffer>,
    /// Retained for lazy derivation (jacobian/jvp). Always present:
    /// bundle views retain the bundle's graph too.
    pub(crate) graph: Py<ExprGraph>,
    pub(crate) root: NodeId,
    /// Memoised JVP tape, wrt y (lazily derived on first jvp call).
    tangent_y: OnceLock<TangentEntry>,
    /// Memoised JVP tape, wrt p (lazily derived on first jvp call with vp).
    tangent_p: OnceLock<TangentEntry>,
    /// Memoised prepared jacobian, wrt y (lazily derived on first call).
    pub(crate) jac_y: PyOnceLock<Py<CompiledJacobian>>,
    /// Memoised prepared jacobian, wrt p (lazily derived on first call).
    pub(crate) jac_p: PyOnceLock<Py<CompiledJacobian>>,
}

impl std::fmt::Debug for CompiledFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledFunction")
            .field("sig", &self.sig)
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
        let g = graph.borrow(py);
        // Scan the ORIGINAL root: simplification can shrink the reachable state
        // extent, and a post-simplify scan would reject valid-length y.
        let usage = scan_state_usage(g.arena(), root);
        let n_states = resolve_n_states(usage.n_states, n_states_override, name.as_deref())?;
        let expr = Arc::new(compile_simplified(g.arena(), root));
        let sig = FunctionSignature {
            input_names: g.input_names(),
            input_widths: g.input_widths(),
            n_states,
            uses_y_dot: usage.uses_y_dot,
            output_len: expr.output_len(),
            name,
        };
        drop(g);
        let pool = ScratchPool::new(Buffer(expr.scratch_len()));
        Ok(Self {
            expr,
            sig,
            pool,
            graph,
            root,
            tangent_y: OnceLock::new(),
            tangent_p: OnceLock::new(),
            jac_y: PyOnceLock::new(),
            jac_p: PyOnceLock::new(),
        })
    }

    /// Bundle-view constructor: shares an existing tape (no recompilation).
    ///
    /// Unlike [`build`](Self::build), the primal expression is already
    /// compiled (the bundle's shared `Arc<CompiledExpr>`); we only scan the
    /// retained graph for the signature's `uses_y_dot` flag and reachable
    /// width. Tangent/jacobian caches start empty and derive lazily on demand.
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
        let usage = scan_state_usage(g.arena(), root);
        let sig = FunctionSignature {
            input_names: g.input_names(),
            input_widths: g.input_widths(),
            n_states, // bundle views always carry the full system width
            uses_y_dot: usage.uses_y_dot,
            output_len: expr.output_len(),
            name,
        };
        drop(g);
        let pool = ScratchPool::new(Buffer(expr.scratch_len()));
        Ok(Self {
            expr,
            sig,
            pool,
            graph,
            root,
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
        self.sig.check_y(y.len())?;
        if self.sig.uses_y_dot {
            match y_dot {
                Some(yd) => self.sig.check_y_dot(yd.len())?,
                None => {
                    return Err(PyValueError::new_err(format!(
                        "{}: expression uses y_dot; pass y_dot=",
                        self.sig.display_name()
                    )));
                },
            }
        }
        let mut scratch = self.pool.acquire();
        self.expr
            .eval_into(&mut scratch, t, y, y_dot.unwrap_or(&[]), p, out);
        self.pool.release(scratch);
        Ok(())
    }

    /// Lazily derive and cache the tangent tape + scratch pool for one
    /// seed class.
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
                self.sig.display_name()
            ))
        })?;
        let mut diff_arena = g.arena().clone();
        drop(g); // release the graph borrow before the compile pipeline
        let root = if wrt_params {
            tangent_wrt_params(&mut diff_arena, self.root)
        } else {
            tangent_wrt_states(&mut diff_arena, self.root)
        };
        let (da, root) = simplify_pipeline(diff_arena, root);
        let expr = CompiledExpr::from_ir(TypedIr::from_arena(&da, root));
        let pool = ScratchPool::new(Buffer(expr.scratch_len()));
        Ok(cell.get_or_init(|| TangentEntry { expr, pool }))
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
        let packed = self.sig.extract_p(p)?;
        let mut out = vec![0.0; self.sig.output_len];
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
        mut out: PyReadwriteArray1<'_, f64>,
        y_dot: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<()> {
        let packed = self.sig.extract_p(p)?;
        let out_slice = out.as_slice_mut()?;
        if out_slice.len() != self.sig.output_len {
            return Err(PyValueError::new_err(format!(
                "{}: expected out of length {}, got {}",
                self.sig.display_name(),
                self.sig.output_len,
                out_slice.len()
            )));
        }
        let yd = y_dot.as_ref().map(|a| a.as_slice()).transpose()?;
        self.eval_inner(t, y.as_slice()?, &packed, yd, out_slice)
    }

    /// Pack a {name: value} mapping into the stacked input layout.
    fn pack<'py>(
        &self,
        py: Python<'py>,
        mapping: &Bound<'_, PyDict>,
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
    #[getter]
    fn name(&self) -> Option<String> {
        self.sig.name.clone()
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
    /// How many dispatches `n_instructions` includes, one per short-circuited
    /// conditional. The part of the reported count that is control flow rather
    /// than always-run work, which `branch_block_lens` cannot recover.
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
        // The tangent tape would slice an empty y_dot and panic;
        // d/d(y_dot) seeding is a solver concern (cj).
        self.sig.reject_y_dot("jvp")?;
        let packed = self.sig.extract_p(p)?;
        let y_slice = y.as_slice()?;
        self.sig.check_y(y_slice.len())?;
        let vy_slice = vy.as_slice()?;
        if vy_slice.len() != self.sig.n_states {
            return Err(PyValueError::new_err(format!(
                "{}: expected vy of length {}, got {}",
                self.sig.display_name(),
                self.sig.n_states,
                vy_slice.len()
            )));
        }

        let ty = self.tangent_entry(py, false)?;
        let mut scratch = ty.pool.acquire();
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
            self.sig.check_vp(vp_slice.len())?;
            let tp = self.tangent_entry(py, true)?;
            let mut scratch_p = tp.pool.acquire();
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
        // Guard BEFORE prep: assembly evaluates with an empty y_dot, which the
        // tape would slice and panic on. cj-weighted systems stay solver-side.
        self.sig.reject_y_dot("jacobian")?;
        let cell = match wrt {
            "y" => &self.jac_y,
            "p" => &self.jac_p,
            other => {
                return Err(PyValueError::new_err(format!(
                    "{}: wrt must be 'y' or 'p', got {:?}",
                    self.sig.display_name(),
                    other
                )));
            },
        };
        cell.get_or_try_init(py, || {
            let g = self.graph.try_borrow(py).map_err(|_| {
                PyValueError::new_err(format!(
                    "{}: graph borrow conflict during jacobian derivation",
                    self.sig.display_name()
                ))
            })?;
            let data = match wrt {
                "y" => pybamm_core::JacobianData::new_wrt_states(
                    g.arena(),
                    self.root,
                    self.sig.output_len,
                    self.sig.n_states,
                ),
                _ => pybamm_core::JacobianData::new_wrt_params(
                    g.arena(),
                    self.root,
                    self.sig.output_len,
                    self.sig.input_names.len(),
                ),
            };
            drop(g);
            Py::new(
                py,
                CompiledJacobian::build(py, Arc::new(data), self.sig.clone())?,
            )
        })
        .map(|j| j.clone_ref(py))
    }

    /// Evaluate along a trajectory: one `extract_p` per sweep, GIL released
    /// for the inner loop, scratch reused across columns.
    fn eval_trajectory<'py>(
        &self,
        py: Python<'py>,
        ts: PyArrayLike1<'_, f64, AllowTypeChange>,
        y_traj: PyReadonlyArray2<'_, f64>,
        p: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let packed = self.sig.extract_p(p)?;
        let ts_view = ts.as_array();
        let ts_slice = contiguous_slice_1d(&ts_view);
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
        self.sig.reject_y_dot("eval_trajectory")?;

        // Zero-copy when Y is already F-contiguous; one gathering copy otherwise.
        let y_cols = columns_slice(&view);

        let out_len = self.sig.output_len;
        let mut out = vec![0.0_f64; out_len * n_t];
        let expr = Arc::clone(&self.expr);
        let scratch_len = expr.scratch_len();
        // Lane-batched tiling: ~1 MiB working set, 8..=64 lanes, capped at n_t.
        let k_max = (TARGET_BYTES / (8 * scratch_len.max(1))).clamp(8, 64);
        let k_alloc = k_max.min(n_t.max(1));
        let mut scratch = vec![0.0_f64; scratch_len * k_alloc];
        let batch_result = py.detach(|| -> Result<(), pybamm_core::BatchEvalError> {
            let mut j0 = 0;
            while j0 < n_t {
                let k = (n_t - j0).min(k_max);
                let ts_tile = &ts_slice[j0..j0 + k];
                let y_tile = &y_cols[j0 * n_rows..(j0 + k) * n_rows];
                let root = expr.eval_batch(&mut scratch, k, ts_tile, y_tile, &packed)?;
                // root is (out_len, k) lane-minor: element e lane l at root[e*k + l].
                for l in 0..k {
                    let dst = &mut out[(j0 + l) * out_len..(j0 + l + 1) * out_len];
                    for (e, o) in dst.iter_mut().enumerate() {
                        *o = root[e * k + l];
                    }
                }
                j0 += k;
            }
            Ok(())
        });
        batch_result.map_err(|e| PyValueError::new_err(e.to_string()))?;

        // out is laid out column-major: (out_len, n_t) in F order.
        // from_shape_vec is infallible here, out has exactly out_len * n_t elements.
        let arr = numpy::ndarray::Array2::from_shape_vec((out_len, n_t).f(), out)
            .map_err(|e| PyValueError::new_err(format!("eval_trajectory shape error: {e}")))?;
        Ok(PyArray2::from_owned_array(py, arr))
    }

    /// Forward-mode JVP swept along a trajectory: for each time column `j`,
    /// `df/dy(t_j, y_j) @ vy_j (+ df/dp(t_j, y_j) @ vp when given)`.
    ///
    /// `vy_traj` is `(n_states, n_t)`, one `yS` parameter-column over time;
    /// `vp` is the constant parameter direction `e_k`, one entry per
    /// registered parameter name (`n_inputs`, not the packed width).
    /// Returns `(output_len, n_t)` F-contiguous, mirroring `eval_trajectory`:
    /// one tangent-tape eval per column, scratch reused, GIL released. The
    /// tangent tapes are the same lazily-derived, cached tapes `jvp` uses.
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
        // The tangent tape slices an empty y_dot and would panic; d/d(y_dot)
        // seeding is a solver concern (cj), mirroring jvp/eval_trajectory.
        self.sig.reject_y_dot("jvp_trajectory")?;
        let packed = self.sig.extract_p(p)?;
        let ts_view = ts.as_array();
        let ts_slice = contiguous_slice_1d(&ts_view);
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
        let vy_view = vy_traj.as_array();
        if vy_view.shape()[0] != self.sig.n_states || vy_view.shape()[1] != n_t {
            return Err(PyValueError::new_err(format!(
                "{}: vy_traj shape must be (n_states, n_t) = ({}, {}), got ({}, {})",
                self.sig.display_name(),
                self.sig.n_states,
                n_t,
                vy_view.shape()[0],
                vy_view.shape()[1]
            )));
        }

        // Zero-copy when F-contiguous; one gathering copy otherwise.
        let y_cols = columns_slice(&view);
        let vy_cols = columns_slice(&vy_view);

        // y-tangent tape (always); p-tangent tape only when vp is supplied.
        let ty = self.tangent_entry(py, false)?;
        let vp_slice: Option<&[f64]> = match &vp {
            Some(arr) => {
                let s = arr.as_slice()?;
                self.sig.check_vp(s.len())?;
                Some(s)
            },
            None => None,
        };
        let tp = match vp_slice {
            Some(_) => Some(self.tangent_entry(py, true)?),
            None => None,
        };

        let out_len = self.sig.output_len;
        let mut out = vec![0.0_f64; out_len * n_t];
        let mut scratch_y = ty.pool.acquire();
        let mut scratch_p = tp.map(|t| t.pool.acquire());
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

        // Column-major (out_len, n_t) in F order; from_shape_vec is infallible
        // here, out has exactly out_len * n_t elements.
        let arr = numpy::ndarray::Array2::from_shape_vec((out_len, n_t).f(), out)
            .map_err(|e| PyValueError::new_err(format!("jvp_trajectory shape error: {e}")))?;
        Ok(PyArray2::from_owned_array(py, arr))
    }

    /// Cubic-Hermite reconstruct the state at each `t_query` from the solver
    /// knots (`ts`, `ys`, `yps`), then evaluate the compiled graph. Mirrors the
    /// C++ `observe.cpp::observe_hermite_interp` math on the Rust evaluator;
    /// returns `(output_len, n_query)` F-contiguous like `eval_trajectory`.
    #[pyo3(signature = (t_query, ts, ys, yps, p))]
    #[allow(clippy::suboptimal_flops)] // mixed-sign cubic-Hermite basis; mul_add would obscure it, matches eval.rs precedent
    fn eval_trajectory_hermite<'py>(
        &self,
        py: Python<'py>,
        t_query: PyArrayLike1<'_, f64, AllowTypeChange>,
        ts: PyArrayLike1<'_, f64, AllowTypeChange>,
        ys: PyReadonlyArray2<'_, f64>,
        yps: PyReadonlyArray2<'_, f64>,
        p: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.sig.reject_y_dot("eval_trajectory_hermite")?;
        let packed = self.sig.extract_p(p)?;
        let tq_view = t_query.as_array();
        let tq = contiguous_slice_1d(&tq_view);
        let ts_view = ts.as_array();
        let ts_slice = contiguous_slice_1d(&ts_view);
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

        // Zero-copy when F-contiguous; one gathering copy otherwise.
        let y_cols = columns_slice(&y_view);
        let yp_cols = columns_slice(&yp_view);

        let out_len = self.sig.output_len;
        let n_query = tq.len();
        let mut out = vec![0.0_f64; out_len * n_query];
        let expr = Arc::clone(&self.expr);
        let scratch_len = expr.scratch_len();
        // Lane-batched tiling: ~1 MiB working set, 8..=64 lanes, capped at n_query.
        let k_max = (TARGET_BYTES / (8 * scratch_len.max(1))).clamp(8, 64);
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
                    &packed,
                )?;
                // root is (out_len, k) lane-minor: element e lane l at root[e*k + l].
                for l in 0..k {
                    let dst = &mut out[(q0 + l) * out_len..(q0 + l + 1) * out_len];
                    for (e, o) in dst.iter_mut().enumerate() {
                        *o = root[e * k + l];
                    }
                }
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
        format!(
            "CompiledFunction(name={:?}, inputs={:?}, n_states={}, output_len={}, uses_y_dot={})",
            self.sig.display_name(),
            self.sig.input_names,
            self.sig.n_states,
            self.sig.output_len,
            self.sig.uses_y_dot
        )
    }

    /// Rebuild from the retained `(graph, root)` derivation source (pickle
    /// protocol). Recompiles the tape and resets the lazily-derived
    /// tangent/jacobian caches; `n_states` is re-pinned exactly so a
    /// user-widened system round-trips at the same width.
    #[staticmethod]
    fn _rebuild(
        py: Python<'_>,
        graph: Py<ExprGraph>,
        root: u32,
        name: Option<String>,
        n_states: Option<usize>,
    ) -> PyResult<Self> {
        let root = NodeId::from(root);
        {
            // Reachable from Python with an arbitrary root: validate before
            // lowering, mirroring the `graph.compile` entry point.
            let g = graph.borrow(py);
            if root.index() >= g.arena().len() {
                return Err(PyValueError::new_err(format!(
                    "_rebuild: root {} is out of range for a graph of {} nodes",
                    root.raw(),
                    g.arena().len()
                )));
            }
            crate::expr::check_supported(g.arena(), root)?;
        }
        Self::build(py, graph, root, name, n_states)
    }

    /// `(callable, args)` pair for the pickle protocol: rebuild from the
    /// retained graph and root rather than serializing derived state.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, RebuildArgs)> {
        let this = slf.get();
        let rebuild = slf.get_type().getattr("_rebuild")?;
        Ok((
            rebuild,
            (
                this.graph.clone_ref(py),
                this.root.raw(),
                this.name(),
                Some(this.n_states()),
            ),
        ))
    }
}

/// Column-major slice of Y: zero-copy when Y is F-contiguous (strides[0] == 1),
/// otherwise at most one gathering copy.
pub fn columns_slice<'a>(
    view: &'a numpy::ndarray::ArrayView2<'a, f64>,
) -> std::borrow::Cow<'a, [f64]> {
    match view.as_slice_memory_order() {
        // strides[0] == 1 and all strides non-negative ⇒ true F-contiguous column-major
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

/// Contiguous view of a 1-D time array: zero-copy when already contiguous,
/// otherwise one gathering copy. `PyArrayLike1<f64, AllowTypeChange>` only
/// materialises a fresh copy on dtype mismatch or non-ndarray input; an
/// already-f64 but strided ndarray passes through unmaterialized, so
/// `.as_slice()` alone would still reject it.
pub fn contiguous_slice_1d<'a>(
    view: &'a numpy::ndarray::ArrayView1<'a, f64>,
) -> std::borrow::Cow<'a, [f64]> {
    view.as_slice().map_or_else(
        || std::borrow::Cow::Owned(view.to_vec()),
        std::borrow::Cow::Borrowed,
    )
}

/// Bracketing interval index for `x` in ascending `ts`, clamped to `[0, n-2]`
/// (extends the boundary segment). `ts.len() >= 2` is enforced by the caller.
pub fn locate_interval(ts: &[f64], x: f64) -> usize {
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

/// Compile prep = simplify + instruction tape: same pass order as
/// `CompiledModel::new`, run on a clone so the retained graph stays
/// the unsimplified derivation source for jvp / jacobian (which re-run
/// the pipeline on their own tangent arenas). Without this, user-compiled
/// functions get systematically worse tapes than the bundle's, and group
/// CSE only catches shared `NodeIds`, not structurally identical nodes
/// from separate builds.
pub fn compile_simplified(arena: &Arena, root: NodeId) -> CompiledExpr {
    let (da, root) = simplify_pipeline(arena.clone(), root);
    CompiledExpr::from_ir(TypedIr::from_arena(&da, root))
}

/// Resolve the user-supplied `n_states` override against the scanned
/// extent: an override below it would admit a too-short `y` and panic
/// inside the tape.
pub fn resolve_n_states(
    scanned: usize,
    requested: Option<usize>,
    name: Option<&str>,
) -> PyResult<usize> {
    match requested {
        Some(n) if n < scanned => Err(PyValueError::new_err(format!(
            "{}: n_states={n} is below the expression's state extent {scanned}",
            name.unwrap_or("<anonymous>"),
        ))),
        Some(n) => Ok(n),
        None => Ok(scanned),
    }
}
