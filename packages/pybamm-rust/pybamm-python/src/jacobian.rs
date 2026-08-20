//! `CompiledJacobian`: the prepared derivative artifact.

// PyO3 bindings require specific argument types that clippy flags incorrectly
#![allow(clippy::needless_pass_by_value)]

use std::sync::Arc;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

use pybamm_core::JacobianData;

use crate::scratch::ScratchPool;
use crate::signature::FunctionSignature;

#[pyclass(frozen, module = "pybamm.rust")]
pub struct CompiledJacobian {
    pub(crate) data: Arc<JacobianData>,
    pub(crate) sig: FunctionSignature,
    scratch_pool: ScratchPool<Arc<JacobianData>>,
    /// CSC index arrays, built once: **int32** so scipy adopts
    /// them without a cast-copy, and **read-only** so in-place
    /// canonicalisation of one returned matrix (`sort_indices`,
    /// `sum_duplicates`) raises instead of corrupting the pattern
    /// shared by every other matrix. Each call allocates only the
    /// data array.
    indptr: Py<PyArray1<i32>>,
    indices: Py<PyArray1<i32>>,
}

impl std::fmt::Debug for CompiledJacobian {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledJacobian")
            .field("sig", &self.sig)
            .field("wrt", &self.data.wrt())
            .field("shape", &(self.data.n_rows(), self.data.n_cols()))
            .field("nnz", &self.data.nnz())
            .finish_non_exhaustive()
    }
}

fn to_readonly_i32(py: Python<'_>, values: &[usize]) -> PyResult<Py<PyArray1<i32>>> {
    let v: Vec<i32> = values
        .iter()
        .map(|&x| {
            i32::try_from(x).map_err(|_| {
                PyValueError::new_err("jacobian pattern exceeds the int32 index range")
            })
        })
        .collect::<PyResult<_>>()?;
    let arr = PyArray1::from_vec(py, v);
    arr.call_method("setflags", (), Some(&[("write", false)].into_py_dict(py)?))?;
    Ok(arr.unbind())
}

impl CompiledJacobian {
    pub(crate) fn build(
        py: Python<'_>,
        data: Arc<JacobianData>,
        sig: FunctionSignature,
    ) -> PyResult<Self> {
        Ok(Self {
            scratch_pool: ScratchPool::new(Arc::clone(&data)),
            indptr: to_readonly_i32(py, &data.csc().colptr)?,
            indices: to_readonly_i32(py, &data.csc().rowind)?,
            data,
            sig,
        })
    }
}

#[pymethods]
impl CompiledJacobian {
    /// Assemble and return a `scipy.sparse.csc_matrix`. Per-call work is
    /// `n_colors` JVP sweeps + linear scatter; only the data array allocates.
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        y: PyReadonlyArray1<'_, f64>,
        p: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let packed = self.sig.extract_p(p)?;
        let y_slice = y.as_slice()?;
        self.sig.check_y(y_slice.len())?;

        let mut values = vec![0.0; self.data.nnz()];
        let mut scratch = self.scratch_pool.acquire();
        let data = &self.data;
        py.detach(|| {
            data.assemble_into(
                &mut scratch,
                data.layout(),
                t,
                y_slice,
                &[],
                &packed,
                &mut values,
            );
        });
        self.scratch_pool.release(scratch);

        let data_arr = PyArray1::from_vec(py, values);
        let csc = py.import("scipy.sparse")?.getattr("csc_matrix")?.call1((
            (data_arr, self.indices.bind(py), self.indptr.bind(py)),
            (self.data.n_rows(), self.data.n_cols()),
        ))?;
        Ok(csc)
    }

    /// CSC pattern as (indptr, indices), the cached read-only arrays.
    fn sparsity<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i32>>) {
        (self.indptr.bind(py).clone(), self.indices.bind(py).clone())
    }

    #[getter]
    fn nnz(&self) -> usize {
        self.data.nnz()
    }
    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.data.n_rows(), self.data.n_cols())
    }
    #[getter]
    fn n_colors(&self) -> usize {
        self.data.n_colors()
    }
    /// Number of dense rows split out of the column coloring; they share one
    /// reverse-mode tape, seeded once per row.
    #[getter]
    fn n_dense_rows(&self) -> usize {
        self.data.n_dense_rows()
    }
    #[getter]
    fn wrt(&self) -> &'static str {
        match self.data.wrt() {
            pybamm_core::DiffTarget::States => "y",
            pybamm_core::DiffTarget::Params => "p",
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CompiledJacobian(of={:?}, wrt={:?}, shape=({}, {}), nnz={}, n_colors={}, n_dense_rows={})",
            self.sig.display_name(),
            self.wrt(),
            self.data.n_rows(),
            self.data.n_cols(),
            self.data.nnz(),
            self.data.n_colors(),
            self.data.n_dense_rows()
        )
    }
}
