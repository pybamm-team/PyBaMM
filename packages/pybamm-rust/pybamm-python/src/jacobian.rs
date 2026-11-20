//! `CompiledJacobian`: the prepared derivative artifact.

// PyO3 bindings require specific argument types that clippy flags incorrectly
#![allow(clippy::needless_pass_by_value)]

use std::sync::Arc;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyType;

use pybamm_core::{DiffTarget, JacobianData, JacobianScratch};

use crate::arrays;
use crate::function_metadata::FunctionMetadata;
use crate::scratch_pool::ScratchPool;

/// `scipy.sparse.csc_matrix`, imported on first assembly rather than on every one.
static CSC_MATRIX: PyOnceLock<Py<PyType>> = PyOnceLock::new();

#[pyclass(frozen, module = "pybamm.rust")]
pub struct CompiledJacobian {
    data: Arc<JacobianData>,
    metadata: FunctionMetadata,
    scratch_pool: ScratchPool<JacobianScratch>,
    /// CSC pattern as int32, the index type scipy adopts without converting.
    indptr: Box<[i32]>,
    indices: Box<[i32]>,
}

impl std::fmt::Debug for CompiledJacobian {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledJacobian")
            .field("metadata", &self.metadata)
            .field("wrt", &self.data.wrt())
            .field("shape", &(self.data.n_rows(), self.data.n_cols()))
            .field("nnz", &self.data.nnz())
            .finish_non_exhaustive()
    }
}

fn to_i32(values: &[usize]) -> PyResult<Box<[i32]>> {
    values
        .iter()
        .map(|&x| {
            i32::try_from(x).map_err(|_| {
                PyValueError::new_err("jacobian pattern exceeds the int32 index range")
            })
        })
        .collect()
}

impl CompiledJacobian {
    pub(crate) fn build(data: Arc<JacobianData>, metadata: FunctionMetadata) -> PyResult<Self> {
        Ok(Self {
            scratch_pool: ScratchPool::default(),
            indptr: to_i32(data.csc().colptr())?,
            indices: to_i32(data.csc().rowind())?,
            data,
            metadata,
        })
    }
}

#[pymethods]
impl CompiledJacobian {
    /// Assemble a `scipy.sparse.csc_matrix` that owns all of its arrays.
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        y: PyReadonlyArray1<'_, f64>,
        p: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let packed = self.metadata.extract_p(p)?;
        let y = arrays::values(&y);
        self.metadata
            .check_len("y", y.len(), self.metadata.n_states)?;

        let mut values = vec![0.0; self.data.nnz()];
        let mut scratch = self
            .scratch_pool
            .acquire(|| JacobianScratch::new(&self.data));
        let data = &self.data;
        py.detach(|| {
            data.assemble_into(
                &mut scratch,
                data.layout(),
                t,
                &y,
                &[],
                &packed,
                &mut values,
            );
        });
        self.scratch_pool.release(scratch);

        let (indptr, indices) = self.sparsity(py);
        CSC_MATRIX.import(py, "scipy.sparse", "csc_matrix")?.call1((
            (PyArray1::from_vec(py, values), indices, indptr),
            self.shape(),
        ))
    }

    /// CSC pattern as `(indptr, indices)`, in new int32 arrays.
    fn sparsity<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i32>>) {
        (
            PyArray1::from_slice(py, &self.indptr),
            PyArray1::from_slice(py, &self.indices),
        )
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
    /// Number of dense rows split out of the column coloring.
    #[getter]
    fn n_dense_rows(&self) -> usize {
        self.data.n_dense_rows()
    }
    #[getter]
    fn wrt(&self) -> &'static str {
        match self.data.wrt() {
            DiffTarget::States => "y",
            DiffTarget::Params => "p",
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CompiledJacobian(of={:?}, wrt={:?}, shape=({}, {}), nnz={}, n_colors={}, n_dense_rows={})",
            self.metadata.display_name(),
            self.wrt(),
            self.data.n_rows(),
            self.data.n_cols(),
            self.data.nnz(),
            self.data.n_colors(),
            self.data.n_dense_rows()
        )
    }
}
