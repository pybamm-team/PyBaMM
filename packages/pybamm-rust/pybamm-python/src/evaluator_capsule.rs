//! Python capsules that hand a [`ModelEvaluator`] to a C consumer, which claims it.

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

use pybamm_core::ModelEvaluator;
use pybamm_core::ffi::{EVALUATOR_CAPSULE, EvaluatorHandle, pybamm_rust_release};

// A consumer runs the evaluator on its own threads, so it must stay `Send`.
const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<ModelEvaluator>();
};

/// Box `evaluator` into a `pybamm.rust.evaluator` capsule, which frees it unless claimed.
#[allow(unsafe_code)]
pub fn evaluator_capsule(
    py: Python<'_>,
    evaluator: ModelEvaluator,
) -> PyResult<Bound<'_, PyCapsule>> {
    let handle = EvaluatorHandle::into_raw(evaluator);
    // SAFETY: `handle` is live until released, and only `release_unclaimed` or the consumer
    // that renames the capsule releases it.
    unsafe {
        PyCapsule::new_with_pointer_and_destructor(
            py,
            handle.cast(),
            EVALUATOR_CAPSULE,
            Some(release_unclaimed),
        )
    }
    // SAFETY: no capsule was made, so nothing else holds `handle`.
    .inspect_err(|_| unsafe { pybamm_rust_release(handle.as_ptr()) })
}

/// Release the capsule's handle, unless a consumer claimed it by renaming the capsule.
#[allow(unsafe_code)]
unsafe extern "C" fn release_unclaimed(capsule: *mut ffi::PyObject) {
    let name = EVALUATOR_CAPSULE.as_ptr();
    // SAFETY: CPython passes the capsule it is destroying; `PyCapsule_IsValid` raises nothing,
    // and only an unclaimed capsule still holds a handle this side must release.
    unsafe {
        if ffi::PyCapsule_IsValid(capsule, name) == 1 {
            pybamm_rust_release(ffi::PyCapsule_GetPointer(capsule, name).cast());
        }
    }
}
