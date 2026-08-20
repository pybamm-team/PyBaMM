//! FFI (Foreign Function Interface) for IDAKLU solver integration.
//!
//! This module provides C ABI wrappers for the `ModelEvaluator` methods,
//! allowing the Rust expression evaluator to be called from C/C++ code
//! (specifically the IDAKLU DAE solver used by `PyBaMM`).
//!
//! # Safety
//!
//! All functions in this module are `unsafe extern "C"` and require:
//! - Valid, non-null pointers for all array arguments
//! - Correctly sized arrays (`n_states` elements)
//! - Valid `user_data` pointer to a `ModelEvaluator` instance
//!
//! Every entry point enters through [`with_model`] (a query) or
//! [`with_model_and_inputs`] (an evaluation), which own the whole boundary
//! contract — null rejection, panic containment, the cast of `user_data`, and
//! the input-parameter buffer — so an entry point body is only its own work.

// FFI code requires unsafe - this module is intentionally using unsafe patterns
// for C interop with proper safety checks at the boundaries.
#![allow(unsafe_code)]
#![allow(clippy::not_unsafe_ptr_arg_deref)] // We check null before deref
#![allow(clippy::missing_panics_doc)] // Panics are caught at boundary
#![allow(clippy::missing_const_for_fn)] // FFI functions cannot be const

use crate::model::ModelEvaluator;
use crate::observable::ObservableKind;
use std::ffi::c_void;
use std::os::raw::c_int;

/// Success return code.
pub const SUCCESS: c_int = 0;

/// Error: null pointer passed to function.
pub const ERROR_NULL_POINTER: c_int = -1;

/// Error: panic occurred during execution.
pub const ERROR_PANIC: c_int = -2;

/// Error: invalid parameter index supplied to a sensitivity-related FFI fn.
pub const ERROR_INVALID_PARAM: c_int = -3;

/// Error: invalid output-variable index supplied to an output-related FFI fn.
pub const ERROR_INVALID_OUTPUT: c_int = -4;

/// Error: caller invoked a sensitivity FFI on a model with no sensitivities.
pub const ERROR_NO_SENS: c_int = -6;

/// Error: caller invoked an output FFI on a model with no output variables.
pub const ERROR_NO_OUTPUTS: c_int = -7;

/// Error: caller invoked an algebraic FFI on a model with no algebraic block.
pub const ERROR_NO_ALG: c_int = -8;

/// Error: caller invoked an event FFI on a model with no events.
pub const ERROR_NO_EVENTS: c_int = -9;

/// ABI contract version for the Rust FFI surface.
///
/// Bump by 1 whenever ANY exported signature changes,
/// arg/return types, or a function added, removed, or reordered. The C++
/// consumer pins the expected value in `PYBAMM_RUST_ABI_VERSION`, and the
/// drift test asserts the two are equal.
///
/// `pybamm_rust_abi_version` itself must forever keep the signature `-> u32` with no
/// arguments: it is the probe the C++ consumer calls to read this version, so it
/// cannot follow the bump rule it enforces.
pub const RUST_ABI_VERSION: u32 = 1;

/// Return the FFI ABI contract version.
///
/// # Safety
///
/// Takes no pointer arguments and is always safe to call.
#[unsafe(no_mangle)]
pub extern "C" fn pybamm_rust_abi_version() -> u32 {
    RUST_ABI_VERSION
}

/// Golden hash of the normalized exported symbol surface, pinned by the
/// `test_ffi_abi_contract` drift test.
///
/// Changing any exported signature changes this hash and fails that test,
/// which then instructs you to update this value AND bump `RUST_ABI_VERSION`
/// / `PYBAMM_RUST_ABI_VERSION` in lockstep. This makes the version bump
/// enforceable rather than a manual convention.
pub const EXPECTED_ABI_HASH: u64 = 0x6584_760b_149a_9779;

/// Run `body` with the model borrowed immutably from `user_data`.
///
/// Returns `ERROR_NULL_POINTER` if `user_data` or any pointer in `required` is
/// null, and `ERROR_PANIC` if `body` unwinds; otherwise `body`'s own status.
///
/// # Safety
///
/// `user_data` must point to a live `ModelEvaluator` that stays valid, and is
/// not mutably aliased, for the call.
#[inline]
unsafe fn with_model<const N: usize, F>(
    user_data: *const c_void,
    required: [*const c_void; N],
    body: F,
) -> c_int
where
    F: FnOnce(&ModelEvaluator) -> c_int,
{
    if user_data.is_null() || any_null(required) {
        return ERROR_NULL_POINTER;
    }
    // SAFETY: null-checked above; the caller guarantees a live `ModelEvaluator`.
    let model = unsafe { &*user_data.cast::<ModelEvaluator>() };
    caught(|| body(model))
}

/// Run `body` with the model borrowed mutably from `user_data` and its input
/// parameters borrowed from `inputs`.
///
/// Rejects nulls and contains panics as [`with_model`] does, and holds `inputs`
/// to the same rule: null is accepted only when the model declares no input
/// parameters. A body handed an empty slice for a model that has parameters
/// would read past it and panic, reporting a caller's null as a Rust fault.
///
/// # Safety
///
/// `user_data` must point to a live `ModelEvaluator` that stays valid, and is
/// not otherwise aliased, for the call. When the model declares input
/// parameters, `inputs` must point to at least that many `f64`.
#[inline]
unsafe fn with_model_and_inputs<const N: usize, F>(
    user_data: *mut c_void,
    required: [*const c_void; N],
    inputs: *const f64,
    body: F,
) -> c_int
where
    F: FnOnce(&mut ModelEvaluator, &[f64]) -> c_int,
{
    if user_data.is_null() || any_null(required) {
        return ERROR_NULL_POINTER;
    }
    // SAFETY: null-checked above; the caller guarantees a live `ModelEvaluator`.
    let model = unsafe { &mut *user_data.cast::<ModelEvaluator>() };
    let n_params = model.n_params();
    if n_params != 0 && inputs.is_null() {
        return ERROR_NULL_POINTER;
    }
    let inputs: &[f64] = if n_params == 0 {
        &[]
    } else {
        // SAFETY: non-null per the check above, and the caller guarantees
        // `n_params` elements.
        unsafe { borrow_slice(inputs, n_params) }
    };
    caught(|| body(model, inputs))
}

/// Whether any caller-supplied buffer pointer is null.
#[inline]
fn any_null<const N: usize>(pointers: [*const c_void; N]) -> bool {
    pointers.iter().any(|pointer| pointer.is_null())
}

/// Run `body`, turning an unwind into `ERROR_PANIC` rather than letting it
/// cross back into C, which would be undefined behaviour.
#[inline]
fn caught<F: FnOnce() -> c_int>(body: F) -> c_int {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)).unwrap_or(ERROR_PANIC)
}

/// Borrow `len` elements of a caller-provided buffer.
///
/// # Safety
///
/// `pointer` must be non-null and point to at least `len` initialized `T`,
/// unaliased for the borrow.
#[inline]
unsafe fn borrow_slice<'a, T>(pointer: *const T, len: usize) -> &'a [T] {
    // SAFETY: guaranteed by this function's own contract.
    unsafe { std::slice::from_raw_parts(pointer, len) }
}

/// Borrow `len` writable elements of a caller-provided buffer.
///
/// # Safety
///
/// `pointer` must be non-null and point to at least `len` writable `T`,
/// unaliased for the borrow.
#[inline]
unsafe fn borrow_slice_mut<'a, T>(pointer: *mut T, len: usize) -> &'a mut [T] {
    // SAFETY: guaranteed by this function's own contract.
    unsafe { std::slice::from_raw_parts_mut(pointer, len) }
}

/// Time the enclosing scope into `$counter`, or expand to nothing when the
/// `profile` feature is off.
macro_rules! profile_scope {
    ($counter:ident) => {
        #[cfg(feature = "profile")]
        let _profile_scope = profile::$counter.scope();
    };
}

/// Compile-time FFI profiling, absent entirely when the feature is disabled.
#[cfg(feature = "profile")]
mod profile {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    /// Calls and accumulated wall time for one FFI entry point.
    #[derive(Debug)]
    pub struct Counter {
        calls: AtomicU64,
        nanos: AtomicU64,
    }

    impl Counter {
        const fn new() -> Self {
            Self {
                calls: AtomicU64::new(0),
                nanos: AtomicU64::new(0),
            }
        }

        /// Start timing a call; the returned scope records it when dropped.
        pub fn scope(&'static self) -> Scope {
            Scope {
                counter: self,
                start: Instant::now(),
            }
        }

        /// Read and reset the accumulated `(calls, nanos)`.
        pub fn take(&self) -> (u64, u64) {
            (
                self.calls.swap(0, Ordering::Relaxed),
                self.nanos.swap(0, Ordering::Relaxed),
            )
        }
    }

    /// Records one call into its counter on drop, including on unwind.
    #[derive(Debug)]
    pub struct Scope {
        counter: &'static Counter,
        start: Instant,
    }

    impl Drop for Scope {
        fn drop(&mut self) {
            self.counter.calls.fetch_add(1, Ordering::Relaxed);
            self.counter.nanos.fetch_add(
                u64::try_from(self.start.elapsed().as_nanos()).unwrap_or(u64::MAX),
                Ordering::Relaxed,
            );
        }
    }

    pub static RESIDUAL: Counter = Counter::new();
    pub static JAC_ASSEMBLE: Counter = Counter::new();
    pub static JAC_MUL: Counter = Counter::new();
    pub static RHS_EVAL: Counter = Counter::new();

    /// Every profiled entry point, in report order.
    pub const ALL: [(&str, &Counter); 4] = [
        ("residual", &RESIDUAL),
        ("jac_assemble", &JAC_ASSEMBLE),
        ("jac_mul", &JAC_MUL),
        ("rhs_eval", &RHS_EVAL),
    ];
}

/// Print accumulated FFI profiling statistics and reset all counters.
///
/// Only available when compiled with `--features profile`.
///
/// # Safety
///
/// This function has no pointer arguments and is always safe to call.
#[cfg(feature = "profile")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pybamm_rust_profile_report() {
    eprintln!("=== pybamm-core FFI profile ===");
    for (name, counter) in profile::ALL {
        let (calls, nanos) = counter.take();
        eprintln!(
            "  {name:<13} {calls:>8} calls, {:.3} ms total",
            nanos as f64 / 1_000_000.0
        );
    }
    eprintln!("===============================");
}

/// Evaluate residual: r = M*y' - f(t,y)
///
/// This computes the DAE residual function for IDAKLU. For a system
/// M*y' = f(t,y), the residual is M*y' - f(t,y).
///
/// # Safety
///
/// - `y`, `yp`, `r`, and `user_data` must be valid and non-null
/// - `y` and `yp` must point to arrays of at least `model.n_states()` elements
/// - `r` must point to an array of at least `model.output_len()` elements
/// - `inputs` may be null when the model has no input parameters; otherwise it
///   must point to an array of at least `model.n_params()` elements
/// - `user_data` must point to a valid `ModelEvaluator` instance
///
/// # Returns
///
/// - `SUCCESS` (0) on success
/// - `ERROR_NULL_POINTER` (-1) if any pointer is null
/// - `ERROR_PANIC` (-2) if a Rust panic occurred
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pybamm_rust_residual(
    t: f64,
    y: *const f64,
    yp: *const f64,
    inputs: *const f64,
    r: *mut f64,
    user_data: *mut c_void,
) -> c_int {
    unsafe {
        with_model_and_inputs(
            user_data,
            [y.cast(), yp.cast(), r.cast()],
            inputs,
            |model, inputs| {
                profile_scope!(RESIDUAL);
                let (n_states, n_out) = (model.n_states(), model.output_len());
                model.eval_residual(
                    t,
                    borrow_slice(y, n_states),
                    borrow_slice(yp, n_states),
                    inputs,
                    borrow_slice_mut(r, n_out),
                );
                SUCCESS
            },
        )
    }
}

/// Compute Jacobian-vector product: (df/dy - cj*M) @ v
///
/// This computes the matrix-vector product needed for Newton iteration
/// in the DAE solver. The Jacobian is J = df/dy - cj*M where cj is a
/// scalar coefficient provided by the solver.
///
/// # Safety
///
/// - `y`, `v`, `jv`, and `user_data` must be valid and non-null
/// - `y`, `v`, and `jv` must point to arrays of at least `model.n_states()` elements
/// - `inputs` may be null when the model has no input parameters; otherwise it
///   must point to an array of at least `model.n_params()` elements
/// - `user_data` must point to a valid `ModelEvaluator` instance
///
/// # Returns
///
/// - `SUCCESS` (0) on success
/// - `ERROR_NULL_POINTER` (-1) if any pointer is null
/// - `ERROR_PANIC` (-2) if a Rust panic occurred
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pybamm_rust_jac_mul(
    t: f64,
    y: *const f64,
    inputs: *const f64,
    cj: f64,
    v: *const f64,
    jv: *mut f64,
    user_data: *mut c_void,
) -> c_int {
    unsafe {
        with_model_and_inputs(
            user_data,
            [y.cast(), v.cast(), jv.cast()],
            inputs,
            |model, inputs| {
                profile_scope!(JAC_MUL);
                let n_states = model.n_states();
                model.set_cj(cj);
                model.jac_mul(
                    t,
                    borrow_slice(y, n_states),
                    inputs,
                    borrow_slice(v, n_states),
                    borrow_slice_mut(jv, n_states),
                );
                SUCCESS
            },
        )
    }
}

/// Evaluate the right-hand side f(t, y).
///
/// This computes f(t, y) for the DAE system M*y' = f(t, y).
///
/// # Safety
///
/// - `y`, `f_out`, and `user_data` must be valid and non-null
/// - `y` must point to an array of at least `model.n_states()` elements
/// - `f_out` must point to an array of at least `model.output_len()` elements
/// - `inputs` may be null when the model has no input parameters; otherwise it
///   must point to an array of at least `model.n_params()` elements
/// - `user_data` must point to a valid `ModelEvaluator` instance
///
/// # Returns
///
/// - `SUCCESS` (0) on success
/// - `ERROR_NULL_POINTER` (-1) if any pointer is null
/// - `ERROR_PANIC` (-2) if a Rust panic occurred
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pybamm_rust_eval_rhs(
    t: f64,
    y: *const f64,
    inputs: *const f64,
    f_out: *mut f64,
    user_data: *mut c_void,
) -> c_int {
    unsafe {
        with_model_and_inputs(
            user_data,
            [y.cast(), f_out.cast()],
            inputs,
            |model, inputs| {
                profile_scope!(RHS_EVAL);
                let (n_states, n_out) = (model.n_states(), model.output_len());
                model.eval_rhs(
                    t,
                    borrow_slice(y, n_states),
                    inputs,
                    borrow_slice_mut(f_out, n_out),
                );
                SUCCESS
            },
        )
    }
}

/// Get the number of states in the model.
///
/// # Safety
///
/// - `user_data` must point to a valid `ModelEvaluator` instance
///
/// # Returns
///
/// - Number of states on success (>= 0)
/// - `ERROR_NULL_POINTER` (-1) if `user_data` is null
/// - `ERROR_PANIC` (-2) if a Rust panic occurred
#[unsafe(no_mangle)]
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub unsafe extern "C" fn pybamm_rust_n_states(user_data: *const c_void) -> c_int {
    unsafe { with_model(user_data, [], |model| model.n_states() as c_int) }
}

/// Get the number of input parameters in the model.
///
/// # Safety
///
/// - `user_data` must point to a valid `ModelEvaluator` instance
///
/// # Returns
///
/// - Number of input parameters on success (>= 0)
/// - `ERROR_NULL_POINTER` (-1) if `user_data` is null
/// - `ERROR_PANIC` (-2) if a Rust panic occurred
#[unsafe(no_mangle)]
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub unsafe extern "C" fn pybamm_rust_n_inputs(user_data: *const c_void) -> c_int {
    unsafe { with_model(user_data, [], |model| model.n_params() as c_int) }
}

/// Write algebraic-state IDs into the provided buffer using IDA's convention:
/// `1.0` for differential, `0.0` for algebraic.
///
/// # Safety
///
/// - `ids_out` must point to an array of at least `model.n_states()` elements
/// - `user_data` must point to a valid `ModelEvaluator` instance
///
/// # Returns
///
/// - `SUCCESS` (0) on success
/// - `ERROR_NULL_POINTER` (-1) if any pointer is null
/// - `ERROR_PANIC` (-2) if a Rust panic occurred
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pybamm_rust_algebraic_ids(
    ids_out: *mut f64,
    user_data: *mut c_void,
) -> c_int {
    unsafe {
        with_model(user_data, [ids_out.cast()], |model| {
            model.algebraic_ids_f64(borrow_slice_mut(ids_out, model.n_states()));
            SUCCESS
        })
    }
}

/// Assemble the Jacobian matrix into a pre-allocated CSC data buffer.
///
/// This is the zero-allocation callback for IDAKLU integration. The Jacobian
/// is computed as `J = df/dy - cj * M` and stored in CSC format.
///
/// # Safety
///
/// - `y`, `jac_data`, and `user_data` must be valid and non-null
/// - `y` must point to an array of at least `model.n_states()` elements
/// - `jac_data` must point to an array of at least `model.nnz()` elements
/// - `inputs` may be null when the model has no input parameters; otherwise it
///   must point to an array of at least `model.n_params()` elements
/// - `user_data` must point to a valid `ModelEvaluator` instance
///
/// # Returns
///
/// - `SUCCESS` (0) on success
/// - `ERROR_NULL_POINTER` (-1) if any pointer is null
/// - `ERROR_PANIC` (-2) if a Rust panic occurred
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pybamm_rust_jac_assemble(
    t: f64,
    y: *const f64,
    inputs: *const f64,
    cj: f64,
    jac_data: *mut f64,
    user_data: *mut c_void,
) -> c_int {
    unsafe {
        with_model_and_inputs(
            user_data,
            [y.cast(), jac_data.cast()],
            inputs,
            |model, inputs| {
                profile_scope!(JAC_ASSEMBLE);
                let (n_states, nnz) = (model.n_states(), model.nnz());
                model.set_cj(cj);
                model.assemble_jacobian_csc_into(
                    t,
                    borrow_slice(y, n_states),
                    inputs,
                    borrow_slice_mut(jac_data, nnz),
                );
                SUCCESS
            },
        )
    }
}

/// Get the number of non-zeros in the Jacobian.
///
/// # Safety
///
/// - `user_data` must point to a valid `ModelEvaluator` instance
///
/// # Returns
///
/// - Number of non-zeros on success (>= 0)
/// - `ERROR_NULL_POINTER` (-1) if `user_data` is null
/// - `ERROR_PANIC` (-2) if a Rust panic occurred
#[unsafe(no_mangle)]
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub unsafe extern "C" fn pybamm_rust_jac_nnz(user_data: *const c_void) -> c_int {
    unsafe { with_model(user_data, [], |model| model.nnz() as c_int) }
}

/// Compute pure Jacobian-vector product: df/dy @ v (no mass term)
///
/// This matches pybammsolvers ABI where mass is subtracted separately.
///
/// # Safety
///
/// - `y`, `v`, `jv`, and `user_data` must be valid and non-null
/// - `y`, `v`, and `jv` must point to arrays of at least `model.n_states()` elements
/// - `inputs` may be null when the model has no input parameters; otherwise it
///   must point to an array of at least `model.n_params()` elements
/// - `user_data` must point to a valid `ModelEvaluator` instance
///
/// # Returns
///
/// - `SUCCESS` (0) on success
/// - `ERROR_NULL_POINTER` (-1) if any pointer is null
/// - `ERROR_PANIC` (-2) if a Rust panic occurred
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pybamm_rust_jac_action(
    t: f64,
    y: *const f64,
    inputs: *const f64,
    v: *const f64,
    jv: *mut f64,
    user_data: *mut c_void,
) -> c_int {
    unsafe {
        with_model_and_inputs(
            user_data,
            [y.cast(), v.cast(), jv.cast()],
            inputs,
            |model, inputs| {
                let n_states = model.n_states();
                model.jac_action(
                    t,
                    borrow_slice(y, n_states),
                    inputs,
                    borrow_slice(v, n_states),
                    borrow_slice_mut(jv, n_states),
                );
                SUCCESS
            },
        )
    }
}

/// Compute mass matrix action: M @ v
///
/// For identity mass matrix (ODE prototype), this copies v to mv.
///
/// # Safety
///
/// - All pointers must be valid and non-null
/// - `v` and `mv` must point to arrays of at least `model.n_states()` elements
/// - `user_data` must point to a valid `ModelEvaluator` instance
///
/// # Returns
///
/// - `SUCCESS` (0) on success
/// - `ERROR_NULL_POINTER` (-1) if any pointer is null
/// - `ERROR_PANIC` (-2) if a Rust panic occurred
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pybamm_rust_mass_action(
    v: *const f64,
    mv: *mut f64,
    user_data: *mut c_void,
) -> c_int {
    unsafe {
        with_model(user_data, [v.cast(), mv.cast()], |model| {
            let n_states = model.n_states();
            model.mass_action(borrow_slice(v, n_states), borrow_slice_mut(mv, n_states));
            SUCCESS
        })
    }
}

/// Copy the CSC column pointers to a pre-allocated buffer.
///
/// The buffer must have length `n_states + 1`.
///
/// # Safety
///
/// - `colptr` must point to an array of at least `model.n_states() + 1` elements
/// - `user_data` must point to a valid `ModelEvaluator` instance
///
/// # Returns
///
/// - `SUCCESS` (0) on success
/// - `ERROR_NULL_POINTER` (-1) if any pointer is null
/// - `ERROR_PANIC` (-2) if a Rust panic occurred
#[unsafe(no_mangle)]
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub unsafe extern "C" fn pybamm_rust_jac_csc_colptr(
    colptr: *mut i64,
    user_data: *const c_void,
) -> c_int {
    unsafe {
        with_model(user_data, [colptr.cast()], |model| {
            let csc = model.csc_sparsity();
            let out = borrow_slice_mut(colptr, csc.ncols + 1);
            for (i, &val) in csc.colptr.iter().enumerate() {
                out[i] = val as i64;
            }
            SUCCESS
        })
    }
}

/// Copy the CSC row indices to a pre-allocated buffer.
///
/// The buffer must have length `nnz`.
///
/// # Safety
///
/// - `rowind` must point to an array of at least `model.nnz()` elements
/// - `user_data` must point to a valid `ModelEvaluator` instance
///
/// # Returns
///
/// - `SUCCESS` (0) on success
/// - `ERROR_NULL_POINTER` (-1) if any pointer is null
/// - `ERROR_PANIC` (-2) if a Rust panic occurred
#[unsafe(no_mangle)]
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub unsafe extern "C" fn pybamm_rust_jac_csc_rowind(
    rowind: *mut i64,
    user_data: *const c_void,
) -> c_int {
    unsafe {
        with_model(user_data, [rowind.cast()], |model| {
            let csc = model.csc_sparsity();
            let out = borrow_slice_mut(rowind, csc.nnz());
            for (i, &val) in csc.rowind.iter().enumerate() {
                out[i] = val as i64;
            }
            SUCCESS
        })
    }
}

/// Get the number of forward-sensitivity parameters configured on the model.
///
/// # Safety
///
/// - `user_data` must point to a valid `ModelEvaluator` instance
///
/// # Returns
///
/// - Number of sensitivity parameters on success (>= 0)
/// - `ERROR_NULL_POINTER` (-1) if `user_data` is null
/// - `ERROR_PANIC` (-2) if a Rust panic occurred
#[unsafe(no_mangle)]
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub unsafe extern "C" fn pybamm_rust_n_sens_params(user_data: *const c_void) -> c_int {
    unsafe { with_model(user_data, [], |model| model.n_sens_params() as c_int) }
}

/// Evaluate `df/dp_i` for a single sensitivity parameter into `df_dp`.
///
/// # Safety
///
/// - `y`, `df_dp`, `user_data` must be valid and non-null
/// - `y` must point to an array of at least `model.n_states()` elements
/// - `df_dp` must point to an array of at least `model.n_states()` elements
/// - `inputs` may be null when the model has no input parameters; otherwise it
///   must point to an array of at least `model.n_params()` elements
///
/// # Returns
///
/// - `SUCCESS` (0) on success
/// - `ERROR_NULL_POINTER` (-1) if any required pointer is null
/// - `ERROR_INVALID_PARAM` (-3) if `param_idx` is out of range
/// - `ERROR_NO_SENS` (-6) if the model has no sensitivities
/// - `ERROR_PANIC` (-2) if a Rust panic occurred
#[unsafe(no_mangle)]
#[allow(clippy::cast_sign_loss)] // param_idx >= 0 is checked before the cast
pub unsafe extern "C" fn pybamm_rust_sens_eval(
    t: f64,
    y: *const f64,
    inputs: *const f64,
    param_idx: c_int,
    df_dp: *mut f64,
    user_data: *mut c_void,
) -> c_int {
    unsafe {
        with_model_and_inputs(
            user_data,
            [y.cast(), df_dp.cast()],
            inputs,
            |model, inputs| {
                if model.n_sens_params() == 0 {
                    return ERROR_NO_SENS;
                }
                if param_idx < 0 || (param_idx as usize) >= model.n_sens_params() {
                    return ERROR_INVALID_PARAM;
                }
                let n_states = model.n_states();
                model.eval_sens(
                    t,
                    borrow_slice(y, n_states),
                    inputs,
                    param_idx as usize,
                    borrow_slice_mut(df_dp, n_states),
                );
                SUCCESS
            },
        )
    }
}

/// Evaluate `df/dp` for all configured sensitivity parameters at once.
///
/// Layout of `df_dp_out`: `df_dp_out[i*n_states + j] = ∂f_j/∂p_i`.
///
/// # Safety
///
/// Same buffer-size and pointer requirements as [`pybamm_rust_sens_eval`], with
/// `df_dp_out` length at least `n_sens_params * n_states`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pybamm_rust_sens_eval_all(
    t: f64,
    y: *const f64,
    inputs: *const f64,
    df_dp_out: *mut f64,
    user_data: *mut c_void,
) -> c_int {
    unsafe {
        with_model_and_inputs(
            user_data,
            [y.cast(), df_dp_out.cast()],
            inputs,
            |model, inputs| {
                if model.n_sens_params() == 0 {
                    return ERROR_NO_SENS;
                }
                let n_states = model.n_states();
                let n_sens = model.n_sens_params();
                model.eval_sens_all(
                    t,
                    borrow_slice(y, n_states),
                    inputs,
                    borrow_slice_mut(df_dp_out, n_sens * n_states),
                );
                SUCCESS
            },
        )
    }
}

/// Project state sensitivities onto output sensitivities for all outputs and
/// all configured sensitivity parameters.
///
/// `dvar/dp_k = dH/dp . e_k + dH/dy . y_sens_k`.
///
/// Layout of `y_sens`: `y_sens[k*n_states + j]` (len `n_sens_params * n_states`).
/// Layout of `out`: `out[k*n_out + o]` (len `n_sens_params * total_output_len`).
///
/// # Safety
///
/// - `y`, `y_sens`, `out`, `user_data` must be valid and non-null
/// - `y` must point to an array of at least `model.n_states()` elements
/// - `y_sens` must point to an array of at least `n_sens_params * n_states` elements
/// - `out` must point to an array of at least `n_sens_params * total_output_len` elements
/// - `inputs` may be null when the model has no input parameters; otherwise it
///   must point to an array of at least `model.n_params()` elements
///
/// # Returns
///
/// - `SUCCESS` (0) on success
/// - `ERROR_NULL_POINTER` (-1) if any required pointer is null
/// - `ERROR_NO_OUTPUTS` (-7) if the model has no output variables
/// - `ERROR_NO_SENS` (-6) if the model has no sensitivities
/// - `ERROR_PANIC` (-2) if a Rust panic occurred
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pybamm_rust_output_sens_project(
    t: f64,
    y: *const f64,
    inputs: *const f64,
    y_sens: *const f64,
    out: *mut f64,
    user_data: *mut c_void,
) -> c_int {
    unsafe {
        with_model_and_inputs(
            user_data,
            [y.cast(), y_sens.cast(), out.cast()],
            inputs,
            |model, inputs| {
                if model.n_outputs() == 0 {
                    return ERROR_NO_OUTPUTS;
                }
                if model.n_sens_params() == 0 {
                    return ERROR_NO_SENS;
                }
                let n_states = model.n_states();
                let (n_sens, n_out) = (model.n_sens_params(), model.total_output_len());
                model.output_sens_project(
                    t,
                    borrow_slice(y, n_states),
                    inputs,
                    borrow_slice(y_sens, n_sens * n_states),
                    borrow_slice_mut(out, n_sens * n_out),
                );
                SUCCESS
            },
        )
    }
}

/// Get the number of compiled output-variable expressions on the model.
///
/// # Safety
///
/// - `user_data` must point to a valid `ModelEvaluator` instance
///
/// # Returns
///
/// - Number of output variables on success (>= 0)
/// - `ERROR_NULL_POINTER` (-1) if `user_data` is null
/// - `ERROR_PANIC` (-2) if a Rust panic occurred
#[unsafe(no_mangle)]
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub unsafe extern "C" fn pybamm_rust_n_outputs(user_data: *const c_void) -> c_int {
    unsafe { with_model(user_data, [], |model| model.n_outputs() as c_int) }
}

/// Get the length of output variable `var_idx`.
///
/// # Safety
///
/// - `user_data` must point to a valid `ModelEvaluator` instance
///
/// # Returns
///
/// - Output length on success (>= 0)
/// - `ERROR_NULL_POINTER` (-1) if `user_data` is null
/// - `ERROR_INVALID_OUTPUT` (-4) if `var_idx` is out of range
/// - `ERROR_PANIC` (-2) if a Rust panic occurred
#[unsafe(no_mangle)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
pub unsafe extern "C" fn pybamm_rust_output_len(user_data: *const c_void, var_idx: c_int) -> c_int {
    unsafe {
        with_model(user_data, [], |model| {
            if var_idx < 0 || (var_idx as usize) >= model.n_outputs() {
                return ERROR_INVALID_OUTPUT;
            }
            model.output_len_at(var_idx as usize) as c_int
        })
    }
}

/// Evaluate output variable `var_idx` into `out`.
///
/// # Safety
///
/// - `y`, `out`, `user_data` must be valid and non-null
/// - `y` must point to an array of at least `model.n_states()` elements
/// - `out` must point to an array of at least `output_len_at(var_idx)` elements
/// - `inputs` may be null when the model has no input parameters; otherwise it
///   must point to an array of at least `model.n_params()` elements
/// - `out_len` may be null; if non-null it receives the count of values written
///
/// # Returns
///
/// - `SUCCESS` (0) on success
/// - `ERROR_NULL_POINTER` (-1) if any required pointer is null
/// - `ERROR_INVALID_OUTPUT` (-4) if `var_idx` is out of range
/// - `ERROR_NO_OUTPUTS` (-7) if the model has no output variables
/// - `ERROR_PANIC` (-2) if a Rust panic occurred
#[unsafe(no_mangle)]
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
pub unsafe extern "C" fn pybamm_rust_output_eval(
    t: f64,
    y: *const f64,
    inputs: *const f64,
    var_idx: c_int,
    out: *mut f64,
    out_len: *mut c_int,
    user_data: *mut c_void,
) -> c_int {
    unsafe {
        with_model_and_inputs(
            user_data,
            [y.cast(), out.cast()],
            inputs,
            |model, inputs| {
                if model.n_outputs() == 0 {
                    return ERROR_NO_OUTPUTS;
                }
                if var_idx < 0 || (var_idx as usize) >= model.n_outputs() {
                    return ERROR_INVALID_OUTPUT;
                }
                let var = var_idx as usize;
                let n_states = model.n_states();
                let len = model.output_len_at(var);
                let written = model.eval_output(
                    t,
                    borrow_slice(y, n_states),
                    inputs,
                    var,
                    borrow_slice_mut(out, len),
                );
                if !out_len.is_null() {
                    *out_len = written as c_int;
                }
                SUCCESS
            },
        )
    }
}

/// Batch-evaluate every output variable over `n_points` trajectory points.
///
/// Amortises interpreter dispatch across the batch instead of walking every
/// output tape once per point; results match per-point evaluation bitwise.
///
/// # Safety
///
/// - `ts`, `ys`, `out`, `user_data` must be valid and non-null
/// - `ts` must point to `n_points` times
/// - `ys` must point to `n_points * n_states` states, each point contiguous
///   (`(n_states, n_points)` F-contiguous)
/// - `out` must point to `n_points * total_output_len` elements and is written
///   with each point's stacked outputs contiguous
/// - `inputs` may be null when the model has no input parameters; otherwise it
///   must point to an array of at least `model.n_params()` elements
///
/// # Returns
///
/// - `SUCCESS` (0) on success
/// - `ERROR_NULL_POINTER` (-1) if any required pointer is null
/// - `ERROR_INVALID_PARAM` (-3) if `n_points` is not positive
/// - `ERROR_NO_OUTPUTS` (-7) if the model has no output variables
/// - `ERROR_PANIC` (-2) if a Rust panic occurred
#[unsafe(no_mangle)]
#[allow(clippy::cast_sign_loss)]
pub unsafe extern "C" fn pybamm_rust_output_eval_batch(
    ts: *const f64,
    ys: *const f64,
    n_points: c_int,
    inputs: *const f64,
    out: *mut f64,
    user_data: *mut c_void,
) -> c_int {
    unsafe {
        with_model_and_inputs(
            user_data,
            [ts.cast(), ys.cast(), out.cast()],
            inputs,
            |model, inputs| {
                if n_points <= 0 {
                    return ERROR_INVALID_PARAM;
                }
                if model.n_outputs() == 0 {
                    return ERROR_NO_OUTPUTS;
                }
                let points = n_points as usize;
                let n_states = model.n_states();
                let total = model.total_output_len();
                model.eval_outputs_batch(
                    points,
                    borrow_slice(ts, points),
                    borrow_slice(ys, points * n_states),
                    inputs,
                    borrow_slice_mut(out, points * total),
                );
                SUCCESS
            },
        )
    }
}

/// Evaluate algebraic residual g(t, y).
///
/// # Safety
///
/// - `y`, `output`, and `user_data` must be valid and non-null
/// - `y` must point to an array of at least `model.n_states()` elements
/// - `output` must point to an array of at least `model.n_algebraic()` elements
/// - `inputs` may be null when the model has no input parameters; otherwise it
///   must point to an array of at least `model.n_params()` elements
/// - `user_data` must point to a valid `ModelEvaluator` instance
///
/// # Returns
///
/// - `SUCCESS` (0) on success
/// - `ERROR_NULL_POINTER` (-1) if any required pointer is null
/// - `ERROR_NO_ALG` (-8) if the model has no algebraic block
/// - `ERROR_PANIC` (-2) if a Rust panic occurred
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pybamm_rust_alg_res(
    t: f64,
    y: *const f64,
    inputs: *const f64,
    output: *mut f64,
    user_data: *mut c_void,
) -> c_int {
    unsafe {
        with_model_and_inputs(
            user_data,
            [y.cast(), output.cast()],
            inputs,
            |model, inputs| {
                if !model.has_algebraic() {
                    return ERROR_NO_ALG;
                }
                let n_states = model.n_states();
                let n_algebraic = model.n_algebraic();
                model.eval_algebraic_residual(
                    t,
                    borrow_slice(y, n_states),
                    inputs,
                    borrow_slice_mut(output, n_algebraic),
                );
                SUCCESS
            },
        )
    }
}

/// Assemble the algebraic Jacobian `dg/dy_alg`.
///
/// # Safety
///
/// - `y`, `output`, and `user_data` must be valid and non-null
/// - `y` must point to an array of at least `model.n_states()` elements
/// - `output` must point to an array of at least `model.algebraic_jacobian_nnz()` elements
/// - `inputs` may be null when the model has no input parameters; otherwise it
///   must point to an array of at least `model.n_params()` elements
/// - `user_data` must point to a valid `ModelEvaluator` instance
///
/// # Returns
///
/// - `SUCCESS` (0) on success
/// - `ERROR_NULL_POINTER` (-1) if any required pointer is null
/// - `ERROR_NO_ALG` (-8) if the model has no algebraic block
/// - `ERROR_PANIC` (-2) if a Rust panic occurred
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pybamm_rust_alg_jac_assemble(
    t: f64,
    y: *const f64,
    inputs: *const f64,
    output: *mut f64,
    user_data: *mut c_void,
) -> c_int {
    unsafe {
        with_model_and_inputs(
            user_data,
            [y.cast(), output.cast()],
            inputs,
            |model, inputs| {
                if !model.has_algebraic() {
                    return ERROR_NO_ALG;
                }
                let n_states = model.n_states();
                let n_alg_jac = model.algebraic_jacobian_nnz();
                model.assemble_algebraic_jacobian_into(
                    t,
                    borrow_slice(y, n_states),
                    inputs,
                    borrow_slice_mut(output, n_alg_jac),
                );
                SUCCESS
            },
        )
    }
}

/// Compute algebraic Jacobian-vector product `(dg/dy_alg) @ v`.
///
/// # Safety
///
/// - `y`, `v`, `jv`, and `user_data` must be valid and non-null
/// - `y` must point to an array of at least `model.n_states()` elements
/// - `v` and `jv` must point to arrays of at least `model.n_algebraic()` elements
/// - `inputs` may be null when the model has no input parameters; otherwise it
///   must point to an array of at least `model.n_params()` elements
/// - `user_data` must point to a valid `ModelEvaluator` instance
///
/// # Returns
///
/// - `SUCCESS` (0) on success
/// - `ERROR_NULL_POINTER` (-1) if any required pointer is null
/// - `ERROR_NO_ALG` (-8) if the model has no algebraic block
/// - `ERROR_PANIC` (-2) if a Rust panic occurred
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pybamm_rust_alg_jac_action(
    t: f64,
    y: *const f64,
    inputs: *const f64,
    v: *const f64,
    jv: *mut f64,
    user_data: *mut c_void,
) -> c_int {
    unsafe {
        with_model_and_inputs(
            user_data,
            [y.cast(), v.cast(), jv.cast()],
            inputs,
            |model, inputs| {
                if !model.has_algebraic() {
                    return ERROR_NO_ALG;
                }
                let n_states = model.n_states();
                let n_algebraic = model.n_algebraic();
                model.eval_algebraic_jacobian_action(
                    t,
                    borrow_slice(y, n_states),
                    inputs,
                    borrow_slice(v, n_algebraic),
                    borrow_slice_mut(jv, n_algebraic),
                );
                SUCCESS
            },
        )
    }
}

/// Get the number of compiled event expressions on the model.
///
/// # Safety
///
/// - `user_data` must point to a valid `ModelEvaluator` instance
///
/// # Returns
///
/// - Number of events on success (>= 0)
/// - `ERROR_NULL_POINTER` (-1) if `user_data` is null
/// - `ERROR_PANIC` (-2) if a Rust panic occurred
#[unsafe(no_mangle)]
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub unsafe extern "C" fn pybamm_rust_n_events(user_data: *const c_void) -> c_int {
    unsafe { with_model(user_data, [], |model| model.n_events() as c_int) }
}

/// Get the total length of all events concatenated.
///
/// # Safety
///
/// - `user_data` must point to a valid `ModelEvaluator` instance
///
/// # Returns
///
/// - Total event length on success (>= 0)
/// - `ERROR_NULL_POINTER` (-1) if `user_data` is null
/// - `ERROR_PANIC` (-2) if a Rust panic occurred
#[unsafe(no_mangle)]
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub unsafe extern "C" fn pybamm_rust_total_event_len(user_data: *const c_void) -> c_int {
    unsafe { with_model(user_data, [], |model| model.total_event_len() as c_int) }
}

/// Evaluate all events and write concatenated results into `output`.
///
/// This is the primary interface for DAE solvers that need all event values
/// at once for root-finding. Events are evaluated in order and their results
/// are concatenated into the output buffer.
///
/// # Safety
///
/// - `y`, `output`, and `user_data` must be valid and non-null
/// - `y` must point to an array of at least `model.n_states()` elements
/// - `output` must point to an array of at least `model.total_event_len()` elements
/// - `inputs` may be null when the model has no input parameters; otherwise it
///   must point to an array of at least `model.n_params()` elements
/// - `user_data` must point to a valid `ModelEvaluator` instance
///
/// # Returns
///
/// - `SUCCESS` (0) on success
/// - `ERROR_NULL_POINTER` (-1) if any required pointer is null
/// - `ERROR_NO_EVENTS` (-9) if the model has no events
/// - `ERROR_PANIC` (-2) if a Rust panic occurred
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pybamm_rust_events_eval(
    t: f64,
    y: *const f64,
    inputs: *const f64,
    output: *mut f64,
    user_data: *mut c_void,
) -> c_int {
    unsafe {
        with_model_and_inputs(
            user_data,
            [y.cast(), output.cast()],
            inputs,
            |model, inputs| {
                if model.n_events() == 0 {
                    return ERROR_NO_EVENTS;
                }
                let n_states = model.n_states();
                let total_len = model.total_event_len();
                model.eval_observables(
                    ObservableKind::Events,
                    t,
                    borrow_slice(y, n_states),
                    inputs,
                    borrow_slice_mut(output, total_len),
                );
                SUCCESS
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;
    use crate::node::{CsrData, Node, Shape};

    #[test]
    fn abi_version_is_stable_and_positive() {
        // Bumping the const is deliberate, paired with a PYBAMM_RUST_ABI_VERSION
        // bump in the C++ consumer.
        assert_eq!(pybamm_rust_abi_version(), RUST_ABI_VERSION);
        assert_eq!(RUST_ABI_VERSION, 1);
    }

    /// Create an identity mass matrix of size n.
    fn identity_mass_matrix(n: usize) -> CsrData {
        CsrData {
            shape: Shape::matrix(n, n),
            indptr: (0..=n).collect(),
            indices: (0..n).collect(),
            data: vec![1.0; n],
        }
    }

    /// Helper to create a simple test model: f(y) = 2*y with identity mass matrix
    fn create_test_model() -> Box<ModelEvaluator> {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let two = arena.alloc(Node::Scalar(2.0));
        let rhs = arena.alloc(Node::Mul(two, y)); // f(y) = 2*y

        let mass = identity_mass_matrix(2);
        Box::new(ModelEvaluator::new(&arena, rhs, mass, 2, 0))
    }

    fn create_algebraic_test_model() -> Box<ModelEvaluator> {
        let mut arena = Arena::new();
        let y_full = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let y0 = arena.alloc(Node::Index {
            child: y_full,
            start: 0,
            end: 1,
        });
        let y1 = arena.alloc(Node::Index {
            child: y_full,
            start: 1,
            end: 2,
        });
        let y2 = arena.alloc(Node::Index {
            child: y_full,
            start: 2,
            end: 3,
        });
        let rhs = y_full;
        let alg0_mul = arena.alloc(Node::Mul(y1, y2));
        let alg0 = arena.alloc(Node::Add(y0, alg0_mul));
        let two = arena.alloc(Node::Scalar(2.0));
        let y2_sq = arena.alloc(Node::Pow(y2, two));
        let alg1 = arena.alloc(Node::Add(y1, y2_sq));
        let alg = arena.alloc(Node::Concat(vec![alg0, alg1]));
        let mass = CsrData {
            shape: Shape::matrix(3, 3),
            indptr: vec![0, 1, 1, 1],
            indices: vec![0],
            data: vec![1.0],
        };

        Box::new(ModelEvaluator::new_with_algebraic(
            &arena,
            rhs,
            mass,
            3,
            0,
            Some(alg),
            &[1, 2],
        ))
    }

    #[test]
    fn test_rust_residual_null_check() {
        // All nulls
        let result = unsafe {
            pybamm_rust_residual(
                0.0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null::<f64>(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(result, ERROR_NULL_POINTER);

        // y is null
        let yp = [1.0, 2.0];
        let mut r = [0.0, 0.0];
        let result = unsafe {
            pybamm_rust_residual(
                0.0,
                std::ptr::null(),
                yp.as_ptr(),
                std::ptr::null::<f64>(),
                r.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(result, ERROR_NULL_POINTER);

        // user_data is null (but other pointers valid)
        let y = [1.0, 2.0];
        let result = unsafe {
            pybamm_rust_residual(
                0.0,
                y.as_ptr(),
                yp.as_ptr(),
                std::ptr::null::<f64>(),
                r.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(result, ERROR_NULL_POINTER);
    }

    #[test]
    fn test_rust_residual_success() {
        let mut model = create_test_model();

        // f(y) = 2*y, M = I: residual = M*y' - f(y) = [3, 4] - [2, 4] = [1, 0]
        let y = [1.0, 2.0];
        let yp = [3.0, 4.0];
        let mut r = [0.0, 0.0];

        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let result = unsafe {
            pybamm_rust_residual(
                0.0,
                y.as_ptr(),
                yp.as_ptr(),
                std::ptr::null::<f64>(),
                r.as_mut_ptr(),
                user_data,
            )
        };

        assert_eq!(result, SUCCESS);
        assert!(
            (r[0] - 1.0).abs() < 1e-14,
            "Expected r[0] = 1.0, got {}",
            r[0]
        );
        assert!(r[1].abs() < 1e-14, "Expected r[1] = 0.0, got {}", r[1]);
    }

    #[test]
    fn test_rust_jac_mul_null_check() {
        let result = unsafe {
            pybamm_rust_jac_mul(
                0.0,
                std::ptr::null(),
                std::ptr::null::<f64>(),
                1.0,
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(result, ERROR_NULL_POINTER);
    }

    #[test]
    fn test_rust_jac_mul_success() {
        let mut model = create_test_model();

        // f(y) = 2*y, so df/dy = 2*I
        // With cj=0.5 and M=I: (df/dy - cj*M) @ v = (2*I - 0.5*I) @ v = 1.5*v
        let y = [1.0, 2.0];
        let v = [1.0, 0.0];
        let mut jv = [0.0, 0.0];

        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let result = unsafe {
            pybamm_rust_jac_mul(
                0.0,
                y.as_ptr(),
                std::ptr::null::<f64>(),
                0.5,
                v.as_ptr(),
                jv.as_mut_ptr(),
                user_data,
            )
        };

        assert_eq!(result, SUCCESS);
        // (2 - 0.5) * [1, 0] = [1.5, 0]
        assert!(
            (jv[0] - 1.5).abs() < 1e-14,
            "Expected jv[0] = 1.5, got {}",
            jv[0]
        );
        assert!(jv[1].abs() < 1e-14, "Expected jv[1] = 0.0, got {}", jv[1]);
    }

    #[test]
    fn test_rust_jac_mul_uses_inputs() {
        // f(y)=k*y (k an input); (df/dy - cj*M)@v = (k-cj)*v. k=7, cj=0.5, v=[1,0] -> [6.5, 0].
        // If inputs were dropped (k read as 0), the result would instead be [-0.5, 0].
        let mut model = create_sens_test_model();
        let y = [3.0, 4.0];
        let inputs = [7.0];
        let v = [1.0, 0.0];
        let mut jv = [0.0, 0.0];

        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let result = unsafe {
            pybamm_rust_jac_mul(
                0.0,
                y.as_ptr(),
                inputs.as_ptr(),
                0.5,
                v.as_ptr(),
                jv.as_mut_ptr(),
                user_data,
            )
        };

        assert_eq!(result, SUCCESS);
        assert!(
            (jv[0] - 6.5).abs() < 1e-12,
            "expected jv[0]=6.5, got {}",
            jv[0]
        );
        assert!(jv[1].abs() < 1e-12, "expected jv[1]=0.0, got {}", jv[1]);
    }

    #[test]
    fn test_rust_eval_rhs_null_check() {
        let result = unsafe {
            pybamm_rust_eval_rhs(
                0.0,
                std::ptr::null(),
                std::ptr::null::<f64>(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(result, ERROR_NULL_POINTER);
    }

    #[test]
    fn test_rust_eval_rhs_success() {
        let mut model = create_test_model();

        // f(y) = 2*y
        // y = [1, 2] -> f(y) = [2, 4]
        let y = [1.0, 2.0];
        let mut f_out = [0.0, 0.0];

        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let result = unsafe {
            pybamm_rust_eval_rhs(
                0.0,
                y.as_ptr(),
                std::ptr::null::<f64>(),
                f_out.as_mut_ptr(),
                user_data,
            )
        };

        assert_eq!(result, SUCCESS);
        assert!(
            (f_out[0] - 2.0).abs() < 1e-14,
            "Expected f_out[0] = 2.0, got {}",
            f_out[0]
        );
        assert!(
            (f_out[1] - 4.0).abs() < 1e-14,
            "Expected f_out[1] = 4.0, got {}",
            f_out[1]
        );
    }

    #[test]
    fn test_rust_alg_res_no_algebraic_block() {
        let mut model = create_test_model();
        let y = [1.0, 2.0];
        let mut out = [0.0, 0.0];
        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let result = unsafe {
            pybamm_rust_alg_res(
                0.0,
                y.as_ptr(),
                std::ptr::null::<f64>(),
                out.as_mut_ptr(),
                user_data,
            )
        };
        assert_eq!(result, ERROR_NO_ALG);
    }

    #[test]
    fn test_rust_alg_res_success() {
        let mut model = create_algebraic_test_model();
        let y = [10.0, 2.0, 3.0];
        let mut out = [0.0, 0.0];
        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let result = unsafe {
            pybamm_rust_alg_res(
                0.0,
                y.as_ptr(),
                std::ptr::null::<f64>(),
                out.as_mut_ptr(),
                user_data,
            )
        };

        assert_eq!(result, SUCCESS);
        assert!((out[0] - 16.0).abs() < 1e-12, "expected 16, got {}", out[0]);
        assert!((out[1] - 11.0).abs() < 1e-12, "expected 11, got {}", out[1]);
    }

    #[test]
    fn test_rust_alg_jac_assemble_success() {
        let mut model = create_algebraic_test_model();
        let y = [10.0, 2.0, 3.0];
        let mut jac = vec![0.0; model.algebraic_jacobian_nnz()];
        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let result = unsafe {
            pybamm_rust_alg_jac_assemble(
                0.0,
                y.as_ptr(),
                std::ptr::null::<f64>(),
                jac.as_mut_ptr(),
                user_data,
            )
        };

        assert_eq!(result, SUCCESS);
        // CSC order, matching the (rows, cols) the model publishes beside it.
        assert_eq!(jac, vec![3.0, 1.0, 2.0, 6.0]);
    }

    #[test]
    fn test_rust_alg_jac_action_success() {
        let mut model = create_algebraic_test_model();
        let y = [10.0, 2.0, 3.0];
        let v = [5.0, 7.0];
        let mut jv = [0.0, 0.0];
        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let result = unsafe {
            pybamm_rust_alg_jac_action(
                0.0,
                y.as_ptr(),
                std::ptr::null::<f64>(),
                v.as_ptr(),
                jv.as_mut_ptr(),
                user_data,
            )
        };

        assert_eq!(result, SUCCESS);
        assert!((jv[0] - 29.0).abs() < 1e-12, "expected 29, got {}", jv[0]);
        assert!((jv[1] - 47.0).abs() < 1e-12, "expected 47, got {}", jv[1]);
    }

    #[test]
    fn test_rust_n_states_null_check() {
        let result = unsafe { pybamm_rust_n_states(std::ptr::null()) };
        assert_eq!(result, ERROR_NULL_POINTER);
    }

    #[test]
    fn test_rust_n_states_success() {
        let model = create_test_model();
        let user_data: *const c_void = std::ptr::from_ref(model.as_ref()).cast();
        let result = unsafe { pybamm_rust_n_states(user_data) };
        assert_eq!(result, 2);
    }

    #[test]
    fn test_rust_jac_assemble_null_check() {
        let result = unsafe {
            pybamm_rust_jac_assemble(
                0.0,
                std::ptr::null(),
                std::ptr::null::<f64>(),
                1.0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(result, ERROR_NULL_POINTER);
    }

    #[test]
    fn test_rust_jac_assemble_success() {
        let mut model = create_test_model();

        // f(y) = 2*y, df/dy = 2*I, M = I
        // With cj=0.5: J = df/dy - cj*M = 2*I - 0.5*I = 1.5*I
        let y = [1.0, 2.0];
        let nnz = model.nnz();
        let mut jac_data = vec![0.0; nnz];

        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let result = unsafe {
            pybamm_rust_jac_assemble(
                0.0,
                y.as_ptr(),
                std::ptr::null::<f64>(),
                0.5,
                jac_data.as_mut_ptr(),
                user_data,
            )
        };

        assert_eq!(result, SUCCESS);
        // Diagonal entries should be 1.5
        for &val in &jac_data {
            assert!((val - 1.5).abs() < 1e-14, "Expected 1.5, got {val}");
        }
    }

    #[test]
    fn test_rust_jac_nnz_null_check() {
        let result = unsafe { pybamm_rust_jac_nnz(std::ptr::null()) };
        assert_eq!(result, ERROR_NULL_POINTER);
    }

    #[test]
    fn test_rust_jac_nnz_success() {
        let model = create_test_model();
        let user_data: *const c_void = std::ptr::from_ref(model.as_ref()).cast();
        let result = unsafe { pybamm_rust_jac_nnz(user_data) };
        // f(y) = 2*y has diagonal Jacobian, so nnz = n_states = 2
        assert_eq!(result, 2);
    }

    #[test]
    fn test_rust_jac_csc_colptr_success() {
        let model = create_test_model();
        let user_data: *const c_void = std::ptr::from_ref(model.as_ref()).cast();

        // For 2x2 diagonal matrix, colptr should be [0, 1, 2]
        let mut colptr = [0i64; 3];
        let result = unsafe { pybamm_rust_jac_csc_colptr(colptr.as_mut_ptr(), user_data) };

        assert_eq!(result, SUCCESS);
        assert_eq!(colptr, [0, 1, 2]);
    }

    #[test]
    fn test_rust_jac_csc_rowind_success() {
        let model = create_test_model();
        let user_data: *const c_void = std::ptr::from_ref(model.as_ref()).cast();

        // For 2x2 diagonal matrix, rowind should be [0, 1]
        let mut rowind = [0i64; 2];
        let result = unsafe { pybamm_rust_jac_csc_rowind(rowind.as_mut_ptr(), user_data) };

        assert_eq!(result, SUCCESS);
        assert_eq!(rowind, [0, 1]);
    }

    #[test]
    fn test_rust_jac_action_null_check() {
        let result = unsafe {
            pybamm_rust_jac_action(
                0.0,
                std::ptr::null(),
                std::ptr::null::<f64>(),
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(result, ERROR_NULL_POINTER);
    }

    #[test]
    fn test_rust_jac_action_success() {
        let mut model = create_test_model();

        // f(y) = 2*y, so df/dy = 2*I
        // jac_action(v) = df/dy @ v = 2*v (no mass term)
        let y = [1.0, 2.0];
        let v = [1.0, 0.0];
        let mut jv = [0.0, 0.0];

        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let result = unsafe {
            pybamm_rust_jac_action(
                0.0,
                y.as_ptr(),
                std::ptr::null::<f64>(),
                v.as_ptr(),
                jv.as_mut_ptr(),
                user_data,
            )
        };

        assert_eq!(result, SUCCESS);
        assert!(
            (jv[0] - 2.0).abs() < 1e-14,
            "Expected jv[0] = 2.0, got {}",
            jv[0]
        );
        assert!(jv[1].abs() < 1e-14, "Expected jv[1] = 0.0, got {}", jv[1]);
    }

    #[test]
    fn test_rust_mass_action_null_check() {
        let result = unsafe {
            pybamm_rust_mass_action(std::ptr::null(), std::ptr::null_mut(), std::ptr::null_mut())
        };
        assert_eq!(result, ERROR_NULL_POINTER);
    }

    #[test]
    fn test_rust_mass_action_success() {
        let model = create_test_model();

        // Identity mass: M @ v = v
        let v = [3.0, 4.0];
        let mut mv = [0.0, 0.0];

        let user_data: *const c_void = std::ptr::from_ref(model.as_ref()).cast();
        let result =
            unsafe { pybamm_rust_mass_action(v.as_ptr(), mv.as_mut_ptr(), user_data.cast_mut()) };

        assert_eq!(result, SUCCESS);
        // Use approx comparison for floats instead of assert_eq!
        assert!(
            (mv[0] - v[0]).abs() < 1e-14,
            "Expected mv[0] = {}, got {}",
            v[0],
            mv[0]
        );
        assert!(
            (mv[1] - v[1]).abs() < 1e-14,
            "Expected mv[1] = {}, got {}",
            v[1],
            mv[1]
        );
    }

    #[test]
    fn test_rust_jac_action_preserves_cj() {
        // Verify that jac_action doesn't permanently change cj
        let mut model = create_test_model();

        // Set a non-zero cj
        model.set_cj(0.5);

        let y = [1.0, 2.0];
        let v = [1.0, 0.0];
        let mut jv = [0.0, 0.0];

        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let result = unsafe {
            pybamm_rust_jac_action(
                0.0,
                y.as_ptr(),
                std::ptr::null::<f64>(),
                v.as_ptr(),
                jv.as_mut_ptr(),
                user_data,
            )
        };

        assert_eq!(result, SUCCESS);
        // jac_action should still give df/dy @ v = 2*v (no mass)
        assert!(
            (jv[0] - 2.0).abs() < 1e-14,
            "Expected jv[0] = 2.0, got {}",
            jv[0]
        );

        // cj should be restored to 0.5
        assert!(
            (model.cj() - 0.5).abs() < 1e-14,
            "Expected cj = 0.5, got {}",
            model.cj()
        );
    }

    #[test]
    fn test_rust_eval_rhs_with_inputs() {
        // f(y) = k*y where k is an input parameter at index 0
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let k = arena.alloc(Node::InputParameter {
            name: "k".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let rhs = arena.alloc(Node::Mul(k, y));
        let mass = identity_mass_matrix(2);
        let mut model = Box::new(ModelEvaluator::new(&arena, rhs, mass, 2, 1));

        // y = [1, 2], k = 3 -> f = [3, 6]
        let y = [1.0, 2.0];
        let inputs = [3.0];
        let mut f_out = [0.0, 0.0];

        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let result = unsafe {
            pybamm_rust_eval_rhs(
                0.0,
                y.as_ptr(),
                inputs.as_ptr(),
                f_out.as_mut_ptr(),
                user_data,
            )
        };

        assert_eq!(result, SUCCESS);
        assert!((f_out[0] - 3.0).abs() < 1e-14);
        assert!((f_out[1] - 6.0).abs() < 1e-14);
    }

    #[test]
    fn test_rust_eval_rhs_null_inputs_when_no_params() {
        // When n_inputs == 0, inputs may be null
        let mut model = create_test_model();
        let y = [1.0, 2.0];
        let mut f_out = [0.0, 0.0];

        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let result = unsafe {
            pybamm_rust_eval_rhs(
                0.0,
                y.as_ptr(),
                std::ptr::null(),
                f_out.as_mut_ptr(),
                user_data,
            )
        };
        assert_eq!(result, SUCCESS);
        assert!((f_out[0] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_rust_n_inputs() {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let k = arena.alloc(Node::InputParameter {
            name: "k".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let rhs = arena.alloc(Node::Mul(k, y));
        let mass = identity_mass_matrix(2);
        let model = Box::new(ModelEvaluator::new(&arena, rhs, mass, 2, 1));
        let user_data: *const c_void = std::ptr::from_ref(model.as_ref()).cast();
        let result = unsafe { pybamm_rust_n_inputs(user_data) };
        assert_eq!(result, 1);
    }

    #[test]
    fn test_rust_algebraic_ids_dae() {
        // Mass with row 1 missing diagonal -> row 1 is algebraic.
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 3 });
        let mass = CsrData {
            shape: Shape::matrix(3, 3),
            indptr: vec![0, 1, 1, 2],
            indices: vec![0, 2],
            data: vec![1.0, 1.0],
        };
        let mut model = Box::new(ModelEvaluator::new(&arena, y, mass, 3, 0));
        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let mut ids = [0.0f64; 3];
        let result = unsafe { pybamm_rust_algebraic_ids(ids.as_mut_ptr(), user_data) };
        assert_eq!(result, SUCCESS);
        // 1.0 = differential, 0.0 = algebraic; values are exact (no FP arithmetic).
        assert!((ids[0] - 1.0).abs() < f64::EPSILON);
        assert!(ids[1].abs() < f64::EPSILON);
        assert!((ids[2] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rust_algebraic_ids_null_check() {
        let result =
            unsafe { pybamm_rust_algebraic_ids(std::ptr::null_mut(), std::ptr::null_mut()) };
        assert_eq!(result, ERROR_NULL_POINTER);
    }

    fn create_sens_test_model() -> Box<ModelEvaluator> {
        // f(y) = k * y with k as InputParameter index 0; sensitivity w.r.t. k.
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let k = arena.alloc(Node::InputParameter {
            name: "k".to_string(),
            index: 0,
            offset: 0,
            width: 1,
        });
        let rhs = arena.alloc(Node::Mul(k, y));
        let mass = identity_mass_matrix(2);
        Box::new(ModelEvaluator::new_with_sens(&arena, rhs, mass, 2, 1, &[0]))
    }

    #[test]
    fn test_rust_n_sens_params() {
        let model = create_sens_test_model();
        let user_data: *const c_void = std::ptr::from_ref(model.as_ref()).cast();
        assert_eq!(unsafe { pybamm_rust_n_sens_params(user_data) }, 1);
    }

    #[test]
    fn test_rust_sens_eval_basic() {
        // f(y) = k*y, ∂f/∂k = y. At y=[3, 4] -> [3, 4].
        let mut model = create_sens_test_model();
        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let y = [3.0, 4.0];
        let inputs = [7.0];
        let mut out = [0.0; 2];
        let result = unsafe {
            pybamm_rust_sens_eval(
                0.0,
                y.as_ptr(),
                inputs.as_ptr(),
                0,
                out.as_mut_ptr(),
                user_data,
            )
        };
        assert_eq!(result, SUCCESS);
        assert!((out[0] - 3.0).abs() < 1e-12);
        assert!((out[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_rust_sens_eval_no_sensitivities() {
        // Model has 0 sens params -> any param_idx returns ERROR_NO_SENS.
        let mut model = create_test_model();
        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let y = [1.0, 2.0];
        let mut out = [0.0; 2];
        let result = unsafe {
            pybamm_rust_sens_eval(
                0.0,
                y.as_ptr(),
                std::ptr::null(),
                0,
                out.as_mut_ptr(),
                user_data,
            )
        };
        assert_eq!(result, ERROR_NO_SENS);
    }

    #[test]
    fn test_rust_sens_eval_invalid_index() {
        let mut model = create_sens_test_model();
        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let y = [3.0, 4.0];
        let inputs = [7.0];
        let mut out = [0.0; 2];
        let result = unsafe {
            pybamm_rust_sens_eval(
                0.0,
                y.as_ptr(),
                inputs.as_ptr(),
                5, // out of range
                out.as_mut_ptr(),
                user_data,
            )
        };
        assert_eq!(result, ERROR_INVALID_PARAM);
    }

    #[test]
    fn test_rust_sens_eval_all_basic() {
        let mut model = create_sens_test_model();
        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let y = [3.0, 4.0];
        let inputs = [7.0];
        let mut out = [0.0; 2]; // n_sens_params=1, n_states=2 -> 2 entries
        let result = unsafe {
            pybamm_rust_sens_eval_all(
                0.0,
                y.as_ptr(),
                inputs.as_ptr(),
                out.as_mut_ptr(),
                user_data,
            )
        };
        assert_eq!(result, SUCCESS);
        assert!((out[0] - 3.0).abs() < 1e-12);
        assert!((out[1] - 4.0).abs() < 1e-12);
    }

    fn create_output_test_model() -> Box<ModelEvaluator> {
        // rhs: f(y) = y; output: 2 * y[0]
        let mut arena = Arena::new();
        let y_full = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let y0 = arena.alloc(Node::Index {
            child: y_full,
            start: 0,
            end: 1,
        });
        let two = arena.alloc(Node::Scalar(2.0));
        let var0 = arena.alloc(Node::Mul(two, y0));
        let mass = identity_mass_matrix(2);
        let mut model = ModelEvaluator::new(&arena, y_full, mass, 2, 0);
        model.add_output(&arena, var0);
        Box::new(model)
    }

    #[test]
    fn test_rust_n_outputs() {
        let model = create_output_test_model();
        let user_data: *const c_void = std::ptr::from_ref(model.as_ref()).cast();
        assert_eq!(unsafe { pybamm_rust_n_outputs(user_data) }, 1);
    }

    #[test]
    fn test_rust_output_len() {
        let model = create_output_test_model();
        let user_data: *const c_void = std::ptr::from_ref(model.as_ref()).cast();
        assert_eq!(unsafe { pybamm_rust_output_len(user_data, 0) }, 1);
        assert_eq!(
            unsafe { pybamm_rust_output_len(user_data, 5) },
            ERROR_INVALID_OUTPUT
        );
    }

    #[test]
    fn test_rust_output_eval_basic() {
        let mut model = create_output_test_model();
        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let y = [3.0, 4.0];
        let mut out = [0.0; 1];
        let mut out_len: c_int = 0;
        let result = unsafe {
            pybamm_rust_output_eval(
                0.0,
                y.as_ptr(),
                std::ptr::null(),
                0,
                out.as_mut_ptr(),
                &raw mut out_len,
                user_data,
            )
        };
        assert_eq!(result, SUCCESS);
        assert_eq!(out_len, 1);
        assert!((out[0] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_rust_output_eval_no_outputs() {
        let mut model = create_test_model(); // no outputs configured
        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let y = [1.0, 2.0];
        let mut out = [0.0];
        let mut out_len: c_int = 0;
        let result = unsafe {
            pybamm_rust_output_eval(
                0.0,
                y.as_ptr(),
                std::ptr::null(),
                0,
                out.as_mut_ptr(),
                &raw mut out_len,
                user_data,
            )
        };
        assert_eq!(result, ERROR_NO_OUTPUTS);
    }

    fn create_event_test_model() -> Box<ModelEvaluator> {
        // rhs: f(y) = y; event: y[0] - 0.5
        let mut arena = Arena::new();
        let y_full = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let y0 = arena.alloc(Node::Index {
            child: y_full,
            start: 0,
            end: 1,
        });
        let thresh = arena.alloc(Node::Scalar(0.5));
        let event_expr = arena.alloc(Node::Sub(y0, thresh));
        let mass = identity_mass_matrix(2);
        let mut model = ModelEvaluator::new(&arena, y_full, mass, 2, 0);
        model.add_event(&arena, event_expr);
        Box::new(model)
    }

    #[test]
    fn test_rust_n_events_null_check() {
        let result = unsafe { pybamm_rust_n_events(std::ptr::null()) };
        assert_eq!(result, ERROR_NULL_POINTER);
    }

    #[test]
    fn test_rust_n_events_success() {
        let model = create_event_test_model();
        let user_data: *const c_void = std::ptr::from_ref(model.as_ref()).cast();
        let result = unsafe { pybamm_rust_n_events(user_data) };
        assert_eq!(result, 1);
    }

    #[test]
    fn test_rust_n_events_zero_when_no_events() {
        let model = create_test_model(); // no events
        let user_data: *const c_void = std::ptr::from_ref(model.as_ref()).cast();
        let result = unsafe { pybamm_rust_n_events(user_data) };
        assert_eq!(result, 0);
    }

    #[test]
    fn test_rust_total_event_len_null_check() {
        let result = unsafe { pybamm_rust_total_event_len(std::ptr::null()) };
        assert_eq!(result, ERROR_NULL_POINTER);
    }

    #[test]
    fn test_rust_total_event_len_success() {
        let model = create_event_test_model();
        let user_data: *const c_void = std::ptr::from_ref(model.as_ref()).cast();
        let result = unsafe { pybamm_rust_total_event_len(user_data) };
        assert_eq!(result, 1);
    }

    #[test]
    fn test_rust_events_eval_null_check() {
        let result = unsafe {
            pybamm_rust_events_eval(
                0.0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(result, ERROR_NULL_POINTER);
    }

    #[test]
    fn test_rust_events_eval_no_events() {
        let mut model = create_test_model(); // no events
        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let y = [1.0, 2.0];
        let mut out = [0.0];
        let result = unsafe {
            pybamm_rust_events_eval(
                0.0,
                y.as_ptr(),
                std::ptr::null(),
                out.as_mut_ptr(),
                user_data,
            )
        };
        assert_eq!(result, ERROR_NO_EVENTS);
    }

    #[test]
    fn test_rust_events_eval_success() {
        let mut model = create_event_test_model();
        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();

        // event = y[0] - 0.5; y = [0.7, 1.0] => 0.7 - 0.5 = 0.2
        let y = [0.7, 1.0];
        let mut out = [0.0];
        let result = unsafe {
            pybamm_rust_events_eval(
                0.0,
                y.as_ptr(),
                std::ptr::null(),
                out.as_mut_ptr(),
                user_data,
            )
        };
        assert_eq!(result, SUCCESS);
        assert!((out[0] - 0.2).abs() < 1e-12, "expected 0.2, got {}", out[0]);
    }

    #[test]
    fn test_rust_events_eval_at_threshold() {
        let mut model = create_event_test_model();
        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();

        // event = y[0] - 0.5; y = [0.5, 1.0] => 0.0 (at threshold)
        let y = [0.5, 1.0];
        let mut out = [0.0];
        let result = unsafe {
            pybamm_rust_events_eval(
                0.0,
                y.as_ptr(),
                std::ptr::null(),
                out.as_mut_ptr(),
                user_data,
            )
        };
        assert_eq!(result, SUCCESS);
        assert!(out[0].abs() < 1e-12, "expected 0.0, got {}", out[0]);
    }

    #[test]
    fn guard_maps_a_panicking_body_to_error_panic() {
        let model = create_test_model();
        let shared: *const c_void = std::ptr::from_ref(model.as_ref()).cast();
        assert_eq!(
            unsafe { with_model(shared, [], |_| panic!("boom")) },
            ERROR_PANIC
        );

        let mut model = create_test_model();
        let owned: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        assert_eq!(
            unsafe { with_model_and_inputs(owned, [], std::ptr::null(), |_, _| panic!("boom")) },
            ERROR_PANIC
        );
    }

    #[test]
    fn guard_rejects_nulls_without_running_the_body() {
        let model = create_test_model();
        let shared: *const c_void = std::ptr::from_ref(model.as_ref()).cast();
        let unreachable = |_: &ModelEvaluator| unreachable!("guard let a null through");

        assert_eq!(
            unsafe { with_model(std::ptr::null(), [], unreachable) },
            ERROR_NULL_POINTER
        );
        assert_eq!(
            unsafe { with_model(shared, [std::ptr::null()], unreachable) },
            ERROR_NULL_POINTER
        );
        assert_eq!(
            unsafe { with_model(shared, [shared, std::ptr::null(), shared], unreachable) },
            ERROR_NULL_POINTER
        );
    }

    /// `inputs` obeys the same null rule as every other buffer, decided against
    /// the model rather than the signature: null is the caller saying "no
    /// inputs", which only a model without input parameters can be told.
    #[test]
    fn null_inputs_are_rejected_for_a_model_that_takes_parameters() {
        let mut model = create_sens_test_model(); // f(y) = k*y, one input
        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let y = [3.0, 4.0];
        let mut f_out = [0.0; 2];

        let mut call = |inputs| unsafe {
            pybamm_rust_eval_rhs(0.0, y.as_ptr(), inputs, f_out.as_mut_ptr(), user_data)
        };
        assert_eq!(call(std::ptr::null()), ERROR_NULL_POINTER);

        let inputs = [3.0];
        assert_eq!(call(inputs.as_ptr()), SUCCESS);
        assert!((f_out[0] - 9.0).abs() < 1e-12, "got {}", f_out[0]);
    }

    #[test]
    fn a_supplied_inputs_buffer_is_ignored_when_the_model_takes_none() {
        let mut model = create_test_model();
        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let y = [1.0, 2.0];
        let stray = [99.0];
        let mut f_out = [0.0; 2];

        let result = unsafe {
            pybamm_rust_eval_rhs(
                0.0,
                y.as_ptr(),
                stray.as_ptr(),
                f_out.as_mut_ptr(),
                user_data,
            )
        };
        assert_eq!(result, SUCCESS);
        assert!((f_out[0] - 2.0).abs() < 1e-14, "got {}", f_out[0]);
    }

    /// Null `user_data` is the one input every entry point shares, so a missing
    /// guard shows up here as a segfault or a stale value instead of the code.
    #[test]
    fn every_entry_point_rejects_a_null_user_data() {
        let nil = std::ptr::null::<f64>();
        let nil_mut = std::ptr::null_mut::<f64>();
        let nil_data = std::ptr::null_mut::<c_void>();
        let checks: [(&str, c_int); 26] = [
            ("pybamm_rust_residual", unsafe {
                pybamm_rust_residual(0.0, nil, nil, nil, nil_mut, nil_data)
            }),
            ("pybamm_rust_jac_mul", unsafe {
                pybamm_rust_jac_mul(0.0, nil, nil, 1.0, nil, nil_mut, nil_data)
            }),
            ("pybamm_rust_eval_rhs", unsafe {
                pybamm_rust_eval_rhs(0.0, nil, nil, nil_mut, nil_data)
            }),
            ("pybamm_rust_n_states", unsafe {
                pybamm_rust_n_states(std::ptr::null())
            }),
            ("pybamm_rust_n_inputs", unsafe {
                pybamm_rust_n_inputs(std::ptr::null())
            }),
            ("pybamm_rust_algebraic_ids", unsafe {
                pybamm_rust_algebraic_ids(nil_mut, nil_data)
            }),
            ("pybamm_rust_jac_assemble", unsafe {
                pybamm_rust_jac_assemble(0.0, nil, nil, 1.0, nil_mut, nil_data)
            }),
            ("pybamm_rust_jac_nnz", unsafe {
                pybamm_rust_jac_nnz(std::ptr::null())
            }),
            ("pybamm_rust_jac_action", unsafe {
                pybamm_rust_jac_action(0.0, nil, nil, nil, nil_mut, nil_data)
            }),
            ("pybamm_rust_mass_action", unsafe {
                pybamm_rust_mass_action(nil, nil_mut, nil_data)
            }),
            ("pybamm_rust_jac_csc_colptr", unsafe {
                pybamm_rust_jac_csc_colptr(std::ptr::null_mut(), std::ptr::null())
            }),
            ("pybamm_rust_jac_csc_rowind", unsafe {
                pybamm_rust_jac_csc_rowind(std::ptr::null_mut(), std::ptr::null())
            }),
            ("pybamm_rust_n_sens_params", unsafe {
                pybamm_rust_n_sens_params(std::ptr::null())
            }),
            ("pybamm_rust_sens_eval", unsafe {
                pybamm_rust_sens_eval(0.0, nil, nil, 0, nil_mut, nil_data)
            }),
            ("pybamm_rust_sens_eval_all", unsafe {
                pybamm_rust_sens_eval_all(0.0, nil, nil, nil_mut, nil_data)
            }),
            ("pybamm_rust_output_sens_project", unsafe {
                pybamm_rust_output_sens_project(0.0, nil, nil, nil, nil_mut, nil_data)
            }),
            ("pybamm_rust_n_outputs", unsafe {
                pybamm_rust_n_outputs(std::ptr::null())
            }),
            ("pybamm_rust_output_len", unsafe {
                pybamm_rust_output_len(std::ptr::null(), 0)
            }),
            ("pybamm_rust_output_eval", unsafe {
                pybamm_rust_output_eval(0.0, nil, nil, 0, nil_mut, std::ptr::null_mut(), nil_data)
            }),
            // Also pins the order: nulls outrank the `n_points > 0` rule.
            ("pybamm_rust_output_eval_batch", unsafe {
                pybamm_rust_output_eval_batch(nil, nil, 0, nil, nil_mut, nil_data)
            }),
            ("pybamm_rust_alg_res", unsafe {
                pybamm_rust_alg_res(0.0, nil, nil, nil_mut, nil_data)
            }),
            ("pybamm_rust_alg_jac_assemble", unsafe {
                pybamm_rust_alg_jac_assemble(0.0, nil, nil, nil_mut, nil_data)
            }),
            ("pybamm_rust_alg_jac_action", unsafe {
                pybamm_rust_alg_jac_action(0.0, nil, nil, nil, nil_mut, nil_data)
            }),
            ("pybamm_rust_n_events", unsafe {
                pybamm_rust_n_events(std::ptr::null())
            }),
            ("pybamm_rust_total_event_len", unsafe {
                pybamm_rust_total_event_len(std::ptr::null())
            }),
            ("pybamm_rust_events_eval", unsafe {
                pybamm_rust_events_eval(0.0, nil, nil, nil_mut, nil_data)
            }),
        ];
        for (name, status) in checks {
            assert_eq!(
                status, ERROR_NULL_POINTER,
                "{name} accepted a null user_data"
            );
        }
    }

    /// Every buffer an entry point dereferences must be named in its guard, and
    /// the model here has no sensitivities, outputs, events or algebraic block —
    /// so these also pin that a null outranks the model-state error codes.
    #[test]
    fn every_required_buffer_of_the_solver_callbacks_is_null_checked() {
        let mut model = create_test_model();
        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let mut scratch = [0.0f64; 4];
        let ok = scratch.as_mut_ptr();
        let ok_const: *const f64 = ok;
        let nil = std::ptr::null::<f64>();
        let nil_mut = std::ptr::null_mut::<f64>();

        let residual = |y, yp, r| unsafe { pybamm_rust_residual(0.0, y, yp, nil, r, user_data) };
        assert_eq!(residual(nil, ok_const, ok), ERROR_NULL_POINTER, "y");
        assert_eq!(residual(ok_const, nil, ok), ERROR_NULL_POINTER, "yp");
        assert_eq!(
            residual(ok_const, ok_const, nil_mut),
            ERROR_NULL_POINTER,
            "r"
        );

        let jac_mul = |y, v, jv| unsafe { pybamm_rust_jac_mul(0.0, y, nil, 1.0, v, jv, user_data) };
        assert_eq!(jac_mul(nil, ok_const, ok), ERROR_NULL_POINTER, "y");
        assert_eq!(jac_mul(ok_const, nil, ok), ERROR_NULL_POINTER, "v");
        assert_eq!(
            jac_mul(ok_const, ok_const, nil_mut),
            ERROR_NULL_POINTER,
            "jv"
        );

        let eval_rhs = |y, f_out| unsafe { pybamm_rust_eval_rhs(0.0, y, nil, f_out, user_data) };
        assert_eq!(eval_rhs(nil, ok), ERROR_NULL_POINTER, "y");
        assert_eq!(eval_rhs(ok_const, nil_mut), ERROR_NULL_POINTER, "f_out");

        let jac_assemble =
            |y, jac| unsafe { pybamm_rust_jac_assemble(0.0, y, nil, 1.0, jac, user_data) };
        assert_eq!(jac_assemble(nil, ok), ERROR_NULL_POINTER, "y");
        assert_eq!(
            jac_assemble(ok_const, nil_mut),
            ERROR_NULL_POINTER,
            "jac_data"
        );

        let jac_action =
            |y, v, jv| unsafe { pybamm_rust_jac_action(0.0, y, nil, v, jv, user_data) };
        assert_eq!(jac_action(nil, ok_const, ok), ERROR_NULL_POINTER, "y");
        assert_eq!(jac_action(ok_const, nil, ok), ERROR_NULL_POINTER, "v");
        assert_eq!(
            jac_action(ok_const, ok_const, nil_mut),
            ERROR_NULL_POINTER,
            "jv"
        );

        let mass_action = |v, mv| unsafe { pybamm_rust_mass_action(v, mv, user_data) };
        assert_eq!(mass_action(nil, ok), ERROR_NULL_POINTER, "v");
        assert_eq!(mass_action(ok_const, nil_mut), ERROR_NULL_POINTER, "mv");

        assert_eq!(
            unsafe { pybamm_rust_algebraic_ids(nil_mut, user_data) },
            ERROR_NULL_POINTER,
            "ids_out"
        );
        assert_eq!(
            unsafe { pybamm_rust_jac_csc_colptr(std::ptr::null_mut(), user_data) },
            ERROR_NULL_POINTER,
            "colptr"
        );
        assert_eq!(
            unsafe { pybamm_rust_jac_csc_rowind(std::ptr::null_mut(), user_data) },
            ERROR_NULL_POINTER,
            "rowind"
        );
    }

    #[test]
    fn every_required_buffer_of_the_sensitivity_and_output_calls_is_null_checked() {
        let mut model = create_test_model();
        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let mut scratch = [0.0f64; 4];
        let ok = scratch.as_mut_ptr();
        let ok_const: *const f64 = ok;
        let nil = std::ptr::null::<f64>();
        let nil_mut = std::ptr::null_mut::<f64>();

        let sens_eval =
            |y, df_dp| unsafe { pybamm_rust_sens_eval(0.0, y, nil, 0, df_dp, user_data) };
        assert_eq!(sens_eval(nil, ok), ERROR_NULL_POINTER, "y");
        assert_eq!(sens_eval(ok_const, nil_mut), ERROR_NULL_POINTER, "df_dp");

        let sens_all = |y, out| unsafe { pybamm_rust_sens_eval_all(0.0, y, nil, out, user_data) };
        assert_eq!(sens_all(nil, ok), ERROR_NULL_POINTER, "y");
        assert_eq!(sens_all(ok_const, nil_mut), ERROR_NULL_POINTER, "df_dp_out");

        let project = |y, y_sens, out| unsafe {
            pybamm_rust_output_sens_project(0.0, y, nil, y_sens, out, user_data)
        };
        assert_eq!(project(nil, ok_const, ok), ERROR_NULL_POINTER, "y");
        assert_eq!(project(ok_const, nil, ok), ERROR_NULL_POINTER, "y_sens");
        assert_eq!(
            project(ok_const, ok_const, nil_mut),
            ERROR_NULL_POINTER,
            "out"
        );

        // `out_len` is the one optional buffer: null there is not an error.
        let output_eval = |y, out| unsafe {
            pybamm_rust_output_eval(0.0, y, nil, 0, out, std::ptr::null_mut(), user_data)
        };
        assert_eq!(output_eval(nil, ok), ERROR_NULL_POINTER, "y");
        assert_eq!(output_eval(ok_const, nil_mut), ERROR_NULL_POINTER, "out");

        let batch =
            |ts, ys, out| unsafe { pybamm_rust_output_eval_batch(ts, ys, 1, nil, out, user_data) };
        assert_eq!(batch(nil, ok_const, ok), ERROR_NULL_POINTER, "ts");
        assert_eq!(batch(ok_const, nil, ok), ERROR_NULL_POINTER, "ys");
        assert_eq!(
            batch(ok_const, ok_const, nil_mut),
            ERROR_NULL_POINTER,
            "out"
        );
    }

    #[test]
    fn every_required_buffer_of_the_algebraic_and_event_calls_is_null_checked() {
        let mut model = create_test_model();
        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let mut scratch = [0.0f64; 4];
        let ok = scratch.as_mut_ptr();
        let ok_const: *const f64 = ok;
        let nil = std::ptr::null::<f64>();
        let nil_mut = std::ptr::null_mut::<f64>();

        let alg_res = |y, out| unsafe { pybamm_rust_alg_res(0.0, y, nil, out, user_data) };
        assert_eq!(alg_res(nil, ok), ERROR_NULL_POINTER, "y");
        assert_eq!(alg_res(ok_const, nil_mut), ERROR_NULL_POINTER, "output");

        let alg_jac = |y, out| unsafe { pybamm_rust_alg_jac_assemble(0.0, y, nil, out, user_data) };
        assert_eq!(alg_jac(nil, ok), ERROR_NULL_POINTER, "y");
        assert_eq!(alg_jac(ok_const, nil_mut), ERROR_NULL_POINTER, "output");

        let alg_action =
            |y, v, jv| unsafe { pybamm_rust_alg_jac_action(0.0, y, nil, v, jv, user_data) };
        assert_eq!(alg_action(nil, ok_const, ok), ERROR_NULL_POINTER, "y");
        assert_eq!(alg_action(ok_const, nil, ok), ERROR_NULL_POINTER, "v");
        assert_eq!(
            alg_action(ok_const, ok_const, nil_mut),
            ERROR_NULL_POINTER,
            "jv"
        );

        let events = |y, out| unsafe { pybamm_rust_events_eval(0.0, y, nil, out, user_data) };
        assert_eq!(events(nil, ok), ERROR_NULL_POINTER, "y");
        assert_eq!(events(ok_const, nil_mut), ERROR_NULL_POINTER, "output");
    }
}
