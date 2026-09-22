//! C ABI entry points exposing `ModelEvaluator` to the C++ IDAKLU solver.
//!
//! Each returns `SUCCESS` (or a count), `ERROR_NULL_POINTER` on a null, or `ERROR_PANIC` on a
//! caught panic. All require `user_data` to point to a live `ModelEvaluator` no other call is
//! using, `inputs` to hold `input_width` elements (or be null at width zero), each buffer to
//! hold the count its `# Safety` names, and no output to overlap another argument. Each
//! `unsafe` block borrows only guard-checked pointers, at those counts.

#![allow(unsafe_code)]

use crate::model::{CompiledModel, Workspace};
use crate::observable::ObservableKind;
use std::ffi::c_void;
use std::os::raw::c_int;
use std::sync::Arc;

/// The `user_data` behind every entry point: a shared model and one caller's scratch.
#[derive(Debug)]
pub struct ModelEvaluator {
    compiled: Arc<CompiledModel>,
    workspace: Workspace,
}

impl ModelEvaluator {
    /// Bind a fresh [`Workspace`] to an already-compiled model, without recompiling.
    pub fn from_compiled(compiled: Arc<CompiledModel>) -> Self {
        let workspace = compiled.create_workspace();
        Self {
            compiled,
            workspace,
        }
    }
}

/// Success return code.
pub const SUCCESS: c_int = 0;

/// Error: null pointer passed to function.
pub const ERROR_NULL_POINTER: c_int = -1;

/// Error: panic occurred during execution.
pub const ERROR_PANIC: c_int = -2;

/// Error: a non-positive block count (`n_dirs` or `n_points`) supplied to a batched FFI fn.
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

/// ABI contract version, bumped with `PYBAMM_RUST_ABI_VERSION` on any exported signature change.
///
/// `pybamm_rust_abi_version` must keep its no-argument `-> u32` signature: it is the probe.
pub const RUST_ABI_VERSION: u32 = 2;

/// Return the FFI ABI contract version.
#[unsafe(no_mangle)]
pub const extern "C" fn pybamm_rust_abi_version() -> u32 {
    RUST_ABI_VERSION
}

/// Golden hash of the exported signatures, pinned by the `test_ffi_abi_contract` drift test.
pub const EXPECTED_ABI_HASH: u64 = 0x51e2_4131_2250_caf6;

/// Run `body` on the model behind `user_data`, rejecting nulls and containing panics.
///
/// # Safety
///
/// `user_data` must point to a live `ModelEvaluator`, not mutably aliased for the call.
#[inline]
unsafe fn with_model<const N: usize, F>(
    user_data: *const c_void,
    required: [*const c_void; N],
    body: F,
) -> c_int
where
    F: FnOnce(&CompiledModel) -> c_int,
{
    if user_data.is_null() || any_null(required) {
        return ERROR_NULL_POINTER;
    }
    // SAFETY: null-checked above; the caller guarantees a live `ModelEvaluator`.
    let model = unsafe { &*user_data.cast::<ModelEvaluator>() };
    caught(|| body(&model.compiled)).unwrap_or(ERROR_PANIC)
}

/// Like [`with_model`], but also lends the evaluator's workspace and its `inputs` buffer.
///
/// Null `inputs` is accepted only when the model declares no input parameters. A caught panic
/// also rebuilds the workspace, so no scratch state from an interrupted call survives.
///
/// # Safety
///
/// `user_data` must point to a live, unaliased `ModelEvaluator`. When the model declares input
/// parameters, `inputs` must point to at least that many `f64`.
#[inline]
unsafe fn with_model_and_inputs<const N: usize, F>(
    user_data: *mut c_void,
    required: [*const c_void; N],
    inputs: *const f64,
    body: F,
) -> c_int
where
    F: FnOnce(&CompiledModel, &mut Workspace, &[f64]) -> c_int,
{
    if user_data.is_null() || any_null(required) {
        return ERROR_NULL_POINTER;
    }
    // SAFETY: null-checked above; the caller guarantees a live `ModelEvaluator`.
    let ModelEvaluator {
        compiled,
        workspace,
    } = unsafe { &mut *user_data.cast::<ModelEvaluator>() };
    let input_width = compiled.input_width().get();
    if input_width != 0 && inputs.is_null() {
        return ERROR_NULL_POINTER;
    }
    let inputs: &[f64] = if input_width == 0 {
        &[]
    } else {
        // SAFETY: non-null per the check above; the caller guarantees `input_width` elements.
        unsafe { borrow_slice(inputs, input_width) }
    };
    caught(|| body(compiled, workspace, inputs)).unwrap_or_else(|| {
        // Cannot unwind: `from_compiled` built the same workspace.
        *workspace = compiled.create_workspace();
        ERROR_PANIC
    })
}

#[inline]
fn any_null<const N: usize>(pointers: [*const c_void; N]) -> bool {
    pointers.iter().any(|pointer| pointer.is_null())
}

/// Run `body`, returning `None` on an unwind so it never crosses back into C.
///
/// `AssertUnwindSafe` holds because a panic is reported, never resumed: the caller discards its
/// outputs, and [`with_model_and_inputs`] replaces the workspace, the only state a body mutates.
/// A payload whose own drop panics escapes this, and the `extern "C"` boundary then aborts.
#[inline]
fn caught<F: FnOnce() -> c_int>(body: F) -> Option<c_int> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)).ok()
}

/// `count` as a C `int`, panicking (so `ERROR_PANIC`) where a cast would wrap it.
///
/// A wrapped positive count would size the caller's buffer smaller than Rust then writes.
#[inline]
fn c_count(count: usize) -> c_int {
    c_int::try_from(count).expect("count exceeds the C int range")
}

/// `index` as a position in `0..len`, or `None` when negative or out of range.
#[inline]
fn checked_index(index: c_int, len: usize) -> Option<usize> {
    usize::try_from(index).ok().filter(|&index| index < len)
}

/// `count` as a block count, or `None` unless it is positive.
#[inline]
fn positive_count(count: c_int) -> Option<usize> {
    usize::try_from(count).ok().filter(|&count| count > 0)
}

/// Element count of `blocks` contiguous blocks of `width`, panicking rather than wrapping.
///
/// No caller buffer can hold an overflowing count, so overflow is a contract violation.
#[inline]
const fn blocks_len(blocks: usize, width: usize) -> usize {
    blocks
        .checked_mul(width)
        .expect("buffer length overflows usize")
}

/// Borrow `len` elements of a caller-provided buffer.
///
/// # Safety
///
/// `pointer` must be non-null and point to `len` initialized `T`, unaliased for the borrow.
#[inline]
const unsafe fn borrow_slice<'a, T>(pointer: *const T, len: usize) -> &'a [T] {
    // SAFETY: guaranteed by this function's own contract.
    unsafe { std::slice::from_raw_parts(pointer, len) }
}

/// Borrow `len` writable elements of a caller-provided buffer.
///
/// # Safety
///
/// `pointer` must be non-null and point to `len` writable `T`, unaliased for the borrow.
#[inline]
const unsafe fn borrow_slice_mut<'a, T>(pointer: *mut T, len: usize) -> &'a mut [T] {
    // SAFETY: guaranteed by this function's own contract.
    unsafe { std::slice::from_raw_parts_mut(pointer, len) }
}

/// Evaluate the right-hand side `f(t, y)` into `f_out`.
///
/// # Safety
///
/// `y` and `f_out` must hold `n_states` elements.
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
            |model, ws, inputs| {
                let n_states = model.n_states();
                model.eval_rhs(
                    ws,
                    t,
                    borrow_slice(y, n_states),
                    inputs,
                    borrow_slice_mut(f_out, n_states),
                );
                SUCCESS
            },
        )
    }
}

/// Number of states in the model.
///
/// # Safety
///
/// `user_data` must point to a live `ModelEvaluator`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pybamm_rust_n_states(user_data: *const c_void) -> c_int {
    unsafe { with_model(user_data, [], |model| c_count(model.n_states())) }
}

/// Packed width of the model's `inputs` buffer.
///
/// # Safety
///
/// `user_data` must point to a live `ModelEvaluator`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pybamm_rust_n_inputs(user_data: *const c_void) -> c_int {
    unsafe { with_model(user_data, [], |model| c_count(model.input_width().get())) }
}

/// Assemble `J = df/dy - cj*M` into a pre-allocated CSC data buffer.
///
/// # Safety
///
/// `y` must hold `n_states` elements and `jac_data` `nnz` elements.
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
            |model, ws, inputs| {
                let (n_states, nnz) = (model.n_states(), model.nnz());
                model.assemble_jacobian_csc_into(
                    ws,
                    t,
                    borrow_slice(y, n_states),
                    inputs,
                    cj,
                    borrow_slice_mut(jac_data, nnz),
                );
                SUCCESS
            },
        )
    }
}

/// Number of non-zeros in the Jacobian.
///
/// # Safety
///
/// `user_data` must point to a live `ModelEvaluator`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pybamm_rust_jac_nnz(user_data: *const c_void) -> c_int {
    unsafe { with_model(user_data, [], |model| c_count(model.nnz())) }
}

/// Jacobian-vector product `df/dy @ v`, without the mass term.
///
/// # Safety
///
/// `y`, `v` and `jv` must hold `n_states` elements.
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
            |model, ws, inputs| {
                let n_states = model.n_states();
                model.jac_action(
                    ws,
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

/// Jacobian-vector products `df/dy @ v` in `n_dirs` directions, sharing one primal pass.
///
/// `vs` and `jvs` hold one `n_states` block per direction, contiguously, and each block matches
/// [`pybamm_rust_jac_action`] bitwise. Also returns `ERROR_INVALID_PARAM` for a non-positive
/// `n_dirs`.
///
/// # Safety
///
/// `y` must hold `n_states` elements, and `vs` and `jvs` `n_dirs * n_states`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pybamm_rust_jac_action_batch(
    t: f64,
    y: *const f64,
    inputs: *const f64,
    n_dirs: c_int,
    vs: *const f64,
    jvs: *mut f64,
    user_data: *mut c_void,
) -> c_int {
    unsafe {
        with_model_and_inputs(
            user_data,
            [y.cast(), vs.cast(), jvs.cast()],
            inputs,
            |model, ws, inputs| {
                let Some(dirs) = positive_count(n_dirs) else {
                    return ERROR_INVALID_PARAM;
                };
                let n_states = model.n_states();
                let len = blocks_len(dirs, n_states);
                model
                    .jac_action_at(ws, t, borrow_slice(y, n_states), inputs)
                    .apply_lanes(
                        borrow_slice(vs, len).chunks_exact(n_states),
                        borrow_slice_mut(jvs, len).chunks_exact_mut(n_states),
                    );
                SUCCESS
            },
        )
    }
}

/// Mass-matrix action `mv = M @ v`.
///
/// # Safety
///
/// `v` and `mv` must hold `n_states` elements.
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

/// Number of forward-sensitivity parameters configured on the model.
///
/// # Safety
///
/// `user_data` must point to a live `ModelEvaluator`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pybamm_rust_n_sens_params(user_data: *const c_void) -> c_int {
    unsafe { with_model(user_data, [], |model| c_count(model.n_sens_params())) }
}

/// Evaluate `df/dp` for all sensitivity parameters into `df_dp_out`.
///
/// Layout: `df_dp_out[i*n_states + j] = ∂f_j/∂p_i`. Also returns `ERROR_NO_SENS`.
///
/// # Safety
///
/// `y` must hold `n_states` elements and `df_dp_out` `n_sens_params * n_states`.
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
            |model, ws, inputs| {
                if model.n_sens_params() == 0 {
                    return ERROR_NO_SENS;
                }
                let n_states = model.n_states();
                let n_sens = model.n_sens_params();
                model.eval_sens_all(
                    ws,
                    t,
                    borrow_slice(y, n_states),
                    inputs,
                    borrow_slice_mut(df_dp_out, blocks_len(n_sens, n_states)),
                );
                SUCCESS
            },
        )
    }
}

/// Project state sensitivities onto output sensitivities for every output.
///
/// `dvar/dp_k = dH/dp . e_k + dH/dy . y_sens_k`, with layouts `y_sens[k*n_states + j]`
/// and `out[k*n_out + o]`. Also returns `ERROR_NO_OUTPUTS` or `ERROR_NO_SENS`.
///
/// # Safety
///
/// `y` must hold `n_states`, `y_sens` `n_sens_params * n_states` and `out`
/// `n_sens_params * total_output_len` elements.
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
            |model, ws, inputs| {
                if model.n_outputs() == 0 {
                    return ERROR_NO_OUTPUTS;
                }
                if model.n_sens_params() == 0 {
                    return ERROR_NO_SENS;
                }
                let n_states = model.n_states();
                let (n_sens, n_out) = (model.n_sens_params(), model.total_output_len());
                model.output_sens_project(
                    ws,
                    t,
                    borrow_slice(y, n_states),
                    inputs,
                    borrow_slice(y_sens, blocks_len(n_sens, n_states)),
                    borrow_slice_mut(out, blocks_len(n_sens, n_out)),
                );
                SUCCESS
            },
        )
    }
}

/// Number of compiled output-variable expressions on the model.
///
/// # Safety
///
/// `user_data` must point to a live `ModelEvaluator`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pybamm_rust_n_outputs(user_data: *const c_void) -> c_int {
    unsafe { with_model(user_data, [], |model| c_count(model.n_outputs())) }
}

/// Length of output variable `var_idx`, or `ERROR_INVALID_OUTPUT` if out of range.
///
/// # Safety
///
/// `user_data` must point to a live `ModelEvaluator`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pybamm_rust_output_len(user_data: *const c_void, var_idx: c_int) -> c_int {
    unsafe {
        with_model(user_data, [], |model| {
            checked_index(var_idx, model.n_outputs()).map_or(ERROR_INVALID_OUTPUT, |var| {
                c_count(model.output_len_at(var))
            })
        })
    }
}

/// Evaluate every output variable at `n_points` trajectory points.
///
/// `ys` is `(n_states, n_points)` F-contiguous and `out` holds each point's stacked outputs
/// contiguously; results match per-point evaluation bitwise. Also returns
/// `ERROR_INVALID_PARAM` for a non-positive `n_points`, or `ERROR_NO_OUTPUTS`.
///
/// # Safety
///
/// `ts` must hold `n_points`, `ys` `n_points * n_states` and `out`
/// `n_points * total_output_len` elements.
#[unsafe(no_mangle)]
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
            |model, ws, inputs| {
                let Some(points) = positive_count(n_points) else {
                    return ERROR_INVALID_PARAM;
                };
                if model.n_outputs() == 0 {
                    return ERROR_NO_OUTPUTS;
                }
                let n_states = model.n_states();
                let total = model.total_output_len();
                model.eval_outputs_batch(
                    ws,
                    points,
                    borrow_slice(ts, points),
                    borrow_slice(ys, blocks_len(points, n_states)),
                    inputs,
                    borrow_slice_mut(out, blocks_len(points, total)),
                );
                SUCCESS
            },
        )
    }
}

/// Evaluate the algebraic residual `g(t, y)`, or return `ERROR_NO_ALG` without one.
///
/// # Safety
///
/// `y` must hold `n_states` elements and `output` `n_algebraic`.
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
            |model, ws, inputs| {
                let Some(block) = model.algebraic() else {
                    return ERROR_NO_ALG;
                };
                block.eval_residual(
                    ws,
                    t,
                    borrow_slice(y, model.n_states()),
                    inputs,
                    borrow_slice_mut(output, block.n_variables()),
                );
                SUCCESS
            },
        )
    }
}

/// Assemble the algebraic Jacobian `dg/dy_alg`, or return `ERROR_NO_ALG` without one.
///
/// # Safety
///
/// `y` must hold `n_states` elements and `output` `algebraic_jacobian_nnz`.
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
            |model, ws, inputs| {
                let Some(block) = model.algebraic() else {
                    return ERROR_NO_ALG;
                };
                block.assemble_jacobian_into(
                    ws,
                    t,
                    borrow_slice(y, model.n_states()),
                    inputs,
                    borrow_slice_mut(output, block.jacobian().nnz()),
                );
                SUCCESS
            },
        )
    }
}

/// Total length of all events concatenated.
///
/// # Safety
///
/// `user_data` must point to a live `ModelEvaluator`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pybamm_rust_total_event_len(user_data: *const c_void) -> c_int {
    unsafe { with_model(user_data, [], |model| c_count(model.total_event_len())) }
}

/// Evaluate all events, concatenated in order, or return `ERROR_NO_EVENTS` without any.
///
/// # Safety
///
/// `y` must hold `n_states` elements and `output` `total_event_len`.
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
            |model, ws, inputs| {
                if model.n_events() == 0 {
                    return ERROR_NO_EVENTS;
                }
                let n_states = model.n_states();
                let total_len = model.total_event_len();
                model.eval_observables(
                    ws,
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
    use crate::arena::{Arena, NodeId};
    use crate::jacobian::tests::banded_nonlinear;
    use crate::model::test_fixtures::{
        compile, compile_with_sens, doubling_model, element, identity_mass_matrix, input,
        split_algebraic_model,
    };
    use crate::node::Node;
    use crate::params::ParamIdx;

    /// An evaluator over `model`, as C++ receives one.
    fn evaluator(model: CompiledModel) -> Box<ModelEvaluator> {
        Box::new(ModelEvaluator::from_compiled(Arc::new(model)))
    }

    /// `f(y) = 2*y` with an identity mass matrix.
    fn create_test_model() -> Box<ModelEvaluator> {
        evaluator(doubling_model(2))
    }

    #[test]
    fn test_rust_jac_action_uses_inputs() {
        // df/dy @ v = k * v with k = 7; dropped inputs would read k = 0.
        let mut model = create_sens_test_model();
        let y = [3.0, 4.0];
        let inputs = [7.0];
        let v = [1.0, 0.0];
        let mut jv = [0.0, 0.0];

        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let result = unsafe {
            pybamm_rust_jac_action(
                0.0,
                y.as_ptr(),
                inputs.as_ptr(),
                v.as_ptr(),
                jv.as_mut_ptr(),
                user_data,
            )
        };

        assert_eq!(result, SUCCESS);
        assert!(
            (jv[0] - 7.0).abs() < 1e-12,
            "expected jv[0]=7, got {}",
            jv[0]
        );
        assert!(jv[1].abs() < 1e-12, "expected jv[1]=0, got {}", jv[1]);
    }

    #[test]
    fn test_rust_eval_rhs_success() {
        let mut model = create_test_model();

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
        let mut model = evaluator(split_algebraic_model());
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
        let mut model = evaluator(split_algebraic_model());
        let y = [10.0, 2.0, 3.0];
        let nnz = (model.compiled.algebraic()).map_or(0, |block| block.jacobian().nnz());
        let mut jac = vec![0.0; nnz];
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
    fn test_rust_jac_assemble_success() {
        let mut model = create_test_model();

        // J = df/dy - cj*M = 2*I - 0.5*I
        let y = [1.0, 2.0];
        let mut jac_data = vec![0.0; model.compiled.nnz()];

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
        for &val in &jac_data {
            assert!((val - 1.5).abs() < 1e-14, "Expected 1.5, got {val}");
        }
    }

    #[test]
    fn jac_action_batch_matches_one_jac_action_per_direction_bitwise() {
        let n = 64;
        let (arena, rhs) = banded_nonlinear(n, 3);
        let mut model = evaluator(compile(&arena, rhs, identity_mass_matrix(n), n, 0));
        let stats = model.compiled.jacobian_stats();
        assert!(stats.jac_lane_width > 1, "the fixture must batch");
        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let y: Vec<f64> = (0..n)
            .map(|i| f64::from(u8::try_from(i).unwrap()) / 27.0)
            .collect();
        // 11 directions: a full 8-lane block, then a 4-lane block with a spare lane.
        let vs: Vec<f64> = (0..11 * n)
            .map(|k| f64::from(u8::try_from((k * 7 + k / n * 3) % 11).unwrap()) - 5.0)
            .collect();
        let mut batched = vec![f64::NAN; vs.len()];
        let mut batch = |n_dirs| unsafe {
            pybamm_rust_jac_action_batch(
                0.5,
                y.as_ptr(),
                std::ptr::null(),
                n_dirs,
                vs.as_ptr(),
                batched.as_mut_ptr(),
                user_data,
            )
        };
        assert_eq!(batch(0), ERROR_INVALID_PARAM);
        assert_eq!(batch(-1), ERROR_INVALID_PARAM);
        assert_eq!(batch(11), SUCCESS);

        let mut single = vec![f64::NAN; n];
        for (v, got) in vs.chunks_exact(n).zip(batched.chunks_exact(n)) {
            let status = unsafe {
                pybamm_rust_jac_action(
                    0.5,
                    y.as_ptr(),
                    std::ptr::null(),
                    v.as_ptr(),
                    single.as_mut_ptr(),
                    user_data,
                )
            };
            assert_eq!(status, SUCCESS);
            assert!(
                got.iter()
                    .map(|x| x.to_bits())
                    .eq(single.iter().map(|x| x.to_bits()))
            );
        }
    }

    #[test]
    fn test_rust_mass_action_success() {
        let model = create_test_model();

        let v = [3.0, 4.0];
        let mut mv = [0.0, 0.0];

        let user_data: *const c_void = std::ptr::from_ref(model.as_ref()).cast();
        let result =
            unsafe { pybamm_rust_mass_action(v.as_ptr(), mv.as_mut_ptr(), user_data.cast_mut()) };

        assert_eq!(result, SUCCESS);
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

    /// `f(y) = k*y` with input `k`, and sensitivities for `k`.
    fn create_sens_test_model() -> Box<ModelEvaluator> {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let k = input(&mut arena, "k", 0, 0, 1);
        let rhs = arena.alloc(Node::Mul(k, y));
        let sens = [ParamIdx::new(0)];
        evaluator(compile_with_sens(
            &arena,
            rhs,
            identity_mass_matrix(2),
            2,
            1,
            &sens,
        ))
    }

    #[test]
    fn test_rust_sens_eval_all_no_sensitivities() {
        let mut model = create_test_model();
        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let y = [1.0, 2.0];
        let mut out = [0.0; 2];
        let result = unsafe {
            pybamm_rust_sens_eval_all(
                0.0,
                y.as_ptr(),
                std::ptr::null(),
                out.as_mut_ptr(),
                user_data,
            )
        };
        assert_eq!(result, ERROR_NO_SENS);
    }

    #[test]
    fn test_rust_sens_eval_all_basic() {
        let mut model = create_sens_test_model();
        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let y = [3.0, 4.0];
        let inputs = [7.0];
        let mut out = [0.0; 2];
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
        observed_test_model(ObservableKind::Outputs, |arena, y| {
            let y0 = element(arena, y, 0);
            let two = arena.alloc(Node::Scalar(2.0));
            arena.alloc(Node::Mul(two, y0))
        })
    }

    fn create_event_test_model() -> Box<ModelEvaluator> {
        observed_test_model(ObservableKind::Events, |arena, y| {
            let y0 = element(arena, y, 0);
            let threshold = arena.alloc(Node::Scalar(0.5));
            arena.alloc(Node::Sub(y0, threshold))
        })
    }

    /// `f(y) = y` on two states, with the root `build` returns as its only observable of `kind`.
    fn observed_test_model(
        kind: ObservableKind,
        build: impl FnOnce(&mut Arena, NodeId) -> NodeId,
    ) -> Box<ModelEvaluator> {
        let mut arena = Arena::new();
        let y = arena.alloc(Node::StateVector { start: 0, end: 2 });
        let root = build(&mut arena, y);
        let mut model = compile(&arena, y, identity_mass_matrix(2), 2, 0);
        model
            .set_observables(kind, &mut arena, &[root])
            .expect("a valid observable");
        evaluator(model)
    }

    #[test]
    fn count_queries_report_the_compiled_sizes() {
        type CountQuery = unsafe extern "C" fn(*const c_void) -> c_int;
        let (plain, sens) = (create_test_model(), create_sens_test_model());
        let (outputs, events) = (create_output_test_model(), create_event_test_model());
        let queries: [(&str, CountQuery, &ModelEvaluator, c_int); 6] = [
            ("n_states", pybamm_rust_n_states, &plain, 2),
            ("jac_nnz", pybamm_rust_jac_nnz, &plain, 2),
            ("n_inputs", pybamm_rust_n_inputs, &sens, 1),
            ("n_sens_params", pybamm_rust_n_sens_params, &sens, 1),
            ("n_outputs", pybamm_rust_n_outputs, &outputs, 1),
            ("total_event_len", pybamm_rust_total_event_len, &events, 1),
        ];
        for (name, query, model, expected) in queries {
            let user_data: *const c_void = std::ptr::from_ref(model).cast();
            assert_eq!(unsafe { query(user_data) }, expected, "{name}");
        }
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
    fn negative_indices_are_rejected_not_wrapped() {
        let model = create_output_test_model();
        let user_data: *const c_void = std::ptr::from_ref(model.as_ref()).cast();
        assert_eq!(
            unsafe { pybamm_rust_output_len(user_data, -1) },
            ERROR_INVALID_OUTPUT
        );
    }

    #[test]
    fn output_eval_batch_lays_out_each_point_contiguously() {
        let mut model = create_output_test_model();
        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let ts = [0.0, 1.0];
        let ys = [3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0; 2];
        let mut batch = |n_points| unsafe {
            pybamm_rust_output_eval_batch(
                ts.as_ptr(),
                ys.as_ptr(),
                n_points,
                std::ptr::null(),
                out.as_mut_ptr(),
                user_data,
            )
        };
        assert_eq!(batch(0), ERROR_INVALID_PARAM);
        assert_eq!(batch(-1), ERROR_INVALID_PARAM);
        assert_eq!(batch(2), SUCCESS);
        assert_eq!(out.map(f64::to_bits), [6.0, 10.0].map(f64::to_bits));
    }

    #[test]
    fn counts_and_lengths_that_do_not_fit_panic_instead_of_wrapping() {
        let max = usize::try_from(c_int::MAX).expect("c_int::MAX fits in usize");
        assert_eq!(c_count(max), c_int::MAX);
        assert_eq!(blocks_len(3, 4), 12);

        let model = create_test_model();
        let shared: *const c_void = std::ptr::from_ref(model.as_ref()).cast();
        assert_eq!(
            unsafe { with_model(shared, [], |_| c_count(max + 1)) },
            ERROR_PANIC
        );
        assert_eq!(
            unsafe { with_model(shared, [], |_| c_count(blocks_len(usize::MAX, 2))) },
            ERROR_PANIC
        );
    }

    #[test]
    fn test_rust_events_eval_no_events() {
        let mut model = create_test_model();
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
        let mut out = [f64::NAN];
        for (y0, expected) in [(0.7, 0.2), (0.5, 0.0)] {
            let y = [y0, 1.0];
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
            assert!(
                (out[0] - expected).abs() < 1e-12,
                "y0 = {y0}: got {}",
                out[0]
            );
        }
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
            unsafe { with_model_and_inputs(owned, [], std::ptr::null(), |_, _, _| panic!("boom")) },
            ERROR_PANIC
        );
    }

    #[test]
    fn guard_rejects_nulls_without_running_the_body() {
        let model = create_test_model();
        let shared: *const c_void = std::ptr::from_ref(model.as_ref()).cast();
        let unreachable = |_: &CompiledModel| unreachable!("guard let a null through");

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

    #[test]
    fn null_inputs_are_rejected_for_a_model_that_takes_parameters() {
        let mut model = create_sens_test_model();
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

    /// Every entry point's status for a null `user_data`, given these buffer pointers.
    fn statuses_for_null_user_data(
        input: *const f64,
        output: *mut f64,
    ) -> [(&'static str, c_int); 18] {
        let nil_data = std::ptr::null_mut::<c_void>();
        [
            ("pybamm_rust_eval_rhs", unsafe {
                pybamm_rust_eval_rhs(0.0, input, input, output, nil_data)
            }),
            ("pybamm_rust_n_states", unsafe {
                pybamm_rust_n_states(std::ptr::null())
            }),
            ("pybamm_rust_n_inputs", unsafe {
                pybamm_rust_n_inputs(std::ptr::null())
            }),
            ("pybamm_rust_jac_assemble", unsafe {
                pybamm_rust_jac_assemble(0.0, input, input, 1.0, output, nil_data)
            }),
            ("pybamm_rust_jac_nnz", unsafe {
                pybamm_rust_jac_nnz(std::ptr::null())
            }),
            ("pybamm_rust_jac_action", unsafe {
                pybamm_rust_jac_action(0.0, input, input, input, output, nil_data)
            }),
            // Also pins the order: nulls outrank the `n_dirs > 0` rule.
            ("pybamm_rust_jac_action_batch", unsafe {
                pybamm_rust_jac_action_batch(0.0, input, input, 0, input, output, nil_data)
            }),
            ("pybamm_rust_mass_action", unsafe {
                pybamm_rust_mass_action(input, output, nil_data)
            }),
            ("pybamm_rust_n_sens_params", unsafe {
                pybamm_rust_n_sens_params(std::ptr::null())
            }),
            ("pybamm_rust_sens_eval_all", unsafe {
                pybamm_rust_sens_eval_all(0.0, input, input, output, nil_data)
            }),
            ("pybamm_rust_output_sens_project", unsafe {
                pybamm_rust_output_sens_project(0.0, input, input, input, output, nil_data)
            }),
            ("pybamm_rust_n_outputs", unsafe {
                pybamm_rust_n_outputs(std::ptr::null())
            }),
            ("pybamm_rust_output_len", unsafe {
                pybamm_rust_output_len(std::ptr::null(), 0)
            }),
            // Also pins the order: nulls outrank the `n_points > 0` rule.
            ("pybamm_rust_output_eval_batch", unsafe {
                pybamm_rust_output_eval_batch(input, input, 0, input, output, nil_data)
            }),
            ("pybamm_rust_alg_res", unsafe {
                pybamm_rust_alg_res(0.0, input, input, output, nil_data)
            }),
            ("pybamm_rust_alg_jac_assemble", unsafe {
                pybamm_rust_alg_jac_assemble(0.0, input, input, output, nil_data)
            }),
            ("pybamm_rust_total_event_len", unsafe {
                pybamm_rust_total_event_len(std::ptr::null())
            }),
            ("pybamm_rust_events_eval", unsafe {
                pybamm_rust_events_eval(0.0, input, input, output, nil_data)
            }),
        ]
    }

    #[test]
    fn every_entry_point_rejects_a_null_user_data() {
        let (source, mut sink) = ([0.0f64; 4], [0.0f64; 4]);
        let all_null = statuses_for_null_user_data(std::ptr::null(), std::ptr::null_mut());
        // Valid buffers isolate `user_data` as the null that must trip the guard.
        let only_user_data = statuses_for_null_user_data(source.as_ptr(), sink.as_mut_ptr());
        for (name, status) in all_null.into_iter().chain(only_user_data) {
            assert_eq!(
                status, ERROR_NULL_POINTER,
                "{name} accepted a null user_data"
            );
        }
    }

    /// The model has no optional blocks, so this also pins nulls outranking those error codes.
    #[test]
    fn every_required_buffer_of_the_solver_callbacks_is_null_checked() {
        let mut model = create_test_model();
        let user_data: *mut c_void = std::ptr::from_mut(model.as_mut()).cast();
        let mut scratch = [0.0f64; 4];
        let ok = scratch.as_mut_ptr();
        let ok_const: *const f64 = ok;
        let nil = std::ptr::null::<f64>();
        let nil_mut = std::ptr::null_mut::<f64>();

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

        let jac_action_batch = |y, vs, jvs| unsafe {
            pybamm_rust_jac_action_batch(0.0, y, nil, 1, vs, jvs, user_data)
        };
        assert_eq!(jac_action_batch(nil, ok_const, ok), ERROR_NULL_POINTER, "y");
        assert_eq!(
            jac_action_batch(ok_const, nil, ok),
            ERROR_NULL_POINTER,
            "vs"
        );
        assert_eq!(
            jac_action_batch(ok_const, ok_const, nil_mut),
            ERROR_NULL_POINTER,
            "jvs"
        );

        let mass_action = |v, mv| unsafe { pybamm_rust_mass_action(v, mv, user_data) };
        assert_eq!(mass_action(nil, ok), ERROR_NULL_POINTER, "v");
        assert_eq!(mass_action(ok_const, nil_mut), ERROR_NULL_POINTER, "mv");
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

        let events = |y, out| unsafe { pybamm_rust_events_eval(0.0, y, nil, out, user_data) };
        assert_eq!(events(nil, ok), ERROR_NULL_POINTER, "y");
        assert_eq!(events(ok_const, nil_mut), ERROR_NULL_POINTER, "output");
    }
}
