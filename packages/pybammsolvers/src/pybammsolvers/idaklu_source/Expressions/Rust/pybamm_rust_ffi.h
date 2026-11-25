#ifndef PYBAMM_RUST_FFI_H
#define PYBAMM_RUST_FFI_H

#include <stdint.h>

/* Return codes */
#define PYBAMM_SUCCESS              0
#define PYBAMM_ERROR_NULL          (-1)
#define PYBAMM_ERROR_PANIC         (-2)
#define PYBAMM_ERROR_INVALID_PARAM (-3)
#define PYBAMM_ERROR_INVALID_OUTPUT (-4)
#define PYBAMM_ERROR_NO_SENS       (-6)
#define PYBAMM_ERROR_NO_OUTPUTS    (-7)
#define PYBAMM_ERROR_NO_ALG        (-8)
#define PYBAMM_ERROR_NO_EVENTS     (-9)
#define PYBAMM_ERROR_NO_UPSAMPLING (-10)

/* Major ABI version: must equal RUST_ABI_VERSION in ffi.rs, checked for every evaluator. */
#define PYBAMM_RUST_ABI_VERSION 2

/* A capsule holding an evaluator nothing has claimed, and the name its owner renames it to. */
#define PYBAMM_RUST_EVALUATOR_CAPSULE "pybamm.rust.evaluator"
#define PYBAMM_RUST_CLAIMED_EVALUATOR_CAPSULE "pybamm.rust.evaluator.claimed"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PybammRustEvaluator PybammRustEvaluator;

/*
 * Mirrors `FfiTable` in ffi.rs. `abi_version` and `size` lead every version's table; entries
 * are only appended within a version, so `size` must cover every entry a consumer reads.
 */
typedef struct PybammRustFfi {
    uint32_t abi_version;
    uint32_t size;
    void (*release)(PybammRustEvaluator*);
    int (*n_states)(const void*);
    int (*n_inputs)(const void*);
    int (*jac_nnz)(const void*);
    int (*n_sens_params)(const void*);
    int (*eval_rhs)(double, const double*, const double*, double*, void*);
    int (*jac_assemble)(double, const double*, const double*, double, double*, void*);
    int (*jac_action)(double, const double*, const double*, const double*, double*, void*);
    int (*jac_action_batch)(
        double, const double*, const double*, int, const double*, double*, void*);
    int (*mass_action)(const double*, double*, void*);
    int (*sens_eval_all)(double, const double*, const double*, double*, void*);
    int (*output_sens_project)(
        double, const double*, const double*, const double*, double*, void*);
    int (*n_outputs)(const void*);
    int (*output_len)(const void*, int);
    int (*output_eval_batch)(const double*, const double*, int, const double*, double*, void*);
    int (*n_algebraic)(const void*);
    int (*alg_jac_nnz)(const void*);
    int (*alg_res)(double, const double*, const double*, double*, void*);
    int (*alg_jac_assemble)(double, const double*, const double*, double*, void*);
    int (*total_event_len)(const void*);
    int (*events_eval)(double, const double*, const double*, const double*, double*, void*);
    int (*upsampling_len)(const void*);
    int (*upsampling_eval)(double, const double*, const double*, double*, void*);
} PybammRustFfi;

/* Mirrors `EvaluatorHandle` in ffi.rs; `ffi` leads in every version. */
struct PybammRustEvaluator {
    const PybammRustFfi* ffi;
    /* The `user_data` every entry point takes. */
    void* model;
};

#ifdef __cplusplus
}

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace pybamm_rust_detail {

inline const char* status_name(int status) {
    switch (status) {
        case PYBAMM_SUCCESS: return "SUCCESS";
        case PYBAMM_ERROR_NULL: return "ERROR_NULL";
        case PYBAMM_ERROR_PANIC: return "ERROR_PANIC";
        case PYBAMM_ERROR_INVALID_PARAM: return "ERROR_INVALID_PARAM";
        case PYBAMM_ERROR_INVALID_OUTPUT: return "ERROR_INVALID_OUTPUT";
        case PYBAMM_ERROR_NO_SENS: return "ERROR_NO_SENS";
        case PYBAMM_ERROR_NO_OUTPUTS: return "ERROR_NO_OUTPUTS";
        case PYBAMM_ERROR_NO_ALG: return "ERROR_NO_ALG";
        case PYBAMM_ERROR_NO_EVENTS: return "ERROR_NO_EVENTS";
        case PYBAMM_ERROR_NO_UPSAMPLING: return "ERROR_NO_UPSAMPLING";
        default: return "unrecognised status";
    }
}

[[noreturn]] inline void throw_ffi_error(const char* name, int status) {
    throw std::runtime_error(
        std::string("pybammsolvers: Rust FFI call '") + name + "' failed with " +
        status_name(status) + " (" + std::to_string(status) +
        "). The output buffer may be unwritten, so the evaluation cannot be trusted.");
}

/* Call a status-returning entry point, throwing on failure: a caught Rust panic may leave the
 * output unwritten, and those stale values must not reach SUNDIALS. */
template <typename Fn, typename... Args>
inline void checked_call(const char* name, Fn fn, Args... args) {
    const int status = fn(args...);
    if (status != PYBAMM_SUCCESS) {
        throw_ffi_error(name, status);
    }
}

/* As above, for entry points returning a non-negative count. */
template <typename Fn, typename... Args>
inline int checked_value(const char* name, Fn fn, Args... args) {
    const int value = fn(args...);
    if (value < 0) {
        throw_ffi_error(name, value);
    }
    return value;
}

/*
 * Throw unless this build can call every entry of the table `handle` points to. Only the fields
 * every version leads with are read, byte-wise, since another version's layout may differ.
 */
inline void check_abi(const void* handle) {
    const unsigned char* table = nullptr;
    std::memcpy(static_cast<void*>(&table), handle, sizeof(table));
    uint32_t abi_version = 0;
    uint32_t size = 0;
    std::memcpy(&abi_version, table, sizeof(abi_version));
    std::memcpy(&size, table + sizeof(abi_version), sizeof(size));
    if (abi_version != PYBAMM_RUST_ABI_VERSION) {
        throw std::runtime_error(
            "pybammsolvers: this Rust evaluator has ABI version " + std::to_string(abi_version) +
            ", but pybammsolvers was built for version " +
            std::to_string(PYBAMM_RUST_ABI_VERSION) +
            ". Install pybamm and pybammsolvers releases that share a Rust ABI version.");
    }
    if (size < sizeof(PybammRustFfi)) {
        throw std::runtime_error(
            "pybammsolvers: this Rust evaluator's function table is " + std::to_string(size) +
            " bytes, older than the " + std::to_string(sizeof(PybammRustFfi)) +
            " bytes pybammsolvers reads. Upgrade the package that built the Rust core (pybamm).");
    }
}

}  // namespace pybamm_rust_detail

/* Releases an evaluator through the table of the build that made it. */
struct RustEvaluatorRelease {
    void operator()(PybammRustEvaluator* handle) const noexcept {
        handle->ffi->release(handle);
    }
};

/* An evaluator C++ owns. */
using RustEvaluatorPtr = std::unique_ptr<PybammRustEvaluator, RustEvaluatorRelease>;

/* An evaluator borrowed from its owner: the table to call, and the model every call takes. */
struct RustModelRef {
    const PybammRustFfi* ffi;
    void* model;
};

inline RustModelRef borrow(const RustEvaluatorPtr& evaluator) {
    return RustModelRef{evaluator->ffi, evaluator->model};
}

/* Checked call to `entry` of `rust`, a RustModelRef; see `checked_call`. */
#define PYBAMM_RUST_CALL(rust, entry, ...) \
    ::pybamm_rust_detail::checked_call(#entry, (rust).ffi->entry, __VA_ARGS__)

/* Checked read of the count-returning `entry` of `rust`. */
#define PYBAMM_RUST_VALUE(rust, entry, ...) \
    ::pybamm_rust_detail::checked_value(#entry, (rust).ffi->entry, __VA_ARGS__)

#endif /* __cplusplus */

#endif /* PYBAMM_RUST_FFI_H */
