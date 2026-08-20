#ifndef PYBAMM_RUST_FFI_H
#define PYBAMM_RUST_FFI_H

#include <stdint.h>

/* Return codes */
#define PYBAMM_SUCCESS              0
#define PYBAMM_ERROR_NULL          -1
#define PYBAMM_ERROR_PANIC         -2
#define PYBAMM_ERROR_INVALID_PARAM -3
#define PYBAMM_ERROR_INVALID_OUTPUT -4
#define PYBAMM_ERROR_BUFFER_SMALL  -5
#define PYBAMM_ERROR_NO_SENS       -6
#define PYBAMM_ERROR_NO_OUTPUTS    -7
#define PYBAMM_ERROR_NO_ALG        -8
#define PYBAMM_ERROR_NO_EVENTS     -9

/* ABI contract version. Must equal the Rust core's RUST_ABI_VERSION; the
 * rust_ffi() resolver throws on mismatch. Bump in lockstep with ffi.rs. */
#define PYBAMM_RUST_ABI_VERSION 1

#ifdef __cplusplus

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
// PSAPI_VERSION 2 binds EnumProcessModules to its kernel32 export
// (K32EnumProcessModules), so no psapi.lib import library is needed.
#ifndef PSAPI_VERSION
#define PSAPI_VERSION 2
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <dlfcn.h>
#endif

#include <stdexcept>
#include <string>
#include <vector>

/*
 * Rust FFI access — runtime symbol resolution (host-plugin model).
 *
 * pybammsolvers cannot link against PyBaMM's Rust core: PyBaMM depends on
 * pybammsolvers, so the dependency only flows one way, and the Rust FFI entry
 * points are compiled into PyBaMM's `pybamm.rust._core` Python extension rather
 * than a standalone shared library. We therefore do NOT declare these functions as
 * `extern "C"` (which would leave undefined symbols in idaklu and break loading
 * the module unless the extension happened to be loaded first). Instead we
 * resolve them at runtime via `find_symbol` below (`dlsym(RTLD_DEFAULT, ...)`
 * on POSIX, a loaded-module walk on Windows).
 *
 * By the time a Rust-backed solver group is constructed, `pybamm.rust._core` is
 * already imported in-process (the caller hands us a live CompiledModel), so
 * its exported symbols are visible to the dynamic linker. This keeps idaklu
 * free of undefined symbols — it loads cleanly standalone (the CasADi path is
 * unaffected) and ships in release wheels, with the Rust path resolved lazily
 * the first time it is used.
 *
 * Note (Linux): symbols are only visible to `dlsym(RTLD_DEFAULT, ...)` if the
 * extension was loaded with global visibility. CPython defaults to RTLD_LOCAL
 * on Linux, so `pybamm/rust/__init__.py` adds RTLD_GLOBAL around the import
 * there (`_global_symbol_visibility`). On macOS the symbols are globally
 * visible by default.
 *
 * Note (Windows): there is no process-global symbol namespace, so
 * `find_symbol` walks the loaded-module list with `GetProcAddress` instead.
 * The Rust extension is a cdylib, whose `#[no_mangle]` entry points sit in
 * its DLL export table, so the walk finds them once the module is loaded.
 */

/*
 * Function-pointer typedefs for the Rust `extern "C"` entry points IDAKLU
 * calls. This is deliberately a subset of `ffi.rs`, not a mirror of it: every
 * name here is resolved eagerly and a missing symbol aborts the whole Rust
 * path, so binding an entry point nothing calls would couple IDAKLU to Rust
 * exports it does not need. Model metadata (state/param/output counts,
 * sparsity, algebraic ids) reaches C++ as explicit `create_rust_solver_group`
 * arguments, read on the Python side through pyo3.
 */
using rust_eval_rhs_t = int (*)(double, const double*, const double*, double*, void*);
using rust_jac_assemble_t = int (*)(double, const double*, const double*, double, double*, void*);
using rust_jac_action_t = int (*)(double, const double*, const double*, const double*, double*, void*);
using rust_mass_action_t = int (*)(const double*, double*, void*);
using rust_sens_eval_all_t = int (*)(double, const double*, const double*, double*, void*);
using rust_output_sens_project_t =
    int (*)(double, const double*, const double*, const double*, double*, void*);
using rust_output_eval_t = int (*)(double, const double*, const double*, int, double*, int*, void*);
using rust_output_eval_batch_t =
    int (*)(const double*, const double*, int, const double*, double*, void*);
using rust_alg_res_t = int (*)(double, const double*, const double*, double*, void*);
using rust_alg_jac_assemble_t = int (*)(double, const double*, const double*, double*, void*);
using rust_total_event_len_t = int (*)(const void*);
using rust_events_eval_t = int (*)(double, const double*, const double*, double*, void*);
using rust_abi_version_t = uint32_t (*)();

/* Resolved table of Rust FFI entry points. */
struct RustFfi {
    rust_eval_rhs_t eval_rhs;
    rust_jac_assemble_t jac_assemble;
    rust_jac_action_t jac_action;
    rust_mass_action_t mass_action;
    rust_sens_eval_all_t sens_eval_all;
    rust_output_sens_project_t output_sens_project;
    rust_output_eval_t output_eval;
    rust_output_eval_batch_t output_eval_batch;
    rust_alg_res_t alg_res;
    rust_alg_jac_assemble_t alg_jac_assemble;
    rust_total_event_len_t total_event_len;
    rust_events_eval_t events_eval;
    rust_abi_version_t abi_version;
};

namespace pybamm_rust_detail {

/* Address of an exported symbol in any module loaded in this process, or
 * nullptr. POSIX has this as `dlsym(RTLD_DEFAULT, ...)`; Windows has no
 * process-global lookup, so every loaded module's export table is tried in
 * turn (`K32EnumProcessModules` is the kernel32 alias of the psapi function,
 * so no extra import library is linked). */
inline void* find_symbol(const char* name) {
#if defined(_WIN32)
    HANDLE process = GetCurrentProcess();
    DWORD bytes_needed = 0;
    if (K32EnumProcessModules(process, nullptr, 0, &bytes_needed) == 0) {
        return nullptr;
    }
    std::vector<HMODULE> modules(bytes_needed / sizeof(HMODULE));
    if (K32EnumProcessModules(process, modules.data(), bytes_needed, &bytes_needed) == 0) {
        return nullptr;
    }
    // The module list can shrink between the size query and the fill.
    modules.resize(bytes_needed / sizeof(HMODULE));
    for (HMODULE module : modules) {
        if (FARPROC sym = GetProcAddress(module, name)) {
            return reinterpret_cast<void*>(sym);
        }
    }
    return nullptr;
#else
    return dlsym(RTLD_DEFAULT, name);
#endif
}

template <typename Fn>
inline Fn load_symbol(const char* name) {
    void* sym = find_symbol(name);
    if (sym == nullptr) {
        throw std::runtime_error(
            std::string("pybammsolvers: Rust FFI symbol '") + name +
            "' could not be resolved. Import `pybamm.rust` before constructing a "
            "Rust-backed IDAKLU solver: it loads pybamm.rust._core and, on Linux, "
            "adds RTLD_GLOBAL so these symbols reach dlsym(RTLD_DEFAULT).");
    }
    return reinterpret_cast<Fn>(sym);
}

/* Spelling of a Rust FFI status code, for error messages. */
inline const char* status_name(int status) {
    switch (status) {
        case PYBAMM_SUCCESS: return "SUCCESS";
        case PYBAMM_ERROR_NULL: return "ERROR_NULL";
        case PYBAMM_ERROR_PANIC: return "ERROR_PANIC";
        case PYBAMM_ERROR_INVALID_PARAM: return "ERROR_INVALID_PARAM";
        case PYBAMM_ERROR_INVALID_OUTPUT: return "ERROR_INVALID_OUTPUT";
        case PYBAMM_ERROR_BUFFER_SMALL: return "ERROR_BUFFER_SMALL";
        case PYBAMM_ERROR_NO_SENS: return "ERROR_NO_SENS";
        case PYBAMM_ERROR_NO_OUTPUTS: return "ERROR_NO_OUTPUTS";
        case PYBAMM_ERROR_NO_ALG: return "ERROR_NO_ALG";
        case PYBAMM_ERROR_NO_EVENTS: return "ERROR_NO_EVENTS";
        default: return "unrecognised status";
    }
}

[[noreturn]] inline void throw_ffi_error(const char* name, int status) {
    throw std::runtime_error(
        std::string("pybammsolvers: Rust FFI call '") + name + "' failed with " +
        status_name(status) + " (" + std::to_string(status) +
        "). The output buffer may be unwritten, so the evaluation cannot be trusted.");
}

/*
 * Call a status-returning entry point, throwing unless it reports success.
 *
 * A panic caught at the Rust boundary yields ERROR_PANIC with the output buffer
 * potentially unwritten, so dropping the status would feed the previous step's
 * stale values to SUNDIALS as a valid evaluation. Throwing matches the CasADi
 * path, whose evaluation errors reach IDAKLUSolverGroup's handler the same way.
 */
template <typename Fn, typename... Args>
inline void checked_call(const char* name, Fn fn, Args... args) {
    const int status = fn(args...);
    if (status != PYBAMM_SUCCESS) {
        throw_ffi_error(name, status);
    }
}

/* As above for entry points returning a non-negative count rather than a status. */
template <typename Fn, typename... Args>
inline int checked_value(const char* name, Fn fn, Args... args) {
    const int value = fn(args...);
    if (value < 0) {
        throw_ffi_error(name, value);
    }
    return value;
}

}  // namespace pybamm_rust_detail

/* Checked call to `RustFfi::entry`; see `pybamm_rust_detail::checked_call`. */
#define PYBAMM_RUST_CALL(entry, ...) \
    ::pybamm_rust_detail::checked_call(#entry, ::rust_ffi().entry, __VA_ARGS__)

/* Checked read of a count-returning `RustFfi::entry`. */
#define PYBAMM_RUST_VALUE(entry, ...) \
    ::pybamm_rust_detail::checked_value(#entry, ::rust_ffi().entry, __VA_ARGS__)

/*
 * Resolve and cache the Rust FFI table. The first call performs the dlsym
 * lookups (thread-safe via the function-local static); subsequent calls return
 * the cached table, so evaluation hot paths only pay an indirect call.
 */
inline const RustFfi& rust_ffi() {
    using pybamm_rust_detail::load_symbol;
    static const RustFfi table = [] {
        RustFfi t{};
        t.eval_rhs = load_symbol<rust_eval_rhs_t>("pybamm_rust_eval_rhs");
        t.jac_assemble = load_symbol<rust_jac_assemble_t>("pybamm_rust_jac_assemble");
        t.jac_action = load_symbol<rust_jac_action_t>("pybamm_rust_jac_action");
        t.mass_action = load_symbol<rust_mass_action_t>("pybamm_rust_mass_action");
        t.sens_eval_all = load_symbol<rust_sens_eval_all_t>("pybamm_rust_sens_eval_all");
        t.output_sens_project =
            load_symbol<rust_output_sens_project_t>("pybamm_rust_output_sens_project");
        t.output_eval = load_symbol<rust_output_eval_t>("pybamm_rust_output_eval");
        t.output_eval_batch =
            load_symbol<rust_output_eval_batch_t>("pybamm_rust_output_eval_batch");
        t.alg_res = load_symbol<rust_alg_res_t>("pybamm_rust_alg_res");
        t.alg_jac_assemble = load_symbol<rust_alg_jac_assemble_t>("pybamm_rust_alg_jac_assemble");
        t.total_event_len = load_symbol<rust_total_event_len_t>("pybamm_rust_total_event_len");
        t.events_eval = load_symbol<rust_events_eval_t>("pybamm_rust_events_eval");
        t.abi_version = load_symbol<rust_abi_version_t>("pybamm_rust_abi_version");
        if (t.abi_version() != PYBAMM_RUST_ABI_VERSION) {
            throw std::runtime_error(
                std::string("pybammsolvers: Rust FFI ABI version mismatch. "
                            "pybammsolvers was built for version ") +
                std::to_string(PYBAMM_RUST_ABI_VERSION) +
                " but the loaded PyBaMM Rust core reports version " +
                std::to_string(t.abi_version()) +
                ". Rebuild pybammsolvers and the PyBaMM Rust extension from the "
                "same source tree.");
        }
        return t;
    }();
    return table;
}

#endif /* __cplusplus */

#endif /* PYBAMM_RUST_FFI_H */
