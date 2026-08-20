# Building the Rust Backend

This guide covers building the Rust compute core, its Python bindings, and the
Rust-capable IDAKLU solver for development. Everything lives in this
repository's `uv` workspace; there are no side-by-side clones or environment
variables to configure.

## Components

| Component | Location | Build tool | Output |
|-----------|----------|------------|--------|
| `pybamm-core` | `packages/pybamm-rust/pybamm-core` | Cargo | Rust library (rlib + cdylib) |
| `pybamm-python` | `packages/pybamm-rust/pybamm-python` | maturin (via `uv sync`) | `pybamm.rust._core` extension module |
| pybammsolvers | `packages/pybammsolvers` | scikit-build-core / CMake | `idaklu` extension module |

**How the pieces connect:**

- `pybamm.rust` provides the Python API that lowers a discretised model to a
  `CompiledModel` and drives the diffsol solver.
- pybammsolvers does **not** link against the Rust core (the package
  dependency only flows the other way). The IDAKLU C++ consumer resolves the
  `extern "C"` entry points at runtime — `dlsym(RTLD_DEFAULT)` on POSIX, a
  loaded-module walk with `GetProcAddress` on Windows — once
  `pybamm.rust._core` is imported in-process.
- The FFI contract's source of truth is the `extern "C"` exports in
  `packages/pybamm-rust/pybamm-core/src/ffi.rs` (symbols carry a
  `pybamm_rust_` prefix). The C++ consumer mirrors them as a function-pointer
  table in
  `packages/pybammsolvers/src/pybammsolvers/idaklu_source/Expressions/Rust/pybamm_rust_ffi.h`.
  A version handshake (`pybamm_rust_abi_version()` /
  `PYBAMM_RUST_ABI_VERSION`) and the `test_ffi_abi_contract` drift test keep
  the two sides in lockstep.

## Prerequisites

- **Rust 1.89+**: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **uv**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **CMake 3.20+** and a C++17 compiler
- **macOS only**: Homebrew with `libomp` (`brew install libomp`)

## Quick start

One sync builds everything — the maturin hook compiles `pybamm.rust._core` in
release mode, and pybammsolvers builds SUNDIALS/SuiteSparse from its bundled
submodules on first build:

```bash
git clone https://github.com/pybamm-team/PyBaMM.git
cd PyBaMM
git submodule update --init --recursive
uv sync --extra all --group dev
```

Verify the two halves see each other:

```bash
uv run python -c "
from pybamm.rust import CompiledModel
from pybammsolvers import idaklu
print('Rust bindings:', CompiledModel)
print('Rust FFI available:', hasattr(idaklu, 'create_rust_solver_group'))
"
```

## Rebuilding after changes

`uv`'s cache keys cover the Rust sources, so a plain `uv sync` rebuilds
`pybamm.rust._core` whenever a `.rs` file changes.

pybammsolvers' cache key is only its `pyproject.toml`. After changing any of
its C++ sources — including `pybamm_rust_ffi.h` — force a rebuild:

```bash
uv cache clean pybammsolvers
uv sync --extra all --group dev --reinstall-package pybammsolvers
```

## Changing the FFI surface

The drift test parses both sides of the boundary, so the workflow is
mechanical:

1. Edit the exports in `ffi.rs` and the consumer table in
   `pybamm_rust_ffi.h` together.
2. Run `cargo test -p pybamm-core --test test_ffi_abi_contract` from
   `packages/pybamm-rust/`. On any change to the exported surface it fails
   and prints the new `EXPECTED_ABI_HASH`; update it in `ffi.rs` and bump
   `RUST_ABI_VERSION` / `PYBAMM_RUST_ABI_VERSION` in lockstep.
3. Rebuild both extensions (previous section). A stale pybammsolvers binary
   fails loudly at first use — either an unresolvable symbol or an ABI
   version mismatch — rather than corrupting an evaluation.

## Running tests

```bash
# Rust unit + contract tests
cd packages/pybamm-rust && cargo test --workspace

# Rust-IDAKLU integration tests (from the repo root)
uv run --group dev pytest packages/pybamm/tests/integration/test_rust_idaklu_parity.py
uv run --group dev pytest packages/pybamm/tests/integration/test_rust_idaklu_spm.py

# Diffsol solver unit tests
uv run --group dev pytest packages/pybamm/tests/unit/test_solvers/test_diffsol_solver.py
```

## Verifying FFI symbols

The entry points are exported by `pybamm.rust._core` and resolved from it at
runtime, so they should be *defined* there and appear nowhere in idaklu:

```bash
# macOS (Linux: nm -D --defined-only)
nm -gU packages/pybamm/src/pybamm/rust/_core.abi3.so | grep pybamm_rust_
```

Expected output lists symbols like `_pybamm_rust_eval_rhs` and
`_pybamm_rust_abi_version`. If a Rust-backed solve fails with an
unresolvable-symbol error instead, the loaded extension predates the current
FFI surface — rebuild both extensions from the same source tree.

## Running benchmarks

```bash
# Rust micro-benchmarks (Criterion)
cd packages/pybamm-rust/pybamm-core
cargo bench

# Python end-to-end observability benchmark (from the repo root)
uv run python benchmarks/run_rust_observability.py
```

## Troubleshooting

### Segfaults in SUNDIALS callbacks

Crashes during `solve()` (especially in residual/Jacobian callbacks) usually
indicate a **SuiteSparse version mismatch**: the `.idaklu/` build directory
holds a stale build while the submodules (or a Homebrew install picked up via
`@rpath`) moved on. Rebuild the native dependencies from the current
submodules:

```bash
cd packages/pybammsolvers
rm -rf .idaklu
uv run python install_KLU_Sundials.py
cd ../.. && uv sync --extra all --group dev --reinstall-package pybammsolvers
```

Do **not** point the build at Homebrew's suite-sparse; version mismatches
between it and the local SUNDIALS build cause runtime crashes.

### `pybamm.rust._core` import fails

```bash
uv sync --extra all --group dev --reinstall-package pybamm
```

### Other symbol errors

An ABI mismatch between components — rebuild both extensions from the same
tree (see "Rebuilding after changes").

## Supported features

Feature coverage (events, DAEs, forward sensitivities, output variables,
experiments) and the expression types the backend can lower are documented in
the [Rust backend user guide](../user_guide/rust_backend.rst).
