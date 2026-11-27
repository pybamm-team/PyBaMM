# Building the Rust Backend

This guide covers building the Rust compute core, its Python bindings, and the
Rust-capable IDAKLU solver for development. Everything lives in this
repository's `uv` workspace; there are no side-by-side clones or environment
variables to configure.

## Components

| Component | Location | Build tool | Output |
|-----------|----------|------------|--------|
| `pybamm-core` | `packages/pybamm-rust/pybamm-core` | Cargo | Rust library (rlib) |
| `pybamm-python` | `packages/pybamm-rust/pybamm-python` | maturin (via `uv sync`) | `pybamm.rust._core` extension module |
| pybammsolvers | `packages/pybammsolvers` | scikit-build-core / CMake | `idaklu` extension module |

**How the pieces connect:**

- `pybamm-python` implements `pybamm.rust`, which provides the Python API over
  the Rust code in `pybamm-core`, which compiles expressions into evaluable
  functions and sparse Jacobians.
- `pybamm.rust` also lowers a discretised model to a `CompiledModel`, which the
  IDAKLU solver takes, and drives the diffsol solver.
- pybammsolvers does **not** link against the Rust core (the package
  dependency only flows the other way). Instead, `CompiledModel.evaluator()`
  hands IDAKLU each evaluator in a Python capsule named
  `pybamm.rust.evaluator`. The capsule holds the evaluator and a pointer to the
  function table of the build that made it, and IDAKLU calls only through that
  table. Nothing is resolved by symbol name, so `pybamm.rust._core` exports no
  FFI symbols.
- IDAKLU claims each capsule by renaming it `pybamm.rust.evaluator.claimed`,
  and then owns the evaluator: each solver frees its own, and a claimed
  evaluator cannot be handed to a second solver.
- The contract's source of truth is `FfiTable` and `EvaluatorHandle` in
  `packages/pybamm-rust/pybamm-core/src/ffi.rs`. The C++ consumer mirrors them
  in
  `packages/pybammsolvers/src/pybammsolvers/idaklu_source/Expressions/Rust/pybamm_rust_ffi.h`,
  and the `test_ffi_abi_contract` drift test compares the two field by field.
- The table leads with its ABI version and its size. IDAKLU refuses an
  evaluator of another version, or one whose table is shorter than the one it
  reads, before calling anything.
- Rust reads and writes the C++ buffers at the Rust model's own sizes, so
  `create_rust_solver_group` checks every count it sizes a buffer from
  (states, inputs, Jacobian non-zeros, sensitivity parameters, outputs and
  events) against the model, raising `ValueError` on a mismatch.

## Prerequisites

- **Rust 1.89+**: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **uv**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **CMake 3.22+** (the bundled SuiteSparse's floor) and a C++17 compiler
- **BLAS and a Fortran compiler** for SuiteSparse, e.g. `libopenblas-dev gfortran`
- **macOS only**: Homebrew with `libomp` (`brew install libomp`)

## Quick start

One sync builds everything — the hatch build hook compiles `pybamm.rust._core`
in release mode, and pybammsolvers builds SUNDIALS/SuiteSparse from its bundled
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

`uv`'s cache keys cover the Rust sources, so `uv sync --extra all --group dev`
or `uv run` rebuilds `pybamm.rust._core` whenever a `.rs` file changes.

pybammsolvers' cache key is only its `pyproject.toml`. After changing any of
its C++ sources — including `pybamm_rust_ffi.h` — force a rebuild:

```bash
uv cache clean pybammsolvers
uv sync --extra all --group dev --reinstall-package pybammsolvers
```

## Changing the FFI surface

The drift test parses both sides of the boundary, so the workflow is
mechanical:

1. Edit `FfiTable` and `FFI_TABLE` in `ffi.rs` and `PybammRustFfi` in
   `pybamm_rust_ffi.h` together. The drift test fails on any field the two
   tables do not share, in order and type, on an `FFI_TABLE` entry wired to
   another entry's function, and on an entry idaklu never calls.
2. Run `cargo test -p pybamm-core --test test_ffi_abi_contract` from
   `packages/pybamm-rust/`. The test holds the layout the current ABI version
   froze:
   - **Appending** table entries passes it unchanged. A pybammsolvers built
     against the shorter table reads only the entries it knows, so it keeps
     working with the newer core.
   - **Any other change** (an entry's signature or meaning, removing or
     reordering entries, or the handle's layout) fails it. Bump
     `RUST_ABI_VERSION` and `PYBAMM_RUST_ABI_VERSION` together; the test then
     prints the new layout to freeze in its place.
3. Rebuild both extensions (previous section). A pybammsolvers built for
   another ABI version refuses the evaluator, naming both versions, rather
   than corrupting an evaluation.

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

## Checking the ABI version

`idaklu.rust_abi_version` is the version a pybammsolvers build reads; it must
equal `RUST_ABI_VERSION` in the `ffi.rs` that built `pybamm.rust._core`:

```bash
uv run python -c "from pybammsolvers import idaklu; print(idaklu.rust_abi_version)"
```

## Running benchmarks

```bash
# Rust micro-benchmarks (Criterion); --all-features runs those that need `serialize`
cd packages/pybamm-rust && cargo bench -p pybamm-core --all-features
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

### Rust ABI version errors

pybammsolvers and `pybamm.rust._core` were built for different ABI versions,
or the core's table is older than the one pybammsolvers reads. Rebuild both
extensions from the same tree (see "Rebuilding after changes").

## Supported features

Feature coverage (events, DAEs, forward sensitivities, output variables,
experiments) and the expression types the backend can lower are documented in
the [Rust backend user guide](../user_guide/rust_backend.rst).
