# Building the Rust Backend

This guide covers building the Rust compute core and its Python bindings for
development. Everything lives in this repository's `uv` workspace; there are
no side-by-side clones or environment variables to configure.

## Components

| Component | Location | Build tool | Output |
|-----------|----------|------------|--------|
| `pybamm-core` | `packages/pybamm-rust/pybamm-core` | Cargo | Rust library (rlib) |
| `pybamm-python` | `packages/pybamm-rust/pybamm-python` | maturin (via `uv sync`) | `pybamm.rust._core` extension module |

**How the pieces connect:**

- `pybamm.rust` provides the Python API that compiles expressions into
  evaluable functions and sparse Jacobians.

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

Verify the extension imports:

```bash
uv run python -c "
from pybamm.rust import ExprGraph
print('Rust bindings:', ExprGraph)
"
```

## Rebuilding after changes

`uv`'s cache keys cover the Rust sources, so a plain `uv sync` rebuilds
`pybamm.rust._core` whenever a `.rs` file changes.

## Running tests

```bash
# Rust unit + contract tests
cd packages/pybamm-rust && cargo test --workspace
```

## Running benchmarks

```bash
# Rust micro-benchmarks (Criterion)
cd packages/pybamm-rust/pybamm-core
cargo bench
```

## Troubleshooting

### `pybamm.rust._core` import fails

```bash
uv sync --extra all --group dev --reinstall-package pybamm
```
