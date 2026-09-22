# PyBaMM Rust Core

Expression compiler and evaluator for PyBaMM models, filling the role CasADi plays on
PyBaMM's other backends. Python hands over a discretised model as an expression DAG;
this workspace turns it into flat instruction tapes it can evaluate, differentiate and
solve.

Models use it by default; set `model.convert_to_format = "casadi"` to opt out.

## Layout

Two crates:

- **`pybamm-core`** — the compiler and interpreter. No Python dependency, so it can be
  tested and benchmarked on its own.
- **`pybamm-python`** — PyO3 bindings, built as the `_core` extension and re-exported
  through `pybamm.rust`. The bindings check Python-side shapes and ownership; core
  validates what it is handed and returns `CoreError`.

## How a model becomes executable

A model passes through five stages, each a module in `pybamm-core`: build, rewrite,
differentiate, lower and evaluate. The crate-level doc walks through them, links each
module, and explains how the Jacobian is assembled (see
[Finding your way around](#finding-your-way-around)).

## Who consumes it

Two paths, and they differ in how they reach the crate:

- **diffsol**, in-process. The `solver` module implements one operator trait per
  callback diffsol needs and runs the integration entirely in Rust.
- **IDAKLU**, through the C ABI in `ffi.rs`. `pybammsolvers` cannot link against this
  crate, because PyBaMM depends on `pybammsolvers` and not the reverse. So the FFI entry
  points are compiled into the Python extension, and the C++ side resolves them at
  runtime with `dlsym` once that extension is loaded. IDAKLU therefore has no undefined
  symbols and loads standalone, with the Rust path resolved lazily on first use. The two
  sides agree on an ABI version and refuse to proceed on a mismatch; bumping the Rust
  constant means bumping the C++ one in the same change.

## Building

Build through the workspace, not `cargo` or `maturin` directly:

```bash
uv sync --extra all --group dev
```

The pybamm package declares `cache-keys` over both crates' sources and manifests, so
`uv sync` and `uv run` rebuild the extension whenever Rust code changes. Reaching for
maturin by hand produces a wheel the Python side will not pick up.

For work that stays inside Rust, the usual cargo commands apply from this directory:

```bash
cargo build --release
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```

## Testing

```bash
cargo test                                                # rust unit + integration
cargo test --all-features                                 # includes feature-gated tests
uv run --group dev pytest -m unit packages/pybamm/tests    # python side, from repo root
```

Rust and Python tests cover different things. The Rust suite owns the compiler and
interpreter, including property tests that check AD against finite differences and the
split primal/tangent tape against a monolithic one. The Python suite owns parity: the
same model solved through Rust and through CasADi must agree.

## Benchmarking

```bash
cargo bench -p pybamm-core --all-features
```

Pass `--all-features`, or cargo silently skips the benches whose `required-features` are
unmet rather than telling you they did not run.

The cargo benches measure the compiler and interpreter in isolation. Quote numbers from a
run on the machine in question, not from documentation.

## Finding your way around

Every module carries a doc comment explaining what it owns and why it exists:

```bash
cargo doc --no-deps --open
```

Start at the crate root for the pipeline overview, then read the module you need. The
doc comments are the reference; this README only orients you.

## Conventions

- **Lints are strict.** The workspace turns on clippy's pedantic and nursery groups and
  denies correctness. Fix warnings at the source; reach for a scoped `allow` only for a
  confirmed false positive, with a comment saying why.
- **MSRV and dependency versions** live in the workspace `Cargo.toml`. The Rust floor is
  set by the dependency graph rather than by the edition, so check there before assuming
  a newer language feature is available.
- **Feature flags**: `diffsol` (on by default) pulls in the integrator and its matrix
  backend; `serialize` enables snapshot support that some tests and benches require. A
  `required-features` error means the target needs one of these.
