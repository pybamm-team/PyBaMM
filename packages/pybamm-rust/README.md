# PyBaMM Rust Core

Expression compiler and evaluator for PyBaMM models, filling the role CasADi plays on
PyBaMM's other backends. Python hands over a discretised model as an expression DAG;
this workspace turns it into flat instruction tapes it can evaluate, differentiate and
solve.

Models opt in with `model.convert_to_format = "rust"`.

## Layout

Two crates:

- **`pybamm-core`** — the compiler and interpreter. No Python dependency, so it can be
  tested and benchmarked on its own.
- **`pybamm-python`** — PyO3 bindings, built as the `_core` extension and re-exported
  through `pybamm.rust`. Owns the boundary checks, so core code can assume array lengths
  and index ranges are already valid.

## How a model becomes executable

Five stages, each a module in `pybamm-core`:

1. **Build** — the bindings allocate nodes into an arena. A node is referenced by a
   small integer id, so sharing a subexpression is repeating an id rather than cloning a
   subtree.
2. **Rewrite** — constant folding and algebraic identities, CSE and DCE, plus a pass
   that proves subtrees identically zero and folds them away.
3. **Differentiate** — forward mode emits a tangent DAG for JVPs; reverse mode fills a
   single wide Jacobian row from one backward pass.
4. **Lower** — the DAG flattens to fixed-size instructions addressing slots in one
   scratch buffer, with constants held in a side table.
5. **Evaluate** — the interpreter walks a tape against a caller-supplied buffer, either
   one time point at a time or several lanes at once.

Assembling a sparse Jacobian is where the design earns its keep: a column coloring
groups columns that never share a row, so assembly costs one primal pass whose result
every color reuses, plus one tangent sweep per color. Rows too wide to color cheaply are
split out and filled by reverse mode instead, so a single dense row cannot force one
color per column.

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
cargo clippy --all-targets -- -D warnings
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
uv run python benchmarks/run_rust_observability.py --lane all   # from repo root
```

Pass `--all-features`, or cargo silently skips the benches whose `required-features` are
unmet rather than telling you they did not run.

The cargo benches measure the compiler and interpreter in isolation. The observability
harness is the one to trust for backend comparisons: it runs the same scenarios across
Rust, CasADi and CasADi's ahead-of-time path, shuffles backend order to keep machine
warm-up from flattering whichever runs last, and reports agreement against a baseline
rather than raw timings alone. Quote numbers from a run on the machine in question, not
from documentation.

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
  backend; `serialize` enables snapshot support that some tests and benches require;
  `profile` adds FFI call counters. A `required-features` error means the target needs
  one of these.
