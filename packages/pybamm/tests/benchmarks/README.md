## Benchmarks

This directory contains the benchmark suite for PyBaMM, using [pytest-benchmark](https://pytest-benchmark.readthedocs.io/) and [pytest-memray](https://pytest-memray.readthedocs.io/).

### Running benchmarks locally

Run timing benchmarks:

```shell
nox -s benchmark-time
```

Run memory benchmarks (Linux/macOS only):

```shell
nox -s benchmark-memory
```

The session passes `--trace-python-allocators`, so memray records each small-object
allocation instead of whole 1 MiB pymalloc arenas; the `limit_memory` values assume it.
Pass the same flag when running the memory benchmarks through `pytest` directly.

### Comparing timing benchmarks against a baseline

To detect regressions between two states of the code:

```shell
# Save baseline results on one branch/commit
nox -s benchmark-time -- --benchmark-save=baseline

# Switch to another branch/commit, then compare
nox -s benchmark-time -- --benchmark-compare=baseline --benchmark-compare-fail=mean:125%
```

`--benchmark-compare-fail=mean:125%` exits with an error if any benchmark is more than 25% slower than the baseline.

### Markers

Benchmarks should be marked as either time or memory tests so they can be grouped correctly. This can either be done at a whole file level using e.g.
```python
pytestmark = pytest.mark.memory_bench
```

Or individual tests can be marked

- `@pytest.mark.time_bench` — timing benchmarks (run via `benchmark-time`).

- `@pytest.mark.memory_bench` — memory benchmarks (run via `benchmark-memory`).

- `@pytest.mark.slow_bench` — the large sweeps, `test_model_options.py`,
  `test_setup_models_and_sims.py` and the extra parameter sets in `test_solve_models.py`.
  Each job takes about 3 minutes on Bencher's `intel-v1` runner, too long to share a 5
  minute job with the core suite, so CI runs them as separate jobs on a weekday schedule.
  `nox -s benchmark-time` still runs every benchmark.

### Measures

`--bencher-json PATH` (with `--benchmark-json`) writes results in [Bencher Metric
Format](https://bencher.dev/docs/reference/bencher-metric-format/). Besides `latency`:

- `scaling-exponent` — `p` in `time ~ size**p` for each `record_scaling` series.
- `import-modules` — modules loaded by `import pybamm`.

### CI

Timing benchmarks are tracked with [Bencher](https://bencher.dev) on [bare metal
hardware](https://bencher.dev/docs/explanation/bare-metal/). CI builds self-contained image (Bencher's runners have no network access), pushes it to Bencher's registry, and `bencher run --image` executes the suite on dedicated hardware.

Bencher's free plan caps each job at 5 minutes; the core suite takes about 3.

- **`benchmarks_main.yml`** — on push to `main`, builds, pushes, and runs the core suite
  to record the baseline every PR is compared against. On a weekday schedule it runs the
  `slow_bench` sweeps instead.
- **`benchmarks_pr.yml`** — on non-draft PRs that touch pybamm's source, the benchmarks,
  the solver or the dependencies. Builds the image and uploads it as an artifact,
  and runs the memray memory benchmarks (which assert fixed limits, so they gain
  nothing from bare metal). Holds no secrets, because it runs fork code.
- **`benchmarks_track.yml`** — on `benchmarks_pr.yml` completing. Pushes the
  artifact image and reports results against the PR's base branch. Split out from `benchmarks_pr.yml` so the
  Bencher API key is never exposed to a fork's code.

#### Registry bandwidth

Bencher implements an image bandwidth quota, so the image is kept small to fit it:

- It installs only pybamm's runtime dependencies and the `bench` group, with
  zstd-compressed layers (about 210 MiB).
- pybamm's source is the last layer and the layers below it are cached, so a commit that
  leaves `uv.lock` and the solver alone uploads about 2 MiB.

Add new benchmark-only dependencies to the `bench` group, not `dev`, or the image can't
import them.

#### Regression alerts on PRs

`benchmarks_main.yml` alerts when latency is more than 10%, a scaling exponent 25%, or
the module count 2% above its historical mean. `benchmarks_track.yml` inherits these
thresholds from main to raise alerts on PRs, posting a GitHub Check and a PR comment.

**PRs only check the core suite.** Regressions in the `slow_bench` sweeps alert on the
next scheduled `main` run.
