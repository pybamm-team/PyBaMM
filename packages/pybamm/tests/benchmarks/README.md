## Benchmarks

This directory contains the benchmark suite for PyBaMM, using [pytest-benchmark](https://pytest-benchmark.readthedocs.io/).

### Running benchmarks locally

Run speed benchmarks (all non-memory benchmarks):

```shell
nox -s benchmark-speed
```

Run memory benchmarks with memray (Linux/macOS only):

```shell
nox -s benchmark-memory
```

### Comparing speed benchmarks against a baseline

To detect regressions between two states of the code:

```shell
# Save baseline results on one branch/commit
nox -s benchmark-speed -- --benchmark-save=baseline

# Switch to another branch/commit, then compare
nox -s benchmark-speed -- --benchmark-compare=baseline --benchmark-compare-fail=mean:125%
```

`--benchmark-compare-fail=mean:125%` exits with an error if any benchmark is more than 25% slower than the baseline.

### Markers

- `@pytest.mark.memory_bench` — memory benchmarks (run via `benchmark-memory`).

### CI

- **`benchmarks_pr.yml`** — triggers on PRs touching `packages/pybamm/`. Runs speed benchmarks against main HEAD (fails on >25% regression) and memory benchmarks with memray (fails if `limit_memory` thresholds are exceeded).
- **`benchmarks_history.yml`** — triggers on push to `main`. Runs the full suite and publishes results to the `gh-pages` branch for time-series visualization via [github-action-benchmark](https://github.com/benchmark-action/github-action-benchmark).
