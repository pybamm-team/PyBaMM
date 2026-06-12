## Benchmarks

This directory contains the benchmark suite for PyBaMM, using [pytest-benchmark](https://pytest-benchmark.readthedocs.io/).

### Running benchmarks locally

Run the fast subset (same set used in PR CI):

```shell
nox -s benchmark-fast
```

Run the full suite including all parameter combinations:

```shell
nox -s benchmark-all
```

### Comparing against a baseline

To detect regressions between two states of the code:

```shell
# Save baseline results on one branch/commit
nox -s benchmark-fast -- --benchmark-save=baseline.json

# Switch to another branch/commit, then compare
nox -s benchmark-fast -- --benchmark-compare=baseline --benchmark-compare-fail=mean:125%
```

`--benchmark-compare-fail=mean:125%` exits with an error if any benchmark is more than 25% slower than the baseline.

### The `slow_bench` marker

Benchmarks parametrised over the full combination grid are marked `@pytest.mark.slow_bench`. These are excluded from PR CI (which uses `-m "not slow_bench"`) but included in the full history run that triggers on every push to `main`.

To add a benchmark that should run in PR CI, leave it unmarked. To add a comprehensive sweep, mark it `@pytest.mark.slow_bench`.

### CI

- **`benchmarks_history.yml`** — triggers on push to `main`. Runs the full suite and publishes results to the `gh-pages` branch for time-series visualization via [github-action-benchmark](https://github.com/benchmark-action/github-action-benchmark).
