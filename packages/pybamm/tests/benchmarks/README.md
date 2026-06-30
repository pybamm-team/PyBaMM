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

### CI

- **`benchmarks_pr.yml`** — triggers on PRs. Runs timing benchmarks against main HEAD (fails on >25% regression) and memory benchmarks with memray (fails if `limit_memory` thresholds are exceeded).
