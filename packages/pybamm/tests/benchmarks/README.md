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

- `@pytest.mark.slow_bench` — timing benchmarks CI skips, selected with
  `-m "time_bench and not slow_bench"`. **Temporary:** Bencher's free tier enforces a
  hard 5 minute job timeout, and on an `intel-v1` runner (used by Bencher's bare-metal setup) the full time benchmark suite exceeds this limit. The marker is currently carried by `test_model_options.py`
  and `test_setup_models_and_sims.py`, as they are the two large sweeps, ~70% of the runtime; the rest finishes in about half the budget. Drop the marker (and the `not slow_bench` selection) if PyBaMM moves to a paid tier, which lifts the timeout entirely. Local runs are unaffected: `nox -s benchmark-time` still runs every benchmark.

### CI

Timing benchmarks are tracked with [Bencher](https://bencher.dev) on [bare metal
hardware](https://bencher.dev/docs/explanation/bare-metal/). CI builds self-contained image (Bencher's runners have no network access), pushes it to Bencher's registry, and `bencher run --image` executes the suite on dedicated hardware.

- **`benchmarks_main.yml`** — on push to `main`. Builds, pushes, and runs the suite
  to record the baseline every PR is compared against.
- **`benchmarks_pr.yml`** — on PRs. Builds the image and uploads it as an artifact,
  and runs the memray memory benchmarks (which assert fixed limits, so they gain
  nothing from bare metal). Holds no secrets, because it runs fork code.
- **`benchmarks_track.yml`** — on `benchmarks_pr.yml` completing. Pushes the
  artifact image and reports results against the PR's base branch. Split out from `benchmarks_pr.yml` so the
  Bencher API key is never exposed to a fork's code.

#### Regression alerts on PRs

`benchmarks_main.yml` sets an upper threshold of `0.1`, so a benchmark alerts once it is more
than 10% slower than its historical mean. `benchmarks_track.yml` inherits that threshold
from main to raise alerts on PRs, posting a GitHub Check and a PR comment.

**Only the benchmarks CI runs are guarded.** Everything marked `slow_bench` is
  skipped, so regressions in the model-option and setup sweeps will currently pass unnoticed.
