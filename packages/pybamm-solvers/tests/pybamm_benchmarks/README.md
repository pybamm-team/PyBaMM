# Scripts

This directory contains utility scripts for development and testing.

## `run_benchmarks.py`

Orchestrates performance benchmarking to compare vanilla PyBaMM against pybammsolvers.

### Usage

Run via nox (recommended):
```bash
nox -s benchmarks
```

Or directly:
```bash
python scripts/run_benchmarks.py
```

### How It Works

1. **Baseline Run**: Uninstalls local pybammsolvers and runs benchmarks with vanilla PyBaMM
2. **Local Install**: Installs the local pybammsolvers package
3. **Current Run**: Re-runs benchmarks with pybammsolvers installed
4. **Comparison**: Compares results and reports any performance regressions
5. **Results**: Saves comparison to `performance_results.json`

### Output

The script will:
- Print benchmark times for both runs
- Compare performance between vanilla and pybammsolvers
- Flag regressions >20% as failures
- Save detailed results to `performance_results.json`

### Exit Codes

- `0`: Success (no significant regressions)
- `1`: Failure (benchmarks failed or >20% regression detected)

### Example Output

```
[1/4] Running baseline benchmarks with vanilla PyBaMM...
SPM 1-hour discharge:
  Average: 2.380s

[2/4] Installing local pybammsolvers...
Local pybammsolvers installed

[3/4] Running benchmarks with local pybammsolvers...
SPM 1-hour discharge:
  Average: 2.290s

[4/4] Comparing results...
SPM 1-hour discharge:
  Baseline: 2.380s
  Current:  2.290s
  Change:   0.090s (3.8% faster)

SUMMARY
============================================================
No significant regressions detected
```

