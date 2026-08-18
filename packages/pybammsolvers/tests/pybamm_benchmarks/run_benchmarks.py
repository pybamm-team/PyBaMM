#!/usr/bin/env python3
"""Orchestrate performance benchmarks comparing vanilla PyBaMM vs pybammsolvers.

This script:
1. Runs benchmarks with vanilla PyBaMM (baseline)
2. Installs local pybammsolvers
3. Runs benchmarks again with pybammsolvers
4. Compares results and reports any regressions
"""

import sys
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime


def run_command(cmd, cwd=None, env=None):
    """Run a command and return the result."""
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    return result


def run_benchmark_suite(output_file, description):
    """Run the benchmark suite and save results."""
    print(f"\n{'=' * 60}")
    print(f"Running benchmarks: {description}")
    print(f"{'=' * 60}\n")

    # Set environment variable for output file
    env = os.environ.copy()
    env["BENCHMARK_OUTPUT"] = str(output_file)

    # Run pytest with benchmark marker
    result = run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/pybamm_benchmarks",
            "-m",
            "benchmark",
            "-v",
            "--tb=short",
        ],
        env=env,
    )

    if result.returncode != 0:
        print("Benchmark run failed!")
        print(result.stdout)
        print(result.stderr)
        return False

    print(result.stdout)
    return True


def compare_results(baseline_file, current_file):
    """Compare baseline and current benchmark results."""
    with open(baseline_file) as f:
        baseline = json.load(f)

    with open(current_file) as f:
        current = json.load(f)

    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON: Vanilla PyBaMM vs pybammsolvers")
    print("=" * 60)

    print(f"\nBaseline: PyBaMM {baseline.get('pybamm_version', 'unknown')}")
    print(
        f"Current:  PyBaMM {current.get('pybamm_version', 'unknown')} + pybammsolvers {current.get('pybammsolvers_version', 'unknown')}"
    )

    baseline_benchmarks = baseline.get("benchmarks", {})
    current_benchmarks = current.get("benchmarks", {})

    regressions = []
    improvements = []

    for name in sorted(baseline_benchmarks.keys()):
        if name not in current_benchmarks:
            print(f"\n{name}: Not found in current results")
            continue

        baseline_time = baseline_benchmarks[name].get("average")
        current_time = current_benchmarks[name].get("average")

        if baseline_time is None or current_time is None:
            print(f"\n{name}: Missing timing data")
            continue

        diff = current_time - baseline_time
        pct_change = (diff / baseline_time) * 100

        status = "WARNING" if abs(pct_change) > 10 else "OK"
        direction = "slower" if diff > 0 else "faster"

        print(f"\n{status} {name}:")
        print(f"  Baseline: {baseline_time:.3f}s")
        print(f"  Current:  {current_time:.3f}s")
        print(f"  Change:   {abs(diff):.3f}s ({abs(pct_change):.1f}% {direction})")

        if pct_change > 50:
            regressions.append((name, pct_change))
            print("  WARNING: >50% performance regression!")
        elif pct_change < -20:
            improvements.append((name, abs(pct_change)))
            print("  Great! >20% performance improvement!")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if regressions:
        print(f"\n{len(regressions)} REGRESSION(S) DETECTED:")
        for name, pct in regressions:
            print(f"  - {name}: {pct:.1f}% slower")
        return False

    if improvements:
        print(f"\n{len(improvements)} IMPROVEMENT(S):")
        for name, pct in improvements:
            print(f"  - {name}: {pct:.1f}% faster")

    print("\nNo significant regressions detected")
    return True


def main():
    """Main orchestration function."""
    print("=" * 60)
    print("PyBaMM Performance Benchmark Suite")
    print("=" * 60)

    repo_root = Path(__file__).parent.parent.parent
    baseline_results = repo_root / "baseline_results.json"
    current_results = repo_root / "current_results.json"

    # Step 1: Run baseline benchmarks (vanilla PyBaMM)
    print("\n[1/4] Running baseline benchmarks with vanilla PyBaMM...")
    print("      (This establishes performance without pybammsolvers)")

    # Uninstall local pybammsolvers if present
    print("\n      Uninstalling local pybammsolvers...")
    result = run_command(
        [sys.executable, "-m", "uv", "pip", "uninstall", "-y", "pybammsolvers"]
    )

    # Install PyBaMM's bundled pybammsolvers
    print("      Installing pybamm (with bundled pybammsolvers)...")
    result = run_command(
        [sys.executable, "-m", "uv", "pip", "install", "--force-reinstall", "pybamm"]
    )

    if not run_benchmark_suite(baseline_results, "Baseline (vanilla PyBaMM)"):
        print("\nBaseline benchmark suite failed!")
        return 1

    # Step 2: Install local pybammsolvers
    print("\n[2/4] Installing local pybammsolvers...")
    result = run_command(
        [sys.executable, "-m", "uv", "pip", "install", "-e", ".", "--no-deps"],
        cwd=repo_root,
    )

    if result.returncode != 0:
        print("Failed to install local pybammsolvers!")
        print(result.stderr)
        return 1

    print("Local pybammsolvers installed")

    # Step 3: Run current benchmarks (with local pybammsolvers)
    print("\n[3/4] Running benchmarks with local pybammsolvers...")

    if not run_benchmark_suite(current_results, "Current (with pybammsolvers)"):
        print("\nCurrent benchmark suite failed!")
        return 1

    # Step 4: Compare results
    print("\n[4/4] Comparing results...")

    success = compare_results(baseline_results, current_results)

    # Save comparison results
    comparison_file = repo_root / "performance_results.json"

    # Load results
    with open(baseline_results) as f:
        baseline = json.load(f)
    with open(current_results) as f:
        current = json.load(f)

    # Create comparison record
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "baseline": baseline,
        "current": current,
        "success": success,
    }

    # Load history if exists
    if comparison_file.exists():
        with open(comparison_file) as f:
            try:
                history = json.load(f)
                if not isinstance(history, list):
                    history = [history]
            except json.JSONDecodeError:
                history = []
    else:
        history = []

    # Append and save
    history.append(comparison)
    with open(comparison_file, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nResults saved to {comparison_file}")

    # Clean up temporary files
    baseline_results.unlink(missing_ok=True)
    current_results.unlink(missing_ok=True)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
