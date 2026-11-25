"""Write benchmark results in Bencher Metric Format with ``--bencher-json``."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

_NANOSECONDS_PER_SECOND = 1e9
_PEAK_MEMORY_ROUNDS = 2


def pytest_addoption(parser):
    parser.addoption(
        "--bencher-json",
        metavar="PATH",
        help="Also write results in Bencher Metric Format; requires --benchmark-json.",
    )
    parser.addoption(
        "--peak-memory-file",
        metavar="PATH",
        help="Trace record_peak_memory in this process and write the peak to PATH.",
    )


@pytest.hookimpl(optionalhook=True)
def pytest_benchmark_update_json(config, benchmarks, output_json):
    path = config.getoption("--bencher-json")
    if path:
        metrics = to_bencher_metric_format(output_json["benchmarks"])
        Path(path).write_text(json.dumps(metrics, indent=2))


@pytest.fixture
def record_scaling(benchmark, request):
    """Mark this benchmark as one size of a scaling series."""

    def record(size: float, label: str | None = None) -> None:
        series = request.node.nodeid.split("[")[0]
        if label is not None:
            series = f"{series}[{label}]"
        benchmark.extra_info["scaling"] = {"series": series, "size": size}

    return record


@pytest.fixture
def record_peak_memory(benchmark, request, tmp_path):
    """Record the peak heap memory of a call as this benchmark's ``peak-memory``.

    The call is traced in a fresh interpreter running only this benchmark: caches
    that earlier benchmarks leave behind move the peak by several percent.
    ``setup`` returns ``(args, kwargs)`` as for ``benchmark.pedantic``.
    """
    config = request.config
    recorded = []

    def record(function: Callable, setup: Callable[[], tuple[tuple, dict]]) -> None:
        if recorded:
            pytest.fail("record_peak_memory records one call per benchmark")
        recorded.append(function)
        # memray is Linux/macOS only, and refuses a second Tracker under --memray
        if sys.platform == "win32" or config.getoption("memray", False):
            return
        peak_file = config.getoption("--peak-memory-file")
        if peak_file:
            Path(peak_file).write_text(str(_traced_peak(function, setup, tmp_path)))
            return
        if benchmark.disabled:
            return

        peak_file = tmp_path / "peak-memory.txt"
        command = [
            sys.executable,
            "-m",
            "pytest",
            request.node.nodeid,
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            f"--basetemp={tmp_path / 'fresh'}",
            # Runs the benchmark once, untimed, so the traced calls start warm
            "--benchmark-disable",
            f"--peak-memory-file={peak_file}",
        ]
        # A fixed seed fixes set and dict iteration order, and so the allocations
        env = {**os.environ, "PYTHONHASHSEED": "0"}
        fresh = subprocess.run(
            command,
            cwd=config.rootpath,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if fresh.returncode != 0:
            output = fresh.stdout + fresh.stderr
            pytest.fail(f"Tracing in a fresh interpreter failed:\n{output}")
        peak = int(peak_file.read_text())
        benchmark.extra_info.setdefault("measures", {})["peak-memory"] = peak

    return record


def _traced_peak(function: Callable, setup: Callable, directory: Path) -> int:
    """The least peak heap memory of ``_PEAK_MEMORY_ROUNDS`` traced calls."""
    import memray

    trace = directory / "peak-memory.bin"
    peaks = []
    for _ in range(_PEAK_MEMORY_ROUNDS):
        args, kwargs = setup()
        # Per-object tracing; pymalloc arenas would make the peak jump by 1 MiB
        with memray.Tracker(
            trace,
            trace_python_allocators=True,
            file_format=memray.FileFormat.AGGREGATED_ALLOCATIONS,
        ):
            function(*args, **kwargs)
        peaks.append(memray.FileReader(trace).metadata.peak_memory)
        trace.unlink()
    # The first traced call also allocates memray's per-function profiler data
    return min(peaks)


def to_bencher_metric_format(benchmarks: list[dict]) -> dict[str, dict]:
    """Convert pytest-benchmark results to BMF, with latency as ``python_pytest``."""
    metrics = {}
    series_points: dict[str, list[tuple[float, float]]] = {}
    for benchmark in benchmarks:
        stats = benchmark["stats"]
        mean = stats["mean"] * _NANOSECONDS_PER_SECOND
        spread = stats["stddev"] * _NANOSECONDS_PER_SECOND
        measures = {
            "latency": {
                "value": mean,
                "lower_value": mean - spread,
                "upper_value": mean + spread,
            }
        }
        extra_info = benchmark.get("extra_info", {})
        for name, value in extra_info.get("measures", {}).items():
            measures[name] = {"value": value}
        metrics[benchmark["fullname"]] = measures

        if "scaling" in extra_info:
            scaling = extra_info["scaling"]
            series_points.setdefault(scaling["series"], []).append(
                (scaling["size"], stats["median"])
            )

    for series, points in series_points.items():
        if len(points) < 2:
            continue
        exponent = growth_exponent(min(points), max(points))
        metrics[series] = {"scaling-exponent": {"value": exponent}}
    return metrics


def growth_exponent(small: tuple[float, float], large: tuple[float, float]) -> float:
    """The ``p`` in ``time ~ size**p`` between two (size, time) points."""
    (small_size, small_time), (large_size, large_time) = small, large
    return math.log(large_time / small_time) / math.log(large_size / small_size)
