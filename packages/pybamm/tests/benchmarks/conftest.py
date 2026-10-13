"""
Bencher Metric Format (BMF) output for the timing benchmarks.

pytest-benchmark records each benchmark's latency. Tests attach hardware-independent
measures through ``benchmark.extra_info``:

- ``extra_info["measures"]`` maps a measure name to a value, e.g. a module count.
- ``extra_info["scaling"]`` (set by the ``record_scaling`` fixture) marks one size of a
  scaling series. Each series is reported as its growth exponent.

``--bencher-json PATH`` writes all of them as BMF for ``bencher run --adapter json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

_NANOSECONDS_PER_SECOND = 1e9


def pytest_addoption(parser):
    parser.addoption(
        "--bencher-json",
        metavar="PATH",
        help="Also write results in Bencher Metric Format; requires --benchmark-json.",
    )


@pytest.hookimpl(optionalhook=True)
def pytest_benchmark_update_json(config, benchmarks, output_json):
    path = config.getoption("--bencher-json")
    if path:
        metrics = to_bencher_metric_format(output_json["benchmarks"])
        Path(path).write_text(json.dumps(metrics, indent=2))


@pytest.fixture
def record_scaling(benchmark, request):
    """
    Mark this benchmark as one size of a scaling series.

    The series is named after the test function, plus ``label`` when one test function
    holds several series.
    """

    def record(size: float, label: str | None = None) -> None:
        series = request.node.nodeid.split("[")[0]
        if label is not None:
            series = f"{series}[{label}]"
        benchmark.extra_info["scaling"] = {"series": series, "size": size}

    return record


def to_bencher_metric_format(benchmarks: list[dict]) -> dict[str, dict]:
    """
    Convert pytest-benchmark's JSON benchmarks to Bencher Metric Format.

    Parameters
    ----------
    benchmarks : list of dict
        The ``benchmarks`` entry of pytest-benchmark's JSON output.

    Returns
    -------
    dict
        BMF results keyed by benchmark name, then by measure. Latency matches
        Bencher's ``python_pytest`` adapter: the mean in nanoseconds, with one standard
        deviation either side.
    """
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
    """
    The exponent ``p`` in ``time ~ size**p`` between two (size, time) points.

    Parameters
    ----------
    small, large : tuple of float
        The (size, time) of the smaller and the larger run.

    Returns
    -------
    float
        About 1 for linear growth and 2 for quadratic growth.
    """
    (small_size, small_time), (large_size, large_time) = small, large
    return math.log(large_time / small_time) / math.log(large_size / small_size)
