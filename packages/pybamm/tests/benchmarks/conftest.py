"""Write benchmark results in Bencher Metric Format with ``--bencher-json``."""

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
    """Mark this benchmark as one size of a scaling series."""

    def record(size: float, label: str | None = None) -> None:
        series = request.node.nodeid.split("[")[0]
        if label is not None:
            series = f"{series}[{label}]"
        benchmark.extra_info["scaling"] = {"series": series, "size": size}

    return record


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
