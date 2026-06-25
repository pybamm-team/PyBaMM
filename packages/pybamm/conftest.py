import os

import numpy as np
import pytest
from hypothesis import settings as hypothesis_settings

import pybamm

# Hypothesis auto-loads its built-in "ci" profile in CI, which derandomizes
# every run (the same inputs forever). Re-register it with randomisation
# enabled so CI explores new inputs each run; the inherited print_blob=True
# keeps any failure reproducible locally via @reproduce_failure.
hypothesis_settings.register_profile(
    "ci", parent=hypothesis_settings.get_profile("ci"), derandomize=False
)
if os.environ.get("CI"):
    hypothesis_settings.load_profile("ci")


def pytest_addoption(parser):
    parser.addoption(
        "--scripts",
        action="store_true",
        default=False,
        help="execute the example scripts",
    )
    parser.addoption(
        "--unit", action="store_true", default=False, help="run unit tests"
    )
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="run integration tests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "scripts: mark test as an example script")
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "memory: mark test as a memory stress test")
    config.addinivalue_line("markers", "speed_bench: mark test as a speed benchmark")
    config.addinivalue_line("markers", "memory_bench: mark test as a memory benchmark")


def pytest_collection_modifyitems(items):
    for item in items:
        if "unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "memory" in item.nodeid:
            item.add_marker(pytest.mark.memory)


@pytest.fixture(autouse=True)
# Set the random seed to 42 for all tests
def set_random_seed():
    np.random.seed(42)


@pytest.fixture(autouse=True)
def set_debug_value():
    pybamm.settings.debug_mode = True


@pytest.fixture(autouse=True)
def disable_telemetry():
    pybamm.telemetry.disable()
