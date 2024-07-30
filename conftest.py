import pytest
import numpy as np
import pybamm


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


def pytest_collection_modifyitems(items):
    for item in items:
        if "unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)


@pytest.fixture(autouse=True)
# Set the random seed to 42 for all tests
def set_random_seed():
    np.random.seed(42)


@pytest.fixture(autouse=True)
def set_debug_value():
    pybamm.settings.debug_mode = True
