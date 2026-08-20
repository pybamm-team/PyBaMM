"""Zoo-wide pytest configuration.

Duplicates the core package's autouse fixtures because
``packages/pybamm/conftest.py`` is not importable from here; extracting a shared
``pybamm.testing`` pytest plugin would mean changing the core package, which this
work deliberately does not.
"""

import numpy as np
import pytest

import pybamm
import pybamm_model_zoo as zoo

MODEL_TESTS_PARENT = "pybamm_model_zoo"


def pytest_addoption(parser):
    parser.addoption(
        "--zoo-model",
        action="store",
        default=None,
        metavar="SLUG",
        help=(
            "run only the tests belonging to one model, so CI can test just what a "
            "pull request changed"
        ),
    )


def _slug_from_path(path):
    """The model a test file belongs to, from ``<slug>/tests/test_*.py``."""
    parts = path.parts
    if "tests" not in parts:
        return None
    index = parts.index("tests")
    if index >= 2 and parts[index - 2] == MODEL_TESTS_PARENT:
        return parts[index - 1]
    return None


def _slug_of(item):
    """The model an item belongs to: its own marker, else its path.

    Parametrized suites mark each case with ``zoo_model(slug)`` rather than
    relying on an argument name, so renaming a fixture cannot silently empty the
    merge gate.
    """
    marker = item.get_closest_marker("zoo_model")
    return marker.args[0] if marker else _slug_from_path(item.path)


def pytest_collection_modifyitems(config, items):
    core_slugs = {entry.slug for entry in zoo.all_entries() if entry.tier == "core"}
    selected = config.getoption("--zoo-model")
    deselected = []
    remaining = []
    for item in items:
        item.add_marker(pytest.mark.zoo)
        if "integration" in item.path.parts:
            item.add_marker(pytest.mark.integration)
        elif "memory" not in item.path.parts:
            item.add_marker(pytest.mark.unit)

        slug = _slug_of(item)
        # Advisory-ness lives in the CI job, never in a marker: `gating` only
        # says whether a failure blocks a merge. A test belonging to no model is
        # the zoo's own machinery -- the registry, the template, the contract
        # suite itself -- which gates however the models happen to be tiered.
        if slug is None or slug in core_slugs:
            item.add_marker(pytest.mark.gating)
        if selected is not None and slug != selected:
            deselected.append(item)
        else:
            remaining.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = remaining


@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(42)


@pytest.fixture(autouse=True)
def set_debug_value():
    pybamm.settings.debug_mode = True


@pytest.fixture(autouse=True)
def disable_telemetry():
    pybamm.telemetry.disable()
