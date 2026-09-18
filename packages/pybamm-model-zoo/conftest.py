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
    parser.addoption(
        "--zoo-tier",
        action="store",
        default=None,
        choices=zoo.TIERS,
        metavar="TIER",
        help=(
            "keep only the models of one tier in collection, so the merge gate "
            "cannot be broken by a model outside it"
        ),
    )


def _model_folders():
    """Every registered model's folder, with its slug and tier.

    Manifests are parsed rather than imported, so this is safe to call before a
    single test module has been loaded.
    """
    return [
        (entry.path.resolve(), entry.slug, entry.tier) for entry in zoo.all_entries()
    ]


def pytest_ignore_collect(collection_path, config):
    """Drop a model's folder before pytest imports anything inside it.

    It does not replace the marker: the contract suite is parametrized over every
    model from a single module, so those cases are filtered by marker instead.
    """
    selected = config.getoption("--zoo-model")
    tier = config.getoption("--zoo-tier")
    if selected is None and tier is None:
        return None
    for folder, slug, model_tier in _model_folders():
        if collection_path != folder and folder not in collection_path.parents:
            continue
        if selected is not None and slug != selected:
            return True
        return True if tier is not None and model_tier != tier else None
    return None


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
