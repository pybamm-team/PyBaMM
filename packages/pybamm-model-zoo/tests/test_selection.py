"""Which models a run collects, and which it never imports.

The tiering is only as good as this: marker-based deselection runs after every
test module has been imported, so a model outside the run has to be pruned from
collection or it can still break it.
"""

from pathlib import Path

import conftest
import pytest

from pybamm_model_zoo._registry import ModelEntry

ROOT = Path("/zoo/src/pybamm_model_zoo")
MACHINERY = Path("/zoo/tests/test_registry.py")


def entry(slug, tier):
    return ModelEntry(
        slug=slug,
        name=slug.title().replace("_", ""),
        path=ROOT / slug,
        raw={"model": {"tier": tier}},
    )


CORE = ROOT / "core_model"
COMMUNITY = ROOT / "community_model"


class FakeConfig:
    """Just the ``getoption`` that :func:`conftest.pytest_ignore_collect` reads."""

    def __init__(self, **options):
        self._options = options

    def getoption(self, name):
        return self._options.get(name)


@pytest.fixture(autouse=True)
def two_tiers(mocker):
    mocker.patch.object(
        conftest,
        "_model_folders",
        return_value=[
            (CORE, entry("core_model", "core")),
            (COMMUNITY, entry("community_model", "community")),
        ],
    )


def ignored(path, **options):
    return conftest.pytest_ignore_collect(path, FakeConfig(**options))


class TestIgnoreCollect:
    def test_an_unfiltered_run_prunes_nothing(self):
        assert ignored(COMMUNITY / "tests" / "test_it.py") is None
        assert ignored(CORE / "tests" / "test_it.py") is None

    @pytest.mark.parametrize(
        ("path", "expected"),
        [(COMMUNITY, True), (COMMUNITY / "tests" / "test_it.py", True), (CORE, None)],
    )
    def test_a_tier_prunes_the_other_tiers_folders(self, path, expected):
        assert ignored(path, **{"--zoo-tier": "core"}) is expected

    @pytest.mark.parametrize(("path", "expected"), [(COMMUNITY, True), (CORE, None)])
    def test_one_model_prunes_the_others(self, path, expected):
        assert ignored(path, **{"--zoo-model": "core_model"}) is expected

    def test_the_zoos_own_tests_are_never_pruned(self):
        """They are the contract suite and the registry: no model owns them."""
        for options in ({"--zoo-tier": "core"}, {"--zoo-model": "core_model"}):
            assert ignored(MACHINERY, **options) is None


class TestUnrecognisedTierIsNeverPruned:
    """A tier typo must not demote a model out of the run that would report it.

    ``check_manifest`` is the check that rejects the bad value, and it only runs
    for models the run kept, so pruning on an unrecognised tier would bury it.
    """

    @pytest.fixture(autouse=True)
    def mistyped_tier(self, mocker):
        mocker.patch.object(
            conftest,
            "_model_folders",
            return_value=[(CORE, entry("core_model", "Core"))],
        )

    @pytest.mark.parametrize("tier", ["core", "community"])
    def test_it_survives_every_tier_filter(self, tier):
        assert ignored(CORE / "tests" / "test_it.py", **{"--zoo-tier": tier}) is None

    def test_a_model_filter_still_applies(self):
        """Only the *tier* is untrusted; selecting one model by slug still works."""
        assert ignored(CORE, **{"--zoo-model": "other_model"}) is True
