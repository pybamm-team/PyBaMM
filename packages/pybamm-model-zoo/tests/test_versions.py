"""Release ordering and the zoo's compatibility window."""

import pytest
from packaging.requirements import Requirement

from pybamm_model_zoo import _paths, _versions
from pybamm_model_zoo._exceptions import ZooError
from pybamm_model_zoo._registry import read_manifest

RELEASES = ["25.12.0", "26.0.0", "26.5.0", "26.7.1", "26.8.0"]


class TestSortedReleases:
    def test_orders_numerically_not_lexically(self):
        assert _versions.sorted_releases(["26.10.0", "26.9.0", "26.8.0"]) == [
            "26.8.0",
            "26.9.0",
            "26.10.0",
        ]

    def test_drops_anything_that_is_not_a_final_calver_release(self):
        assert _versions.sorted_releases(
            ["26.8.0", "26.9.0rc1", "26.9.0.dev0", _versions.MAIN]
        ) == ["26.8.0"]


class TestWindow:
    def test_keeps_the_oldest_admitted_and_the_newest(self):
        assert _versions.window(RELEASES, ">=26.0", 2) == [
            "26.0.0",
            "26.7.1",
            "26.8.0",
        ]

    def test_an_oldest_already_among_the_newest_is_not_repeated(self):
        assert _versions.window(RELEASES, ">=26.7", 2) == ["26.7.1", "26.8.0"]

    def test_no_newest_still_tests_the_floor(self):
        """`--releases 0` keeps the floor; `[-0:]` would have kept every release."""
        assert _versions.window(RELEASES, ">=26.0", 0) == ["26.0.0"]

    def test_a_floor_nothing_released_meets_gets_no_cells(self):
        assert _versions.window(RELEASES, ">=99.0", 2) == []

    def test_an_empty_specifier_admits_everything(self):
        assert _versions.window(RELEASES, "", 1) == ["25.12.0", "26.8.0"]


class TestPybammSpecifier:
    def test_is_the_zoo_pybamm_dependency(self):
        dependencies = read_manifest(_paths.ZOO_PYPROJECT)["project"]["dependencies"]
        pybamm = next(
            Requirement(item)
            for item in dependencies
            if Requirement(item).name == "pybamm"
        )
        assert _versions.pybamm_specifier() == str(pybamm.specifier)

    def test_a_pyproject_without_pybamm_is_reported(self, tmp_path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\ndependencies = ["packaging>=23.0"]\n')
        with pytest.raises(ZooError, match=r"no pybamm dependency"):
            _versions.pybamm_specifier(pyproject)
