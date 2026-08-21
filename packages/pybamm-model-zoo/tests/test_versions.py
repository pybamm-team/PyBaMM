"""Release ordering and the per-model compatibility window."""

from pathlib import Path

import pytest

from pybamm_model_zoo import _versions
from pybamm_model_zoo._exceptions import ManifestError
from pybamm_model_zoo._registry import ModelEntry

RELEASES = ["25.12.0", "26.0.0", "26.5.0", "26.7.1", "26.8.0"]


def entry(pybamm_requires):
    return ModelEntry(
        slug="a_model",
        name="AModel",
        path=Path("a_model"),
        raw={"model": {"pybamm_requires": pybamm_requires}},
        external=False,
    )


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


class TestWindowFor:
    def test_takes_the_newest_releases_a_model_admits(self):
        assert _versions.window_for(entry(">=26.0"), RELEASES, 2) == [
            "26.7.1",
            "26.8.0",
        ]

    def test_an_upper_bound_keeps_the_releases_below_it(self):
        """The bug this guards: filtering after the window leaves no cells at all."""
        assert _versions.window_for(entry(">=26.0,<26.7"), RELEASES, 2) == [
            "26.0.0",
            "26.5.0",
        ]

    def test_a_window_wider_than_the_admitted_set_is_not_padded(self):
        assert _versions.window_for(entry("<26.0"), RELEASES, 3) == ["25.12.0"]

    def test_a_model_admitting_nothing_released_gets_no_cells(self):
        assert _versions.window_for(entry(">=99.0"), RELEASES, 2) == []

    def test_an_empty_specifier_admits_everything(self):
        assert _versions.window_for(entry(""), RELEASES, 1) == ["26.8.0"]

    def test_a_malformed_specifier_is_reported_against_the_manifest(self):
        with pytest.raises(ManifestError, match=r"not a valid specifier"):
            _versions.window_for(entry("=>26.0"), RELEASES, 1)
