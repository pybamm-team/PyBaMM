"""What counts as "not installed", which gates loading a model and its checks."""

from importlib.metadata import PackageNotFoundError

import pytest

from pybamm_model_zoo import _dependencies

METADATA = {
    "zoo": [
        "missing-base-xyz>=1.0",
        'pybamm[fem]; extra == "zoo-nested"',
        'absent-xyz>=1.0; extra == "zoo-absent"',
        'packaging>=9999; extra == "zoo-too-old"',
        'packaging>=23.0; extra == "zoo-installed"',
        'absent-xyz; extra == "zoo-elsewhere" and python_version < "3"',
        'zoo[zoo-nested]; extra == "zoo-all"',
    ],
    "pybamm": ['scikit-fem>=12.0.2; extra == "fem"', "numpy>=2.0"],
}
INSTALLED = {"zoo": "0.1.0", "pybamm": "26.10.0.0", "packaging": "25.0"}


@pytest.fixture
def environment(monkeypatch):
    """A fake set of installed distributions, editable per test."""
    installed = dict(INSTALLED)

    def lookup(table, name):
        try:
            return table[name]
        except KeyError:
            raise PackageNotFoundError(name) from None

    monkeypatch.setattr(_dependencies, "requires", lambda name: lookup(METADATA, name))
    monkeypatch.setattr(_dependencies, "version", lambda name: lookup(installed, name))
    return installed


class TestUnsatisfied:
    def test_an_absent_package_is_missing(self, environment):
        assert _dependencies.unsatisfied("zoo", ["zoo-absent"]) == ["absent-xyz>=1.0"]

    def test_an_installed_package_too_old_is_missing(self, environment):
        assert _dependencies.unsatisfied("zoo", ["zoo-too-old"]) == ["packaging>=9999"]

    def test_an_installed_package_is_not_missing(self, environment):
        assert _dependencies.unsatisfied("zoo", ["zoo-installed"]) == []

    def test_a_nested_extra_is_followed(self, environment):
        """PyBaMM itself is installed, but not the extra the model asks it for."""
        assert _dependencies.unsatisfied("zoo", ["zoo-nested"]) == [
            "scikit-fem>=12.0.2"
        ]
        environment["scikit-fem"] = "12.0.2"
        assert _dependencies.unsatisfied("zoo", ["zoo-nested"]) == []

    def test_a_requirement_whose_marker_is_false_is_not_missing(self, environment):
        """Otherwise a platform-specific dependency fails everywhere else."""
        assert _dependencies.unsatisfied("zoo", ["zoo-elsewhere"]) == []

    def test_base_dependencies_are_not_the_extras(self, environment):
        assert _dependencies.unsatisfied("zoo", ["zoo-undeclared"]) == []

    def test_an_aggregate_naming_its_own_distribution_terminates(self, environment):
        assert _dependencies.unsatisfied("zoo", ["zoo-all"]) == ["scikit-fem>=12.0.2"]

    def test_a_distribution_that_is_not_installed_declares_nothing(self, environment):
        assert _dependencies.unsatisfied("not-installed", ["zoo-absent"]) == []


class TestInstalledMetadata:
    def test_the_fem_extra_reaches_scikit_fem(self, monkeypatch):
        """The zoo's own extra, through PyBaMM's, as `pip` installs them."""
        real_version = _dependencies.version

        def version(name):
            if name == "scikit-fem":
                raise PackageNotFoundError(name)
            return real_version(name)

        monkeypatch.setattr(_dependencies, "version", version)
        missing = _dependencies.unsatisfied(
            "pybamm-model-zoo", ["zoo-multilayer-3d-thermal"]
        )
        assert [item.split(">")[0] for item in missing] == ["scikit-fem"]
