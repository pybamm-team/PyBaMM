import pytest
import importlib
import os
import sys
import pybamm
import tempfile
from io import StringIO

from tests import (
    get_optional_distribution_deps,
    get_required_distribution_deps,
    get_present_optional_import_deps,
)


class TestUtil:
    """
    Test the functionality in util.py
    """

    def test_is_constant_and_can_evaluate(self):
        symbol = pybamm.PrimaryBroadcast(0, "negative electrode")
        assert not pybamm.is_constant_and_can_evaluate(symbol)
        symbol = pybamm.StateVector(slice(0, 1))
        assert not pybamm.is_constant_and_can_evaluate(symbol)
        symbol = pybamm.Scalar(0)
        assert pybamm.is_constant_and_can_evaluate(symbol)

    def test_fuzzy_dict(self):
        d = pybamm.FuzzyDict(
            {
                "test": 1,
                "test2": 2,
                "SEI current": 3,
                "Lithium plating current": 4,
                "A dimensional variable [m]": 5,
                "Positive particle diffusivity [m2.s-1]": 6,
            }
        )
        d2 = pybamm.FuzzyDict(
            {
                "Positive electrode diffusivity [m2.s-1]": 6,
            }
        )
        assert d["test"] == 1
        with pytest.raises(KeyError, match="'test3' not found. Best matches are "):
            d.__getitem__("test3")

        with pytest.raises(KeyError, match="stoichiometry"):
            d.__getitem__("Negative electrode SOC")

        with pytest.raises(KeyError, match="dimensional version"):
            d.__getitem__("A dimensional variable")

        with pytest.raises(KeyError, match="open circuit voltage"):
            d.__getitem__("Measured open circuit voltage [V]")

        with pytest.raises(KeyError, match="Lower voltage"):
            d.__getitem__("Open-circuit voltage at 0% SOC [V]")

        with pytest.raises(KeyError, match="Upper voltage"):
            d.__getitem__("Open-circuit voltage at 100% SOC [V]")

        assert (
            d2["Positive particle diffusivity [m2.s-1]"]
            == d["Positive particle diffusivity [m2.s-1]"]
        )

        assert (
            d2["Positive electrode diffusivity [m2.s-1]"]
            == d["Positive electrode diffusivity [m2.s-1]"]
        )

        with pytest.warns(DeprecationWarning):
            assert (
                d["Positive electrode diffusivity [m2.s-1]"]
                == d["Positive particle diffusivity [m2.s-1]"]
            )

    def test_get_parameters_filepath(self):
        with tempfile.NamedTemporaryFile("w", dir=".") as tempfile_obj:
            assert (
                pybamm.get_parameters_filepath(tempfile_obj.name) == tempfile_obj.name
            )

        package_dir = os.path.join(pybamm.root_dir(), "pybamm")
        with tempfile.NamedTemporaryFile("w", dir=package_dir) as tempfile_obj:
            path = os.path.join(package_dir, tempfile_obj.name)
            assert pybamm.get_parameters_filepath(tempfile_obj.name) == path

    @pytest.mark.skipif(pybamm.have_jax(), reason="The JAX solver is not installed")
    def test_is_jax_compatible(self):
        assert True

    def test_git_commit_info(self):
        git_commit_info = pybamm.get_git_commit_info()
        assert isinstance(git_commit_info, str)
        assert git_commit_info[:2] == "v2"

    def test_import_optional_dependency(self):
        optional_distribution_deps = get_optional_distribution_deps("pybamm")
        present_optional_import_deps = get_present_optional_import_deps(
            "pybamm", optional_distribution_deps=optional_distribution_deps
        )

        # Save optional dependencies, then make them not importable
        modules = {}
        for import_pkg in present_optional_import_deps:
            modules[import_pkg] = sys.modules.get(import_pkg)
            sys.modules[import_pkg] = None

        # Test import optional dependency
        for import_pkg in present_optional_import_deps:
            with pytest.raises(
                ModuleNotFoundError,
                match=f"Optional dependency {import_pkg} is not available.",
            ):
                pybamm.util.import_optional_dependency(import_pkg)

        # Restore optional dependencies
        for import_pkg in present_optional_import_deps:
            sys.modules[import_pkg] = modules[import_pkg]

    def test_pybamm_import(self):
        optional_distribution_deps = get_optional_distribution_deps("pybamm")
        present_optional_import_deps = get_present_optional_import_deps(
            "pybamm", optional_distribution_deps=optional_distribution_deps
        )

        # Save optional dependencies and their sub-modules, then make them not importable
        modules = {}
        for module_name, module in sys.modules.items():
            base_module_name = module_name.split(".")[0]
            if base_module_name in present_optional_import_deps:
                modules[module_name] = module
                sys.modules[module_name] = None

        # Unload pybamm and its sub-modules
        for module_name in list(sys.modules.keys()):
            base_module_name = module_name.split(".")[0]
            if base_module_name == "pybamm":
                sys.modules.pop(module_name)

        # Test pybamm is still importable
        try:
            importlib.import_module("pybamm")
        except ModuleNotFoundError as error:
            pytest.fail(
                f"Import of 'pybamm' shouldn't require optional dependencies. Error: {error}"
            )
        finally:
            # Restore optional dependencies and their sub-modules
            for module_name, module in modules.items():
                sys.modules[module_name] = module

    def test_optional_dependencies(self):
        optional_distribution_deps = get_optional_distribution_deps("pybamm")
        required_distribution_deps = get_required_distribution_deps("pybamm")

        # Get nested required dependencies
        for distribution_dep in list(required_distribution_deps):
            required_distribution_deps.update(
                get_required_distribution_deps(distribution_dep)
            )

        # Check that optional dependencies are not present in the core PyBaMM installation
        optional_present_deps = bool(
            optional_distribution_deps & required_distribution_deps
        )
        assert not optional_present_deps, (
            f"Optional dependencies installed: {optional_present_deps}.\n"
            "Please ensure that optional dependencies are not present in the core PyBaMM installation, "
            "or list them as required."
        )


class TestSearch:
    def test_url_gets_to_stdout(self, mocker):
        model = pybamm.BaseModel()
        model.variables = {"Electrolyte concentration": 1, "Electrode potential": 0}

        param = pybamm.ParameterValues({"test": 10, "b": 2})

        # Test variables search (default returns key)
        with mocker.patch("sys.stdout", new=StringIO()) as fake_out:
            model.variables.search("Electrode")
            assert fake_out.getvalue() == "Electrode potential\n"

        # Test bad var search (returns best matches)
        with mocker.patch("sys.stdout", new=StringIO()) as fake_out:
            model.variables.search("Electrolyte cot")
            out = (
                "No results for search using 'Electrolyte cot'. "
                "Best matches are ['Electrolyte concentration', "
                "'Electrode potential']\n"
            )
            assert fake_out.getvalue() == out

        # Test param search (default returns key, value)
        with mocker.patch("sys.stdout", new=StringIO()) as fake_out:
            param.search("test")
            assert fake_out.getvalue() == "test\t10\n"
