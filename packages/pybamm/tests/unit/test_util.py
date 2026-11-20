import importlib
import importlib.util
import json
import os
import pickle
import pkgutil
import subprocess
import sys
import types
from io import StringIO

import numpy as np
import pytest

import pybamm
from tests import (
    get_optional_distribution_deps,
    get_present_optional_import_deps,
    get_required_distribution_deps,
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
                "Primary: Open circuit voltage [V]": 7,
            }
        )
        d2 = pybamm.FuzzyDict(
            {
                "Positive electrode diffusivity [m2.s-1]": 6,
            }
        )
        assert d["test"] == 1
        with pytest.raises(KeyError, match=r"'test3' not found. Best matches are "):
            d.__getitem__("test3")

        with pytest.raises(KeyError, match=r"stoichiometry"):
            d.__getitem__("Negative electrode SOC")

        with pytest.raises(KeyError, match=r"dimensional version"):
            d.__getitem__("A dimensional variable")

        with pytest.raises(KeyError, match=r"composite model"):
            d.__getitem__("Open circuit voltage [V]")

        with pytest.raises(KeyError, match=r"open circuit voltage"):
            d.__getitem__("Measured open circuit voltage [V]")

        with pytest.raises(KeyError, match=r"Lower voltage"):
            d.__getitem__("Open-circuit voltage at 0% SOC [V]")

        with pytest.raises(KeyError, match=r"Upper voltage"):
            d.__getitem__("Open-circuit voltage at 100% SOC [V]")

        assert (
            d2["Positive particle diffusivity [m2.s-1]"]
            == d["Positive particle diffusivity [m2.s-1]"]
        )

        assert (
            d2["Positive electrode diffusivity [m2.s-1]"]
            == d["Positive electrode diffusivity [m2.s-1]"]
        )

        with pytest.warns(
            DeprecationWarning,
            match=r"'Positive electrode diffusivity \[m2.s-1\]' has been renamed to "
            r"'Positive particle diffusivity \[m2.s-1\]'.*Using "
            r"'Positive particle diffusivity \[m2.s-1\]'",
        ):
            assert (
                d["Positive electrode diffusivity [m2.s-1]"]
                == d["Positive particle diffusivity [m2.s-1]"]
            )

    def test_get_parameters_filepath(self, tmp_path):
        temppath = tmp_path / "temp_file.txt"
        assert pybamm.get_parameters_filepath(temppath) == str(temppath)

        temppath = "random.txt"
        assert pybamm.get_parameters_filepath(temppath) == str(
            os.path.join(pybamm.root_dir(), "src", "pybamm", temppath)
        )

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
        original_symbol = pybamm.Symbol

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
        unloaded = {}
        for module_name in list(sys.modules.keys()):
            base_module_name = module_name.split(".")[0]
            if base_module_name == "pybamm":
                unloaded[module_name] = sys.modules.pop(module_name)

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
            # Restore the original classes: the re-import built a second set, and code
            # imported earlier still holds the first, so `isinstance` fails across them.
            sys.modules.update(unloaded)

        assert pybamm.Symbol is original_symbol
        assert importlib.import_module("pybamm").Symbol is original_symbol

    @staticmethod
    def _fake_module(monkeypatch, name, **objects):
        """Register a stand-in module whose objects pickle under its name."""
        module = types.ModuleType(name)
        for attribute, obj in objects.items():
            obj.__module__, obj.__qualname__ = name, attribute
            setattr(module, attribute, obj)
        monkeypatch.setitem(sys.modules, name, module)

    def test_load_drops_xarray_caches_of_older_files(self, tmp_path, monkeypatch):
        # Solutions saved by 26.10 or earlier can cache xarray and pandas objects,
        # which pickle in each of these ways
        class DataArray:  # rebuilt as an object, then its state set
            def __init__(self, values, index, names, attrs):
                self.values, self.index = values, index
                self.names, self.attrs = names, attrs

        class Index:  # rebuilt by calling a function
            def __init__(self, data):
                self.data = data

            def __reduce__(self):
                return new_index, (type(self), {"data": self.data})

        def new_index(cls, state):
            return cls(**state)

        class FrozenList(list):  # rebuilt, then extended
            pass

        class Attrs(dict):  # rebuilt, then its items set
            pass

        self._fake_module(monkeypatch, "xarray", DataArray=DataArray)
        self._fake_module(
            monkeypatch,
            "pandas",
            Index=Index,
            new_index=new_index,
            FrozenList=FrozenList,
            Attrs=Attrs,
        )
        solution = pybamm.Simulation(pybamm.lithium_ion.SPM()).solve([0, 60])
        voltage = solution["Voltage [V]"]
        voltage._xr_array_raw = DataArray(
            voltage.entries,
            Index(voltage.t_pts),
            FrozenList(["t"]),
            Attrs(units="V"),
        )
        path = tmp_path / "solution.pkl"
        solution.save(path)

        # Without xarray or pandas installed
        monkeypatch.setitem(sys.modules, "xarray", None)
        monkeypatch.setitem(sys.modules, "pandas", None)
        with pytest.raises(ImportError):
            pickle.loads(path.read_bytes())
        loaded = pybamm.load(path)["Voltage [V]"]
        np.testing.assert_array_equal(loaded(voltage.t_pts), voltage.entries)
        assert "_xr_array_raw" not in vars(loaded)

        # Any other missing package still fails to load
        class Other:
            pass

        self._fake_module(monkeypatch, "some_missing_package", Other=Other)
        voltage._xr_array_raw = Other()
        solution.save(path)
        monkeypatch.setitem(sys.modules, "some_missing_package", None)
        with pytest.raises(ImportError):
            pybamm.load(path)

    @staticmethod
    def _cache_holder_class():
        class Holder:
            def __setstate__(self, state):
                # Discards its cache on loading, as processed variables do
                state.pop("cache", None)
                self.__dict__.update(state)

        return Holder

    def test_load_tolerates_classes_the_installed_package_lacks(
        self, tmp_path, monkeypatch
    ):
        class DataFrame:
            pass

        class Index:
            pass

        Holder = self._cache_holder_class()
        self._fake_module(monkeypatch, "pandas", DataFrame=DataFrame, Index=Index)
        self._fake_module(monkeypatch, "older_package", Holder=Holder)
        holder = Holder()
        holder.cache, holder.frame = Index(), DataFrame()
        cache_discarded = pickle.dumps(holder)
        holder.kept = Index()
        index_kept = pickle.dumps(holder)

        # A newer pandas that renamed or removed Index
        newer_pandas = types.ModuleType("pandas")
        newer_pandas.DataFrame = DataFrame
        monkeypatch.setitem(sys.modules, "pandas", newer_pandas)
        with pytest.raises(AttributeError):
            pickle.loads(cache_discarded)
        path = tmp_path / "holder.pkl"
        path.write_bytes(cache_discarded)
        loaded = pybamm.load(path)
        assert isinstance(loaded.frame, DataFrame)
        assert "cache" not in vars(loaded)
        path.write_bytes(index_kept)
        with pytest.raises(ImportError, match=r"holds pandas\.Index, which could not"):
            pybamm.load(path)

        # Any other package lacking a class still fails to load
        class Other:
            pass

        self._fake_module(monkeypatch, "some_package", Other=Other)
        path.write_bytes(pickle.dumps(Other()))
        monkeypatch.setitem(sys.modules, "some_package", types.ModuleType("x"))
        with pytest.raises(AttributeError):
            pybamm.load(path)

    def test_load_raises_if_the_object_keeps_what_it_cannot_import(
        self, tmp_path, monkeypatch
    ):
        class DataFrame:
            pass

        class Index:
            pass

        Holder = self._cache_holder_class()
        self._fake_module(monkeypatch, "pandas", DataFrame=DataFrame, Index=Index)
        self._fake_module(monkeypatch, "older_package", Holder=Holder)
        holder = Holder()
        # A cache in a reference cycle outlives loading until garbage collection
        holder.cache = [Index()]
        holder.cache.append(holder.cache)
        cache_only = pickle.dumps(holder)
        holder.frames = [DataFrame(), DataFrame()]
        frames_kept = pickle.dumps(holder)
        del holder.frames
        holder.kind = Index
        class_kept = pickle.dumps(holder)

        monkeypatch.setitem(sys.modules, "pandas", None)
        path = tmp_path / "holder.pkl"
        path.write_bytes(cache_only)
        assert vars(pybamm.load(path)) == {}
        for kept, name in [(frames_kept, "DataFrame"), (class_kept, "Index")]:
            path.write_bytes(kept)
            with pytest.raises(
                ImportError,
                match=(
                    rf"^The loaded object holds pandas\.{name}, which could not be "
                    r"imported\. Install a version of pandas that provides them\.$"
                ),
            ):
                pybamm.load(path)

    def test_import_defers_heavy_dependencies(self):
        deferred = ["black", "pandas", "pooch", "posthog", "sympy", "xarray"]
        code = (
            "import json, sys, pybamm; "
            f"print(json.dumps([m for m in {deferred!r} if m in sys.modules]))"
        )
        env = {**os.environ, "PYBAMM_DISABLE_TELEMETRY": "true"}
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        # An editable install's rebuild check can print before the result
        assert json.loads(result.stdout.splitlines()[-1]) == []

    def test_package_all_names_exist(self):
        # `from package import *` raises for a name that is neither an attribute
        # nor a submodule
        for _, package_name, is_package in pkgutil.walk_packages(
            pybamm.__path__, "pybamm."
        ):
            if not is_package:
                continue
            package = importlib.import_module(package_name)
            for name in getattr(package, "__all__", []):
                assert hasattr(package, name) or importlib.util.find_spec(
                    f"{package_name}.{name}"
                ), f"{package_name}.__all__ lists {name!r}, which does not exist"

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
            assert (
                fake_out.getvalue()
                == "Results for 'Electrode': ['Electrode potential']\n"
            )
        # Test bad var search (returns best matches)
        with mocker.patch("sys.stdout", new=StringIO()) as fake_out:
            model.variables.search("Electrolyte cot")
            out = (
                "No exact matches found for 'Electrolyte cot'. "
                "Best matches are: ['Electrolyte concentration', 'Electrode potential']\n"
            )
            assert fake_out.getvalue() == out

        # Test for multiple strings as input (default returns key)
        with mocker.patch("sys.stdout", new=StringIO()) as fake_out:
            model.variables.search(["Electrolyte", "Concentration"], print_values=True)
            assert (
                fake_out.getvalue()
                == "Results for 'Electrolyte Concentration': ['Electrolyte concentration']\n"
                "Electrolyte concentration -> 1\n"
            )

        # Test for multiple strings as input (default returns best matches)
        with mocker.patch("sys.stdout", new=StringIO()) as fake_out:
            model.variables.search(["Electrolyte", "Potenteel"], print_values=True)
            out = (
                "Exact matches for 'Electrolyte': ['Electrolyte concentration']\n"
                "Electrolyte concentration -> 1\n"
                "No exact matches found for 'Potenteel'. Best matches are: ['Electrode potential']\n"
            )
            assert fake_out.getvalue() == out

        # Test param search (default returns key, value)
        with mocker.patch("sys.stdout", new=StringIO()) as fake_out:
            param.search("test")
            out = "Results for 'test': ['test']\ntest -> 10\n"
            assert fake_out.getvalue() == out

        # Test no matches and no best matches
        with mocker.patch("sys.stdout", new=StringIO()) as fake_out:
            model.variables.search("NonexistentKey")
            assert fake_out.getvalue() == "No matches found for 'NonexistentKey'\n"

        # Test print_values=True with partial matches
        with mocker.patch("sys.stdout", new=StringIO()) as fake_out:
            model.variables.search("Electrolyte", print_values=True)
            out = (
                "Results for 'Electrolyte': ['Electrolyte concentration']\n"
                "Electrolyte concentration -> 1\n"
            )
            assert fake_out.getvalue() == out

        # Test for empty string input (raises ValueError)
        with pytest.raises(
            ValueError,
            match=r"The search term cannot be an empty or whitespace-only string",
        ):
            model.variables.search("", print_values=True)

        # Test for list with all empty strings (raises ValueError)
        with pytest.raises(
            ValueError,
            match=r"The 'keys' list cannot contain only empty or whitespace strings",
        ):
            model.variables.search(["", "   ", "\t"], print_values=True)

        # Test for list with a mix of empty and valid strings
        with mocker.patch("sys.stdout", new=StringIO()) as fake_out:
            model.variables.search(["", "Electrolyte"], print_values=True)
            out = (
                "Results for 'Electrolyte': ['Electrolyte concentration']\n"
                "Electrolyte concentration -> 1\n"
            )
            assert fake_out.getvalue() == out

        # Blank terms are dropped without shifting the labels of the other terms
        with mocker.patch("sys.stdout", new=StringIO()) as fake_out:
            model.variables.search(["", "Electrolyte", "Potenteel"])
            out = (
                "Exact matches for 'Electrolyte': ['Electrolyte concentration']\n"
                "No exact matches found for 'Potenteel'. Best matches are: ['Electrode potential']\n"
            )
            assert fake_out.getvalue() == out

        # Labels show each term without its surrounding whitespace
        with mocker.patch("sys.stdout", new=StringIO()) as fake_out:
            model.variables.search([" Electrolyte", "Potenteel "])
            out = (
                "Exact matches for 'Electrolyte': ['Electrolyte concentration']\n"
                "No exact matches found for 'Potenteel'. Best matches are: ['Electrode potential']\n"
            )
            assert fake_out.getvalue() == out

        with mocker.patch("sys.stdout", new=StringIO()) as fake_out:
            model.variables.search(" Electrolyte ")
            assert fake_out.getvalue() == (
                "Results for 'Electrolyte': ['Electrolyte concentration']\n"
            )

        # Test invalid input type
        with pytest.raises(
            TypeError,
            match=r"'keys' must be a string or a list of strings, got <class 'int'>",
        ):
            model.variables.search(123)

        # Test smaller strings
        with mocker.patch("sys.stdout", new=StringIO()) as fake_out:
            model.variables.search(["El", "co"], print_values=True)
            out = "No matches found for 'El'\nNo matches found for 'co'\n"
            assert fake_out.getvalue() == out

        # Case where min_similarity is high (0.9)
        with mocker.patch("sys.stdout", new=StringIO()) as fake_out:
            model.variables.search("electro", min_similarity=0.9)
            assert fake_out.getvalue() == "No matches found for 'electro'\n"

        # Case where min_similarity is low (0.3)
        with mocker.patch("sys.stdout", new=StringIO()) as fake_out:
            model.variables.search("electro", min_similarity=0.3)
            assert (
                fake_out.getvalue()
                == "Results for 'electro': ['Electrolyte concentration', 'Electrode potential']\n"
            )
