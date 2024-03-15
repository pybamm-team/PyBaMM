import importlib
import re

from tests import TestCase


class TestDependencies(TestCase):
    """
    This class tests the dependencies required by PyBaMM for specific versions.

    **Note:** This test module is **not run automatically** with other tests.
    Its functions are intended to be tested manually on different PyBaMM installations that are installed with various sets of extra dependencies.
    """

    def test_core_optional_dependencies(self):
        """
        Ensure optional dependencies are not installed in the core PyBaMM version.
        It scan all dependencies for PyBaMM and checks that the ones listed as optional are not installed.
        """

        pattern = re.compile(
            r"(?!.*pybamm\b|.*docs\b|.*dev\b)^([^>=;\[]+)\b.*$"
        )  # do not consider pybamm, [docs] and [dev] dependencies
        json_deps = importlib.metadata.metadata("pybamm").json["requires_dist"]

        optional_deps = {
            m.group(1)
            for dep_name in json_deps
            if (m := pattern.match(dep_name)) and "extra" in m.group(0)
        }

        present_deps = set()
        for _, distribution_pkgs in importlib.metadata.packages_distributions().items():
            present_deps.update(set(distribution_pkgs))

        optional_present_deps = optional_deps & present_deps
        self.assertFalse(
            bool(optional_present_deps),
            f"Optional dependencies installed: {optional_present_deps}.\n"
            "Please ensure that optional dependencies are not installed in the core version, or list them as required.",
        )

    def test_core_pybamm_import(self):
        """Verify successful import of 'pybamm' without optional dependencies in the core PyBaMM version."""

        try:
            importlib.import_module("pybamm")
        except ModuleNotFoundError as error:
            self.fail(
                f"Import of 'pybamm' shouldn't require optional dependencies. Error: {error}"
            )
