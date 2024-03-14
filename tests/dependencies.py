import importlib
import re

from tests import TestCase
import sys
import unittest


class TestDependencies(TestCase):
    """
    This class tests the dependencies required by PyBaMM for specific versions.

    **Note:** This test module is **not run automatically** with other tests.
    Its functions are intended to be tested manually on different PyBaMM versions.
    """

    def test_core_optional_dependencies(self):
        """Ensure optional dependencies are not installed in the core PyBaMM version."""

        pattern = re.compile(r"^([^>=;\[]+)\b.*$")
        json_deps = importlib.metadata.metadata("pybamm").json["requires_dist"]

        optional_distribution_deps = {
            pattern.match(dep_name).group(1)
            for dep_name in json_deps
            if "extra" in dep_name and "pybamm" not in dep_name
        }

        present_distribution_deps = set()
        for _, distribution_pkgs in importlib.metadata.packages_distributions().items():
            present_distribution_deps.update(set(distribution_pkgs))

        self.assertFalse(bool(optional_distribution_deps & present_distribution_deps))

    def test_core_pybamm_import(self):
        """Verify successful import of 'pybamm' without optional dependencies in the core PyBaMM version."""

        try:
            importlib.import_module("pybamm")
        except ImportError as error:
            self.fail(
                f"Import of 'pybamm' shouldn't require optional dependencies. Error: {error}"
            )


if __name__ == "__main__":
    print("Add -v for more debug output")

    if "-v" in sys.argv:
        debug = True
    unittest.main()
