import importlib
import re

from tests import TestCase
import sys
import unittest


class TestDependencies(TestCase):
    """
    Test of dependencies for specific versions of PyBaMM
    """

    # Test that optional dependencies are not installed in the core version of PyBaMM
    def test_optional_dependencies(self):
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


if __name__ == "__main__":
    print("Add -v for more debug output")

    if "-v" in sys.argv:
        debug = True
    unittest.main()
