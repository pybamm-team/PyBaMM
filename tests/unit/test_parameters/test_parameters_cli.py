#
# Tests for the PyBaMM parameters management
# command line interface
#

import os
import subprocess
import pybamm
import csv
import tempfile
import unittest
import platform


@unittest.skipUnless(platform.system() != "Windows", "Skipped for Windows")
class TestParametersCLI(unittest.TestCase):
    def test_add_rm_param(self):
        # Read a parameter file thta is shipped with PyBaMM
        param_pkg_dir = os.path.join(pybamm.__path__[0], "input", "parameters")
        param_filename = os.path.join(
            param_pkg_dir,
            "lithium-ion",
            "anodes",
            "graphite_mcmb2528_Marquis2019",
            "parameters.csv",
        )

        anode = pybamm.ParameterValues({}).read_parameters_csv(param_filename)

        # Write these parameters in current working dir. to mimic
        # user-defined parameters
        tempdir = tempfile.TemporaryDirectory()
        new_parameter_file = os.path.join(tempdir.name, "parameters.csv")
        with open(new_parameter_file, "w", newline="") as csvfile:
            fieldnames = ["Name [units]", "Value"]
            writer = csv.writer(csvfile)
            writer.writerow(fieldnames)
            for row in anode.items():
                writer.writerow(row)

        # Use pybamm command line to add new parameters under
        # test_parameters_dir directory
        cmd = ["pybamm_add_parameter", "-f", tempdir.name, "lithium-ion", "anodes"]
        subprocess.run(cmd, check=True)

        # Check that the new parameters can be accessed from the package
        # and that content is correct
        new_parameter_filename = os.path.join(
            param_pkg_dir,
            "lithium-ion",
            "anodes",
            os.path.basename(tempdir.name),
            "parameters.csv",
        )
        self.assertTrue(os.path.isfile(new_parameter_filename))

        new_anode = pybamm.ParameterValues({}).read_parameters_csv(
            new_parameter_filename
        )
        self.assertEqual(new_anode["Negative electrode porosity"], "0.3")

        # Now delete added parameter
        cmd = ["pybamm_rm_parameter", "-f", tempdir.name, "lithium-ion", "anodes"]
        subprocess.run(cmd, check=True)
        self.assertFalse(os.path.isfile(new_parameter_filename))

        # Clean up directories
        tempdir.cleanup()  # Remove temporary local directory

    def test_edit_param(self):
        anodes_dir = os.path.join("input", "parameters", "lithium-ion", "anodes")
        chemistry = "lithium-ion"
        # Write dummy parameters.csv file in temporary directory
        # in package input dir
        tempdir = tempfile.TemporaryDirectory(
            dir=os.path.join(pybamm.__path__[0], anodes_dir)
        )
        with open(os.path.join(tempdir.name, "parameters.csv"), "w") as f:
            f.write("hello")

        # Create a temporary directory to perform this test in isolation
        sandbox_dir = tempfile.TemporaryDirectory()

        # Copy temporary dir in package to current working directory
        cmd = [
            "pybamm_edit_parameter",
            "-f",
            chemistry,
        ]
        subprocess.run(cmd, cwd=sandbox_dir.name)

        # Read and compare copied parameters.csv file
        copied_path_parameters_file = os.path.join(
            sandbox_dir.name,
            chemistry,
            "anodes",
            os.path.basename(tempdir.name),
            "parameters.csv",
        )
        with open(copied_path_parameters_file, "r") as f:
            content = f.read()
            self.assertTrue(content == "hello")

        # Clean up temporary dicts
        sandbox_dir.cleanup()
        tempdir.cleanup()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
