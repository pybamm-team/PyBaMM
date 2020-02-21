#
# Tests for the PyBaMM parameters management
# command line interface
#

import os
import subprocess
import pybamm
import csv
import tempfile
import pkg_resources
import unittest


class TestParametersCLI(unittest.TestCase):
    def test_add_param(self):
        # Read a parameter file thta is shipped with PyBaMM
        param_filename = pkg_resources.resource_filename(
            "pybamm",
            "input/parameters/lithium-ion/anodes/"
            "graphite_mcmb2528_Marquis2019/parameters.csv",
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
        cmd = ["pybamm_add_param", "-f", tempdir.name, "lithium-ion", "anodes"]
        subprocess.run(cmd, check=True)

        # Check that the new parameters can be accessed from the package
        # and that content is correct
        new_parameter_filename = pkg_resources.resource_filename(
            "pybamm",
            os.path.join(
                "input/parameters/lithium-ion/anodes/",
                os.path.basename(tempdir.name),
                "parameters.csv",
            ),
        )

        new_anode = pybamm.ParameterValues({}).read_parameters_csv(
            new_parameter_filename
        )
        self.assertEqual(new_anode["Reference temperature [K]"], "298.15")

        # Clean up directories
        tempdir.cleanup()  # Remove temporary local directory
        os.remove(new_parameter_filename)  # Remove parameters.csv file
        os.rmdir(os.path.dirname(new_parameter_filename))  # Remove (now empty) dir
