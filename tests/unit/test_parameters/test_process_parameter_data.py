#
# Tests for the parameter processing functions
#


import os
import numpy as np
import pybamm

import unittest


class TestProcessParameterData(unittest.TestCase):
    def test_process_1D_data(self):
        name = "lico2_ocv_example"
        path = os.path.join(pybamm.root_dir(), "tests", "unit", "test_parameters")
        processed = pybamm.parameters.process_1D_data(name, path)
        self.assertEqual(processed[0], name)
        self.assertIsInstance(processed[1], tuple)
        self.assertIsInstance(processed[1][0][0], np.ndarray)
        self.assertIsInstance(processed[1][1], np.ndarray)

    def test_process_2D_data(self):
        name = "lico2_diffusivity_Dualfoil1998_2D"
        path = os.path.join(pybamm.root_dir(), "tests", "unit", "test_parameters")
        processed = pybamm.parameters.process_2D_data(name, path)
        self.assertEqual(processed[0], name)
        self.assertIsInstance(processed[1], tuple)
        self.assertIsInstance(processed[1][0][0], np.ndarray)
        self.assertIsInstance(processed[1][0][1], np.ndarray)
        self.assertIsInstance(processed[1][1], np.ndarray)

    def test_process_2D_data_csv(self):
        name = "data_for_testing_2D"
        path = os.path.join(pybamm.root_dir(), "tests", "unit", "test_parameters")
        processed = pybamm.parameters.process_2D_data_csv(name, path)

        self.assertEqual(processed[0], name)
        self.assertIsInstance(processed[1], tuple)
        self.assertIsInstance(processed[1][0][0], np.ndarray)
        self.assertIsInstance(processed[1][0][1], np.ndarray)
        self.assertIsInstance(processed[1][1], np.ndarray)

    def test_process_3D_data_csv(self):
        name = "data_for_testing_3D"
        path = os.path.join(pybamm.root_dir(), "tests", "unit", "test_parameters")
        processed = pybamm.parameters.process_3D_data_csv(name, path)

        self.assertEqual(processed[0], name)
        self.assertIsInstance(processed[1], tuple)
        self.assertIsInstance(processed[1][0][0], np.ndarray)
        self.assertIsInstance(processed[1][0][1], np.ndarray)
        self.assertIsInstance(processed[1][0][2], np.ndarray)
        self.assertIsInstance(processed[1][1], np.ndarray)

    def test_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "Could not find file"):
            pybamm.parameters.process_1D_data("not_a_real_file", "not_a_real_path")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
