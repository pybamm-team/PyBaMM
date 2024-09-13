#
# Tests for the parameter processing functions
#


import os
import numpy as np
import pybamm

import pytest


class TestProcessParameterData:
    def test_process_1D_data(self):
        name = "lico2_ocv_example"
        path = os.path.abspath(os.path.dirname(__file__))
        processed = pybamm.parameters.process_1D_data(name, path)
        assert processed[0] == name
        assert isinstance(processed[1], tuple)
        assert isinstance(processed[1][0][0], np.ndarray)
        assert isinstance(processed[1][1], np.ndarray)

    def test_process_2D_data(self):
        name = "lico2_diffusivity_Dualfoil1998_2D"
        path = os.path.abspath(os.path.dirname(__file__))
        processed = pybamm.parameters.process_2D_data(name, path)
        assert processed[0] == name
        assert isinstance(processed[1], tuple)
        assert isinstance(processed[1][0][0], np.ndarray)
        assert isinstance(processed[1][0][1], np.ndarray)
        assert isinstance(processed[1][1], np.ndarray)

    def test_process_2D_data_csv(self):
        name = "data_for_testing_2D"
        path = os.path.abspath(os.path.dirname(__file__))
        processed = pybamm.parameters.process_2D_data_csv(name, path)

        assert processed[0] == name
        assert isinstance(processed[1], tuple)
        assert isinstance(processed[1][0][0], np.ndarray)
        assert isinstance(processed[1][0][1], np.ndarray)
        assert isinstance(processed[1][1], np.ndarray)

    def test_process_3D_data_csv(self):
        name = "data_for_testing_3D"
        path = os.path.abspath(os.path.dirname(__file__))
        processed = pybamm.parameters.process_3D_data_csv(name, path)

        assert processed[0] == name
        assert isinstance(processed[1], tuple)
        assert isinstance(processed[1][0][0], np.ndarray)
        assert isinstance(processed[1][0][1], np.ndarray)
        assert isinstance(processed[1][0][2], np.ndarray)
        assert isinstance(processed[1][1], np.ndarray)

    def test_error(self):
        with pytest.raises(FileNotFoundError, match="Could not find file"):
            pybamm.parameters.process_1D_data("not_a_real_file", "not_a_real_path")
