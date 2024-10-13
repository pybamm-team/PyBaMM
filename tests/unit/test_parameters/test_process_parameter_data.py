#
# Tests for the parameter processing functions
#

import numpy as np
import pybamm
import pytest
from pathlib import Path

@pytest.fixture
def parameters_path():
    return Path(__file__).parent.resolve()

@pytest.fixture(params=[
    ("lico2_ocv_example", pybamm.parameters.process_1D_data),
    ("lico2_diffusivity_Dualfoil1998_2D", pybamm.parameters.process_2D_data),
    ("data_for_testing_2D", pybamm.parameters.process_2D_data_csv),
    ("data_for_testing_3D", pybamm.parameters.process_3D_data_csv),
])
def parameter_data(request, path):
    name, processing_function = request.param
    processed = processing_function(name, parameters_path)
    return name, processed


class TestProcessParameterData:
    def test_processed_name(self, parameter_data):
        name, processed = parameter_data
        assert processed[0] == name

    def test_processed_structure(self, parameter_data):
        name, processed = parameter_data
        assert isinstance(processed[1], tuple)
        assert isinstance(processed[1][0][0], np.ndarray)
        assert isinstance(processed[1][1], np.ndarray)

        if len(processed[1][0]) > 1:
            assert isinstance(processed[1][0][1], np.ndarray)

        elif len(processed[1]) == 3:
            assert isinstance(processed[1][0][1], np.ndarray)
            assert isinstance(processed[1][0][2], np.ndarray)

    def test_error(self):
        with pytest.raises(FileNotFoundError, match="Could not find file"):
            pybamm.parameters.process_1D_data("not_a_real_file", "not_a_real_path")
