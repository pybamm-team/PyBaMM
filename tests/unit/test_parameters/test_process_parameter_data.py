#
# Tests for the parameter processing functions
#

from pathlib import Path

import numpy as np
import pytest

import pybamm


@pytest.fixture
def parameters_path():
    return Path(__file__).parent.resolve()


@pytest.fixture(
    params=[
        ("lico2_ocv_example", pybamm.parameters.process_1D_data),
        ("lico2_diffusivity_Dualfoil1998_2D", pybamm.parameters.process_2D_data),
        ("data_for_testing_2D", pybamm.parameters.process_2D_data_csv),
        ("data_for_testing_3D", pybamm.parameters.process_3D_data_csv),
    ]
)
def parameter_data(request, parameters_path):
    name, processing_function = request.param
    processed = processing_function(name, parameters_path)
    return name, processed


class TestProcessParameterData:
    def test_processed_name(self, parameter_data):
        name, processed = parameter_data
        assert processed[0] == name

    def test_processed_structure(self, parameter_data):
        _name, processed = parameter_data
        coordinates, values = processed[1][:2]

        assert isinstance(processed[1], tuple)
        assert all(isinstance(coordinate, np.ndarray) for coordinate in coordinates)
        assert isinstance(values, np.ndarray)

    def test_error(self):
        with pytest.raises(FileNotFoundError, match=r"Could not find file"):
            pybamm.parameters.process_1D_data("not_a_real_file", "not_a_real_path")
