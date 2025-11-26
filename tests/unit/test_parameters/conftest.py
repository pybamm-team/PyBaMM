import pytest

import pybamm


@pytest.fixture
def simple_param():
    return pybamm.ParameterValues({"a": 1})
