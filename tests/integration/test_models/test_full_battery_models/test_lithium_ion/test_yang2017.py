import pybamm
import tests


class TestYang2017:
    def test_basic_processing(self):
        model = pybamm.lithium_ion.Yang2017()
        parameter_values = pybamm.ParameterValues("OKane2022")
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all()
