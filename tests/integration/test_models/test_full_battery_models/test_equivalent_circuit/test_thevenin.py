import pybamm
import tests


class TestThevenin:
    def test_basic_processing(self):
        model = pybamm.equivalent_circuit.Thevenin()
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_diffusion(self):
        model = pybamm.equivalent_circuit.Thevenin(
            options={"diffusion element": "true"}
        )
        parameter_values = model.default_parameter_values

        parameter_values.update(
            {"Diffusion time constant [s]": 580}, check_already_exists=False
        )
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all()
