import pybamm
import tests


class TestThevenin:
    def test_basic_processing(self):
        model = pybamm.equivalent_circuit.Thevenin()
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()
