#
# Tests for the lithium-ion MPM model
#
from tests import TestCase
import pybamm
import tests
import numpy as np
import unittest


class TestMPM(TestCase):
    def test_basic_processing(self):
        options = {"thermal": "isothermal"}
        model = pybamm.lithium_ion.MPM(options)
        # use Ecker parameters for nonlinear diffusion
        param = pybamm.ParameterValues("Ecker2015")
        param = pybamm.get_size_distribution_parameters(param)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_optimisations(self):
        options = {"thermal": "isothermal"}
        model = pybamm.lithium_ion.MPM(options)
        optimtest = tests.OptimisationsTest(model)

        original = optimtest.evaluate_model()
        to_python = optimtest.evaluate_model(to_python=True)
        np.testing.assert_array_almost_equal(original, to_python)

        if pybamm.have_jax():
            to_jax = optimtest.evaluate_model(to_jax=True)
            np.testing.assert_array_almost_equal(original, to_jax)

    def test_set_up(self):
        model = pybamm.lithium_ion.MPM()
        optimtest = tests.OptimisationsTest(model)
        optimtest.set_up_model(to_python=True)
        optimtest.set_up_model(to_python=False)

    def test_particle_uniform(self):
        options = {"particle": "uniform profile"}
        model = pybamm.lithium_ion.MPM(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_differential_surface_form(self):
        options = {"surface form": "differential"}
        model = pybamm.lithium_ion.MPM(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_current_sigmoid_ocp(self):
        options = {"open-circuit potential": ("current sigmoid", "single")}
        model = pybamm.lithium_ion.MPM(options)
        parameter_values = pybamm.ParameterValues("Chen2020")
        parameter_values = pybamm.get_size_distribution_parameters(parameter_values)
        parameter_values.update(
            {
                "Negative electrode lithiation OCP [V]"
                "": parameter_values["Negative electrode OCP [V]"],
                "Negative electrode delithiation OCP [V]"
                "": parameter_values["Negative electrode OCP [V]"],
            },
            check_already_exists=False,
        )
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all(skip_output_tests=True)

    def test_voltage_control(self):
        options = {"operating mode": "voltage"}
        model = pybamm.lithium_ion.MPM(options)
        param = model.default_parameter_values
        param.update({"Voltage function [V]": 3.8}, check_already_exists=False)
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        modeltest.test_all(skip_output_tests=True)

    def test_conservation_each_electrode(self):
        # Test that surface areas are being calculated from the distribution correctly
        # for any discretization in the size domain.
        # We test that the amount of lithium removed or added to each electrode
        # is the same as for the SPM with the same parameters
        models = [pybamm.lithium_ion.SPM(), pybamm.lithium_ion.MPM()]

        # reduce number of particle sizes, for a crude discretization
        var_pts = {"R_n": 3, "R_p": 3}
        solver = pybamm.CasadiSolver(mode="fast")

        # solve
        neg_Li = []
        pos_Li = []
        for model in models:
            sim = pybamm.Simulation(model, solver=solver)
            sim.var_pts.update(var_pts)
            solution = sim.solve([0, 3500])
            neg = solution["Total lithium in negative electrode [mol]"].entries[-1]
            pos = solution["Total lithium in positive electrode [mol]"].entries[-1]
            neg_Li.append(neg)
            pos_Li.append(pos)

        # compare
        np.testing.assert_array_almost_equal(neg_Li[0], neg_Li[1], decimal=13)
        np.testing.assert_array_almost_equal(pos_Li[0], pos_Li[1], decimal=13)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
