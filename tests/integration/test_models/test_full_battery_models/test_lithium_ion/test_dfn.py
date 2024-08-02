#
# Tests for the lithium-ion DFN model
#

import pybamm
import tests
import numpy as np
import unittest
from tests import BaseIntegrationTestLithiumIon


class TestDFN(BaseIntegrationTestLithiumIon, unittest.TestCase):
    def setUp(self):
        self.model = pybamm.lithium_ion.DFN

    def test_particle_distribution_in_x(self):
        model = pybamm.lithium_ion.DFN()
        param = model.default_parameter_values
        L_n = model.param.n.L
        L_p = model.param.p.L
        L = model.param.L_x

        def negative_radius(x):
            return (1 + x / L_n) * 1e-5

        def positive_radius(x):
            return (1 + (x - L_p) / (L - L_p)) * 1e-5

        param["Negative particle radius [m]"] = negative_radius
        param["Positive particle radius [m]"] = positive_radius
        # Only get 3dp of accuracy in some tests at 1C with particle distribution
        # TODO: investigate if there is a bug or some way to improve the
        # implementation
        param["Current function [A]"] = 0.5 * param["Nominal cell capacity [A.h]"]
        self.run_basic_processing_test({}, parameter_values=param)


class TestDFNWithSizeDistribution(unittest.TestCase):
    def setUp(self):
        params = pybamm.ParameterValues("Marquis2019")
        self.params = pybamm.get_size_distribution_parameters(params)

        self.var_pts = {
            "x_n": 5,
            "x_s": 5,
            "x_p": 5,
            "r_n": 5,
            "r_p": 5,
            "R_n": 3,
            "R_p": 3,
            "y": 5,
            "z": 5,
        }

    def test_basic_processing(self):
        options = {"particle size": "distribution"}
        model = pybamm.lithium_ion.DFN(options)
        modeltest = tests.StandardModelTest(
            model, parameter_values=self.params, var_pts=self.var_pts
        )
        modeltest.test_all()

    def test_basic_processing_tuple(self):
        options = {"particle size": ("single", "distribution")}
        model = pybamm.lithium_ion.DFN(options)
        modeltest = tests.StandardModelTest(
            model, parameter_values=self.params, var_pts=self.var_pts
        )
        modeltest.test_all()

    def test_uniform_profile(self):
        options = {"particle size": "distribution", "particle": "uniform profile"}
        model = pybamm.lithium_ion.DFN(options)
        modeltest = tests.StandardModelTest(
            model, parameter_values=self.params, var_pts=self.var_pts
        )
        modeltest.test_all()

    def test_basic_processing_4D(self):
        # 4 dimensions: particle, particle size, electrode, current collector
        options = {
            "particle size": "distribution",
            "current collector": "potential pair",
            "dimensionality": 1,
        }
        model = pybamm.lithium_ion.DFN(options)
        modeltest = tests.StandardModelTest(
            model, parameter_values=self.params, var_pts=self.var_pts
        )
        modeltest.test_all(skip_output_tests=True)

    def test_conservation_each_electrode(self):
        # Test that surface areas are being calculated from the distribution correctly
        # for any discretization in the size domain.
        # We test that the amount of lithium removed or added to each electrode
        # is the same as for the standard DFN with the same parameters
        models = [
            pybamm.lithium_ion.DFN(),
            pybamm.lithium_ion.DFN(options={"particle size": "distribution"}),
        ]

        # reduce number of particle sizes, for a crude discretization
        var_pts = {"R_n": 3, "R_p": 3}
        solver = pybamm.CasadiSolver(mode="fast")

        # solve
        neg_Li = []
        pos_Li = []
        for model in models:
            sim = pybamm.Simulation(
                model, parameter_values=self.params, var_pts=self.var_pts, solver=solver
            )
            sim.var_pts.update(var_pts)
            solution = sim.solve([0, 3500])
            neg = solution["Total lithium in negative electrode [mol]"].entries[-1]
            pos = solution["Total lithium in positive electrode [mol]"].entries[-1]
            neg_Li.append(neg)
            pos_Li.append(pos)

        # compare
        np.testing.assert_array_almost_equal(neg_Li[0], neg_Li[1], decimal=12)
        np.testing.assert_array_almost_equal(pos_Li[0], pos_Li[1], decimal=12)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
