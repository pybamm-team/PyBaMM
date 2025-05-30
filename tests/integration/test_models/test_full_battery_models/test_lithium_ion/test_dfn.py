#
# Tests for the lithium-ion DFN model
#
import numpy as np
import pytest

import pybamm
import tests
from tests import BaseIntegrationTestLithiumIon


class TestDFN(BaseIntegrationTestLithiumIon):
    @pytest.fixture(autouse=True)
    def setup(self):
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


class TestDFNWithSizeDistribution:
    @pytest.fixture(autouse=True)
    def setup(self):
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
            "R_n_prim": 3,
            "R_n_sec": 3,
            "R_p_prim": 3,
            "R_p_sec": 3,
        }

    def test_basic_processing(self):
        options = {"particle size": "distribution"}
        model = pybamm.lithium_ion.DFN(options)
        modeltest = tests.StandardModelTest(
            model, parameter_values=self.params, var_pts=self.var_pts
        )
        modeltest.test_all()

    def test_composite(self):
        options = {
            "particle phases": ("2", "1"),
            "open-circuit potential": (("single", "current sigmoid"), "single"),
            "particle size": "distribution",
        }
        parameter_values = pybamm.ParameterValues("Chen2020_composite")
        name = "Negative electrode active material volume fraction"
        x = 0.1
        parameter_values.update(
            {f"Primary: {name}": (1 - x) * 0.75, f"Secondary: {name}": x * 0.75}
        )
        parameter_values = pybamm.get_size_distribution_parameters(
            parameter_values,
            composite="negative",
            R_min_n_prim=0.9,
            R_min_n_sec=0.9,
            R_max_n_prim=1.1,
            R_max_n_sec=1.1,
        )
        # self.run_basic_processing_test(options, parameter_values=parameter_values)
        model = pybamm.lithium_ion.DFN(options)
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
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

        # solve
        neg_Li = []
        pos_Li = []
        t_eval = [0, 3500]
        t_interp = np.linspace(t_eval[0], t_eval[-1], 100)
        for model in models:
            sim = pybamm.Simulation(
                model, parameter_values=self.params, var_pts=self.var_pts
            )
            sim.var_pts.update(var_pts)
            solution = sim.solve(t_eval, t_interp=t_interp)
            neg = solution["Total lithium in negative electrode [mol]"].entries[-1]
            pos = solution["Total lithium in positive electrode [mol]"].entries[-1]
            neg_Li.append(neg)
            pos_Li.append(pos)

        # compare
        np.testing.assert_allclose(neg_Li[0], neg_Li[1], rtol=1e-13, atol=1e-12)
        np.testing.assert_allclose(pos_Li[0], pos_Li[1], rtol=1e-13, atol=1e-12)

    def test_basic_processing_nonlinear_diffusion(self):
        options = {
            "particle size": "distribution",
        }
        model = pybamm.lithium_ion.DFN(options)
        # Ecker2015 has a nonlinear diffusion coefficient
        parameter_values = pybamm.ParameterValues("Ecker2015")
        parameter_values = pybamm.get_size_distribution_parameters(parameter_values)
        modeltest = tests.StandardModelTest(
            model, parameter_values=parameter_values, var_pts=self.var_pts
        )
        modeltest.test_all(skip_output_tests=True)
