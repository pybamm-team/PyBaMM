#
# Tests for the lithium-ion MPM model
#
import pybamm
import tests
import numpy as np
import unittest
from platform import system


class TestMPM(unittest.TestCase):
    def test_basic_processing(self):
        options = {"thermal": "isothermal"}
        model = pybamm.lithium_ion.MPM(options)
        # use Ecker parameters for nonlinear diffusion
        #param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Ecker2015)
        #param = self.add_distribution_params_for_test(param)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_basic_processing_1plus1D(self):
        options = {"current collector": "potential pair", "dimensionality": 1}

        model = pybamm.lithium_ion.MPM(options)
        var = pybamm.standard_spatial_vars
        var_pts = {
            var.x_n: 5,
            var.x_s: 5,
            var.x_p: 5,
            var.r_n: 5,
            var.r_p: 5,
            var.R_n: 5,
            var.R_p: 5,
            var.y: 5,
            var.z: 5,
        }
        modeltest = tests.StandardModelTest(model, var_pts=var_pts)
        modeltest.test_all(skip_output_tests=True)

    def test_basic_processing_2plus1D(self):
        options = {"current collector": "potential pair", "dimensionality": 2}
        model = pybamm.lithium_ion.MPM(options)
        var = pybamm.standard_spatial_vars
        var_pts = {
            var.x_n: 5,
            var.x_s: 5,
            var.x_p: 5,
            var.r_n: 5,
            var.r_p: 5,
            var.R_n: 5,
            var.R_p: 5,
            var.y: 5,
            var.z: 5,
        }
        modeltest = tests.StandardModelTest(model, var_pts=var_pts)
        modeltest.test_all(skip_output_tests=True)

    def test_optimisations(self):
        options = {"thermal": "isothermal"}
        model = pybamm.lithium_ion.MPM(options)
        optimtest = tests.OptimisationsTest(model)

        original = optimtest.evaluate_model()
        using_known_evals = optimtest.evaluate_model(use_known_evals=True)
        to_python = optimtest.evaluate_model(to_python=True)
        np.testing.assert_array_almost_equal(original, using_known_evals)
        np.testing.assert_array_almost_equal(original, to_python)

        if system() != "Windows":
            to_jax = optimtest.evaluate_model(to_jax=True)
            np.testing.assert_array_almost_equal(original, to_jax)

    def test_set_up(self):
        model = pybamm.lithium_ion.MPM()
        optimtest = tests.OptimisationsTest(model)
        optimtest.set_up_model(to_python=True)
        optimtest.set_up_model(to_python=False)

    def test_zero_current(self):
        options = {"thermal": "isothermal"}
        model = pybamm.lithium_ion.MPM(options)
        parameter_values = model.default_parameter_values
        parameter_values.update({"Current function [A]": 0})
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all()

    def test_thermal_lumped(self):
        options = {"thermal": "lumped"}
        model = pybamm.lithium_ion.MPM(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_particle_uniform(self):
        options = {"particle": "uniform profile"}
        model = pybamm.lithium_ion.MPM(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_loss_active_material_stress_negative(self):
        options = {"loss of active material": ("none", "stress-driven")}
        model = pybamm.lithium_ion.MPM(options)
        chemistry = pybamm.parameter_sets.Ai2020
        parameter_values = pybamm.ParameterValues(chemistry=chemistry)
        param = self.add_distribution_params_for_test(parameter_values)
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        modeltest.test_all()

    def test_loss_active_material_stress_positive(self):
        options = {"loss of active material": ("stress-driven", "none")}
        model = pybamm.lithium_ion.MPM(options)
        chemistry = pybamm.parameter_sets.Ai2020
        parameter_values = pybamm.ParameterValues(chemistry=chemistry)
        param = self.add_distribution_params_for_test(parameter_values)
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        modeltest.test_all()

    def test_loss_active_material_stress_both(self):
        options = {"loss of active material": "stress-driven"}
        model = pybamm.lithium_ion.MPM(options)
        chemistry = pybamm.parameter_sets.Ai2020
        parameter_values = pybamm.ParameterValues(chemistry=chemistry)
        param = self.add_distribution_params_for_test(parameter_values)
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        modeltest.test_all()

    def add_distribution_params_for_test(self, param):
        R_n_dim = param["Negative particle radius [m]"]
        R_p_dim = param["Positive particle radius [m]"]
        sd_a_n = 0.3
        sd_a_p = 0.3

        # Min and max radii
        R_min_n = 0
        R_min_p = 0
        R_max_n = 1 + sd_a_n * 5
        R_max_p = 1 + sd_a_p * 5

        def lognormal_distribution(R, R_av, sd):
            import numpy as np

            mu_ln = pybamm.log(R_av ** 2 / pybamm.sqrt(R_av ** 2 + sd ** 2))
            sigma_ln = pybamm.sqrt(pybamm.log(1 + sd ** 2 / R_av ** 2))
            return (
                pybamm.exp(-((pybamm.log(R) - mu_ln) ** 2) / (2 * sigma_ln ** 2))
                / pybamm.sqrt(2 * np.pi * sigma_ln ** 2)
                / (R)
            )

        # Set the dimensional (area-weighted) particle-size distributions
        def f_a_dist_n_dim(R):
            return lognormal_distribution(R, R_n_dim, sd_a_n * R_n_dim)

        def f_a_dist_p_dim(R):
            return lognormal_distribution(R, R_p_dim, sd_a_p * R_p_dim)

        # Append to parameter set
        param.update(
            {
                "Negative minimum particle radius [m]": R_min_n * R_n_dim,
                "Positive minimum particle radius [m]": R_min_p * R_p_dim,
                "Negative maximum particle radius [m]": R_max_n * R_n_dim,
                "Positive maximum particle radius [m]": R_max_p * R_p_dim,
                "Negative area-weighted "
                + "particle-size distribution [m-1]": f_a_dist_n_dim,
                "Positive area-weighted "
                + "particle-size distribution [m-1]": f_a_dist_p_dim,
            },
            check_already_exists=False,
        )
        return param


class TestMPMWithCrack(unittest.TestCase):
    def test_well_posed_negative_cracking(self):
        options = {"particle mechanics": ("swelling and cracking", "none")}
        model = pybamm.lithium_ion.MPM(options)
        chemistry = pybamm.parameter_sets.Ai2020
        parameter_values = pybamm.ParameterValues(chemistry=chemistry)
        param = self.add_distribution_params_for_test(parameter_values)
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        modeltest.test_all()

    def test_well_posed_positive_cracking(self):
        options = {"particle mechanics": ("none", "swelling and cracking")}
        model = pybamm.lithium_ion.MPM(options)
        chemistry = pybamm.parameter_sets.Ai2020
        parameter_values = pybamm.ParameterValues(chemistry=chemistry)
        param = self.add_distribution_params_for_test(parameter_values)
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        modeltest.test_all()

    def test_well_posed_both_cracking(self):
        options = {"particle mechanics": "swelling and cracking"}
        model = pybamm.lithium_ion.MPM(options)
        chemistry = pybamm.parameter_sets.Ai2020
        parameter_values = pybamm.ParameterValues(chemistry=chemistry)
        param = self.add_distribution_params_for_test(parameter_values)
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        modeltest.test_all()

    def test_well_posed_both_swelling_only(self):
        options = {"particle mechanics": "swelling only"}
        model = pybamm.lithium_ion.MPM(options)
        chemistry = pybamm.parameter_sets.Ai2020
        parameter_values = pybamm.ParameterValues(chemistry=chemistry)
        param = self.add_distribution_params_for_test(parameter_values)
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        modeltest.test_all()

    def add_distribution_params_for_test(self, param):
        R_n_dim = param["Negative particle radius [m]"]
        R_p_dim = param["Positive particle radius [m]"]
        sd_a_n = 0.3
        sd_a_p = 0.3

        # Min and max radii
        R_min_n = 0
        R_min_p = 0
        R_max_n = 1 + sd_a_n * 5
        R_max_p = 1 + sd_a_p * 5

        def lognormal_distribution(R, R_av, sd):
            import numpy as np

            mu_ln = pybamm.log(R_av ** 2 / pybamm.sqrt(R_av ** 2 + sd ** 2))
            sigma_ln = pybamm.sqrt(pybamm.log(1 + sd ** 2 / R_av ** 2))
            return (
                pybamm.exp(-((pybamm.log(R) - mu_ln) ** 2) / (2 * sigma_ln ** 2))
                / pybamm.sqrt(2 * np.pi * sigma_ln ** 2)
                / (R)
            )

        # Set the dimensional (area-weighted) particle-size distributions
        def f_a_dist_n_dim(R):
            return lognormal_distribution(R, R_n_dim, sd_a_n * R_n_dim)

        def f_a_dist_p_dim(R):
            return lognormal_distribution(R, R_p_dim, sd_a_p * R_p_dim)

        # Append to parameter set
        param.update(
            {
                "Negative minimum particle radius [m]": R_min_n * R_n_dim,
                "Positive minimum particle radius [m]": R_min_p * R_p_dim,
                "Negative maximum particle radius [m]": R_max_n * R_n_dim,
                "Positive maximum particle radius [m]": R_max_p * R_p_dim,
                "Negative area-weighted "
                + "particle-size distribution [m-1]": f_a_dist_n_dim,
                "Positive area-weighted "
                + "particle-size distribution [m-1]": f_a_dist_p_dim,
            },
            check_already_exists=False,
        )
        return param


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    sys.setrecursionlimit(10000)

    if "-v" in sys.argv:
        debug = True
    unittest.main()
