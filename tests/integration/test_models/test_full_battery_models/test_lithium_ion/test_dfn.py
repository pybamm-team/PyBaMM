#
# Tests for the lithium-ion DFN model
#
import pybamm
import tests

import numpy as np
import unittest


class TestDFN(unittest.TestCase):
    def test_basic_processing(self):
        options = {"thermal": "isothermal"}
        model = pybamm.lithium_ion.DFN(options)
        # use Ecker parameters for nonlinear diffusion
        param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Ecker2015)
        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 5, var.r_p: 5}
        modeltest = tests.StandardModelTest(
            model, parameter_values=param, var_pts=var_pts
        )
        modeltest.test_all()

    def test_sensitivities(self):
        options = {"thermal": "isothermal"}
        model = pybamm.lithium_ion.DFN(options)
        # use Ecker parameters for nonlinear diffusion
        param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Ecker2015)
        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 5, var.r_p: 5}
        modeltest = tests.StandardModelTest(
            model, parameter_values=param, var_pts=var_pts
        )
        modeltest.test_sensitivities(
            "Current function [A]",
            0.15652,
        )

    def test_basic_processing_1plus1D(self):
        options = {"current collector": "potential pair", "dimensionality": 1}
        model = pybamm.lithium_ion.DFN(options)
        var = pybamm.standard_spatial_vars
        var_pts = {
            var.x_n: 5,
            var.x_s: 5,
            var.x_p: 5,
            var.r_n: 5,
            var.r_p: 5,
            var.y: 5,
            var.z: 5,
        }
        modeltest = tests.StandardModelTest(model, var_pts=var_pts)
        modeltest.test_all(skip_output_tests=True)

    def test_basic_processing_2plus1D(self):
        options = {"current collector": "potential pair", "dimensionality": 2}
        model = pybamm.lithium_ion.DFN(options)
        var = pybamm.standard_spatial_vars
        var_pts = {
            var.x_n: 5,
            var.x_s: 5,
            var.x_p: 5,
            var.r_n: 5,
            var.r_p: 5,
            var.y: 5,
            var.z: 5,
        }
        modeltest = tests.StandardModelTest(model, var_pts=var_pts)
        modeltest.test_all(skip_output_tests=True)

    def test_optimisations(self):
        options = {"thermal": "isothermal"}
        model = pybamm.lithium_ion.DFN(options)
        optimtest = tests.OptimisationsTest(model)

        original = optimtest.evaluate_model()
        using_known_evals = optimtest.evaluate_model(use_known_evals=True)
        to_python = optimtest.evaluate_model(to_python=True)
        np.testing.assert_array_almost_equal(original, using_known_evals)
        np.testing.assert_array_almost_equal(original, to_python)

    def test_set_up(self):
        model = pybamm.lithium_ion.DFN()
        optimtest = tests.OptimisationsTest(model)
        optimtest.set_up_model(to_python=True)
        optimtest.set_up_model(to_python=False)

    def test_full_thermal(self):
        options = {"thermal": "x-full"}
        model = pybamm.lithium_ion.DFN(options)
        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 5, var.r_p: 5}
        modeltest = tests.StandardModelTest(model, var_pts=var_pts)
        modeltest.test_all()

    def test_lumped_thermal(self):
        options = {"thermal": "lumped"}
        model = pybamm.lithium_ion.DFN(options)
        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 5, var.r_p: 5}
        modeltest = tests.StandardModelTest(model, var_pts=var_pts)
        modeltest.test_all()

    def test_particle_uniform(self):
        options = {"particle": "uniform profile"}
        model = pybamm.lithium_ion.DFN(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_particle_quadratic(self):
        options = {"particle": "quadratic profile"}
        model = pybamm.lithium_ion.DFN(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_particle_quartic(self):
        options = {"particle": "quartic profile"}
        model = pybamm.lithium_ion.DFN(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_loss_active_material_stress_negative(self):
        options = {"loss of active material": ("stress-driven", "none")}
        model = pybamm.lithium_ion.DFN(options)
        chemistry = pybamm.parameter_sets.Ai2020
        parameter_values = pybamm.ParameterValues(chemistry=chemistry)
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all()

    def test_loss_active_material_stress_positive(self):
        options = {"loss of active material": ("none", "stress-driven")}
        model = pybamm.lithium_ion.DFN(options)
        chemistry = pybamm.parameter_sets.Ai2020
        parameter_values = pybamm.ParameterValues(chemistry=chemistry)
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all()

    def test_loss_active_material_stress_both(self):
        options = {"loss of active material": "stress-driven"}
        model = pybamm.lithium_ion.DFN(options)
        chemistry = pybamm.parameter_sets.Ai2020
        parameter_values = pybamm.ParameterValues(chemistry=chemistry)
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all()

    def test_loss_active_material_reaction_both(self):
        options = {"loss of active material": "reaction-driven"}
        model = pybamm.lithium_ion.DFN(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_surface_form_differential(self):
        options = {"surface form": "differential"}
        model = pybamm.lithium_ion.DFN(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_surface_form_algebraic(self):
        options = {"surface form": "algebraic"}
        model = pybamm.lithium_ion.DFN(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_particle_distribution_in_x(self):
        model = pybamm.lithium_ion.DFN()
        param = model.default_parameter_values
        L_n = model.param.L_n
        L_p = model.param.L_p
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
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        modeltest.test_all()

    def test_well_posed_irreversible_plating_with_porosity(self):
        options = {
            "lithium plating": "irreversible",
            "lithium plating porosity change": "true",
        }
        model = pybamm.lithium_ion.DFN(options)
        param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Chen2020_plating)
        modeltest = tests.StandardModelTest(model, parameter_values=param)
        modeltest.test_all()


class TestDFNWithSEI(unittest.TestCase):
    def test_well_posed_constant(self):
        options = {"SEI": "constant"}
        model = pybamm.lithium_ion.DFN(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_well_posed_reaction_limited(self):
        options = {"SEI": "reaction limited"}
        model = pybamm.lithium_ion.DFN(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_well_posed_reaction_limited_average_film_resistance(self):
        options = {"SEI": "reaction limited", "SEI film resistance": "average"}
        model = pybamm.lithium_ion.DFN(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_well_posed_solvent_diffusion_limited(self):
        options = {"SEI": "solvent-diffusion limited"}
        model = pybamm.lithium_ion.DFN(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_well_posed_electron_migration_limited(self):
        options = {"SEI": "electron-migration limited"}
        model = pybamm.lithium_ion.DFN(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_well_posed_interstitial_diffusion_limited(self):
        options = {"SEI": "interstitial-diffusion limited"}
        model = pybamm.lithium_ion.DFN(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_well_posed_ec_reaction_limited(self):
        options = {"SEI": "ec reaction limited", "SEI porosity change": "true"}
        model = pybamm.lithium_ion.DFN(options)
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()


class TestDFNWithMechanics(unittest.TestCase):
    def test_well_posed_negative_cracking(self):
        options = {"particle mechanics": ("swelling and cracking", "none")}
        model = pybamm.lithium_ion.DFN(options)
        chemistry = pybamm.parameter_sets.Ai2020
        parameter_values = pybamm.ParameterValues(chemistry=chemistry)
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all()

    def test_well_posed_positive_cracking(self):
        options = {"particle mechanics": ("none", "swelling and cracking")}
        model = pybamm.lithium_ion.DFN(options)
        chemistry = pybamm.parameter_sets.Ai2020
        parameter_values = pybamm.ParameterValues(chemistry=chemistry)
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all()

    def test_well_posed_both_cracking(self):
        options = {"particle mechanics": "swelling and cracking"}
        model = pybamm.lithium_ion.DFN(options)
        chemistry = pybamm.parameter_sets.Ai2020
        parameter_values = pybamm.ParameterValues(chemistry=chemistry)
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all()

    def test_well_posed_both_swelling_only(self):
        options = {"particle mechanics": "swelling only"}
        model = pybamm.lithium_ion.DFN(options)
        chemistry = pybamm.parameter_sets.Ai2020
        parameter_values = pybamm.ParameterValues(chemistry=chemistry)
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all()


class TestDFNWithSizeDistribution(unittest.TestCase):
    def setUp(self):
        params = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Marquis2019)
        self.params = pybamm.get_size_distribution_parameters(params)

        var = pybamm.standard_spatial_vars
        self.var_pts = {
            var.x_n: 5,
            var.x_s: 5,
            var.x_p: 5,
            var.r_n: 5,
            var.r_p: 5,
            var.R_n: 3,
            var.R_p: 3,
            var.y: 5,
            var.z: 5,
        }

    def test_basic_processing(self):
        options = {"particle size": "distribution"}
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
        var = pybamm.standard_spatial_vars

        # reduce number of particle sizes, for a crude discretization
        var_pts = {
            var.R_n: 3,
            var.R_p: 3,
        }
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
