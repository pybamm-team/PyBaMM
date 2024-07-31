#
# Base integration tests for lithium-ion models
#
import pybamm
import tests

import numpy as np


class BaseIntegrationTestLithiumIon:
    def run_basic_processing_test(self, options, **kwargs):
        model = self.model(options)
        modeltest = tests.StandardModelTest(model, **kwargs)
        modeltest.test_all()

    def test_basic_processing(self):
        options = {}
        self.run_basic_processing_test(options)

    def test_sensitivities(self):
        model = self.model()
        param = pybamm.ParameterValues("Ecker2015")
        rtol = 1e-6
        atol = 1e-6
        if pybamm.have_idaklu():
            solver = pybamm.IDAKLUSolver(rtol=rtol, atol=atol)
        else:
            solver = pybamm.CasadiSolver(rtol=rtol, atol=atol)
        modeltest = tests.StandardModelTest(
            model, parameter_values=param, solver=solver
        )
        modeltest.test_sensitivities("Current function [A]", 0.15652)

    def test_basic_processing_1plus1D(self):
        options = {"current collector": "potential pair", "dimensionality": 1}
        var_pts = {"x_n": 5, "x_s": 5, "x_p": 5, "r_n": 5, "r_p": 5, "y": 5, "z": 5}
        model = self.model(options)
        modeltest = tests.StandardModelTest(model, var_pts=var_pts)
        modeltest.test_all(skip_output_tests=True)

    def test_basic_processing_2plus1D(self):
        options = {"current collector": "potential pair", "dimensionality": 2}
        var_pts = {"x_n": 5, "x_s": 5, "x_p": 5, "r_n": 5, "r_p": 5, "y": 5, "z": 5}
        model = self.model(options)
        modeltest = tests.StandardModelTest(model, var_pts=var_pts)
        modeltest.test_all(skip_output_tests=True)

    def test_optimisations(self):
        model = self.model()
        optimtest = tests.OptimisationsTest(model)

        original = optimtest.evaluate_model()
        to_python = optimtest.evaluate_model(to_python=True)
        np.testing.assert_array_almost_equal(original, to_python)

        if pybamm.have_jax():
            to_jax = optimtest.evaluate_model(to_jax=True)
            np.testing.assert_array_almost_equal(original, to_jax)

    def test_set_up(self):
        model = self.model()
        optimtest = tests.OptimisationsTest(model)
        optimtest.set_up_model(to_python=True)
        optimtest.set_up_model(to_python=False)

    def test_charge(self):
        options = {"thermal": "isothermal"}
        parameter_values = pybamm.ParameterValues("Marquis2019")
        parameter_values.update({"Current function [A]": -1})
        self.run_basic_processing_test(options, parameter_values=parameter_values)

    def test_zero_current(self):
        options = {"thermal": "isothermal"}
        parameter_values = pybamm.ParameterValues("Marquis2019")
        parameter_values.update({"Current function [A]": 0})
        self.run_basic_processing_test(options, parameter_values=parameter_values)

    def test_lumped_thermal(self):
        options = {"thermal": "lumped"}
        self.run_basic_processing_test(options)

    def test_full_thermal(self):
        options = {"thermal": "x-full"}
        self.run_basic_processing_test(options)

    def test_particle_uniform(self):
        options = {"particle": "uniform profile"}
        self.run_basic_processing_test(options)

    def test_particle_quadratic(self):
        options = {"particle": "quadratic profile"}
        self.run_basic_processing_test(options)

    def test_particle_quartic(self):
        options = {"particle": "quartic profile"}
        self.run_basic_processing_test(options)

    def test_constant_utilisation(self):
        options = {"interface utilisation": "constant"}
        parameter_values = pybamm.ParameterValues("Marquis2019")
        parameter_values.update(
            {
                "Initial negative electrode interface utilisation": 0.9,
                "Initial positive electrode interface utilisation": 0.8,
            },
            check_already_exists=False,
        )
        self.run_basic_processing_test(options, parameter_values=parameter_values)

    def test_current_driven_utilisation(self):
        options = {"interface utilisation": "current-driven"}
        parameter_values = pybamm.ParameterValues("Marquis2019")
        parameter_values.update(
            {
                "Initial negative electrode interface utilisation": 0.9,
                "Initial positive electrode interface utilisation": 0.8,
                "Negative electrode current-driven interface utilisation factor "
                "[m3.mol-1]": -1e-5,
                "Positive electrode current-driven interface utilisation factor "
                "[m3.mol-1]": 1e-5,
            },
            check_already_exists=False,
        )
        self.run_basic_processing_test(options, parameter_values=parameter_values)

    def test_surface_form_differential(self):
        options = {"surface form": "differential"}
        self.run_basic_processing_test(options)

    def test_surface_form_algebraic(self):
        options = {"surface form": "algebraic"}
        self.run_basic_processing_test(options)

    def test_kinetics_asymmetric_butler_volmer(self):
        options = {"intercalation kinetics": "asymmetric Butler-Volmer"}
        solver = pybamm.CasadiSolver(atol=1e-14, rtol=1e-14)

        parameter_values = pybamm.ParameterValues("Marquis2019")
        parameter_values.update(
            {
                "Negative electrode Butler-Volmer transfer coefficient": 0.6,
                "Positive electrode Butler-Volmer transfer coefficient": 0.6,
            },
            check_already_exists=False,
        )
        self.run_basic_processing_test(
            options, parameter_values=parameter_values, solver=solver
        )

    def test_kinetics_linear(self):
        options = {"intercalation kinetics": "linear"}
        self.run_basic_processing_test(options)

    def test_kinetics_mhc(self):
        options = {"intercalation kinetics": "Marcus-Hush-Chidsey"}
        parameter_values = pybamm.ParameterValues("Marquis2019")
        parameter_values.update(
            {
                "Negative electrode reorganization energy [eV]": 0.35,
                "Positive electrode reorganization energy [eV]": 0.35,
                "Positive electrode exchange-current density [A.m-2]": 5,
            },
            check_already_exists=False,
        )
        self.run_basic_processing_test(options, parameter_values=parameter_values)

    def test_irreversible_plating_with_porosity(self):
        options = {
            "lithium plating": "irreversible",
            "lithium plating porosity change": "true",
        }
        param = pybamm.ParameterValues("OKane2022")
        self.run_basic_processing_test(options, parameter_values=param)

    def test_sei_reaction_limited(self):
        options = {"SEI": "reaction limited"}
        self.run_basic_processing_test(options)

    def test_sei_asymmetric_reaction_limited(self):
        options = {"SEI": "reaction limited (asymmetric)"}
        parameter_values = pybamm.ParameterValues("Marquis2019")
        parameter_values.update(
            {"SEI growth transfer coefficient": 0.2},
            check_already_exists=False,
        )
        self.run_basic_processing_test(options, parameter_values=parameter_values)

    def test_sei_solvent_diffusion_limited(self):
        options = {"SEI": "solvent-diffusion limited"}
        self.run_basic_processing_test(options)

    def test_sei_electron_migration_limited(self):
        options = {"SEI": "electron-migration limited"}
        self.run_basic_processing_test(options)

    def test_sei_interstitial_diffusion_limited(self):
        options = {"SEI": "interstitial-diffusion limited"}
        self.run_basic_processing_test(options)

    def test_sei_ec_reaction_limited(self):
        options = {
            "SEI": "ec reaction limited",
            "SEI porosity change": "true",
        }
        self.run_basic_processing_test(options)

    def test_sei_asymmetric_ec_reaction_limited(self):
        options = {
            "SEI": "ec reaction limited (asymmetric)",
            "SEI porosity change": "true",
        }
        parameter_values = pybamm.ParameterValues("Marquis2019")
        parameter_values.update(
            {"SEI growth transfer coefficient": 0.2},
            check_already_exists=False,
        )
        self.run_basic_processing_test(options, parameter_values=parameter_values)

    def test_loss_active_material_stress_positive(self):
        options = {"loss of active material": ("none", "stress-driven")}
        parameter_values = pybamm.ParameterValues("Ai2020")
        self.run_basic_processing_test(options, parameter_values=parameter_values)

    def test_loss_active_material_stress_negative(self):
        options = {"loss of active material": ("stress-driven", "none")}
        parameter_values = pybamm.ParameterValues("Ai2020")
        self.run_basic_processing_test(options, parameter_values=parameter_values)

    def test_loss_active_material_stress_both(self):
        options = {"loss of active material": "stress-driven"}
        parameter_values = pybamm.ParameterValues("Ai2020")
        self.run_basic_processing_test(options, parameter_values=parameter_values)

    def test_loss_active_material_reaction(self):
        options = {"loss of active material": "reaction-driven"}
        self.run_basic_processing_test(options)

    def test_loss_active_material_stress_and_reaction(self):
        options = {"loss of active material": "stress and reaction-driven"}
        parameter_values = pybamm.ParameterValues("Ai2020")
        self.run_basic_processing_test(options, parameter_values=parameter_values)

    def test_well_posed_loss_active_material_current_negative(self):
        options = {"loss of active material": ("current-driven", "none")}
        parameter_values = pybamm.ParameterValues("Chen2020")

        def current_LAM(i, T):
            return -1e-10 * abs(i)

        parameter_values.update(
            {"Negative electrode current-driven LAM rate": current_LAM},
            check_already_exists=False,
        )

        self.run_basic_processing_test(options, parameter_values=parameter_values)

    def test_well_posed_loss_active_material_current_positive(self):
        options = {"loss of active material": ("none", "current-driven")}
        parameter_values = pybamm.ParameterValues("Chen2020")

        def current_LAM(i, T):
            return -1e-10 * abs(i)

        parameter_values.update(
            {"Positive electrode current-driven LAM rate": current_LAM},
            check_already_exists=False,
        )

        self.run_basic_processing_test(options, parameter_values=parameter_values)

    def test_negative_cracking(self):
        options = {"particle mechanics": ("swelling and cracking", "none")}
        parameter_values = pybamm.ParameterValues("Ai2020")
        var_pts = {
            "x_n": 20,  # negative electrode
            "x_s": 20,  # separator
            "x_p": 20,  # positive electrode
            "r_n": 26,  # negative particle
            "r_p": 26,  # positive particle
        }
        self.run_basic_processing_test(
            options, parameter_values=parameter_values, var_pts=var_pts
        )

    def test_positive_cracking(self):
        options = {"particle mechanics": ("none", "swelling and cracking")}
        parameter_values = pybamm.ParameterValues("Ai2020")
        var_pts = {
            "x_n": 20,  # negative electrode
            "x_s": 20,  # separator
            "x_p": 20,  # positive electrode
            "r_n": 26,  # negative particle
            "r_p": 26,  # positive particle
        }
        self.run_basic_processing_test(
            options, parameter_values=parameter_values, var_pts=var_pts
        )

    def test_both_cracking(self):
        options = {"particle mechanics": "swelling and cracking"}
        parameter_values = pybamm.ParameterValues("Ai2020")
        var_pts = {
            "x_n": 20,  # negative electrode
            "x_s": 20,  # separator
            "x_p": 20,  # positive electrode
            "r_n": 26,  # negative particle
            "r_p": 26,  # positive particle
        }
        self.run_basic_processing_test(
            options, parameter_values=parameter_values, var_pts=var_pts
        )

    def test_both_swelling_only(self):
        options = {"particle mechanics": "swelling only"}
        parameter_values = pybamm.ParameterValues("Ai2020")
        self.run_basic_processing_test(options, parameter_values=parameter_values)

    def test_composite_graphite_silicon(self):
        options = {
            "particle phases": ("2", "1"),
            "open-circuit potential": (("single", "current sigmoid"), "single"),
        }
        parameter_values = pybamm.ParameterValues("Chen2020_composite")
        name = "Negative electrode active material volume fraction"
        x = 0.1
        parameter_values.update(
            {f"Primary: {name}": (1 - x) * 0.75, f"Secondary: {name}": x * 0.75}
        )
        self.run_basic_processing_test(options, parameter_values=parameter_values)

    def test_composite_graphite_silicon_sei(self):
        options = {
            "particle phases": ("2", "1"),
            "open-circuit potential": (("single", "current sigmoid"), "single"),
            "SEI": "ec reaction limited",
        }
        parameter_values = pybamm.ParameterValues("Chen2020_composite")
        name = "Negative electrode active material volume fraction"
        x = 0.1
        parameter_values.update(
            {f"Primary: {name}": (1 - x) * 0.75, f"Secondary: {name}": x * 0.75}
        )
        self.run_basic_processing_test(options, parameter_values=parameter_values)

    def test_basic_processing_msmr(self):
        options = {
            "open-circuit potential": "MSMR",
            "particle": "MSMR",
            "intercalation kinetics": "MSMR",
            "number of MSMR reactions": ("6", "4"),
        }
        parameter_values = pybamm.ParameterValues("MSMR_Example")
        model = self.model(options)
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all(skip_output_tests=True)

    def test_basic_processing_temperature_interpolant(self):
        times = np.arange(0, 4000, 10)
        tmax = max(times)

        def temp_drive_cycle(y, z, t):
            return pybamm.Interpolant(
                times,
                298.15 + 20 * (times / tmax),
                t,
            )

        parameter_values = pybamm.ParameterValues("Chen2020")
        parameter_values.update(
            {
                "Initial temperature [K]": 298.15,
                "Ambient temperature [K]": temp_drive_cycle,
            }
        )
        model = self.model()
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all(skip_output_tests=True)
