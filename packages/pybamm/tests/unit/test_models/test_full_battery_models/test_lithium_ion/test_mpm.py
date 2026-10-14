#
# Tests for the lithium-ion MPM model
#

import numpy as np
import pytest

import pybamm


class TestMPM:
    def test_well_posed(self):
        options = {"thermal": "isothermal"}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

        # Test build after init
        model = pybamm.lithium_ion.MPM(build=False)
        model.build_model()
        model.check_well_posedness()

    def test_default_parameter_values(self):
        # check default parameters are added correctly
        model = pybamm.lithium_ion.MPM()
        assert (
            model.default_parameter_values["Negative minimum particle radius [m]"]
            == 0.0
        )

    def test_mean_particle_radius_input_sensitivity(self):
        # The size distribution must follow an input mean radius, not its default
        name = "Negative particle radius [m]"
        var_pts = {"x_n": 5, "x_s": 5, "x_p": 5, "r_n": 6, "r_p": 6, "R_n": 6, "R_p": 6}
        t_interp = np.linspace(0, 1200, 7)

        def solve(radius, calculate_sensitivities):
            model = pybamm.lithium_ion.MPM()
            parameter_values = model.default_parameter_values
            parameter_values.update({name: "[input]"})
            submesh_types = model.default_submesh_types
            for domain in ["negative particle", "negative particle size"]:
                submesh_types[domain] = pybamm.SymbolicUniform1DSubMesh
            sim = pybamm.Simulation(
                model,
                parameter_values=parameter_values,
                submesh_types=submesh_types,
                var_pts=var_pts,
                solver=pybamm.IDAKLUSolver(rtol=1e-10, atol=1e-12),
            )
            return sim.solve(
                [0, 1200],
                inputs={name: radius},
                calculate_sensitivities=calculate_sensitivities,
                t_interp=t_interp,
            )

        radius = pybamm.lithium_ion.MPM().default_parameter_values[name]
        sensitivity = solve(radius, True)["Voltage [V]"].sensitivities[name].ravel()
        h = 1e-4 * radius
        voltage_plus = solve(radius + h, False)["Voltage [V]"](t_interp)
        voltage_minus = solve(radius - h, False)["Voltage [V]"](t_interp)
        finite_difference = (voltage_plus - voltage_minus) / (2 * h)
        assert np.max(np.abs(finite_difference)) > 1e2
        np.testing.assert_allclose(sensitivity, finite_difference, rtol=1e-4, atol=1e-1)

    def test_lumped_thermal_model_1D(self):
        options = {"thermal": "lumped"}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_x_full_thermal_not_implemented(self):
        options = {"thermal": "x-full"}
        with pytest.raises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)

    def test_thermal_1plus1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "x-lumped",
        }
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_particle_uniform(self):
        options = {"particle": "uniform profile"}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_particle_quadratic(self):
        options = {"particle": "quadratic profile"}
        with pytest.raises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)

    def test_differential_surface_form(self):
        options = {"surface form": "differential"}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_current_sigmoid(self):
        options = {"open-circuit potential": "current sigmoid"}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_necessary_options(self):
        options = {"particle size": "single"}
        with pytest.raises(pybamm.OptionError):
            pybamm.lithium_ion.MPM(options)

        options = {"surface form": "false"}
        with pytest.raises(pybamm.OptionError):
            pybamm.lithium_ion.MPM(options)

    def test_nonspherical_particle_not_implemented(self):
        options = {"particle shape": "user"}
        with pytest.raises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)

    def test_reversible_plating_with_porosity_not_implemented(self):
        options = {
            "lithium plating": "reversible",
            "lithium plating porosity change": "true",
        }
        with pytest.raises(pybamm.OptionError, match=r"distributions"):
            pybamm.lithium_ion.MPM(options)

    def test_msmr(self):
        options = {
            "open-circuit potential": "MSMR",
            "particle": "MSMR",
            "number of MSMR reactions": ("6", "4"),
            "intercalation kinetics": "MSMR",
        }
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_one_state_differential_capacity_hysteresis_ocp(self):
        options = {
            "open-circuit potential": "one-state differential capacity hysteresis",
        }
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_one_state_hysteresis_ocp(self):
        options = {
            "open-circuit potential": "one-state hysteresis",
        }
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_one_state_hysteresis_ocp_lumped_thermal(self):
        # Regression test: MPM always uses particle size distribution, and
        # combining hysteresis OCP with a non-isothermal thermal submodel
        # previously raised a DomainError in the hysteresis heating term.
        options = {
            "open-circuit potential": "one-state hysteresis",
            "thermal": "lumped",
        }
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_mpm_with_lithium_plating(self):
        options = {
            "lithium plating": "irreversible",
        }
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()
        options = {
            "lithium plating": "reversible",
        }
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()


class TestMPMExternalCircuits:
    def test_well_posed_voltage(self):
        options = {"operating mode": "voltage"}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_well_posed_power(self):
        options = {"operating mode": "power"}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_well_posed_function(self):
        def external_circuit_function(variables):
            I = variables["Current [A]"]
            V = variables["Voltage [V]"]
            return V + I - pybamm.FunctionParameter("Function", {"Time [s]": pybamm.t})

        options = {"operating mode": external_circuit_function}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()


class TestMPMWithSEI:
    def test_reaction_limited(self):
        options = {"SEI": "reaction limited"}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_solvent_diffusion_limited(self):
        options = {"SEI": "solvent-diffusion limited"}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_electron_migration_limited(self):
        options = {"SEI": "electron-migration limited"}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_interstitial_diffusion_limited(self):
        options = {"SEI": "interstitial-diffusion limited"}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_ec_reaction_limited_not_implemented(self):
        options = {
            "SEI": "ec reaction limited",
            "SEI porosity change": "true",
        }
        with pytest.raises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)
