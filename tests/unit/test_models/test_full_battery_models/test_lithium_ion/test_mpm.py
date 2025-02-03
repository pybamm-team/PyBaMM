#
# Tests for the lithium-ion MPM model
#

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
        with pytest.raises(pybamm.OptionError, match="distributions"):
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

    def test_wycisk_ocp(self):
        options = {
            "open-circuit potential": "Wycisk",
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
