#
# Tests for the lithium-ion MPM model
#
import pybamm
import unittest


class TestMPM(unittest.TestCase):
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
        self.assertEqual(
            model.default_parameter_values[
                "Negative area-weighted mean particle radius [m]"
            ],
            1e-05,
        )

    def test_lumped_thermal_model_1D(self):
        options = {"thermal": "lumped"}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_x_full_thermal_not_implemented(self):
        options = {"thermal": "x-full"}
        with self.assertRaises(NotImplementedError):
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
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)

    def test_differential_surface_form(self):
        options = {"surface form": "differential"}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_necessary_options(self):
        options = {"particle size": "single"}
        with self.assertRaises(pybamm.OptionError):
            pybamm.lithium_ion.MPM(options)

        options = {"surface form": "false"}
        with self.assertRaises(pybamm.OptionError):
            pybamm.lithium_ion.MPM(options)

    def test_nonspherical_particle_not_implemented(self):
        options = {"particle shape": "user"}
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)

    def test_loss_active_material_stress_negative_not_implemented(self):
        options = {"loss of active material": ("stress-driven", "none")}
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)

    def test_loss_active_material_stress_positive_not_implemented(self):
        options = {"loss of active material": ("none", "stress-driven")}
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)

    def test_loss_active_material_stress_both_not_implemented(self):
        options = {"loss of active material": "stress-driven"}
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)

    def test_reversible_plating_with_porosity_not_implemented(self):
        options = {
            "lithium plating": "reversible",
            "lithium plating porosity change": "true",
        }
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)

    def test_stress_induced_diffusion_not_implemented(self):
        options = {"stress-induced diffusion": "true"}
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)


class TestMPMExternalCircuits(unittest.TestCase):
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
            V = variables["Terminal voltage [V]"]
            return V + I - pybamm.FunctionParameter("Function", {"Time [s]": pybamm.t})

        options = {"operating mode": external_circuit_function}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()


class TestMPMWithSEI(unittest.TestCase):
    def test_reaction_limited_not_implemented(self):
        options = {"SEI": "reaction limited"}
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)

    def test_solvent_diffusion_limited_not_implemented(self):
        options = {"SEI": "solvent-diffusion limited"}
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)

    def test_electron_migration_limited_not_implemented(self):
        options = {"SEI": "electron-migration limited"}
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)

    def test_interstitial_diffusion_limited_not_implemented(self):
        options = {"SEI": "interstitial-diffusion limited"}
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)

    def test_ec_reaction_limited_not_implemented(self):
        options = {
            "SEI": "ec reaction limited",
            "SEI porosity change": "true",
        }
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)


class TestMPMWithMechanics(unittest.TestCase):
    def test_well_posed_negative_cracking_not_implemented(self):
        options = {"particle mechanics": ("swelling and cracking", "none")}
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)

    def test_well_posed_positive_cracking_not_implemented(self):
        options = {"particle mechanics": ("none", "swelling and cracking")}
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)

    def test_well_posed_both_cracking_not_implemented(self):
        options = {"particle mechanics": "swelling and cracking"}
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)

    def test_well_posed_both_swelling_only_not_implemented(self):
        options = {"particle mechanics": "swelling only"}
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)


class TestMPMWithPlating(unittest.TestCase):
    def test_well_posed_reversible_plating_not_implemented(self):
        options = {"lithium plating": "reversible"}
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)

    def test_well_posed_irreversible_plating_not_implemented(self):
        options = {"lithium plating": "irreversible"}
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
