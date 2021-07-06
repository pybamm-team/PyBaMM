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

    def test_lumped_thermal_model_1D(self):
        options = {"thermal": "lumped"}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_x_full_thermal_not_implemented(self):
        options = {"thermal": "x-full"}
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)

    def test_lumped_thermal_1plus1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "lumped",
        }
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

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

    def test_loss_active_material_stress_negative(self):
        options = {"loss of active material": ("stress-driven", "none")}
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)

    def test_loss_active_material_stress_positive(self):
        options = {"loss of active material": ("none", "stress-driven")}
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)

    def test_loss_active_material_stress_both(self):
        options = {"loss of active material": "stress-driven"}
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)

    def test_well_posed_reversible_plating_with_porosity(self):
        options = {
            "lithium plating": "reversible",
            "lithium plating porosity change": "true",
        }
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
        options = {"SEI": "ec reaction limited", "SEI porosity change": "true"}
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)


class TestMPMWithCrack(unittest.TestCase):
    def test_well_posed_negative_cracking(self):
        options = {"particle mechanics": ("swelling and cracking", "none")}
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)

    def test_well_posed_positive_cracking(self):
        options = {"particle mechanics": ("none", "swelling and cracking")}
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)

    def test_well_posed_both_cracking(self):
        options = {"particle mechanics": "swelling and cracking"}
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)

    def test_well_posed_both_swelling_only(self):
        options = {"particle mechanics": "swelling only"}
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)


class TestMPMWithPlating(unittest.TestCase):
    def test_well_posed_reversible_plating(self):
        options = {"lithium plating": "reversible"}
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)

    def test_well_posed_irreversible_plating(self):
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
