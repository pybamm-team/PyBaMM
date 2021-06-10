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

    def test_well_posed_2plus1D(self):
        options = {"current collector": "potential pair", "dimensionality": 1}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

        options = {"current collector": "potential pair", "dimensionality": 2}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_lumped_thermal_model_1D(self):
        options = {"thermal": "lumped"}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_x_full_thermal_not_implemented(self):
        options = {"thermal": "x-full"}
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)

    def test_x_full_Nplus1D_not_implemented(self):
        # 1plus1D
        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "x-full",
        }
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)
        # 2plus1D
        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "x-full",
        }
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

    def test_lumped_thermal_2plus1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
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

    def test_thermal_2plus1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "x-lumped",
        }
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_particle_uniform(self):
        options = {"particle": "uniform profile"}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_particle_shape_user(self):
        options = {"particle shape": "user"}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_loss_active_material_stress_negative(self):
        options = {"loss of active material": ("stress-driven", "none")}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_loss_active_material_stress_positive(self):
        options = {"loss of active material": ("none", "stress-driven")}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_loss_active_material_stress_both(self):
        options = {"loss of active material": "stress-driven"}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_loss_active_material_stress_reaction_both(self):
        options = {"loss of active material": "reaction-driven"}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_electrolyte_options(self):
        options = {"electrolyte conductivity": "full"}
        with self.assertRaisesRegex(pybamm.OptionError, "electrolyte conductivity"):
            pybamm.lithium_ion.MPM(options)

    def test_new_model(self):
        model = pybamm.lithium_ion.MPM({"thermal": "x-lumped"})
        new_model = model.new_copy()
        model_T_eqn = model.rhs[model.variables["Volume-averaged cell temperature"]]
        new_model_T_eqn = new_model.rhs[
            new_model.variables["Volume-averaged cell temperature"]
        ]
        self.assertEqual(new_model_T_eqn.id, model_T_eqn.id)
        self.assertEqual(new_model.name, model.name)
        self.assertEqual(new_model.use_jacobian, model.use_jacobian)
        self.assertEqual(new_model.convert_to_format, model.convert_to_format)
        self.assertEqual(new_model.timescale.id, model.timescale.id)

        # with custom submodels
        model = pybamm.lithium_ion.MPM({"thermal": "x-lumped"}, build=False)
        model.submodels[
            "negative particle"
        ] = pybamm.particle.FastSingleSizeDistribution(
            model.param, "Negative"
        )
        model.build_model()
        new_model = model.new_copy()
        new_model_cs_eqn = list(new_model.rhs.values())[1]
        model_cs_eqn = list(model.rhs.values())[1]
        self.assertEqual(new_model_cs_eqn.id, model_cs_eqn.id)

    def test_well_posed_reversible_plating_with_porosity(self):
        options = {
            "lithium plating": "reversible",
            "lithium plating porosity change": "true",
        }
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()


class TestMPMExternalCircuits(unittest.TestCase):
    def test_voltage_not_implemented(self):
        options = {"operating mode": "voltage"}
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.MPM(options)

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
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_well_posed_positive_cracking(self):
        options = {"particle mechanics": ("none", "swelling and cracking")}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_well_posed_both_cracking(self):
        options = {"particle mechanics": "swelling and cracking"}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_well_posed_both_swelling_only(self):
        options = {"particle mechanics": "swelling only"}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()


class TestMPMWithPlating(unittest.TestCase):
    def test_well_posed_none_plating(self):
        options = {"lithium plating": "none"}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_well_posed_reversible_plating(self):
        options = {"lithium plating": "reversible"}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()

    def test_well_posed_irreversible_plating(self):
        options = {"lithium plating": "irreversible"}
        model = pybamm.lithium_ion.MPM(options)
        model.check_well_posedness()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
