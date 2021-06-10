#
# Tests for the lithium-ion SPM model
#
import pybamm
import unittest


class TestSPM(unittest.TestCase):
    def test_well_posed(self):
        options = {"thermal": "isothermal"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

        # Test build after init
        model = pybamm.lithium_ion.SPM(build=False)
        model.build_model()
        model.check_well_posedness()

    def test_well_posed_2plus1D(self):
        options = {"current collector": "potential pair", "dimensionality": 1}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

        options = {"current collector": "potential pair", "dimensionality": 2}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_lumped_thermal_model_1D(self):
        options = {"thermal": "lumped"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_x_full_thermal_model(self):
        options = {"thermal": "x-full"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_x_full_Nplus1D_not_implemented(self):
        # 1plus1D
        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "x-full",
        }
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.SPM(options)
        # 2plus1D
        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "x-full",
        }
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.SPM(options)

    def test_lumped_thermal_1plus1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "lumped",
        }
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_lumped_thermal_2plus1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "lumped",
        }
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_thermal_1plus1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "x-lumped",
        }
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_thermal_2plus1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "x-lumped",
        }
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_particle_uniform(self):
        options = {"particle": "uniform profile"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_particle_quadratic(self):
        options = {"particle": "quadratic profile"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_particle_quartic(self):
        options = {"particle": "quartic profile"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_particle_mixed(self):
        options = {"particle": ("Fickian diffusion", "quartic profile")}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_particle_shape_user(self):
        options = {"particle shape": "user"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_loss_active_material_stress_negative(self):
        options = {"loss of active material": ("stress-driven", "none")}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_loss_active_material_stress_positive(self):
        options = {"loss of active material": ("none", "stress-driven")}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_loss_active_material_stress_both(self):
        options = {"loss of active material": "stress-driven"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_loss_active_material_stress_reaction_both(self):
        options = {"loss of active material": "reaction-driven"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_surface_form_differential(self):
        options = {"surface form": "differential"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_surface_form_algebraic(self):
        options = {"surface form": "algebraic"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_electrolyte_options(self):
        options = {"electrolyte conductivity": "full"}
        with self.assertRaisesRegex(pybamm.OptionError, "electrolyte conductivity"):
            pybamm.lithium_ion.SPM(options)

    def test_new_model(self):
        model = pybamm.lithium_ion.SPM({"thermal": "x-full"})
        new_model = model.new_copy()
        model_T_eqn = model.rhs[model.variables["Cell temperature"]]
        new_model_T_eqn = new_model.rhs[new_model.variables["Cell temperature"]]
        self.assertEqual(new_model_T_eqn.id, model_T_eqn.id)
        self.assertEqual(new_model.name, model.name)
        self.assertEqual(new_model.use_jacobian, model.use_jacobian)
        self.assertEqual(new_model.convert_to_format, model.convert_to_format)
        self.assertEqual(new_model.timescale.id, model.timescale.id)

        # with custom submodels
        model = pybamm.lithium_ion.SPM({"thermal": "x-full"}, build=False)
        model.submodels["negative particle"] = pybamm.particle.PolynomialSingleParticle(
            model.param, "Negative", "quadratic profile"
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
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()


class TestSPMExternalCircuits(unittest.TestCase):
    def test_well_posed_voltage(self):
        options = {"operating mode": "voltage"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_well_posed_power(self):
        options = {"operating mode": "power"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_well_posed_function(self):
        def external_circuit_function(variables):
            I = variables["Current [A]"]
            V = variables["Terminal voltage [V]"]
            return V + I - pybamm.FunctionParameter("Function", {"Time [s]": pybamm.t})

        options = {"operating mode": external_circuit_function}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()


class TestSPMWithSEI(unittest.TestCase):
    def test_well_posed_reaction_limited(self):
        options = {"SEI": "reaction limited"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_well_posed_solvent_diffusion_limited(self):
        options = {"SEI": "solvent-diffusion limited"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_well_posed_electron_migration_limited(self):
        options = {"SEI": "electron-migration limited"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_well_posed_interstitial_diffusion_limited(self):
        options = {"SEI": "interstitial-diffusion limited"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_well_posed_ec_reaction_limited(self):
        options = {"SEI": "ec reaction limited", "SEI porosity change": "true"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()


class TestSPMWithCrack(unittest.TestCase):
    def test_well_posed_negative_cracking(self):
        options = {"particle mechanics": ("swelling and cracking", "none")}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_well_posed_positive_cracking(self):
        options = {"particle mechanics": ("none", "swelling and cracking")}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_well_posed_both_cracking(self):
        options = {"particle mechanics": "swelling and cracking"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_well_posed_both_swelling_only(self):
        options = {"particle mechanics": "swelling only"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()


class TestSPMWithPlating(unittest.TestCase):
    def test_well_posed_none_plating(self):
        options = {"lithium plating": "none"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_well_posed_reversible_plating(self):
        options = {"lithium plating": "reversible"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()

    def test_well_posed_irreversible_plating(self):
        options = {"lithium plating": "irreversible"}
        model = pybamm.lithium_ion.SPM(options)
        model.check_well_posedness()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
