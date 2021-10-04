#
# Tests for the lithium-ion Newman-Tobias model
#
import pybamm
import unittest


class TestNewmanTobias(unittest.TestCase):
    def test_well_posed(self):
        options = {"thermal": "isothermal"}
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

    def test_well_posed_2plus1D(self):
        options = {"current collector": "potential pair", "dimensionality": 1}
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

        options = {"current collector": "potential pair", "dimensionality": 2}
        pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

        options = {"bc_options": {"dimensionality": 5}}
        with self.assertRaises(pybamm.OptionError):
            model = pybamm.lithium_ion.NewmanTobias(options)

    def test_lumped_thermal_model_1D(self):
        options = {"thermal": "x-lumped"}
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

    def test_x_full_thermal_model(self):
        options = {"thermal": "x-full"}
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

    def test_x_full_Nplus1D_not_implemented(self):
        # 1plus1D
        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "x-full",
        }
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.NewmanTobias(options)
        # 2plus1D
        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "x-full",
        }
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.NewmanTobias(options)

    def test_lumped_thermal_1plus1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "lumped",
        }
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

    def test_lumped_thermal_2plus1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "lumped",
        }
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

    def test_thermal_1plus1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "x-lumped",
        }
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

    def test_thermal_2plus1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "x-lumped",
        }
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

    def test_particle_fickian(self):
        options = {"particle": "Fickian diffusion"}
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

    def test_particle_quadratic(self):
        options = {"particle": "quadratic profile"}
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

    def test_particle_quartic(self):
        options = {"particle": "quartic profile"}
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

    def test_particle_shape_user(self):
        options = {"particle shape": "user"}
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

    def test_loss_active_material_stress_negative(self):
        options = {"loss of active material": ("stress-driven", "none")}
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

    def test_loss_active_material_stress_positive(self):
        options = {"loss of active material": ("none", "stress-driven")}
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

    def test_loss_active_material_stress_both(self):
        options = {"loss of active material": "stress-driven"}
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

    def test_loss_active_material_stress_reaction_both(self):
        options = {"loss of active material": "reaction-driven"}
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

    def test_surface_form_differential(self):
        options = {"surface form": "differential"}
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

    def test_surface_form_algebraic(self):
        options = {"surface form": "algebraic"}
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

    def test_electrolyte_options(self):
        options = {"electrolyte conductivity": "integrated"}
        with self.assertRaisesRegex(pybamm.OptionError, "electrolyte conductivity"):
            pybamm.lithium_ion.NewmanTobias(options)


class TestNewmanTobiasWithSEI(unittest.TestCase):
    def test_well_posed_constant(self):
        options = {"SEI": "constant"}
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

    def test_well_posed_reaction_limited(self):
        options = {"SEI": "reaction limited"}
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

    def test_well_posed_reaction_limited_average_film_resistance(self):
        options = {"SEI": "reaction limited", "SEI film resistance": "average"}
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

    def test_well_posed_solvent_diffusion_limited(self):
        options = {"SEI": "solvent-diffusion limited"}
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

    def test_well_posed_electron_migration_limited(self):
        options = {"SEI": "electron-migration limited"}
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

    def test_well_posed_interstitial_diffusion_limited(self):
        options = {"SEI": "interstitial-diffusion limited"}
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

    def test_well_posed_ec_reaction_limited(self):
        options = {"SEI": "ec reaction limited", "SEI porosity change": "true"}
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()


class TestNewmanTobiasWithMechanics(unittest.TestCase):
    def test_well_posed_negative_cracking(self):
        options = {"particle mechanics": ("swelling and cracking", "none")}
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

    def test_well_posed_positive_cracking(self):
        options = {"particle mechanics": ("none", "swelling and cracking")}
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

    def test_well_posed_both_cracking(self):
        options = {"particle mechanics": "swelling and cracking"}
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

    def test_well_posed_both_swelling_only(self):
        options = {"particle mechanics": "swelling only"}
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()


class TestNewmanTobiasWithPlating(unittest.TestCase):
    def test_well_posed_none_plating(self):
        options = {"lithium plating": "none"}
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

    def test_well_posed_reversible_plating(self):
        options = {"lithium plating": "reversible"}
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()

    def test_well_posed_irreversible_plating(self):
        options = {"lithium plating": "irreversible"}
        model = pybamm.lithium_ion.NewmanTobias(options)
        model.check_well_posedness()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
