#
# Tests for the lithium-ion LithiumIonModel model
#
import pybamm
import unittest


class TestLithiumIonModel(unittest.TestCase):
    def setUp(self):
        # Will be overwritten by subclasses
        self.model = None

    def check_well_posedness(self, options):
        if self.model is not None:
            model = self.model(options)
            model.check_well_posedness()

    def test_well_posed(self):
        options = {"thermal": "isothermal"}
        self.check_well_posedness(options)

    def test_well_posed_2plus1D(self):
        options = {"current collector": "potential pair", "dimensionality": 1}
        self.check_well_posedness(options)

        options = {"current collector": "potential pair", "dimensionality": 2}
        self.check_well_posedness(options)

    def test_lumped_thermal_model_1D(self):
        options = {"thermal": "x-lumped"}
        self.check_well_posedness(options)

    def test_x_full_thermal_model(self):
        options = {"thermal": "x-full"}
        self.check_well_posedness(options)

    def test_lumped_thermal_1plus1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "lumped",
        }
        self.check_well_posedness(options)

    def test_lumped_thermal_2plus1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "lumped",
        }
        self.check_well_posedness(options)

    def test_thermal_1plus1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "x-lumped",
        }
        self.check_well_posedness(options)

    def test_thermal_2plus1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "x-lumped",
        }
        self.check_well_posedness(options)

    def test_particle_uniform(self):
        options = {"particle": "uniform profile"}
        self.check_well_posedness(options)

    def test_particle_quadratic(self):
        options = {"particle": "quadratic profile"}
        self.check_well_posedness(options)

    def test_particle_quartic(self):
        options = {"particle": "quartic profile"}
        self.check_well_posedness(options)

    def test_particle_mixed(self):
        options = {"particle": ("Fickian diffusion", "quartic profile")}
        self.check_well_posedness(options)

    def test_particle_shape_user(self):
        options = {"particle shape": "user"}
        self.check_well_posedness(options)

    def test_loss_active_material_stress_negative(self):
        options = {"loss of active material": ("stress-driven", "none")}
        self.check_well_posedness(options)

    def test_loss_active_material_stress_positive(self):
        options = {"loss of active material": ("none", "stress-driven")}
        self.check_well_posedness(options)

    def test_loss_active_material_stress_both(self):
        options = {"loss of active material": "stress-driven"}
        self.check_well_posedness(options)

    def test_loss_active_material_stress_reaction_both(self):
        options = {"loss of active material": "reaction-driven"}
        self.check_well_posedness(options)

    def test_surface_form_differential(self):
        options = {"surface form": "differential"}
        self.check_well_posedness(options)

    def test_surface_form_algebraic(self):
        options = {"surface form": "algebraic"}
        self.check_well_posedness(options)

    def test_well_posed_reversible_plating_with_porosity(self):
        options = {
            "lithium plating": "reversible",
            "lithium plating porosity change": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_irreversible_plating_with_porosity(self):
        options = {
            "lithium plating": "irreversible",
            "lithium plating porosity change": "true",
        }
        self.check_well_posedness(options)

    def test_well_posed_sei_constant(self):
        options = {"SEI": "constant"}
        self.check_well_posedness(options)

    def test_well_posed_sei_reaction_limited(self):
        options = {"SEI": "reaction limited"}
        self.check_well_posedness(options)

    def test_well_posed_sei_reaction_limited_average_film_resistance(self):
        options = {"SEI": "reaction limited", "SEI film resistance": "average"}
        self.check_well_posedness(options)

    def test_well_posed_sei_solvent_diffusion_limited(self):
        options = {"SEI": "solvent-diffusion limited"}
        self.check_well_posedness(options)

    def test_well_posed_sei_electron_migration_limited(self):
        options = {"SEI": "electron-migration limited"}
        self.check_well_posedness(options)

    def test_well_posed_sei_interstitial_diffusion_limited(self):
        options = {"SEI": "interstitial-diffusion limited"}
        self.check_well_posedness(options)

    def test_well_posed_sei_ec_reaction_limited(self):
        options = {"SEI": "ec reaction limited", "SEI porosity change": "true"}
        self.check_well_posedness(options)

    def test_well_posed_mechanics_negative_cracking(self):
        options = {"particle mechanics": ("swelling and cracking", "none")}
        self.check_well_posedness(options)

    def test_well_posed_mechanics_positive_cracking(self):
        options = {"particle mechanics": ("none", "swelling and cracking")}
        self.check_well_posedness(options)

    def test_well_posed_mechanics_both_cracking(self):
        options = {"particle mechanics": "swelling and cracking"}
        self.check_well_posedness(options)

    def test_well_posed_mechanics_both_swelling_only(self):
        options = {"particle mechanics": "swelling only"}
        self.check_well_posedness(options)

    def test_well_posed_reversible_plating(self):
        options = {"lithium plating": "reversible"}
        self.check_well_posedness(options)

    def test_well_posed_irreversible_plating(self):
        options = {"lithium plating": "irreversible"}
        self.check_well_posedness(options)

    def test_well_posed_size_distribution(self):
        options = {"particle size": "distribution"}
        self.check_well_posedness(options)

    def test_well_posed_size_distribution_uniform_profile(self):
        options = {"particle size": "distribution", "particle": "uniform profile"}
        self.check_well_posedness(options)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
