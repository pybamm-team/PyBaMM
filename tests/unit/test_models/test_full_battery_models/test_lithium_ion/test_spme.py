#
# Tests for the lithium-ion SPMe model
#
import pybamm
import unittest


class TestSPMe(unittest.TestCase):
    def test_well_posed(self):
        options = {"thermal": "isothermal"}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

    def test_well_posed_2plus1D(self):
        options = {"current collector": "potential pair", "dimensionality": 1}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

        options = {"current collector": "potential pair", "dimensionality": 2}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

        options = {"bc_options": {"dimensionality": 5}}
        with self.assertRaises(pybamm.OptionError):
            model = pybamm.lithium_ion.SPMe(options)

    def test_lumped_thermal_model_1D(self):
        options = {"thermal": "lumped"}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

    def test_x_full_thermal_model(self):
        options = {"thermal": "x-full"}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

    def test_x_full_Nplus1D_not_implemented(self):
        # 1plus1D
        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "x-full",
        }
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.SPMe(options)
        # 2plus1D
        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "x-full",
        }
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.SPMe(options)

    def test_lumped_thermal_1plus1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "lumped",
        }
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

    def test_lumped_thermal_2plus1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "lumped",
        }
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

    def test_thermal_1plus1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "x-lumped",
        }
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

    def test_thermal_2plus1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "x-lumped",
        }
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

    def test_particle_uniform(self):
        options = {"particle": "uniform profile"}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

    def test_particle_quadratic(self):
        options = {"particle": "quadratic profile"}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

    def test_particle_quartic(self):
        options = {"particle": "quartic profile"}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

    def test_particle_shape_user(self):
        options = {"particle shape": "user"}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

    def test_surface_form_differential(self):
        options = {"surface form": "differential"}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

    def test_surface_form_algebraic(self):
        options = {"surface form": "algebraic"}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()


class TestSPMeWithSEI(unittest.TestCase):
    def test_well_posed_reaction_limited(self):
        options = {"sei": "reaction limited"}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

    def test_well_posed_solvent_diffusion_limited(self):
        options = {"sei": "solvent-diffusion limited"}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

    def test_well_posed_electron_migration_limited(self):
        options = {"sei": "electron-migration limited"}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

    def test_well_posed_interstitial_diffusion_limited(self):
        options = {"sei": "interstitial-diffusion limited"}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

    def test_well_posed_ec_reaction_limited(self):
        options = {"sei": "ec reaction limited", "sei porosity change": True}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
