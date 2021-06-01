#
# Tests for Landesfeind (2019) electrolytes parameter set loads
#
import pybamm
import unittest
import os


class TestLandesfeind(unittest.TestCase):
    def test_load_params(self):
        data_D_e = {
            "EC_DMC_1_1": [1.94664e-10, 1.94233e-10],
            "EC_EMC_3_7": [2.01038e-10, 1.78391e-10],
            "EMC_FEC_19_1": [2.16871e-10, 1.8992e-10],
        }
        data_sigma_e = {
            "EC_DMC_1_1": [0.870352, 0.839076],
            "EC_EMC_3_7": [0.695252, 0.668677],
            "EMC_FEC_19_1": [0.454054, 0.632419],
        }
        data_TDF = {
            "EC_DMC_1_1": [1.84644, 4.16915],
            "EC_EMC_3_7": [1.82671, 3.9218],
            "EMC_FEC_19_1": [0.92532, 3.22481],
        }
        data_tplus = {
            "EC_DMC_1_1": [0.17651, 0.241924],
            "EC_EMC_3_7": [0.0118815, 0.151879],
            "EMC_FEC_19_1": [-0.0653014, 0.0416203],
        }

        T = [273.15 + 10.0, 273.15 + 30.0]
        c = [1000.0, 2000.0]

        for solvent in ["EC_DMC_1_1", "EC_EMC_3_7", "EMC_FEC_19_1"]:
            root = pybamm.root_dir()
            p = (
                "pybamm/input/parameters/lithium_ion/electrolytes/lipf6_"
                + solvent
                + "_Landesfeind2019/"
            )
            k_path = os.path.join(root, p)

            sigma_e = pybamm.load_function(
                os.path.join(
                    k_path,
                    "electrolyte_conductivity_" + solvent + "_Landesfeind2019.py",
                )
            )
            D_e = pybamm.load_function(
                os.path.join(
                    k_path, "electrolyte_diffusivity_" + solvent + "_Landesfeind2019.py"
                )
            )
            TDF = pybamm.load_function(
                os.path.join(
                    k_path, "electrolyte_TDF_" + solvent + "_Landesfeind2019.py"
                )
            )
            tplus = pybamm.load_function(
                os.path.join(
                    k_path,
                    "electrolyte_transference_number_"
                    + solvent
                    + "_Landesfeind2019.py",
                )
            )

            for i, _ in enumerate(T):
                self.assertAlmostEqual(
                    sigma_e(c[i], T[i]).value, data_sigma_e[solvent][i], places=5
                )
                self.assertAlmostEqual(
                    D_e(c[i], T[i]).value, data_D_e[solvent][i], places=5
                )
                self.assertAlmostEqual(TDF(c[i], T[i]), data_TDF[solvent][i], places=5)
                self.assertAlmostEqual(
                    tplus(c[i], T[i]), data_tplus[solvent][i], places=5
                )

    def test_standard_lithium_parameters(self):
        for solvent in ["EC_DMC_1_1", "EC_EMC_3_7", "EMC_FEC_19_1"]:
            chemistry = pybamm.parameter_sets.Chen2020
            chemistry["electrolyte"] = "lipf6_" + solvent + "_Landesfeind2019"
            parameter_values = pybamm.ParameterValues(chemistry=chemistry)
            model = pybamm.lithium_ion.DFN()
            sim = pybamm.Simulation(model, parameter_values=parameter_values)
            sim.set_parameters()
            sim.build()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
