#
# Tests for Landesfeind (2019) electrolytes parameter set loads
#
import pybamm
import unittest
import numpy as np


class TestLandesfeind(unittest.TestCase):
    def test_load_params(self):
        data_D_e = {"EC-DMC_1-1": [1.94664e-10, 1.94233e-10]}
        data_sigma_e = {"EC-DMC_1-1": [0.870352, 0.839076]}
        data_TDF = {"EC-DMM_1-1": [1.84644, 4.16915]}
        data_tplus = {"EC-DMC_1-1": [0.17651, 0.241924]}

        T1 = 273.15 + 10.0
        c1 = 1000.0

        T2 = 273.15 + 30.0
        c2 = 2000.0

        for solvent in ["EC_DMC_1_1"]:
            electrolyte = pybamm.ParameterValues({}).read_parameters_csv(
                pybamm.get_parameters_filepath(
                    "input/parameters/lithium-ion/electrolytes/lipf6_"
                    + solvent
                    + "_Landesfeind2019/parameters.csv"
                )
            )

            sigma_e = electrolyte["Electrolyte conductivity [S.m-1]"]
            D_e = electrolyte["Electrolyte diffusivity [m2.s-1]"]
            TDF = electrolyte["1 + dlnf/dlnc"]
            tplus = electrolyte["Cation transference number"]

            self.assertAlmostEqual(
                [sigma_e(c1, T1).value, sigma_e(c2, T2).value],
                data_sigma_e[solvent],
                places=5,
            )
            self.assertAlmostEqual(
                [D_e(c1, T1).value, D_e(c2, T2).value], data_D_e[solvent], places=15
            )
            self.assertAlmostEqual(
                [TDF(c1, T1).value, TDF(c2, T2).value], data_TDF[solvent], places=5
            )
            self.assertAlmostEqual(
                [tplus(c1, T1).value, tplus(c2, T2).value],
                data_tplus[solvent],
                places=5,
            )

    def test_standard_lithium_parameters(self):
        for solvent in ["EC_DMC_1_1"]:
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
