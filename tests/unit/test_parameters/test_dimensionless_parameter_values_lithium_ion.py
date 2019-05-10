#
# Tests for the Base Parameter Values class
#
import pybamm

import os

import unittest
import numpy as np


class TestDimensionlessParameterValues(unittest.TestCase):
    def test_lithium_ion(self):
        """This test checks that all the dimensionless parameters are being calculated
        correctly for the specific set of parameters for LCO from dualfoil. The values
        are those converted from those in Scott's transfer which previous versions of
        the DFN work with. A 1C rate corresponds to a 24A/m^2 current density"""
        input_path = os.path.join(os.getcwd(), "input", "parameters", "lithium-ion")
        values = pybamm.ParameterValues(
            os.path.join(
                input_path, "mcmb2528_lif6-in-ecdmc_lico2_parameters_Dualfoil.csv"
            ),
            {
                "Typical current [A]": 1,
                "Current function": os.path.join(
                    os.getcwd(),
                    "pybamm",
                    "parameters",
                    "standard_current_functions",
                    "constant_current.py",
                ),
                "Electrolyte diffusivity": os.path.join(
                    input_path, "electrolyte_diffusivity_Capiglia1999.py"
                ),
                "Electrolyte conductivity": os.path.join(
                    input_path, "electrolyte_conductivity_Capiglia1999.py"
                ),
                "Negative electrode OCV": os.path.join(
                    input_path, "graphite_mcmb2528_ocp_Dualfoil.py"
                ),
                "Positive electrode OCV": os.path.join(
                    input_path, "lico2_ocp_Dualfoil.py"
                ),
                "Negative electrode diffusivity": os.path.join(
                    input_path, "graphite_mcmb2528_diffusivity_Dualfoil.py"
                ),
                "Positive electrode diffusivity": os.path.join(
                    input_path, "lico2_diffusivity_Dualfoil.py"
                ),
            },
        )

        param = pybamm.standard_parameters_lithium_ion

        c_rate = param.i_typ / 24  # roughly for the numbers I used before

        "particle geometry"
        # a_n dimensional
        np.testing.assert_almost_equal(
            values.process_symbol(param.a_n_dim).evaluate(None, None),
            0.18 * 10 ** (6),
            2,
        )
        # R_n dimensional
        np.testing.assert_almost_equal(
            values.process_symbol(param.R_n).evaluate(None, None), 1 * 10 ** (-5), 2
        )

        # a_n
        np.testing.assert_almost_equal(
            values.process_symbol(param.a_n).evaluate(None, None), 1.8, 2
        )

        # a_p dimensional
        np.testing.assert_almost_equal(
            values.process_symbol(param.a_p_dim).evaluate(None, None),
            0.15 * 10 ** (6),
            2,
        )

        # R_p dimensional
        np.testing.assert_almost_equal(
            values.process_symbol(param.R_n).evaluate(None, None), 1 * 10 ** (-5), 2
        )

        # a_p
        np.testing.assert_almost_equal(
            values.process_symbol(param.a_p).evaluate(None, None), 1.5, 2
        )

        "reaction rates"
        # m_n*
        np.testing.assert_almost_equal(
            values.process_symbol(param.m_n_dimensional).evaluate(None, None),
            2 * 10 ** (-5),
            8,
        )

        np.testing.assert_almost_equal(
            values.process_symbol(1 / param.C_r_n * c_rate).evaluate(None, None),
            26.6639,
            3,
        )

        # m_p*
        np.testing.assert_almost_equal(
            values.process_symbol(param.m_p_dimensional).evaluate(None, None),
            6 * 10 ** (-7),
            8,
        )

        # gamma_p / C_r_p
        np.testing.assert_almost_equal(
            values.process_symbol(param.gamma_p / param.C_r_p * c_rate).evaluate(
                None, None
            ),
            1.366,
            3,
        )

        "particle dynamics"
        # neg diffusion coefficient
        np.testing.assert_almost_equal(
            values.process_symbol(param.D_n_dimensional(param.c_n_init)).evaluate(
                None, None
            ),
            3.9 * 10 ** (-14),
            2,
        )

        # neg diffusion timescale
        np.testing.assert_almost_equal(
            values.process_symbol(param.tau_diffusion_n).evaluate(None, None),
            2.5641 * 10 ** (3),
            2,
        )

        # tau_n / tau_d (1/gamma_n in Scott's transfer)
        np.testing.assert_almost_equal(
            values.process_symbol(param.C_n / c_rate).evaluate(None, None), 0.11346, 3
        )

        # pos diffusion coefficient
        np.testing.assert_almost_equal(
            values.process_symbol(param.D_p_dimensional(param.c_p_init)).evaluate(
                None, None
            ),
            1 * 10 ** (-13),
            2,
        )

        # pos diffusion timescale
        np.testing.assert_almost_equal(
            values.process_symbol(param.tau_diffusion_p).evaluate(None, None),
            1 * 10 ** (3),
            2,
        )

        # tau_p / tau_d (1/gamma_p in Scott's transfer)
        np.testing.assert_almost_equal(
            values.process_symbol(param.C_p / c_rate).evaluate(None, None), 0.044249, 3
        )

        "electrolyte dynamics"
        # typical diffusion coefficient (we should change the typ value in paper to
        # match this one. We take this parameter excluding the exp(-0.65) in the
        # paper at the moment
        np.testing.assert_almost_equal(
            values.process_symbol(param.D_e_dimensional(param.c_e_typ)).evaluate(
                None, None
            ),
            5.34 * 10 ** (-10) * np.exp(-0.65),
            10,
        )

        # electrolyte diffusion timescale (accounting for np.exp(-0.65) in
        # diffusion_typ). Change value in paper to this.
        np.testing.assert_almost_equal(
            values.process_symbol(param.tau_diffusion_e).evaluate(None, None),
            181.599,
            3,
        )

        # C_e
        np.testing.assert_almost_equal(
            values.process_symbol(param.C_e / c_rate).evaluate(None, None), 0.008, 3
        )

        # electrolyte conductivity
        np.testing.assert_almost_equal(
            values.process_symbol(param.kappa_e_dimensional(param.c_e_typ)).evaluate(
                None, None
            ),
            1.1045,
            3,
        )

        "potential scale"
        # F R / T (should be equal to old 1 / Lambda)
        old_Lambda = 38
        np.testing.assert_almost_equal(
            values.process_symbol(param.potential_scale).evaluate(None, None),
            1 / old_Lambda,
            3,
        )

        "electrode conductivities"
        # neg dimensional
        np.testing.assert_almost_equal(
            values.process_symbol(param.sigma_n_dimensional).evaluate(None, None),
            100,
            3,
        )

        # neg dimensionless (old sigma_n / old_Lambda ) (this is different to values
        # in paper so check again, it is close enough though for now)
        np.testing.assert_almost_equal(
            values.process_symbol(param.sigma_n * c_rate).evaluate(None, None), 475.7, 1
        )

        # neg dimensional
        np.testing.assert_almost_equal(
            values.process_symbol(param.sigma_p_dimensional).evaluate(None, None), 10, 3
        )

        # neg dimensionless (old sigma_n / old_Lambda ) (this is different to values in
        # paper so check again, it is close enough for now though)
        np.testing.assert_almost_equal(
            values.process_symbol(param.sigma_p * c_rate).evaluate(None, None), 47.57, 1
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
