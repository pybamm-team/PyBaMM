#
# Tests for the create_from_bpx function
#

import tempfile
import unittest
import json
import pybamm
import copy


class TestBPX(unittest.TestCase):
    def setUp(self):
        self.base = {
            "Header": {
                "BPX": 1.0,
                "Title": "Parametrisation example",
                "Description": "Test",
                "Model": "DFN",
            },
            "Parameterisation": {
                "Cell": {
                    "Ambient temperature [K]": 298.15,
                    "Initial temperature [K]": 298.15,
                    "Reference temperature [K]": 298.15,
                    "Lower voltage cut-off [V]": 2.8,
                    "Upper voltage cut-off [V]": 4.2,
                    "Nominal cell capacity [A.h]": 12.5,
                    "Specific heat capacity [J.K-1.kg-1]": 913,
                    "Thermal conductivity [W.m-1.K-1]": 2,
                    "Density [kg.m-3]": 1847,
                    "Electrode area [m2]": 0.016808,
                    "Number of electrode pairs connected "
                    "in parallel to make a cell": 34,
                    "External surface area [m2]": 3.79e-2,
                    "Volume [m3]": 1.28e-4,
                },
                "Electrolyte": {
                    "Initial concentration [mol.m-3]": 1000,
                    "Cation transference number": 0.259,
                    "Conductivity [S.m-1]": (
                        "0.1297 * (x / 1000) ** 3 "
                        "- 2.51 * (x / 1000) ** 1.5 + 3.329 * (x / 1000)"
                    ),
                    "Diffusivity [m2.s-1]": (
                        "8.794e-11 * (x / 1000) ** 2 "
                        "- 3.972e-10 * (x / 1000) + 4.862e-10"
                    ),
                    "Conductivity activation energy [J.mol-1]": 34200,
                    "Diffusivity activation energy [J.mol-1]": 34200,
                },
                "Negative electrode": {
                    "Particle radius [m]": 4.12e-06,
                    "Thickness [m]": 5.62e-05,
                    "Diffusivity [m2.s-1]": "8.6e-13 * exp(-13.5 * x) + 9.5e-15",
                    "OCP [V]": (
                        "7.84108819e-01 * exp(-5.16822591e01 * x) + "
                        "5.99914745e02 + "
                        "2.62306941e-02 * tanh(-1.71992993e01 * (x - 5.48687033e-01)) +"
                        "9.41099327e02 * tanh(-6.91080049e-01 * (x + 2.49433043e00)) + "
                        "3.40646063e02 * tanh(7.27243978e-01 * (x + 1.64297574e00))"
                    ),
                    "Entropic change coefficient [V.K-1]": "-0.001*x",
                    "Conductivity [S.m-1]": 0.39,
                    "Surface area per unit volume [m-1]": 487864,
                    "Porosity": 0.33,
                    "Transport efficiency": 0.19,
                    "Reaction rate constant [mol.m-2.s-1]": 5.75e-06,
                    "Minimum stoichiometry": 0.008536,
                    "Maximum stoichiometry": 0.824874,
                    "Maximum concentration [mol.m-3]": 28500,
                    "Diffusivity activation energy [J.mol-1]": 108000,
                    "Reaction rate constant activation energy [J.mol-1]": 53400,
                },
                "Positive electrode": {
                    "Particle radius [m]": 4.6e-06,
                    "Thickness [m]": 5.23e-05,
                    "Diffusivity [m2.s-1]": (
                        "5e-13 - x * 8e-14 - 4.1e-13 * exp(-12 * (x - 0.98) ** 2)"
                    ),
                    "OCP [V]": (
                        "-2.59073509 * x + "
                        "4.17659428 - "
                        "11.03429916 * tanh(-9.343666 * (x - 0.79475919)) - "
                        "1.63480454 * tanh(82.26606342 * (x - 1.00945121)) - "
                        "10.70641562 * tanh(9.43939843 * (x - 0.79469384))"
                    ),
                    "Entropic change coefficient [V.K-1]": "-0.001*x",
                    "Conductivity [S.m-1]": 1.464,
                    "Surface area per unit volume [m-1]": 404348,
                    "Porosity": 0.385,
                    "Transport efficiency": 0.2389,
                    "Reaction rate constant [mol.m-2.s-1]": 2.5e-05,
                    "Minimum stoichiometry": 0.462455,
                    "Maximum stoichiometry": 0.940627,
                    "Maximum concentration [mol.m-3]": 56500,
                    "Diffusivity activation energy [J.mol-1]": 62400,
                    "Reaction rate constant activation energy [J.mol-1]": 27010,
                },
                "Separator": {
                    "Thickness [m]": 2e-5,
                    "Porosity": 0.47,
                    "Transport efficiency": 0.3222,
                },
            },
        }

    def test_bpx(self):
        bpx_obj = copy.copy(self.base)

        filename = "tmp.json"
        with tempfile.NamedTemporaryFile(
            suffix=filename, delete=False, mode="w"
        ) as tmp:
            # write to a tempory file so we can
            # get the source later on using inspect.getsource
            # (as long as the file still exists)
            json.dump(bpx_obj, tmp)
            tmp.flush()

            pv = pybamm.ParameterValues.create_from_bpx(tmp.name)

            model = pybamm.lithium_ion.DFN()
            experiment = pybamm.Experiment(
                [
                    "Discharge at C/5 for 1 hour",
                ]
            )
            sim = pybamm.Simulation(model, parameter_values=pv, experiment=experiment)
            sim.solve()

    def test_constant_functions(self):
        bpx_obj = copy.copy(self.base)
        bpx_obj["Parameterisation"]["Electrolyte"].update(
            {
                "Conductivity [S.m-1]": 1,
                "Diffusivity [m2.s-1]": 1,
            }
        )
        bpx_obj["Parameterisation"]["Negative electrode"].update(
            {
                "Diffusivity [m2.s-1]": 1,
                "Entropic change coefficient [V.K-1]": 1,
            }
        )
        bpx_obj["Parameterisation"]["Positive electrode"].update(
            {
                "Diffusivity [m2.s-1]": 1,
                "Entropic change coefficient [V.K-1]": 1,
            }
        )

        filename = "tmp.json"
        with tempfile.NamedTemporaryFile(
            suffix=filename, delete=False, mode="w"
        ) as tmp:
            # write to a tempory file so we can
            # get the source later on using inspect.getsource
            # (as long as the file still exists)
            json.dump(bpx_obj, tmp)
            tmp.flush()

            pybamm.ParameterValues.create_from_bpx(tmp.name)

    def test_table_data(self):
        bpx_obj = copy.copy(self.base)
        bpx_obj["Parameterisation"]["Electrolyte"].update(
            {"Conductivity [S.m-1]": {"x": [800, 1000, 1200], "y": [0.9, 1, 1.1]}}
        )

        filename = "tmp.json"
        with tempfile.NamedTemporaryFile(
            suffix=filename, delete=False, mode="w"
        ) as tmp:
            # write to a tempory file so we can
            # get the source later on using inspect.getsource
            # (as long as the file still exists)
            json.dump(bpx_obj, tmp)
            tmp.flush()

            pybamm.ParameterValues.create_from_bpx(tmp.name)

    def test_bpx_soc_error(self):
        with self.assertRaisesRegex(ValueError, "Target SOC"):
            pybamm.ParameterValues.create_from_bpx("blah.json", target_soc=10)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
