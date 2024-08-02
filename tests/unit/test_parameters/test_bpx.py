#
# Tests for the create_from_bpx function
#


import tempfile
import unittest
import json
import pybamm
import copy
import numpy as np
import pytest


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
        bpx_objs = [
            {
                **copy.deepcopy(self.base),
                "Parameterisation": {
                    **copy.deepcopy(self.base["Parameterisation"]),
                    "Negative electrode": {
                        **copy.deepcopy(
                            self.base["Parameterisation"]["Negative electrode"]
                        ),
                        "Diffusivity [m2.s-1]": "8.3e-13 * exp(-13.4 * x) + 9.6e-15",  # new diffusivity
                    },
                },
            },
            copy.copy(self.base),
        ]

        model = pybamm.lithium_ion.DFN()
        experiment = pybamm.Experiment(
            [
                "Discharge at C/5 for 1 hour",
            ]
        )

        filename = "tmp.json"
        sols = []
        for obj in bpx_objs:
            with tempfile.NamedTemporaryFile(
                suffix=filename, delete=False, mode="w"
            ) as tmp:
                # write to a temporary file so we can
                # get the source later on using inspect.getsource
                # (as long as the file still exists)
                json.dump(obj, tmp)
                tmp.flush()

                pv = pybamm.ParameterValues.create_from_bpx(tmp.name)
                sim = pybamm.Simulation(
                    model, parameter_values=pv, experiment=experiment
                )
                sols.append(sim.solve())

        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                sols[0]["Voltage [V]"].data, sols[1]["Voltage [V]"].data, atol=1e-7
            )

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

            param = pybamm.ParameterValues.create_from_bpx(tmp.name)

            # Function to check that functional parameters output constants
            def check_constant_output(func):
                stos = [0, 1]
                T = 298.15
                p_vals = [func(sto, T) for sto in stos]
                self.assertEqual(p_vals[0], p_vals[1])

            for electrode in ["Negative", "Positive"]:
                D = param[f"{electrode} particle diffusivity [m2.s-1]"]
                dUdT = param[f"{electrode} electrode OCP entropic change [V.K-1]"]
                check_constant_output(D)
                check_constant_output(dUdT)

            kappa = param["Electrolyte conductivity [S.m-1]"]
            De = param["Electrolyte diffusivity [m2.s-1]"]
            check_constant_output(kappa)
            check_constant_output(De)

    def test_table_data(self):
        bpx_obj = copy.copy(self.base)
        data = {"x": [0, 1], "y": [0, 1]}
        bpx_obj["Parameterisation"]["Electrolyte"].update(
            {
                "Conductivity [S.m-1]": data,
                "Diffusivity [m2.s-1]": data,
            }
        )
        bpx_obj["Parameterisation"]["Negative electrode"].update(
            {
                "Diffusivity [m2.s-1]": data,
                "OCP [V]": data,
                "Entropic change coefficient [V.K-1]": data,
            }
        )
        bpx_obj["Parameterisation"]["Positive electrode"].update(
            {
                "Diffusivity [m2.s-1]": data,
                "OCP [V]": data,
                "Entropic change coefficient [V.K-1]": data,
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

            param = pybamm.ParameterValues.create_from_bpx(tmp.name)

            # Check that the electrolyte conductivity is an Interpolant with the
            # correct child
            c = pybamm.Variable("c")
            kappa = param["Electrolyte conductivity [S.m-1]"](c, 298.15)
            self.assertIsInstance(kappa, pybamm.Interpolant)
            self.assertEqual(kappa.children[0], c)
            # Check other parameters give interpolants
            D = param["Electrolyte diffusivity [m2.s-1]"](c, 298.15)
            self.assertIsInstance(D, pybamm.Interpolant)
            for electrode in ["Negative", "Positive"]:
                D = param[f"{electrode} particle diffusivity [m2.s-1]"](c, 298.15)
                self.assertIsInstance(D, pybamm.Interpolant)
                OCP = param[f"{electrode} electrode OCP [V]"](c)
                self.assertIsInstance(OCP, pybamm.Interpolant)
                dUdT = param[f"{electrode} electrode OCP entropic change [V.K-1]"](
                    c, 10000
                )
                self.assertIsInstance(dUdT, pybamm.Interpolant)

    def test_bpx_soc_error(self):
        with self.assertRaisesRegex(ValueError, "Target SOC"):
            pybamm.ParameterValues.create_from_bpx("blah.json", target_soc=10)

    def test_bpx_arrhenius(self):
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

        def arrhenius_assertion(pv, param_key, Ea_key):
            sto = 0.5
            T = 300
            c_e = 1000
            c_s_surf = 15000
            c_s_max = 20000
            T_ref = pv["Reference temperature [K]"]
            Ea = pv[Ea_key]

            if "exchange-current" in param_key:
                eval_ratio = (
                    pv[param_key](c_e, c_s_surf, c_s_max, T).value
                    / pv[param_key](c_e, c_s_surf, c_s_max, T_ref).value
                )
            else:
                eval_ratio = (
                    pv[param_key](sto, T).value / pv[param_key](sto, T_ref).value
                )

            calc_ratio = pybamm.exp(Ea / pybamm.constants.R * (1 / T_ref - 1 / T)).value

            self.assertAlmostEqual(eval_ratio, calc_ratio)

        param_keys = [
            "Electrolyte conductivity [S.m-1]",
            "Electrolyte diffusivity [m2.s-1]",
            "Negative particle diffusivity [m2.s-1]",
            "Positive particle diffusivity [m2.s-1]",
            "Positive electrode exchange-current density [A.m-2]",
            "Negative electrode exchange-current density [A.m-2]",
        ]

        Ea_keys = [
            "Electrolyte conductivity activation energy [J.mol-1]",
            "Electrolyte diffusivity activation energy [J.mol-1]",
            "Negative particle diffusivity activation energy [J.mol-1]",
            "Positive particle diffusivity activation energy [J.mol-1]",
            "Positive electrode reaction rate constant activation energy [J.mol-1]",
            "Negative electrode reaction rate constant activation energy [J.mol-1]",
        ]

        for param_key, Ea_key in zip(param_keys, Ea_keys):
            arrhenius_assertion(pv, param_key, Ea_key)

    def test_bpx_blended(self):
        bpx_obj = copy.copy(self.base)
        bpx_obj["Parameterisation"]["Positive electrode"] = {
            "Thickness [m]": 5.23e-05,
            "Conductivity [S.m-1]": 0.789,
            "Porosity": 0.277493,
            "Transport efficiency": 0.1462,
            "Particle": {
                "Large Particles": {
                    "Diffusivity [m2.s-1]": 3.2e-14,
                    "Particle radius [m]": 8e-06,
                    "OCP [V]": "-3.04420906 * x + 10.04892207 - 0.65637536 * tanh(-4.02134095 * (x - 0.80063948)) + 4.24678547 * tanh(12.17805062 * (x - 7.57659337)) - 0.3757068 * tanh(59.33067782 * (x - 0.99784492))",
                    "Entropic change coefficient [V.K-1]": -1e-4,
                    "Surface area per unit volume [m-1]": 186331,
                    "Reaction rate constant [mol.m-2.s-1]": 2.305e-05,
                    "Minimum stoichiometry": 0.42424,
                    "Maximum stoichiometry": 0.96210,
                    "Maximum concentration [mol.m-3]": 46200,
                    "Diffusivity activation energy [J.mol-1]": 15000,
                    "Reaction rate constant activation energy [J.mol-1]": 3500,
                },
                "Small Particles": {
                    "Diffusivity [m2.s-1]": 3.2e-14,
                    "Particle radius [m]": 1e-06,
                    "OCP [V]": "-3.04420906 * x + 10.04892207 - 0.65637536 * tanh(-4.02134095 * (x - 0.80063948)) + 4.24678547 * tanh(12.17805062 * (x - 7.57659337)) - 0.3757068 * tanh(59.33067782 * (x - 0.99784492))",
                    "Entropic change coefficient [V.K-1]": -1e-4,
                    "Surface area per unit volume [m-1]": 496883,
                    "Reaction rate constant [mol.m-2.s-1]": 2.305e-05,
                    "Minimum stoichiometry": 0.42424,
                    "Maximum stoichiometry": 0.96210,
                    "Maximum concentration [mol.m-3]": 46200,
                    "Diffusivity activation energy [J.mol-1]": 15000,
                    "Reaction rate constant activation energy [J.mol-1]": 3500,
                },
            },
        }

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
            # initial concentration must be set manually for blended models (for now)
            pv.update(
                {
                    "Initial concentration in negative electrode [mol.m-3]": 22000,
                    "Primary: Initial concentration in positive electrode [mol.m-3]": 19404,
                    "Secondary: Initial concentration in positive electrode [mol.m-3]": 19404,
                },
                check_already_exists=False,
            )
            model = pybamm.lithium_ion.SPM({"particle phases": ("1", "2")})
            experiment = pybamm.Experiment(
                [
                    "Discharge at C/5 for 1 hour",
                ]
            )
            sim = pybamm.Simulation(model, parameter_values=pv, experiment=experiment)
            sim.solve(calc_esoh=False)

    def test_bpx_blended_error(self):
        bpx_obj = copy.copy(self.base)
        bpx_obj["Parameterisation"]["Positive electrode"] = {
            "Thickness [m]": 5.23e-05,
            "Conductivity [S.m-1]": 0.789,
            "Porosity": 0.277493,
            "Transport efficiency": 0.1462,
            "Particle": {
                "Large Particles": {
                    "Diffusivity [m2.s-1]": 3.2e-14,
                    "Particle radius [m]": 8e-06,
                    "OCP [V]": "-3.04420906 * x + 10.04892207 - 0.65637536 * tanh(-4.02134095 * (x - 0.80063948)) + 4.24678547 * tanh(12.17805062 * (x - 7.57659337)) - 0.3757068 * tanh(59.33067782 * (x - 0.99784492))",
                    "Entropic change coefficient [V.K-1]": -1e-4,
                    "Surface area per unit volume [m-1]": 186331,
                    "Reaction rate constant [mol.m-2.s-1]": 2.305e-05,
                    "Minimum stoichiometry": 0.42424,
                    "Maximum stoichiometry": 0.96210,
                    "Maximum concentration [mol.m-3]": 46200,
                    "Diffusivity activation energy [J.mol-1]": 15000,
                    "Reaction rate constant activation energy [J.mol-1]": 3500,
                },
                "Medium Particles": {
                    "Diffusivity [m2.s-1]": 3.2e-14,
                    "Particle radius [m]": 4e-06,
                    "OCP [V]": "-3.04420906 * x + 10.04892207 - 0.65637536 * tanh(-4.02134095 * (x - 0.80063948)) + 4.24678547 * tanh(12.17805062 * (x - 7.57659337)) - 0.3757068 * tanh(59.33067782 * (x - 0.99784492))",
                    "Entropic change coefficient [V.K-1]": -1e-4,
                    "Surface area per unit volume [m-1]": 186331,
                    "Reaction rate constant [mol.m-2.s-1]": 2.305e-05,
                    "Minimum stoichiometry": 0.42424,
                    "Maximum stoichiometry": 0.96210,
                    "Maximum concentration [mol.m-3]": 46200,
                    "Diffusivity activation energy [J.mol-1]": 15000,
                    "Reaction rate constant activation energy [J.mol-1]": 3500,
                },
                "Small Particles": {
                    "Diffusivity [m2.s-1]": 3.2e-14,
                    "Particle radius [m]": 1e-06,
                    "OCP [V]": "-3.04420906 * x + 10.04892207 - 0.65637536 * tanh(-4.02134095 * (x - 0.80063948)) + 4.24678547 * tanh(12.17805062 * (x - 7.57659337)) - 0.3757068 * tanh(59.33067782 * (x - 0.99784492))",
                    "Entropic change coefficient [V.K-1]": -1e-4,
                    "Surface area per unit volume [m-1]": 186331,
                    "Reaction rate constant [mol.m-2.s-1]": 2.305e-05,
                    "Minimum stoichiometry": 0.42424,
                    "Maximum stoichiometry": 0.96210,
                    "Maximum concentration [mol.m-3]": 46200,
                    "Diffusivity activation energy [J.mol-1]": 15000,
                    "Reaction rate constant activation energy [J.mol-1]": 3500,
                },
            },
        }

        filename = "tmp.json"
        with tempfile.NamedTemporaryFile(
            suffix=filename, delete=False, mode="w"
        ) as tmp:
            # write to a tempory file so we can
            # get the source later on using inspect.getsource
            # (as long as the file still exists)
            json.dump(bpx_obj, tmp)
            tmp.flush()

            with self.assertRaisesRegex(NotImplementedError, "PyBaMM does not support"):
                pybamm.ParameterValues.create_from_bpx(tmp.name)

    def test_bpx_user_defined(self):
        bpx_obj = copy.copy(self.base)
        data = {"x": [0, 1], "y": [0, 1]}
        bpx_obj["Parameterisation"]["User-defined"] = {
            "User-defined scalar parameter": 1.0,
            "User-defined parameter data": data,
            "User-defined parameter data function": "x**2",
        }

        filename = "tmp.json"
        with tempfile.NamedTemporaryFile(
            suffix=filename, delete=False, mode="w"
        ) as tmp:
            # write to a tempory file so we can
            # get the source later on using inspect.getsource
            # (as long as the file still exists)
            json.dump(bpx_obj, tmp)
            tmp.flush()

            param = pybamm.ParameterValues.create_from_bpx(tmp.name)

            self.assertEqual(param["User-defined scalar parameter"], 1.0)
            var = pybamm.Variable("var")
            self.assertIsInstance(
                param["User-defined parameter data"](var), pybamm.Interpolant
            )
            self.assertIsInstance(
                param["User-defined parameter data function"](var), pybamm.Power
            )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
