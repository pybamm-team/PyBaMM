import copy
import json
import math
from typing import Any

import numpy as np
import pytest

import pybamm


class TestBPX:
    def setup_method(self):
        self.base: dict[str, Any] = {
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
                    "Number of electrode pairs connected in parallel to make a cell": 34,
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

    def test_bpx(self, tmp_path):
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
            copy.deepcopy(self.base),
        ]

        model = pybamm.lithium_ion.DFN()
        experiment = pybamm.Experiment(
            [
                "Discharge at C/5 for 1 hour",
            ]
        )

        sols = []
        for obj in bpx_objs:
            temp_file = tmp_path / "tmp.json"
            temp_file.write_text(json.dumps(obj))
            pv = pybamm.ParameterValues.create_from_bpx(temp_file)
            sim = pybamm.Simulation(model, parameter_values=pv, experiment=experiment)
            sols.append(sim.solve())

        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                sols[0]["Voltage [V]"].data, sols[1]["Voltage [V]"].data, atol=1e-7
            )

    def test_no_already_exists_in_BPX(self, tmp_path):
        temp_file = tmp_path / "tmp.json"
        temp_file.write_text(json.dumps(copy.deepcopy(self.base)))

        params = pybamm.ParameterValues.create_from_bpx(temp_file)
        assert "check_already_exists" not in params.keys()

    def test_constant_functions(self, tmp_path):
        bpx_obj = copy.deepcopy(self.base)
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

        temp_file = tmp_path / "tmp.json"
        temp_file.write_text(json.dumps(bpx_obj))

        param = pybamm.ParameterValues.create_from_bpx(temp_file)

        # Function to check that functional parameters output constants
        def check_constant_output(func):
            stos = [0, 1]
            T = 298.15
            p_vals = [func(sto, T) for sto in stos]
            assert p_vals[0] == p_vals[1]

        for electrode in ["Negative", "Positive"]:
            D = param[f"{electrode} particle diffusivity [m2.s-1]"]
            dUdT = param[f"{electrode} electrode OCP entropic change [V.K-1]"]
            check_constant_output(D)
            assert dUdT == 1

        kappa = param["Electrolyte conductivity [S.m-1]"]
        De = param["Electrolyte diffusivity [m2.s-1]"]
        check_constant_output(kappa)
        check_constant_output(De)

    def test_table_data(self, tmp_path):
        bpx_obj = copy.deepcopy(self.base)
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

        temp_file = tmp_path / "tmp.json"
        temp_file.write_text(json.dumps(bpx_obj))

        param = pybamm.ParameterValues.create_from_bpx(temp_file)

        # Check that the electrolyte conductivity is an Interpolant with the
        # correct child
        c = pybamm.Variable("c")
        kappa = param["Electrolyte conductivity [S.m-1]"](c, 298.15)
        assert isinstance(kappa, pybamm.Interpolant)
        assert kappa.children[0] == c
        # Check other parameters give interpolants
        D = param["Electrolyte diffusivity [m2.s-1]"](c, 298.15)
        assert isinstance(D, pybamm.Interpolant)
        for electrode in ["Negative", "Positive"]:
            D = param[f"{electrode} particle diffusivity [m2.s-1]"](c, 298.15)
            assert isinstance(D, pybamm.Interpolant)
            OCP = param[f"{electrode} electrode OCP [V]"](c)
            assert isinstance(OCP, pybamm.Interpolant)
            dUdT = param[f"{electrode} electrode OCP entropic change [V.K-1]"](c)
            assert isinstance(dUdT, pybamm.Interpolant)

    def test_bpx_soc_error(self):
        bpx_obj = copy.deepcopy(self.base)
        with pytest.raises(ValueError, match=r"Target SOC"):
            pybamm.ParameterValues.create_from_bpx_obj(bpx_obj, target_soc=10)

    def test_bpx_arrhenius(self, tmp_path):
        bpx_obj = copy.deepcopy(self.base)

        temp_file = tmp_path / "tmp.json"
        temp_file.write_text(json.dumps(bpx_obj))

        pv = pybamm.ParameterValues.create_from_bpx(temp_file)

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
                    pv[param_key](c_e, c_s_surf, c_s_max, T).evaluate()
                    / pv[param_key](c_e, c_s_surf, c_s_max, T_ref).evaluate()
                )
            else:
                eval_ratio = (
                    pv[param_key](sto, T).evaluate()
                    / pv[param_key](sto, T_ref).evaluate()
                )

            calc_ratio = pybamm.exp(
                Ea / pybamm.constants.R * (1 / T_ref - 1 / T)
            ).evaluate()

            assert eval_ratio == pytest.approx(calc_ratio)

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

        for param_key, Ea_key in zip(param_keys, Ea_keys, strict=False):
            arrhenius_assertion(pv, param_key, Ea_key)

    def test_bpx_blended(self, tmp_path):
        bpx_obj = copy.deepcopy(self.base)
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

        temp_file = tmp_path / "tmp.json"
        temp_file.write_text(json.dumps(bpx_obj))

        pv = pybamm.ParameterValues.create_from_bpx(temp_file)

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

    def test_bpx_blended_error(self, tmp_path):
        bpx_obj = copy.deepcopy(self.base)
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

        temp_file = tmp_path / "tmp.json"
        temp_file.write_text(json.dumps(bpx_obj))
        with pytest.raises(NotImplementedError, match=r"PyBaMM does not support"):
            pybamm.ParameterValues.create_from_bpx(temp_file)

    def test_bpx_user_defined(self, tmp_path):
        bpx_obj = copy.deepcopy(self.base)
        data = {"x": [0, 1], "y": [0, 1]}
        bpx_obj["Parameterisation"]["User-defined"] = {
            "User-defined scalar parameter": 1.0,
            "User-defined parameter data": data,
            "User-defined parameter data function": "x**2",
        }

        temp_file = tmp_path / "tmp.json"
        temp_file.write_text(json.dumps(bpx_obj))

        param = pybamm.ParameterValues.create_from_bpx(temp_file)

        assert param["User-defined scalar parameter"] == pytest.approx(1.0, rel=1e-12)
        var = pybamm.Variable("var")
        assert isinstance(param["User-defined parameter data"](var), pybamm.Interpolant)
        assert isinstance(
            param["User-defined parameter data function"](var), pybamm.Power
        )

    def test_bpx_activation_energy_default(self, tmp_path):
        bpx_obj = copy.deepcopy(self.base)
        del bpx_obj["Parameterisation"]["Negative electrode"][
            "Diffusivity activation energy [J.mol-1]"
        ]

        test_file = tmp_path / "test.json"
        test_file.write_text(json.dumps(bpx_obj))

        param = pybamm.ParameterValues.create_from_bpx(test_file)

        assert param[
            "Negative electrode diffusivity activation energy [J.mol-1]"
        ] == pytest.approx(0.0, rel=1e-12)

    def test_bpx_from_obj(self):
        bpx_obj = copy.deepcopy(self.base)
        param = pybamm.ParameterValues.create_from_bpx_obj(bpx_obj)
        assert isinstance(param, pybamm.ParameterValues)

    def test_bruggeman_coefficient_calculation(self):
        bpx_obj = copy.deepcopy(self.base)
        domains = {
            "Negative electrode": {"Porosity": 0.42, "Transport efficiency": 0.75},
            "Separator": {"Porosity": 0.50, "Transport efficiency": 0.80},
            "Positive electrode": {"Porosity": 0.38, "Transport efficiency": 0.70},
        }
        for domain, values in domains.items():
            bpx_obj["Parameterisation"][domain].update(values)

        param = pybamm.ParameterValues.create_from_bpx_obj(bpx_obj)
        pre_names = ["Negative electrode ", "Separator ", "Positive electrode "]
        for pre_name, domain in zip(pre_names, domains.values(), strict=True):
            expected = math.log(domain["Transport efficiency"]) / math.log(
                domain["Porosity"]
            )
            calculated = param[pre_name + "Bruggeman coefficient (electrolyte)"]
            np.testing.assert_allclose(calculated, expected, atol=1e-6)

    def test_bruggeman_invalid_values_raise(self):
        bpx_obj = copy.deepcopy(self.base)
        bpx_obj["Parameterisation"]["Negative electrode"]["Porosity"] = 0  # Invalid

        with pytest.raises(
            ValueError, match=r"math domain error"
        ):  # Matches log(0) error
            pybamm.ParameterValues.create_from_bpx_obj(bpx_obj)
