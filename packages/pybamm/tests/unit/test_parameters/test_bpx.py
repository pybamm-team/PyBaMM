import copy
import json
import math
import warnings
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
                    "Reference temperature [K]": 298.15,
                    "Lower voltage cut-off [V]": 2.8,
                    "Upper voltage cut-off [V]": 4.2,
                    "Nominal cell capacity [A.h]": 12.5,
                    "Specific heat capacity [J.K-1.kg-1]": 913,
                    "Density [kg.m-3]": 1847,
                    "Electrode area [m2]": 0.016808,
                    "Number of electrode pairs connected in parallel to make a cell": 34,
                    "External surface area [m2]": 3.79e-2,
                    "Volume [m3]": 1.28e-4,
                },
                "Electrolyte": {
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
            "State": {
                "Initial conditions": {
                    "Initial temperature [K]": 298.15,
                    "Initial electrolyte concentration [mol.m-3]": 1000,
                    "Initial hysteresis state: Positive electrode": 0.0,
                    "Initial hysteresis state: Negative electrode": 0.0,
                },
                "Thermal environment": {
                    "Ambient temperature [K]": 298.15,
                    "Heat transfer coefficient [W.m-2.K-1]": 10.0,
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

    def test_bpx_emits_current_particle_diffusivity_name(self, tmp_path):
        # a BPX-derived set must use only the current "particle diffusivity" name;
        # emitting the deprecated "electrode diffusivity" alias too would let it
        # silently clobber the current value on any later re-normalisation
        temp_file = tmp_path / "tmp.json"
        temp_file.write_text(json.dumps(copy.deepcopy(self.base)))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            param = pybamm.ParameterValues.create_from_bpx(temp_file)
        assert not [
            w
            for w in caught
            if "diffusivity" in str(w.message) and "renamed" in str(w.message)
        ]

        for electrode in ["Negative", "Positive"]:
            assert f"{electrode} particle diffusivity [m2.s-1]" in param
            assert f"{electrode} electrode diffusivity [m2.s-1]" not in param
            assert (
                f"{electrode} particle diffusivity activation energy [J.mol-1]" in param
            )
            assert (
                f"{electrode} electrode diffusivity activation energy [J.mol-1]"
                not in param
            )

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

    @pytest.mark.parametrize("form", ["function", "constant", "table"])
    def test_bpx_to_json_roundtrip(self, tmp_path, form):
        """create_from_bpx stores functional parameters as keyword-bound partials;
        serialising them must not raise (regression for the partial arg handling
        in convert_function_to_symbolic_expression)."""
        bpx_obj = copy.deepcopy(self.base)
        if form != "function":
            value = 1 if form == "constant" else {"x": [0, 1], "y": [0, 1]}
            for section in ["Electrolyte", "Negative electrode", "Positive electrode"]:
                bpx_obj["Parameterisation"][section]["Diffusivity [m2.s-1]"] = value
            bpx_obj["Parameterisation"]["Electrolyte"]["Conductivity [S.m-1]"] = value

        temp_file = tmp_path / "tmp.json"
        temp_file.write_text(json.dumps(bpx_obj))
        param = pybamm.ParameterValues.create_from_bpx(temp_file)

        out_file = tmp_path / "out.json"
        param.to_json(out_file)
        # the round-tripped values load back without error
        pybamm.ParameterValues.from_json(out_file)

    def test_bpx_soc_error(self):
        bpx_obj = copy.deepcopy(self.base)
        with (
            pytest.warns(DeprecationWarning, match="target_soc"),
            pytest.raises(ValueError, match=r"Target SOC"),
        ):
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
        # Update hysteresis states for blended electrode
        bpx_obj["State"]["Initial conditions"][
            "Initial hysteresis state: Positive electrode"
        ] = {"Large Particles": 0.0, "Small Particles": 0.0}

        temp_file = tmp_path / "tmp.json"
        temp_file.write_text(json.dumps(bpx_obj))

        pv = pybamm.ParameterValues.create_from_bpx(temp_file)

        # initial concentration must be set manually for blended models (for now)
        pv.update(
            {
                "Initial concentration in negative electrode [mol.m-3]": 22000,
                "Primary: Initial concentration in positive electrode [mol.m-3]": 19404,
                "Secondary: Initial concentration in positive electrode [mol.m-3]": 19404,
            }
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
        # Update hysteresis states for blended electrode with 3 phases
        bpx_obj["State"]["Initial conditions"][
            "Initial hysteresis state: Positive electrode"
        ] = {"Large Particles": 0.0, "Medium Particles": 0.0, "Small Particles": 0.0}

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
            ValueError, match=r"math domain error|expected a positive input"
        ):  # Matches log(0) error (message changed in Python 3.14)
            pybamm.ParameterValues.create_from_bpx_obj(bpx_obj)

    @staticmethod
    def _blended_positive_phase(radius, sav, *, with_hysteresis=False, decay=None):
        phase = {
            "Diffusivity [m2.s-1]": 3.2e-14,
            "Particle radius [m]": radius,
            "OCP [V]": "4.0 - 0.5 * x",
            "Entropic change coefficient [V.K-1]": -1e-4,
            "Surface area per unit volume [m-1]": sav,
            "Reaction rate constant [mol.m-2.s-1]": 2.305e-05,
            "Minimum stoichiometry": 0.42424,
            "Maximum stoichiometry": 0.96210,
            "Maximum concentration [mol.m-3]": 46200,
            "Diffusivity activation energy [J.mol-1]": 15000,
            "Reaction rate constant activation energy [J.mol-1]": 3500,
        }
        if with_hysteresis:
            phase.update(
                {
                    "OCP (lithiation) [V]": "4.0 - 0.5 * x",
                    "OCP (delithiation) [V]": "4.05 - 0.5 * x",
                    "OCP hysteresis decay constant": decay,
                }
            )
        return phase

    def _blended_positive_electrode(self, *, with_hysteresis=False):
        return {
            "Thickness [m]": 5.23e-05,
            "Conductivity [S.m-1]": 0.789,
            "Porosity": 0.277493,
            "Transport efficiency": 0.1462,
            "Particle": {
                "Large Particles": self._blended_positive_phase(
                    8e-06, 186331, with_hysteresis=with_hysteresis, decay=0.03
                ),
                "Small Particles": self._blended_positive_phase(
                    1e-06, 496883, with_hysteresis=with_hysteresis, decay=0.04
                ),
            },
        }

    def test_bpx_hysteresis_names(self):
        bpx_obj = copy.deepcopy(self.base)
        bpx_obj["Parameterisation"]["Negative electrode"].update(
            {
                "OCP (lithiation) [V]": {"x": [0, 1], "y": [0.1, 1.5]},
                "OCP (delithiation) [V]": {"x": [0, 1], "y": [0.15, 1.55]},
                "OCP hysteresis decay constant": 0.01,
            }
        )
        bpx_obj["Parameterisation"]["Positive electrode"].update(
            {
                "OCP (lithiation) [V]": {"x": [0, 1], "y": [4.2, 3.0]},
                "OCP (delithiation) [V]": {"x": [0, 1], "y": [4.25, 3.05]},
                "OCP hysteresis decay constant": 0.02,
            }
        )

        param = pybamm.ParameterValues.create_from_bpx_obj(bpx_obj)

        for electrode in ("Negative", "Positive"):
            assert f"{electrode} electrode lithiation OCP [V]" in param
            assert f"{electrode} electrode delithiation OCP [V]" in param
            # interpolated-table OCP branches must be converted to usable
            # expression-tree builders, not left as raw (name, (x, y)) tuples
            for branch in ("lithiation", "delithiation"):
                U = param[f"{electrode} electrode {branch} OCP [V]"]
                assert not isinstance(U, tuple)
                assert callable(U)
            assert f"{electrode} particle lithiation hysteresis decay rate" in param
            assert f"{electrode} particle delithiation hysteresis decay rate" in param
            assert f"{electrode} electrode OCP (lithiation) [V]" not in param
            assert f"{electrode} electrode OCP (delithiation) [V]" not in param
            assert f"{electrode} electrode OCP hysteresis decay constant" not in param

        assert param["Negative particle lithiation hysteresis decay rate"] == 0.01
        assert param["Negative particle delithiation hysteresis decay rate"] == 0.01
        assert param["Positive particle lithiation hysteresis decay rate"] == 0.02
        assert param["Positive particle delithiation hysteresis decay rate"] == 0.02

    def test_bpx_hysteresis_missing_fields_skipped(self):
        bpx_obj = copy.deepcopy(self.base)
        param = pybamm.ParameterValues.create_from_bpx_obj(bpx_obj)

        for electrode in ("Negative", "Positive"):
            assert f"{electrode} electrode lithiation OCP [V]" not in param
            assert f"{electrode} electrode delithiation OCP [V]" not in param
            assert f"{electrode} particle lithiation hysteresis decay rate" not in param
            assert (
                f"{electrode} particle delithiation hysteresis decay rate" not in param
            )

    def test_bpx_hysteresis_blended(self):
        bpx_obj = copy.deepcopy(self.base)
        bpx_obj["Parameterisation"]["Positive electrode"] = (
            self._blended_positive_electrode(with_hysteresis=True)
        )
        bpx_obj["State"]["Initial conditions"][
            "Initial hysteresis state: Positive electrode"
        ] = {"Large Particles": 0.0, "Small Particles": 0.0}

        param = pybamm.ParameterValues.create_from_bpx_obj(bpx_obj)

        for phase, value in (("Primary", 0.03), ("Secondary", 0.04)):
            assert (
                param[f"{phase}: Positive particle lithiation hysteresis decay rate"]
                == value
            )
            assert (
                param[f"{phase}: Positive particle delithiation hysteresis decay rate"]
                == value
            )
            assert f"{phase}: Positive electrode lithiation OCP [V]" in param
            assert f"{phase}: Positive electrode delithiation OCP [V]" in param

    def test_bpx_initial_hysteresis_state_scalar(self):
        bpx_obj = copy.deepcopy(self.base)
        bpx_obj["State"]["Initial conditions"][
            "Initial hysteresis state: Negative electrode"
        ] = 0.5
        bpx_obj["State"]["Initial conditions"][
            "Initial hysteresis state: Positive electrode"
        ] = -0.25

        param = pybamm.ParameterValues.create_from_bpx_obj(bpx_obj)

        assert param["Initial hysteresis state in negative electrode"] == 0.5
        assert param["Initial hysteresis state in positive electrode"] == -0.25

    def test_bpx_initial_hysteresis_state_blended(self):
        bpx_obj = copy.deepcopy(self.base)
        bpx_obj["Parameterisation"]["Positive electrode"] = (
            self._blended_positive_electrode()
        )
        bpx_obj["State"]["Initial conditions"][
            "Initial hysteresis state: Positive electrode"
        ] = {"Large Particles": 0.7, "Small Particles": 0.3}
        bpx_obj["State"]["Initial conditions"][
            "Initial hysteresis state: Negative electrode"
        ] = 1.0

        param = pybamm.ParameterValues.create_from_bpx_obj(bpx_obj)

        assert param["Primary: Initial hysteresis state in positive electrode"] == 0.7
        assert param["Secondary: Initial hysteresis state in positive electrode"] == 0.3
        assert param["Initial hysteresis state in negative electrode"] == 1.0

    def test_bpx_heat_transfer_coefficient_preserved(self):
        bpx_obj = copy.deepcopy(self.base)
        bpx_obj["State"]["Thermal environment"][
            "Heat transfer coefficient [W.m-2.K-1]"
        ] = 42.0

        param = pybamm.ParameterValues.create_from_bpx_obj(bpx_obj)

        assert param["Total heat transfer coefficient [W.m-2.K-1]"] == 42.0

    def test_bpx_axen_hysteresis_end_to_end(self):
        bpx_obj = copy.deepcopy(self.base)
        for electrode in ("Negative electrode", "Positive electrode"):
            ocp = bpx_obj["Parameterisation"][electrode]["OCP [V]"]
            bpx_obj["Parameterisation"][electrode].update(
                {
                    "OCP (lithiation) [V]": ocp,
                    "OCP (delithiation) [V]": ocp,
                    "OCP hysteresis decay constant": 0.01,
                }
            )

        pv = pybamm.ParameterValues.create_from_bpx_obj(bpx_obj)
        model = pybamm.lithium_ion.SPM(
            {"open-circuit potential": "one-state hysteresis"}
        )
        sim = pybamm.Simulation(
            model,
            parameter_values=pv,
            experiment=pybamm.Experiment(["Discharge at C/10 for 5 minutes"]),
        )
        sol = sim.solve()
        v_final = float(sol["Voltage [V]"].entries[-1])
        assert (
            pv["Lower voltage cut-off [V]"] < v_final < pv["Upper voltage cut-off [V]"]
        )

    def test_bpx_default_initial_concentrations_at_full_charge(self):
        # self.base gives no initial SOC, so concentrations default to full charge
        # (theta_max negative / theta_min positive) without invoking
        # set_initial_state.
        bpx_obj = copy.deepcopy(self.base)
        neg = bpx_obj["Parameterisation"]["Negative electrode"]
        pos = bpx_obj["Parameterisation"]["Positive electrode"]
        expected_neg = (
            neg["Maximum stoichiometry"] * neg["Maximum concentration [mol.m-3]"]
        )
        expected_pos = (
            pos["Minimum stoichiometry"] * pos["Maximum concentration [mol.m-3]"]
        )

        param = pybamm.ParameterValues.create_from_bpx_obj(bpx_obj)

        assert param["Initial concentration in negative electrode [mol.m-3]"] == (
            expected_neg
        )
        assert param["Initial concentration in positive electrode [mol.m-3]"] == (
            expected_pos
        )

    def test_bpx_initial_soc_applied_from_state(self):
        # A BPX initial state-of-charge other than full charge is applied via
        # set_initial_state, matching a manual set_initial_state on the same set.
        bpx_obj = copy.deepcopy(self.base)
        bpx_obj["State"]["Initial conditions"]["Initial state-of-charge"] = 0.25
        neg = bpx_obj["Parameterisation"]["Negative electrode"]
        full_charge_neg = (
            neg["Maximum stoichiometry"] * neg["Maximum concentration [mol.m-3]"]
        )
        param = pybamm.ParameterValues.create_from_bpx_obj(bpx_obj)

        reference = pybamm.ParameterValues.create_from_bpx_obj(copy.deepcopy(self.base))
        reference.set_initial_state(0.25)

        for key in (
            "Initial concentration in negative electrode [mol.m-3]",
            "Initial concentration in positive electrode [mol.m-3]",
        ):
            assert param[key] == pytest.approx(reference[key])
        # and it is not the full-charge value
        assert (
            param["Initial concentration in negative electrode [mol.m-3]"]
            != full_charge_neg
        )

    def test_bpx_initial_soc_out_of_range_raises(self):
        bpx_obj = copy.deepcopy(self.base)
        bpx_obj["State"]["Initial conditions"]["Initial state-of-charge"] = 1.5
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            pybamm.ParameterValues.create_from_bpx_obj(bpx_obj)

    def test_bpx_target_soc_is_deprecated_but_preserves_behaviour(self):
        from bpx import get_electrode_concentrations, parse_bpx_obj

        bpx_obj = copy.deepcopy(self.base)
        target_soc = 0.25
        expected_neg, expected_pos = get_electrode_concentrations(
            target_soc, parse_bpx_obj(copy.deepcopy(bpx_obj))
        )

        with pytest.warns(DeprecationWarning, match="target_soc"):
            param = pybamm.ParameterValues.create_from_bpx_obj(
                bpx_obj, target_soc=target_soc
            )

        assert (
            param["Initial concentration in negative electrode [mol.m-3]"]
            == expected_neg
        )
        assert (
            param["Initial concentration in positive electrode [mol.m-3]"]
            == expected_pos
        )

    def _blended_bpx_obj(self):
        bpx_obj = copy.deepcopy(self.base)
        bpx_obj["Parameterisation"]["Positive electrode"] = (
            self._blended_positive_electrode()
        )
        bpx_obj["State"]["Initial conditions"][
            "Initial hysteresis state: Positive electrode"
        ] = {"Large Particles": 0.0, "Small Particles": 0.0}
        return bpx_obj

    def test_bpx_blended_initial_soc_applied(self):
        # a BPX initial SOC on a blended electrode is applied per phase via the
        # composite electrode SOH path, not left at full charge
        bpx_obj = self._blended_bpx_obj()
        bpx_obj["State"]["Initial conditions"]["Initial state-of-charge"] = 0.5
        particle = bpx_obj["Parameterisation"]["Positive electrode"]["Particle"]
        full_charge = {
            phase: particle[phase]["Minimum stoichiometry"]
            * particle[phase]["Maximum concentration [mol.m-3]"]
            for phase in ("Large Particles", "Small Particles")
        }

        param = pybamm.ParameterValues.create_from_bpx_obj(bpx_obj)

        # both phases are initialised away from full charge by the composite path
        assert param[
            "Primary: Initial concentration in positive electrode [mol.m-3]"
        ] != pytest.approx(full_charge["Large Particles"])
        assert param[
            "Secondary: Initial concentration in positive electrode [mol.m-3]"
        ] != pytest.approx(full_charge["Small Particles"])

    def test_bpx_blended_with_target_soc_raises(self):
        bpx_obj = self._blended_bpx_obj()
        with (
            pytest.warns(DeprecationWarning, match="target_soc"),
            pytest.raises(NotImplementedError, match="blended electrodes"),
        ):
            pybamm.ParameterValues.create_from_bpx_obj(bpx_obj, target_soc=0.5)

    # Legacy BPX v0.x backward-compatibility conversion
    def _make_v0_obj(self, v1_obj=None):
        """Recast a v1.x BPX dict into the legacy v0.x layout."""
        obj = copy.deepcopy(v1_obj if v1_obj is not None else self.base)
        state = obj.pop("State")
        # v0.x used a numeric version field
        obj["Header"]["BPX"] = 0.4
        cell = obj["Parameterisation"]["Cell"]
        ic = state["Initial conditions"]
        te = state["Thermal environment"]
        cell["Initial temperature [K]"] = ic["Initial temperature [K]"]
        cell["Ambient temperature [K]"] = te["Ambient temperature [K]"]
        # v0.x described thermal behaviour via a lumped thermal conductivity
        cell["Thermal conductivity [W.m-1.K-1]"] = 1.5
        obj["Parameterisation"]["Electrolyte"]["Initial concentration [mol.m-3]"] = ic[
            "Initial electrolyte concentration [mol.m-3]"
        ]
        return obj

    def test_bpx_v0_obj_is_converted_with_warning(self):
        v0_obj = self._make_v0_obj()
        with pytest.warns(UserWarning, match="legacy BPX v0"):
            param = pybamm.ParameterValues.create_from_bpx_obj(v0_obj)
        # moved fields are preserved
        assert param["Ambient temperature [K]"] == 298.15
        assert param["Initial temperature [K]"] == 298.15
        # v0.x has no heat transfer coefficient; PyBaMM applies its adiabatic default
        assert param["Total heat transfer coefficient [W.m-2.K-1]"] == 0

    def test_bpx_v0_file_is_converted_with_warning(self, tmp_path):
        temp_file = tmp_path / "tmp.json"
        temp_file.write_text(json.dumps(self._make_v0_obj()))
        with pytest.warns(UserWarning, match="legacy BPX v0"):
            param = pybamm.ParameterValues.create_from_bpx(temp_file)
        assert param["Ambient temperature [K]"] == 298.15

    def test_bpx_v0_missing_initial_temperature_falls_back_to_ambient(self):
        v0_obj = self._make_v0_obj()
        del v0_obj["Parameterisation"]["Cell"]["Initial temperature [K]"]
        with pytest.warns(UserWarning, match="legacy BPX v0"):
            param = pybamm.ParameterValues.create_from_bpx_obj(v0_obj)
        assert param["Initial temperature [K]"] == param["Ambient temperature [K]"]

    def test_bpx_v0_blended_is_converted(self):
        v0_obj = self._make_v0_obj(self._blended_bpx_obj())
        with pytest.warns(UserWarning, match="legacy BPX v0"):
            param = pybamm.ParameterValues.create_from_bpx_obj(v0_obj)
        assert param["Ambient temperature [K]"] == 298.15

    def test_bpx_v1_float_version_not_converted(self):
        # self.base uses a float version (1.0) but is a v1.x object: it must not
        # trigger the legacy conversion path.
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            pybamm.ParameterValues.create_from_bpx_obj(copy.deepcopy(self.base))
        assert not any("legacy BPX v0" in str(r.message) for r in records)

    def test_bpx_invalid_header_raises(self):
        with pytest.raises(ValueError, match="Header"):
            pybamm.ParameterValues.create_from_bpx_obj({"Parameterisation": {}})

    # Optional BPX State handling (bpx>=1.1.1). All State fields are optional and
    # the section may be omitted entirely; PyBaMM applies its own defaults.
    def test_bpx_state_section_can_be_omitted(self):
        bpx_obj = copy.deepcopy(self.base)
        del bpx_obj["State"]
        T_ref = bpx_obj["Parameterisation"]["Cell"]["Reference temperature [K]"]
        with pytest.warns(UserWarning, match="PyBaMM applied defaults"):
            param = pybamm.ParameterValues.create_from_bpx_obj(bpx_obj)
        # temperatures fall back to the reference temperature
        assert param["Ambient temperature [K]"] == T_ref
        assert param["Initial temperature [K]"] == T_ref
        # electrolyte concentration falls back to 1 M
        assert param["Initial concentration in electrolyte [mol.m-3]"] == 1000
        # heat transfer coefficient falls back to the adiabatic default
        assert param["Total heat transfer coefficient [W.m-2.K-1]"] == 0

    def test_bpx_state_omitted_temperatures_default_to_reference(self):
        bpx_obj = copy.deepcopy(self.base)
        bpx_obj["Parameterisation"]["Cell"]["Reference temperature [K]"] = 300.0
        del bpx_obj["State"]["Initial conditions"]["Initial temperature [K]"]
        del bpx_obj["State"]["Thermal environment"]["Ambient temperature [K]"]
        param = pybamm.ParameterValues.create_from_bpx_obj(bpx_obj)
        assert param["Ambient temperature [K]"] == 300.0
        assert param["Initial temperature [K]"] == 300.0

    def test_bpx_state_omitted_electrolyte_concentration_defaults_to_1M(self):
        # PyBaMM needs c_e0 to normalise the exchange-current density, so an
        # absent electrolyte concentration defaults to 1000 mol.m-3 (1 M).
        bpx_obj = copy.deepcopy(self.base)
        del bpx_obj["State"]["Initial conditions"][
            "Initial electrolyte concentration [mol.m-3]"
        ]
        with pytest.warns(UserWarning, match="Initial concentration in electrolyte"):
            param = pybamm.ParameterValues.create_from_bpx_obj(bpx_obj)
        assert param["Initial concentration in electrolyte [mol.m-3]"] == 1000

    def test_bpx_state_omitted_hysteresis_does_not_warn(self):
        # hysteresis state is optional and usually absent; omitting it must not warn
        bpx_obj = copy.deepcopy(self.base)
        ic = bpx_obj["State"]["Initial conditions"]
        del ic["Initial hysteresis state: Positive electrode"]
        del ic["Initial hysteresis state: Negative electrode"]
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            pybamm.ParameterValues.create_from_bpx_obj(bpx_obj)
        assert not any("hysteresis" in str(r.message) for r in records)

    def test_bpx_state_omitted_heat_transfer_coefficient_defaults_to_zero(self):
        bpx_obj = copy.deepcopy(self.base)
        del bpx_obj["State"]["Thermal environment"][
            "Heat transfer coefficient [W.m-2.K-1]"
        ]
        param = pybamm.ParameterValues.create_from_bpx_obj(bpx_obj)
        assert param["Total heat transfer coefficient [W.m-2.K-1]"] == 0
