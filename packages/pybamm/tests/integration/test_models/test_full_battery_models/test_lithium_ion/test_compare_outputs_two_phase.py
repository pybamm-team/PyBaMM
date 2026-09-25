#
# Tests for the surface formulation
#
import numpy as np
import pytest

import pybamm


class TestCompareOutputsTwoPhase:
    def compare_outputs_two_phase_graphite_graphite(self, model_class):
        """
        Check that a two-phase graphite-graphite model gives the same results as a
        standard one-phase graphite model
        """
        # Standard model
        model = model_class()
        parameter_values = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(model, parameter_values=parameter_values)
        t_eval = [0, 3600]
        t_interp = np.linspace(0, 3600)
        sol = sim.solve(t_eval=t_eval, t_interp=t_interp)

        # Two phase model
        model_two_phase = model_class({"particle phases": ("2", "1")})

        ratio = pybamm.InputParameter("ratio")
        parameter_values_two_phase = pybamm.ParameterValues("Chen2020")

        for parameter in [
            "Negative electrode OCP [V]",
            "Negative electrode OCP entropic change [V.K-1]",
            "Maximum concentration in negative electrode [mol.m-3]",
            "Initial concentration in negative electrode [mol.m-3]",
            "Negative particle radius [m]",
            "Negative particle diffusivity [m2.s-1]",
            "Negative electrode exchange-current density [A.m-2]",
        ]:
            parameter_values_two_phase.update(
                {
                    f"Primary: {parameter}": parameter_values_two_phase[parameter],
                    f"Secondary: {parameter}": parameter_values_two_phase[parameter],
                }
            )
            del parameter_values_two_phase[parameter]
        parameter_values_two_phase.update(
            {
                "Primary: Negative electrode active material volume fraction"
                "": parameter_values_two_phase[
                    "Negative electrode active material volume fraction"
                ]
                * ratio,
                "Secondary: Negative electrode active material volume "
                "fraction": parameter_values_two_phase[
                    "Negative electrode active material volume fraction"
                ]
                * (1 - ratio),
            }
        )
        del parameter_values_two_phase[
            "Negative electrode active material volume fraction"
        ]

        sim = pybamm.Simulation(
            model_two_phase, parameter_values=parameter_values_two_phase
        )
        for x in [0.1, 0.3, 0.5]:
            sol_two_phase = sim.solve(
                t_eval=t_eval, t_interp=t_interp, inputs={"ratio": x}
            )
            # Compare two phase model to standard model
            for variable in [
                "X-averaged negative electrode active material volume fraction",
                (
                    "X-averaged negative electrode volumetric "
                    "interfacial current density [A.m-3]"
                ),
                "Voltage [V]",
            ]:
                np.testing.assert_allclose(
                    sol[variable].entries, sol_two_phase[variable].entries, rtol=1e-2
                )

            # Compare each phase in the two-phase model
            np.testing.assert_allclose(
                sol_two_phase[
                    "Negative primary particle concentration [mol.m-3]"
                ].entries,
                sol_two_phase[
                    "Negative secondary particle concentration [mol.m-3]"
                ].entries,
                rtol=1e-6,
            )
            np.testing.assert_allclose(
                sol_two_phase[
                    "Negative electrode primary volumetric "
                    "interfacial current density [A.m-3]"
                ].entries
                / x,
                sol_two_phase[
                    "Negative electrode secondary volumetric "
                    "interfacial current density [A.m-3]"
                ].entries
                / (1 - x),
                rtol=1e-6,
            )
            np.testing.assert_allclose(
                sol_two_phase[
                    "Negative electrode primary active material volume fraction"
                ].entries
                / x,
                sol_two_phase[
                    "Negative electrode secondary active material volume fraction"
                ].entries
                / (1 - x),
                rtol=1e-6,
            )

    def test_compare_SPM_graphite_graphite(self):
        model_class = pybamm.lithium_ion.SPM
        self.compare_outputs_two_phase_graphite_graphite(model_class)

    def test_compare_SPMe_graphite_graphite(self):
        model_class = pybamm.lithium_ion.SPMe
        self.compare_outputs_two_phase_graphite_graphite(model_class)

    def test_compare_DFN_graphite_graphite(self):
        model_class = pybamm.lithium_ion.DFN
        self.compare_outputs_two_phase_graphite_graphite(model_class)

    def compare_outputs_two_phase_silicon_graphite(self, model_class):
        # Check that increasing silicon content has the expected effect
        options = {
            "particle phases": ("2", "1"),
            "open-circuit potential": (("single", "current sigmoid"), "single"),
        }
        model = model_class(options)

        name = "Negative electrode active material volume fraction"
        param = pybamm.ParameterValues("Chen2020_composite")
        x = pybamm.InputParameter("x")
        param.update(
            {
                f"Primary: {name}": (1 - x) * 0.75,
                f"Secondary: {name}": x * 0.75,
                "Current function [A]": 5 / 2,
            }
        )

        sim = pybamm.Simulation(model, parameter_values=param)
        t_eval = [0, 8000]
        t_interp = np.linspace(0, 8000, 1000)
        inputs = [{"x": 0.01}, {"x": 0.1}]
        sol = sim.solve(t_eval=t_eval, t_interp=t_interp, inputs=inputs)

        # Starting values should be close
        for var in [
            "Voltage [V]",
            "Average negative primary particle concentration",
            "Average negative secondary particle concentration",
        ]:
            np.testing.assert_allclose(
                sol[0][var].data[:20], sol[1][var].data[:20], rtol=1e-2
            )

        # More silicon means longer sim
        assert sol[0]["Time [s]"].data[-1] < sol[1]["Time [s]"].data[-1]

    def test_compare_SPM_silicon_graphite(self):
        model_class = pybamm.lithium_ion.SPM
        self.compare_outputs_two_phase_silicon_graphite(model_class)

    def test_compare_SPMe_silicon_graphite(self):
        model_class = pybamm.lithium_ion.SPMe
        self.compare_outputs_two_phase_silicon_graphite(model_class)

    def test_compare_DFN_silicon_graphite(self):
        model_class = pybamm.lithium_ion.DFN
        self.compare_outputs_two_phase_silicon_graphite(model_class)

    def compare_heat_sources_two_phase_graphite_graphite(
        self, model_class, thermal_options, parameter_set="Chen2020"
    ):
        """
        Every heat source is a sum over the particle phases, so splitting one graphite
        phase into two identical halves must reproduce the one-phase heat sources
        """
        # composite models default to the algebraic surface form, so pin it for the
        # one-phase model too and compare like with like
        options = {
            **thermal_options,
            "heat of mixing": "true",
            "surface form": "algebraic",
        }
        t_eval = [0, 3600]
        t_interp = np.linspace(0, 3600)

        parameter_values = pybamm.ParameterValues(parameter_set)
        if parameter_set == "Chen2020":
            # Chen2020 has zero entropic change, which would leave the reversible
            # heat and the temperature dependence of the heat of mixing untested
            entropic_change = pybamm.ParameterValues("Ai2020")
            for domain in ["Negative", "Positive"]:
                name = f"{domain} electrode OCP entropic change [V.K-1]"
                parameter_values[name] = entropic_change[name]

        # the default rtol lets a temperature state near 300 K drift by ~0.03 K,
        # which is comparable to the one- and two-phase differences being tested
        solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-8)
        sol = pybamm.Simulation(
            model_class(options), parameter_values=parameter_values, solver=solver
        ).solve(t_eval=t_eval, t_interp=t_interp)

        parameter_values_two_phase = parameter_values.copy()
        for parameter in [
            "Negative electrode OCP [V]",
            "Negative electrode OCP entropic change [V.K-1]",
            "Maximum concentration in negative electrode [mol.m-3]",
            "Initial concentration in negative electrode [mol.m-3]",
            "Negative particle radius [m]",
            "Negative particle diffusivity [m2.s-1]",
            "Negative electrode exchange-current density [A.m-2]",
            "Negative electrode active material volume fraction",
        ]:
            value = parameter_values_two_phase[parameter]
            if parameter.endswith("active material volume fraction"):
                value = value / 2
            parameter_values_two_phase.update(
                {f"Primary: {parameter}": value, f"Secondary: {parameter}": value}
            )
            del parameter_values_two_phase[parameter]

        sol_two_phase = pybamm.Simulation(
            model_class({"particle phases": ("2", "1"), **options}),
            parameter_values=parameter_values_two_phase,
            solver=solver,
        ).solve(t_eval=t_eval, t_interp=t_interp)

        for variable in [
            "Volume-averaged irreversible electrochemical heating [W.m-3]",
            "Volume-averaged reversible heating [W.m-3]",
            "Volume-averaged hysteresis electrochemical heating [W.m-3]",
            "Volume-averaged heat of mixing [W.m-3]",
            "Volume-averaged total heating [W.m-3]",
            "Voltage [V]",
        ]:
            one_phase = sol[variable](t_interp)
            # scale atol to the variable, since the reversible heat changes sign
            np.testing.assert_allclose(
                sol_two_phase[variable](t_interp),
                one_phase,
                rtol=1e-2,
                atol=1e-3 * np.nanmax(np.abs(one_phase)) + 1e-8,
            )

        if options["thermal"] != "isothermal":
            temperature = "Volume-averaged cell temperature [K]"
            temperature_rise = sol[temperature](t_interp) - sol[temperature](0)
            temperature_rise_two_phase = sol_two_phase[temperature](
                t_interp
            ) - sol_two_phase[temperature](0)
            np.testing.assert_allclose(
                temperature_rise_two_phase,
                temperature_rise,
                rtol=1e-2,
                atol=1e-3 * np.nanmax(temperature_rise),
            )

    @pytest.mark.parametrize(
        ("model_class", "thermal_options", "parameter_set"),
        [
            pytest.param(model_class, options, parameter_set, id=f"{name}-{label}")
            for name, model_class in [
                ("SPM", pybamm.lithium_ion.SPM),
                ("SPMe", pybamm.lithium_ion.SPMe),
                ("DFN", pybamm.lithium_ion.DFN),
            ]
            for label, options, parameter_set in [
                (
                    "isothermal",
                    {
                        "thermal": "isothermal",
                        "calculate heat source for isothermal models": "true",
                    },
                    "Chen2020",
                ),
                ("lumped", {"thermal": "lumped"}, "Chen2020"),
                # x-full resolves the temperature in x, so the heat of mixing uses
                # the local rather than the x-averaged electrode temperature
                (
                    "x-full",
                    {"thermal": "x-full", "cell geometry": "pouch"},
                    "Marquis2019",
                ),
                (
                    "x-lumped-1plus1D",
                    {
                        "thermal": "x-lumped",
                        "cell geometry": "pouch",
                        "current collector": "potential pair",
                        "dimensionality": 1,
                    },
                    "Marquis2019",
                ),
            ]
        ],
    )
    def test_compare_heat_sources_graphite_graphite(
        self, model_class, thermal_options, parameter_set
    ):
        self.compare_heat_sources_two_phase_graphite_graphite(
            model_class, thermal_options, parameter_set
        )

    def heat_of_mixing_silicon_graphite(self, model_class):
        # Check the heat of mixing of two genuinely different phases heats the cell
        options = {
            "particle phases": ("2", "1"),
            "open-circuit potential": (("single", "current sigmoid"), "single"),
            "thermal": "lumped",
        }
        parameter_values = pybamm.ParameterValues("Chen2020_composite")
        # thermal properties are per layer, but the set only has per-phase densities
        parameter_values.update(
            {"Negative electrode density [kg.m-3]": 1657.0},
            check_already_exists=False,
        )
        # the shipped graphite OCP is a cubic Interpolant, whose derivative cannot
        # yet be converted for the solver, so use the analytic Chen2020 fit
        parameter_values["Primary: Negative electrode OCP [V]"] = (
            pybamm.ParameterValues("Chen2020")["Negative electrode OCP [V]"]
        )
        t_eval = [0, 3600]
        t_interp = np.linspace(0, 3600)

        solutions = {
            heat_of_mixing: pybamm.Simulation(
                model_class({**options, "heat of mixing": heat_of_mixing}),
                parameter_values=parameter_values,
            ).solve(t_eval=t_eval, t_interp=t_interp)
            for heat_of_mixing in ["false", "true"]
        }

        heat_of_mixing = solutions["true"]["Volume-averaged heat of mixing [W.m-3]"](
            t_interp
        )
        heat_of_mixing = heat_of_mixing[~np.isnan(heat_of_mixing)]
        assert np.all(heat_of_mixing > 0)
        temperature = "Volume-averaged cell temperature [K]"
        assert solutions["true"][temperature](t_interp[-2]) > solutions["false"][
            temperature
        ](t_interp[-2])

    def test_heat_of_mixing_SPM_silicon_graphite(self):
        self.heat_of_mixing_silicon_graphite(pybamm.lithium_ion.SPM)

    def test_heat_of_mixing_DFN_silicon_graphite(self):
        self.heat_of_mixing_silicon_graphite(pybamm.lithium_ion.DFN)
