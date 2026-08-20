#
# Sweep the composite electrode SOH solver densely over its targets
#
from __future__ import annotations

import numpy as np
import pytest

import pybamm
from pybamm.models.full_battery_models.lithium_ion import electrode_soh_composite as esc
from pybamm.models.full_battery_models.lithium_ion.util import (
    get_lithiation_delithiation,
)

# Dense enough that a bracket that fails only on a sliver of the range is still hit.
SWEEP = 1000

OPTIONS = {
    "particle phases": ("2", "1"),
    "open-circuit potential": (("single", "current sigmoid"), "single"),
}

# (Q_n_1, Q_n_2, Q_p_1, Q_Li) multipliers on the nominal capacities
WEAR = {
    "nominal": (1.0, 1.0, 1.0, 1.0),
    "very low lithium": (1.0, 1.0, 1.0, 0.60),
    "lost secondary": (1.0, 0.50, 1.0, 0.90),
    "worn all": (0.70, 1.0, 0.90, 0.75),
}


class Case:
    """A composite SOH solver bound to one direction and initialisation method."""

    def __init__(self, direction, method):
        self.options = pybamm.BatteryModelOptions(OPTIONS)
        self.parameter_values = pybamm.ParameterValues("Chen2020_composite")
        self.param = pybamm.LithiumIonParameters(self.options)
        self.method = method
        model = pybamm.lithium_ion.ElectrodeSOHComposite(
            self.options, direction, initialization_method=method
        )
        self.names, self.input_names, self.function = esc._esoh_evaluator(
            model, self.parameter_values
        )

        # mirror the temperature and branch choice in ElectrodeSOHComposite.__init__
        reference = self.parameter_values.evaluate(self.param.T_ref)
        temperature = (
            self.parameter_values.evaluate(self.param.T_init)
            if method == "voltage"
            else reference
        )

        def potential(side, phase, electrode):
            stoichiometry = pybamm.InputParameter("sto")
            branch = get_lithiation_delithiation(
                direction, electrode, self.options, phase=phase
            )
            processed = self.parameter_values.process_symbol(
                (side.prim if phase == "primary" else side.sec).U(
                    stoichiometry, temperature, branch
                )
            )
            return lambda value: float(
                np.asarray(processed.evaluate(inputs={"sto": value})).reshape(-1)[0]
            )

        self.U_n = potential(self.param.n, "primary", "negative")
        self.U_n2 = potential(self.param.n, "secondary", "negative")
        self.U_p = potential(self.param.p, "primary", "positive")

    def capacities(self, scales):
        evaluate = self.parameter_values.evaluate
        return {
            "Q_n_1": evaluate(self.param.n.prim.Q_init) * scales[0],
            "Q_n_2": evaluate(self.param.n.sec.Q_init) * scales[1],
            "Q_p_1": evaluate(self.param.p.prim.Q_init) * scales[2],
            "Q_Li": evaluate(self.param.Q_Li_particles_init) * scales[3],
        }

    def solve(self, capacities, target):
        key = "V_init" if self.method == "voltage" else "SOC_init"
        inputs = {**capacities, key: float(target)}
        values = np.asarray(
            self.function(*[inputs[name] for name in self.input_names])
        ).reshape(-1)
        return dict(zip(self.names, values, strict=True))

    def lithium(self, capacities, state):
        return (
            capacities["Q_n_1"] * state["x_init_1"]
            + capacities["Q_n_2"] * state["x_init_2"]
            + capacities["Q_p_1"] * state["y_init_1"]
        )

    def state_of_charge(self, capacities, state):
        def charge(tag):
            return (
                capacities["Q_n_1"] * state[f"x_{tag}_1"]
                + capacities["Q_n_2"] * state[f"x_{tag}_2"]
            )

        return (charge("init") - charge("0")) / (charge("100") - charge("0"))


class TestCompositeElectrodeSOHSweeps:
    """Dense sweeps checking physics rather than stored answers."""

    @pytest.mark.parametrize("direction", ["discharge", "charge", None])
    @pytest.mark.parametrize("wear", list(WEAR))
    def test_a_thousand_states_of_charge(self, direction, wear):
        case = Case(direction, "SOC")
        capacities = case.capacities(WEAR[wear])
        for target in np.linspace(0.0, 1.0, SWEEP):
            state = case.solve(capacities, target)
            assert all(np.isfinite(v) for v in state.values()), target
            assert case.lithium(capacities, state) == pytest.approx(
                capacities["Q_Li"], rel=1e-9
            ), target
            assert case.state_of_charge(capacities, state) == pytest.approx(
                target, abs=1e-7
            )
            assert case.U_n(state["x_init_1"]) == pytest.approx(
                case.U_n2(state["x_init_2"]), abs=1e-6
            ), target

    @pytest.mark.parametrize("direction", ["discharge", "charge", None])
    @pytest.mark.parametrize("wear", list(WEAR))
    def test_a_thousand_voltages(self, direction, wear):
        case = Case(direction, "voltage")
        capacities = case.capacities(WEAR[wear])
        for target in np.linspace(2.5, 4.2, SWEEP):
            state = case.solve(capacities, target)
            assert all(np.isfinite(v) for v in state.values()), target
            assert case.lithium(capacities, state) == pytest.approx(
                capacities["Q_Li"], rel=1e-9
            ), target
            voltage = case.U_p(state["y_init_1"]) - case.U_n(state["x_init_1"])
            assert voltage == pytest.approx(target, abs=1e-7)

    def test_repeating_a_solve_returns_identical_bits(self):
        # the rootfinder caches its last solve, which must not change an answer
        case = Case("discharge", "SOC")
        capacities = case.capacities(WEAR["nominal"])
        for target in np.linspace(0.0, 1.0, SWEEP):
            first = case.solve(capacities, target)
            assert case.solve(capacities, target) == first

    def test_the_answer_moves_smoothly_with_the_target(self):
        # a bracket that flipped to another root would show up as a jump
        case = Case("discharge", "SOC")
        capacities = case.capacities(WEAR["nominal"])
        targets = np.linspace(0.0, 1.0, SWEEP)
        x_init = np.array([case.solve(capacities, t)["x_init_1"] for t in targets])
        steps = np.diff(x_init)
        assert np.all(steps > 0), "x_init_1 must increase with state of charge"
        assert np.max(steps) < 50 * np.median(steps), "discontinuity in x_init_1"

    def test_a_state_of_charge_round_trips_through_its_voltage(self):
        by_soc = Case("discharge", "SOC")
        by_voltage = Case("discharge", "voltage")
        soc_capacities = by_soc.capacities(WEAR["nominal"])
        voltage_capacities = by_voltage.capacities(WEAR["nominal"])
        for target in np.linspace(0.05, 0.95, SWEEP):
            state = by_soc.solve(soc_capacities, target)
            voltage = by_voltage.U_p(state["y_init_1"]) - by_voltage.U_n(
                state["x_init_1"]
            )
            back = by_voltage.solve(voltage_capacities, voltage)
            assert back["x_init_1"] == pytest.approx(state["x_init_1"], abs=1e-6)

    @pytest.mark.parametrize(
        ("method", "targets"),
        [
            ("SOC", np.linspace(-5.0, 5.0, SWEEP)),
            ("voltage", np.linspace(0.5, 6.0, SWEEP)),
        ],
    )
    def test_a_target_outside_the_window_still_solves_exactly(self, method, targets):
        """A non-physical target has an exact answer, and must be given it."""
        case = Case("discharge", method)
        capacities = case.capacities(WEAR["nominal"])
        for target in targets:
            state = case.solve(capacities, target)
            assert all(np.isfinite(v) for v in state.values()), target
            assert case.lithium(capacities, state) == pytest.approx(
                capacities["Q_Li"], rel=1e-9
            ), target
            if method == "voltage":
                voltage = case.U_p(state["y_init_1"]) - case.U_n(state["x_init_1"])
                assert voltage == pytest.approx(target, abs=1e-6), target
            else:
                assert case.state_of_charge(capacities, state) == pytest.approx(
                    target, abs=1e-7
                ), target

    def test_a_non_physical_target_returns_a_non_physical_stoichiometry(self):
        case = Case("discharge", "SOC")
        capacities = case.capacities(WEAR["nominal"])
        assert case.solve(capacities, -3.0)["x_init_1"] < 0
        assert case.solve(capacities, 3.0)["x_init_1"] > 1


class TestCompositeElectrodeSOHReuse:
    """Ageing a cell must not rebuild the evaluator.

    The capacities reach the compiled function as inputs, so only the parameters its
    graph substitutes can invalidate it.
    """

    # both fall as a cell ages, and both feed the capacities the solve is given
    FADE = (
        "Initial concentration in positive electrode [mol.m-3]",
        "Primary: Negative electrode active material volume fraction",
    )

    @staticmethod
    def _reused_simulation(parameter_values):
        model = pybamm.lithium_ion.ElectrodeSOHComposite(
            pybamm.BatteryModelOptions(OPTIONS),
            "discharge",
            initialization_method="SOC",
        )
        return pybamm.Simulation(model, parameter_values=parameter_values)

    def _call(self, parameter_values, simulation):
        return pybamm.lithium_ion.get_initial_stoichiometries_composite(
            0.5,
            parameter_values,
            direction="discharge",
            options=OPTIONS,
            esoh_sim=simulation,
        )

    def test_fading_capacities_reuse_the_compiled_evaluator(self):
        parameter_values = pybamm.ParameterValues("Chen2020_composite")
        simulation = self._reused_simulation(parameter_values)
        self._call(parameter_values, simulation)
        built = simulation.model._evaluator[-1]

        nominal = {key: parameter_values[key] for key in self.FADE}
        for step in range(1, 11):
            for key in self.FADE:
                parameter_values[key] = nominal[key] * (1 - 0.002 * step)
            state = self._call(parameter_values, simulation)
            assert np.isfinite(state["x_init_1"])
            assert simulation.model._evaluator[-1] is built

    def test_a_new_open_circuit_potential_does_rebuild(self):
        parameter_values = pybamm.ParameterValues("Chen2020_composite")
        simulation = self._reused_simulation(parameter_values)
        self._call(parameter_values, simulation)
        built = simulation.model._evaluator[-1]

        parameter_values["Primary: Negative electrode OCP [V]"] = lambda sto: 1.5 - sto
        self._call(parameter_values, simulation)
        assert simulation.model._evaluator[-1] is not built
