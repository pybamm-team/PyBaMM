#
# IDAKLU forward sensitivities of full battery models against finite differences
#
from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import pytest

import pybamm

lithium_ion = pybamm.lithium_ion

# Tight enough that the finite-difference noise, about rtol / (step * dimensionless
# sensitivity), stays well below RELATIVE_TOLERANCE in every case
SOLVER_TOLERANCES = {"rtol": 1e-11, "atol": 1e-13}
# Allowed error relative to the largest finite difference of each output
RELATIVE_TOLERANCE = 1e-4
DEFAULT_STEP = 1e-4
# Below this dimensionless sensitivity, (p / max|y|) dy/dp, a finite difference is noise
SENSITIVITY_FLOOR = 1e-5
COARSE_MESH = {
    "x_n": 6,
    "x_s": 5,
    "x_p": 6,
    "r_n": 8,
    "r_p": 8,
    "r_n_prim": 6,
    "r_n_sec": 6,
    "R_n": 6,
    "R_p": 6,
}


@dataclasses.dataclass(frozen=True)
class SensitivityCase:
    """A simulation that runs to ``t_end``, with a sensitivity to every input.

    ``steps`` overrides ``DEFAULT_STEP``, relative to the input's value or absolute
    for an input whose value is zero.
    """

    model: Callable[[], pybamm.BaseModel]
    parameter_values: Callable[[], pybamm.ParameterValues]
    inputs: dict[str, float]
    outputs: tuple[str, ...] = ("Voltage [V]",)
    t_end: float = 1800.0
    steps: dict[str, float] = dataclasses.field(default_factory=dict)
    var_pts: dict[str, int] | None = dataclasses.field(
        default_factory=lambda: COARSE_MESH
    )

    def build(self) -> pybamm.BaseModel:
        parameter_values = self.parameter_values()
        parameter_values.update(
            {name: "[input]" for name in self.inputs}, check_already_exists=False
        )
        simulation = pybamm.Simulation(
            self.model(), parameter_values=parameter_values, var_pts=self.var_pts
        )
        simulation.build()
        return simulation.built_model

    def output_times(self) -> np.ndarray:
        # Shifted off the even grid so outputs miss interpolant knots
        t = np.linspace(0, self.t_end, 21)
        t[1:-1] += 0.137 * self.t_end / 21
        return t

    def solve(self, solver, model, inputs, **kwargs):
        solution = solver.solve(
            model,
            [0, self.t_end],
            inputs=inputs,
            t_interp=self.output_times(),
            **kwargs,
        )
        for sol in solution if isinstance(solution, list) else [solution]:
            # Event-terminated solves would move the end time with the input
            assert sol.termination == "final time"
        return solution


def _parameter_values(name, **updates):
    def parameter_values():
        parameter_values = pybamm.ParameterValues(name)
        parameter_values.update(updates, check_already_exists=False)
        return parameter_values

    return parameter_values


def _function_inputs():
    """Chen2020 with an input inside an interpolant and inside a function."""
    parameter_values = pybamm.ParameterValues("Chen2020")
    stoichiometry = np.linspace(0, 1, 41)
    ocp = parameter_values["Negative electrode OCP [V]"]
    potential = np.array(
        [ocp(pybamm.Scalar(x)).evaluate() for x in stoichiometry]
    ).ravel()

    def negative_ocp(sto):
        scaled = sto * pybamm.InputParameter("Stoichiometry scale")
        return pybamm.Interpolant(
            stoichiometry, potential, scaled, interpolator="cubic"
        )

    def electrolyte_conductivity(c_e, T):
        c = c_e / 1000
        return (
            pybamm.InputParameter("Conductivity scale")
            * c
            * pybamm.exp(-0.5 * (c - 1) ** 2)
        )

    parameter_values.update(
        {
            "Negative electrode OCP [V]": negative_ocp,
            "Electrolyte conductivity [S.m-1]": electrolyte_conductivity,
        }
    )
    return parameter_values


def _drive_cycle():
    """Chen2020 with a drive-cycle current scaled by an input."""
    parameter_values = pybamm.ParameterValues("Chen2020")
    time = np.linspace(0, 1800, 61)
    current = 5 + 2 * np.sin(time / 150)

    def current_function(t):
        cycle = pybamm.Interpolant(time, current, t, interpolator="linear")
        return pybamm.InputParameter("Current scale") * cycle

    parameter_values.update({"Current function [A]": current_function})
    return parameter_values


CHEN2020 = _parameter_values("Chen2020")
NEGATIVE_DIFFUSIVITY = "Negative particle diffusivity [m2.s-1]"
NEGATIVE_INITIAL_CONCENTRATION = "Initial concentration in negative electrode [mol.m-3]"
# Diffusivities and conductivities barely bend voltage, so their finite differences
# are noise-limited at small steps; a larger step costs little truncation.
WEAK_STEP = 1e-3

CASES = {
    # An integer current; c_max enters many places, the cut-off only an event and
    # the nominal capacity only the C-rate
    "SPM": SensitivityCase(
        lithium_ion.SPM,
        CHEN2020,
        {
            "Current function [A]": 5,
            NEGATIVE_DIFFUSIVITY: 3.3e-14,
            NEGATIVE_INITIAL_CONCENTRATION: 29866.0,
            "Maximum concentration in negative electrode [mol.m-3]": 33133.0,
            "Lower voltage cut-off [V]": 2.5,
            "Nominal cell capacity [A.h]": 5.0,
        },
        outputs=("Voltage [V]", "C-rate"),
    ),
    "SPMe": SensitivityCase(
        lithium_ion.SPMe,
        _parameter_values("Ecker2015"),
        {"Current function [A]": 0.15652, NEGATIVE_INITIAL_CONCENTRATION: 26120.05},
    ),
    "SPMe lumped thermal": SensitivityCase(
        lambda: lithium_ion.SPMe({"thermal": "lumped"}),
        CHEN2020,
        {
            "Current function [A]": 5.0,
            NEGATIVE_DIFFUSIVITY: 3.3e-14,
            "Ambient temperature [K]": 298.15,
            "Total heat transfer coefficient [W.m-2.K-1]": 10.0,
            "Initial temperature [K]": 298.15,
        },
        outputs=("Voltage [V]", "X-averaged cell temperature [K]"),
    ),
    "DFN": SensitivityCase(
        lithium_ion.DFN,
        CHEN2020,
        {
            "Current function [A]": 5.0,
            NEGATIVE_DIFFUSIVITY: 3.3e-14,
            "Electrolyte conductivity [S.m-1]": 1.0,
            "Positive electrode conductivity [S.m-1]": 0.18,
        },
        outputs=("Voltage [V]", "Electrolyte concentration [mol.m-3]"),
        steps={
            NEGATIVE_DIFFUSIVITY: WEAK_STEP,
            "Electrolyte conductivity [S.m-1]": WEAK_STEP,
            "Positive electrode conductivity [S.m-1]": WEAK_STEP,
        },
    ),
    "DFN lumped thermal": SensitivityCase(
        lambda: lithium_ion.DFN({"thermal": "lumped"}),
        CHEN2020,
        {
            "Current function [A]": 5.0,
            "Ambient temperature [K]": 298.15,
            "Total heat transfer coefficient [W.m-2.K-1]": 10.0,
        },
        outputs=("Voltage [V]", "X-averaged cell temperature [K]"),
    ),
    "DFN quadratic particle": SensitivityCase(
        lambda: lithium_ion.DFN({"particle": "quadratic profile"}),
        CHEN2020,
        {NEGATIVE_DIFFUSIVITY: 3.3e-14, NEGATIVE_INITIAL_CONCENTRATION: 29866.0},
        steps={NEGATIVE_DIFFUSIVITY: WEAK_STEP},
    ),
    "DFN differential surface form": SensitivityCase(
        lambda: lithium_ion.DFN({"surface form": "differential"}),
        CHEN2020,
        {
            NEGATIVE_DIFFUSIVITY: 3.3e-14,
            "Positive electrode conductivity [S.m-1]": 0.18,
        },
        steps={
            NEGATIVE_DIFFUSIVITY: WEAK_STEP,
            "Positive electrode conductivity [S.m-1]": WEAK_STEP,
        },
    ),
    "DFN contact resistance": SensitivityCase(
        lambda: lithium_ion.DFN({"contact resistance": "true"}),
        CHEN2020,
        {"Contact resistance [Ohm]": 0.0, "Current function [A]": 5.0},
    ),
    "SPMe composite": SensitivityCase(
        lambda: lithium_ion.SPMe(
            {"particle phases": ("2", "1"), "surface form": "algebraic"}
        ),
        _parameter_values("Chen2020_composite"),
        {
            "Primary: Negative particle diffusivity [m2.s-1]": 5.5e-14,
            "Secondary: Negative electrode active material volume fraction": 0.015,
            "Secondary: Initial concentration in negative electrode [mol.m-3]": (
                276610.0
            ),
        },
        steps={
            "Primary: Negative particle diffusivity [m2.s-1]": WEAK_STEP,
            "Secondary: Negative electrode active material volume fraction": WEAK_STEP,
            # Close to the secondary c_max, so the voltage curves sharply
            "Secondary: Initial concentration in negative electrode [mol.m-3]": 2e-5,
        },
    ),
    "SPM SEI": SensitivityCase(
        lambda: lithium_ion.SPM(
            {
                "SEI": "ec reaction limited",
                "SEI film resistance": "distributed",
                "total interfacial current density as a state": "true",
            }
        ),
        _parameter_values("OKane2022"),
        {
            "SEI kinetic rate constant [m.s-1]": 1e-12,
            "EC diffusivity [m2.s-1]": 2e-18,
            "Initial SEI thickness [m]": 5e-9,
        },
        outputs=("Voltage [V]", "Loss of lithium to negative SEI [mol]"),
        steps={
            "SEI kinetic rate constant [m.s-1]": WEAK_STEP,
            "EC diffusivity [m2.s-1]": WEAK_STEP,
            "Initial SEI thickness [m]": WEAK_STEP,
        },
    ),
    "SPM lithium plating": SensitivityCase(
        lambda: lithium_ion.SPM({"lithium plating": "irreversible"}),
        _parameter_values(
            "OKane2022",
            **{
                "Initial concentration in positive electrode [mol.m-3]": 50000.0,
                NEGATIVE_INITIAL_CONCENTRATION: 6000.0,
            },
        ),
        {
            "Current function [A]": -5.0,
            "Lithium plating kinetic rate constant [m.s-1]": 1e-9,
            "Initial plated lithium concentration [mol.m-3]": 0.0,
        },
        outputs=("Voltage [V]", "Loss of lithium to negative lithium plating [mol]"),
        t_end=1200.0,
        steps={
            # Plating switches on sharply with the overpotential
            "Current function [A]": 3e-5,
            "Lithium plating kinetic rate constant [m.s-1]": WEAK_STEP,
        },
    ),
    "MPM": SensitivityCase(
        lithium_ion.MPM,
        lambda: lithium_ion.MPM().default_parameter_values,
        {
            "Current function [A]": 0.68,
            NEGATIVE_INITIAL_CONCENTRATION: 19986.609595075,
            NEGATIVE_DIFFUSIVITY: 3.9e-14,
        },
        steps={NEGATIVE_DIFFUSIVITY: WEAK_STEP},
    ),
    "MSMR": SensitivityCase(
        lambda: lithium_ion.SPM(
            {
                "open-circuit potential": "MSMR",
                "particle": "MSMR",
                "intercalation kinetics": "MSMR",
                "number of MSMR reactions": ("6", "4"),
                "surface form": "algebraic",
            }
        ),
        _parameter_values("MSMR_Example"),
        {
            "Current function [A]": 5.0,
            "Negative electrode host site standard potential (0) [V]": 0.08843,
            "Positive particle diffusivity [m2.s-1]": 4e-15,
        },
        steps={"Positive particle diffusivity [m2.s-1]": WEAK_STEP},
    ),
    "lead-acid LOQS": SensitivityCase(
        pybamm.lead_acid.LOQS,
        _parameter_values("Sulzer2019"),
        {
            "Current function [A]": 1.0,
            "Initial concentration in electrolyte [mol.m-3]": 5650.0,
        },
        t_end=3600.0,
    ),
    # The initial SoC is kept off the OCV and resistance table knots
    "Thevenin": SensitivityCase(
        pybamm.equivalent_circuit.Thevenin,
        _parameter_values("ECM_Example"),
        {
            "Current function [A]": 100.0,
            "Cell capacity [A.h]": 100.0,
            "Initial SoC": 0.537,
            "Element-1 initial overpotential [V]": 0.0,
            "Cell thermal mass [J/K]": 1000.0,
        },
        outputs=("Voltage [V]", "SoC"),
        t_end=1200.0,
        steps={"Cell thermal mass [J/K]": WEAK_STEP},
        var_pts=None,
    ),
    "half-cell SPMe": SensitivityCase(
        lambda: lithium_ion.SPMe({"working electrode": "positive"}),
        _parameter_values("Xu2019"),
        {
            "Current function [A]": 0.0024,
            "Positive particle diffusivity [m2.s-1]": 1e-14,
            "Initial concentration in positive electrode [mol.m-3]": 4631.0,
        },
    ),
    "SPM voltage control": SensitivityCase(
        lambda: lithium_ion.SPM({"operating mode": "voltage"}),
        CHEN2020,
        {"Voltage function [V]": 4.0, NEGATIVE_DIFFUSIVITY: 3.3e-14},
        outputs=("Current [A]",),
        # The current is very nonlinear in the held voltage
        steps={"Voltage function [V]": 1e-5, NEGATIVE_DIFFUSIVITY: WEAK_STEP},
    ),
    "SPMe power control": SensitivityCase(
        lambda: lithium_ion.SPMe({"operating mode": "power"}),
        CHEN2020,
        {
            "Power function [W]": 15.0,
            NEGATIVE_DIFFUSIVITY: 3.3e-14,
            "Electrolyte conductivity [S.m-1]": 1.0,
        },
        outputs=("Voltage [V]", "Current [A]"),
        steps={
            NEGATIVE_DIFFUSIVITY: WEAK_STEP,
            "Electrolyte conductivity [S.m-1]": WEAK_STEP,
        },
    ),
    "SPM CCCV": SensitivityCase(
        lambda: lithium_ion.SPM({"operating mode": "CCCV"}),
        _parameter_values(
            "Chen2020",
            **{
                "Initial concentration in positive electrode [mol.m-3]": 30000.0,
                NEGATIVE_INITIAL_CONCENTRATION: 20000.0,
            },
        ),
        {"CCCV current function [A]": -5.0, "Voltage function [V]": 4.1},
        outputs=("Voltage [V]", "Current [A]"),
        t_end=2400.0,
        steps={"Voltage function [V]": 1e-5},
    ),
    "SPMe function inputs": SensitivityCase(
        lithium_ion.SPMe,
        _function_inputs,
        {"Stoichiometry scale": 1.0, "Conductivity scale": 1.0},
    ),
    "SPM drive cycle": SensitivityCase(
        lithium_ion.SPM,
        _drive_cycle,
        {"Current scale": 1.0},
    ),
}


def _solver(**kwargs):
    return pybamm.IDAKLUSolver(**SOLVER_TOLERANCES, **kwargs)


def _values(solution, output, t):
    """``output`` at times ``t``, at its mesh nodes if it varies in space."""
    variable = solution[output]
    mesh = variable.mesh
    if mesh is None or len(mesh.nodes) == 1:
        return np.asarray(variable(t))
    return np.asarray(variable(t=t, x=mesh.nodes))


def _sensitivity(solution, output, name):
    """d(output)/d(name), shaped like ``_values``."""
    sensitivity = np.asarray(solution[output].sensitivities[name])
    # Stored time-major, one row per time and node
    sensitivity = sensitivity.reshape(len(solution.t), -1).T
    return sensitivity[0] if len(sensitivity) == 1 else sensitivity


def _central_differences(case, solver, model, name, step):
    value = case.inputs[name]
    h = step * abs(value) if value else step
    plus, minus = (
        case.solve(solver, model, {**case.inputs, name: value + sign * h})
        for sign in (1, -1)
    )
    t = case.output_times()
    return {
        output: (_values(plus, output, t) - _values(minus, output, t)) / (2 * h)
        for output in case.outputs
    }


def _perturbed(inputs):
    # Downwards, as the composite case starts close to its secondary c_max
    return {name: 0.99 * value if value else 1e-3 for name, value in inputs.items()}


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
class TestIDAKLUSensitivities:
    def test_sensitivities_match_central_differences(self, case):
        model = case.build()
        solution = case.solve(
            _solver(), model, case.inputs, calculate_sensitivities=True
        )
        t = case.output_times()
        solver = _solver()
        for name, value in case.inputs.items():
            step = case.steps.get(name, DEFAULT_STEP)
            fine = _central_differences(case, solver, model, name, step)
            coarse = _central_differences(case, solver, model, name, 2 * step)
            for output in case.outputs:
                label = f"d[{output}]/d[{name}]"
                sensitivity = _sensitivity(solution, output, name)
                if not fine[output].any():
                    # An input that leaves the solution unchanged has no gradient
                    np.testing.assert_array_equal(sensitivity, 0, err_msg=label)
                    continue
                floor = (
                    SENSITIVITY_FLOOR
                    * np.abs(_values(solution, output, t)).max()
                    / (abs(value) or 1.0)
                )
                scale = max(np.abs(fine[output]).max(), floor)
                convergence = np.abs(fine[output] - coarse[output]).max() / scale
                assert convergence <= RELATIVE_TOLERANCE, (
                    f"{label}: finite differences not converged ({convergence:.1e})"
                )
                np.testing.assert_allclose(
                    sensitivity,
                    fine[output],
                    rtol=0,
                    atol=RELATIVE_TOLERANCE * scale,
                    err_msg=label,
                )

    def test_output_variables_match_full_state(self, case):
        model = case.build()
        full = case.solve(_solver(), model, case.inputs, calculate_sensitivities=True)
        outputs_only = case.solve(
            _solver(output_variables=list(case.outputs)),
            model,
            case.inputs,
            calculate_sensitivities=True,
        )
        for output in case.outputs:
            for name in case.inputs:
                expected = _sensitivity(full, output, name)
                np.testing.assert_allclose(
                    _sensitivity(outputs_only, output, name),
                    expected,
                    rtol=1e-12,
                    atol=1e-12 * np.abs(expected).max(),
                    err_msg=f"d[{output}]/d[{name}]",
                )

    def test_input_sets_match_single_solves(self, case):
        model = case.build()
        input_sets = [case.inputs, _perturbed(case.inputs)]
        solutions = case.solve(
            _solver(options={"num_threads": 2}),
            model,
            input_sets,
            calculate_sensitivities=True,
        )
        solver = _solver()
        for inputs, solution in zip(input_sets, solutions, strict=True):
            single = case.solve(solver, model, inputs, calculate_sensitivities=True)
            for output in case.outputs:
                for name in case.inputs:
                    expected = _sensitivity(single, output, name)
                    np.testing.assert_allclose(
                        _sensitivity(solution, output, name),
                        expected,
                        rtol=1e-12,
                        atol=1e-12 * np.abs(expected).max(),
                        err_msg=f"d[{output}]/d[{name}] at {inputs}",
                    )
