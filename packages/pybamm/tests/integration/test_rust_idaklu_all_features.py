import numpy as np
import pytest

import pybamm

pytest.importorskip("casadi")


def _build_spm_with_input_current():
    """SPM without events; ``_chen_with_input_current`` makes the current an input."""
    model = pybamm.lithium_ion.SPM()
    model.events = []
    return model


def _chen_with_input_current():
    param = pybamm.ParameterValues("Chen2020")
    param["Current function [A]"] = pybamm.InputParameter("I")
    return param


T_EVAL = np.linspace(0, 100, 15)
INPUTS = {"I": 0.5}


class TestRustIDAKLUAllFeatures:
    def test_spm_inputs_plus_outputs_parity(self):
        output_vars = ["Discharge capacity [A.h]", "Current [A]"]

        model_casadi = _build_spm_with_input_current()
        model_casadi.convert_to_format = "casadi"
        sol_casadi = pybamm.Simulation(
            model_casadi,
            parameter_values=_chen_with_input_current(),
            solver=pybamm.IDAKLUSolver(output_variables=output_vars),
        ).solve(T_EVAL, inputs=INPUTS)

        model_rust = _build_spm_with_input_current()
        model_rust.convert_to_format = "rust"
        sol_rust = pybamm.Simulation(
            model_rust,
            parameter_values=_chen_with_input_current(),
            solver=pybamm.IDAKLUSolver(output_variables=output_vars),
        ).solve(T_EVAL, inputs=INPUTS)

        np.testing.assert_allclose(sol_rust.y, sol_casadi.y, rtol=1e-5, atol=1e-8)
        for var in output_vars:
            np.testing.assert_allclose(
                sol_rust[var].entries,
                sol_casadi[var].entries,
                rtol=1e-5,
                atol=1e-7,
            )

    def test_spm_inputs_plus_sensitivities_parity(self):
        model_casadi = _build_spm_with_input_current()
        model_casadi.convert_to_format = "casadi"
        sol_casadi = pybamm.Simulation(
            model_casadi,
            parameter_values=_chen_with_input_current(),
            solver=pybamm.IDAKLUSolver(),
        ).solve(T_EVAL, inputs=INPUTS, calculate_sensitivities=["I"])

        model_rust = _build_spm_with_input_current()
        model_rust.convert_to_format = "rust"
        sol_rust = pybamm.Simulation(
            model_rust,
            parameter_values=_chen_with_input_current(),
            solver=pybamm.IDAKLUSolver(),
        ).solve(T_EVAL, inputs=INPUTS, calculate_sensitivities=["I"])

        np.testing.assert_allclose(sol_rust.y, sol_casadi.y, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(
            sol_rust.sensitivities["I"],
            sol_casadi.sensitivities["I"],
            rtol=1e-4,
            atol=1e-7,
        )
