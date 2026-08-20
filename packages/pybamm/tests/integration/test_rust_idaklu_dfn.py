"""Integration test: DFN solved via Rust-backed IDAKLU.

DFN exercises the `RegPower` Rust converter (used in OCP / regularised
intercalation flux); without it the model build raises `TypeError` at
`_to_rust`. This test pins parity against the CasADi backend.
"""

import numpy as np
import pytest

import pybamm


class TestRustIDAKLUDFN:
    """Full DFN simulation via Rust IDAKLU backend; parity vs CasADi."""

    @pytest.fixture
    def dfn_model_no_events(self):
        model = pybamm.lithium_ion.DFN()
        model.events = []
        return model

    @pytest.fixture
    def parameter_values(self):
        return pybamm.ParameterValues("Chen2020")

    def test_dfn_uses_reg_power(self, dfn_model_no_events, parameter_values):
        """Sanity-check: DFN's processed equations contain RegPower.

        If a refactor removes RegPower from DFN, the Rust converter
        coverage for it is no longer load-bearing for this model — but
        this test fails so we notice and re-evaluate the integration.
        """
        param = parameter_values.copy()
        model = dfn_model_no_events.new_copy()
        param.process_model(model)

        def _has_reg_power(symbol):
            if isinstance(symbol, pybamm.expression_tree.functions.RegPower):
                return True
            return any(_has_reg_power(c) for c in symbol.children)

        equations = list(model.rhs.values()) + list(model.algebraic.values())
        assert any(_has_reg_power(eq) for eq in equations), (
            "DFN no longer contains RegPower; this gate test is stale."
        )

    def test_dfn_rust_vs_casadi_voltage(self, dfn_model_no_events, parameter_values):
        """Voltage trajectory matches CasADi within IDA tolerance."""
        t_eval = np.linspace(0, 100, 25)

        model_casadi = dfn_model_no_events.new_copy()
        model_casadi.convert_to_format = "casadi"
        sim_casadi = pybamm.Simulation(
            model_casadi,
            parameter_values=parameter_values,
            solver=pybamm.IDAKLUSolver(),
        )
        sol_casadi = sim_casadi.solve(t_eval)

        model_rust = dfn_model_no_events.new_copy()
        model_rust.convert_to_format = "rust"
        sim_rust = pybamm.Simulation(
            model_rust,
            parameter_values=parameter_values,
            solver=pybamm.IDAKLUSolver(),
        )
        sol_rust = sim_rust.solve(t_eval)

        np.testing.assert_allclose(
            sol_rust["Voltage [V]"](t_eval),
            sol_casadi["Voltage [V]"](t_eval),
            rtol=1e-4,
            atol=1e-5,
            err_msg="DFN voltage trajectory differs between Rust and CasADi",
        )
