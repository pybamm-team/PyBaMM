"""Integration test: DFN solved via Rust-backed IDAKLU."""

import numpy as np
import pytest

import pybamm


class TestRustIDAKLUDFN:
    @pytest.fixture
    def dfn_model_no_events(self):
        model = pybamm.lithium_ion.DFN()
        model.events = []
        return model

    @pytest.fixture
    def parameter_values(self):
        return pybamm.ParameterValues("Chen2020")

    def test_dfn_uses_reg_power(self, dfn_model_no_events, parameter_values):
        """Guards that DFN still exercises the Rust `RegPower` converter."""
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
