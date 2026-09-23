import pytest

import pybamm


class TestMSMRExchangeCurrentDensityFixes:
    """Guards for MSMR exchange current density calculation bug fixes."""

    @staticmethod
    def _walk_symbols(symbol):
        yield symbol
        for child in getattr(symbol, "children", []):
            yield from TestMSMRExchangeCurrentDensityFixes._walk_symbols(child)

    @classmethod
    def _contains_name(cls, symbol, name):
        return any(name in node.name for node in cls._walk_symbols(symbol))

    @pytest.mark.parametrize("electrode", ["negative", "positive"])
    def test_msmr_exchange_current_density_formula_uses_exact_power(self, electrode):
        """
        Verify MSMR j0 calculation uses xj**wj (not reg_power) for occupancy fraction.

        The fix changed from reg_power(xj, wj) to xj**wj for the occupancy fraction term.
        reg_power is still used for the concentration term c_e/c_e_ref which is acceptable.
        """
        param = pybamm.LithiumIonParameters()
        phase_param = getattr(param, electrode[0]).prim
        j0_expr = phase_param.j0_j(
            pybamm.Variable("c_e"),
            pybamm.Variable("U"),
            pybamm.Variable("T"),
            0,
        )

        occupancy_powers = [
            node
            for node in self._walk_symbols(j0_expr)
            if isinstance(node, pybamm.Power)
            and self._contains_name(node.left, "host site occupancy fraction")
            and self._contains_name(node.right, "host site ideality factor")
        ]
        assert len(occupancy_powers) == 1

        reg_power_terms = [
            node
            for node in self._walk_symbols(j0_expr)
            if isinstance(node, pybamm.RegPower)
        ]
        assert len(reg_power_terms) == 1
        assert self._contains_name(reg_power_terms[0], "c_e")
        assert not self._contains_name(
            reg_power_terms[0], "host site occupancy fraction"
        )
