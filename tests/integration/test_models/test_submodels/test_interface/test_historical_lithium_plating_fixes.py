"""
Regression tests for historical lithium plating bug fixes.

These tests guard against the reintroduction of bugs that were fixed but
did not have regression tests added at the time of the fix.
"""

import numpy as np
import pytest

import pybamm


class TestLithiumPlatingSPMDomainFixes:
    """Guards for lithium plating domain mismatch bug fixes."""

    @staticmethod
    def _get_plating_params():
        """Get Chen2020 parameters with lithium plating enabled."""
        param = pybamm.ParameterValues("Chen2020")

        def plating_exchange_current_density(c_e, c_Li, T):
            k_plating = pybamm.Parameter(
                "Lithium plating kinetic rate constant [m.s-1]"
            )
            return pybamm.constants.F * k_plating * c_e

        param.update(
            {
                "Lithium metal partial molar volume [m3.mol-1]": 1.3e-05,
                "Lithium plating kinetic rate constant [m.s-1]": 1e-09,
                "Exchange-current density for plating [A.m-2]": plating_exchange_current_density,
                "Initial plated lithium concentration [mol.m-3]": 0.0,
                "Typical plated lithium concentration [mol.m-3]": 1000.0,
                "Lithium plating transfer coefficient": 0.65,
                # Low temperature promotes plating
                "Ambient temperature [K]": 268.15,
                "Initial temperature [K]": 268.15,
            }
        )
        return param

    def test_lithium_plating_with_spm_builds_and_solves(self):
        """
        Guards against: PR #4844 - Fixed domain mismatch bug when using
        lithium plating with SPM

        The bug was that the x_average operation alone wasn't sufficient for
        SPM (which doesn't resolve the electrode domain spatially). The
        plating variables needed yz_average after x_average to produce
        scalar quantities that work with SPM's domain structure.

        The fix changed:
        - c_plated_Li_av = pybamm.x_average(c_plated_Li)
        + c_plated_Li_xav = pybamm.x_average(c_plated_Li)
        + c_plated_Li_av = pybamm.yz_average(c_plated_Li_xav)

        This test verifies SPM with lithium plating builds and solves
        without domain mismatch errors.
        """
        model = pybamm.lithium_ion.SPM({"lithium plating": "irreversible"})
        param = self._get_plating_params()

        sim = pybamm.Simulation(model, parameter_values=param)

        # This should not raise DomainError
        sol = sim.solve([0, 600])

        assert len(sol.t) > 0

        # Voltage should be reasonable
        V = sol["Voltage [V]"].data
        assert np.all(V > 2.5)
        assert np.all(V < 4.5)

    def test_spm_plating_variables_are_scalar(self):
        """
        Verify that averaged plating variables are scalar-valued for SPM.

        The bug caused x_averaged variables to have incorrect domain, leading
        to shape mismatches. This test verifies the averaged variables are
        truly scalar (0-dimensional in space).
        """
        model = pybamm.lithium_ion.SPM({"lithium plating": "irreversible"})
        param = self._get_plating_params()

        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 600])

        # These "volume-averaged" variables should be scalar (no spatial dimensions)
        # When called at a single time, should return a single number
        t_mid = sol.t[len(sol.t) // 2]

        L_plating = sol["X-averaged negative lithium plating thickness [m]"](t_mid)
        c_plating = sol["X-averaged negative lithium plating concentration [mol.m-3]"](
            t_mid
        )

        # Should be scalar values, not arrays with spatial dimensions
        assert np.isscalar(L_plating) or L_plating.ndim == 0 or L_plating.size == 1, (
            f"Plating thickness should be scalar, got shape {np.shape(L_plating)}"
        )
        assert np.isscalar(c_plating) or c_plating.ndim == 0 or c_plating.size == 1, (
            f"Plating concentration should be scalar, got shape {np.shape(c_plating)}"
        )

    def test_spm_plating_increases_during_charge_at_low_temp(self):
        """
        Verify lithium plating increases during charging at low temperature.
        """
        model = pybamm.lithium_ion.SPM({"lithium plating": "irreversible"})
        param = self._get_plating_params()

        # Use negative current (charging)
        param.update({"Current function [A]": -5.0})

        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 600], initial_soc=0.2)

        # Plating thickness should increase during charging at low temp
        L_plating = sol["X-averaged negative lithium plating thickness [m]"].data

        # Initial plating is 0, should increase during charge
        assert L_plating[-1] >= L_plating[0], (
            "Lithium plating should increase during charging at low temperature"
        )

    def test_dfn_plating_also_works(self):
        """
        Verify DFN with lithium plating still works (sanity check).
        """
        model = pybamm.lithium_ion.DFN({"lithium plating": "irreversible"})
        param = self._get_plating_params()

        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 600])

        assert len(sol.t) > 0

        # DFN should also have valid plating variables
        L_plating = sol["X-averaged negative lithium plating thickness [m]"].data
        assert not np.any(np.isnan(L_plating))

    def test_spme_plating_works(self):
        """
        Verify SPMe with lithium plating works.
        """
        model = pybamm.lithium_ion.SPMe({"lithium plating": "irreversible"})
        param = self._get_plating_params()

        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 600])

        assert len(sol.t) > 0

        V = sol["Voltage [V]"].data
        assert np.all(V > 2.5)
        assert np.all(V < 4.5)


class TestLithiumPlatingReversibility:
    """Tests for lithium plating reversibility options."""

    @staticmethod
    def _get_plating_params(needs_stripping=False):
        """Get Chen2020 parameters with lithium plating enabled.

        Parameters
        ----------
        needs_stripping : bool
            If True, include stripping parameters for reversible options.
        """
        param = pybamm.ParameterValues("Chen2020")

        def plating_exchange_current_density(c_e, c_Li, T):
            k_plating = pybamm.Parameter(
                "Lithium plating kinetic rate constant [m.s-1]"
            )
            return pybamm.constants.F * k_plating * c_e

        param.update(
            {
                "Lithium metal partial molar volume [m3.mol-1]": 1.3e-05,
                "Lithium plating kinetic rate constant [m.s-1]": 1e-09,
                "Exchange-current density for plating [A.m-2]": plating_exchange_current_density,
                "Initial plated lithium concentration [mol.m-3]": 0.0,
                "Typical plated lithium concentration [mol.m-3]": 1000.0,
                "Lithium plating transfer coefficient": 0.65,
                "Ambient temperature [K]": 268.15,
                "Initial temperature [K]": 268.15,
            }
        )

        if needs_stripping:

            def stripping_exchange_current_density(c_e, c_Li, T):
                k_stripping = pybamm.Parameter(
                    "Lithium stripping kinetic rate constant [m.s-1]"
                )
                return pybamm.constants.F * k_stripping * c_Li

            def dead_lithium_decay_rate(L_sei):
                gamma_0 = pybamm.Parameter("Dead lithium decay constant [s-1]")
                return gamma_0

            param.update(
                {
                    "Lithium stripping kinetic rate constant [m.s-1]": 1e-09,
                    "Exchange-current density for stripping [A.m-2]": stripping_exchange_current_density,
                    "Dead lithium decay constant [s-1]": 1e-06,
                    "Dead lithium decay rate [s-1]": dead_lithium_decay_rate,
                }
            )

        return param

    @pytest.mark.parametrize(
        "plating_option,needs_stripping",
        [
            ("irreversible", False),
            ("partially reversible", True),
            ("reversible", True),
        ],
    )
    def test_plating_options_build_with_spm(self, plating_option, needs_stripping):
        """
        Verify all lithium plating options work with SPM.

        This tests the domain fix from PR #4844 for all plating options.
        """
        model = pybamm.lithium_ion.SPM({"lithium plating": plating_option})
        param = self._get_plating_params(needs_stripping=needs_stripping)

        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 300])

        assert len(sol.t) > 0
        V = sol["Voltage [V]"].data
        assert not np.any(np.isnan(V))
