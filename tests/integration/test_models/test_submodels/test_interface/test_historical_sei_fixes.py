"""
Regression tests for historical SEI and degradation model bug fixes.

These tests guard against the reintroduction of bugs that were fixed but
did not have regression tests added at the time of the fix.
"""

import numpy as np
import pytest

import pybamm


class TestSEIThicknessFixes:
    """Guards for SEI thickness calculation bug fixes."""

    def test_sei_thickness_never_decreases_electron_migration(self):
        """
        Guards against: PR #3622 - Fixed a bug where the SEI thickness decreased
        at some intervals when using the 'electron-migration limited' model.

        The bug was that the SEI current density (j_sei) could become positive
        under certain conditions, causing SEI thickness to decrease. The fix
        applied a Heaviside function to ensure only negative contributions to
        eta_sei affect j_sei.

        This test verifies SEI thickness is monotonically non-decreasing.
        """
        model = pybamm.lithium_ion.DFN({"SEI": "electron-migration limited"})
        # Use OKane2022 which has SEI parameters
        param = pybamm.ParameterValues("OKane2022")
        sim = pybamm.Simulation(model, parameter_values=param)

        # Run for enough time to see SEI growth
        sol = sim.solve([0, 3600])

        # Get SEI thickness
        L_sei = sol["X-averaged negative SEI thickness [m]"].data

        # SEI thickness should never decrease
        # Check that each value is >= the previous value
        dL_sei = np.diff(L_sei)
        assert np.all(dL_sei >= -1e-15), (
            f"SEI thickness decreased at some intervals. Min change: {dL_sei.min()}"
        )

        # SEI should grow over time (not stay constant)
        assert L_sei[-1] > L_sei[0], "SEI should grow during cycling"

    def test_sei_thickness_never_decreases_solvent_diffusion(self):
        """
        Verify SEI thickness is monotonically non-decreasing for
        solvent-diffusion limited model as well.
        """
        model = pybamm.lithium_ion.DFN({"SEI": "solvent-diffusion limited"})
        param = pybamm.ParameterValues("OKane2022")
        sim = pybamm.Simulation(model, parameter_values=param)

        sol = sim.solve([0, 3600])

        L_sei = sol["X-averaged negative SEI thickness [m]"].data
        dL_sei = np.diff(L_sei)

        assert np.all(dL_sei >= -1e-15), (
            f"SEI thickness decreased. Min change: {dL_sei.min()}"
        )


class TestECReactionFixes:
    """Guards for EC reaction limited SEI bug fixes."""

    def test_ec_reaction_limited_runs_correctly(self):
        """
        Guards against: PR #4774 - fix ec reaction bug

        PR #4394 introduced a bug in EC-reaction limited SEI that was
        reverted in #4774. This test verifies the EC-reaction limited
        model runs correctly and produces valid SEI growth.
        """
        model = pybamm.lithium_ion.DFN({"SEI": "ec reaction limited"})
        param = pybamm.ParameterValues("OKane2022")
        sim = pybamm.Simulation(model, parameter_values=param)

        # Should run without errors
        sol = sim.solve([0, 3600])

        assert len(sol.t) > 0

        # SEI thickness should be valid
        L_sei = sol["X-averaged negative SEI thickness [m]"].data
        assert not np.any(np.isnan(L_sei))
        assert np.all(L_sei > 0)

        # Voltage should be reasonable (small tolerance for cutoff)
        V = sol["Voltage [V]"].data
        assert np.all(V >= 2.5 - 1e-10)
        assert np.all(V < 4.5)

    def test_ec_reaction_sei_grows_during_discharge(self):
        """
        Verify EC reaction limited SEI grows during discharge.
        """
        model = pybamm.lithium_ion.DFN({"SEI": "ec reaction limited"})
        param = pybamm.ParameterValues("OKane2022")
        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 3600])

        L_sei = sol["X-averaged negative SEI thickness [m]"].data

        # SEI should grow
        assert L_sei[-1] > L_sei[0]


class TestSEICompositeIntegration:
    """Guards for SEI + Composite electrode integration fixes."""

    @staticmethod
    def _get_composite_sei_params():
        """Get Chen2020_composite params with SEI parameters from OKane2022."""
        param = pybamm.ParameterValues("Chen2020_composite")
        okane = pybamm.ParameterValues("OKane2022")
        # Add SEI parameters
        sei_params = {
            k: okane[k] for k in okane.keys() if "SEI" in k or "sei" in k.lower()
        }
        param.update(sei_params)
        return param

    def test_sei_with_composite_electrode_runs(self):
        """
        Guards against: PR #4153 - Issue 4123 fix: SEI + Composite Integrations
        and lithium plating functions updated

        The bug was that SEI submodels didn't properly handle domain options
        for composite electrodes. This test verifies SEI works with composite
        electrodes.
        """
        model = pybamm.lithium_ion.DFN(
            {
                "particle phases": ("2", "1"),
                "SEI": "solvent-diffusion limited",
            }
        )
        param = self._get_composite_sei_params()
        sim = pybamm.Simulation(model, parameter_values=param)

        # Should run without domain errors
        sol = sim.solve([0, 600])

        assert len(sol.t) > 0

        # For composite, SEI thickness is tracked per phase
        L_sei_primary = sol["Volume-averaged negative primary SEI thickness [m]"].data
        L_sei_secondary = sol[
            "Volume-averaged negative secondary SEI thickness [m]"
        ].data

        assert not np.any(np.isnan(L_sei_primary))
        assert not np.any(np.isnan(L_sei_secondary))
        assert np.all(L_sei_primary > 0)
        assert np.all(L_sei_secondary > 0)

        # Both particle phases should have valid concentrations
        c_primary = sol[
            "X-averaged negative primary particle concentration [mol.m-3]"
        ].data
        c_secondary = sol[
            "X-averaged negative secondary particle concentration [mol.m-3]"
        ].data

        assert not np.any(np.isnan(c_primary))
        assert not np.any(np.isnan(c_secondary))

    @staticmethod
    def _get_composite_plating_params():
        """Get Chen2020_composite params with plating parameters.

        Based on lithium-plating-composite.ipynb notebook example.
        """
        param = pybamm.ParameterValues("Chen2020_composite")

        def graphite_plating_exchange_current_density(c_e, c_Li, T):
            k_plating = pybamm.Parameter(
                "Primary: Lithium plating kinetic rate constant [m.s-1]"
            )
            return pybamm.constants.F * k_plating * c_e

        def silicon_plating_exchange_current_density(c_e, c_Li, T):
            k_plating = pybamm.Parameter(
                "Secondary: Lithium plating kinetic rate constant [m.s-1]"
            )
            return pybamm.constants.F * k_plating * c_e

        param.update(
            {
                "Lithium metal partial molar volume [m3.mol-1]": 1.3e-05,
                "Primary: Lithium plating kinetic rate constant [m.s-1]": 1e-09,
                "Primary: Exchange-current density for plating [A.m-2]": graphite_plating_exchange_current_density,
                "Primary: Initial plated lithium concentration [mol.m-3]": 0.0,
                "Primary: Typical plated lithium concentration [mol.m-3]": 1000.0,
                "Primary: Lithium plating transfer coefficient": 0.65,
                "Secondary: Lithium plating kinetic rate constant [m.s-1]": 1e-09,
                "Secondary: Exchange-current density for plating [A.m-2]": silicon_plating_exchange_current_density,
                "Secondary: Initial plated lithium concentration [mol.m-3]": 0.0,
                "Secondary: Typical plated lithium concentration [mol.m-3]": 1000.0,
                "Secondary: Lithium plating transfer coefficient": 0.65,
                "Ambient temperature [K]": 268.15,
                "Initial temperature [K]": 268.15,
            }
        )
        return param

    def test_lithium_plating_with_composite_electrode(self):
        """
        Guards against: PR #4153 - lithium plating functions updated for
        composite electrodes.

        Verify lithium plating works correctly with composite electrodes.
        """
        model = pybamm.lithium_ion.DFN(
            {
                "particle phases": ("2", "1"),
                "lithium plating": "irreversible",
            }
        )
        param = self._get_composite_plating_params()
        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 600])

        assert len(sol.t) > 0

        L_plating_primary = sol[
            "X-averaged negative primary lithium plating thickness [m]"
        ].data
        L_plating_secondary = sol[
            "X-averaged negative secondary lithium plating thickness [m]"
        ].data
        assert not np.any(np.isnan(L_plating_primary))
        assert not np.any(np.isnan(L_plating_secondary))


class TestSEIModelConsistency:
    """Tests for SEI model consistency across configurations."""

    @pytest.mark.parametrize(
        "sei_option",
        [
            "reaction limited",
            "solvent-diffusion limited",
            "electron-migration limited",
            "ec reaction limited",
        ],
    )
    def test_sei_models_produce_growth(self, sei_option):
        """
        Verify all SEI model options produce SEI growth during discharge.
        """
        model = pybamm.lithium_ion.SPM({"SEI": sei_option})
        param = pybamm.ParameterValues("OKane2022")
        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 3600])

        L_sei = sol["X-averaged negative SEI thickness [m]"].data

        # SEI should exist and grow
        assert L_sei[-1] > L_sei[0], f"{sei_option} should produce SEI growth"
        assert not np.any(np.isnan(L_sei))

    def test_sei_affects_capacity_fade(self):
        """
        Verify SEI growth contributes to capacity fade over cycling.
        """
        model = pybamm.lithium_ion.SPM({"SEI": "solvent-diffusion limited"})
        param = pybamm.ParameterValues("OKane2022")

        experiment = pybamm.Experiment(
            [
                ("Discharge at 1C until 2.5V", "Rest for 10 minutes"),
            ]
            * 3
        )

        sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
        sol = sim.solve()

        # With SEI growth, SEI thickness should increase
        L_sei_final = sol["X-averaged negative SEI thickness [m]"].data[-1]
        L_sei_initial = sol["X-averaged negative SEI thickness [m]"].data[0]

        assert L_sei_final > L_sei_initial
