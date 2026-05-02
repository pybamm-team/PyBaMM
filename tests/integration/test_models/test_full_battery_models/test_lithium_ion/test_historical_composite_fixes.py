"""
Regression tests for historical composite electrode bug fixes.

These tests guard against the reintroduction of bugs that were fixed but
did not have regression tests added at the time of the fix.
"""

import numpy as np
import pytest

import pybamm


class TestParticlePhasesOptionFixes:
    """Guards for particle phases option bug fixes."""

    def test_particle_phases_tuple_of_ones_works(self):
        """
        Guards against: PR #3534 - #3532 fix bug

        The bug was that specifying particle phases as ('1', '1') triggered
        the multi-phase validation even though it effectively means single
        phase for both electrodes. The check used `!= "1"` which failed for
        tuples.

        This test verifies that ('1', '1') is treated as single-phase.
        """
        # This should NOT raise an OptionError
        model = pybamm.lithium_ion.DFN(
            {
                "particle phases": ("1", "1"),
            }
        )
        param = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 600])

        # Should run successfully
        assert len(sol.t) > 0
        V = sol["Voltage [V]"].data
        assert np.all(V > 2.5)
        assert np.all(V < 4.5)

    def test_particle_phases_string_one_works(self):
        """
        Verify single string "1" for particle phases works.
        """
        model = pybamm.lithium_ion.SPM({"particle phases": "1"})
        param = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 600])

        assert len(sol.t) > 0


class TestParticleSizeDistributionFixes:
    """Guards for particle size distribution with multiple phases fixes."""

    @staticmethod
    def _get_psd_composite_params():
        """Get Chen2020_composite with PSD parameters for both phases.

        Uses different bounds for primary vs secondary to verify they're
        independently set in the geometry.
        """
        param = pybamm.ParameterValues("Chen2020_composite")

        # Get particle radii from composite parameter set
        R_primary = param["Primary: Negative particle radius [m]"]
        R_secondary = param["Secondary: Negative particle radius [m]"]
        R_positive = param["Positive particle radius [m]"]

        # Use DIFFERENT max bounds for primary vs secondary to verify
        # the geometry correctly uses independent bounds
        param.update(
            {
                # Primary phase (graphite) - use 3x typical radius
                "Primary: Negative minimum particle radius [m]": 0.0,
                "Primary: Negative maximum particle radius [m]": 3 * R_primary,
                "Primary: Negative area-weighted particle-size distribution [m-1]": lambda R: (
                    pybamm.lognormal(R, R_primary, 0.3 * R_primary)
                ),
                # Secondary phase (silicon) - use 2x typical radius (DIFFERENT)
                "Secondary: Negative minimum particle radius [m]": 0.0,
                "Secondary: Negative maximum particle radius [m]": 2 * R_secondary,
                "Secondary: Negative area-weighted particle-size distribution [m-1]": lambda R: (
                    pybamm.lognormal(R, R_secondary, 0.3 * R_secondary)
                ),
                # Positive electrode
                "Positive minimum particle radius [m]": 0.0,
                "Positive maximum particle radius [m]": 3 * R_positive,
                "Positive area-weighted particle-size distribution [m-1]": lambda R: (
                    pybamm.lognormal(R, R_positive, 0.3 * R_positive)
                ),
            }
        )
        return param

    def test_psd_secondary_phase_bounds_correct(self):
        """
        Guards against: PR #5415 - Fix psd 2 phases

        The bug was that when >2 phases were selected, the secondary particle
        size distribution bounds were not implemented - it was incorrectly
        using the primary PSD bounds for the secondary phase.

        This test verifies that secondary phase uses its own PSD bounds by:
        1. Setting different max bounds for primary vs secondary phases
        2. Checking the geometry has distinct bounds for each phase
        3. Running the model successfully
        """
        options = pybamm.BatteryModelOptions(
            {
                "particle phases": ("2", "1"),
                "particle size": "distribution",
            }
        )
        param = self._get_psd_composite_params()

        # Verify geometry has different bounds for primary vs secondary
        geometry = pybamm.battery_geometry(options=options)
        param.process_geometry(geometry)

        # Get the bounds from geometry after parameter processing
        R_max_primary_sym = geometry["negative primary particle size"]["R_n_prim"][
            "max"
        ]
        R_max_secondary_sym = geometry["negative secondary particle size"]["R_n_sec"][
            "max"
        ]

        # Evaluate the symbolic values
        R_max_primary = float(R_max_primary_sym.evaluate())
        R_max_secondary = float(R_max_secondary_sym.evaluate())

        # The bounds should be different (primary=3x, secondary=2x typical)
        assert R_max_primary != pytest.approx(R_max_secondary, rel=0.01), (
            f"Primary ({R_max_primary}) and secondary ({R_max_secondary}) "
            "bounds should differ - the bug aliased them incorrectly"
        )

        # Also verify the model runs correctly
        model = pybamm.lithium_ion.DFN(
            {
                "particle phases": ("2", "1"),
                "particle size": "distribution",
            }
        )
        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 600])

        assert len(sol.t) > 0
        V = sol["Voltage [V]"].data
        assert np.all(V > 2.5)
        assert np.all(V < 4.5)


class TestCompositeSwellingFixes:
    """Guards for composite electrode swelling/mechanics fixes."""

    @staticmethod
    def _get_swelling_composite_params():
        """Get Chen2020_composite with swelling parameters from OKane2022."""
        param = pybamm.ParameterValues("Chen2020_composite")
        okane = pybamm.ParameterValues("OKane2022")

        # Add mechanics params for both phases
        param.update(
            {
                # Partial molar volume
                "Primary: Negative electrode partial molar volume [m3.mol-1]": okane[
                    "Negative electrode partial molar volume [m3.mol-1]"
                ],
                "Secondary: Negative electrode partial molar volume [m3.mol-1]": okane[
                    "Negative electrode partial molar volume [m3.mol-1]"
                ],
                "Positive electrode partial molar volume [m3.mol-1]": okane[
                    "Positive electrode partial molar volume [m3.mol-1]"
                ],
                # Young's modulus
                "Primary: Negative electrode Young's modulus [Pa]": okane[
                    "Negative electrode Young's modulus [Pa]"
                ],
                "Secondary: Negative electrode Young's modulus [Pa]": okane[
                    "Negative electrode Young's modulus [Pa]"
                ],
                "Positive electrode Young's modulus [Pa]": okane[
                    "Positive electrode Young's modulus [Pa]"
                ],
                # Poisson's ratio
                "Primary: Negative electrode Poisson's ratio": okane[
                    "Negative electrode Poisson's ratio"
                ],
                "Secondary: Negative electrode Poisson's ratio": okane[
                    "Negative electrode Poisson's ratio"
                ],
                "Positive electrode Poisson's ratio": okane[
                    "Positive electrode Poisson's ratio"
                ],
                # Reference concentration for free of deformation
                "Primary: Negative electrode reference concentration for free of deformation [mol.m-3]": okane[
                    "Negative electrode reference concentration for free of deformation [mol.m-3]"
                ],
                "Secondary: Negative electrode reference concentration for free of deformation [mol.m-3]": okane[
                    "Negative electrode reference concentration for free of deformation [mol.m-3]"
                ],
                "Positive electrode reference concentration for free of deformation [mol.m-3]": okane[
                    "Positive electrode reference concentration for free of deformation [mol.m-3]"
                ],
                # Volume change functions
                "Primary: Negative electrode volume change": okane[
                    "Negative electrode volume change"
                ],
                "Secondary: Negative electrode volume change": okane[
                    "Negative electrode volume change"
                ],
                "Positive electrode volume change": okane[
                    "Positive electrode volume change"
                ],
            }
        )
        return param

    def test_cell_thickness_change_with_composite(self):
        """
        Guards against: PR #5272 - Composite electrode bug-fix for swelling
        submodel

        The bug was a mismatched key inspection when creating the
        'Cell thickness change [m]' variable with composite electrodes.

        This test verifies cell thickness change is correctly computed.
        """
        model = pybamm.lithium_ion.DFN(
            {
                "particle phases": ("2", "1"),
                "particle mechanics": "swelling only",
            }
        )
        param = self._get_swelling_composite_params()
        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 1800])

        # Cell thickness change should exist and be valid
        dL = sol["Cell thickness change [m]"].data
        assert not np.any(np.isnan(dL))

        # During discharge, cell typically expands slightly
        # Just verify it's a reasonable value (not huge or negative)
        assert np.all(np.abs(dL) < 1e-3)  # Less than 1mm change

    def test_composite_electrode_mechanics_variables(self):
        """
        Verify mechanics variables exist for both phases in composite.
        """
        model = pybamm.lithium_ion.DFN(
            {
                "particle phases": ("2", "1"),
                "particle mechanics": "swelling only",
            }
        )
        param = self._get_swelling_composite_params()
        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 600])

        # Both phases should have strain variables
        strain_primary = sol[
            "X-averaged negative primary particle surface tangential stress [Pa]"
        ].data
        strain_secondary = sol[
            "X-averaged negative secondary particle surface tangential stress [Pa]"
        ].data

        assert not np.any(np.isnan(strain_primary))
        assert not np.any(np.isnan(strain_secondary))


class TestCompositeThermalFixes:
    """Guards for composite electrode thermal model fixes."""

    @staticmethod
    def _get_thermal_composite_params():
        """Get Chen2020_composite with thermal parameters."""
        param = pybamm.ParameterValues("Chen2020_composite")
        chen = pybamm.ParameterValues("Chen2020")
        marquis = pybamm.ParameterValues("Marquis2019")

        # Add non-prefixed thermal params needed for lumped thermal model
        param.update(
            {
                "Negative electrode density [kg.m-3]": chen[
                    "Negative electrode density [kg.m-3]"
                ],
                "Negative electrode specific heat capacity [J.kg-1.K-1]": chen[
                    "Negative electrode specific heat capacity [J.kg-1.K-1]"
                ],
                "Negative electrode thermal conductivity [W.m-1.K-1]": chen[
                    "Negative electrode thermal conductivity [W.m-1.K-1]"
                ],
                # Add entropic change for both phases to enable reversible heating
                "Primary: Negative electrode OCP entropic change [V.K-1]": marquis[
                    "Negative electrode OCP entropic change [V.K-1]"
                ],
                "Secondary: Negative electrode OCP entropic change [V.K-1]": marquis[
                    "Negative electrode OCP entropic change [V.K-1]"
                ],
                "Positive electrode OCP entropic change [V.K-1]": marquis[
                    "Positive electrode OCP entropic change [V.K-1]"
                ],
            }
        )
        return param

    def test_thermal_with_composite_two_phases(self):
        """
        Guards against: PR #3586 - Fix Issue 3543 - Thermal model not working
        for a composite electrode particle model with two (or more) phases

        The bug was that irreversible and reversible heating terms weren't
        being summed over the phases in BaseThermal. The fix stores dUdT
        for each phase and sums heating contributions.

        This test verifies thermal model works with composite electrodes.
        """
        model = pybamm.lithium_ion.DFN(
            {
                "particle phases": ("2", "1"),
                "thermal": "lumped",
            }
        )
        param = self._get_thermal_composite_params()
        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 1800])

        # Temperature should be valid and increase during discharge
        T = sol["Volume-averaged cell temperature [K]"].data
        assert not np.any(np.isnan(T))
        assert T[-1] > T[0]  # Should heat up during discharge

        # Heating terms should be valid
        Q_irrev = sol[
            "Volume-averaged irreversible electrochemical heating [W.m-3]"
        ].data
        assert not np.any(np.isnan(Q_irrev))

    def test_reversible_heating_with_composite(self):
        """
        Verify reversible heating is computed for composite electrodes.
        """
        model = pybamm.lithium_ion.DFN(
            {
                "particle phases": ("2", "1"),
                "thermal": "lumped",
            }
        )
        param = self._get_thermal_composite_params()
        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 1800])

        Q_rev = sol["Volume-averaged reversible heating [W.m-3]"].data
        assert not np.any(np.isnan(Q_rev))

        # Reversible heating oscillates (can be positive or negative)
        # Just verify it's not all zeros
        assert not np.allclose(Q_rev, 0, atol=1e-10)


class TestCompositeSurfaceFormFixes:
    """Guards for composite surface form conductivity fixes."""

    def test_composite_surface_form_conductivity(self):
        """
        Guards against: 7661ed966 - fix bug in composite surface form model
        (#4293)

        The bug was in the composite surface form conductivity calculation.
        This test verifies the model runs correctly and produces reasonable
        results with composite electrodes and surface form.
        """
        model = pybamm.lithium_ion.DFN(
            {
                "particle phases": ("2", "1"),
                "surface form": "differential",
            }
        )
        param = pybamm.ParameterValues("Chen2020_composite")
        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 600])

        # Model should run successfully without errors
        assert len(sol.t) > 0

        # Voltage should be reasonable
        V = sol["Voltage [V]"].data
        assert np.all(V > 2.5)
        assert np.all(V < 4.5)

        # Surface form potentials should exist
        phi_s = sol["Negative electrode potential [V]"].data
        assert not np.any(np.isnan(phi_s))
