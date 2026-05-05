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
        phase. The check used `!= "1"` which failed for tuples.
        """
        model = pybamm.lithium_ion.DFN({"particle phases": ("1", "1")})
        param = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 600])

        assert len(sol.t) > 0
        V = sol["Voltage [V]"].data
        assert np.all(V > 2.5)
        assert np.all(V < 4.5)


class TestParticleSizeDistributionFixes:
    """Guards for particle size distribution with multiple phases fixes."""

    @staticmethod
    def _get_psd_composite_params():
        """Get Chen2020_composite with PSD parameters for both phases."""
        param = pybamm.ParameterValues("Chen2020_composite")

        R_primary = param["Primary: Negative particle radius [m]"]
        R_secondary = param["Secondary: Negative particle radius [m]"]
        R_positive = param["Positive particle radius [m]"]

        param.update(
            {
                "Primary: Negative minimum particle radius [m]": 0.0,
                "Primary: Negative maximum particle radius [m]": 3 * R_primary,
                "Primary: Negative area-weighted particle-size distribution [m-1]": lambda R: (
                    pybamm.lognormal(R, R_primary, 0.3 * R_primary)
                ),
                "Secondary: Negative minimum particle radius [m]": 0.0,
                "Secondary: Negative maximum particle radius [m]": 2 * R_secondary,
                "Secondary: Negative area-weighted particle-size distribution [m-1]": lambda R: (
                    pybamm.lognormal(R, R_secondary, 0.3 * R_secondary)
                ),
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

        The bug was that secondary particle size distribution bounds were
        incorrectly using the primary PSD bounds.
        """
        options = pybamm.BatteryModelOptions(
            {"particle phases": ("2", "1"), "particle size": "distribution"}
        )
        param = self._get_psd_composite_params()

        geometry = pybamm.battery_geometry(options=options)
        param.process_geometry(geometry)

        R_max_primary_sym = geometry["negative primary particle size"]["R_n_prim"][
            "max"
        ]
        R_max_secondary_sym = geometry["negative secondary particle size"]["R_n_sec"][
            "max"
        ]

        R_max_primary = float(R_max_primary_sym.evaluate())
        R_max_secondary = float(R_max_secondary_sym.evaluate())

        assert R_max_primary != pytest.approx(R_max_secondary, rel=0.01)

        model = pybamm.lithium_ion.DFN(
            {"particle phases": ("2", "1"), "particle size": "distribution"}
        )
        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 600])

        assert len(sol.t) > 0


class TestCompositeSwellingFixes:
    """Guards for composite electrode swelling/mechanics fixes."""

    @staticmethod
    def _get_swelling_composite_params():
        """Get Chen2020_composite with swelling parameters from OKane2022."""
        param = pybamm.ParameterValues("Chen2020_composite")
        okane = pybamm.ParameterValues("OKane2022")

        param.update(
            {
                "Primary: Negative electrode partial molar volume [m3.mol-1]": okane[
                    "Negative electrode partial molar volume [m3.mol-1]"
                ],
                "Secondary: Negative electrode partial molar volume [m3.mol-1]": okane[
                    "Negative electrode partial molar volume [m3.mol-1]"
                ],
                "Positive electrode partial molar volume [m3.mol-1]": okane[
                    "Positive electrode partial molar volume [m3.mol-1]"
                ],
                "Primary: Negative electrode Young's modulus [Pa]": okane[
                    "Negative electrode Young's modulus [Pa]"
                ],
                "Secondary: Negative electrode Young's modulus [Pa]": okane[
                    "Negative electrode Young's modulus [Pa]"
                ],
                "Positive electrode Young's modulus [Pa]": okane[
                    "Positive electrode Young's modulus [Pa]"
                ],
                "Primary: Negative electrode Poisson's ratio": okane[
                    "Negative electrode Poisson's ratio"
                ],
                "Secondary: Negative electrode Poisson's ratio": okane[
                    "Negative electrode Poisson's ratio"
                ],
                "Positive electrode Poisson's ratio": okane[
                    "Positive electrode Poisson's ratio"
                ],
                "Primary: Negative electrode reference concentration for free of deformation [mol.m-3]": okane[
                    "Negative electrode reference concentration for free of deformation [mol.m-3]"
                ],
                "Secondary: Negative electrode reference concentration for free of deformation [mol.m-3]": okane[
                    "Negative electrode reference concentration for free of deformation [mol.m-3]"
                ],
                "Positive electrode reference concentration for free of deformation [mol.m-3]": okane[
                    "Positive electrode reference concentration for free of deformation [mol.m-3]"
                ],
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
        Also verifies stress variables exist for both phases.
        """
        model = pybamm.lithium_ion.DFN(
            {"particle phases": ("2", "1"), "particle mechanics": "swelling only"}
        )
        param = self._get_swelling_composite_params()
        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 1800])

        dL = sol["Cell thickness change [m]"].data
        assert not np.any(np.isnan(dL))
        assert np.all(np.abs(dL) < 1e-3)

        strain_primary = sol[
            "X-averaged negative primary particle surface tangential stress [Pa]"
        ].data
        strain_secondary = sol[
            "X-averaged negative secondary particle surface tangential stress [Pa]"
        ].data

        assert not np.any(np.isnan(strain_primary))
        assert not np.any(np.isnan(strain_secondary))


class TestCompositeSurfaceFormFixes:
    """Guards for composite surface form conductivity fixes."""

    @staticmethod
    def _symbol_names(symbol):
        names = {symbol.name}
        for child in getattr(symbol, "children", []):
            names.update(TestCompositeSurfaceFormFixes._symbol_names(child))
        return names

    def test_composite_surface_form_conductivity(self):
        """
        Guards against: 7661ed966 - fix bug in composite surface form model
        (#4293)

        The bug was in the composite surface form conductivity calculation.
        """
        model = pybamm.lithium_ion.DFN(
            {"particle phases": ("2", "1"), "surface form": "differential"}
        )
        negative_delta_phi = next(
            key
            for key in model.rhs
            if key.name == "Negative electrode surface potential difference [V]"
        )
        rhs_names = self._symbol_names(model.rhs[negative_delta_phi])
        assert (
            "Primary: Negative electrode active material volume fraction" in rhs_names
        )
        assert (
            "Secondary: Negative electrode active material volume fraction" in rhs_names
        )
        assert "Primary: Negative particle radius [m]" in rhs_names
        assert "Secondary: Negative particle radius [m]" in rhs_names

        param = pybamm.ParameterValues("Chen2020_composite")
        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 600])

        assert len(sol.t) > 0

        V = sol["Voltage [V]"].data
        assert np.all(V > 2.5)
        assert np.all(V < 4.5)

        phi_s = sol["Negative electrode potential [V]"].data
        assert not np.any(np.isnan(phi_s))
