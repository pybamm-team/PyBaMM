"""
Regression guard for issue #5414.

In half-cell mode (planar negative electrode) with `surface form` set to
either ``"differential"`` or ``"algebraic"`` and any non-isothermal thermal
option enabled (e.g. ``"thermal": "lumped"``), :class:`pybamm.lithium_ion.DFN`
used to fail to build with::

    pybamm.expression_tree.exceptions.ShapeError: Cannot find shape
    (original error: matmul: dimension mismatch with signature
    (n,k=21),(k=19,m)->(n,m))

Two separate boundary-condition gaps caused this:

1. ``BaseSurfaceFormConductivity.set_boundary_conditions`` (the parent of
   ``FullAlgebraic`` and ``FullDifferential``) only registered the
   inter-domain ``phi_e`` BCs from inside the negative-domain branch.  In
   the half-cell case the negative submodel is skipped, so ``phi_e`` had
   no BCs and ``Discretisation.set_internal_boundary_conditions`` could
   not split them across the separator/positive orphans -- ``grad(phi_e_s)``
   then discretised to the internal edges only (size ``n_s - 1``) instead
   of the full mesh edges (size ``n_s + 1``) expected by the Ohmic
   heating term in :class:`pybamm.thermal.base_thermal.BaseThermal`.

2. ``base_ohm.BaseModel.set_boundary_conditions`` bailed out as soon as
   the negative electrode was planar, silently dropping the BCs of the
   positive electrode potential ``phi_s_p`` in half-cell models.  The
   ``surface_form_ohm`` submodel builds ``phi_s_p`` as
   ``-IndefiniteIntegral(i_s / sigma, x_p) + boundary_value(...)``, so
   ``grad(phi_s_p)`` -- needed by the solid Ohmic heating term -- ended
   up internal-edge-only as well.
"""

import numpy as np

import pybamm


_HALF_CELL_EXTRA_PARAMS = {
    # Xu2019 is a half-cell parameter set but is missing thermal /
    # current-collector data; supply just enough to build a lumped-thermal
    # DFN without raising "Parameter ... not found".
    "Negative current collector thickness [m]": 1.2e-05,
    "Positive current collector thickness [m]": 1.6e-05,
    "Cell cooling surface area [m2]": 0.00531,
    "Cell volume [m3]": 2.42e-05,
    "Cell thermal expansion coefficient [m.K-1]": 1.1e-06,
    "Negative current collector conductivity [S.m-1]": 58411000.0,
    "Positive current collector conductivity [S.m-1]": 36914000.0,
    "Negative current collector density [kg.m-3]": 8960.0,
    "Positive current collector density [kg.m-3]": 2700.0,
    "Negative current collector specific heat capacity [J.kg-1.K-1]": 385.0,
    "Positive current collector specific heat capacity [J.kg-1.K-1]": 897.0,
    "Negative current collector thermal conductivity [W.m-1.K-1]": 401.0,
    "Positive current collector thermal conductivity [W.m-1.K-1]": 237.0,
    "Negative electrode density [kg.m-3]": 534.0,
    "Negative electrode specific heat capacity [J.kg-1.K-1]": 3580.0,
    "Positive electrode density [kg.m-3]": 3262.0,
    "Positive electrode specific heat capacity [J.kg-1.K-1]": 700.0,
    "Positive electrode thermal conductivity [W.m-1.K-1]": 2.1,
    "Separator density [kg.m-3]": 397.0,
    "Separator specific heat capacity [J.kg-1.K-1]": 700.0,
    "Separator thermal conductivity [W.m-1.K-1]": 0.16,
    "Total heat transfer coefficient [W.m-2.K-1]": 10.0,
}


def _half_cell_params():
    params = pybamm.ParameterValues("Xu2019")
    params.update(_HALF_CELL_EXTRA_PARAMS, check_already_exists=False)
    return params


class TestHalfCellSurfaceFormLumpedThermal:
    """
    Guards against #5414: half-cell DFN with surface form + lumped thermal
    used to fail with a shape-mismatch ``ShapeError`` during discretisation.
    """

    def test_phi_e_bc_registered_in_half_cell(self):
        """
        The full surface-form electrolyte conductivity must register an
        inter-domain BC entry for ``phi_e`` (the electrolyte potential
        concatenation) even when the negative submodel is skipped (planar
        Li metal). Without it, ``set_internal_boundary_conditions`` cannot
        split the concatenation's BCs across the separator/positive
        orphans, and ``grad(phi_e_s)`` ends up internal-edge-only.
        """
        model = pybamm.lithium_ion.DFN(
            {
                "working electrode": "positive",
                "surface form": "algebraic",
            }
        )
        phi_e = model.variables["Electrolyte potential [V]"]
        assert phi_e in model.boundary_conditions, (
            "Half-cell `phi_e` must be a BC key so the internal-boundary "
            "split applies to its separator/positive orphans."
        )

    def test_phi_s_bc_registered_for_positive_in_half_cell(self):
        """
        ``base_ohm.BaseModel.set_boundary_conditions`` must register BCs
        for the porous positive electrode potential ``phi_s_p`` even
        when the negative electrode is planar; otherwise ``grad(phi_s_p)``
        in the solid Ohmic heating term has no BCs.
        """
        model = pybamm.lithium_ion.DFN(
            {
                "working electrode": "positive",
                "surface form": "false",
            }
        )
        phi_s_p = model.variables["Positive electrode potential [V]"]
        assert phi_s_p in model.boundary_conditions, (
            "Half-cell `phi_s_p` must carry BCs from base_ohm even when the "
            "negative electrode is planar."
        )

    def test_half_cell_surface_form_lumped_builds_and_solves(self):
        """
        End-to-end: half-cell DFN with each (surface_form, thermal)
        combination must build and solve. This is the exact scenario from
        the bug report.
        """
        params = _half_cell_params()
        for surface_form in ["algebraic", "differential"]:
            for thermal in ["isothermal", "lumped"]:
                opts = {
                    "working electrode": "positive",
                    "surface form": surface_form,
                    "thermal": thermal,
                }
                model = pybamm.lithium_ion.DFN(opts)
                sim = pybamm.Simulation(
                    model, parameter_values=params, C_rate=1.0
                )
                sol = sim.solve([0, 60])
                voltage = sol["Voltage [V]"].entries
                assert np.all(np.isfinite(voltage)), (
                    f"Voltage went non-finite for {opts}"
                )
                # Half-cell vs Li metal positive at full SOC should sit
                # well above 3 V.
                assert voltage[0] > 3.0, (
                    f"Initial voltage too low for {opts}: {voltage[0]}"
                )

    def test_full_cell_lumped_thermal_unaffected(self):
        """
        The fixes only add BCs that were missing in the half-cell case --
        full-cell numerics must be untouched.
        """
        params = pybamm.ParameterValues("Chen2020")
        baselines = {}
        for opts in [
            {"thermal": "lumped"},
            {"surface form": "algebraic", "thermal": "lumped"},
            {"surface form": "differential", "thermal": "lumped"},
        ]:
            model = pybamm.lithium_ion.DFN(opts)
            sim = pybamm.Simulation(model, parameter_values=params, C_rate=1.0)
            sol = sim.solve([0, 600])
            key = repr(sorted(opts.items()))
            baselines[key] = float(sol["Voltage [V]"].entries[-1])
            assert np.isfinite(baselines[key]), f"NaN voltage for {opts}"
            # Full-cell at the end of a 10-min 1C discharge from full SOC
            # should still be comfortably above the 2.5 V cutoff.
            assert 2.5 < baselines[key] < 4.5, (
                f"Voltage {baselines[key]} out of physical range for {opts}"
            )
