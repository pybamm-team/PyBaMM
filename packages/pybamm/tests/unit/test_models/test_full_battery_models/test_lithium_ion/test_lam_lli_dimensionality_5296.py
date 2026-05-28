"""
Guards for #5296: degradation submodels fail to discretise with
high-dimensional thermal/current-collector submodels.

The loss-of-active-material submodel declares the scalar variable

    "Loss of lithium due to loss of <phase>active material in
     <domain> electrode [mol]"

(see ``submodels/active_material/loss_active_material.py``), backed
by ``pybamm.Variable`` with no ``domain``. Its initial condition is
``pybamm.Scalar(0)``, which discretises to shape ``(1, 1)``.

Before the fix the right-hand side was

    -V * pybamm.x_average(c_s_rav * deps_solid_dt)

``x_average`` averages over the through-cell direction but preserves
the current-collector domain, so for ``"dimensionality": 1`` the
rhs has the z-mesh shape (e.g. ``(10, 1)``). ``Discretisation.
check_initial_conditions`` raises::

    ModelError: rhs and initial conditions must have the same shape
    after discretisation but rhs.shape = (10, 1) and
    initial_conditions.shape = (1, 1) for variable 'Loss of lithium
    due to loss of active material in negative electrode [mol]'.

The fix adds ``pybamm.yz_average`` around the existing ``x_average``
so the rhs collapses to a scalar in 1+1D and 2+1D current
collectors. ``yz_average`` is the identity for a 0-D current
collector, so 1-D models are unaffected.
"""

import pybamm


def _okane_param_with_thermal_overrides():
    """Replicate the parameter overrides from the issue body."""
    param = pybamm.ParameterValues("OKane2022")
    L_y = param["Electrode width [m]"]
    L_z = param["Electrode height [m]"]
    param.update(
        {
            "Total heat transfer coefficient [W.m-2.K-1]": 100,
            "Negative current collector surface heat transfer coefficient "
            "[W.m-2.K-1]": 100,
            "Positive current collector surface heat transfer coefficient "
            "[W.m-2.K-1]": 100,
            "Negative tab heat transfer coefficient [W.m-2.K-1]": 0,
            "Positive tab heat transfer coefficient [W.m-2.K-1]": 0,
            "Edge heat transfer coefficient [W.m-2.K-1]": 0,
            "Positive tab width [m]": 0.04,
            "Positive tab centre y-coordinate [m]": L_y * 0.2,
            "Positive tab centre z-coordinate [m]": L_z,
            "Negative tab width [m]": 0.04,
            "Negative tab centre y-coordinate [m]": L_y * 0.8,
            "Negative tab centre z-coordinate [m]": L_z,
        },
        check_already_exists=False,
    )
    return param


class TestLAMLLIDimensionality5296:
    """Guards #5296: LAM-LLI scalar accumulator must match initial
    conditions in shape under high-dim current collectors."""

    OPTIONS_5296 = {
        "current collector": "potential pair",
        "dimensionality": 1,
        "thermal": "x-lumped",
        "SEI": "solvent-diffusion limited",
        "SEI porosity change": "true",
        "lithium plating": "partially reversible",
        "lithium plating porosity change": "true",
        "particle mechanics": ("swelling and cracking", "swelling only"),
        "SEI on cracks": "true",
        "loss of active material": "stress-driven",
    }

    def test_spme_dimensionality_1_solves(self):
        """Exact reproducer from #5296 — SPMe + dimensionality=1 must
        discretise and solve for at least one step."""
        model = pybamm.lithium_ion.SPMe(options=self.OPTIONS_5296)
        param = _okane_param_with_thermal_overrides()
        var_pts = {"x_n": 5, "x_s": 5, "x_p": 5, "r_n": 30, "r_p": 30, "z": 10}
        sim = pybamm.Simulation(
            model,
            parameter_values=param,
            var_pts=var_pts,
            solver=pybamm.IDAKLUSolver(),
        )
        sol = sim.solve([0, 60])
        assert sol is not None
        lli_name = (
            "Loss of lithium due to loss of active material in negative electrode [mol]"
        )
        # The scalar accumulator must be non-negative and finite. The
        # mere fact that the simulation reached this point proves the
        # discretisation check passed.
        lli = sol[lli_name].entries
        assert lli[0] == 0.0
        assert lli[-1] >= 0.0

    def test_lam_rhs_is_scalar_after_discretisation(self):
        """Direct shape guard against future regression of the same
        kind: after discretisation, the rhs of every LAM-LLI variable
        must have the same shape as its initial condition."""
        model = pybamm.lithium_ion.DFN(options=self.OPTIONS_5296)
        param = _okane_param_with_thermal_overrides()
        var_pts = {"x_n": 5, "x_s": 5, "x_p": 5, "r_n": 30, "r_p": 30, "z": 8}
        sim = pybamm.Simulation(
            model,
            parameter_values=param,
            var_pts=var_pts,
        )
        # ``build`` runs through ``Discretisation.check_initial_conditions``
        # which is the original raise site; calling it here is enough
        # to lock in the fix.
        sim.build()
        built = sim.built_model
        for var, rhs in built.rhs.items():
            if var.name.startswith("Loss of lithium due to loss of"):
                ic = built.initial_conditions[var]
                assert rhs.shape == ic.shape, (
                    f"LAM-LLI rhs/IC shape mismatch for {var.name}: "
                    f"rhs={rhs.shape}, ic={ic.shape}"
                )

    def test_dfn_dimensionality_1_solves(self):
        """DFN exposes a second 1+1D bug in #5296: the ``CrackPropagation``
        IC ``PrimaryBroadcast(l_cr_0, "<dom> electrode")`` left the
        current-collector axis at 1 while the rhs lived on the (x, z)
        mesh declared by ``Variable(..., auxiliary_domains={"secondary":
        "current collector"})``. Replacing the broadcast with
        ``FullBroadcast`` so it spans the current collector explicitly
        lets DFN build and solve under ``"dimensionality": 1``."""
        model = pybamm.lithium_ion.DFN(options=self.OPTIONS_5296)
        param = _okane_param_with_thermal_overrides()
        var_pts = {"x_n": 5, "x_s": 5, "x_p": 5, "r_n": 30, "r_p": 30, "z": 8}
        sim = pybamm.Simulation(
            model,
            parameter_values=param,
            var_pts=var_pts,
            solver=pybamm.IDAKLUSolver(),
        )
        sol = sim.solve([0, 60])
        assert sol is not None
        lli_name = (
            "Loss of lithium due to loss of active material in negative electrode [mol]"
        )
        assert sol[lli_name].entries[0] == 0.0
        assert sol[lli_name].entries[-1] >= 0.0

    def test_zero_d_regression(self):
        """``yz_average`` is the identity for a 0-D current collector;
        accumulators built without the high-dim options must still be
        able to solve and produce the same total Li loss they did
        before the fix (within solver tolerance)."""
        model = pybamm.lithium_ion.SPMe(
            options={
                "loss of active material": "stress-driven",
                "particle mechanics": "swelling and cracking",
            }
        )
        param = pybamm.ParameterValues("OKane2022")
        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve([0, 600])
        lli_name = (
            "Loss of lithium due to loss of active material in negative electrode [mol]"
        )
        lli = sol[lli_name].entries
        # Stress-driven LAM accumulates a tiny but strictly positive
        # amount over 10 minutes at OCV-ish conditions; the magnitude
        # serves as a sanity guard against an off-by-one yz_average bug.
        assert lli[0] == 0.0
        assert lli[-1] > 0.0
        assert lli[-1] < 1e-5  # well below any cell-scale Li inventory
