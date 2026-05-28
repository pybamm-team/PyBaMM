"""
Guards for #5281: SEI on cracks looks up the roughness ratio variable
without the per-phase prefix, so models with composite electrodes
(``"particle phases": "2"``) fail to build with::

    KeyError: 'Negative primary SEI on cracks thickness [m]'

The actual missing variable is ``"Negative electrode roughness ratio"``;
PyBaMM's submodel-loop reports the downstream SEI on cracks thickness
key once the build runs out of retries. The particle mechanics submodel
registers the roughness as ``"Negative <phase> electrode roughness
ratio"`` (see ``particle_mechanics/base_mechanics.py``), and the
volumetric current density helper in ``interface/base_interface.py``
already looks it up with the phase prefix. The SEI submodels were the
only consumers that dropped the prefix.

The fix is in:
- ``src/pybamm/models/submodels/interface/sei/sei_thickness.py``
- ``src/pybamm/models/submodels/interface/sei/sei_growth.py``

which now read ``f"{Domain} {self.phase_name}electrode roughness ratio"``
(and the X-averaged variant for the ``x-average`` reaction location).
"""

import pybamm

# Reproducer from issue #5281 (user satishrapol). Single-phase positive,
# composite negative with primary cracking + ec-reaction-limited SEI.
COMPOSITE_OPTS_5281 = {
    "particle phases": ("2", "1"),
    "open-circuit potential": (("single", "current sigmoid"), "single"),
    "SEI": "ec reaction limited",
    "particle mechanics": ("swelling and cracking", "swelling only"),
    "loss of active material": "stress-driven",
    "SEI on cracks": "true",
    "surface form": "algebraic",
}


class TestSEIOnCracksCompositePhase5281:
    """Guards #5281: composite electrode + SEI on cracks must build."""

    def test_spme_composite_ec_reaction_limited_builds(self):
        """Exact reproducer from #5281 — SPMe with composite negative."""
        # Before the fix this raised ``ModelError: Missing variable for
        # submodel 'negative primary sei on cracks': 'Negative primary SEI
        # on cracks thickness [m]'`` because ``SEIThickness`` (and the
        # ``set_rhs`` branch of ``SEIGrowth`` in the ``"x-average"`` case)
        # was looking up ``"Negative electrode roughness ratio"`` instead
        # of ``"Negative primary electrode roughness ratio"``.
        model = pybamm.lithium_ion.SPMe(options=COMPOSITE_OPTS_5281)
        assert "Negative primary SEI on cracks thickness [m]" in model.variables
        assert "Negative secondary SEI on cracks thickness [m]" in model.variables

    def test_dfn_composite_ec_reaction_limited_builds(self):
        """DFN reaches the same SEI submodels through the full-electrode
        branch of ``SEIGrowth.set_rhs`` rather than the x-averaged branch,
        so it covers the second occurrence of the bug in sei_growth.py."""
        model = pybamm.lithium_ion.DFN(options=COMPOSITE_OPTS_5281)
        assert "Negative primary SEI on cracks thickness [m]" in model.variables
        assert "Negative secondary SEI on cracks thickness [m]" in model.variables

    def test_spm_composite_solvent_diffusion_builds(self):
        """SEI options other than 'ec reaction limited' must not regress."""
        opts = dict(COMPOSITE_OPTS_5281, **{"SEI": "solvent-diffusion limited"})
        # SPM uses x-averaged SEI; the 'x-average' branch of
        # SEIGrowth.set_rhs hits the X-averaged roughness ratio lookup.
        model = pybamm.lithium_ion.SPM(options=opts)
        assert "Negative primary SEI on cracks thickness [m]" in model.variables
        assert "Negative secondary SEI on cracks thickness [m]" in model.variables

    def test_per_phase_roughness_keys_are_consumed(self):
        """The fixed lookups must reference the phased roughness keys
        (``"Negative primary/secondary electrode roughness ratio"``) and
        not the un-phased ``"Negative electrode roughness ratio"``."""
        model = pybamm.lithium_ion.DFN(options=COMPOSITE_OPTS_5281)
        # Particle mechanics registers the phased keys; SEI on cracks
        # consumes them. We verify the model exposes the registered keys.
        assert "Negative primary electrode roughness ratio" in model.variables
        assert "Negative secondary electrode roughness ratio" in model.variables
        # The phase-stripped key historically existed only via the
        # single-phase code path; for composite electrodes the SEI
        # submodels must not depend on it.
        assert "Negative electrode roughness ratio" not in model.variables

    def test_single_phase_regression(self):
        """Non-composite models that previously worked must keep working
        after switching the lookups to use ``self.phase_name``. For a
        single-phase electrode ``phase_name`` is empty, recovering the
        original variable name verbatim."""
        opts = {
            "SEI": "solvent-diffusion limited",
            "particle mechanics": "swelling and cracking",
            "SEI on cracks": "true",
        }
        for cls in (
            pybamm.lithium_ion.SPM,
            pybamm.lithium_ion.SPMe,
            pybamm.lithium_ion.DFN,
        ):
            model = cls(options=opts)
            assert "Negative SEI on cracks thickness [m]" in model.variables
            assert "Negative electrode roughness ratio" in model.variables
