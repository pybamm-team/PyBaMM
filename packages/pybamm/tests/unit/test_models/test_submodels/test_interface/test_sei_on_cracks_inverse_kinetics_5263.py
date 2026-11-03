"""
Regression guard for issue #5263.

The SPM / SPMe x-averaged ``CurrentForInverseKinetics`` submodel computes
the main intercalation current density as ``j_tot - j_sei - j_stripping``
and used to omit ``j_sei_on_cracks`` entirely. SEI on cracks consumes
lithium from the main reaction, so leaving it out gave the simulated cell
a free pass on the SEI-on-cracks Li loss: the variable
``"Loss of capacity to negative SEI on cracks [A.h]"`` was reported but
never actually shrank the deliverable capacity, while DFN's surface-form
mass balance subtracted it through charge conservation. Comparing a
30-cycle ageing experiment, DFN's time-integrated charge capacity loss
matched ``SEI + SEI-on-cracks``, but SPM's only matched ``SEI`` alone.
"""

import numpy as np
import pytest

import pybamm


def _extract_charge_capacities(solution):
    time = solution["Time [s]"].entries
    current = solution["Current [A]"].entries
    discharge_mask = current < 0
    edges = np.where(np.diff(discharge_mask.astype(int)) != 0)[0] + 1
    if discharge_mask[0]:
        edges = np.insert(edges, 0, 0)
    if discharge_mask[-1]:
        edges = np.append(edges, len(current))
    caps = []
    for i in range(0, len(edges), 2):
        s, e = edges[i], edges[i + 1]
        if e - s > 1:
            caps.append(np.trapezoid(-current[s:e], time[s:e]) / 3600)
    return np.asarray(caps)


def _run_cracks_experiment(ModCls):
    options = {
        "SEI": "solvent-diffusion limited",
        "SEI porosity change": "true",
        "particle mechanics": ("swelling and cracking", "swelling only"),
        "SEI on cracks": "true",
    }
    param = pybamm.ParameterValues("OKane2022")
    var_pts = {"x_n": 5, "x_s": 5, "x_p": 5, "r_n": 30, "r_p": 30}
    cycles = 30
    exp = pybamm.Experiment(
        [
            (
                "Discharge at 1C until 2.5 V",
                "Charge at 0.3C until 4.2 V",
                "Hold at 4.2 V until C/100",
            )
        ]
        * cycles
        + ["Discharge at 0.1C until 2.5 V"]
    )
    model = ModCls(options)
    sim = pybamm.Simulation(
        model, parameter_values=param, experiment=exp, var_pts=var_pts
    )
    sol = sim.solve()
    sei = float(sol["Loss of capacity to negative SEI [A.h]"].entries[-1])
    sei_c = float(sol["Loss of capacity to negative SEI on cracks [A.h]"].entries[-1])
    caps = _extract_charge_capacities(sol)
    actual_loss = caps[0] - caps[-2]
    return sei, sei_c, actual_loss


class TestSEIOnCracksInverseKineticsFix:
    """Guards against #5263: SPM ignored SEI-on-cracks Li loss."""

    def test_sei_on_cracks_current_subtracted_in_inverse_kinetics(self):
        """
        Inspect the model graph directly: when SEI-on-cracks is enabled,
        the ``Negative electrode SEI on cracks interfacial current density``
        variable must appear inside the symbol tree of the negative
        electrode's main-reaction interfacial current density used by the
        SPM/SPMe x-averaged inverse kinetics. Prior to #5263's fix this
        was absent, leaving SEI-on-cracks Li loss unaccounted for.
        """
        options = {
            "SEI": "solvent-diffusion limited",
            "particle mechanics": ("swelling and cracking", "swelling only"),
            "SEI on cracks": "true",
        }
        model = pybamm.lithium_ion.SPM(options)
        j_main = model.variables[
            "X-averaged negative electrode interfacial current density [A.m-2]"
        ]
        flat = {n.name for n in (j_main, *j_main.pre_order())}
        assert any("SEI on cracks" in n for n in flat), (
            "The SPM main-reaction current density does not depend on the "
            "SEI on cracks current; SEI-on-cracks Li loss is silently dropped."
        )

    @pytest.mark.parametrize(
        "ModCls", [pybamm.lithium_ion.SPM, pybamm.lithium_ion.SPMe]
    )
    def test_capacity_loss_matches_sei_plus_cracks(self, ModCls):
        """
        Time-integrated charge-capacity loss of a 30-cycle ageing
        experiment must agree with the *sum* of the standard SEI and
        SEI-on-cracks losses (not SEI alone). Acceptance band is wide
        because the time-integration only samples the cycles, and the
        DFN baseline runs at ~0.77 -- well above the pre-fix SPM value
        of ~0.20.
        """
        sei, sei_c, actual = _run_cracks_experiment(ModCls)
        ratio_total = actual / (sei + sei_c)
        ratio_sei_only = actual / sei
        assert ratio_total > 0.6, (
            f"{ModCls.__name__}: actual cap loss {actual:.5f} Ah is only "
            f"{ratio_total:.2f}x the reported SEI + SEI-on-cracks loss "
            f"{sei + sei_c:.5f} Ah, indicating SEI-on-cracks Li loss is "
            f"still being ignored by the main reaction."
        )
        assert ratio_sei_only > 1.5, (
            f"{ModCls.__name__}: actual cap loss {actual:.5f} Ah is only "
            f"{ratio_sei_only:.2f}x the reported standalone SEI loss "
            f"{sei:.5f} Ah -- inconsistent with SEI-on-cracks contributing."
        )
