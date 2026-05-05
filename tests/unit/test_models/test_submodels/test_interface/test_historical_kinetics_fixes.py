"""
Regression tests for historical kinetics submodel bug fixes.

These tests guard against the reintroduction of bugs that were fixed but
did not have regression tests added at the time of the fix.
"""

import pybamm


def _walk_symbols(symbol):
    yield symbol
    for child in getattr(symbol, "children", []):
        yield from _walk_symbols(child)


class TestHalfCellKineticsOCPBroadcastFixes:
    """Guards for half-cell kinetics OCP broadcast domain mismatch fixes."""

    def test_ocp_broadcast_orphan_extracted_for_lithium_metal_plating(self):
        """
        Guards against: PR #3211 - #3128 fix half cell models

        The bug was that OCP broadcast orphan extraction didn't handle
        the lithium metal electrode (half-cell) reaction correctly.
        The original code was:

            if isinstance(ocp, pybamm.Broadcast) and delta_phi.domains["secondary"] == [
                "current collector"
            ]:
                ocp = ocp.orphans[0]

        This failed for the "lithium metal plating" reaction where
        delta_phi's secondary domain differs. The fix added:

            if isinstance(ocp, pybamm.Broadcast):
                if self.reaction == "lithium metal plating":
                    ocp = ocp.orphans[0]
                elif delta_phi.domains["secondary"] == ["current collector"]:
                    ocp = ocp.orphans[0]

        This test directly verifies that in a half-cell model, the lithium
        metal interface overpotential (eta_r = delta_phi - ocp) is NOT a
        domain mismatch error (which would happen if OCP remained a Broadcast).
        """
        model = pybamm.lithium_ion.DFN({"working electrode": "positive"})

        eta_li = model.variables["Lithium metal interface reaction overpotential [V]"]
        assert eta_li is not None

        broadcasted_ocp_terms = [
            node
            for node in _walk_symbols(eta_li)
            if isinstance(node, pybamm.Broadcast)
            and any(
                "open-circuit potential" in child.name.lower()
                for child in _walk_symbols(node)
            )
        ]
        assert broadcasted_ocp_terms == []


class TestInverseButlerVolmerHalfCellFixes:
    """Guards for inverse Butler-Volmer half-cell fixes."""

    def test_half_cell_uses_correct_current_density_submodel(self):
        """
        Guards against: PR #1665 - fix bug in inverse_butler_volmer.py

        The bug was in variable update ordering for half-cell mode.
        The original code computed j_tot_av unconditionally, but for
        half-cells (lithium metal electrode), j_tot should directly equal
        i_boundary_cc (current collector current density), not the computed
        j_tot_av.

        The fix restructured the code so half-cell current is handled
        separately via CurrentForInverseKineticsLithiumMetal. This test
        verifies that half-cell SPMe models correctly use the boundary
        current for the lithium metal electrode interfacial current.
        """
        model = pybamm.lithium_ion.SPMe({"working electrode": "positive"})

        j_li = model.variables.get(
            "Lithium metal total interfacial current density [A.m-2]"
        )
        i_boundary = model.variables.get("Current collector current density [A.m-2]")

        assert j_li is not None, "Half-cell model should have Li metal current density"
        assert i_boundary is not None, "Model should have boundary current density"

        j_li_str = str(j_li)

        assert j_li == i_boundary, (
            f"Li metal interfacial current should equal or derive from boundary "
            f"current. Got j_li={j_li_str}, i_boundary={i_boundary}"
        )

    def test_inverse_kinetics_uses_boundary_current_for_half_cell(self):
        """
        Additional guard for PR #1665.

        The original bug caused j_tot_av to be computed unconditionally and
        used for the lithium metal electrode. The fix ensures half-cell
        models use i_boundary_cc (current collector current density) directly.

        This verifies that the lithium metal interfacial current derives
        from the applied current (Current function), not from computed
        average interfacial current density.
        """
        model = pybamm.lithium_ion.SPMe({"working electrode": "positive"})

        j_li = model.variables.get(
            "Lithium metal total interfacial current density [A.m-2]"
        )

        assert j_li is not None, "Half-cell model should have Li metal current density"

        applied_current_terms = [
            node for node in _walk_symbols(j_li) if node.name == "Current function [A]"
        ]

        assert len(applied_current_terms) == 1, (
            f"Li metal interfacial current should derive from 'Current function' "
            f"(the applied current), not from computed average. Got: {j_li}"
        )
