#
# Tests for the total active material submodel
#
import pytest

import pybamm
from pybamm.models.submodels.active_material.total_active_material import Total

 
class TestTotalActiveMaterial:
    def test_surface_area_to_volume_ratio_is_sum_of_phase_ratios(self):
        # Regression test for #5802: for spherical particles the total
        # surface area to volume ratio must be the sum of the per-phase
        # ratios 3 * eps_k / R_k, not sum(eps_k * R_k**2) / (3 * sum(R_k**3))
        options = {"particle phases": ("2", "1"), "particle shape": "spherical"}
        submodel = Total(None, "negative", options)

        # distinct radii so that "ratio of sums" differs from "sum of ratios"
        eps = {"primary": 0.4, "secondary": 0.35}
        R = {"primary": 5e-6, "secondary": 1e-5}
        phases = ("primary", "secondary")

        variables = {}
        for phase in phases:
            variables[f"Negative electrode {phase} active material volume fraction"] = (
                pybamm.Scalar(eps[phase])
            )
            variables[
                f"X-averaged negative electrode {phase} active material volume fraction"
            ] = pybamm.Scalar(eps[phase])
            variables[
                f"Negative electrode {phase} active material volume fraction change [s-1]"
            ] = pybamm.Scalar(0)
            variables[
                f"X-averaged negative electrode {phase} "
                "active material volume fraction change [s-1]"
            ] = pybamm.Scalar(0)
            variables[
                f"Loss of lithium due to loss of {phase} "
                "active material in negative electrode [mol]"
            ] = pybamm.Scalar(0)
            variables[f"Negative electrode {phase} phase capacity [A.h]"] = (
                pybamm.Scalar(1)
            )
            # per-phase ratio, as reported by the base active material submodel
            variables[
                f"Negative electrode {phase} surface area to volume ratio [m-1]"
            ] = pybamm.Scalar(3 * eps[phase] / R[phase])
            variables[f"Negative {phase} particle radius [m]"] = pybamm.Scalar(R[phase])

        variables = submodel.get_coupled_variables(variables)

        expected = sum(3 * eps[phase] / R[phase] for phase in phases)
        total = variables["Negative electrode surface area to volume ratio [m-1]"]
        assert total.evaluate() == pytest.approx(expected)

        # the old (buggy) expression gives a different answer here
        buggy = sum(eps[phase] * R[phase] ** 2 for phase in phases) / (
            3 * sum(R[phase] ** 3 for phase in phases)
        )
        assert buggy != pytest.approx(expected)

    def test_total_active_material_volume_fraction_sums_phases(self):
        options = {"particle phases": ("2", "1"), "particle shape": "spherical"}
        submodel = Total(None, "negative", options)

        variables = {}
        for phase in ("primary", "secondary"):
            variables[f"Negative electrode {phase} active material volume fraction"] = (
                pybamm.Scalar(0.3)
            )
            variables[
                f"X-averaged negative electrode {phase} active material volume fraction"
            ] = pybamm.Scalar(0.3)
            variables[
                f"Negative electrode {phase} active material volume fraction change [s-1]"
            ] = pybamm.Scalar(0)
            variables[
                f"X-averaged negative electrode {phase} "
                "active material volume fraction change [s-1]"
            ] = pybamm.Scalar(0)
            variables[
                f"Loss of lithium due to loss of {phase} "
                "active material in negative electrode [mol]"
            ] = pybamm.Scalar(0)
            variables[f"Negative electrode {phase} phase capacity [A.h]"] = (
                pybamm.Scalar(1)
            )
            variables[
                f"Negative electrode {phase} surface area to volume ratio [m-1]"
            ] = pybamm.Scalar(1000)

        variables = submodel.get_coupled_variables(variables)

        total_eps = variables["Negative electrode active material volume fraction"]
        assert total_eps.evaluate() == pytest.approx(0.6)
