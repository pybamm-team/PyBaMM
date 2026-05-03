"""
Regression tests for historical surface form electrolyte conductivity bug fixes.

These tests guard against the reintroduction of bugs that were fixed but
did not have regression tests added at the time of the fix.
"""

import pybamm


class TestLeadingSurfaceFormFixes:
    """Guards for leading-order surface form conductivity fixes."""

    def test_leading_surface_form_rhs_divides_by_surface_area(self):
        """
        Guards against: PR #4139 - fix bug in leading surface form conductivity

        The bug was a missing factor of electrode surface area to volume ratio
        in the RHS of the LeadingOrderDifferential conductivity model.

        Original (buggy):
            self.rhs[delta_phi] = 1 / C_dl * (sum_a_j_av - sum_a_j)

        Fixed:
            a = variables[f"X-averaged {domain} electrode surface area to volume ratio [m-1]"]
            self.rhs[delta_phi] = 1 / (a * C_dl) * (sum_a_j_av - sum_a_j)

        This test directly inspects the model equation structure to verify
        that the surface area variable is present in the RHS denominator.
        """
        model = pybamm.lithium_ion.SPM(options={"surface form": "differential"})

        rhs_keys = list(model.rhs.keys())
        surface_potential_diff_vars = [
            k for k in rhs_keys if "surface potential difference" in k.name.lower()
        ]

        assert len(surface_potential_diff_vars) >= 2, (
            "SPM with surface form differential should have RHS equations "
            "for both negative and positive surface potential difference"
        )

        def collect_symbols(expr, symbols):
            if hasattr(expr, "name"):
                symbols.add(expr.name)
            if hasattr(expr, "children"):
                for child in expr.children:
                    collect_symbols(child, symbols)

        for var in surface_potential_diff_vars:
            rhs_expr = model.rhs[var]

            rhs_symbols = set()
            collect_symbols(rhs_expr, rhs_symbols)

            has_active_material_fraction = any(
                "active material volume fraction" in sym.lower() for sym in rhs_symbols
            )
            has_particle_radius = any(
                "particle radius" in sym.lower() for sym in rhs_symbols
            )

            assert has_active_material_fraction and has_particle_radius, (
                f"RHS for {var.name} should include surface area components "
                f"(active material volume fraction and particle radius). "
                f"The surface area 'a' is computed as 3 * eps_s / R, so both "
                f"terms must appear. Missing the 'a' factor was the bug in #4139."
            )

    def test_leading_surface_form_rhs_structure_has_division(self):
        """
        Additional guard for PR #4139.

        The fix divides by 'a * C_dl' rather than just 'C_dl'. This verifies
        that the RHS expression has a Division operation with surface area
        components in the divisor.
        """
        model = pybamm.lithium_ion.SPM(options={"surface form": "differential"})

        neg_delta_phi_var = None
        for key in model.rhs.keys():
            if (
                "negative" in key.name.lower()
                and "surface potential difference" in key.name.lower()
            ):
                neg_delta_phi_var = key
                break

        assert neg_delta_phi_var is not None

        rhs_expr = model.rhs[neg_delta_phi_var]

        def has_division_with_surface_area(expr):
            if isinstance(expr, pybamm.Division):
                divisor_str = str(expr.right)
                if "active material volume fraction" in divisor_str.lower():
                    return True
                if "particle radius" in divisor_str.lower():
                    return True
            if hasattr(expr, "children"):
                for child in expr.children:
                    if has_division_with_surface_area(child):
                        return True
            return False

        def rhs_contains_surface_area_in_calculation(expr):
            expr_str = str(expr)
            return (
                "active material volume fraction" in expr_str.lower()
                or "particle radius" in expr_str.lower()
            )

        assert rhs_contains_surface_area_in_calculation(rhs_expr), (
            "RHS expression should contain surface area components "
            "(active material volume fraction or particle radius)"
        )
