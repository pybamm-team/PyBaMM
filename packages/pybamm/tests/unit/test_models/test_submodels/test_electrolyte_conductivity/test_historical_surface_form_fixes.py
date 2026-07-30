import pybamm


class TestLeadingSurfaceFormFixes:
    """Guards for leading-order surface form conductivity fixes."""

    @staticmethod
    def _walk_symbols(symbol):
        yield symbol
        for child in getattr(symbol, "children", []):
            yield from TestLeadingSurfaceFormFixes._walk_symbols(child)

    @classmethod
    def _contains_name(cls, symbol, name):
        return any(name in node.name for node in cls._walk_symbols(symbol))

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

        This test directly inspects the model equation structure to verify the
        surface area expression is in the denominator with C_dl for each electrode.
        """
        model = pybamm.lithium_ion.SPM(options={"surface form": "differential"})

        for domain in ["negative", "positive"]:
            var = next(
                key
                for key in model.rhs
                if key.name
                == f"X-averaged {domain} electrode surface potential difference [V]"
            )
            rhs_expr = model.rhs[var]

            matching_divisions = [
                node
                for node in self._walk_symbols(rhs_expr)
                if isinstance(node, pybamm.Division)
                and isinstance(node.left, pybamm.Scalar)
                and node.left.value == 1
                and self._contains_name(
                    node.right,
                    f"{domain.capitalize()} electrode active material volume fraction",
                )
                and self._contains_name(
                    node.right, f"{domain.capitalize()} particle radius"
                )
                and self._contains_name(
                    node.right,
                    f"{domain.capitalize()} electrode double-layer capacity",
                )
            ]

            assert len(matching_divisions) == 1
