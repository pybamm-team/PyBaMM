#
# Tests for the asymptotic convergence of the simplified models
#
from tests import TestCase
import pybamm

import numpy as np
import unittest


class TestAsymptoticConvergence(TestCase):
    def test_leading_order_convergence(self):
        """
        Check that the leading-order model solution converges linearly in C_e to the
        full model solution
        """
        # Create models
        leading_order_model = pybamm.lead_acid.LOQS()
        full_model = pybamm.lead_acid.Full()

        # Same parameters, same geometry
        parameter_values = full_model.default_parameter_values
        parameter_values["Current function [A]"] = "[input]"
        parameter_values.process_model(leading_order_model)
        parameter_values.process_model(full_model)
        geometry = full_model.default_geometry
        parameter_values.process_geometry(geometry)

        # Discretise (same mesh, create different discretisations)
        var_pts = {"x_n": 3, "x_s": 3, "x_p": 3}
        mesh = pybamm.Mesh(geometry, full_model.default_submesh_types, var_pts)

        method_options = {"extrapolation": {"order": "linear", "use bcs": False}}
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(method_options),
            "current collector": pybamm.ZeroDimensionalSpatialMethod(method_options),
        }
        loqs_disc = pybamm.Discretisation(mesh, spatial_methods)
        loqs_disc.process_model(leading_order_model)
        full_disc = pybamm.Discretisation(mesh, spatial_methods)
        full_disc.process_model(full_model)

        def get_max_error(current):
            pybamm.logger.info("current = {}".format(current))
            # Solve, make sure times are the same and use tight tolerances
            t_eval = np.linspace(0, 3600 * 17 / current)
            solver = pybamm.CasadiSolver()
            solver.rtol = 1e-8
            solver.atol = 1e-8
            solution_loqs = solver.solve(
                leading_order_model, t_eval, inputs={"Current function [A]": current}
            )
            solution_full = solver.copy().solve(
                full_model, t_eval, inputs={"Current function [A]": current}
            )

            # Post-process variables
            voltage_loqs = solution_loqs["Voltage [V]"]
            voltage_full = solution_full["Voltage [V]"]

            # Compare
            t_loqs = solution_loqs.t
            t_full = solution_full.t
            t = t_full[: np.min([len(t_loqs), len(t_full)])]
            loqs_error = np.max(np.abs(voltage_loqs(t) - voltage_full(t)))
            return loqs_error

        # Get errors
        currents = 0.5 / (2 ** np.arange(3))
        loqs_errs = np.array([get_max_error(current) for current in currents])
        # Get rates: expect linear convergence for loqs
        loqs_rates = np.log2(loqs_errs[:-1] / loqs_errs[1:])

        np.testing.assert_array_less(0.99 * np.ones_like(loqs_rates), loqs_rates)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
