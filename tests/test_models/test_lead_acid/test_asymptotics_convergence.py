#
# Tests for the asymptotic convergence of the simplified models
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import tests

import numpy as np
import unittest


class TestAsymptoticConvergence(unittest.TestCase):
    def test_leading_order_convergence(self):
        """
        Check that the leading-order model solution converges linearly in C_e to the
        full model solution
        """
        # Create models
        leading_order_model = pybamm.lead_acid.LOQS()
        full_model = pybamm.lead_acid.NewmanTiedemann()
        # Same parameters, same geometry
        parameter_values = full_model.default_parameter_values
        parameter_values.process_model(leading_order_model)
        parameter_values.process_model(full_model)
        geometry = full_model.default_geometry
        parameter_values.process_geometry(geometry)

        # Discretise (same mesh, create two discretisations)
        var_pts = {
            pybamm.standard_spatial_vars.x_n: 3,
            pybamm.standard_spatial_vars.x_s: 3,
            pybamm.standard_spatial_vars.x_p: 3,
            pybamm.standard_spatial_vars.r_n: 1,
            pybamm.standard_spatial_vars.r_p: 1,
        }
        mesh = pybamm.Mesh(geometry, full_model.default_submesh_types, var_pts)
        loqs_disc = pybamm.Discretisation(mesh, full_model.default_spatial_methods)
        loqs_disc.process_model(leading_order_model)
        full_disc = pybamm.Discretisation(mesh, full_model.default_spatial_methods)
        full_disc.process_model(full_model)

        def get_l2_error(current):
            # Update current (and hence C_e) in the parameters
            param = pybamm.ParameterValues(
                base_parameters=full_model.default_parameter_values,
                optional_parameters={"Typical current density": current},
            )
            param.process_discretised_model(leading_order_model, loqs_disc)
            param.process_discretised_model(full_model, full_disc)
            # Solve, make sure times are the same
            t_eval = np.linspace(0, 0.1, 10)
            solver_loqs = leading_order_model.default_solver
            solver_loqs.solve(leading_order_model, t_eval)
            solver_full = full_model.default_solver
            solver_full.solve(full_model, t_eval)
            np.testing.assert_array_equal(solver_loqs.t, solver_full.t)

            # Post-process variables
            t, y_loqs = solver_loqs.t, solver_loqs.y
            y_full = solver_full.y
            voltage_loqs = pybamm.ProcessedVariable(
                leading_order_model.variables["Voltage"], t, y_loqs
            )
            voltage_full = pybamm.ProcessedVariable(
                full_model.variables["Voltage"], t, y_full, mesh=full_disc.mesh
            )

            # Compare
            return np.linalg.norm(
                voltage_loqs.evaluate(t) - voltage_full.evaluate(t)
            ) / np.linalg.norm(voltage_full.evaluate(t))

        # Get errors
        currents = 0.5 / (2 ** np.arange(3))
        errs = np.array([get_l2_error(current) for current in currents])

        # Get rates: expect linear convergence
        rates = np.log2(errs[:-1] / errs[1:])
        np.testing.assert_array_less(0.99 * np.ones_like(rates), rates)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
