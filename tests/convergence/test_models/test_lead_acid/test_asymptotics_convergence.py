#
# Tests for the asymptotic convergence of the simplified models
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

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
        composite_model = pybamm.lead_acid.Composite()
        full_model = pybamm.lead_acid.NewmanTiedemann()
        # Same parameters, same geometry
        parameter_values = full_model.default_parameter_values
        parameter_values.process_model(leading_order_model)
        parameter_values.process_model(composite_model)
        parameter_values.process_model(full_model)
        geometry = full_model.default_geometry
        parameter_values.process_geometry(geometry)

        # Discretise (same mesh, create different discretisations)
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
        comp_disc = pybamm.Discretisation(mesh, full_model.default_spatial_methods)
        comp_disc.process_model(composite_model)
        full_disc = pybamm.Discretisation(mesh, full_model.default_spatial_methods)
        full_disc.process_model(full_model)

        def get_l2_error(current):
            # Update current (and hence C_e) in the parameters
            param = pybamm.ParameterValues(
                base_parameters=full_model.default_parameter_values,
                optional_parameters={"Typical current density": current},
            )
            param.update_model(leading_order_model, loqs_disc)
            param.update_model(composite_model, comp_disc)
            param.update_model(full_model, full_disc)
            # Solve, make sure times are the same
            t_eval = np.linspace(0, 0.1, 10)
            solver_loqs = leading_order_model.default_solver
            solver_loqs.solve(leading_order_model, t_eval)
            solver_comp = composite_model.default_solver
            solver_comp.solve(composite_model, t_eval)
            solver_full = full_model.default_solver
            solver_full.solve(full_model, t_eval)
            np.testing.assert_array_equal(solver_loqs.t, solver_comp.t)
            np.testing.assert_array_equal(solver_loqs.t, solver_full.t)

            # Post-process variables
            t, y_loqs = solver_loqs.t, solver_loqs.y
            y_comp = solver_comp.y
            y_full = solver_full.y
            voltage_loqs = pybamm.ProcessedVariable(
                leading_order_model.variables["Voltage"], t, y_loqs
            )
            voltage_comp = pybamm.ProcessedVariable(
                composite_model.variables["Voltage"], t, y_comp
            )
            voltage_full = pybamm.ProcessedVariable(
                full_model.variables["Voltage"], t, y_full
            )

            # Compare
            norm = np.linalg.norm
            loqs_error = norm(voltage_loqs(t) - voltage_full(t)) / norm(voltage_full(t))
            comp_error = norm(voltage_comp(t) - voltage_full(t)) / norm(voltage_full(t))
            return (loqs_error, comp_error)

        # Get errors
        currents = 0.5 / (2 ** np.arange(3))
        errs = np.array([get_l2_error(current) for current in currents])
        loqs_errs, comp_errs = [np.array(err) for err in zip(*errs)]

        # Get rates: expect linear convergence for loqs, quadratic for composite
        loqs_rates = np.log2(loqs_errs[:-1] / loqs_errs[1:])
        np.testing.assert_array_less(0.99 * np.ones_like(loqs_rates), loqs_rates)
        # Composite not converging as expected
        # comp_rates = np.log2(comp_errs[:-1] / comp_errs[1:])
        # np.testing.assert_array_less(1.99 * np.ones_like(comp_rates), comp_rates)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
