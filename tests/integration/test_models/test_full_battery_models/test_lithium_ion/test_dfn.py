#
# Tests for the lithium-ion DFN model
#
import pybamm
import tests

import numpy as np
import unittest


@unittest.skipIf(pybamm.have_scikits_odes(), "scikits.odes not installed")
class TestDFN(unittest.TestCase):
    # def test_basic_processing(self):
    #     options = {"thermal": None}
    #     model = pybamm.lithium_ion.DFN(options)
    #     var = pybamm.standard_spatial_vars
    #     var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 5, var.r_p: 5}
    #     modeltest = tests.StandardModelTest(model, var_pts=var_pts)
    #     modeltest.test_all()

    # def test_optimisations(self):
    #     options = {"thermal": None}
    #     model = pybamm.lithium_ion.DFN(options)
    #     optimtest = tests.OptimisationsTest(model)

    #     original = optimtest.evaluate_model()
    #     simplified = optimtest.evaluate_model(simplify=True)
    #     using_known_evals = optimtest.evaluate_model(use_known_evals=True)
    #     simp_and_known = optimtest.evaluate_model(simplify=True, use_known_evals=True)
    #     simp_and_python = optimtest.evaluate_model(simplify=True, to_python=True)
    #     np.testing.assert_array_almost_equal(original, simplified)
    #     np.testing.assert_array_almost_equal(original, using_known_evals)
    #     np.testing.assert_array_almost_equal(original, simp_and_known)

    #     np.testing.assert_array_almost_equal(original, simp_and_python)

    # def test_full_thermal(self):
    #     options = {"thermal": "full"}
    #     model = pybamm.lithium_ion.DFN(options)
    #     var = pybamm.standard_spatial_vars
    #     var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 5, var.r_p: 5}
    #     modeltest = tests.StandardModelTest(model, var_pts=var_pts)
    #     modeltest.test_all()

    # def test_lumped_thermal(self):
    #     options = {"thermal": "lumped"}
    #     model = pybamm.lithium_ion.DFN(options)
    #     var = pybamm.standard_spatial_vars
    #     var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 5, var.r_p: 5}
    #     modeltest = tests.StandardModelTest(model, var_pts=var_pts)
    #     modeltest.test_all()

    def test_gradient_of_subdomain_var(self):
        # a hacky test which will be replaced by a more general one in
        # standard output tests when #481 is completed

        options = {"thermal": None}
        model = pybamm.lithium_ion.DFN(options)
        c_e = model.variables["Electrolyte concentration"]
        c_e_n, c_e_s, c_e_p = c_e.orphans
        phi_e = model.variables["Electrolyte potential"]
        phi_e_n, phi_e_s, phi_e_p = phi_e.orphans
        model.variables.update(
            {
                "Test grad c_e": pybamm.grad(c_e),
                "Test grad c_e_n": pybamm.grad(c_e_n),
                "Test grad c_e_s": pybamm.grad(c_e_s),
                "Test grad c_e_p": pybamm.grad(c_e_p),
                "Test grad phi_e": pybamm.grad(phi_e),
                "Test grad phi_e_n": pybamm.grad(phi_e_n),
                "Test grad phi_e_s": pybamm.grad(phi_e_s),
                "Test grad phi_e_p": pybamm.grad(phi_e_p),
            }
        )
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.process_model(model)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        t_eval = np.linspace(0, 2, 100)
        solution = model.default_solver.solve(model, t_eval)
        t = solution.t
        x_n = disc.mesh["negative electrode"][0].nodes
        x_s = disc.mesh["separator"][0].nodes
        x_p = disc.mesh["positive electrode"][0].nodes
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        x = disc.mesh.combine_submeshes(*whole_cell)[0].nodes

        variables = pybamm.post_process_variables(
            model.variables, solution.t, solution.y, mesh=disc.mesh
        )

        grad_c_e_n = variables["Test grad c_e_n"]
        grad_c_e_s = variables["Test grad c_e_s"]
        grad_c_e_p = variables["Test grad c_e_p"]
        grad_c_e = variables["Test grad c_e"]

        grad_c_e_combined = np.concatenate(
            (grad_c_e_n(t, x_n), grad_c_e_s(t, x_s), grad_c_e_p(t, x_p)), axis=0
        )

        np.testing.assert_array_almost_equal(
            grad_c_e(t, x), grad_c_e_combined, decimal=6
        )

        grad_phi_e_n = variables["Test grad phi_e_n"]
        grad_phi_e_s = variables["Test grad phi_e_s"]
        grad_phi_e_p = variables["Test grad phi_e_p"]
        grad_phi_e = variables["Test grad phi_e"]

        grad_phi_e_combined = np.concatenate(
            (grad_phi_e_n(t, x_n), grad_phi_e_s(t, x_s), grad_phi_e_p(t, x_p)), axis=0
        )

        np.testing.assert_array_almost_equal(
            grad_phi_e(t, x), grad_phi_e_combined, decimal=6
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
