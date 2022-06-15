#
# Tests for the base model class
#
import platform
import unittest
import pybamm

have_julia = True  # pybamm.have_julia()
if have_julia and platform.system() != "Windows":
    from julia.api import Julia

    Julia(compiled_modules=False)
    from julia import Main

    # load julia libraries required for evaluating the strings
    Main.eval("using SparseArrays, LinearAlgebra")


@unittest.skipIf(not have_julia, "Julia not installed")
class TestBaseModelGenerateJuliaDiffEq(unittest.TestCase):
    def test_generate_ode(self):
        # ODE model with no input parameters
        model = pybamm.BaseModel(name="ode test model")
        a = pybamm.Variable("a")
        b = pybamm.Variable("b")
        model.rhs = {a: -a, b: a - b}
        model.initial_conditions = {a: 1, b: 2}

        # Generate rhs and ics for the Julia model
        rhs_str, ics_str = model.generate_julia_diffeq()
        self.assertIsInstance(rhs_str, str)
        self.assertIn("ode_test_model", rhs_str)
        self.assertIn("(dy, y, p, t)", rhs_str)
        self.assertIsInstance(ics_str, str)
        self.assertIn("ode_test_model_u0", ics_str)
        self.assertIn("(u0, p)", ics_str)

        # ODE model with input parameters
        model = pybamm.BaseModel(name="ode test model 2")
        a = pybamm.Variable("a")
        b = pybamm.Variable("b")
        p = pybamm.InputParameter("p")
        q = pybamm.InputParameter("q")
        model.rhs = {a: -a * p, b: a - b}
        model.initial_conditions = {a: q, b: 2}

        # Generate rhs and ics for the Julia model
        rhs_str, ics_str = model.generate_julia_diffeq(input_parameter_order=["p", "q"])
        self.assertIsInstance(rhs_str, str)
        self.assertIn("ode_test_model_2", rhs_str)
        self.assertIn("p, q = p", rhs_str)

        self.assertIsInstance(ics_str, str)
        self.assertIn("ode_test_model_2_u0", ics_str)
        self.assertIn("p, q = p", ics_str)

    def test_generate_dae(self):
        # ODE model with no input parameters
        model = pybamm.BaseModel(name="dae test model")
        a = pybamm.Variable("a")
        b = pybamm.Variable("b")
        model.rhs = {a: -a}
        model.algebraic = {b: a - b}
        model.initial_conditions = {a: 1, b: 2}

        # Generate eqn and ics for the Julia model (semi-explicit)
        eqn_str, ics_str = model.generate_julia_diffeq()
        self.assertIsInstance(eqn_str, str)
        self.assertIn("dae_test_model", eqn_str)
        self.assertIn("(dy, y, p, t)", eqn_str)
        self.assertIsInstance(ics_str, str)
        self.assertIn("dae_test_model_u0", ics_str)
        self.assertIn("(u0, p)", ics_str)
        self.assertIn("[1.,2.]", ics_str)

        # Generate eqn and ics for the Julia model (implicit)
        eqn_str, ics_str = model.generate_julia_diffeq(dae_type="implicit")
        self.assertIsInstance(eqn_str, str)
        self.assertIn("dae_test_model", eqn_str)
        self.assertIn("(out, dy, y, p, t)", eqn_str)
        self.assertIsInstance(ics_str, str)
        self.assertIn("dae_test_model_u0", ics_str)
        self.assertIn("(u0, p)", ics_str)

        # Calculate initial conditions in python
        eqn_str, ics_str = model.generate_julia_diffeq(
            get_consistent_ics_solver=pybamm.CasadiSolver()
        )
        # Check that the initial conditions are consistent
        self.assertIn("[1.,1.]", ics_str)

    def test_generate_pde(self):
        # ODE model with no input parameters
        model = pybamm.BaseModel(name="pde test model")
        a = pybamm.Variable("a", domain="line")
        b = pybamm.Variable("b", domain="line")
        model.rhs = {a: pybamm.div(pybamm.grad(a)) + b, b: a - b}
        model.boundary_conditions = {
            a: {"left": (-1, "Dirichlet"), "right": (1, "Neumann")}
        }
        model.initial_conditions = {a: 1, b: 2}

        # Discretize
        x = pybamm.SpatialVariable("x", domain=["line"])
        geometry = pybamm.Geometry(
            {"line": {x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}}
        )
        submesh_types = {"line": pybamm.Uniform1DSubMesh}
        var_pts = {x: 10}
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
        disc = pybamm.Discretisation(mesh, {"line": pybamm.FiniteVolume()})
        disc.process_model(model)

        # Generate rhs and ics for the Julia model
        rhs_str, ics_str = model.generate_julia_diffeq()
        self.assertIsInstance(rhs_str, str)
        self.assertIn("pde_test_model", rhs_str)
        self.assertIn("(dy, y, p, t)", rhs_str)
        self.assertIsInstance(ics_str, str)
        self.assertIn("pde_test_model_u0", ics_str)
        self.assertIn("(u0, p)", ics_str)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
