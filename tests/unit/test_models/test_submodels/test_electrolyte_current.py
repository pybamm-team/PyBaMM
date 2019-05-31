#
# Tests for the electrolyte current submodel
#
import pybamm
from pybamm.solvers.scikits_ode_solver import scikits_odes_spec
import unittest


@unittest.skipIf(scikits_odes_spec is None, "scikits.odes not installed")
class TestMacInnesStefanMaxwell(unittest.TestCase):
    def test_default_solver(self):
        param = pybamm.standard_parameters_lithium_ion
        model = pybamm.electrolyte_current.MacInnesStefanMaxwell(param)
        self.assertIsInstance(model.default_solver, pybamm.ScikitsDaeSolver)

    def test_use_epsilon_parameter(self):
        param = pybamm.standard_parameters_lithium_ion
        model = pybamm.electrolyte_current.MacInnesCapacitance(param)

        c_e = pybamm.standard_variables.c_e
        i_boundary_cc = param.current_with_time
        delta_phi_n = pybamm.Variable("delta phi", "negative electrode")
        delta_phi_p = pybamm.Variable("delta phi", "positive electrode")

        variables = {
            "Negative electrode surface potential difference": delta_phi_n,
            "Positive electrode surface potential difference": delta_phi_p,
            "Electrolyte concentration": c_e,
            "Current collector current density": i_boundary_cc,
        }
        reactions = {
            "main": {"neg": {"aj": pybamm.Scalar(1)}, "pos": {"aj": pybamm.Scalar(1)}}
        }

        model.set_full_system(variables, reactions, ["negative electrode"])
        model.set_full_system(variables, reactions, ["positive electrode"])

        model.set_post_processed()

    def test_exceptions(self):
        param = pybamm.standard_parameters_lithium_ion
        model = pybamm.electrolyte_current.MacInnesCapacitance(param)

        c_e = pybamm.standard_variables.c_e
        i_boundary_cc = param.current_with_time
        delta_phi_n = pybamm.Variable("delta phi", "negative electrode")
        delta_phi_p = pybamm.Variable("delta phi", "positive electrode")

        variables = {
            "Negative electrode surface potential difference": delta_phi_n,
            "Positive electrode surface potential difference": delta_phi_p,
            "Electrolyte concentration": c_e,
            "Current collector current density": i_boundary_cc,
        }
        reactions = {
            "main": {"neg": {"aj": pybamm.Scalar(1)}, "pos": {"aj": pybamm.Scalar(1)}}
        }

        with self.assertRaises(pybamm.DomainError):
            model.set_full_system(variables, reactions, ["not a domain"])

        with self.assertRaises(pybamm.DomainError):
            model.set_leading_order_system(variables, reactions, ["not a domain"])


@unittest.skipIf(scikits_odes_spec is None, "scikits.odes not installed")
class TestMacInnesCapacitance(unittest.TestCase):
    def test_default_solver(self):
        param = pybamm.standard_parameters_lithium_ion
        model = pybamm.electrolyte_current.MacInnesCapacitance(param, "differential")
        self.assertIsInstance(model.default_solver, pybamm.ScikitsOdeSolver)
        model = pybamm.electrolyte_current.MacInnesCapacitance(param, "algebraic")
        self.assertIsInstance(model.default_solver, pybamm.ScikitsDaeSolver)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
