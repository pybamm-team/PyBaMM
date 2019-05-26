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
        c_e_n = pybamm.standard_variables.c_e_n
        c_e_p = pybamm.standard_variables.c_e_p
        c_e = pybamm.standard_variables.c_e

        delta_phi = pybamm.Variable("delta phi", "negative electrode")
        model.set_full_system(
            delta_phi, c_e_n, {"main": {"neg": {"aj": pybamm.Scalar(1)}}}
        )

        delta_phi = pybamm.Variable("delta phi", "positive electrode")
        model.set_full_system(
            delta_phi, c_e_p, {"main": {"pos": {"aj": pybamm.Scalar(1)}}}
        )

        model.set_post_processed(c_e)

    def test_exceptions(self):
        param = pybamm.standard_parameters_lithium_ion
        model = pybamm.electrolyte_current.MacInnesCapacitance(param)
        c_e = pybamm.Variable("c_e")
        delta_phi = pybamm.Variable("delta phi", "not a domain")

        with self.assertRaises(pybamm.DomainError):
            model.set_full_system(
                delta_phi, c_e, {"main": {"pos": {"aj": pybamm.Scalar(1)}}}
            )

        with self.assertRaises(pybamm.DomainError):
            model.set_leading_order_system(
                delta_phi, {"main": {"pos": {"aj": pybamm.Scalar(1)}}}, "not a domain"
            )


@unittest.skipIf(scikits_odes_spec is None, "scikits.odes not installed")
class TestMacInnesCapacitance(unittest.TestCase):
    def test_default_solver(self):
        param = pybamm.standard_parameters_lithium_ion
        model = pybamm.electrolyte_current.MacInnesCapacitance(param, True)
        self.assertIsInstance(model.default_solver, pybamm.ScikitsOdeSolver)
        model = pybamm.electrolyte_current.MacInnesCapacitance(param, False)
        self.assertIsInstance(model.default_solver, pybamm.ScikitsDaeSolver)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
