#
# Tests for the electrode-electrolyte interface equations
#
import pybamm
from tests import get_discretisation_for_testing

import unittest
import numpy as np


class TestOldHomogeneousReaction(unittest.TestCase):
    def test_set_parameters(self):
        param = pybamm.standard_parameters_lithium_ion
        current = param.current_with_time
        model = pybamm.old_interface.OldInterfacialReaction(param)
        j_n = model.get_homogeneous_interfacial_current(current, ["negative electrode"])
        j_p = model.get_homogeneous_interfacial_current(current, ["positive electrode"])
        parameter_values = model.default_parameter_values

        j_n = parameter_values.process_symbol(j_n)
        j_p = parameter_values.process_symbol(j_p)

        self.assertFalse(
            any([isinstance(x, pybamm.Parameter) for x in j_n.pre_order()])
        )
        self.assertFalse(
            any([isinstance(x, pybamm.Parameter) for x in j_p.pre_order()])
        )
        self.assertEqual(j_n.domain, [])
        self.assertEqual(j_p.domain, [])

    def test_discretisation(self):
        disc = get_discretisation_for_testing()

        param = pybamm.standard_parameters_lithium_ion
        current = param.current_with_time
        model = pybamm.old_interface.OldInterfacialReaction(param)
        j_n = model.get_homogeneous_interfacial_current(current, ["negative electrode"])
        j_p = model.get_homogeneous_interfacial_current(current, ["positive electrode"])
        parameter_values = model.default_parameter_values

        j_n = disc.process_symbol(parameter_values.process_symbol(j_n))
        j_p = disc.process_symbol(parameter_values.process_symbol(j_p))

        # test values
        l_n = parameter_values.process_symbol(param.l_n)
        l_p = parameter_values.process_symbol(param.l_p)
        np.testing.assert_array_equal((l_n * j_n).evaluate(0, None), 1)
        np.testing.assert_array_equal((l_p * j_p).evaluate(0, None), -1)

    def test_simplify_constant_current(self):
        disc = get_discretisation_for_testing()
        mesh = disc.mesh

        param = pybamm.standard_parameters_lithium_ion
        current = param.current_with_time
        model = pybamm.old_interface.OldInterfacialReaction(param)
        j_n = model.get_homogeneous_interfacial_current(current, ["negative electrode"])
        j_p = model.get_homogeneous_interfacial_current(current, ["positive electrode"])
        parameter_values = model.default_parameter_values
        j = pybamm.Concatenation(
            pybamm.Broadcast(j_n, ["negative electrode"]),
            pybamm.Broadcast(0, ["separator"]),
            pybamm.Broadcast(j_p, ["positive electrode"]),
        )

        j = disc.process_symbol(parameter_values.process_symbol(j))

        # Simplifiy, since current is constant this should give a vector
        j_simp = j.simplify()
        self.assertIsInstance(j_simp, pybamm.Vector)
        # test values
        l_n = parameter_values.process_symbol(param.l_n)
        l_p = parameter_values.process_symbol(param.l_p)
        npts_n = mesh["negative electrode"][0].npts
        npts_s = mesh["separator"][0].npts
        np.testing.assert_array_equal((l_n * j_simp).evaluate(0, None)[:npts_n], 1)
        np.testing.assert_array_equal(
            j_simp.evaluate(0, None)[npts_n : npts_n + npts_s], 0
        )
        np.testing.assert_array_equal(
            (l_p * j_simp).evaluate(0, None)[npts_n + npts_s :], -1
        )

    def test_failure(self):
        model = pybamm.old_interface.OldInterfacialReaction(None)
        with self.assertRaises(pybamm.DomainError):
            model.get_homogeneous_interfacial_current(None, "not a domain")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
