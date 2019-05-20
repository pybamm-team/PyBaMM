#
# Tests for the electrolyte current submodel
#
import pybamm
import unittest


@unittest.skipIf(scikits_odes_spec is None, "scikits.odes not installed")
class TestMacInnesStefanMaxwell(unittest.TestCase):
    def setUp(self):
        # Parameters
        param = pybamm.standard_parameters_lithium_ion

        # Variables and reactions
        phi_e = pybamm.standard_variables.phi_e
        c_e = pybamm.standard_variables.c_e
        onen = pybamm.Broadcast(1, ["negative electrode"])
        onep = pybamm.Broadcast(1, ["positive electrode"])
        reactions = {
            "main": {"neg": {"s_plus": 1, "aj": onen}, "pos": {"s_plus": 1, "aj": onep}}
        }

        # Set up model
        self.model = pybamm.electrolyte_current.MacInnesStefanMaxwell(param)

    def test_default_solver(self):
        self.assertTrue(isinstance(self.model.default_solver, pybamm.ScikitsDaeSolver))


@unittest.skipIf(scikits_odes_spec is None, "scikits.odes not installed")
class TestMacInnesCapacitance(unittest.TestCase):
    def test_basic_processing(self):
        # Parameters
        param = pybamm.standard_parameters_lithium_ion

        # Variables
        delta_phi_n = pybamm.standard_variables.delta_phi_n
        delta_phi_p = pybamm.standard_variables.delta_phi_p
        c_e_n = pybamm.standard_variables.c_e_n
        c_e_p = pybamm.standard_variables.c_e_p
        c_s_n_surf = pybamm.Scalar(0.8)
        c_s_p_surf = pybamm.Scalar(0.8)

        # Exchange-current density
        neg = ["negative electrode"]
        pos = ["positive electrode"]
        int_curr_model = pybamm.interface.LithiumIonReaction(param)
        j0_n = int_curr_model.get_exchange_current_densities(c_e_n, c_s_n_surf, neg)
        j0_p = int_curr_model.get_exchange_current_densities(c_e_p, c_s_p_surf, pos)

        # Open-circuit potential and reaction overpotential
        ocp_n = param.U_n(c_s_n_surf)
        ocp_p = param.U_p(c_s_p_surf)
        eta_r_n = delta_phi_n - ocp_n
        eta_r_p = delta_phi_p - ocp_p

        # Interfacial current density
        j_n = int_curr_model.get_butler_volmer(j0_n, eta_r_n, ["negative electrode"])
        j_p = int_curr_model.get_butler_volmer(j0_p, eta_r_p, ["positive electrode"])
        reactions = {"main": {"neg": {"aj": j_n}, "pos": {"aj": j_p}}}

        self.model_cap = pybamm.electrolyte_current.MacInnesCapacitance(param, True)
        self.model_no_cap = pybamm.electrolyte_current.MacInnesCapacitance(param, False)

    def test_default_solver(self):
        self.assertTrue(isinstance(
            self.model_cap.default_solver, pybamm.ScikitsOdeSolver))
        self.assertTrue(isinstance(
            self.model_no_cap.default_solver, pybamm.ScikitsDaeSolver))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
