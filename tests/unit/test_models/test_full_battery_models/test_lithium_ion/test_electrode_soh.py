#
# Tests for the lithium-ion electrode-specific SOH model
#
import pybamm
import unittest


class TestElectrodeSOH(unittest.TestCase):
    def test_known_solution(self):
        model = pybamm.lithium_ion.ElectrodeSOH()

        param = pybamm.LithiumIonParameters()
        parameter_values = pybamm.ParameterValues(
            chemistry=pybamm.parameter_sets.Mohtat2020
        )
        sim = pybamm.Simulation(model, parameter_values=parameter_values)

        V_min = 3
        V_max = 4.2
        param
        C_n = parameter_values.evaluate(param.C_n_init)
        C_p = parameter_values.evaluate(param.C_p_init)
        n_Li = parameter_values.evaluate(param.n_Li_particles_init)

        # Solve the model and check outputs
        sol = sim.solve(
            [0],
            inputs={
                "V_min": V_min,
                "V_max": V_max,
                "C_n": C_n,
                "C_p": C_p,
                "n_Li": n_Li,
            },
        )
        x_0 = sol["x_0"].data
        y_0 = sol["y_0"].data
        F = parameter_values.evaluate(param.F)
        self.assertAlmostEqual(sol["Up(y_100) - Un(x_100)"].data[0], V_max)
        self.assertAlmostEqual(sol["Up(y_0) - Un(x_0)"].data[0], V_min)
        self.assertEqual(sol["n_Li"].data, n_Li)
        self.assertEqual(3600 / F * (x_0 * C_n + y_0 * C_p), n_Li)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
