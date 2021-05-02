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
            chemistry=pybamm.parameter_sets.Marquis2019
        )
        sim = pybamm.Simulation(model, parameter_values=parameter_values)

        V_min = 3
        V_max = 4.2
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
        self.assertAlmostEqual(sol["Up(y_100) - Un(x_100)"].data[0], V_max, places=5)
        self.assertAlmostEqual(sol["Up(y_0) - Un(x_0)"].data[0], V_min, places=5)
        self.assertAlmostEqual(sol["n_Li_100"].data[0], n_Li, places=5)
        self.assertAlmostEqual(sol["n_Li_0"].data[0], n_Li, places=5)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
