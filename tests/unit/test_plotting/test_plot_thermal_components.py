import pybamm
import unittest
import numpy as np
from tests import TestCase
import matplotlib.pyplot as plt
from matplotlib import use

use("Agg")


class TestPlotThermalComponents(TestCase):
    def test_plot_with_solution(self):
        model = pybamm.lithium_ion.SPM({"thermal": "lumped"})
        sim = pybamm.Simulation(model)
        sol = sim.solve([0, 3600])
        for input_data in [sim, sol]:
            _, ax = pybamm.plot_thermal_components(input_data, show_plot=False)
            t, T = ax[0].get_lines()[-1].get_data()
            np.testing.assert_array_almost_equal(t, sol["Time [h]"].data)
            np.testing.assert_array_almost_equal(
                T, sol["X-averaged cell temperature [K]"].data
            )

            _, ax = plt.subplots(1, 2)
            _, ax_out = pybamm.plot_thermal_components(sol, ax=ax, show_legend=True)
            self.assertEqual(ax_out[0], ax[0])
            self.assertEqual(ax_out[1], ax[1])

    def test_not_implemented(self):
        model = pybamm.lithium_ion.SPM({"thermal": "x-full"})
        sim = pybamm.Simulation(model)
        sol = sim.solve([0, 3600])
        with self.assertRaises(NotImplementedError):
            pybamm.plot_thermal_components(sol)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
