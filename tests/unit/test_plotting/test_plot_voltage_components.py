import pybamm
import unittest
import numpy as np
from tests import TestCase
import matplotlib.pyplot as plt


class TestPlotVoltageComponents(TestCase):
    def test_plot(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)
        sol = sim.solve([0, 3600])
        for split in [True, False]:
            _, ax = pybamm.plot_voltage_components(
                sol, testing=True, split_by_electrode=split
            )
            t, V = ax.get_lines()[0].get_data()
            np.testing.assert_array_equal(t, sol["Time [h]"].data)
            np.testing.assert_array_equal(V, sol["Battery voltage [V]"].data)

        _, ax = plt.subplots()
        _, ax_out = pybamm.plot_voltage_components(sol, ax=ax, show_legend=True)
        self.assertEqual(ax_out, ax)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
