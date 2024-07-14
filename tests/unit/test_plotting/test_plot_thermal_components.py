import pytest
import pybamm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import use

use("Agg")


class TestPlotThermalComponents:
    def test_plot_with_solution(self):
        model = pybamm.lithium_ion.SPM({"thermal": "lumped"})
        sim = pybamm.Simulation(
            model, parameter_values=pybamm.ParameterValues("ORegan2022")
        )
        sol = sim.solve([0, 3600])
        for input_data in [sim, sol]:
            _, ax = pybamm.plot_thermal_components(input_data, show_plot=False)
            t, cumul_heat = ax[1].get_lines()[-1].get_data()
            np.testing.assert_array_almost_equal(t, sol["Time [h]"].data)
            t, cumul_heat = ax[1].get_lines()[-1].get_data()
            T_sol = sol["X-averaged cell temperature [K]"].data
            np.testing.assert_array_almost_equal(t, sol["Time [h]"].data)
            rho_c_p_eff = sol[
                "Volume-averaged effective heat capacity [J.K-1.m-3]"
            ].data
            T_plot = T_sol[0] + cumul_heat / rho_c_p_eff
            np.testing.assert_allclose(T_sol, T_plot, rtol=1e-2)

            _, ax = plt.subplots(1, 2)
            _, ax_out = pybamm.plot_thermal_components(sol, ax=ax, show_legend=True)
            assert ax_out[0] == ax[0]
            assert ax_out[1] == ax[1]

    def test_not_implemented(self):
        model = pybamm.lithium_ion.SPM({"thermal": "x-full"})
        sim = pybamm.Simulation(model)
        sol = sim.solve([0, 3600])
        with pytest.raises(NotImplementedError):
            pybamm.plot_thermal_components(sol)
