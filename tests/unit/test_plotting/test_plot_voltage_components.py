#
# Tests for the voltage plot components functions
#

import pytest
import pybamm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import use

use("Agg")


@pytest.fixture
def solved_simulation():
    model = pybamm.lithium_ion.SPM()
    sim = pybamm.Simulation(model)
    sol = sim.solve([0, 3600])
    return sim, sol


@pytest.mark.parametrize(
    "from_solution", [True, False], ids=["from_solution", "from_simulation"]
)
@pytest.mark.parametrize("split_by_electrode", [True, False], ids=["split", "no_split"])
def test_plot_voltage_components(solved_simulation, from_solution, split_by_electrode):
    sim, sol = solved_simulation
    target = sol if from_solution else sim

    _, ax = target.plot_voltage_components(
        show_plot=False, split_by_electrode=split_by_electrode
    )
    t, V = ax.get_lines()[0].get_data()
    np.testing.assert_array_equal(t, sol["Time [h]"].data)
    np.testing.assert_array_equal(V, sol["Battery voltage [V]"].data)

    _, ax = plt.subplots()
    _, ax_out = target.plot_voltage_components(ax=ax, show_legend=True)
    assert ax_out == ax


def test_plot_without_solution():
    model = pybamm.lithium_ion.SPM()
    sim = pybamm.Simulation(model)

    with pytest.raises(ValueError, match="The simulation has not been solved yet."):
        sim.plot_voltage_components()
