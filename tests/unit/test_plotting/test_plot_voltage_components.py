import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import use

import pybamm

use("Agg")


@pytest.fixture
def solved_simulations():
    model = pybamm.lithium_ion.SPM()
    sim = pybamm.Simulation(model)
    sol = sim.solve([0, 3600])

    model_composite = pybamm.lithium_ion.SPM({"particle phases": ("2", "1")})
    params = pybamm.ParameterValues("Chen2020_composite")
    sim_composite = pybamm.Simulation(model_composite, parameter_values=params)
    sol_composite = sim_composite.solve([0, 3600])
    return sim, sol, sim_composite, sol_composite


@pytest.mark.parametrize(
    "split_by_electrode",
    [True, False],
    ids=["with_split_by_electrode", "with_no_split_by_electrode"],
)
@pytest.mark.parametrize(
    "from_solution", [True, False], ids=["from_solution", "from_simulation"]
)
@pytest.mark.parametrize(
    "anode",
    ["primary", "secondary", "single"],
    ids=["composite_anode_primary", "composite_anode_secondary", "single_phase_anode"],
)
def test_plot_voltage_components(
    solved_simulations, from_solution, split_by_electrode, anode
):
    sim, sol, sim_composite, sol_composite = solved_simulations
    if anode != "single":
        sim, sol = (sim_composite, sol_composite)
    target = sol if from_solution else sim

    _, ax = target.plot_voltage_components(
        # If anode not composite then the string value does not matter
        show_plot=False,
        split_by_electrode=split_by_electrode,
        electrode_phases=(anode, "primary"),
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

    with pytest.raises(ValueError, match=r"The simulation has not been solved yet."):
        sim.plot_voltage_components()
