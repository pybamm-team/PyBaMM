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


def test_half_cell_voltage_components():
    # Guard: half-cell models must work with plot_voltage_components (7e4c5ffdf)
    model = pybamm.lithium_ion.SPM({"working electrode": "positive"})
    sim = pybamm.Simulation(model)
    sol = sim.solve([0, 3600])

    _fig, ax = sol.plot_voltage_components(show_plot=False)

    assert ax is not None
    assert len(ax.get_lines()) > 0

    pybamm.close_plots()


def test_time_not_starting_at_zero_with_experiment():
    # OCV baseline must use solution.t (seconds), not Time [h] (time[0] bug)
    model = pybamm.lithium_ion.SPM()
    experiment = pybamm.Experiment(
        [
            "Discharge at 1C for 5 minutes",
            "Rest for 1 minute",
            "Discharge at 0.5C for 5 minutes",
        ]
    )
    sim = pybamm.Simulation(model, experiment=experiment)
    sol = sim.solve()

    # Use a discharge cycle that starts at non-zero time to trigger the bug
    cycle_sol = sol.cycles[2]
    cycle_time_hours = cycle_sol["Time [h]"].entries
    cycle_time_seconds = cycle_sol.t

    assert cycle_time_hours[0] > 0, "Test requires solution starting at non-zero time"
    assert cycle_time_seconds[0] > 0, "Test requires solution.t starting at non-zero"

    def assert_fill_baseline(collection, expected_baseline, n_times):
        vertices = collection.get_paths()[0].vertices
        baseline = vertices[n_times + 1 : -1, 1]
        np.testing.assert_allclose(baseline, expected_baseline, rtol=1e-12)

    # Before fix: ocv(time[0]) where time[0] is hours would be wrong
    # After fix: ocv(solution.t[0]) uses seconds correctly
    _fig, ax = cycle_sol.plot_voltage_components(show_plot=False)

    assert ax is not None
    assert len(ax.get_lines()) > 0

    ocv_var = cycle_sol["Battery open-circuit voltage [V]"]
    correct_initial_ocv = ocv_var(cycle_time_seconds[0])
    fill_collections = [c for c in ax.collections if hasattr(c, "get_paths")]
    assert len(fill_collections) > 0, "Plot should have fill regions"
    assert_fill_baseline(
        fill_collections[0], correct_initial_ocv, len(cycle_time_hours)
    )

    pybamm.close_plots()

    _fig, ax = cycle_sol.plot_voltage_components(
        show_plot=False, split_by_electrode=True
    )

    num_cells = (
        cycle_sol["Battery voltage [V]"].entries[0]
        / cycle_sol["Voltage [V]"].entries[0]
    )
    ocp_n = cycle_sol["Negative electrode bulk open-circuit potential [V]"]
    ocp_p = cycle_sol["Positive electrode bulk open-circuit potential [V]"]
    correct_initial_ocv = (
        ocp_p(cycle_time_seconds[0]) - ocp_n(cycle_time_seconds[0])
    ) * num_cells

    fill_collections = [c for c in ax.collections if hasattr(c, "get_paths")]
    assert_fill_baseline(
        fill_collections[0], correct_initial_ocv, len(cycle_time_hours)
    )

    pybamm.close_plots()
