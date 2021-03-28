import pybamm
import matplotlib.pyplot as plt
import numpy as np

pybamm.set_logging_level("NOTICE")

# parameter_values = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Chen2020)
# parameter_values.update(
#     {
#         "SEI kinetic rate constant [m.s-1]": 1e-15,
#         "SEI resistivity [Ohm.m]": 0,
#     }
# )
# spm = pybamm.lithium_ion.SPM()  # {"SEI": "ec reaction limited"})
spm = pybamm.lithium_ion.Yang2017()
param = spm.param


# experiment = pybamm.Experiment(
#     [
#         (
#             f"Discharge at 1C until {Vmin}V",
#             "Rest for 1 hour",
#             f"Charge at 1C until {Vmax}V",
#             f"Hold at {Vmax}V until C/50",
#         )
#     ]
#     * 200,
#     termination="80% capacity",
# )
# sim = pybamm.Simulation(
#     spm,
#     experiment=experiment,
#     parameter_values=parameter_values,
#     solver=pybamm.CasadiSolver("fast with events"),
# )
# sim.solve()

Vmin = 2.5
Vmax = 4.2
experiment = pybamm.Experiment(
    [
        (
            f"Discharge at 1C until {Vmin}V",
            "Rest for 1 hour",
            f"Charge at 1C until {Vmax}V",
            f"Hold at {Vmax}V until C/50",
        )
    ]
)

parameter_values = spm.default_parameter_values
esoh_model = pybamm.lithium_ion.ElectrodeSOH()
esoh_sim = pybamm.Simulation(esoh_model, parameter_values=parameter_values)

sim_acc = pybamm.Simulation(
    spm, experiment=experiment, solver=pybamm.CasadiSolver("fast with events")
)
sim_acc.build_for_experiment()

sol_acc = []
cycle_nums = []
cycle = 0
n_cycles_step = 20
Cn = parameter_values.evaluate(param.C_n_init)
Cp = parameter_values.evaluate(param.C_p_init)

while cycle < 101:
    print(cycle)
    # Simulate one cycle
    sol = sim_acc.solve()

    n_Li_cycle = sol["Total lithium in particles [mol]"].data
    n_Li_cycle_init = n_Li_cycle[0]
    delta_nLi_cycle = n_Li_cycle[-1] - n_Li_cycle[0]

    n_Li = n_Li_cycle_init + delta_nLi_cycle * n_cycles_step

    esoh_sol = esoh_sim.solve(
        [0],
        inputs={"V_min": Vmin, "V_max": Vmax, "C_n": Cn, "C_p": Cp, "n_Li": n_Li},
    )
    esoh_sim.built_model.set_initial_conditions_from(esoh_sol)
    ics = {}
    x_100 = esoh_sol["x_100"].data[0]
    y_100 = esoh_sol["y_100"].data[0]
    for var in sol.all_models[0].initial_conditions:
        if var.name == "X-averaged negative particle concentration":
            ics[var.name] = x_100 * np.ones(
                (sol.all_models[0].variables[var.name].size, 2)
            )
        elif var.name == "X-averaged positive particle concentration":
            ics[var.name] = y_100 * np.ones(
                (sol.all_models[0].variables[var.name].size, 2)
            )
        else:
            first = sim_acc.solution.first_state[var.name].data
            last = sim_acc.solution.last_state[var.name].data
            ics[var.name] = first + (last - first) * n_cycles_step
    #             ics[var.name] = sim_acc.solution[var.name].data
    sim_acc.op_conds_to_built_models[
        experiment.operating_conditions[0][:2]
    ].set_initial_conditions_from(ics)

    cycle_nums.append(cycle)
    sol_acc.append(sol)
    cycle += n_cycles_step
