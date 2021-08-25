import pybamm
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "../figures/"


def lico2_volume_change_Ai2020(sto):
    omega = pybamm.Parameter("Positive electrode partial molar volume [m3.mol-1]")
    c_p_max = pybamm.Parameter("Maximum concentration in positive electrode [mol.m-3]")
    t_change = omega * c_p_max * sto
    return t_change


def graphite_volume_change_Ai2020(sto):
    p1 = 145.907
    p2 = -681.229
    p3 = 1334.442
    p4 = -1415.710
    p5 = 873.906
    p6 = -312.528
    p7 = 60.641
    p8 = -5.706
    p9 = 0.386
    p10 = -4.966e-05
    t_change = (
        p1 * sto ** 9
        + p2 * sto ** 8
        + p3 * sto ** 7
        + p4 * sto ** 6
        + p5 * sto ** 5
        + p6 * sto ** 4
        + p7 * sto ** 3
        + p8 * sto ** 2
        + p9 * sto
        + p10
    )
    return t_change


def get_parameter_values():
    parameter_values = pybamm.ParameterValues(
        chemistry=pybamm.parameter_sets.Mohtat2020
    )
    parameter_values.update(
        {
            # mechanical properties
            "Positive electrode Poisson's ratio": 0.3,
            "Positive electrode Young's modulus [Pa]": 375e9,
            "Positive electrode reference concentration for free of deformation [mol.m-3]": 0,
            "Positive electrode partial molar volume [m3.mol-1]": -7.28e-7,
            "Positive electrode volume change": lico2_volume_change_Ai2020,
            # Loss of active materials (LAM) model
            "Positive electrode LAM constant exponential term": 2,
            "Positive electrode critical stress [Pa]": 375e6,
            # mechanical properties
            "Negative electrode Poisson's ratio": 0.2,
            "Negative electrode Young's modulus [Pa]": 15e9,
            "Negative electrode reference concentration for free of deformation [mol.m-3]": 0,
            "Negative electrode partial molar volume [m3.mol-1]": 3.1e-6,
            "Negative electrode volume change": graphite_volume_change_Ai2020,
            # Loss of active materials (LAM) model
            "Negative electrode LAM constant exponential term": 2,
            "Negative electrode critical stress [Pa]": 60e6,
            # Other
            "Cell thermal expansion coefficient [m.K-1]": 1.48e-6,
        },
        check_already_exists=False,
    )
    return parameter_values


spm = pybamm.lithium_ion.SPM(
    {
        "SEI": "ec reaction limited",
        "loss of active material": "stress-driven",
    }
)

parameter_values = get_parameter_values()
parameter_values.update(
    {
        "SEI kinetic rate constant [m.s-1]": 1.3e-14,
        "Positive electrode LAM constant propotional term": 5e-5,
        "Negative electrode LAM constant propotional term": 5e-5,
        "EC diffusivity [m2.s-1]": 5e-20,
    },
    check_already_exists=False,
)
esoh_model = pybamm.lithium_ion.ElectrodeSOH()
esoh_sim = pybamm.Simulation(esoh_model, parameter_values=parameter_values)
param = spm.param

Vmin = 3.0
Vmax = 4.2
Cn = parameter_values.evaluate(param.C_n_init)
Cp = parameter_values.evaluate(param.C_p_init)
n_Li_init = parameter_values.evaluate(param.n_Li_particles_init)
c_n_max = parameter_values.evaluate(param.c_n_max)
c_p_max = parameter_values.evaluate(param.c_p_max)

esoh_sol = esoh_sim.solve(
    [0],
    inputs={"V_min": Vmin, "V_max": Vmax, "C_n": Cn, "C_p": Cp, "n_Li": n_Li_init},
)
# print(esoh_sol["x_100"].data[0])
# print(esoh_sol["y_100"].data[0])

eps_n = parameter_values["Negative electrode active material volume fraction"]
eps_p = parameter_values["Positive electrode active material volume fraction"]
C_over_eps_n = Cn / eps_n
C_over_eps_p = Cp / eps_p


def split_long_string(title, max_words=None):
    """Get title in a nice format"""
    max_words = max_words or pybamm.settings.max_words_in_line
    words = title.split()
    # Don't split if fits on one line, don't split just for units
    if len(words) <= max_words or words[max_words].startswith("["):
        return title
    else:
        first_line = (" ").join(words[:max_words])
        second_line = (" ").join(words[max_words:])
        return first_line + "\n" + second_line


pybamm.set_logging_level("NOTICE")
experiment = pybamm.Experiment(
    [
        (
            f"Discharge at 1C until {Vmin}V",
            "Rest for 1 hour",
            f"Charge at 1C until {Vmax}V",
            f"Hold at {Vmax}V until C/50",
        )
    ]
    * 1200,
    termination="60% capacity",
    cccv_handling="ode",
)

sim = pybamm.Simulation(
    spm,
    experiment=experiment,
    parameter_values=parameter_values,
    # solver=pybamm.CasadiSolver("fast with events"),
    solver=pybamm.ScipySolver(),
)

sol = pybamm.load("about_to_fail.sav")
sim.solve(initial_soc=1, starting_solution=sol)

# from scipy.integrate import solve_ivp


# def cycle_adaptive_simulation(model, parameter_values, experiment, save_at_cycles=None):
#     experiment_one_cycle = pybamm.Experiment(
#         experiment.operating_conditions_cycles[:1],
#         termination=experiment.termination_string,
#     )

#     esoh_model = pybamm.lithium_ion.ElectrodeSOH()
#     esoh_sim = pybamm.Simulation(esoh_model, parameter_values=parameter_values)
#     param = model.param

#     esoh_sol = esoh_sim.solve(
#         [0],
#         inputs={"V_min": Vmin, "V_max": Vmax, "C_n": Cn, "C_p": Cp, "n_Li": n_Li_init},
#         solver=pybamm.AlgebraicSolver(),
#     )

#     parameter_values.update(
#         {
#             "Initial concentration in negative electrode [mol.m-3]": esoh_sol[
#                 "x_100"
#             ].data[0]
#             * c_n_max,
#             "Initial concentration in positive electrode [mol.m-3]": esoh_sol[
#                 "y_100"
#             ].data[0]
#             * c_p_max,
#         }
#     )

#     sim_ode = pybamm.Simulation(
#         model, experiment=experiment_one_cycle, parameter_values=parameter_values
#     )
#     sol0 = sim_ode.solve()
#     model = sim_ode.solution.all_models[0]
#     cap0 = sol0.summary_variables["Capacity [A.h]"]

#     def sol_to_y(sol, loc="end"):
#         if loc == "start":
#             pos = 0
#         elif loc == "end":
#             pos = -1
#         model = sol.all_models[0]
#         n_Li = sol["Total lithium in particles [mol]"].data[pos].flatten()
#         Cn = sol["Negative electrode capacity [A.h]"].data[pos].flatten()
#         Cp = sol["Positive electrode capacity [A.h]"].data[pos].flatten()
#         # y = np.concatenate([n_Li, Cn, Cp])
#         y = n_Li
#         for var in model.initial_conditions:
#             if var.name not in [
#                 "X-averaged negative particle concentration",
#                 "X-averaged positive particle concentration",
#                 "Discharge capacity [A.h]",
#             ]:
#                 value = sol[var.name].data
#                 if value.ndim == 1:
#                     value = value[pos]
#                 elif value.ndim == 2:
#                     value = value[:, pos]
#                 elif value.ndim == 3:
#                     value = value[:, :, pos]
#                 y = np.concatenate([y, value.flatten()])
#         return y

#     def y_to_sol(y, esoh_sim, model):
#         n_Li = y[0]
#         Cn = C_over_eps_n * y[1]
#         Cp = C_over_eps_p * y[2]

#         # print({"V_min": Vmin, "V_max": Vmax, "C_n": Cn, "C_p": Cp, "n_Li": n_Li})
#         # print(y)
#         # print(y[1], C_over_eps_n * y[3])
#         # print(y[2], C_over_eps_p * y[4])
#         print(Cn, Cp)

#         esoh_sol = esoh_sim.solve(
#             [0],
#             inputs={"V_min": Vmin, "V_max": Vmax, "C_n": Cn, "C_p": Cp, "n_Li": n_Li},
#         )
#         esoh_sim.built_model.set_initial_conditions_from(esoh_sol)
#         ics = {}
#         x_100 = esoh_sol["x_100"].data[0]
#         y_100 = esoh_sol["y_100"].data[0]
#         start = 1
#         for var in model.initial_conditions:
#             if var.name == "X-averaged negative particle concentration":
#                 ics[var.name] = x_100 * np.ones((model.variables[var.name].size, 2))
#             elif var.name == "X-averaged positive particle concentration":
#                 ics[var.name] = y_100 * np.ones((model.variables[var.name].size, 2))
#             elif var.name == "Discharge capacity [A.h]":
#                 ics[var.name] = np.zeros(1)
#             else:
#                 end = start + model.variables[var.name].size
#                 ics[var.name] = y[start:end, np.newaxis]
#                 start = end
#         model.set_initial_conditions_from(ics)
#         return pybamm.Solution(
#             [np.array([0])],
#             model.concatenated_initial_conditions.evaluate()[:, np.newaxis],
#             model,
#             {},
#         )

#     def dydt(t, y):
#         if y[1] < 0 or y[2] < 0:
#             return 0 * y
#         print(t)
#         # print(t,y)
#         # Set up based on current value of y
#         y_to_sol(
#             y,
#             esoh_sim,
#             sim_ode.op_conds_to_built_models[
#                 experiment_one_cycle.operating_conditions[0][:2]
#             ],
#         )

#         # Simulate one cycle
#         sol = sim_ode.solve()

#         dy = sol_to_y(sol) - y
#         # print(y[1:3])
#         #         print(dy)

#         return dy

#     if experiment.termination == {}:
#         event = None
#     else:

#         def capacity_cutoff(t, y):
#             sol = y_to_sol(y, esoh_sim, model)
#             cap = pybamm.make_cycle_solution([sol], esoh_sim, True)[1]["Capacity [A.h]"]
#             #             print(cap / cap0)
#             return cap / cap0 - experiment_one_cycle.termination["capacity"][0] / 100

#         capacity_cutoff.terminal = True

#         def eps_n(t, y):
#             return y[1].flatten()

#         eps_n.terminal = True

#         def eps_p(t, y):
#             return y[2].flatten()

#         eps_p.terminal = True

#         def events(t, y):
#             return np.concatenate([capacity_cutoff(t, y), eps_n(t, y), eps_p(t, y)])

#         events.terminal = True
#     num_cycles = len(experiment.operating_conditions_cycles)
#     if save_at_cycles is None:
#         t_eval = np.arange(1, num_cycles + 1)
#     elif save_at_cycles == -1:
#         t_eval = None
#     else:
#         t_eval = np.arange(1, num_cycles + 1, save_at_cycles)
#     y0 = sol_to_y(sol0, loc="start")
#     sol = solve_ivp(
#         dydt,
#         [1, num_cycles],
#         y0,
#         t_eval=t_eval,
#         # events=events,
#         events=capacity_cutoff,
#         first_step=10,
#         method="RK23",
#         atol=1e-2,
#         rtol=1e-2,
#     )

#     all_sumvars = []
#     for idx in range(sol.y.shape[1]):
#         fullsol = y_to_sol(sol.y[:, idx], esoh_sim, model)
#         sumvars = pybamm.make_cycle_solution([fullsol], esoh_sim, True)[1]
#         all_sumvars.append(sumvars)

#     all_sumvars_dict = {
#         key: np.array([sumvars[key] for sumvars in all_sumvars])
#         for key in all_sumvars[0].keys()
#     }
#     all_sumvars_dict["Cycle number"] = sol.t
#     return all_sumvars_dict

# all_sumvars_dict = cycle_adaptive_simulation(
#     spm, parameter_values, experiment, save_at_cycles=10
# )
# plot(all_sumvars_dict)
# def extrapolation_simulation(
#     model, parameter_values, experiment, n_cycles_step, modes=False
# ):
#     experiment_one_cycle = pybamm.Experiment(
#         experiment.operating_conditions_cycles[:1],
#         termination=experiment.termination_string,
#     )
#     num_cycles = len(experiment.operating_conditions_cycles)

#     esoh_model = pybamm.lithium_ion.ElectrodeSOH()
#     esoh_sim = pybamm.Simulation(esoh_model, parameter_values=parameter_values)
#     param = model.param

#     Cn = parameter_values.evaluate(param.C_n_init)
#     Cp = parameter_values.evaluate(param.C_p_init)
#     n_Li_init = parameter_values.evaluate(param.n_Li_particles_init)

#     esoh_sol = esoh_sim.solve(
#         [0],
#         inputs={"V_min": Vmin, "V_max": Vmax, "C_n": Cn, "C_p": Cp, "n_Li": n_Li_init},
#         solver=pybamm.AlgebraicSolver(),
#     )
#     cap0 = esoh_sol["C"].data[0]
#     cap_cutoff = experiment.termination["capacity"][0] / 100 * cap0

#     parameter_values.update(
#         {
#             "Initial concentration in negative electrode [mol.m-3]": esoh_sol[
#                 "x_100"
#             ].data[0]
#             * c_n_max,
#             "Initial concentration in positive electrode [mol.m-3]": esoh_sol[
#                 "y_100"
#             ].data[0]
#             * c_p_max,
#         }
#     )
#     sim_extrap = pybamm.Simulation(
#         spm, experiment=experiment_one_cycle, parameter_values=parameter_values
#     )
#     sol_extrap = []
#     cycle_nums = []
#     cycle = 1
#     while cycle <= num_cycles:
#         print(cycle)
#         # Simulate one cycle
#         sol = sim_extrap.solve()

#         n_Li_cycle = sol["Total lithium in particles [mol]"].data
#         n_Li_cycle_init = n_Li_cycle[0]
#         delta_nLi_cycle = n_Li_cycle[-1] - n_Li_cycle[0]
#         n_Li = n_Li_cycle_init + delta_nLi_cycle * n_cycles_step

#         Cn_cycle = sol["Negative electrode capacity [A.h]"].data
#         Cn_init = Cn_cycle[0]
#         delta_Cn_cycle = Cn_cycle[-1] - Cn_cycle[0]
#         Cn = Cn_init + delta_Cn_cycle * n_cycles_step

#         Cp_cycle = sol["Positive electrode capacity [A.h]"].data
#         Cp_init = Cp_cycle[0]
#         delta_Cp_cycle = Cp_cycle[-1] - Cp_cycle[0]
#         Cp = Cp_init + delta_Cp_cycle * n_cycles_step

#         esoh_sol = esoh_sim.solve(
#             [0],
#             inputs={"V_min": Vmin, "V_max": Vmax, "C_n": Cn, "C_p": Cp, "n_Li": n_Li},
#         )
#         esoh_sim.built_model.set_initial_conditions_from(esoh_sol)
#         x_100 = esoh_sol["x_100"].data[0]
#         y_100 = esoh_sol["y_100"].data[0]
#         if esoh_sol["C"].data[0] < cap_cutoff:
#             print("Capacity cut-off")
#             break

#         ics = {}
#         if sol is None:
#             break
#         for var in sol.all_models[0].initial_conditions:
#             if var.name == "Discharge capacity [A.h]":
#                 ics[var.name] = np.zeros(1)
#             elif (
#                 modes is True
#                 and var.name == "X-averaged negative particle concentration"
#             ):
#                 ics[var.name] = x_100 * np.ones(
#                     (sol.all_models[0].variables[var.name].size, 2)
#                 )
#             elif (
#                 modes is True
#                 and var.name == "X-averaged positive particle concentration"
#             ):
#                 ics[var.name] = y_100 * np.ones(
#                     (sol.all_models[0].variables[var.name].size, 2)
#                 )
#             else:
#                 first = sim_extrap.solution.first_state[var.name].data
#                 last = sim_extrap.solution.last_state[var.name].data
#                 ics[var.name] = first + (last - first) * n_cycles_step
#         #             ics[var.name] = sim_acc.solution[var.name].data
#         #         print(ics)
#         sim_extrap.op_conds_to_built_models[
#             experiment.operating_conditions[0][:2]
#         ].set_initial_conditions_from(ics)

#         cycle_nums.append(cycle)
#         sol_extrap.append(sol)
#         cycle += n_cycles_step

#     #     return sol_extrap
#     all_sumvars = []
#     for sol in sol_extrap:
#         sumvars = sol.summary_variables
#         all_sumvars.append(sumvars)

#     all_sumvars_dict = {
#         key: np.array([sumvars[key] for sumvars in all_sumvars])
#         for key in all_sumvars[0].keys()
#     }
#     all_sumvars_dict["Cycle number"] = cycle_nums
#     return all_sumvars_dict


# extrapolation_simulation(
#     spm, parameter_values, experiment, n_cycles_step=30, modes=False
# )
