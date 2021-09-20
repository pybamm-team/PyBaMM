import pybamm
import pandas as pd

# setup -----------------------------------------------------------------------

# load 1+1D model (standard DFN)
dfn = pybamm.lithium_ion.DFN(
    options={
        "thermal": "lumped",
    },
    name="1+1D DFN",
)

# load 1+1+1D model
dfn_pouch = pybamm.lithium_ion.DFN(
    options={
        # "current collector": "uniform",  # no potential drop in current collector
        "current collector": "potential pair",
        "dimensionality": 1,
        "thermal": "x-lumped",
    },
    name="1+1+1D DFN",
)

# pick parameters
chemistry = pybamm.parameter_sets.Ecker2015
params = pybamm.ParameterValues(chemistry=chemistry)

# pick grid - 16 finite volumes per domain
var = pybamm.standard_spatial_vars
var_pts = {
    var.x_n: 16,
    var.x_s: 16,
    var.x_p: 16,
    var.r_n: 16,
    var.r_p: 16,
}

# pick solver
solver = pybamm.CasadiSolver(mode="fast", atol=1e-6, rtol=1e-6)

# lists to store times and number of states
set_up_times = []
solve_times = []
integration_times = []
states = []

# solve 1+1D model (standard DFN) ---------------------------------------------
sim = pybamm.Simulation(dfn, parameter_values=params, var_pts=var_pts, solver=solver)
print("Solving 1+1D DFN...")
sol = sim.solve([0, 3600])  # 1hr 1C discharge
set_up_times.append(sol.set_up_time.value)
solve_times.append(sol.solve_time.value)
integration_times.append(sol.integration_time.value)
states.append(sim.built_model.concatenated_initial_conditions.shape[0])
# print times
print(
    f"Set-up time: {sol.set_up_time},",
    f"Solve time: {sol.solve_time} "
    + f"(of which integration time: {sol.integration_time}),",
    f"Total time: {sol.total_time}",
)

# solve 1+1+1D model ----------------------------------------------------------

# number of finite volume in current colector domain (must be >2)
npts = [3, 4, 8]  # , 16, 32, 64, 128, 256, 512]

for N in npts:
    # update grid
    var_pts[var.z] = N

    # set up and solve simulation
    sim = pybamm.Simulation(
        dfn_pouch, parameter_values=params, var_pts=var_pts, solver=solver
    )
    print(f"Solving 1+1+1D DFN with {N} coupled models...")
    sol = sim.solve([0, 3600])  # 1hr 1C discharge
    set_up_times.append(sol.set_up_time.value)
    solve_times.append(sol.solve_time.value)
    integration_times.append(sol.integration_time.value)
    states.append(sim.built_model.concatenated_initial_conditions.shape[0])
    # print times
    print(
        f"Set-up time: {sol.set_up_time},",
        f"Solve time: {sol.solve_time} "
        + f"(of which integration time: {sol.integration_time}),",
        f"Total time: {sol.total_time}",
    )


# Save times etc. to csv ------------------------------------------------------
npts.insert(0, 1)  # add N =1 to list

# dictionary of lists
dict = {
    "N": npts,
    "Setup time": set_up_times,
    "Solve time": solve_times,
    "Integration time": integration_times,
    "States": states,
}

df = pd.DataFrame(dict)
df.to_csv("pouch_bench.csv", index=False)
