import pybamm
import os
import sys
import pickle
import matplotlib.pyplot as plt
import shared

# change working directory to the root of pybamm
os.chdir(pybamm.root_dir())

# increase recursion limit for large expression trees
sys.setrecursionlimit(100000)

pybamm.set_logging_level("INFO")


"-----------------------------------------------------------------------------"
"Load comsol data"

try:
    comsol_variables = pickle.load(
        open("input/comsol_results/comsol_thermal_1plus1D_1C.pickle", "rb")
    )
except FileNotFoundError:
    raise FileNotFoundError("COMSOL data not found. Try running load_comsol_data.py")

"-----------------------------------------------------------------------------"
"Load or set up pybamm simulation"

compute = True
filename = "results/2019_xx_1plus1D_pouch/pybamm_thermal_1plus1D_1C.pickle.pickle"

if compute is False:
    try:
        simulation = pybamm.load_sim(filename)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Run script with compute=True first to generate results"
        )
else:
    # model
    options = {
        "current collector": "potential pair",
        "dimensionality": 1,
        "thermal": "x-lumped",
    }
    pybamm_model = pybamm.lithium_ion.DFN(options)

    # parameters
    param = pybamm_model.default_parameter_values
    param.update({"C-rate": 1})

    # set npts
    var = pybamm.standard_spatial_vars
    var_pts = {
        var.x_n: 10,
        var.x_s: 5,
        var.x_p: 10,
        var.r_n: 15,
        var.r_p: 15,
        var.z: 25,
    }

    # solver
    solver = pybamm.CasadiSolver(
        atol=1e-6, rtol=1e-6, root_tol=1e-6, root_method="hybr", mode="fast"
    )

    # simulation object
    simulation = pybamm.Simulation(
        pybamm_model, parameter_values=param, var_pts=var_pts, solver=solver
    )

    # build and save simulation
    simulation.build(check_model=False)
    # simulation.save(filename)

"-----------------------------------------------------------------------------"
"Solve model if not already solved"

force_solve = False  # if True, then model is re-solved

# discharge timescale
tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)

# solve model at comsol times
t_eval = comsol_variables["time"] / tau

if force_solve is True:
    simulation.solve(t_eval=t_eval)
elif simulation._solution is None:
    simulation.solve(t_eval=t_eval)

"-----------------------------------------------------------------------------"
"Make Comsol 'model' for comparison"

mesh = simulation._mesh
comsol_model = shared.make_comsol_model(comsol_variables, mesh, param, thermal=True)


"-----------------------------------------------------------------------------"
"Make plots"

pybamm_model = simulation.built_model
solution = simulation._solution
plot = pybamm.QuickPlot(
    [pybamm_model, comsol_model],
    mesh,
    [solution, solution],
    output_variables=comsol_model.variables.keys(),
    labels=["PyBaMM", "Comsol"],
)
plot.dynamic_plot(testing=True)  # testing=True to suppress show

shared.plot_t_var(
    "Terminal voltage [V]",
    pybamm_model,
    comsol_model,
    mesh,
    solution,
    param,
    plot_error="both",
)
shared.plot_t_var(
    "Volume-averaged cell temperature [K]",
    pybamm_model,
    comsol_model,
    mesh,
    solution,
    param,
    plot_error="both",
)

plot_times = [600, 1200, 1800, 2400, 3000]  # dimensional in seconds, must be a list
shared.plot_cc_var(
    "Negative current collector potential [V]",
    pybamm_model,
    comsol_model,
    mesh,
    solution,
    param,
    plot_times=plot_times,
    plot_error="both",
    # scale=0.0001,  # typical variation in negative potential
    scale="auto",
)
shared.plot_cc_var(
    "Positive current collector potential [V]",
    pybamm_model,
    comsol_model,
    mesh,
    solution,
    param,
    plot_times=plot_times,
    plot_error="both",
    # scale=0.0001,  # typical variation in positive potential
    scale="auto",
)
shared.plot_cc_var(
    "Current collector current density [A.m-2]",
    pybamm_model,
    comsol_model,
    mesh,
    solution,
    param,
    plot_times=plot_times,
    plot_error="both",
    # scale=0.06,  # typical variation in current density
    scale="auto",
)
shared.plot_cc_var(
    "X-averaged cell temperature [K]",
    pybamm_model,
    comsol_model,
    mesh,
    solution,
    param,
    plot_times=plot_times,
    plot_error="both",
    # scale=0.1,  # typical variation in current density
    scale="auto",
)
plt.show()
