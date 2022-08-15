import pybamm
import numpy as np
import matplotlib.pyplot as plt
import itertools


parameters = [
    "Marquis2019",
    "Prada2013",
    "Ramadass2004",
    "Chen2020",
]
models = ["SPM", "DFN"]
abstols = [
    0.001,
    0.0001,
    1.0e-5,
    1.0e-6,
    1.0e-7,
    1.0e-8,
    1.0e-9,
    1.0e-10,
    1.0e-11,
    1.0e-12,
    1.0e-13,
]
solvers = [pybamm.IDAKLUSolver() , pybamm.CasadiSolver(), pybamm.CasadiSolver( mode="fast")]
nrow = 3
ncol = 2
fig, axs = plt.subplots(nrow, ncol)
for model_ in models:
    # for tol in abstols:
    # for solver_ in solvers:
        # solvers = [
        #     pybamm.IDAKLUSolver(atol=tol) , 
        #     pybamm.CasadiSolver(atol=tol), pybamm.CasadiSolver(atol=tol, mode="fast")]
        for params in parameters:
            print(params)
            time_points = []
            if model_ == "SPM":
                model = pybamm.lithium_ion.SPM()
            else:
                model = pybamm.lithium_ion.DFN()

            c_rate = 10
            tmax = 3600 / c_rate
            nb_points = 500
            t_eval = np.linspace(0, tmax, nb_points)
            geometry = model.default_geometry

            # load parameter values and process model and geometry
            param = pybamm.ParameterValues(params)
            param.process_model(model)
            param.process_geometry(geometry)

            # set mesh
            var_pts = {
                "x_n": 20,
                "x_s": 20,
                "x_p": 20,
                "r_n": 30,
                "r_p": 30,
                "y": 10,
                "z": 10,
            }
            mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

            # discretise model
            disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
            disc.process_model(model)
            # for solver_ in solvers:
            for tol in abstols:
                solvers = [pybamm.IDAKLUSolver(atol=tol) , pybamm.CasadiSolver(atol=tol), pybamm.CasadiSolver(atol=tol, mode="fast")]
                for  ax,i in zip(axs.ravel(),itertools.product(models,solvers)):
                    # ax.plot()
                    print(ax)
                    i = list(i)
                    print(list(i))
                    solver = i[1]
                    solver.solve(model, t_eval=t_eval)
                    time = 0
                    runs = 20
                    for k in range(0, runs):

                        solution = solver.solve(model, t_eval=t_eval)
                        time += solution.solve_time.value
                    time = time / runs

                    time_points.append(time)


           

                    ax.plot(abstols, time_points)
            # plt.title(f"Work Precision Sets for {model_} with {solver_} solver")
            # plt.xlabel("abstols")
            # plt.xticks(abstols)
            # plt.xscale("log")
            # plt.yscale("log")
            # plt.ylabel("time(s)")

plt.gca().legend(
    parameters,
    loc="upper right",
)

plt.tight_layout()
N = 1.5
params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches((plSize[0] * N, plSize[1] * N))
plt.show()