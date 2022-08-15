import pybamm
import numpy as np
import matplotlib.pyplot as plt


parameters = [
    "Marquis2019",
    "Prada2013",
    "Ramadass2004",
    "Chen2020",
]
models = ["SPM", "DFN"]
reltols = [
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
solvers = ["IDAKLU", "Casadi - safe", "Casadi - fast"]

for model_ in models:
    for solver_ in solvers:

        for params in parameters:

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

            for tol in reltols:

                if solver_ == "IDAKLU" and model_ == "SPM":
                    solver = pybamm.IDAKLUSolver(rtol=tol)
                    x = 1
                elif solver_ == "IDAKLU" and model_ == "DFN":
                    solver = pybamm.IDAKLUSolver(rtol=tol)
                    x = 2
                elif solver_ == "Casadi - safe" and model_ == "SPM":
                    solver = pybamm.CasadiSolver(rtol=tol)
                    x = 3
                elif solver_ == "Casadi - safe" and model_ == "DFN":
                    solver = pybamm.CasadiSolver(rtol=tol)
                    x = 4
                elif solver_ == "Casadi - fast" and model_ == "SPM":
                    solver = pybamm.CasadiSolver(rtol=tol, mode="fast")
                    x = 5
                else:
                    solver = pybamm.CasadiSolver(rtol=tol, mode="fast")
                    x = 6
                # solve first
                solver.solve(model, t_eval=t_eval)
                time = 0
                runs = 20
                for k in range(0, runs):

                    solution = solver.solve(model, t_eval=t_eval)
                    time += solution.solve_time.value
                time = time / runs

                time_points.append(time)
            plt.subplot(3, 2, x)

            plt.plot(reltols, time_points)
            plt.title(f"{model_} with {solver_} solver")
            plt.xlabel("reltols")
            plt.xticks(reltols)
            plt.xscale("log")
            plt.yscale("log")
            plt.ylabel("time(s)")

plt.gca().legend(
    parameters,
    loc="upper right",
)

plt.tight_layout()
N = 1.5
params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches((plSize[0] * N, plSize[1] * N))

plt.savefig(f"benchmarks/benchmark_images/time_vs_reltols_{pybamm.__version__}.png")


content = f"## Solve Time vs Reltols\n<img src='./benchmark_images/time_vs_reltols_{pybamm.__version__}.png'>\n"  # noqa

with open("./benchmarks/release_work_precision_sets.md", "r") as original:
    data = original.read()
with open("./benchmarks/release_work_precision_sets.md", "w") as modified:
    modified.write(f"{content}\n{data}")
