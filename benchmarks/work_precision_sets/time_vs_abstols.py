import pybamm
import numpy as np
import matplotlib.pyplot as plt
import itertools


parameters = ["Marquis2019", "Ecker2015", "Ramadass2004", "Chen2020"]

models = {"SPM": pybamm.lithium_ion.SPM(), "DFN": pybamm.lithium_ion.DFN()}

abstols = [
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

solvers = {
    "IDAKLUSolver": pybamm.IDAKLUSolver(),
    "Casadi - safe": pybamm.CasadiSolver(),
    "Casadi - fast": pybamm.CasadiSolver(mode="fast"),
}


fig, axs = plt.subplots(len(solvers), len(models), figsize=(8, 10))

for ax, i, j in zip(
    axs.ravel(),
    itertools.product(solvers.values(), models.values()),
    itertools.product(solvers, models),
):
    for params in parameters:
        time_points = []
        solver = i[0]

        model = i[1].new_copy()
        c_rate = 1
        tmax = 3500 / c_rate
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

        for tol in abstols:
            solver.atol = tol
            solver.solve(model, t_eval=t_eval)
            time = 0
            runs = 20
            for k in range(0, runs):
                solution = solver.solve(model, t_eval=t_eval)
                time += solution.solve_time.value
            time = time / runs

            time_points.append(time)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("abstols")
        ax.set_ylabel("time(s)")
        ax.set_title(f"{j[1]} with {j[0]}")
        ax.plot(abstols, time_points)

plt.tight_layout()
plt.gca().legend(
    parameters,
    loc="lower right",
)


plt.savefig(f"benchmarks/benchmark_images/time_vs_abstols_{pybamm.__version__}.png")


content = f"# PyBaMM {pybamm.__version__}\n## Solve Time vs Abstols\n<img src='./benchmark_images/time_vs_abstols_{pybamm.__version__}.png'>\n"  # noqa

with open("./benchmarks/release_work_precision_sets.md", "r") as original:
    data = original.read()
with open("./benchmarks/release_work_precision_sets.md", "w") as modified:
    modified.write(f"{content}\n{data}")
