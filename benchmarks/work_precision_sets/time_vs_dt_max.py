import pybamm
import numpy as np
import matplotlib.pyplot as plt


parameters = [
    "Marquis2019",
    "Prada2013",
    "Ramadass2004",
    "Chen2020",
]

models = {"SPM": pybamm.lithium_ion.SPM(), "DFN": pybamm.lithium_ion.DFN()}

dt_max = [
    10,
    20,
    50,
    80,
    100,
    150,
    250,
    400,
    600,
    900,
    1200,
    1600,
    2100,
    2600,
    3000,
    3600,
]


fig, axs = plt.subplots(1, len(models), figsize=(8, 3))

for ax, model_, model_name in zip(
    axs.ravel(),
    models.values(),
    models,
):
    for params in parameters:
        time_points = []
        # solver = pybamm.CasadiSolver()

        model = model_.new_copy()
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

        for t in dt_max:
            solver = pybamm.CasadiSolver(dt_max=t)

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
        ax.set_xlabel("dt_max")
        ax.set_ylabel("time(s)")
        ax.set_title(f"{model_name}")
        ax.plot(dt_max, time_points)

plt.tight_layout()
plt.gca().legend(
    parameters,
    loc="upper right",
)
plt.savefig(f"benchmarks/benchmark_images/time_vs_dt_max_{pybamm.__version__}.png")


content = f"## Solve Time vs dt_max\n<img src='./benchmark_images/time_vs_dt_max_{pybamm.__version__}.png'>\n"  # noqa

with open("./benchmarks/release_work_precision_sets.md", "r") as original:
    data = original.read()
with open("./benchmarks/release_work_precision_sets.md", "w") as modified:
    modified.write(f"{content}\n{data}")
