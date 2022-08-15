import pybamm
import numpy as np
import matplotlib.pyplot as plt


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
models = ["SPM", "DFN"]

for model_ in models:
    if model_ == "SPM":
        x = 1
        parameters = [
            "Marquis2019",
            "NCA_Kim2011",
            "Ramadass2004",
            "Mohtat2020",
            "Chen2020",
            "Chen2020_plating",
            "Ecker2015",
        ]
    else:
        x = 2
        parameters = [
            "Marquis2019",
            "Ramadass2004",
        ]
    for params in parameters:
        time_points = []

        if model_ == "SPM":
            model = pybamm.lithium_ion.SPM()
        else:
            model = pybamm.lithium_ion.DFN()
        c_rate = 1
        tmax = 4000 / c_rate
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
            # solve first
            solver.solve(model, t_eval=t_eval)
            time = 0
            runs = 5
            for k in range(0, runs):
                try:
                    solution = solver.solve(model, t_eval=t_eval)
                except Exception:
                    pass

                time += solution.solve_time.value
            time = time / runs

            time_points.append(time)
        plt.subplot(1, 2, x)
        plt.plot(dt_max, time_points)
        plt.title(f"{model_}")
        plt.xlabel("dt_max")
        plt.xticks(dt_max)
        plt.xscale("log")
        plt.yscale("log")
        plt.ylabel("time(s)")


plt.gca().legend(
    parameters,
    loc="upper right",
)

plt.tight_layout()
# plt.show()
plt.savefig(f"benchmarks/benchmark_images/time_vs_dt_max_{pybamm.__version__}.png")


content = f"## Solve Time vs dt_max\n<img src='./benchmark_images/time_vs_dt_max_{pybamm.__version__}.png'>\n"  # noqa

with open("./benchmarks/release_work_precision_sets.md", "r") as original:
    data = original.read()
with open("./benchmarks/release_work_precision_sets.md", "w") as modified:
    modified.write(f"{content}\n{data}")
