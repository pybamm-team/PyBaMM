import pybamm
import numpy as np
import matplotlib.pyplot as plt


parameters = [
    # "Marquis2019",
    "NCA_Kim2011",
    # "Prada2013",-
    "Ramadass2004",
    # "Mohtat2020",-
    # "Chen2020",-
    # "Chen2020_plating",
    # "Ecker2015",
]

dt_max = [1.0e-6,1.0e-5,1.0e-4,1.0e-3,0.01,0.1,1,10,100,1000,1.0e4,1.0e5,1.0e6,1.0e7,1.0e9,1.0e10,1.0e11,1.0e12]
models = [ "DFN"]

for model_ in models:
    if model_ == "SPM":
        x = 1
    else:
        x = 2
    for params in parameters:
        time_points = []
        print(params)
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
            print("a")
            solver = pybamm.CasadiSolver(dt_max = t)
            # solve first
            solver.solve(model, t_eval=t_eval)
            time = 0
            runs = 5
            for k in range(0, runs):

                solution = solver.solve(model, t_eval=t_eval)
                time += solution.solve_time.value
            time = time / runs

            time_points.append(time)
        plt.subplot(1, 2, x)
        plt.plot(dt_max, time_points)
        plt.title(f"Work Precision Sets for {model_}")
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


content = f"<img src='./benchmark_images/time_vs_dt_max_{pybamm.__version__}.png'>"

with open("./benchmarks/release_work_precision_sets.md", "r") as original:
    data = original.read()
with open("./benchmarks/release_work_precision_sets.md", "w") as modified:
    modified.write(f"{content}\n{data}")
