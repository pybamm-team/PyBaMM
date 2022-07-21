import pybamm
import numpy as np
import matplotlib.pyplot as plt


parameters = [
    "Marquis2019",
    "Prada2013",
    "Ramadass2004",
    # "Mohtat2020",
    "Chen2020",
    "Ecker2015",
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

for model_ in models:
    if model_ == "SPM":
        x = 1
    else:
        x = 2
    for params in parameters:
        print(params)
        time_points = []
        if model_ == "SPM":
            model = pybamm.lithium_ion.SPM()
        else:
            model = pybamm.lithium_ion.DFN()
        
        c_rate = 1 / 10
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

        for tol in abstols:
            print("a")
            solver = pybamm.IDAKLUSolver(atol=tol)
            # solve first
            solver.solve(model, t_eval=t_eval)
            time = 0
            runs = 100
            for k in range(0, runs):

                solution = solver.solve(model, t_eval=t_eval)
                time += solution.solve_time.value
            time = time / runs

            time_points.append(time)
        plt.subplot(1, 2, x)
        plt.plot(abstols, time_points)
        plt.title(f"Work Precision Sets for {model_}")
        plt.xlabel("abstols")
        plt.xticks(abstols)
        plt.xscale("log")
        plt.yscale("log")
        plt.ylabel("time(s)")

plt.gca().legend(
    parameters,
    loc="upper right",
)
plt.tight_layout()
plt.show()
# plt.savefig(f"benchmarks/benchmark_images/time_vs_abstols_{pybamm.__version__}.png")


# content = f"## PyBaMM {pybamm.__version__}\n<img src='./benchmark_images/time_vs_abstols_{pybamm.__version__}.png'>"  # noqa

# with open("./benchmarks/release_work_precision_sets.md", "r") as original:
#     data = original.read()
# with open("./benchmarks/release_work_precision_sets.md", "w") as modified:
#     modified.write(f"{content}\n{data}")
