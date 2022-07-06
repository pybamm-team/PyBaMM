import pybamm
import numpy as np
import matplotlib.pyplot as plt


parameters = [
    "Marquis2019",
    "NCA_Kim2011",
    "Prada2013",
    "Ramadass2004",
    "Mohtat2020",
    "Chen2020",
    "Chen2020_plating",
    "Ecker2015",
]
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


for params in parameters:
    time_points = []

    for tol in abstols:

        solver = pybamm.CasadiSolver(atol=tol)
        model = pybamm.lithium_ion.SPM()
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
        time = 0
        for k in range(0, 20):

            solution = solver.solve(model, t_eval=t_eval)
            time += solution.solve_time.value
        time = time / 20

        time_points.append(time)
    plt.plot(abstols, time_points)


plt.gca().legend(
    (
        "Marquis2019",
        "NCA_Kim2011",
        "Prada2013",
        "Ramadass2004",
        "Mohtat2020",
        "Chen2020",
        "Chen2020_plating",
        "Ecker2015",
    ),
    loc="upper right",
)
plt.title("Work Precision Sets")
plt.xlabel("abstols")
plt.xscale("log")
plt.xticks(abstols)
plt.ylabel("time(s)")
plt.savefig(f"benchmarks/benchmark_images/time_vs_abstols_{pybamm.__version__}.png")


x = f"## PyBaMM {pybamm.__version__}\n<img src='./benchmark_images/time_vs_abstols_{pybamm.__version__}.png'>"

with open("./benchmarks/release_work_precision_sets.md", "r") as original:
    data = original.read()
with open("./benchmarks/release_work_precision_sets.md", "w") as modified:
    modified.write(f"{x}\n{data}")
