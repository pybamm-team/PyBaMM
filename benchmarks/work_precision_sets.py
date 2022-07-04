import time
import pybamm
import numpy as np
import matplotlib.pyplot as plt

parameters = [
    "Marquis2019",
    # "ORegan2021",
    "NCA_Kim2011",
    "Prada2013",
    "Ramadass2004",
    "Mohtat2020",
    "Chen2020",
    "Chen2020_plating",
    "Ecker2015",
]
# abstols = [
    # 0.001,
    # 0.0001,
    # 1.0e-5,
    # 1.0e-6,
    # 1.0e-7,
    # 1.0e-8,
    # 1.0e-9,
    # 1.0e-10,
    # 1.0e-11,
    # 1.0e-12,
    # 1.0e-13,
# ]
abstols = [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009]
reltols = [1, 0.1, 0.01, 0.001, 0.0001, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10]
time2 = []
time3 = {}

for i in parameters:
    print(i)
    for j in abstols:
        t = time.time()
        solver = pybamm.CasadiSolver(atol=j)
        model = pybamm.lithium_ion.SPM()
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        t_eval = np.linspace(0, tmax, nb_points)
        geometry = model.default_geometry

        # load parameter values and process model and geometry
        param = pybamm.ParameterValues(i)
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
        solver.solve(model, t_eval=t_eval)

        ti = time.time()
        print(ti - t)
        if i in time3:

            time3[i].append(ti - t)

        else:

            time3[i] = [ti - t]
# print(time3)
plt.plot(
    abstols,
    time3["Marquis2019"],
    abstols,
    time3["NCA_Kim2011"],
    abstols,
    time3["Prada2013"],
    abstols,
    time3["Ramadass2004"],
    abstols,
    time3["Mohtat2020"],
    abstols,
    time3["Chen2020"],
    abstols,
    time3["Chen2020_plating"],
    abstols,
    time3["Ecker2015"]

)
plt.gca().legend(('Marquis2019','NCA_Kim2011','Prada2013','Ramadass2004','Mohtat2020','Chen2020','Chen2020_plating','Ecker2015'),loc="upper right")
plt.title("Work Precision Sets")
plt.xlabel("abstols")
plt.xticks(abstols)
plt.ylabel("time(s)")
plt.show()
# fig, axis = plt.subplots(1, 1)
# ax = axis
# ax.plot(
#     abstols,
#     time3["Marquis2019"],
#     abstols,
#     time3["NCA_Kim2011"],
#     abstols,
#     time3["Prada2013"],
#     abstols,
#     time3["Ramadass2004"],
#     abstols,
#     time3["Mohtat2020"],
#     abstols,
#     time3["Chen2020"],
#     abstols,
#     time3["Chen2020_plating"],
#     abstols,
#     time3["Ecker2015"],
# )
# ax.set_xlabel("abstols")
# ax.set_xticks([0.001, 1.0e-5])
# ax.set_ylabel("time(s)")
# fig.tight_layout()
# fig.legend(
#     (
#         "Marquis2019",
#         "NCA_Kim2011",
#         "Prada2013",
#         "Ramadass2004",
#         "Mohtat2020",
#         "Chen2020",
#         "Chen2020_plating",
#         "Ecker2015",
#     ),
#     loc="upper right",
#     bbox_to_anchor=(0, -0.1),
# )
# plt.savefig("benchmarks/plot.png", dpi=300, bbox_inches="tight")
