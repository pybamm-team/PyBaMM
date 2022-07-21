import sys
from io import StringIO
import pybamm
import numpy as np
import matplotlib.pyplot as plt


parameters = [
    "Marquis2019",
    # "NCA_Kim2011",
    # # "Prada2013",
    # "Ramadass2004",
    # "Mohtat2020",
    # "Chen2020",
    # # "Chen2020_plating",
    # "Ecker2015",
]

# dt_max = [18,20,22,24,50,80]
dt_max = [1.e-8
# 1.0e-7,1.0e-6,1.0e-5,1.0e-4,1.0e-3,0.01,0.1,1,10,100,1000,1.0e4,1.0e5,1.0e6,1.0e7,1.0e8,1.0e9,1.0e10,1.0e11,1.0e12,1.0e45,1.0e65
]

for params in parameters:
    time_points = []
    print(params)
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
        print("a======================================================")
        
        stdout_backup = sys.stdout
        sys.stdout = string_buffer = StringIO()
        solver = pybamm.CasadiSolver(dt_max = t,
        extra_options_setup = 
        {"print_stats": True}
        
        # extra_options_call = {"print_stats": True}
        )
        sys.stdout = stdout_backup  # restore old sys.stdout

        string_buffer.seek(0)
        sys.stdout=open("solver_stats.txt","w")
        print (string_buffer.read())
        sys.stdout.close()
        
        
        
        # solve first
        solver.solve(model, t_eval=t_eval)
        time = 0
        runs = 1
        for k in range(0, runs):

            solution = solver.solve(model, t_eval=t_eval)
            time += solution.solve_time.value
        time = time / runs

        time_points.append(time)
    plt.plot(dt_max, time_points)


plt.gca().legend(
    parameters,
    loc="upper right",
)
plt.title("Work Precision Sets")
plt.xlabel("dt_max")
plt.xticks(dt_max)
plt.xscale("log")
plt.yscale("log")
plt.ylabel("time(s)")
plt.show()
# import sys
# from io import StringIO
# def my_function():
#     print('Hello World')
# stdout_backup = sys.stdout
# sys.stdout = string_buffer = StringIO()

# # this function prints something and I cannot change it:


# my_function()  # <-- call the function, sys.stdout is redirected now

# sys.stdout = stdout_backup  # restore old sys.stdout

# string_buffer.seek(0)
# print('The output was:', string_buffer.read())