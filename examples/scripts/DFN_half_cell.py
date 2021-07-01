#
# Example showing how to load and solve the DFN for the half cell
#

import pybamm
import numpy as np

pybamm.set_logging_level("INFO")

# load model
options = {"working electrode": "positive"}
model1 = pybamm.lithium_ion.DFN(options=options)
model2 = pybamm.lithium_ion.BasicDFNHalfCell(options=options)

sols = []
for model in [model1, model2]:
    # for model in [model2]:
    # create geometry
    geometry = model.default_geometry

    # load parameter values
    chemistry = pybamm.parameter_sets.Xu2019
    param = pybamm.ParameterValues(chemistry=chemistry)

    # add lithium counter electrode parameter values
    param.update(
        {
            "Negative electrode exchange-current density [A.m-2]": 12.6,
            "Negative electrode conductivity [S.m-1]": 1.0776e7,
            "Negative electrode thickness [m]": 250e-6,
        },
        check_already_exists=False,
    )
    param.update({"Negative electrode OCP [V]": 0})

    param["Current function [A]"] = 2.5e-3

    # process model and geometry
    param.process_model(model)
    param.process_geometry(geometry)

    # set mesh
    var_pts = model.default_var_pts
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

    # discretise model
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

    # solve model
    t_eval = np.linspace(0, 7200, 1000)
    solver = pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-3)
    solution = solver.solve(model, t_eval)
    sols.append(solution)

# plot
plot = pybamm.QuickPlot(
    sols,
    [
        # "Positive particle concentration [mol.m-3]",
        "Electrolyte concentration [mol.m-3]",
        "Current [A]",
        "Positive electrode potential [V]",
        "Electrolyte potential [V]",
        "Total lithium in electrolyte [mol]",
        # "Total lithium in positive electrode [mol]",
        "Positive electrode open circuit potential [V]",
        ["Terminal voltage [V]"],  # , "Voltage drop in the cell [V]"],
    ],
    time_unit="seconds",
    spatial_unit="um",
)
plot.dynamic_plot()
