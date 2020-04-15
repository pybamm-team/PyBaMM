#
# Example showing how to load and solve the SPMe
#

import pybamm
import numpy as np

pybamm.set_logging_level("INFO")

# load model
options = {
    "thermal": "x-lumped",
    "anode decomposition": True,
    # "cathode decomposition": True,
}
models = [
    pybamm.lithium_ion.SPM(options, name="with decomposition"),
    pybamm.lithium_ion.SPM({"thermal": "x-lumped"}, name="without decomposition"),
]
# add ambient temperature
def ambient_temperature(t):
    return 350 + t * 100 / 3600


solutions = []
for model in models:
    # create geometry
    geometry = model.default_geometry

    # load parameter values and process model and geometry
    param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Cai2019)
    param.update(
        {
            "Ambient temperature [K]": 200 + 273,  # ambient_temperature,
            # "Frequency factor for anode decomposition [s-1]": 5e9,
        },
        check_already_exists=False,
    )
    param.process_model(model)
    param.process_geometry(geometry)

    # set mesh
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

    # discretise model
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

    # solve model for 1 hour
    t_eval = np.linspace(0, 3600, 100)
    solution = model.default_solver.solve(model, t_eval)
    solutions.append(solution)

# plot
plot = pybamm.QuickPlot(
    solutions,
    [
        "Negative particle surface concentration [mol.m-3]",
        "Electrolyte concentration [mol.m-3]",
        "Positive particle surface concentration [mol.m-3]",
        "Current [A]",
        "Negative electrode potential [V]",
        "Electrolyte potential [V]",
        "Positive electrode potential [V]",
        "Terminal voltage [V]",
        "Anode decomposition reaction rate",
        "Cathode decomposition reaction rate",
        "X-averaged cell temperature [K]",
        "Ambient temperature [K]",
        "Relative SEI thickness",
        "Degree of conversion of cathode decomposition",
        "Anode decomposition heating",
    ],
    time_unit="seconds",
    spatial_unit="um",
)
plot.dynamic_plot()
