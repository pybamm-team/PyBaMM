#
# Example showing how to load and solve the DFN for a half cell
#

import pybamm
import numpy as np

pybamm.set_logging_level("INFO")

# load model
model = pybamm.lithium_ion.BasicDFNHalfCell(working_electrode="cathode")

# define experiment
Crate = 0.5
tpulse = 150
period = tpulse // 50
trest = 2 * 3600
experiment = pybamm.Experiment(
    [
        "Discharge at {}C for {} seconds ({} seconds period)".format(
            Crate, tpulse, period
        ),
        "Rest for {} seconds".format(trest),
    ]
    * 3,
)

# load parameter values and process model and geometry
chemistry = pybamm.parameter_sets.Chen2020
param = pybamm.ParameterValues(chemistry=chemistry)
param.update(
    {
        "Lithium counter electrode exchange-current density [A.m-2]": 12.6,
        "Lithium counter electrode conductivity [S.m-1]": 1.0776e7,
        "Lithium counter electrode thickness [m]": 250e-6,
    },
    check_already_exists=False,
)

# set simulation
sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
sim.solve()

# plot
plot = pybamm.QuickPlot(
    sim.solution,
    [
        "Negative particle surface concentration [mol.m-3]",
        "Electrolyte concentration [mol.m-3]",
        "Positive particle surface concentration [mol.m-3]",
        "Current [A]",
        "Negative electrode potential [V]",
        "Electrolyte potential [V]",
        "Positive electrode potential [V]",
        ["Terminal voltage [V]", "Voltage drop [V]"],
    ],
    time_unit="seconds",
    spatial_unit="um",
)
plot.dynamic_plot()
