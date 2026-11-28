import pybamm

pybamm.set_logging_level("INFO")

model = pybamm.equivalent_circuit.Thevenin(
    options={"open-circuit voltage": "one-state hysteresis"}
)
parameter_values = model.default_parameter_values

# Use the default OCV as the charge branch and a copy shifted down by 0.15 V
# as the discharge branch (constant offset, for testing only)
ocv_function = parameter_values["Open-circuit voltage [V]"]
ocv_expression = ocv_function(
    pybamm.Variable("Cell Temperature [degC]"), pybamm.Variable("SoC")
)
soc_points = ocv_expression.x[0]
ocv_values = ocv_expression.y


def ocv_dch_function(T_cell, soc):
    return pybamm.Interpolant(soc_points, ocv_values - 0.15, soc, "OCV dch.")


parameter_values.update(
    {
        "Initial hysteresis state": 1,
        "Hysteresis decay rate charge": 10,
        "Hysteresis decay rate discharge": 10,
        "Open-circuit voltage charge [V]": ocv_function,
        "Open-circuit voltage discharge [V]": ocv_dch_function,
    },
    check_already_exists=False,
)

experiment = pybamm.Experiment(
    [
        (
            "Discharge at C/10 for 10 hours or until 3.3 V",
            "Rest for 30 minutes",
            "Rest for 2 hours",
            "Charge at 100 A until 4.1 V",
            "Hold at 4.1 V until 5 A",
            "Rest for 30 minutes",
            "Rest for 1 hour",
        ),
    ]
)

# Add variables to the plot
default_vars = model.default_quick_plot_variables
default_vars[1] += [
    "Open-circuit voltage charge [V]",
    "Open-circuit voltage discharge [V]",
]
default_vars[2] = ["SoC", "Hysteresis state"]
default_vars[-1].append("Hysteresis heat generation [W]")

sim = pybamm.Simulation(model, experiment=experiment, parameter_values=parameter_values)
sim.solve()
sim.plot(output_variables=default_vars)
