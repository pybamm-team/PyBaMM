import pybamm

options = {
    "cell geometry": "pouch",
    "current collector": "potential pair",
    "dimensionality": 2,
}
model = pybamm.lithium_ion.DFN(options=options)
parameter_values = pybamm.ParameterValues("Ecker2015")
experiment = pybamm.Experiment(
    [
        "Discharge at 1C until 2.8V",
        "Charge at C/2 until 4.2V",
    ]
)
var_pts = {"x_n": 8, "x_s": 8, "x_p": 8, "r_n": 8, "r_p": 8, "y": 8, "z": 8}
sim = pybamm.Simulation(
    model,
    parameter_values=parameter_values,
    experiment=experiment,
    var_pts=var_pts,
)
sol = sim.solve()
output_variables = [
    "Current collector current density [A.m-2]",
    "X-averaged negative particle surface stoichiometry",
    "X-averaged negative electrode surface potential difference [V]",
    "Voltage [V]",
]
plot = sol.plot(output_variables, variable_limits="tight", shading="auto")
