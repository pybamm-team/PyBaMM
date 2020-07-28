import pybamm

model = pybamm.lithium_ion.DFN()
param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Marquis2019)
param.update(
    {
        "Negative electrode active material volume fraction": "[input]",
        "Positive electrode active material volume fraction": "[input]",
    }
)

sim = pybamm.Simulation(model, parameter_values=param)

inputs = {
    "Negative electrode active material volume fraction": 0.7,
    "Positive electrode active material volume fraction": 0.7,
}
sol_old = sim.solve(t_eval=[0, 3600], inputs=inputs)

inputs = {
    "Negative electrode active material volume fraction": 0.6,
    "Positive electrode active material volume fraction": 0.5,
}
sol_new = sim.solve(t_eval=[0, 3600], inputs=inputs)

pybamm.dynamic_plot([sol_old, sol_new])
