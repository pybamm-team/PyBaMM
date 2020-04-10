import pybamm


x_n = pybamm.standard_spatial_vars.x_n
x_s = pybamm.standard_spatial_vars.x_s
x_p = pybamm.standard_spatial_vars.x_p

var_pts = {x_n: 10, x_s: 3, x_p: 10}
model_lin = pybamm.lead_acid.Full()
sim_lin = pybamm.Simulation(model_lin, var_pts=var_pts)
sim_lin.solve()

model_quad = pybamm.lead_acid.Full()
method_options = {"extrapolation": {"order": "quadratic", "use bcs": False}}
spatial_methods = {
    "negative particle": pybamm.FiniteVolume(method_options),
    "positive particle": pybamm.FiniteVolume(method_options),
    "macroscale": pybamm.FiniteVolume(method_options),
    "current collector": pybamm.ZeroDimensionalSpatialMethod(),
}
sim_quad = pybamm.Simulation(
    model_quad, spatial_methods=spatial_methods, var_pts=var_pts
)
sim_quad.solve()


# plot the two sols
solutions = [sim_lin.solution, sim_quad.solution]
plot = pybamm.QuickPlot(solutions)
plot.dynamic_plot()
