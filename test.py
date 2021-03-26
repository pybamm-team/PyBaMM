import pybamm

model = pybamm.lithium_ion.DFN()
parameter_values = model.default_parameter_values
parameter_values._replace_callable_function_parameters = False
sim = pybamm.Simulation(model, parameter_values=parameter_values)
sim.set_parameters()
mtk_str = pybamm.get_julia_mtk_model(sim.model, geometry=sim.geometry, tspan=(0, 3600))


print(mtk_str)

# list(sim.model.rhs.values())[3].render()
# sim.build()

# rhs_str, u0_str = sim.built_model.generate_julia_diffeq(
#     get_consistent_ics_solver=pybamm.CasadiSolver()
# )
# print(rhs_str)
# print(u0_str)