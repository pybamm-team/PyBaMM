import pybamm

model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model)
sim.set_parameters()
mtk_str = pybamm.get_julia_mtk_model(sim.model, geometry=sim.geometry, tspan=(0, 3600))


print(mtk_str)

# list(sim.model.rhs.values())[1].render()
# sim.build()

# rhs_str, u0_str = sim.built_model.generate_julia_diffeq(
#     get_consistent_ics_solver=pybamm.CasadiSolver()
# )
# print(rhs_str)
# print(u0_str)