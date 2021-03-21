import pybamm

model = pybamm.lithium_ion.DFN(name="DFN")
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 10, var.r_p: 10}
sim = pybamm.Simulation(model, var_pts=var_pts)
# sim.set_parameters()

# model = sim.model
# list(model.rhs.values())[-1].render()

# mtk_str = pybamm.get_julia_mtk_model(model, geometry=sim.geometry, tspan=(0, 3600))
# print(mtk_str)

sim.build()

rhs_str, u0_str = sim.built_model.generate_julia_diffeq(
    get_consistent_ics_solver=pybamm.CasadiSolver()
)
print(rhs_str)
print(u0_str)