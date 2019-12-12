import pybamm
import matplotlib.pyplot as plt

model_options = {"thermal": "x-full"}
model = pybamm.lithium_ion.DFN(options=model_options)


var_pts = {
    pybamm.standard_spatial_vars.x_n: 10,
    pybamm.standard_spatial_vars.x_s: 10,
    pybamm.standard_spatial_vars.x_p: 10,
    pybamm.standard_spatial_vars.r_n: 10,
    pybamm.standard_spatial_vars.r_p: 10,
}

sim = pybamm.Simulation(model, var_pts=var_pts)

sim.solve()

x = sim.get_variable_array("x [m]")
Q_ohm = sim.get_variable_array("Ohmic heating [W.m-3]")

plt.plot(x, Q_ohm, "*")
plt.xlabel("x [m]")
plt.ylabel("Ohmic heating [W.m-3]")
plt.show()
