import pybamm as pb

pb.set_logging_level("INFO")
model = pb.lithium_ion.DFN()

parameter_values = pb.ParameterValues("NCA_Kim2011")

sim = pb.Simulation(model, parameter_values=parameter_values, C_rate=1)
sim.solve([0, 3600])
sim.plot()
