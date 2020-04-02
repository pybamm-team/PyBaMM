import pybamm as pb

pb.set_logging_level("INFO")
model = pb.lithium_ion.DFN()

chemistry = pb.parameter_sets.NCA_Kim2011
parameter_values = pb.ParameterValues(chemistry=chemistry)

sim = pb.Simulation(model, parameter_values=parameter_values, C_rate=1)
sim.solve()
sim.plot()
