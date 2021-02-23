import pybamm
import matplotlib.pyplot as plt

model = pybamm.lithium_ion.SPM()

sim = pybamm.Simulation(model)
sim.solve([0, 3600])
sim.plot(testing=True)  # testing=True stops the plot showing up
sim.quick_plot.axes[7].set_xlim([0, 0.5])
plt.show()
