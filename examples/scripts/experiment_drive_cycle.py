#
# Constant-current constant-voltage charge with US06 Drive Cycle using Experiment Class.
#
import pybamm
import numpy as np
import pandas as pd
import os

os.chdir(pybamm.__path__[0] + "/..")

pybamm.set_logging_level("INFO")

# import drive cycle from file
drive_cycle_current = pd.read_csv(
    "pybamm/input/drive_cycles/US06.csv", comment="#", header=None
).to_numpy()


# Map Drive Cycle
def Map_Drive_Cycle(x, min_ip_value, max_ip_value, min_op_value, max_op_value):
    return (x - min_ip_value) * (max_op_value - min_op_value) / (
        max_ip_value - min_ip_value
    ) + min_op_value


# Map Current to Voltage
temp_volts = np.array([])
min_ip_value = drive_cycle_current[:, 1].min()
max_ip_value = drive_cycle_current[:, 1].max()
min_Voltage = 3.5
max_Voltage = 4.1
for I in drive_cycle_current[:, 1]:
    temp_volts = np.append(
        temp_volts,
        Map_Drive_Cycle(I, min_ip_value, max_ip_value, min_Voltage, max_Voltage),
    )

drive_cycle_voltage = drive_cycle_current
drive_cycle_voltage = np.column_stack(
    (np.delete(drive_cycle_voltage, 1, 1), np.array(temp_volts))
)

# Map Current to Power
temp_volts = np.array([])
min_ip_value = drive_cycle_current[:, 1].min()
max_ip_value = drive_cycle_current[:, 1].max()
min_Power = 2.5
max_Power = 5.5
for I in drive_cycle_current[:, 1]:
    temp_volts = np.append(
        temp_volts, Map_Drive_Cycle(I, min_ip_value, max_ip_value, min_Power, max_Power)
    )

drive_cycle_power = drive_cycle_current
drive_cycle_power = np.column_stack(
    (np.delete(drive_cycle_power, 1, 1), np.array(temp_volts))
)
experiment = pybamm.Experiment(
    [
        "Charge at 1 A until 4.1 V",
        "Hold at 4.1 V until 50 mA",
        "Rest for 1 hour",
        "Run US06_A (A)",
        "Rest for 1 hour",
    ]
    # + [
    #     "Charge at 1 A until 4.1 V",
    #     "Hold at 4.1 V until 50 mA",
    #     "Rest for 1 hour",
    #     "Run US06_V (V)",
    #     "Rest for 1 hour",
    # ]
    + [
        # "Charge at 1 A until 4.1 V",
        # "Hold at 4.1 V until 50 mA",
        # "Rest for 1 hour",
        "Run US06_W (W)",
        "Rest for 1 hour",
    ],
    drive_cycles={
        "US06_A": drive_cycle_current,
        "US06_V": drive_cycle_voltage,
        "US06_W": drive_cycle_power,
    },
)

model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model, experiment=experiment, solver=pybamm.CasadiSolver())
sim.solve()

# Show all plots
sim.plot()
