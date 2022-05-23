#
# ECM model with parameters from csv
#

import numpy as np
import matplotlib.pyplot as plt
import pybamm
import pandas as pd

plt.close('all')
param = pd.read_csv(r'ecm_parameters.csv')

exp_z = np.array(param['z'])
exp_ocv = np.array(param['ocv'])
exp_R0 = np.array(param['R0'])
exp_R1 = np.array(param['R1'])
exp_C1 = np.array(param['C1'])
exp_R2 = np.array(param['R2'])
exp_C2 = np.array(param['C2'])

Q = 5.0

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0][0].plot(param['z'], exp_ocv)
axes[0][1].plot(exp_z, exp_C1)
axes[0][2].plot(exp_z, exp_C2)
axes[1][0].plot(exp_z, exp_R0)
axes[1][1].plot(exp_z, exp_R1)
axes[1][2].plot(exp_z, exp_R2)


# Create an ECM in PyBaMM

model = pybamm.BaseModel(name='ECM')

# Time
# t = pybamm.Variable("t")
# t_init = 0.0
# dtdt = 1.0
t = pybamm.t
# Current Mode
mode = 'step'

if mode == 'cc':
    i = pybamm.Variable("i")
    didt = 0.0
    z_init = 0.95 # Start near fully charged 
    i_init = 2.5
    iR_1_init = i_init
    iR_2_init = i_init
    t_eval = np.linspace(0, 3600, 1001)
    
elif mode == 'sin':
    i = pybamm.Variable("i")
    f = 2 * np.pi / 3600
    n_period = 10
    I_mag = Q
    didt = pybamm.cos(t * n_period * f) * (n_period * f) * I_mag
    z_init = 0.5 # start in middle so we don't go over limits
    i_init = 0.0
    iR_1_init = 0.0
    iR_2_init = 0.0
    t_eval = np.linspace(0, 3600, 1001)
    
elif mode == 'step':
    dt = 60
    pulse_sequence = [0, 0, 3, 0, 0, 0]
    cycle = []
    for ps in pulse_sequence:
        cycle = cycle + [ps] * dt
    ncycle = 3
    I_app_i = np.asarray(cycle * ncycle).flatten().astype(float)
    I_app_t = np.arange(len(I_app_i))
    rnd = (np.random.random(len(I_app_i))-0.5) / 100
    I_app_i += rnd
    i = pybamm.Interpolant(I_app_t, I_app_i, children=t, interpolator="linear")
    iR_1_init = I_app_i[0]
    iR_2_init = I_app_i[0]
    z_init = 0.90 # Start near fully charged
    t_eval = np.linspace(0, I_app_t.max(),  int(I_app_t.max() + 1))


# SoC
z = pybamm.Variable("z")
dzdt = -i / (3600 * Q)

# Interpolated Data
# Interpolated ocv
kind = "linear"
# kind = "cubic spline"
ocv = pybamm.Interpolant(exp_z, exp_ocv, children=z, interpolator=kind)
# Interpolated R0
R0 = pybamm.Interpolant(exp_z, exp_R0, children=z, interpolator=kind)
# Inetrpolated R1
R1 = pybamm.Interpolant(exp_z, exp_R1, children=z, interpolator=kind)
# Inetrpolated C1
C1 = pybamm.Interpolant(exp_z, exp_C1, children=z, interpolator=kind)
# Inetrpolated R2
R2 = pybamm.Interpolant(exp_z, exp_R2, children=z, interpolator=kind)
# Inetrpolated C2
C2 = pybamm.Interpolant(exp_z, exp_C2, children=z, interpolator=kind)


# Current through resistor 1
iR_1 = pybamm.Variable("iR_1")
alpha_1 = (1 / (R1 * C1))
diR_1dt = -alpha_1 * iR_1 + alpha_1 * i

# Current through resistor 2
iR_2 = pybamm.Variable("iR_2")
alpha_2 = (1 / (R2 * C2))
diR_2dt = -alpha_2 * iR_2 + alpha_2 * i


if mode in ["cc", "sin"]:
    # Current is a time-dependent variable
    model.rhs = {
        # t: dtdt,
        z: dzdt,
        i: didt,
        iR_1: diR_1dt,
        iR_2: diR_2dt
        }
    
    model.initial_conditions = {
        # t: pybamm.Scalar(t_init),
        z: pybamm.Scalar(z_init),
        i: pybamm.Scalar(i_init),
        iR_1: pybamm.Scalar(iR_1_init),
        iR_2: pybamm.Scalar(iR_2_init),
        }
elif mode in ["step", "data"]:
    # Current is interpolated
    model.rhs = {
        # t: dtdt,
        z: dzdt,
        iR_1: diR_1dt,
        iR_2: diR_2dt
        }
    
    model.initial_conditions = {
        # t: pybamm.Scalar(t_init),
        z: pybamm.Scalar(z_init),
        iR_1: pybamm.Scalar(iR_1_init),
        iR_2: pybamm.Scalar(iR_2_init),
        }


# Overpotentials and cell voltage
eta_0 = i * R0
eta_1 = iR_1 * R1
eta_2 = iR_2 * R2
v = ocv - eta_0 - eta_1 - eta_2

model.variables = {
    "t": t,
    "z": z,
    "i": i,
    "ocv":ocv,
    "v": v,
    "iR_1":iR_1,
    "iR_2":iR_2,
    "eta_0": eta_0,
    "eta_1": eta_1,
    "eta_2": eta_2,
    }

disc = pybamm.Discretisation()
disc.process_model(model)


output_variables = ["z", "i", "ocv", "v"]


solver = pybamm.ScipySolver()
sol = solver.solve(model, t_eval)

solver2 = pybamm.CasadiSolver()
sol2 = solver2.solve(model, t_eval)

plot = pybamm.QuickPlot(solutions=[sol, sol2], output_variables=output_variables)
plot.dynamic_plot()

# set all print names
t.print_name = "t"
z.print_name = "z"
i.print_name = "i"
ocv.print_name = "ocv"
v.print_name = "v"
R0.print_name = "R_0"
R1.print_name = "R_1"
R2.print_name = "R_2"
iR_1.print_name = "i_{R1}"
iR_2.print_name = "i_{R2}"
eta_0.print_name = "\eta_0"
eta_1.print_name = "\eta_1"
eta_2.print_name = "\eta_2"
C1.print_name = "C_1"
C2.print_name = "C_2"

model.latexify()
