#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:45:13 2019

@author: thomas
"""

import pybamm
import numpy as np
import matplotlib.pyplot as plt

mysolver = pybamm.ScipySolver()
# load model
model = pybamm.lithium_ion.SPMe()

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values
param.process_model(model)
param.process_geometry(geometry)

# set mesh
mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
t_eval = np.linspace(0, 2, 100)
solution = mysolver.solve(model, t_eval)



print('SPM model variables:')
for v in model.variables.keys():
    print('\t-',v)
    
I_neg_s = pybamm.ProcessedVariable(model.variables['Negative electrode current density [A.m-2]'], solution.t, solution.y, mesh=mesh)
I_pos_s = pybamm.ProcessedVariable(model.variables['Positive electrode current density [A.m-2]'], solution.t, solution.y, mesh=mesh)
I_l = pybamm.ProcessedVariable(model.variables['Electrolyte current density [A.m-2]'], solution.t, solution.y, mesh=mesh)
I_i = pybamm.ProcessedVariable(model.variables['Interfacial current density [A.m-2]'], solution.t, solution.y, mesh=mesh)
phis_neg = pybamm.ProcessedVariable(model.variables['Negative electrode potential [V]'], solution.t, solution.y, mesh=mesh)
grad_phis = pybamm.Gradient(model.variables['Negative electrode potential [V]'])

phis_pos = pybamm.ProcessedVariable(model.variables['Positive electrode potential [V]'], solution.t, solution.y, mesh=mesh)
phil = pybamm.ProcessedVariable(model.variables['Electrolyte potential [V]'], solution.t, solution.y, mesh=mesh)
eta_neg = pybamm.ProcessedVariable(model.variables['Negative electrode reaction overpotential [V]'], solution.t, solution.y, mesh=mesh)
eta_pos = pybamm.ProcessedVariable(model.variables['Positive electrode reaction overpotential [V]'], solution.t, solution.y, mesh=mesh)
particle_conc_neg = pybamm.ProcessedVariable(model.variables['Negative particle concentration'], solution.t, solution.y, mesh=mesh)
particle_conc_pos = pybamm.ProcessedVariable(model.variables['Positive particle concentration'], solution.t, solution.y, mesh=mesh)

plt.figure()
x = model.variables['x'].evaluate().flatten()
x_neg = ~np.isnan(I_neg_s(solution.t[0], x=x))
x_pos = ~np.isnan(I_pos_s(solution.t[0], x=x))
plt.plot(x, I_neg_s(solution.t[0], x=x))
plt.plot(x, I_pos_s(solution.t[0], x=x))
plt.plot(x, I_l(solution.t[0], x=x))
plt.plot(x, I_i(solution.t[0], x=x))

#out_var = ['Negative electrode current density',
#           'Positive electrode current density',
#           'Electrolyte current density',
#           'Interfacial current density',
#           'Discharge capacity [A.h]',
#           'Negative electrode potential',
#           'Positive electrode potential',
#           'Electrolyte potential',
#           'Negative electrode reaction overpotential',
#           'Positive electrode reaction overpotential',
#           'Average reaction overpotential [V]',
#           'Average open circuit potential [V]',
#           'x_n',
#           'x_p']
out_var = ['Interfacial current density [A.m-2]']
plot = pybamm.QuickPlot(model, mesh, solution, output_variables=out_var)
plot.dynamic_plot()

# plot
#plot = pybamm.QuickPlot(model, mesh, solution)
#plot.dynamic_plot()

# Entropy changes
param = pybamm.standard_parameters_lithium_ion
#values = pybamm.LithiumIonBaseModel().default_parameter_values
#output_n = []
#output_p = []
#Cs = np.linspace(0,1,101)
#for C in Cs:
#    c_test = pybamm.Scalar(C)
#    output_n.append(values.process_symbol(param.dUdT_n(c_test)).evaluate())
#    output_p.append(values.process_symbol(param.dUdT_p(c_test)).evaluate())
#plt.figure()
##plt.plot(Cs, output_n)
#plt.plot(Cs, output_p)

# Particle Concentrations 
c_ave_neg = []
plt.figure()
for time in solution.t:
    temp_c_neg = particle_conc_neg(time, r=x)
    plt.plot(temp_c_neg)
    c_ave_neg.append(np.mean(temp_c_neg))

c_ave_pos = []
plt.figure()
for time in solution.t:
    temp_c_pos = particle_conc_pos(time, r=x)
    plt.plot(temp_c_pos)
    c_ave_pos.append(np.mean(temp_c_pos))
    
plt.figure()
plt.plot(solution.t, c_ave_neg)
plt.plot(solution.t, c_ave_pos)

(fig, (ax1, ax2)) = plt.subplots(1, 2)
Temp = values['Reference temperature [K]']
w_neg = values['Negative electrode width [m]']
w_pos = values['Positive electrode width [m]']
sa_neg = values['Negative electrode surface area density [m-1]']
sa_pos = values['Positive electrode surface area density [m-1]']

processed_grad = values.process_symbol(grad_phis)

for i, time in enumerate(solution.t):
    # Reversible Entropic Heat Source Neg
    c_neg = pybamm.Scalar(c_ave_neg[i])
    Qrev_neg = I_i(time, x)*Temp*values.process_symbol(param.dUdT_n(c_neg)).evaluate()*sa_neg
    Qrev_neg *= x_neg
    # Reversible Entropic Heat Source Pos
    c_pos = pybamm.Scalar(c_ave_pos[i])
    Qrev_pos = I_i(time, x)*Temp*values.process_symbol(param.dUdT_p(c_pos)).evaluate()*sa_pos
    Qrev_pos *= x_pos
    ax1.plot(x, Qrev_neg+Qrev_pos)
    # Irreversible Entropic Heat Source Neg
    Qirrev_neg = I_i(time, x)*sa_neg*eta_neg(time, x)
    Qirrev_neg[np.isnan(Qirrev_neg)] = 0.0
    # irreversible Entropic Heat Source Pos
    Qirrev_pos = I_i(time, x)*sa_pos*eta_pos(time, x)
    Qirrev_pos[np.isnan(Qirrev_pos)] = 0.0
    ax2.plot(x, Qirrev_neg+Qirrev_pos)
    # Electrolyte Ohmic Heat Source
    