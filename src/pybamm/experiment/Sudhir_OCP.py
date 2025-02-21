#%%
import pybamm
import numpy as np
import matplotlib.pyplot as plt
from pybamm.models.submodels.interface.open_circuit_potential.RK_Polynomial_OCP import RK_Open_Circuit_Potential

#%%
def rk_polynomial(sto,T):
    func=0
    T=T
    k_b = pybamm.constants.k_b
    A=[ 0.10534196, -0.03267566,  0.05709727, -0.32679114,  0.22005231,  1.3713085,
    -1.80127017, -3.7394661,   5.71767423,  4.9354072,  -7.57306322, -2.60555098,
    3.76719326,  3.24557473]
    #make R-K polynomial
    for i in range (len(A)):
        func=func+A[i]*(((1-2*sto)**(i+1))-((2*i*sto*(1-sto)*((1-2*sto)**(i-1)))))
    # add entropy and standard chemical potential term
    function=k_b*T*(np.log(sto/(1-sto)))+func+A[-1]
    return function

#%%
def run_temp ():
    model=pybamm.lithium_ion.DFN(build=False)
    print(model.submodels["positive primary open-circuit potential"])
    model.submodels["positive primary open-circuit potential"] = RK_Open_Circuit_Potential(model.param, "positive",
    "lithium-ion main", model.options, "primary")
    model.build_model()
    parameter_values = pybamm.ParameterValues("Chen2020")
    parameter_values.update({"Positive electrode RK_OCP [V]": rk_polynomial}, check_already_exists=False)
    parameter_values["Current function [A]"] = 4
    print(model.submodels["positive primary open-circuit potential"])

    sim = pybamm.Simulation(model, parameter_values=parameter_values)
    solutions=sim.solve([0,3600])
    t = solutions["Time [s]"]
    V = solutions["Voltage [V]"]
    return t.entries, V.entries


#%%
t,V=run_temp()
#%%
plt.plot(t,V)
plt.show()