# need to restart jupyter server? kernal? whenever a change is made to the pybamm module? 
import sys
import os
import numpy as np
import pandas as pd
import os
from scipy import integrate
import matplotlib.pyplot as plt
os.chdir(sys.path[0] + '\..') # change our working directory to the root of the pybamm folder
import pybamm
from pybamm import exp, constants, Parameter

print(pybamm.__path__[0])
# %matplotlib widget
# inline
import inspect

class ExternalCircuitResistanceFunction():
    def __call__(self, variables):
        I = variables["Current [A]"]
        V = variables["Terminal voltage [V]"]
        R_ext = pybamm.FunctionParameter("External resistance [Ohm]",  {"Time [s]": pybamm.t}) 
        R_tab = pybamm.FunctionParameter("Tabbing resistance [Ohm]",  {"Time [s]": pybamm.t})
        return V/I - (R_ext + R_tab)
    

def modified_graphite_diffusivity_PeymanMPM(sto, T):
    D_ref =  Parameter("Negative electrode diffusion coefficient [m2.s-1]")
    # D_ref = 16*5.0 * 10 ** (-15)
    E_D_s = 42770/10
    arrhenius = exp(E_D_s / constants.R * (1 / 298.15 - 1 / T))

    # soc = (sto - 0.036)/(0.842-0.036)
    soc = (sto - 0)/(0.8321-0)


    # k = -11.79206386*soc**3 + 19.74642427*soc**2 -9.58613417*soc + 2.18105841 #fitted without thermal model
    # k = 0.86610992*soc + 0.15162466 #11?
    k = 0.99*soc + 0.01 #
    # k = 0.86681782*soc + 0.16565809 #13

    # p_high = -15.88891693*soc**3 + 32.73897665*soc**2 -22.08987841*soc + 5.8916456 # high SOC
    # p_low = 15.47540036*soc**2 -10.8508352*soc + 2.13736774 # low SOC
    # k = p_high*(1/(1 + exp(-(soc-0.4)*150)))+ p_low*(1-1/(1 + exp(((0.4-soc)*150))))

    # Ds11
    # p_high = 1.3792875#-15.85192692*soc**3 + 34.1354626*soc**2 -23.71872136*soc + 6.33107728 # high SOC
    # p_low = 0.23807837 #0.35006582 # low SOC
    # # p_high = -0.9128685*soc + 1.95773523
    # # p_low = -1.40355492 + 0.68102741
    # k = p_high*(1/(1 + exp(-(soc-0.4)*20)))+ p_low*(1-1/(1 + exp(((0.4-soc)*20))))

    return D_ref*k # *(-0.9 * sto + 1)

def modified_NMC_diffusivity_PeymanMPM(sto, T):
    D_ref =  Parameter("Positive electrode diffusion coefficient [m2.s-1]")
    E_D_s = 18550/10
    soc = (0.837-sto)/(0.837-0.034)
    # k = 3.77967906*soc**2 - 3.54525492*soc + 1.04220164 # 8 fitted with thermal model 
    # k = 2.61225029*soc**2 - 2.17500416*soc + 0.69677927 # 13 fitted with thermal model 
    k = 3.73478491*soc**2 - 3.50522373*soc + 1.03247363 # 8?fitted without thermal model

    # k = 1.73857835*soc**2 -0.95867106*soc + 0.29889581 # 9 removing high error fits. fitted without thermal model
    # k =  3.32316507*soc**2 -3.06075959*soc + 0.94787949 # 11 removing high error fits k0=[SOC_0]*2
    # k = 0.62889645*soc +0.08790276

    # p_high = -7.54090264*soc**3 +17.18863735*soc**2 -10.57421123*soc + 1.97512672 # high SOC
    # p_low = 6.12915318*soc**2 -4.58479402*soc + 1.06832757# low SOC
    # k = p_high*(1/(1 + exp(-(soc-0.6)*150)))+ p_low*(1-1/(1 + exp(((0.6-soc)*150))))
    # arrhenius = exp(E_D_s / constants.R * (1 / 298.15 - 1 / T))
    
    return D_ref *k

def modified_electrolyte_diffusivity_PeymanMPM(c_e, T):
    # D_c_e = 5.35 * 10 ** (-10)
    D_c_e =  Parameter("Typical electrolyte diffusivity [m2.s-1]")
    E_D_e = 37040
    arrhenius = exp(E_D_e / constants.R * (1 / 298.15 - 1 / T))
    # k = Parameter("Electrolyte diffusion scalar")
    # k_T =  1/(1 + exp(-10*((114+273.15)-T))*(1-0.5)+0.5)
    # k_T=1
    # D_c_e = 8.794e-11 * (c_e/ 1000) ** 2 - 3.972e-10 * (c_e / 1000) + 4.862e-10
    # k_T =  pybamm.sigmoid(T,114+273.15,10).evaluate()*(1-0.1) + 0.5
    return D_c_e#*arrhenius

def modified_electrolyte_conductivity_PeymanMPM(c_e, T):
    # sigma_e = 1.3
    sigma_e = Parameter("Typical electrolyte conductivity [m2.s-1]")
    E_k_e = 34700
    arrhenius = exp(E_k_e / constants.R * (1 / 298.15 - 1 / T))

    # k_T = 1*(1/(1 + exp(-(T-(114+273.15))*150)))+ 1*(1-1/(1 + exp(((114+273.15)-T)*150)))
    # k_T =  1#pybamm.sigmoid(T,114+273.15,10).evaluate()*(1-0.1) + 0.5
    # k_T =  1/(1 + exp(-5*((114+273.15)-T))*(1-0.9)+0.9)
    return sigma_e#*k_T# (1-k)#*arrhenius

def modified_NMC_electrolyte_exchange_current_density_PeymanMPM(c_e, c_s_surf, c_s_max, T):
    m_ref =  Parameter("Positive electrode reference exchange-current density [A.m-2(m3.mol)1.5]")
    # m_ref = 4.824 * 10 ** (-6)  # (A/m2)(mol/m3)**1.5 - includes ref concentrations
    E_r = 39570
    arrhenius = exp(E_r / constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    )

def modified_graphite_electrolyte_exchange_current_density_PeymanMPM(c_e, c_s_surf, c_s_max, T):
    m_ref =  Parameter("Negative electrode reference exchange-current density [A.m-2(m3.mol)1.5]")
    # m_ref = 4*1.061 * 10 ** (-6)  # unit has been converted
    # units are (A/m2)(mol/m3)**1.5 - includes ref concentrations
    E_r = 37480
    arrhenius = exp(E_r / constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    )

solutions = []
SOC_0 = 1
SOC_name = SOC_0
Q_nom = 4.5
h = 0.72753047-0.05 #0.77996386
Cp = 1.51997699 #1.94584251
R_tab = pybamm.Parameter("Tabbing resistance [Ohm]")
R_ext = pybamm.Parameter("External resistance [Ohm]")
data_sets = []
# R_tab_list =  [0.0085]#, 0.015] # 0.0085 (100%), 0.0078 (75%), 0.007? (50%)

# load data import ESC data from file
raw_data = pd.read_csv("./fast_discharge/ESC_"+  str(int(SOC_name*100)) + "SOC_full.csv")
esc_start = raw_data[-raw_data['Current Shunt']>1].index[0]
esc_end = len(raw_data)
# T_amb = np.mean(raw_data['Cell Temperature'][(raw_data.index < esc_start) & (raw_data['Cell Temperature'] >0)])
data = raw_data[['Time (s)', 'Voltage (V)', 'Cell Temperature', 'Current Shunt', 'Force']].loc[esc_start:esc_end].copy()
data['Current Shunt'] = -data['Current Shunt']
data['Time (s)'] = data['Time (s)'] - data['Time (s)'].loc[esc_start]
df_labels = ['t', 'V', 'Temp','I', 'F']
data.set_axis(df_labels, axis=1, inplace=True)
data['I_C'] = data.I/Q_nom 
AhT_calculated = integrate.cumtrapz(abs(data.I), data.t-data.t.iloc[0])/3600
AhT_calculated = np.append(AhT_calculated,AhT_calculated[-1])
data['SOC'] = SOC_0 - AhT_calculated/Q_nom 
T_amb = data.Temp.iloc[0]
data_sets.append(data)

options = {
    # "thermal": "x-lumped",
#         "side reactions": "decomposition", 
    "operating mode": ExternalCircuitResistanceFunction(),
}
model = pybamm.lithium_ion.SPMe(options = options)
chemistry = pybamm.parameter_sets.Mohtat2020
# chemistry["electrolyte"] = "lipf6_Nyman2008"
# chemistry["electrolyte"] = "LiPF6_Mohtat2020"
param = pybamm.ParameterValues(chemistry)

param.update({
    "Tabbing resistance [Ohm]":  0.0085,#0.0041,D
    "External resistance [Ohm]": 0.0067, # 0.0067
    "Cell capacity [A.h]": 4.6, #nominal
    "Typical current [A]": 4.6,
    "Negative electrode thickness [m]":62E-06*4.2/5,
    "Positive electrode thickness [m]":67E-06*4.2/5,
    "Lower voltage cut-off [V]": 0,
    "Ambient temperature [K]":T_amb + 273.15,
    "Initial temperature [K]": T_amb + 273.15,
    "Negative tab width [m]":2.5e-2,
    "Positive tab width [m]":2.5e-2,
    "Negative current collector surface heat transfer coefficient [W.m-2.K-1]": h,  
    "Positive current collector surface heat transfer coefficient [W.m-2.K-1]": h,  
    "Negative tab heat transfer coefficient [W.m-2.K-1]":h,  
    "Positive tab heat transfer coefficient [W.m-2.K-1]":h,  
    "Edge heat transfer coefficient [W.m-2.K-1]":h,
    "Total heat transfer coefficient [W.m-2.K-1]":h,
    "Negative electrode specific heat capacity [J.kg-1.K-1]": 1100*Cp,
    "Positive electrode specific heat capacity [J.kg-1.K-1]": 1100*Cp,
    "Negative electrode diffusivity [m2.s-1]": 8e-14,
    "Positive electrode diffusivity [m2.s-1]": 8e-15,
    "Electrolyte diffusivity [m2.s-1]": modified_electrolyte_diffusivity_PeymanMPM, #set constant. ignore temperature dependence
    "Electrolyte conductivity [S.m-1]": modified_electrolyte_conductivity_PeymanMPM, #set constant. ignore temperature dependence
    "Negative electrode exchange-current density [A.m-2]": modified_graphite_electrolyte_exchange_current_density_PeymanMPM,
    "Positive electrode exchange-current density [A.m-2]": modified_NMC_electrolyte_exchange_current_density_PeymanMPM,
    # "Diffusion stoichiometry scalar": 2,
    # "Diffusion stoichiometry offset": 1,
    # "Electrolyte diffusion scalar":100,
}, check_already_exists = False)

V = model.variables["Terminal voltage [V]"]
I = model.variables["Current [A]"]
model.variables.update({
    "Terminal voltage [V]": V - I*R_tab,
    "Actual resistance [Ohm]":V/I,
    }
)

dt = 0.1
t_eval = np.arange(0, 8*60, dt)
solver = pybamm.CasadiSolver(mode="safe") #, extra_options_setup={"max_num_steps": 10000}
sim = pybamm.Simulation(model, parameter_values = param, solver=solver) 
solution = sim.solve( initial_soc=SOC_0, t_eval = t_eval)
solutions.append(solution)


param.update({
        "Positive electrode diffusivity [m2.s-1]": modified_NMC_diffusivity_PeymanMPM,
})
solver = pybamm.CasadiSolver(mode="safe", dt_max=0.1) #, extra_options_setup={"max_num_steps": 10000}
sim = pybamm.Simulation(model, parameter_values = param, solver=solver) 
solution = sim.solve( initial_soc=SOC_0, t_eval = t_eval)
solutions.append(solution)


param.update({
        "Negative electrode diffusivity [m2.s-1]": modified_graphite_diffusivity_PeymanMPM,
})
solver = pybamm.CasadiSolver(mode="safe", dt_max=1) #, extra_options_setup={"max_num_steps": 10000}
sim = pybamm.Simulation(model, parameter_values = param, solver=solver) 
solution = sim.solve( initial_soc=SOC_0, t_eval = t_eval)
solutions.append(solution)


# %matplotlib inline

labels = ["Constant $D_p$ and $D_n$", "$D_p(c_s)$", "$D_p(c_s)$ and $D_n(c_s)$"]

fig, ax = plt.subplots(6,1, figsize=(4,10), sharex=True)
ax = ax.flatten()
linestyles = ['--',':','-.']*2
sim_colors = ['r','b','g']

t_transition = []
for l, solution in enumerate(solutions):
    # data = data_sets[l]
    t = solution["Time [s]"].entries
    x = solution["x [m]"].entries[:, 0]
    # AhT = solution["Discharge capacity [A.h]"].entries
    x_plot = t
    xlabel = "Time [s]"
    # x_plot = solution["Discharge capacity [A.h]"].entries
    # xlabel = "AhT"
    data_color = 'k' 
    sim_color = sim_colors[l]
    sim_ls = linestyles[l]
    I = solution['C-rate']
    if l ==0:
        ax[0].semilogx(data.t,data.I/param['Nominal cell capacity [A.h]'], label='Data',color = data_color)
    ax[0].semilogx(x_plot, I(t), linestyle=sim_ls, color = sim_color, label=labels[l])
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel('C-rate [A/Ah]')
    # ax[0].legend(loc = "lower left",  prop={'size': 10})

    V = solution['Terminal voltage [V]']
    if l ==0:
        ax[1].semilogx(data.t, data.V,color = data_color)
    ax[1].semilogx(x_plot, V(t),linestyle=sim_ls, color = sim_color)
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel('Terminal voltage [V]')
    ax[1].set_ylim([0,2])
    # ax[0,1].legend(bbox_to_anchor=(0.5, 1.2), fancybox=True, ncol=len(SOC_0), prop={'size': 10}) #, 

    c_n_avg = solution['R-averaged negative particle concentration'](t=t, x=x[0])
    soc = (c_n_avg - 0)/(0.8321-0)
    e_soc = (solution['R-averaged negative particle concentration'](t=data.t, x=x[0])- 0)/(0.8321-0) -  data.SOC
    x = solution["x [m]"].entries[:, 0]
    if l ==0:
        ax[2].semilogx(data.t, data.SOC,color = data_color, label = 'Data')
    ax[2].semilogx(x_plot, soc,linestyle=sim_ls, color = sim_color, label = labels[l])  # can evaluate at arbitrary x (single representative particle)
    ax[2].set_xlabel(xlabel)
    ax[2].set_ylabel('SOC')

    c_e_p= solution['Positive electrolyte concentration']
    # x = solution["x [m]"].entries[:, 0]
    ax[3].semilogx(x_plot, c_e_p (t=t, x = x[-1]),color = sim_color )  # can evaluate at arbitrary x (single representative particle)
    ax[3].set_xlabel(xlabel)
    ax[3].set_ylabel('$c_{e,p}(L_p)$') # [mol.m-3]
    ax[3].set_ylim([0,1])

    # c_e_n = solution['X-averaged negative electrolyte concentration']
    # # x = solution["x [m]"].entries[:, 0]
    # ax[4].semilogx(x_plot, c_e_n (t=t),color = sim_color )  # can evaluate at arbitrary x (single representative particle)
    # ax[4].set_xlabel(xlabel)
    # ax[4].set_ylabel('$c_{e,n}$')

    c_s_p_surf = solution['Positive particle surface concentration']
    x = solution["x [m]"].entries[:, 0]
    ax[4].semilogx(x_plot, c_s_p_surf(t=t, x=x[-1]),color = sim_color )  # can evaluate at arbitrary x (single representative particle)
    ax[4].set_xlabel(xlabel)
    ax[4].set_ylabel('$c_{s,surf,p}$')


    # # Dn = solution['X-averaged negative particle effective diffusivity [m2.s-1]'](t=t, x=x[0],r=[0])
    # Dp = solution['Positive particle effective diffusivity [m2.s-1]'](t=t, x=x[-21],r=[0])
    # # ax[7].semilogx(x_plot, Dn, linestyle='-', color = sim_color, label = "$D_n$")  # can evaluate at arbitrary x (single representative particle)
    # ax[5].semilogx(x_plot, Dp, color = sim_color, label = "$D_p$")  # can evaluate at arbitrary x (single representative particle)
    # ax[5].set_ylabel('$D_p$')
    # # ax[5].legend(loc = 'upper left', ncol=3)
    # ax[5].set_ylim([0, 1.5e-14])

    c_s_n_surf = solution['Negative particle surface concentration']
    x = solution["x [m]"].entries[:, 0]
    ax[5].semilogx(x_plot, c_s_n_surf(t=t, x=x[0]),color = sim_color )  # can evaluate at arbitrary x (single representative particle)
    ax[5].set_xlabel(xlabel)
    ax[5].set_ylabel('$c_{s,surf,n}$')


    # i0 = solution[ 'Negative electrode exchange current density [A.m-2]']
    # x = solution["x [m]"].entries[:, 0]
    # ax[7].plot(x_plot, i0(t=t, x=x[0]),color = sim_color )  # can evaluate at arbitrary x (single representative particle)
    # ax[7].set_xlabel(xlabel)
    # ax[7].set_ylabel('$i_{0,n}$ [A.m-2]')

    # i0 = solution[ 'Positive electrode exchange current density [A.m-2]']
    # x = solution["x [m]"].entries[:, 0]
    # ax[8].plot(x_plot, i0(t=t, x=x[-1]),color = sim_color )  # can evaluate at arbitrary x (single representative particle)
    # ax[8].set_xlabel(xlabel)
    # ax[8].set_ylabel('$i_{0,p}$ [A.m-2]')

    # eta = solution['Negative electrode reaction overpotential [V]']
    # phi = solution['Negative electrolyte potential [V]']
    # x = solution["x [m]"].entries[:, 0]
    # ax[1,3].plot(x_plot, eta(t=t, x=x[0]),color = sim_color )  # can evaluate at arbitrary x (single representative particle)
    # ax[1,3].plot(x_plot, phi(t=t, x=x[0]),color = sim_color, linestyle = ':' )  # can evaluate at arbitrary x (single representative particle)
    # ax[1,3].set_xlabel(xlabel)
    # ax[1,3].set_ylabel('$\eta_n$ [V]')

    # plot transition lines 
    c = 'grey'
    if l  == len(solutions)-1:
        t_transition.append(t[np.where(c_e_p (t=t, x = x[-1]) <=0)[0][0]]) # cep first =0
        t_transition.append(t[np.where(abs(np.diff(c_s_p_surf(t=t, x = x[-1])))<1e-3)[0][0]]) # cep first =0
        # t_transition.append(t[np.where(c_s_n_surf(t=t, x = x[0]) < min(c_s_n_surf(t=t, x = x[0])*1.1))[0][0]])
        # t_transition.append(t[np.where(i0(t=t, x = x[0]) == min(i0(t=t, x = x[0])))[0][0]])
        t_transition.append(t[np.where(np.diff(c_s_n_surf(t=t, x = x[0]))>-1e-6)][0])
    
        #for each subplot
        for i, axs in enumerate(fig.get_axes()):
            ylims = axs.get_ylim()
            # make a verical line with annotation at each transition time
            if i>2: 
                t_trans = t_transition[i-3]
                axs.plot([t_trans]*2, list(ylims), color = c, linestyle = ':')
                if i in [3,5]: # top annotation
                    axs.annotate(str(i-2), xy=(t_trans, np.diff(ylims)*0.90+ ylims[0]), color = c) #
                else: # bottom annotation
                    axs.annotate(str(i-2), xy=(t_trans, np.diff(ylims)*0.05+ ylims[0]), color = c) #
                axs.set_ylim(ylims) # keep original limits

            else:
                ylims = [ylims[0], ylims[1]*1.05] #make more room at the top
                for r, t_trans in enumerate(t_transition):
                    axs.plot([t_trans]*2, list(ylims), color = c, linestyle = ':')
                    axs.annotate(str(r+1), xy=(t_trans, np.diff(ylims)*0.90+ ylims[0]), color = c)
                axs.plot([80]*2, list(ylims), color = 'm', linestyle = ':')
                axs.annotate('Venting', xy=(90, np.diff(ylims)*0.57+ ylims[0]), color = 'm', rotation=-90) #
                axs.set_ylim(ylims) # keep original limits
            # venting line
            # if i ==0:
            # axs.plot([80]*2, list(ylims), color = 'm', linestyle = ':')
            # axs.annotate('Venting', xy=(90, np.diff(ylims)*0.57+ ylims[0]), color = 'm', rotation=-90) #
        # axs.grid()



# ax[0,1].legend(handles,loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=True, ncol=3, prop={'size': 10})
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', prop={'size': 10}, ncol = 2, bbox_to_anchor=(0.5, 1.05), fancybox=True) #ncol=len(labels)
plt.xlim([t_eval[0],t_eval[-1]])
plt.tight_layout()
plt.show()

plt.savefig('./fast_discharge/figs/effect_D_c_long.eps', format='eps')