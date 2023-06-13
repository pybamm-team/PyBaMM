import pybamm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import dfols
import signal
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy import interpolate
from stopit import threading_timeoutable as timeoutable
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))))
sys.path.append("C:/Users/spannala/PyBaMM/gmproj/")
from stopit import threading_timeoutable as timeoutable
from batfuns import *
plt.rcParams = set_rc_params(plt.rcParams)

eSOH_DIR = "../data/esoh/"
oCV_DIR = "../data/ocv/"
fig_DIR = "../figures/figures_fit/"
res_DIR = "../data/results_fit/"
cyc_DIR = "../data/cycling/"

eSOH_DIR = "C:/Users/spannala/PyBaMM/gmproj/data/esoh/"
oCV_DIR =  "C:/Users/spannala/PyBaMM/gmproj/data/ocv/"
fig_DIR =  "C:/Users/spannala/PyBaMM/gmproj/figures/figures_fit/"
res_DIR = "C:/Users/spannala/PyBaMM/gmproj/data/results_fit/"
cyc_DIR = "C:/Users/spannala/PyBaMM/gmproj/data/cycling/"
# %matplotlib widget

parameter_values = get_parameter_values()

spm = pybamm.lithium_ion.SPM(
    {
        "SEI": "ec reaction limited",
        # "loss of active material": ("stress-driven","none"),
        "loss of active material": "stress-driven",
        "lithium plating": "irreversible",
        "stress-induced diffusion": "false",
    }
)
# spm.print_parameter_info()
param=spm.param

cell = 1

cell_no,dfe,dfe_0,dfo_0,N,N_0 = load_data(cell,eSOH_DIR,oCV_DIR)
eps_n_data,eps_p_data,c_rate_c,c_rate_d,dis_set,Temp,SOC_0 = init_exp(cell_no,dfe,spm,parameter_values)

pybamm.set_logging_level("WARNING")
# pybamm.set_logging_level("NOTICE")
drive_cycle = pd.read_csv(cyc_DIR+'peyman_drive_cycle_current'+'.csv', comment="#", header=None).to_numpy()
experiment = pybamm.Experiment(
    [
        ("Discharge at "+c_rate_d+dis_set,
         "Rest for 5 min",
         "Charge at "+c_rate_c+" until 4.2V", 
         "Hold at 4.2V until C/100")
    ] *dfe.N.iloc[-1],
    termination="50% capacity",
#     cccv_handling="ode",
)


ic = 2
blam_p = [1e-6,1e-7,1e-8]
blam_n = [1e-5,1e-6,1e-7]
alam_p = [1e-7,1e-8,1e-9]
alam_n = [1e-6,1e-7,1e-8]
k_pl = 1e-9
x = np.array([1.0,1.0,1.0,1.0,1.0,1.0])

parameter_values = get_parameter_values()
parameter_values.update(
    {
        "Negative electrode active material volume fraction": eps_n_data,
        "Positive electrode active material volume fraction": eps_p_data,
        "Initial temperature [K]": 273.15+Temp,
        "Ambient temperature [K]": 273.15+Temp,
        # "Positive electrode LAM constant proportional term [s-1]": 1.27152e-07,
        # "Negative electrode LAM constant proportional term [s-1]": 1.27272e-06,
        # "Positive electrode LAM constant exponential term": 1.1992,
        # "Negative electrode LAM constant exponential term": 1.1992,
        "SEI kinetic rate constant [m.s-1]":  4.60788219e-16, #1.08494281e-16 , 
        "EC diffusivity [m2.s-1]": 4.56607447e-19,#8.30909086e-19,
        "SEI growth activation energy [J.mol-1]": 1.87422275e+04,#1.58777981e+04,
        # "Lithium plating kinetic rate constant [m.s-1]": 0,
        "Initial inner SEI thickness [m]": 0e-09,
        "Initial outer SEI thickness [m]": 5e-09,
        "SEI resistivity [Ohm.m]": 30000.0,
        "Positive electrode LAM additional term [s-1]": 1.27152e-07,
        "Negative electrode LAM additional term [s-1]": 1.27152e-07,
        "Positive electrode LAM constant proportional term [s-1]": x[0]*blam_p[ic],
        "Negative electrode LAM constant proportional term [s-1]": x[1]*blam_n[ic],
        "Positive electrode LAM constant exponential term": x[2]*2,
        "Negative electrode LAM constant exponential term": x[2]*2,
        "Lithium plating kinetic rate constant [m.s-1]": x[3]*k_pl,
        "Positive electrode LAM additional term [s-1]": x[4]*alam_p[ic],
        "Negative electrode LAM additional term [s-1]": x[5]*alam_n[ic],
    },
    check_already_exists=False,
)


all_sumvars_dict = cycle_adaptive_simulation(spm, parameter_values, experiment,SOC_0, save_at_cycles=1)

fig = plotc(all_sumvars_dict,dfe);
fig.savefig(fig_DIR +'fast_sim_'+cell_no+'_new.png')

ic = 1
blam_p = [1e-6,1e-7,1e-8]
blam_n = [1e-5,1e-6,1e-7]
alam_p = [1e-7,1e-9,1e-9]
alam_n = [1e-6,1e-8,1e-9]
k_pl = 1e-9
 # variables = ["Capacity [A.h]", "Loss of lithium inventory [%]","x_100","y_0"]
    # weights = [1,1/20,5,5]

def objective(model, data):
    return np.array(model.loc[data['N_mod']]["Capacity [A.h]"]) - np.array(data["Capacity [A.h]"])

def multi_objective(model, data):
    variables = ["C_n","C_p","x_100","y_0"]
    weights = [1,1,5,5]
    # variables = ["Capacity [A.h]", "Loss of lithium inventory [%]"]
    # # weights = [1,1/20]
    variables = ["Capacity [A.h]", "Loss of lithium inventory [%]", "C_n", "C_p"]
    weights = [1,1/20,1,1]
    return np.concatenate([
        (np.array(model.loc[data['N_mod']][var]) - np.array(data[var])) * w
        for w,var in zip(weights,variables)
    ]
    )
@timeoutable()
def simulate(x,eps_n_data,eps_p_data,SOC_0,Temp,experiment):
    parameter_values.update(
        {
            "Positive electrode LAM constant proportional term [s-1]": x[0]*blam_p[ic],
            "Negative electrode LAM constant proportional term [s-1]": x[1]*blam_n[ic],
            "Positive electrode LAM constant exponential term": x[2]*2,
            "Negative electrode LAM constant exponential term": x[2]*2,
            "Lithium plating kinetic rate constant [m.s-1]": x[3]*k_pl,
            "Positive electrode LAM additional term [s-1]": 0*x[4]*alam_p[ic],
            "Negative electrode LAM additional term [s-1]": x[4]*alam_n[ic],
            
            "Negative electrode active material volume fraction": eps_n_data,
            "Positive electrode active material volume fraction": eps_p_data,
            "Initial temperature [K]": 273.15+Temp,
            "Ambient temperature [K]": 273.15+Temp,
        },
        check_already_exists=False,
    )
    return cycle_adaptive_simulation(spm, parameter_values, experiment, SOC_0,save_at_cycles=1,drive_cycle=None)
def prediction_error(x):
    try:
        out=[]
        for cell in [1,4,10]:
            cell_no,dfe,dfe_0,dfo_0,N,N_0 = load_data(cell,eSOH_DIR,oCV_DIR)
            eps_n_data,eps_p_data,c_rate_c,c_rate_d,dis_set,Temp,SOC_0 = init_exp(cell_no,dfe,spm,parameter_values)
            # print(f"Cell: {cell_no}")
            if cell == 19:
                experiment = pybamm.Experiment(
                    [
                        ("Run DriveCycle (A)",
                        "Rest for 5 min",
                        "Charge at "+c_rate_c+" until 4.2V", 
                        "Hold at 4.2V until C/50")
                    ] *dfe.N.iloc[-1],
                    # ] *1,
                    drive_cycles={"DriveCycle": drive_cycle},
                    termination="50% capacity",
                #     cccv_handling="ode",
                )
            else:
                experiment = pybamm.Experiment(
                    [
                        ("Discharge at "+c_rate_d+dis_set,
                        "Rest for 5 min",
                        "Charge at "+c_rate_c+" until 4.2V", 
                        "Hold at 4.2V until C/50")
                    ] *dfe.N.iloc[-1],
                    termination="50% capacity",
                #     cccv_handling="ode",
                )
            # print(f"Model")
            model = simulate(x,eps_n_data,eps_p_data,SOC_0,Temp,experiment,timeout=30)
            # print(f"Objective")
            out_t =   multi_objective(pd.DataFrame(model), dfe)
            # out_t[-1] = 5*out_t[-1]
            # print(f"Concat")
            if cell==1:
                out_t = 2*out_t
            out=np.concatenate([out,out_t])
        print(f"x={x}, norm={np.linalg.norm(out)}")
    # except pybamm.SolverError:
    except:
        out=[]
        for cell in [1,4,10]:
            cell_no,dfe,dfe_0,dfo_0,N,N_0 = load_data(cell,eSOH_DIR,oCV_DIR)
            # out_t = np.concatenate([np.array(dfe['Cap'])]*2)
            out_t = np.concatenate([np.array(dfe['Cap'])]*4)
            out=np.concatenate([out, out_t])
        out = 2*np.ones_like(out)
        print(f"Error")
        print(f"x={x}, norm={np.linalg.norm(out)}")
    return out

def train_model():
    timer = pybamm.Timer()
    x0 = np.array([1.0,1.0,0.75,1.0,1.0])
    # print(prediction_error(x0))
    lower = np.array([1e-2, 1e-2, 0.51, 1e-1, 1e-2])
    upper = np.array([1e+2, 1e+2, 1.5, 1e+1, 1e+2])
    dfo_opts = {
        "init.random_initial_directions":True,
        "init.run_in_parallel": True,
    }
    soln_dfols = dfols.solve(prediction_error, x0,bounds=(lower, upper), rhoend=1e-2, user_params=dfo_opts)
    print(timer.time())
    return soln_dfols
def sim_train(df):
    soln_dfols = train_model()
    xsol = soln_dfols.x
    # print(xsol[0]*2e-2/3600)
    # print(xsol[1]*2e-1/3600)
    # print(xsol[2]*1.6e-16)
    df['x_0'][0]=round(xsol[0],4)*blam_p[ic]
    df['x_1'][0]=round(xsol[1],4)*blam_n[ic]
    df['x_2'][0]=round(xsol[2],4)*2
    df['x_3'][0]=round(xsol[3],4)*k_pl
    # df['x_4'][0]=round(xsol[4],4)*alam_p[ic]
    df['x_4'][0]=round(xsol[4],4)*alam_n[ic]
    df['obj'][0]=soln_dfols.f
    return xsol,df

df_x = pd.DataFrame(columns=['x_0','x_1','x_2','x_3','x_4','obj'], index=[0])

x,df_x = sim_train(df_x)

sim_des="plating_mech_1_4_10_addn_LAM_split"
df_x.to_csv(res_DIR + "cycl_train_"+sim_des+".csv")