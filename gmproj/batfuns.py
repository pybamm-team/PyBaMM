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

def set_rc_params(rcParams):

    rcParams["lines.markersize"] = 5
    rcParams["lines.linewidth"] = 2
    rcParams["xtick.minor.visible"] = True
    rcParams["ytick.minor.visible"] = True
    rcParams["font.size"] = 12
    rcParams["legend.fontsize"] = 10
    rcParams["legend.frameon"] = False
    rcParams["font.family"] = 'serif'
    rcParams['font.serif'] = 'Times New Roman'
    rcParams['mathtext.rm'] = 'serif'
    rcParams['mathtext.it'] = 'serif:italic'
    rcParams['mathtext.bf'] = 'serif:bold'
    rcParams['mathtext.fontset'] = 'custom'
    rcParams["savefig.bbox"] = "tight"
    rcParams["axes.grid"] = True
    rcParams["axes.axisbelow"] = True
    rcParams["grid.linestyle"] = "--"
    rcParams["grid.color"] = (0.8, 0.8, 0.8)
    rcParams["grid.alpha"] = 0.5
    rcParams["grid.linewidth"] = 0.5
    rcParams['figure.dpi'] = 150
    rcParams['savefig.dpi'] = 600
    rcParams['figure.max_open_warning']=False
    
    return rcParams

def nmc_volume_change_mohtat(sto,c_s_max):
    t_change = -1.10/100*(1-sto)
    return t_change

def graphite_volume_change_mohtat(sto,c_s_max):
    stoichpoints = np.array([0,0.12,0.18,0.24,0.50,1])
    thicknesspoints = np.array([0,2.406/100,3.3568/100,4.3668/100,5.583/100,13.0635/100])
    x = [sto]
    t_change = pybamm.Interpolant(stoichpoints, thicknesspoints, x, name=None, interpolator='linear', extrapolate=True, entries_string=None)
    return t_change


def get_parameter_values():
    parameter_values = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Mohtat2020)
    parameter_values.update(
        {
            # mechanical properties
            "Positive electrode Poisson's ratio": 0.3,
            "Positive electrode Young's modulus [Pa]": 375e9,
            "Positive electrode reference concentration for free of deformation [mol.m-3]": 0,
            "Positive electrode partial molar volume [m3.mol-1]": -7.28e-7,
            "Positive electrode volume change": nmc_volume_change_mohtat,
            # Loss of active materials (LAM) model
            "Positive electrode LAM constant exponential term": 2,
            "Positive electrode critical stress [Pa]": 375e6,
            # mechanical properties
            "Negative electrode Poisson's ratio": 0.2,
            "Negative electrode Young's modulus [Pa]": 15e9,
            "Negative electrode reference concentration for free of deformation [mol.m-3]": 0,
            "Negative electrode partial molar volume [m3.mol-1]": 3.1e-6,
            "Negative electrode volume change": graphite_volume_change_mohtat,
            # Loss of active materials (LAM) model
            "Negative electrode LAM constant exponential term": 2,
            "Negative electrode critical stress [Pa]": 60e6,
            # Other
            "Cell thermal expansion coefficient [m.K-1]": 1.48E-6,
            "Lower voltage cut-off [V]": 3.0,
            # Initializing Particle Concentration
            # "Initial concentration in negative electrode [mol.m-3]": x100*parameter_values["Maximum concentration in negative electrode [mol.m-3]"],
            # "Initial concentration in positive electrode [mol.m-3]": y100*parameter_values["Maximum concentration in positive electrode [mol.m-3]"]
        },
        check_already_exists=False,
    )
    return parameter_values


def split_long_string(title, max_words=None):
    """Get title in a nice format"""
    max_words = max_words or pybamm.settings.max_words_in_line
    words = title.split()
    # Don't split if fits on one line, don't split just for units
    if len(words) <= max_words or words[max_words].startswith("["):
        return title
    else:
        first_line = (" ").join(words[:max_words])
        second_line = (" ").join(words[max_words:])
        return first_line + "\n" + second_line

def cycle_adaptive_simulation(model, parameter_values, experiment,SOC_0=1, save_at_cycles=None):
    experiment_one_cycle = pybamm.Experiment(
        experiment.operating_conditions_cycles[:1],
        termination=experiment.termination_string,
        cccv_handling=experiment.cccv_handling,
    )
    Vmin = 3.0
    Vmax = 4.2
    esoh_model = pybamm.lithium_ion.ElectrodeSOH()
    esoh_sim = pybamm.Simulation(esoh_model, parameter_values=parameter_values)
    param = model.param
    esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(parameter_values, param)
    Cn = parameter_values.evaluate(param.n.cap_init)
    Cp = parameter_values.evaluate(param.p.cap_init)
    eps_n = parameter_values["Negative electrode active material volume fraction"]
    eps_p = parameter_values["Positive electrode active material volume fraction"]
    C_over_eps_n = Cn / eps_n
    C_over_eps_p = Cp / eps_p
    c_n_max = parameter_values.evaluate(param.n.prim.c_max)
    c_p_max = parameter_values.evaluate(param.p.prim.c_max)
    n_Li_init = parameter_values.evaluate(param.n_Li_particles_init)
    
    esoh_sol = esoh_sim.solve(
        [0],
        inputs={"V_min": Vmin, "V_max": Vmax, "C_n": Cn, "C_p": Cp, "n_Li": n_Li_init},
        solver=pybamm.AlgebraicSolver(),
    )

    parameter_values.update(
        {
            "Initial concentration in negative electrode [mol.m-3]": esoh_sol[
                "x_100"
            ].data[0]
            * c_n_max,
            "Initial concentration in positive electrode [mol.m-3]": esoh_sol[
                "y_100"
            ].data[0]
            * c_p_max,
            
        }
    )

    sim_ode = pybamm.Simulation(
        model, experiment=experiment_one_cycle, parameter_values=parameter_values,
        solver=pybamm.CasadiSolver("safe")
    )
    sol0 = sim_ode.solve(initial_soc=SOC_0)
    model = sim_ode.solution.all_models[0]
    cap0 = sol0.summary_variables["Capacity [A.h]"]

    def sol_to_y(sol, loc="end"):
        if loc == "start":
            pos = 0
        elif loc == "end":
            pos = -1
        model = sol.all_models[0]
        n_Li = sol["Total lithium in particles [mol]"].data[pos].flatten()
        Cn = sol["Negative electrode capacity [A.h]"].data[pos].flatten()
        Cp = sol["Positive electrode capacity [A.h]"].data[pos].flatten()
        # y = np.concatenate([n_Li, Cn, Cp])
        y = n_Li
        for var in model.initial_conditions:
            if var.name not in [
                "X-averaged negative particle concentration",
                "X-averaged positive particle concentration",
                "Discharge capacity [A.h]",
            ]:
                value = sol[var.name].data
                if value.ndim == 1:
                    value = value[pos]
                elif value.ndim == 2:
                    value = value[:, pos]
                elif value.ndim == 3:
                    value = value[:, :, pos]
                y = np.concatenate([y, value.flatten()])
        return y

    def y_to_sol(y, esoh_sim, model):
        n_Li = y[0]
        Cn = C_over_eps_n * y[1]
        Cp = C_over_eps_p * y[2]

        esoh_sol = esoh_sim.solve(
            [0],
            inputs={"V_min": Vmin, "V_max": Vmax, "C_n": Cn, "C_p": Cp, "n_Li": n_Li},
        )
        esoh_sim.built_model.set_initial_conditions_from(esoh_sol)
        ics = {}
        x_100 = esoh_sol["x_100"].data[0]
        y_100 = esoh_sol["y_100"].data[0]
        x_0 = esoh_sol["x_0"].data[0]
        y_0 = esoh_sol["y_0"].data[0]
        start = 1
        for var in model.initial_conditions:
            if var.name == "X-averaged negative particle concentration":
                ics[var.name] = ((x_100-x_0)*SOC_0+x_0) * np.ones((model.variables[var.name].size, 2))
            elif var.name == "X-averaged positive particle concentration":
                ics[var.name] = ((y_100-y_0)*SOC_0+y_0)  * np.ones((model.variables[var.name].size, 2))
            elif var.name == "Discharge capacity [A.h]":
                ics[var.name] = np.zeros(1)
            else:
                end = start + model.variables[var.name].size
                ics[var.name] = y[start:end, np.newaxis]
                start = end
        model.set_initial_conditions_from(ics)
        return pybamm.Solution(
            [np.array([0])],
            model.concatenated_initial_conditions.evaluate()[:, np.newaxis],
            model,
            {},
        )

    def dydt(t, y):
        if y[0] < 0 or y[1] < 0 or y[2] < 0:
            return 0 * y

        # print(t)
        # Set up based on current value of y
        y_to_sol(
            y,
            esoh_sim,
            sim_ode.op_conds_to_built_models[
                experiment_one_cycle.operating_conditions[0]["electric"]
            ],
        )

        # Simulate one cycle
        sol = sim_ode.solve()

        dy = sol_to_y(sol) - y

        return dy

    if experiment.termination == {}:
        event = None
    else:

        def capacity_cutoff(t, y):
            sol = y_to_sol(y, esoh_sim, model)
            cap = pybamm.make_cycle_solution([sol], esoh_solver, True)[1]["Capacity [A.h]"]
            return cap / cap0 - experiment_one_cycle.termination["capacity"][0] / 100

        capacity_cutoff.terminal = True

    num_cycles = len(experiment.operating_conditions_cycles)
    if save_at_cycles is None:
        t_eval = np.arange(1, num_cycles + 1)
    elif save_at_cycles == -1:
        t_eval = None
    else:
        t_eval = np.arange(1, num_cycles + 1, save_at_cycles)
    y0 = sol_to_y(sol0, loc="start")
    timer = pybamm.Timer()
    sol = solve_ivp(
        dydt,
        [1, num_cycles],
        y0,
        t_eval=t_eval,
        events=capacity_cutoff,
        first_step=10,
        method="RK23",
        atol=1e-2,
        rtol=1e-2,
    )
    time = timer.time()

    all_sumvars = []
    for idx in range(sol.y.shape[1]):
        fullsol = y_to_sol(sol.y[:, idx], esoh_sim, model)
        sumvars = pybamm.make_cycle_solution([fullsol], esoh_solver, True)[1]
        all_sumvars.append(sumvars)

    all_sumvars_dict = {
        key: np.array([sumvars[key] for sumvars in all_sumvars])
        for key in all_sumvars[0].keys()
    }
    all_sumvars_dict["Cycle number"] = sol.t
    
    all_sumvars_dict["cycles evaluated"] = sol.nfev
    all_sumvars_dict["solution time"] = time
    
    return all_sumvars_dict

def plot(all_sumvars_dict,esoh_data):
    esoh_vars = ["x_0", "y_0", "x_100", "y_100", "C_n", "C_p"]
    # esoh_vars = ["Capacity [A.h]", "Loss of lithium inventory [%]",
    #              "Loss of active material in negative electrode [%]",
    #              "Loss of active material in positive electrode [%]"]
    fig, axes = plt.subplots(3,2,figsize=(7,7))
    for k, name in enumerate(esoh_vars):
        ax = axes.flat[k]
        ax.plot(all_sumvars_dict["Cycle number"],all_sumvars_dict[name],"ro")
        ax.plot(esoh_data["N"],esoh_data[name],"kx")
#         ax.scatter(all_sumvars_dict["Cycle number"],all_sumvars_dict[name],color="r")
    #     ax.plot(long_sol.summary_variables[name],"b-")
        ax.set_title(split_long_string(name))
        if k>3:
            ax.set_xlabel("Cycle number")
    # fig.subplots_adjust(bottom=0.4)
    fig.legend(["Acc Sim"] + ["Reported"], 
           loc="lower center", ncol=1, fontsize=11)
    fig.tight_layout()
    return fig

def plot1(all_sumvars_dict,esoh_data):
    esoh_vars = ["Capacity [A.h]","n_Li"]
    esoh_data["Capacity [A.h]"]=esoh_data["Cap"]
    param = pybamm.LithiumIonParameters()
    esoh_data["n_Li"]= 3600/param.F.value*(esoh_data["y_100"]*esoh_data["C_p"]+esoh_data["x_100"]*esoh_data["C_n"])
    fig, axes = plt.subplots(2,1,figsize=(7,7))
    for k, name in enumerate(esoh_vars):
        ax = axes.flat[k]
        ax.plot(all_sumvars_dict["Cycle number"],all_sumvars_dict[name],"ro")
        ax.plot(esoh_data["N"],esoh_data[name],"kx")
        ax.set_title(split_long_string(name))
        ax.set_xlabel("Cycle number")
    fig.legend(["Acc Sim"] + ["Reported"], 
           loc="upper right", ncol=1, fontsize=11)
    fig.tight_layout()
    return fig
def plotc(all_sumvars_dict,esoh_data):
    esoh_vars = ["x_100", "y_0", "C_n", "C_p", "Capacity [A.h]", "Loss of lithium inventory [%]"]
    fig, axes = plt.subplots(3,2,figsize=(7,7))
    for k, name in enumerate(esoh_vars):
        ax = axes.flat[k]
        ax.plot(all_sumvars_dict["Cycle number"],all_sumvars_dict[name],"ro")
        ax.plot(esoh_data["N"],esoh_data[name],"kx")
        ax.set_title(split_long_string(name))
        if k ==2 or k==3:
            ax.set_ylim([3,6.2])
        if k>3:
            ax.set_xlabel("Cycle number")
    fig.legend(["Sim"] + ["Data"], 
           loc="lower center",bbox_to_anchor=[0.5,-0.02], ncol=1, fontsize=11)
    fig.tight_layout()
    return fig

def plotc2(all_sumvars_dict1,all_sumvars_dict2,esoh_data):
    esoh_vars = ["x_100", "y_0", "C_n", "C_p", "Capacity [A.h]", "Loss of lithium inventory [%]"]
    fig, axes = plt.subplots(3,2,figsize=(7,7))
    for k, name in enumerate(esoh_vars):
        ax = axes.flat[k]
        ax.plot(all_sumvars_dict1["Cycle number"],all_sumvars_dict1[name],"r.")
        ax.plot(all_sumvars_dict2["Cycle number"],all_sumvars_dict2[name],"bv")
        ax.plot(esoh_data["N"],esoh_data[name],"kx")
        ax.set_title(split_long_string(name))
        if k ==2 or k==3:
            ax.set_ylim([3,6.2])
        if k>3:
            ax.set_xlabel("Cycle number")
    fig.legend(["Old", "New" , "Data"], 
           loc="lower center",bbox_to_anchor=[0.5,-0.02], ncol=1, fontsize=11)
    fig.tight_layout()
    return fig

def plotcomp(all_sumvars_dict0,all_sumvars_dict1):
    esoh_vars = ["x_100", "y_0", "C_n", "C_p", "Capacity [A.h]", "Loss of lithium inventory [%]"]
    fig, axes = plt.subplots(3,2,figsize=(7,7))
    for k, name in enumerate(esoh_vars):
        ax = axes.flat[k]
        ax.plot(all_sumvars_dict0["Cycle number"],all_sumvars_dict0[name],"k")
        ax.plot(all_sumvars_dict1["Cycle number"],all_sumvars_dict1[name],"b")
        ax.set_title(split_long_string(name))
        if k ==2 or k==3:
            ax.set_ylim([3,6.2])
        if k>3:
            ax.set_xlabel("Cycle number")
    fig.legend(["Baseline"] + ["Sim"], 
           loc="lower center",bbox_to_anchor=[0.5,-0.02], ncol=1, fontsize=11)
    fig.tight_layout()
    return fig

def plotcomplong(all_sumvars_dict0,all_sumvars_dict1,all_sumvars_dict2):
    esoh_vars = ["x_100", "y_0", "C_n", "C_p", "Capacity [A.h]", "Loss of lithium inventory [%]"]
    fig, axes = plt.subplots(3,2,figsize=(7,7))
    for k, name in enumerate(esoh_vars):
        ax = axes.flat[k]
        ax.plot(all_sumvars_dict0["Cycle number"],all_sumvars_dict0[name],"kx")
        ax.plot(all_sumvars_dict1["Cycle number"],all_sumvars_dict1[name],"bo")
        ax.plot(all_sumvars_dict2["Cycle number"],all_sumvars_dict2[name],"m.")
        ax.set_title(split_long_string(name))
        if k ==2 or k==3:
            ax.set_ylim([3,6.2])
        if k>3:
            ax.set_xlabel("Cycle number")
    fig.legend(["Baseline"] + ["Accl Sim"] + ["Long Sim"], 
            loc="lower center",bbox_to_anchor=[0.5,-0.02], ncol=1, fontsize=11)
    fig.tight_layout()
    return fig


def load_data_calendar(cell,eSOH_DIR,oCV_DIR):
    param = pybamm.LithiumIonParameters() 
    cell_no = f'{cell:02d}'
    dfe=pd.read_csv(eSOH_DIR+"aging_param_cell_"+cell_no+".csv")
    dfo=pd.read_csv(oCV_DIR+"ocv_data_cell_"+cell_no+".csv")
    # if cell_no=='24':
    #     dfe = dfe.drop(dfe.index[0])
    #     dfe = dfe.reset_index(drop=True)
    #     dfe['Time']=dfe['Time']-dfe['Time'][0]
    dfe['N']=dfe['Time']
    dfe["Capacity [A.h]"]=dfe["Cap"]
    dfe["n_Li"]= 3600/param.F.value*(dfe["y_100"]*dfe["C_p"]+dfe["x_100"]*dfe["C_n"])
    dfe["Loss of lithium inventory [%]"]=(1-dfe["n_Li"]/dfe["n_Li"][0])*100
    N =dfe.N.unique()

    # print("Cycle Numbers:")
    # print(*N, sep = ", ") 
    # print(len(N_0))
    # print(len(dfo_0))
    # rev_exp = []
    # irrev_exp = []

    return cell_no,dfe,N,dfo

def init_exp_calendar(cell_no,dfe,param,parameter_values):
    # dfe_0 = dfe[dfe['N']==N[0]]
    C_n_init = dfe['C_n'][0]
    C_p_init = dfe['C_p'][0]
    y_0_init = dfe['y_0'][0] 
    eps_n_data = parameter_values.evaluate(C_n_init*3600/(param.n.L * param.n.prim.c_max * param.F* param.A_cc))
    eps_p_data = parameter_values.evaluate(C_p_init*3600/(param.p.L * param.p.prim.c_max * param.F* param.A_cc))
    # cs_p_init = parameter_values.evaluate(y_0_init* param.c_p_max)
    if cell_no=='22':
        SOC_0 = 1
        Temp = 45
    #     x_init = esoh_sol["x_100"].data[0] 
    #     y_init = esoh_sol["y_100"].data[0] 
    elif cell_no=='23':
        SOC_0 = 0.5
        Temp = 45
    elif cell_no=='24':
        SOC_0 = 1
        Temp = -5
    elif cell_no=='25':
        SOC_0 = 0.5
        Temp = -5
        
    return eps_n_data,eps_p_data,SOC_0,Temp#,x_init,y_init


def load_data(cell,eSOH_DIR,oCV_DIR): 
    param = pybamm.LithiumIonParameters() 
    cell_no = f'{cell:02d}'
    dfe=pd.read_csv(eSOH_DIR+"aging_param_cell_"+cell_no+".csv")
    dfe_0=pd.read_csv(eSOH_DIR+"aging_param_cell_"+cell_no+".csv")
    dfo_0=pd.read_csv(oCV_DIR+"ocv_data_cell_"+cell_no+".csv")
    # if cell_no=='13':
    #     dfo_d=dfo_0[dfo_0['N']==dfe['N'].iloc[-5]]
    #     dfo_0=dfo_0.drop(dfo_d.index.values)
    #     dfo_0=dfo_0.reset_index(drop=True)
    #     dfe = dfe.drop(dfe.index[-5])
    #     dfe = dfe.reset_index(drop=True)
    # Remove first RPT
    dfe = dfe.drop(dfe.index[0])
    dfe = dfe.reset_index(drop=True)
    # dfo_d=dfo_0[dfo_0['N']==0]
    # dfo_0=dfo_0.drop(dfo_d.index.values)
    if cell_no=='13':
        dfe = dfe.drop(dfe.index[-1])
        dfe = dfe.reset_index(drop=True)
        dfe_0 = dfe_0.drop(dfe_0.index[-1])
        dfe_0 = dfe_0.reset_index(drop=True)
    dfe['N']=dfe['N']-dfe['N'][0]
    cycles = np.array(dfe['N'].astype('int'))
    cycles=cycles-1
    cycles[0]=cycles[0]+1
    dfe['N_mod'] = cycles
    N =dfe.N.unique()
    N_0 = dfe_0.N.unique()
    # print("Cycle Numbers:")
    # print(*N, sep = ", ") 
    # print(len(N_0))
    # print(len(dfo_0))
    rev_exp = []
    irrev_exp = []
    dfe["Capacity [A.h]"]=dfe["Cap"]
    dfe["n_Li"]= 3600/param.F.value*(dfe["y_100"]*dfe["C_p"]+dfe["x_100"]*dfe["C_n"])
    dfe["Loss of lithium inventory [%]"]=(1-dfe["n_Li"]/dfe["n_Li"][0])*100

    for i in range(len(N_0)-1):
        # print(i)
        dfo = dfo_0[dfo_0['N']==N_0[i+1]]
        # print(max(dfo['E'])-min(dfo['E']))
        rev_exp.append(max(dfo['E'])-min(dfo['E']))
    dfe['rev_exp']=rev_exp
    # print('Reversible Expansion')

    dfo_1 = dfo_0[dfo_0['N']==N_0[1]]
    for i in range(len(N_0)-1):
        # print(i)
        dfo = dfo_0[dfo_0['N']==N_0[i+1]]
        # print(max(dfo['E'])-min(dfo['E']))
        irrev_exp.append(min(dfo['E'])-min(dfo_1['E']))
    dfe['irrev_exp']=irrev_exp
    # print('Irreversible Expansion')
    return cell_no,dfe,dfe_0,dfo_0,N,N_0

def init_exp(cell_no,dfe,spm,parameter_values):
    # dfe_0 = dfe[dfe['N']==N[0]]
    param = spm.param
    C_n_init = dfe['C_n'][0]
    C_p_init = dfe['C_p'][0]
    y_0_init = dfe['y_0'][0] 
    eps_n_data = parameter_values.evaluate(C_n_init*3600/(param.n.L * param.n.prim.c_max * param.F* param.A_cc))
    eps_p_data = parameter_values.evaluate(C_p_init*3600/(param.p.L * param.p.prim.c_max * param.F* param.A_cc))
    # cs_p_init = parameter_values.evaluate(y_0_init* param.c_p_max) 
    if (int(cell_no)-1)//3 ==0:
        c_rate_c = 'C/5'
        c_rate_d = 'C/5'
        dis_set = " until 3V"
    elif (int(cell_no)-1)//3 ==1:
        c_rate_c = '1.5C'
        c_rate_d = '1.5C'
        dis_set = " until 3V"
    elif (int(cell_no)-1)//3 ==2:
        c_rate_c = '2C'
        c_rate_d = '2C'
        dis_set = " until 3V"
    elif (int(cell_no)-1)//3 ==3:
        c_rate_c = 'C/5'
        c_rate_d = '1.5C'
        dis_set = " until 3V"
    elif (int(cell_no)-1)//3 ==4:
        c_rate_c = 'C/5'
        c_rate_d = 'C/5'
        dis_set = " for 150 min"
    elif (int(cell_no)-1)//3 ==5:
        c_rate_c = 'C/5'
        c_rate_d = '1.5C'
        dis_set = " for 20 min"
    elif (int(cell_no)-1)//3 ==6:
        c_rate_c = '1.5C'
        c_rate_d = '1.5C'
        dis_set = " for 20 min"
    if int(cell_no)%3 == 0:
        Temp = 45
    if int(cell_no)%3 == 1:
        Temp = 25
    if int(cell_no)%3 == 2:
        Temp = -5
    SOC_0 = 1
    return eps_n_data,eps_p_data,c_rate_c,c_rate_d,dis_set,Temp,SOC_0