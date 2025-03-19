import numpy as np
import torch
import torch.nn as nn
from torch import autograd
import torch.optim as optim
import os
import pandas as pd
import matplotlib.pyplot as plt
from energy import sampling, convex_hull, CommonTangent, calculate_S_config_total, calculate_S_vib_total


global  _eps
_eps = 1e-7


# loss function 
def collocation_loss_all_pts(mu, x, phase_boundarys_fixed_point, GibbsFunction, params_list, alpha_miscibility, T=300):
    """
    Calculate the collocation points loss for all datapoints (that way we don't need hessian loss and common tangent loss, everything is converted into collocation loss)
    mu is the measured OCV data times Farady constant
    x is the measured SOC data
    phase_boundarys_fixed_point is the list of starting and end point of miscibility gap(s)
    GibbsFunction is the Gibbs free energy landscape
    params_list contains the RK params and G0, in the sequence of [Omega0, Omega1, ..., G0]
    alpha_miscibility: weight of miscibility loss
    T: temperature
    """
    # see if x is in any gaps
    def _is_inside_gaps(_x, _gaps_list):
        _is_inside = False
        _index = -99999
        if len(_gaps_list) == 0:
            return False, -99999
        for i in range(0, len(_gaps_list)):
            if _x >= _gaps_list[i][0] and _x <= _gaps_list[i][1]:
                _is_inside = True
                _index = i
                break
        return _is_inside, _index
    # calculate loss
    loss_ = 0.0
    n_count = 0
    for i in range(0, len(x)):
        x_now = x[i]
        mu_now = mu[i]
        is_inside, index = _is_inside_gaps(x_now, phase_boundarys_fixed_point)
        if is_inside == False:
            # outside miscibility gap 
            x_now = x_now.requires_grad_()
            g_now = GibbsFunction(x_now, params_list, T)
            mu_pred_now = autograd.grad(outputs=g_now, inputs=x_now, create_graph=True)[0]
            loss_ = loss_ + ((mu_pred_now-mu_now)/(8.314*T))**2 
            # print(x_now, mu_now, mu_pred_now)
            n_count = n_count + 1
        else: 
            # inside miscibility gap
            x_alpha = phase_boundarys_fixed_point[index][0]
            x_beta = phase_boundarys_fixed_point[index][1]
            ct_pred = (GibbsFunction(x_alpha, params_list, T) - GibbsFunction(x_beta, params_list, T))/(x_alpha - x_beta) 
            if torch.isnan(ct_pred):
                print("Common tangent is NaN")
                x_alpha = 99999.9
                x_beta = -99999.9
            if x_alpha > x_beta:
                print("Error in phase equilibrium boundary, x_left %.4f larger than x_right %.4f. If Hessian loss is not 0, it's fine. Otherwise check code carefully!" %(x_alpha, x_beta))
                x_alpha = 99999.9
                x_beta = -99999.9
            if torch.isnan(ct_pred):
                print("Warning: skipped for loss calculation at a filling fraction x")
            else:
                loss_ = loss_ + alpha_miscibility*((ct_pred - mu_now)/(8.314*T))**2
                # print(x_now, mu_now, ct_pred)
                n_count = n_count + 1
    return loss_/n_count


# entropy loss calculation
def calc_loss_entropy(x_measured, dsdx_measured, S_config_params_list, n_list, Theta_E_list, T_dsdx=320):
    # calculate collocation loss
    dsdx_calculated = torch.zeros(len(dsdx_measured))
    for i in range(0, len(x_measured)):
        x = x_measured[i]
        x = x.requires_grad_()
        s_config_tot, _, _ = calculate_S_config_total(x, S_config_params_list)
        s_vib_tot = calculate_S_vib_total(x, n_list, Theta_E_list, T=T_dsdx, Theta_Li_scaled_100_times=True)
        s_tot = s_config_tot + s_vib_tot
        ds_dx = autograd.grad(outputs=s_tot, inputs=x, create_graph=True)[0]
        dsdx_calculated[i] = ds_dx
    # impose constraints
    x_calculated = np.linspace(0.0001,0.9999,100).astype("float32")
    x_calculated = torch.from_numpy(x_calculated)
    s_calculated = torch.zeros(len(x_calculated))
    s_upper_bound = torch.zeros(len(x_calculated))
    for i in range(0, len(x_calculated)):
        x = x_calculated[i]
        x = x.requires_grad_()
        s_tot, _, _ = calculate_S_config_total(x, S_config_params_list)
        s_calculated[i] = s_tot
        s_upper_bound_now, _, _ = calculate_S_config_total(x, [0.0])
        s_upper_bound[i] = s_upper_bound_now
    ## calculate loss
    # s_config should be larger than 0    
    mask_s_lower_bound = (s_calculated <= 0).int()
    loss_s_config_leq_0 = torch.sum((s_calculated*mask_s_lower_bound)**2)*1000000
    # s_config should be smaller than ideal configurational entropy
    mask_s_upper_bound = (s_calculated >= s_upper_bound).int()
    loss_s_config_geq_upper_bound = torch.sum(((s_calculated-s_upper_bound)*mask_s_upper_bound)**2)*1000000
    # minimize data loss
    loss_dsdx = torch.sum((dsdx_calculated-dsdx_measured)**2)
    # total loss
    loss = loss_s_config_leq_0 + loss_s_config_geq_upper_bound + loss_dsdx
    return loss, loss_dsdx, loss_s_config_leq_0, loss_s_config_geq_upper_bound

def train_multiple_OCVs_and_dS_dx(
          datafile1_name='graphite.csv', 
          T1 = 300,
          datafile2_name=None,
          T2 = None,
          datafile3_name=None,
          T3 = None,
          datafile_dsdx_name=None,
          T_dsdx = None,
          number_of_Omegas=6, 
          number_of_S_config_excess_omegas=5,
          order_of_Theta_LiHM_polynomial_expansion=6,
          learning_rate = 1000.0, 
          learning_rate_other_than_H_mix_excess = 0.01,
          epoch_optimize_params_other_than_H_mix_only_after_this_epoch = 1000,
          alpha_dsdx = 1.0/1000, # if dsdx has been pretrained, then set it as a small number
          total_training_epochs = 8000,
          loss_threshold = 0.01,
          G0_rand_range=[-10*5000,-5*5000], 
          Omegas_rand_range=[-10*100,10*100],
          records_y_lims = [0.0,0.6],
          n_list = [9999.9, 6.0, 1.0], 
          pretrained_value_of_S_config_excess_omegas = None,
          pretrained_value_of_Theta_LiHM_polynomial_expasion = None,
          pretrained_value_of_ThetaHM = None,
          pretrained_value_of_ThetaLi = None,
          ):
    """
    Fit the diffthermo OCV function with up to 3 OCV meausured at different temperatures & 1 dS/dx data

    Inputs:
    datafile1_name, datafile2_name, datafile3_name: the csv file which contains OCV and SoC data, first column must be Li filling fraction (Be careful that Li filling fraction might be SoC or 1-SoC!), second column must be OCV. Must not with header & index. 
    T1, T2, T3: temperature of OCV contained in datafile1_name, datafile2_name, datafile3_name
    datafile_dsdx_name: the csv file which contains OCV and SoC data, first column must be Li filling fraction (Be careful that Li filling fraction might be SoC or 1-SoC!), second column must be dS/dx (unit must be J/(K.mol)). Must not with header & index.
    T_dsdx: temperature of ds/dx data in datafile_dsdx_name
    number_of_Omegas: number of R-K parameters. Note that the order of R-K expansion = number_of_Omegas - 1
    number_of_S_config_excess_omegas: number of excess configurational entropy omegas
    order_of_Theta_LiHM_polynomial_expansion: maximum order of polynomial used to expand Theta_LiHM (Einstein temperature)
    polynomial_style: style of polynomials to expand excess thermo, can be "Legendre", "R-K", "Chebyshev".
    learning_rate: learning rate for updating parameters
    learning_rate_other_than_H_mix_excess: learning rate for other parameters
    epoch_optimize_params_other_than_H_mix_only_after_this_epoch: before this number of epochs, only H excess mix params are optimized; After this epoch, all params are optimized
    alpha_dsdx: weight of dsdx loss
    total_training_epochs: total epochs for fitting
    loss_threshold: threshold of loss, below which training stops automatically
    G0_rand_range: the range for randomly initialize G0
    Omegas_rand_range: the range for randomly initialize R-K parameters
    ns_list_values: the amount of atoms in 1 mole of substance, in the sequence of LixHM, HM and Li (always 1). The first item can be arbitrary, it will be fixed later in the code
    records_y_lims: the range for records y axis lims
    pretrained_value_of_S_config_excess_omegas: pretrained values, a list of float
    pretrained_value_of_ThetaHM and pretrained_value_of_ThetaLi: pretrained value, a float

    Outputs: 
    params_list: contains all fitted params, which can be put into write_ocv_functions function to get your PyBaMM OCV function
    """
    
    from energy import GibbsFE_Legendre as GibbsFE
    
    working_dir = os.getcwd()
    os.chdir(working_dir)
    try:
        os.mkdir("records")
    except:
        pass
    with open("log",'w') as fin:
        fin.write("")

    # read data
    x1, mu1 = read_OCV_data(datafile1_name)
    if datafile2_name != None:
        x2, mu2 = read_OCV_data(datafile2_name)
    if datafile3_name != None:
        x3, mu3 = read_OCV_data(datafile3_name)
    if datafile_dsdx_name != None:
        x_measured, dsdx_measured = _convert_JMCA_Tdsdx_data_to_dsdx(
                                        datafile_name=datafile_dsdx_name, 
                                        T=T_dsdx)

    # declare all params
    ## excess enthalpy params
    params_list = []
    if number_of_Omegas <=10:
        for _ in range(0, number_of_Omegas):
            Omegai_start = np.random.randint(Omegas_rand_range[0], Omegas_rand_range[1])
            params_list.append(nn.Parameter( torch.from_numpy(np.array([Omegai_start],dtype="float32")) )) 
    else:
        for _ in range(0, 10):
            Omegai_start = np.random.randint(Omegas_rand_range[0], Omegas_rand_range[1])
            params_list.append(nn.Parameter( torch.from_numpy(np.array([Omegai_start],dtype="float32")) )) 
        for _ in range(10, number_of_Omegas):
            Omegai_start = 0.0
            params_list.append(nn.Parameter( torch.from_numpy(np.array([Omegai_start],dtype="float32")) )) 
    G0_start = np.random.randint(G0_rand_range[0], G0_rand_range[1]) # G0 is the pure substance gibbs free energy 
    G0 = nn.Parameter( torch.from_numpy(np.array([G0_start],dtype="float32")) ) 
    params_list.append(G0)
    ## omegas for excess configurational entropy
    S_config_params_list = []
    if pretrained_value_of_S_config_excess_omegas == None:
        for _ in range(0, number_of_S_config_excess_omegas):
            S_config_params_list.append(nn.Parameter( torch.from_numpy(np.array([np.random.randint(-100,100)*0.01],dtype="float32")) ) )
    else:
        for _ in range(0, number_of_S_config_excess_omegas):
            S_config_params_list.append(nn.Parameter( torch.from_numpy(np.array([pretrained_value_of_S_config_excess_omegas[_]],dtype="float32")) ) )
    ## Thetas (Einstein temperature)
    Theta_LiHM = []
    if pretrained_value_of_Theta_LiHM_polynomial_expasion == None:
        coeff_now = nn.Parameter( torch.from_numpy(np.array([200.0/100],dtype="float32")) )# Theta_Li is scaled by 100 time
        Theta_LiHM.append(coeff_now)
        for i in range(1, order_of_Theta_LiHM_polynomial_expansion+1):
            coeff_now = nn.Parameter( torch.from_numpy(np.array([(np.random.random()*2-1)/100],dtype="float32")) )# Theta_Li is scaled by 100 time
            Theta_LiHM.append(coeff_now)
    else:
        for i in range(0, order_of_Theta_LiHM_polynomial_expansion+1):
            coeff_now = nn.Parameter( torch.from_numpy(np.array([pretrained_value_of_Theta_LiHM_polynomial_expasion[i]/100],dtype="float32")) )# Theta_Li is scaled by 100 time
            Theta_LiHM.append(coeff_now)
    if pretrained_value_of_ThetaHM == None:
        Theta_HM = nn.Parameter( torch.from_numpy(np.array([300.0/100],dtype="float32")) )# Theta_Li is scaled by 100 time
    else:
        Theta_HM = nn.Parameter( torch.from_numpy(np.array([pretrained_value_of_ThetaHM/100],dtype="float32")) )# Theta_Li is scaled by 100 time
    if pretrained_value_of_ThetaLi == None:
        ## see how to convert Debye temperature into Einstein Temperature: https://en.wikipedia.org/wiki/Debye_model?utm_source=chatgpt.com#Debye_versus_Einstein
        ## Li Debye temperature: https://www.sciencedirect.com/science/article/pii/S0378775303002854
        Theta_Li = nn.Parameter( torch.from_numpy(np.array([380*0.805995977/100],dtype="float32")) )# Theta_Li is scaled by 100 time
    else:
        Theta_Li = nn.Parameter( torch.from_numpy(np.array([pretrained_value_of_ThetaLi/100],dtype="float32")) )# Theta_Li is scaled by 100 time
    Theta_E_list = [Theta_LiHM, Theta_HM, Theta_Li] 
    params_list_other_than_H_mix_excess = []
    for item in S_config_params_list:
        params_list_other_than_H_mix_excess.append(item)
    # this is Theta_LiHM which is expanded as a polynomial
    for j in range(0, len(Theta_E_list[0])):
        params_list_other_than_H_mix_excess.append(Theta_E_list[0][j])
    ## we don't train Theta_HM and Theta_Li, they should be always looked up

    # init optimizer
    optimizer_before_critical_epochs = optim.Adam([{'params': params_list, 'lr': learning_rate}] )
    optimizer_after_critical_epochs = optim.Adam([{'params': params_list, 'lr': learning_rate},
                                                  {'params': params_list_other_than_H_mix_excess, 'lr': learning_rate_other_than_H_mix_excess},] )

    # train
    params_record = []
    for i in range(0, len(params_list)):
        params_record.append([])
    epoch_record = []
    loss = 9999.9 # init total loss
    epoch = -1
    while loss > loss_threshold and epoch < total_training_epochs:
        # use current params to calculate predicted phase boundary
        epoch = epoch + 1
        # clean grad info
        if epoch <= epoch_optimize_params_other_than_H_mix_only_after_this_epoch:
            optimizer_before_critical_epochs.zero_grad()
        else:
            optimizer_after_critical_epochs.zero_grad()
        # init loss components
        loss = 0.0 # init total loss
        # for datafile 1
        phase_boundary_fixed_point = _get_phase_boundaries(GibbsFE, params_list, S_config_params_list, n_list, Theta_E_list, T1)    
        loss_collocation_1 = collocation_loss_all_pts(mu1, x1, phase_boundary_fixed_point, GibbsFE, [params_list, S_config_params_list, n_list, Theta_E_list], 1.0, T=T1)
        loss = loss + loss_collocation_1
        # datafile 2
        if datafile2_name != None:
            phase_boundary_fixed_point2 = _get_phase_boundaries(GibbsFE, params_list, S_config_params_list, n_list, Theta_E_list, T2)    
            loss_collocation_2 = collocation_loss_all_pts(mu2, x2, phase_boundary_fixed_point2, GibbsFE, [params_list, S_config_params_list, n_list, Theta_E_list], 1.0, T=T2)
            loss = loss + loss_collocation_2
        else:
            loss_collocation_2 = 0
        # datafile 3
        if datafile3_name != None:
            phase_boundary_fixed_point3 = _get_phase_boundaries(GibbsFE, params_list, S_config_params_list, n_list, Theta_E_list, T3)    
            loss_collocation_3 = collocation_loss_all_pts(mu3, x3, phase_boundary_fixed_point3, GibbsFE, [params_list, S_config_params_list, n_list, Theta_E_list], 1.0, T=T3)
            loss = loss + loss_collocation_3
        else:
            loss_collocation_3 = 0
        # dsdx file
        if datafile_dsdx_name != None:
            loss_entropy, _loss_dsdx, _loss_s_config_leq_0, _loss_s_config_geq_upper_bound = calc_loss_entropy(x_measured, dsdx_measured, S_config_params_list, n_list, Theta_E_list, T_dsdx=T_dsdx)
            loss = loss + alpha_dsdx * loss_entropy 
        else:
            loss_entropy = 0
        # backprop & update
        loss.backward()
        if epoch <= epoch_optimize_params_other_than_H_mix_only_after_this_epoch:
            optimizer_before_critical_epochs.step()
        else:
            optimizer_after_critical_epochs.step()
        # record
        for i in range(0, len(params_list)):
            params_record[i].append(params_list[i].item()/1000.0)
        epoch_record.append(epoch)
        # print output
        output_txt = "Epoch %3d  Loss %.4f  OCV1 %.4f  OCV2 %.4f  OCV3 %.4f  dSdx_col %.4f s<0 %.4f s>smax%.4f     " %(epoch, loss, loss_collocation_1, loss_collocation_2, loss_collocation_3, _loss_dsdx, _loss_s_config_leq_0, _loss_s_config_geq_upper_bound )
        # H excess params
        for i in range(0, len(params_list)-1):
            output_txt = output_txt + "Omega%d %.4f "%(i, params_list[i].item())
        output_txt  = output_txt + "G0 %.4f "%(params_list[-1].item())
        # S_config_excess params
        for i in range(0, len(S_config_params_list)):
            output_txt = output_txt + "omega%d %.4f "%(i, S_config_params_list[i].item())
        # Theta_E params
        for i in range(0, len(Theta_E_list[0])):
            output_txt = output_txt + "ThetaE0_%d %.4f "%(i, Theta_E_list[0][i].item()*100)
        for i in range(1, len(Theta_E_list)):
            output_txt = output_txt + "ThetaE%d %.4f "%(i, Theta_E_list[i].item()*100)
        output_txt = output_txt + "      "
        print(output_txt)
        with open("log",'a') as fin:
            fin.write(output_txt)
            fin.write("\n")
        # check training for every 100 epochs
        if epoch % 100 == 0:
            # # draw the fitted results, but only draw datafile 1
            mu_pred = []
            for i in range(0, len(x1)):
                x_now = x1[i]
                mu_now = mu1[i]
                x_now = x_now.requires_grad_()
                g_now = GibbsFE(x_now, [params_list, S_config_params_list, n_list, Theta_E_list], T=T1)
                mu_pred_now = autograd.grad(outputs=g_now, inputs=x_now, create_graph=True)[0]
                mu_pred.append(mu_pred_now.detach().numpy())
            mu_pred = np.array(mu_pred)
            SOC = x1.clone().numpy()
            # plot figure
            plt.figure(figsize=(5,4))
            # plot the one before common tangent construction
            U_pred_before_ct = mu_pred/(-96485)
            plt.plot(SOC, U_pred_before_ct, 'k--', label="Prediction Before CT Construction")
            # plot the one after common tangent construction
            mu_pred_after_ct = []
            # see if x is inside any gaps
            def _is_inside_gaps(_x, _gaps_list):
                _is_inside = False
                _index = -99999
                for i in range(0, len(_gaps_list)):
                    if _x >= _gaps_list[i][0] and _x <= _gaps_list[i][1]:
                        _is_inside = True
                        _index = i
                        break
                return _is_inside, _index
            # pred
            for i in range(0, len(x1)):
                x_now = x1[i]
                mu_now = mu1[i]
                is_inside, index = _is_inside_gaps(x_now, phase_boundary_fixed_point)
                if is_inside == False:
                    # outside miscibility gap 
                    mu_pred_after_ct.append(mu_pred[i])
                else: 
                    # inside miscibility gap
                    x_alpha = phase_boundary_fixed_point[index][0]
                    x_beta = phase_boundary_fixed_point[index][1]
                    ct_pred = (GibbsFE(x_alpha, [params_list, S_config_params_list, n_list, Theta_E_list], T=T1) - GibbsFE(x_beta, [params_list, S_config_params_list, n_list, Theta_E_list], T=T1))/(x_alpha - x_beta) 
                    if torch.isnan(ct_pred) == False:
                        mu_pred_after_ct.append(ct_pred.clone().detach().numpy()[0]) 
                    else:
                        mu_pred_after_ct.append(mu_pred[i])
            mu_pred_after_ct = np.array(mu_pred_after_ct)
            U_pred_after_ct = mu_pred_after_ct/(-96485)
            plt.plot(SOC, U_pred_after_ct, 'r-', label="Prediction After CT Construction")
            U_true_value = mu1.numpy()/(-96485) # plot the true value
            plt.plot(SOC, U_true_value, 'b-', label="True OCV")
            plt.xlim([0,1])
            plt.ylim(records_y_lims)
            plt.legend()
            plt.xlabel("SOC")
            plt.ylabel("OCV")
            fig_name = "At_Epoch_%d_datafile1.png" %(epoch)
            os.chdir("records")
            plt.savefig(fig_name, bbox_inches='tight')
            plt.close()
            os.chdir("../")
            # # draw the H excess params VS epochs
            total_epochs = len(epoch_record)
            for i in range(0, len(params_list)-1):
                plt.figure(figsize=(5,4))
                param_name = "Omega%d" %(i)
                plt.plot(epoch_record, params_record[i], 'r-', label=param_name)
                plt.xlim([0,total_epochs])
                plt.xlabel("Epoch")
                plt.ylabel("Param")
                plt.legend()
                fig_name = param_name+".png"
                plt.savefig(fig_name, bbox_inches='tight')
                plt.close()
            # # draw G0 VS epochs
            plt.figure(figsize=(5,4))
            plt.plot(epoch_record, params_record[-1], 'r-', label="G0")
            plt.xlim([0,total_epochs])
            plt.xlabel("Epoch")
            plt.ylabel("Param")
            plt.legend()
            fig_name = "G0.png" 
            plt.savefig(fig_name, bbox_inches='tight')
            plt.close()
    print("Training Complete.\n")
    return params_list, S_config_params_list, n_list, Theta_E_list



def write_ocv_functions(params_list, T = 300, outpyfile_name = "fitted_ocv_functions.py"):
    """
    T is temperature
    """
    from .energy import GibbsFE_Legendre as GibbsFE

    # sample the Gibbs free energy landscape
    print("Temperature is %.4f" %(T))
    sample = sampling(GibbsFE, params_list, T=T, sampling_id=1, ngrid=199)
    # give the initial guess of miscibility gap
    phase_boundarys_init, _ = convex_hull(sample, ngrid=199) 
    # refinement & calculate loss
    if phase_boundarys_init != []:
        # There is at least one phase boundary predicted 
        phase_boundary_fixed_point = []
        for phase_boundary_init in phase_boundarys_init:
            common_tangent = CommonTangent(GibbsFE, params_list, T = T) # init common tangent model
            phase_boundary_now = phase_boundary_init.requires_grad_()
            phase_boundary_fixed_point_now = common_tangent(phase_boundary_now) 
            phase_boundary_fixed_point.append(phase_boundary_fixed_point_now)
    else:
        # No boundary find.
        phase_boundary_fixed_point = []

    # print detected phase boundary
    cts = []
    if len(phase_boundary_fixed_point) > 0:
        print("Found %d phase coexistence region(s):" %(len(phase_boundary_fixed_point)))
        for i in range(0, len(phase_boundary_fixed_point)):
            x_alpha = phase_boundary_fixed_point[i][0]
            x_beta = phase_boundary_fixed_point[i][1]
            ct_now = (GibbsFE(x_alpha, params_list, T=T) - GibbsFE(x_beta, params_list, T=T))/(x_alpha - x_beta) 
            cts.append(ct_now)
            print("From x=%.16f to x=%.16f, mu_coex=%.16f" %(phase_boundary_fixed_point[i][0], phase_boundary_fixed_point[i][1], ct_now))
    else:
        print("No phase separation region detected.")

    # print output function in python
    with open(outpyfile_name, "w") as fout:
        fout.write("import numpy as np\nimport pybamm\nfrom pybamm import exp, log, tanh, constants, Parameter, ParameterValues\n#from numpy import log, exp\n#import matplotlib.pyplot as plt\n\n")
        fout.write("def fitted_OCP(sto):\n")
        fout.write("    _eps = 1e-7\n")
        fout.write("    # params\n")
        # write fitted params
        if isinstance(params_list[0], list) == True and len(params_list) == 4:
            # this means the first one is excess enthalpy free energy params, 
            # the second one is excess config entropy params
            # the thrid and fourth are param for S_vib
            energy_params_list = params_list[0]
            entropy_params_list = params_list[1]
            ns_list = params_list[2] 
            Theta_Es_list = params_list[3]
            # excess G params
            fout.write("    # excess enthalpy params\n")
            fout.write("    G0 = %.6f # G0 is the pure substance gibbs free energy \n" %(energy_params_list[-1].item()))
            for i in range(0, len(energy_params_list)-1):
                fout.write("    Omega%d = %.6f \n" %(i, energy_params_list[i].item()))
            text = "    Omegas =["
            for i in range(0, len(energy_params_list)-1):
                text=text+"Omega"+str(i)
                if i!= len(energy_params_list)-2:
                    text=text+", "
                else:
                    text=text+"]\n"
            fout.write(text)
            # excess configurational S params
            fout.write("    # configurational entropy params\n")
            for i in range(0, len(entropy_params_list)):
                fout.write("    omega%d = %.6f \n" %(i, entropy_params_list[i].item()))
            text = "    omegas =["
            for i in range(0, len(entropy_params_list)):
                text=text+"omega"+str(i)
                if i!= len(entropy_params_list)-1:
                    text=text+", "
                else:
                    text=text+"]\n"
            fout.write(text)
            # vibrational S param, ns
            fout.write("    # vibrational entropy params, ns (ns[0] is place holder)\n")
            for i in range(0, len(ns_list)):
                fout.write("    n%d = %.6f \n" %(i, ns_list[i]))
            text = "    ns =["
            for i in range(0, len(ns_list)):
                text=text+"n"+str(i)
                if i!= len(ns_list)-1:
                    text=text+", "
                else:
                    text=text+"]\n"
            fout.write(text)
            # vibrational S param, ThetaEs
            fout.write("    # vibrational entropy params, ThetaEs\n")
            # Theta_LiHM as a function of x
            for i in range(0, len(Theta_Es_list[0])):
                fout.write("    ThetaE0_%d = %.6f \n" %(i, Theta_Es_list[0][i].item()))
            text = "    ThetaE0 =["
            for i in range(0, len(Theta_Es_list[0])):
                text=text+"ThetaE0_"+str(i)
                if i!= len(Theta_Es_list[0])-1:
                    text=text+", "
                else:
                    text=text+"]\n"
            fout.write(text)
            # Theta_HM and Theta_Li
            for i in range(1, len(Theta_Es_list)):
                fout.write("    ThetaE%d = %.6f \n" %(i, Theta_Es_list[i].item()))
            text = "    ThetaEs =["
            for i in range(0, len(Theta_Es_list)):
                text=text+"ThetaE"+str(i)
                if i!= len(Theta_Es_list)-1:
                    text=text+", "
                else:
                    text=text+"]\n"
            fout.write(text)
        else:
            # no entropy params, only excess enthalpy parameters & temperature independent
            fout.write("    G0 = %.6f # G0 is the pure substance gibbs free energy \n" %(params_list[-1].item()))
            for i in range(0, len(params_list)-1):
                fout.write("    Omega%d = %.6f \n" %(i, params_list[i].item()))
            text = "    Omegas =["
            for i in range(0, len(params_list)-1):
                text=text+"Omega"+str(i)
                if i!= len(params_list)-2:
                    text=text+", "
                else:
                    text=text+"]\n"
            fout.write(text)
            
        # write phase boundaries & addition part
        if len(phase_boundary_fixed_point)>0:
            for i in range(0, len(phase_boundary_fixed_point)):
                fout.write("    # phase boundary %d\n" %(i))
                fout.write("    x_alpha_%d = %.16f\n" %(i, phase_boundary_fixed_point[i][0]))
                fout.write("    x_beta_%d = %.16f\n" %(i, phase_boundary_fixed_point[i][1]))
                fout.write("    mu_coex_%d = %.16f\n" %(i, cts[i]))
                fout.write("    is_outside_miscibility_gap_%d = (sto<x_alpha_%d) + (sto>x_beta_%d)\n" %(i,i,i))
            fout.write("    # whether is outside all gap\n")
            text = "    is_outside_miscibility_gaps = "
            for i in range(0, len(phase_boundary_fixed_point)):
                text = text + "is_outside_miscibility_gap_%d " %(i)
                if i!=len(phase_boundary_fixed_point)-1:
                    text = text + "* "
            fout.write(text)
            fout.write("    \n")
            fout.write("    mu_outside = G0 + 8.314*%.4f*log((sto+_eps)/(1-sto+_eps))\n" %(T))
            if isinstance(params_list[0], list) == True and len(params_list) == 4:
                # this means the first one is excess enthalpy free energy params, 
                # the second one is excess config entropy params
                # the thrid and fourth are param for S_vib
                ## write S_vib contribution
                # LiHM S_vib
                # first sum up the temperature at composition x
                fout.write("    ## S_vib\n")
                fout.write("    # Theta_LiHM\n")
                
                fout.write("    _t = 1 - 2 * sto  # Transform x to (1 - 2x) for legendre expansion\n")
                fout.write("    Pn_values = legendre_poly_recurrence(_t,len(ThetaEs[0])-1)\n")
                fout.write("    Pn_derivatives_values = legendre_derivative_poly_recurrence(_t, len(ThetaEs[0])-1)  # Compute Legendre polynomials up to degree len(coeffs) - 1\n")
                
                fout.write("    Theta_LiHM = 0.0\n")
                fout.write("    Theta_LiHM_derivative = 0.0\n")
                fout.write("    for i in range(0, len(ThetaEs[0])):\n")
                fout.write("        Theta_LiHM = Theta_LiHM + ThetaEs[0][i]*Pn_values[i]\n")
                fout.write("        Theta_LiHM_derivative = Theta_LiHM_derivative + (-2)*ThetaEs[0][i]*Pn_derivatives_values[i]\n")
                fout.write("    t_Theta_LiHM = -Theta_LiHM*100/%.4f\n" %(T))
                fout.write("    t_Theta_LiHM_derivative = Theta_LiHM_derivative*100/%.4f  # NOTE that there is no negative sign here\n" %(T))
                fout.write("    mu_outside = mu_outside - %.4f*(-3)*(ns[2])*8.314*( log(1.0 - exp(t_Theta_LiHM)) + t_Theta_LiHM*1.0/(exp(-t_Theta_LiHM)-1) ) \n" %(T))
                fout.write("    mu_outside = mu_outside - %.4f*(-3)*(1.0*ns[1]+sto*ns[2])*8.314*( 1/(1-exp(t_Theta_LiHM))* exp(t_Theta_LiHM)*t_Theta_LiHM_derivative  - t_Theta_LiHM_derivative*1/(exp(-t_Theta_LiHM) -1)   -t_Theta_LiHM * exp(-t_Theta_LiHM)/(exp(-t_Theta_LiHM) -1)**2 *t_Theta_LiHM_derivative  ) \n" %(T))                    
                ## mu does not depend on HM,
                # fout.write("    # Theta_HM \n")
                # fout.write("    t = -ThetaEs[1]*100/%.4f\n" %(T))
                # fout.write("    mu_outside = mu_outside - As[1] * %.4f*(-3)*8.314*( log(1.0 - exp(t)) + t*1.0/(exp(-t)-1) ) \n" %(T))
                ## mu does depend on Li, because it's (-x*s_Li)
                fout.write("    # Theta_Li\n")
                fout.write("    t = -ThetaEs[2]*100/%.4f\n" %(T))
                fout.write("    mu_outside = mu_outside +  %.4f*(-3)*ns[2]*8.314*( log(1.0 - exp(t)) + t*1.0/(exp(-t)-1) ) \n" %(T))        
                ## write H_vib contribution
                fout.write("    ## H_vib\n")
                fout.write("    # Theta_LiHM\n")
                fout.write("    Theta_LiHM_real = Theta_LiHM*100\n")
                fout.write("    Theta_LiHM_derivative_real = Theta_LiHM_derivative*100\n")
                fout.write("    mu_outside = mu_outside +  3*(ns[2])*8.314*( 1/2*Theta_LiHM_real + Theta_LiHM_real/(exp(Theta_LiHM_real/%.4f)-1) ) \n" %(T))           
                fout.write("    mu_outside = mu_outside +  3*(1.0*ns[1]+sto*ns[2])*8.314*( 0.5*Theta_LiHM_derivative_real + Theta_LiHM_derivative_real/(exp(Theta_LiHM_real/%.4f) -1) - Theta_LiHM_real/(exp(Theta_LiHM_real/%.4f) -1)**2 *  exp(Theta_LiHM_real/%.4f) * Theta_LiHM_derivative_real/%.4f ) \n" %(T,T,T,T))           
                ## mu does not depend on HM
                # fout.write("    # Theta_HM \n")
                # fout.write("    Theta_Li_real = ThetaEs[1]*100\n")
                # fout.write("    mu_outside = mu_outside + As[1] *3*8.314*( 1/2*Theta_Li_real + Theta_Li_real/(exp(Theta_Li_real/%.4f)-1) ) \n" %(T))           
                ## mu does depend on Li, because it's (x*s_Li)
                fout.write("    # Theta_Li\n")
                fout.write("    Theta_Li_real = ThetaEs[2]*100\n")
                fout.write("    mu_outside = mu_outside - 3*ns[2]*8.314*( 1/2*Theta_Li_real + Theta_Li_real/(exp(Theta_Li_real/%.4f)-1) ) \n" %(T))                
                ## write S_config contribution
                fout.write("    t = 1 - 2 * sto  # Transform x to (1 - 2x) for legendre expansion\n")
                fout.write("    Pn_values = legendre_poly_recurrence(t,len(omegas)-1)\n")
                fout.write("    Pn_derivatives_values = legendre_derivative_poly_recurrence(t, len(omegas)-1)  # Compute Legendre polynomials up to degree len(coeffs) - 1\n")
                fout.write("    for i in range(0, len(omegas)):\n")
                fout.write("        mu_outside = mu_outside + 8.314*%.4f*log((sto+_eps)/(1-sto+_eps))*(omegas[i]*Pn_values[i]) -2.0*8.314*%.4f*(sto*log(sto) + (1-sto)*log(1-sto))*(omegas[i]*Pn_derivatives_values[i]) \n" %(T, T))
            # write excess G part              
            fout.write("    t = 1 - 2 * sto  # Transform x to (1 - 2x) for legendre expansion\n")
            fout.write("    Pn_values = legendre_poly_recurrence(t,len(Omegas)-1)\n")
            fout.write("    Pn_derivatives_values = legendre_derivative_poly_recurrence(t, len(Omegas)-1)  # Compute Legendre polynomials up to degree len(coeffs) - 1\n")
            fout.write("    for i in range(0, len(Omegas)):\n")
            fout.write("        mu_outside = mu_outside -2*sto*(1-sto)*(Omegas[i]*Pn_derivatives_values[i]) + (1-2*sto)*(Omegas[i]*Pn_values[i])\n")

            text0 = "    mu_e = is_outside_miscibility_gaps * mu_outside + (1-is_outside_miscibility_gaps) *   "
            text1 = ""
            for i in range(0, len(cts)):
                text1 = text1 + "(1-is_outside_miscibility_gap_%d)*mu_coex_%d " %(i, i)
                if i != len(cts)-1:
                    text1 = text1 + " + "
            text = text0 + "(" + text1 + ")\n"
            fout.write(text)
            fout.write("    return -mu_e/96485.0\n\n\n\n")
        else:
            # no phase boundaries required, just mu and return -mu/F
            fout.write("    mu = G0 + 8.314*%.4f*log((sto+_eps)/(1-sto+_eps))\n" %(T))
            if isinstance(params_list[0], list) == True and len(params_list) == 4:
                # this means the first one is excess enthalpy free energy params, 
                # the second one is excess config entropy params
                # the thrid and fourth are param for S_vib
                ## write S_vib contribution
                # LiHM S_vib
                # first sum up the temperature at composition x
                fout.write("    ## S_vib\n")
                fout.write("    # Theta_LiHM\n")
                fout.write("    _t = 1 - 2 * sto  # Transform x to (1 - 2x) for legendre expansion\n")
                fout.write("    Pn_values = legendre_poly_recurrence(_t,len(ThetaEs[0])-1)\n")
                fout.write("    Pn_derivatives_values = legendre_derivative_poly_recurrence(_t, len(ThetaEs[0])-1)  # Compute Legendre polynomials up to degree len(coeffs) - 1\n")
                fout.write("    Theta_LiHM = 0.0\n")
                fout.write("    Theta_LiHM_derivative = 0.0\n")
                fout.write("    for i in range(0, len(ThetaEs[0])):\n")
                fout.write("        Theta_LiHM = Theta_LiHM + ThetaEs[0][i]*Pn_values[i]\n")
                fout.write("        Theta_LiHM_derivative = Theta_LiHM_derivative + (-2)*ThetaEs[0][i]*Pn_derivatives_values[i]\n")
                fout.write("    t_Theta_LiHM = -Theta_LiHM*100/%.4f\n" %(T))
                fout.write("    t_Theta_LiHM_derivative = Theta_LiHM_derivative*100/%.4f  # NOTE that there is no negative sign here\n" %(T))
                fout.write("    mu = mu - %.4f*(-3)*(ns[2])*8.314*( log(1.0 - exp(t_Theta_LiHM)) + t_Theta_LiHM*1.0/(exp(-t_Theta_LiHM)-1) ) \n" %(T))
                fout.write("    mu = mu - %.4f*(-3)*(1.0*ns[1]+sto*ns[2])*8.314*( 1/(1-exp(t_Theta_LiHM))* exp(t_Theta_LiHM)*t_Theta_LiHM_derivative  - t_Theta_LiHM_derivative*1/(exp(-t_Theta_LiHM) -1)   -t_Theta_LiHM * exp(-t_Theta_LiHM)/(exp(-t_Theta_LiHM) -1)**2 *t_Theta_LiHM_derivative  ) \n" %(T))                    
                ## mu does not depend on HM,
                # fout.write("    # Theta_HM \n")
                # fout.write("    t = -ThetaEs[1]*100/%.4f\n" %(T))
                # fout.write("    mu = mu - As[1] * %.4f*(-3)*8.314*( log(1.0 - exp(t)) + t*1.0/(exp(-t)-1) ) \n" %(T))
                ## mu does depend on Li, because it's (x*s_Li)
                fout.write("    # Theta_Li\n")
                fout.write("    t = -ThetaEs[2]*100/%.4f\n" %(T))
                fout.write("    mu = mu -  %.4f*(-3)*ns[2]*8.314*2*sto*( log(1.0 - exp(t)) + t*1.0/(exp(-t)-1) ) \n" %(T))        
                ## write H_vib contribution
                fout.write("    ## H_vib\n")
                fout.write("    # Theta_LiHM\n")
                fout.write("    Theta_LiHM_real = Theta_LiHM*100\n")
                fout.write("    Theta_LiHM_derivative_real = Theta_LiHM_derivative*100\n")
                fout.write("    mu = mu +  3*(ns[2])*8.314*( 1/2*Theta_LiHM_real + Theta_LiHM_real/(exp(Theta_LiHM_real/%.4f)-1) ) \n" %(T))           
                fout.write("    mu = mu +  3*(1.0*ns[1]+sto*ns[2])*8.314*( 0.5*Theta_LiHM_derivative_real + Theta_LiHM_derivative_real/(exp(Theta_LiHM_real/%.4f) -1) - Theta_LiHM_real/(exp(Theta_LiHM_real/%.4f) -1)**2 *  exp(Theta_LiHM_real/%.4f) * Theta_LiHM_derivative_real/%.4f ) \n" %(T,T,T,T))           
                ## mu does not depend on HM
                # fout.write("    # Theta_HM \n")
                # fout.write("    Theta_Li_real = ThetaEs[1]*100\n")
                # fout.write("    mu = mu + As[1] *3*8.314*( 1/2*Theta_Li_real + Theta_Li_real/(exp(Theta_Li_real/%.4f)-1) ) \n" %(T))           
                ## mu does depend on Li, because it's (x*s_Li)
                fout.write("    # Theta_Li\n")
                fout.write("    Theta_Li_real = ThetaEs[2]*100\n")
                fout.write("    mu = mu + 3*ns[2]*8.314*2*sto*( 1/2*Theta_Li_real + Theta_Li_real/(exp(Theta_Li_real/%.4f)-1) ) \n" %(T))                
                ## write S_config contribution
                # this means the first one is excess gibbs free energy params, 
                # the second one is excess config entropy params
                # the thrid one is Theta_Li param for S_vib
                fout.write("    t = 1 - 2 * sto  # Transform x to (1 - 2x) for legendre expansion\n")
                fout.write("    Pn_values = legendre_poly_recurrence(t,len(omegas)-1)\n")
                fout.write("    Pn_derivatives_values = legendre_derivative_poly_recurrence(t, len(omegas)-1)  # Compute Legendre polynomials up to degree len(coeffs) - 1\n")
                fout.write("    for i in range(0, len(omegas)):\n")
                fout.write("        mu = mu + 8.314*%.4f*log((sto+_eps)/(1-sto+_eps))*(omegas[i]*Pn_values[i]) -2.0*8.314*%.4f*(sto*log(sto) + (1-sto)*log(1-sto))*(omegas[i]*Pn_derivatives_values[i]) \n" %(T, T))         
            # excess G             
            fout.write("    t = 1 - 2 * sto  # Transform x to (1 - 2x) for legendre expansion\n")
            fout.write("    Pn_values = legendre_poly_recurrence(t,len(Omegas)-1)\n")
            fout.write("    Pn_derivatives_values = legendre_derivative_poly_recurrence(t, len(Omegas)-1)  # Compute Legendre polynomials up to degree len(coeffs) - 1\n")
            fout.write("    for i in range(0, len(Omegas)):\n")
            fout.write("        mu = mu -2*sto*(1-sto)*(Omegas[i]*Pn_derivatives_values[i]) + (1-2*sto)*(Omegas[i]*Pn_values[i])\n")
            fout.write("    return -mu/96485.0\n\n\n\n")
            
            
    
    abs_path = os.path.abspath(__file__)[:-8]+"__legendre_derivatives.py"
    with open(abs_path,'r') as fin:
        lines = fin.readlines()
    with open(outpyfile_name, "a") as fout:
        for line in lines:
            fout.write(line)
            
    # write complete
    print("\n\n\n\n\n Fitting Complete.\n")
    print("Fitted OCV function written in PyBaMM function (copy and paste readay!):\n")
    print("###################################\n")
    with open(outpyfile_name, "r") as fin:
        lines = fin.readlines()
    for line in lines:
        print(line, end='')
    print("\n\n###################################\n")
    print("Or check %s and fitted_ocv_functions.m (if polynomial style = R-K) for fitted thermodynamically consistent OCV model in PyBaMM & Matlab formats. " %(outpyfile_name))





def read_OCV_data(datafile_name):
    # read data1
    df = pd.read_csv(datafile_name,header=None)
    data = df.to_numpy()
    x = data[:,0]
    mu = -data[:,1]*96485 # because -mu_e- = OCV*F, -OCV*F = mu
    # convert to torch.tensor
    x = x.astype("float32")
    x = torch.from_numpy(x)
    mu = mu.astype("float32")
    mu = torch.from_numpy(mu)
    return x, mu


def _convert_JMCA_Tdsdx_data_to_dsdx(datafile_name="TdS_dx_lithiation_320K_modified.csv", T=320):
    # read hysterisis data
    working_dir = os.getcwd()
    df = pd.read_csv(datafile_name,header=None) # deleted those datapoints within miscibility gaps
    data = df.to_numpy()
    x_measured = data[:,0] # Li filling fraction of graphite, from 0 to 1
    TdSdx = data[:,1] # unit is eV/6C, measured at 320K -- 6C means 6 carbons, i.e. per formular
    dsdx_measured = TdSdx/T*96485 # now eV*96485 = J/mol
    # convert to torch.tensor
    x_measured = x_measured.astype("float32")
    x_measured = torch.from_numpy(x_measured)
    dsdx_measured = dsdx_measured.astype("float32")
    dsdx_measured = torch.from_numpy(dsdx_measured)
    return x_measured, dsdx_measured


def _get_phase_boundaries(GibbsFE, params_list, S_config_params_list, n_list, Theta_E_list, T):
    # sample the Gibbs free energy landscape
    sample = sampling(GibbsFE, [params_list, S_config_params_list, n_list, Theta_E_list], T=T, sampling_id=1)
    # give the initial guess of miscibility gap
    phase_boundarys_init, _ = convex_hull(sample) 
    # refinement & calculate loss
    if phase_boundarys_init != []:
        # There is at least one phase boundary predicted 
        phase_boundary_fixed_point = []
        for phase_boundary_init in phase_boundarys_init:
            common_tangent = CommonTangent(GibbsFE, [params_list, S_config_params_list, n_list, Theta_E_list], T = T) # init common tangent model
            phase_boundary_now = phase_boundary_init.requires_grad_()
            phase_boundary_fixed_point_now = common_tangent(phase_boundary_now) 
            phase_boundary_fixed_point.append(phase_boundary_fixed_point_now)
    else:
        # No boundary find.
        phase_boundary_fixed_point = []
    return phase_boundary_fixed_point