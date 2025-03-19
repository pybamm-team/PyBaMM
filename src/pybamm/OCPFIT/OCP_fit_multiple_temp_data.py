import numpy as np
import torch
import torch.nn as nn
from torch import autograd
import torch.optim as optim
import pandas as pd
from .energy import sampling, convex_hull, CommonTangent, calculate_S_config_total, calculate_S_vib_total

# Global variable
global _eps
_eps = 1e-7

def train_multiple_OCVs_and_dS_dx(
        datafile1_name, T1, datafile2_name, T2, datafile3_name, T3,
        datafile_dsdx_name, T_dsdx, number_of_Omegas, 
        number_of_S_config_excess_omegas, order_of_Theta_LiHM_polynomial_expansion,
        polynomial_style, learning_rate, learning_rate_other_than_H_mix_excess,
        epoch_optimize_params_other_than_H_mix_only_after_this_epoch, alpha_dsdx,
        total_training_epochs, loss_threshold, G0_rand_range, Omegas_rand_range,
        records_y_lims, n_list, pretrained_value_of_S_config_excess_omegas,
        pretrained_value_of_Theta_LiHM_polynomial_expasion, pretrained_value_of_ThetaHM,
        pretrained_value_of_ThetaLi):
    
    from .energy import GibbsFE_Legendre as GibbsFE
    
    x1, mu1 = read_OCV_data(datafile1_name)
    x2, mu2 = read_OCV_data(datafile2_name) if datafile2_name else (None, None)
    x3, mu3 = read_OCV_data(datafile3_name) if datafile3_name else (None, None)
    x_measured, dsdx_measured = _convert_JMCA_Tdsdx_data_to_dsdx(datafile_dsdx_name, T_dsdx) if datafile_dsdx_name else (None, None)
    
    params_list = [nn.Parameter(torch.tensor(np.random.uniform(*Omegas_rand_range), dtype=torch.float32)) for _ in range(number_of_Omegas)]
    G0 = nn.Parameter(torch.tensor(np.random.uniform(*G0_rand_range), dtype=torch.float32))
    params_list.append(G0)
    
    S_config_params_list = [
        nn.Parameter(torch.tensor(pretrained_value_of_S_config_excess_omegas[i], dtype=torch.float32)) 
        for i in range(number_of_S_config_excess_omegas)
    ] if pretrained_value_of_S_config_excess_omegas else [
        nn.Parameter(torch.tensor(np.random.uniform(-1, 1), dtype=torch.float32)) 
        for _ in range(number_of_S_config_excess_omegas)
    ]
    
    Theta_LiHM = [
        nn.Parameter(torch.tensor(pretrained_value_of_Theta_LiHM_polynomial_expasion[i] / 100, dtype=torch.float32))
        for i in range(order_of_Theta_LiHM_polynomial_expansion + 1)
    ] if pretrained_value_of_Theta_LiHM_polynomial_expasion else [
        nn.Parameter(torch.tensor(np.random.uniform(-1, 1) / 100, dtype=torch.float32))
        for _ in range(order_of_Theta_LiHM_polynomial_expansion + 1)
    ]
    
    Theta_HM = nn.Parameter(torch.tensor(pretrained_value_of_ThetaHM / 100, dtype=torch.float32)) if pretrained_value_of_ThetaHM else nn.Parameter(torch.tensor(3.0, dtype=torch.float32))
    Theta_Li = nn.Parameter(torch.tensor(pretrained_value_of_ThetaLi / 100, dtype=torch.float32)) if pretrained_value_of_ThetaLi else nn.Parameter(torch.tensor(3.0, dtype=torch.float32))
    Theta_E_list = [Theta_LiHM, Theta_HM, Theta_Li]
    
    return params_list, S_config_params_list, n_list, Theta_E_list

def read_OCV_data(datafile_name):
    df = pd.read_csv(datafile_name, header=None)
    data = df.to_numpy()
    x = torch.tensor(data[:, 0], dtype=torch.float32)
    mu = torch.tensor(-data[:, 1] * 96485, dtype=torch.float32)
    return x, mu

def _convert_JMCA_Tdsdx_data_to_dsdx(datafile_name, T):
    df = pd.read_csv(datafile_name, header=None)
    data = df.to_numpy()
    x_measured = torch.tensor(data[:, 0], dtype=torch.float32)
    dsdx_measured = torch.tensor(data[:, 1] / T * 96485, dtype=torch.float32)
    return x_measured, dsdx_measured

def write_ocv_functions(params_list, polynomial_style, T, outpyfile_name):
    from .energy import GibbsFE_Legendre as GibbsFE
    sample = sampling(GibbsFE, params_list, T=T, sampling_id=1, ngrid=199)
    phase_boundarys_init, _ = convex_hull(sample, ngrid=199) 
    phase_boundary_fixed_point = []
    
    if phase_boundarys_init:
        for phase_boundary_init in phase_boundarys_init:
            common_tangent = CommonTangent(GibbsFE, params_list, T=T)
            phase_boundary_now = phase_boundary_init.requires_grad_()
            phase_boundary_fixed_point.append(common_tangent(phase_boundary_now))
    
    with open(outpyfile_name, "w") as fout:
        fout.write("import numpy as np\n")
        fout.write("def fitted_OCP(sto):\n")
        fout.write("    _eps = 1e-7\n")
        fout.write("    mu = ")
        fout.write("    return -mu/96485.0\n")