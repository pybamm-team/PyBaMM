import os
import numpy as np
import torch
import torch.nn as nn
from torch import autograd
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

def calculate_S_config_total(x, S_config_params_list):
    """ 
    S_total = -R(x*log(x) + (1-x)*log(1-x))*(1+\sum_{i=0}^n \Omega_i P_i(1-2x))
    where P_i is i-th order Legendre/Chebyshev polynomial
    S_config_params_list = [omega0, omega1, ... omega_n], length n+1, up to nth order
    """
    S_ideal = -8.314*(x*torch.log(x) + (1-x)*torch.log(1-x) )
    S_expand = 1.0
    t = 1-2*x
    Pn_values = legendre_poly_recurrence(t, len(S_config_params_list)-1) 
    for i in range(0, len(S_config_params_list)):
        S_expand = S_expand + S_config_params_list[i] * Pn_values[i]
    S_total = S_ideal * S_expand
    return S_total, S_ideal, S_expand

def _S_vib_single_Einstein_model(x, Theta_Li, T=320, Theta_Li_scaled_100_times = True):
    """ 
    similar to derivation in https://pubs.acs.org/doi/10.1021/acs.jpcc.1c10414, Equation 20 & 21
    For an Einstein model we have 
    S_vib = -3nk_B[log(1-exp(-Theta_E/T)) - Theta_E/T * 1/(exp(Theta_E/T) -1)]
    x is the filling fraction
    Theta_Li is the learnable parameter (or a list that contains the learnable params), might be scaled by 100 times 
    T is the temperature
    style: if it is Theta_Li, what polynomial will be used
    """
    if isinstance(Theta_Li, list) == True:
        # this is a polynomial expansion
        _t = 1-2*x
        Pn_values = legendre_poly_recurrence(_t, len(Theta_Li)-1) 
        # calculate temperature
        t = 0.0
        # print(Theta_Li)
        for i in range(0, len(Theta_Li)):
            t = t + Theta_Li[i] *Pn_values[i]
    else:
        # it's a constant temperature
        t = Theta_Li * torch.tensor(1.0)
    if Theta_Li_scaled_100_times == True:
        t = -(t*100)/T # remember Theta_Li initialized 100 times smaller!!
    else:
        t = -t/T
    S_vib = -3*8.314*( torch.log(1.0 - torch.exp(t)) + t*1.0/(torch.exp(-t)-1) )
    return S_vib

def calculate_S_vib_total(x, n_list, Theta_E_list, T=320, Theta_Li_scaled_100_times=True):
    """
    Weighted average of several Einstein models, the number of Einstein models is len(n_list) = len(Theta_E_list)
    x is the filling fraction
    n_list is how many moles of atoms are there in 1 mole of substance, the first element is wrong (should be deduced from the other two elements)
    Theta_E_list is the list of learnable effective Einstein temperature for each of the Einstein model
    T is the temperature
    """
    assert len(n_list) == len(Theta_E_list)
    S_vib = 0.0
    ## we have s_excess = s_LixHM - s_HM - x*s_Li
    ## LixHM: there is 1 mole of HM, and x mole of Li
    S_vib = S_vib + (1.0*n_list[1] + x*n_list[2])* _S_vib_single_Einstein_model(x, Theta_E_list[0], T=T, Theta_Li_scaled_100_times=Theta_Li_scaled_100_times)
    ## HM: there is 1 mole of HM
    S_vib = S_vib - (1.0*n_list[1])* _S_vib_single_Einstein_model(x, Theta_E_list[1], T=T, Theta_Li_scaled_100_times=Theta_Li_scaled_100_times)
    ## Li: there is x mole of Li
    S_vib = S_vib - (x*n_list[2])* _S_vib_single_Einstein_model(x, Theta_E_list[2], T=T, Theta_Li_scaled_100_times=Theta_Li_scaled_100_times)
    return S_vib

def legendre_poly_recurrence(x, n):
    """
    Compute the Legendre polynomials up to degree n 
    using the Bonnet's recursion formula (i+1)P_(i+1)(x) = (2i+1)xP_i(x) - iP_(i-1)(x)
    and return all n functions in a list
    """
    P = [torch.ones_like(x), x]  # P_0(x) = 1, P_1(x) = x
    for i in range(1, n):
        P_i_plus_one = ((2 * i + 1) * x * P[i] - i * P[i - 1]) / (i + 1)
        P.append(P_i_plus_one)
    return P

def legendre_derivative_poly_recurrence(x, n):
    """
    Compute the Legendre polynomials derivatives up to degree n 
    using (x^2-1)/n P'n(x) = xP_n(x) - P_(n-1)(x),
    and return all n functions in a list
    """
    Pn_values = legendre_poly_recurrence(x,n)
    Pn_derivatives = [0.0]
    for i in range(1, n+1):
        Pn_derivative_next = (x*Pn_values[i] - Pn_values[i-1])/((x**2-1)/i)
        Pn_derivatives.append(Pn_derivative_next)
    return Pn_derivatives

# read hysterisis data
path = r'C:\UM\Study_Material\Capstone\OCP_FIT\Data'
loc = os.path.join(path,"TdS_dx_lithiation_320K_modified.csv") # deleted those datapoints within miscibility gaps
working_dir = os.getcwd()
df = pd.read_csv(loc,header=None) 
data = df.to_numpy()
x_measured = data[:,0] # Li filling fraction of graphite, from 0 to 1
TdSdx = data[:,1] # unit is eV/6C, measured at 320K -- 6C means 6 carbons, i.e. per formular
dsdx_measured = TdSdx/320*96485 # now eV*96485 = J/mol


# convert to torch.tensor
x_measured = x_measured.astype("float32")
x_measured = torch.from_numpy(x_measured)
dsdx_measured = dsdx_measured.astype("float32")
dsdx_measured = torch.from_numpy(dsdx_measured)
T = 320 # measured at 320

with open("log_entropy",'w') as fout:
    fout.write("")

os.makedirs("records_entropy", exist_ok=True)

## omegas for excess configurational entropy
## load a good guess
omega0 = nn.Parameter( torch.from_numpy(np.array([-0.5600],dtype="float32")) ) 
omega1 = nn.Parameter( torch.from_numpy(np.array([-0.1245],dtype="float32")) ) 
omega2 = nn.Parameter( torch.from_numpy(np.array([0.3012],dtype="float32")) ) 
omega3 = nn.Parameter( torch.from_numpy(np.array([-0.0237],dtype="float32")) )
omega4 = nn.Parameter( torch.from_numpy(np.array([-0.5114],dtype="float32")) )
S_config_params_list = [omega0, omega1, omega2, omega3, omega4]

## Theta_Es and As for Einstein models
## Theta_Li is scaled by 100 time
## initialize Theta_LiHM as a function of x
# Theta_LiHM = []
# Theta_LiHM_order_expansion = 8
# coeff_now = nn.Parameter( torch.from_numpy(np.array([200.0/100],dtype="float32")) )# Theta_Li is scaled by 100 time
# Theta_LiHM.append(coeff_now)
# for i in range(1, Theta_LiHM_order_expansion+1):
#     coeff_now = nn.Parameter( torch.from_numpy(np.array([(np.random.random()*2-1)/100],dtype="float32")) )# Theta_Li is scaled by 100 time
#     Theta_LiHM.append(coeff_now)
# ## load a good guess
ThetaLiHM0 = nn.Parameter( torch.from_numpy(np.array([252.5322/100],dtype="float32")) ) 
ThetaLiHM1 = nn.Parameter( torch.from_numpy(np.array([-26.5938/100],dtype="float32")) ) 
ThetaLiHM2 = nn.Parameter( torch.from_numpy(np.array([-1.2728/100],dtype="float32")) ) 
ThetaLiHM3 = nn.Parameter( torch.from_numpy(np.array([1.9412/100],dtype="float32")) ) 
ThetaLiHM4 = nn.Parameter( torch.from_numpy(np.array([-0.3446/100],dtype="float32")) ) 
ThetaLiHM5 = nn.Parameter( torch.from_numpy(np.array([-0.2208/100],dtype="float32")) ) 
ThetaLiHM6 = nn.Parameter( torch.from_numpy(np.array([0.7924/100],dtype="float32")) ) 
ThetaLiHM7 = nn.Parameter( torch.from_numpy(np.array([0.6615/100],dtype="float32")) ) 
ThetaLiHM8 = nn.Parameter( torch.from_numpy(np.array([0.4697/100],dtype="float32")) ) 
Theta_LiHM = [ThetaLiHM0, ThetaLiHM1, ThetaLiHM2, ThetaLiHM3, ThetaLiHM4, ThetaLiHM5, ThetaLiHM6, ThetaLiHM7, ThetaLiHM8]

## see how to convert Debye temperature into Einstein Temperature: https://en.wikipedia.org/wiki/Debye_model?utm_source=chatgpt.com#Debye_versus_Einstein
## Li Debye temperature: https://www.sciencedirect.com/science/article/pii/S0378775303002854
## graphite Debye temperature: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.73.064304
Theta_HM = nn.Parameter( torch.from_numpy(np.array([402*0.805995977/100],dtype="float32")) )# Theta is scaled by 100 time
Theta_Li = nn.Parameter( torch.from_numpy(np.array([380*0.805995977/100],dtype="float32")) )# Theta is scaled by 100 time  
Theta_E_list = [Theta_LiHM, Theta_HM, Theta_Li] 
## how many moles of atoms are there in one mole of LixHM, HM and Li
n_list = [9999.9, 6.0, 1.0]  # there are 6 moles of C in 1 mole of C6, and 1 mole of Li in 1 mole of lithium
# the first element of n_list should be recalculated by 1*n_list[1] + x*n_list[2]

## declare all parameters
params_list = [] # for optimizer
for item in S_config_params_list:
    params_list.append(item)
# this is Theta_LiHM which is expanded as a polynomial
for j in range(0, len(Theta_E_list[0])):
    params_list.append(Theta_E_list[0][j])
## we don't train Theta_HM and Theta_Li

# init optimizer
learning_rate = 0.1
optimizer = optim.Adam(params_list, lr=learning_rate)

# train
loss = 9999.9 # init total loss
epoch = -1
while loss > 0.01 and epoch < 20000:
    # clean grad info
    optimizer.zero_grad()
    # use current params to calculate predicted phase boundary
    epoch = epoch + 1
    # init loss components
    loss = 0.0 # init total loss
    dsdx_calculated = torch.zeros(len(dsdx_measured))
    for i in range(0, len(x_measured)):
        x = x_measured[i]
        x = x.requires_grad_()
        s_config_tot, _, _ = calculate_S_config_total(x, S_config_params_list)
        s_vib_tot = calculate_S_vib_total(x, n_list, Theta_E_list, T=T)
        s_tot = s_config_tot + s_vib_tot
        ds_dx = autograd.grad(outputs=s_tot, inputs=x, create_graph=True)[0]
        dsdx_calculated[i] = ds_dx
    
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
    loss.backward()
    optimizer.step()
    # print output
    output_txt = "Epoch %3d  Loss tot %.4f dsdx %.4f s>0 %.4f s<s_max %.4f    " %(epoch, loss, loss_dsdx, loss_s_config_leq_0, loss_s_config_geq_upper_bound)
    for i in range(0, len(S_config_params_list)): 
        output_txt = output_txt + "omega%d %.4f "%(i, S_config_params_list[i].item())
    for i in range(0, len(Theta_E_list[0])): 
        output_txt = output_txt + "ThetaLiHM%d %.4f "%(i, Theta_E_list[0][i].item()*100)   
    for i in range(1, len(Theta_E_list)): 
        output_txt = output_txt + "ThetaE%d %.4f "%(i, Theta_E_list[i].item()*100)    
    output_txt = output_txt + "      "
    print(output_txt)
    with open("log_entropy",'a') as fout:
        fout.write(output_txt)
        fout.write("\n")
    
    out_path = r'C:\UM\Study_Material\Capstone\OCP_FIT\entropy_epoch'
    if epoch % 100 == 0:
        _x = x_measured.numpy()
        _y = dsdx_measured.numpy()
        plt.plot(_x, _y, "b*")
        _y1 = dsdx_calculated.detach().numpy()
        plt.plot(_x, _y1, "k-")

        name = f"{epoch}.png"
        save_path = os.path.join(out_path, name)
        plt.savefig(save_path)  # Save to remote location
        plt.close()