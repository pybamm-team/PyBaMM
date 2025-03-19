from utils import train_multiple_OCVs_and_dS_dx, write_ocv_functions
import os
path = r'C:\UM\Project\PyBaMM\src\pybamm\OCPFIT'
""" 
in Theta_LiHM_8_pretrain/log_entropy
Epoch 19999  Loss tot 156.9237 dsdx 156.2216 s>0 0.7021 s<s_max 0.0000    
omega0 -0.4028 omega1 0.1629 omega2 -0.0982 omega3 -0.2342 omega4 -0.5278 
ThetaLiHM0 1200.0280 ThetaLiHM1 276.7914 ThetaLiHM2 17.1939 
ThetaLiHM3 -5.3231 ThetaLiHM4 15.7622 ThetaLiHM5 11.0748 ThetaLiHM6 18.4938 ThetaLiHM7 23.4894 
ThetaLiHM8 13.8011 ThetaE1 324.0104 ThetaE2 306.2785       
"""
params_list, S_config_params_list, n_list, Theta_E_list = train_multiple_OCVs_and_dS_dx(
                                    datafile1_name=os.path.join(path,'47C_lithiation.csv'), 
                                    T1 = 320,
                                    datafile2_name=os.path.join(path,'25C_lithiation.csv'),
                                    T2 = 273+25,
                                    datafile3_name=os.path.join(path,'57C_lithiation.csv'),
                                    T3 = 273+57,
                                    datafile_dsdx_name=os.path.join(path,"TdS_dx_lithiation_320K_modified.csv"),
                                    T_dsdx = 320,
                                    number_of_Omegas=9, 
                                    number_of_S_config_excess_omegas=5,
                                    order_of_Theta_LiHM_polynomial_expansion=8,
                                    learning_rate = 1000.0,  
                                    learning_rate_other_than_H_mix_excess = 0.01, # we have pretrained good values, don't need too large
                                    epoch_optimize_params_other_than_H_mix_only_after_this_epoch = 1000, # use pretrained params for T dependency for a while
                                    alpha_dsdx = 1.0/10000, # weight of dsdx loss, its usually on the order of 100, so 100/10000 ~ 0.01, ~ same level of OCV loss
                                    total_training_epochs = 2000,
                                    loss_threshold = 0.0001,
                                    G0_rand_range=[-10*5000,-5*5000], 
                                    Omegas_rand_range=[-10*100,10*100],
                                    records_y_lims = [0.0,0.3],
                                    n_list = [9999.9, 6.0, 1.0],  # first one is random, okay
                                    pretrained_value_of_S_config_excess_omegas = [-0.5172, 0.3202, -0.1400, -0.1026, 0.0786],   
                                    pretrained_value_of_Theta_LiHM_polynomial_expasion = [1203.4978, 287.2618, 13.5151, 10.9238, 67.2186, 17.4357, -3.0777, 14.6925, 10.7577], # NO NEED TO SCALE BY 100 times!
                                    pretrained_value_of_ThetaHM = 324.0104, # NO NEED TO SCALE BY 100 times!
                                    pretrained_value_of_ThetaLi = 306.2785, # NO NEED TO SCALE BY 100 times!
                                    )

write_ocv_functions([params_list, S_config_params_list, n_list, Theta_E_list],  T=273+10, outpyfile_name="10C.py")
write_ocv_functions([params_list, S_config_params_list, n_list, Theta_E_list],  T=273+25, outpyfile_name="25C.py")
write_ocv_functions([params_list, S_config_params_list, n_list, Theta_E_list],  T=273+47, outpyfile_name="47C.py")
write_ocv_functions([params_list, S_config_params_list, n_list, Theta_E_list],  T=273+57, outpyfile_name="57C.py")