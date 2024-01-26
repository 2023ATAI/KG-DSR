#!/usr/bin/env python
# coding: utf-8


# External packages
import numpy as np
from configs import config
import torch
from sym_r import sr
from utils import read_logs
from data_utils import data

def SquashedNRMSE (y_target, y_pred,):
    """
    Squashed NRMSE reward.
    Parameters
    ----------
    y_target : torch.tensor of shape (?,) of float
        Target output data_utils.
    y_pred   : torch.tensor of shape (?,) of float
        Predicted data_utils.
    Returns
    -------
    reward : torch.tensor float
        Reward encoding prediction vs target discrepancy in [0,1].
    """
    sigma_targ = y_target.std()
    RMSE = np.sqrt(np.mean((y_pred-y_target)**2))
    NRMSE = (1/sigma_targ)*RMSE
    reward = 1/(1 + NRMSE)
    return reward,RMSE,sigma_targ

def SquashedNRMSE_to_R2 (reward):
    """
    Converts SquashedNRMSE reward to R2 score.
    Parameters
    ----------
    reward : torch.tensor float
        Reward encoding prediction vs target discrepancy in [0,1].
    Returns
    -------
    R2 : torch.tensor float
        R2 score.
    """
    R2 = 2/reward - (1/reward)**2
    return R2
if __name__ == '__main__':

    # Seed
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    if config.data_gen:
        input_dir = 'dataset/'
        # obs_dir = 'D:/code/Symbolic Regression/FLUXNET_PLUMBER2_WEI/flux/'
        # obs_name = 'Qle_cor'
        x_train,y_train = data.data_gen(input_dir)
        np.save("./dataset/x_train.npy",x_train)
        np.save("./dataset/y_train.npy",y_train)
       # np.save("./dataset/x_test.npy",x_test)
       # np.save("./dataset/y_test.npy",y_test)
        # y_train_Gs_ = np.load("./dataset/y_train_Gs.npy")
    else:
        x_train = np.load("./dataset/x_train.npy")
        y_train = np.load("./dataset/y_train.npy")
        #x_test = np.load("./dataset/x_test.npy")
        #y_test = np.load("./dataset/y_test.npy")
        # y_train_Gs_ = np.load("./dataset/y_train_Gs.npy")
    ###### Gs=[self.met.delta⋅(self.met.Psy⋅(LE−(self.met.delta−LE))−self.met.rho⋅self.met.Cp⋅self.met.VPD⋅Ga)]/
    # [LE⋅self.met.Psy−self.met.rho⋅self.met.Cp⋅self.met.VPD]


    XX = x_train
    YY = y_train
    aa = np.isnan(XX).any(axis=0)
    # bb = np.isnan(y_train_Gs_)
    a = aa #| bb
    XX = XX[:,~a]
    YY = YY[~a]
    # y_train_Gs__ = y_train_Gs_[~a]
    b = np.isnan(YY)
    XX = XX[:, ~b]
    YY = YY[~b]
    # y_train = y_train_Gs__[~b]
# ______________________________________________________________________________________________________________________________________
    X_names = ["Rn", "G", "delta", "rho", "Cp", "VPD", "Ga", "Psy", "LAI", "SWdown", "swc_root", "WP", "FC", "Tair_K","F1","F2","F3","F4","Wind","RH","CO2air","albedo"]
    var_indices = {var: i for i, var in enumerate(X_names)}
    Rnet = XX[var_indices["Rn"]]
    Qg = XX[var_indices["G"]]
    delta = XX[var_indices["delta"]]
    rho = XX[var_indices["rho"]]
    Cp = XX[var_indices["Cp"]]
    VPD = XX[var_indices["VPD"]]
    Ga = XX[var_indices["Ga"]]
    Psy = XX[var_indices["Psy"]]
    LAI = XX[var_indices["LAI"]]
    SWdown = XX[var_indices["SWdown"]]
    swc_root = XX[var_indices["swc_root"]]
    WP = XX[var_indices["WP"]]
    Tair_K = XX[var_indices["Tair_K"]]
    FC = XX[var_indices["FC"]]
    F2 = XX[var_indices["F2"]]
    F3 = XX[var_indices["F3"]]
    F4 = XX[var_indices["F4"]]
    RH = XX[var_indices["RH"]]
    Wind = XX[var_indices["Wind"]]
    albedo = XX[var_indices["albedo"]]
    CO2air = XX[var_indices["CO2air"]]
    rsmin = 72.  # sm-1  see Alfieri et al. 2008 abstract
    gD = 0.1914  # Kpa  see Alfieri et al. 2008
    p1 = data.canopy_conductance_Jarvis1976(SWdown, FC, swc_root, WP, gD,VPD, rsmin, Tair_K, LAI)
    rc, F1, F2, F3, F4 = p1.canopy_conductance()
    Gs_  = 1/rc
    LE = YY

    LE_sim = (delta * (Rnet - Qg) + rho * Cp * VPD * Ga) / (delta + Psy * (1 + Ga / Gs_))
    R_LE_label, RMSE_LE_label, sigma_targ = SquashedNRMSE(LE, LE_sim)
    R2_LE_label = SquashedNRMSE_to_R2(R_LE_label)
    print('R_LE_label is', R_LE_label)
    print('RMSE_LE_label is', RMSE_LE_label)
    print('R2_label is', R2_LE_label)
    print('sigma_targ is', sigma_targ)

    Gs = Ga/(((((delta * (Rnet - Qg) + rho * Cp * VPD * Ga)/LE)-delta)/Psy)-1)
    X = XX
    Y = LE
    # aaa = np.mean(Gs)
    # bbb = np.mean(y_train)

    # R_label = SquashedNRMSE(Gs,y_train)

    # Running SR task
    expression, logs = sr.SR(X, Y,
                             # Giving names of variables (for display purposes)
                             # w/m2 , w/m2 , kPa/K ,  kg/m3 , J/(kg.K), Kpa, m/s ,kPa/K , - , w/m2 , - , - , K  ,
                             X_names=X_names,
                             # Giving units of input variables
                             # K,J,m,s,kpa,kg
                             X_units=[[0, 1, -2, -1, 0, 0],  # Rn
                                      [0, 1, -2, -1, 0, 0],  # Q
                                      [-1, 0, 0, 0, 1, 0],  # delta
                                      [0, 0, -3, 0, 0, 1],  # rho
                                      [-1, 1, 0, 0, 0, -1],  # Cp
                                      [0, 0, 0, 0, 1, 0],  # VPD
                                      [0, 0, 1, -1, 0, 0],  # Ga
                                      [-1, 0, 0, 0, 1, 0],  # Psy
                                      [0, 0, 0, 0, 0, 0],  # LAI
                                      [0, 1, -2, -1, 0, 0],  # SWdown
                                      [0, 0, 0, 0, 0, 0],  # swc_root
                                      [0, 0, 0, 0, 0, 0],  # WP
                                      [0, 0, 0, 0, 0, 0],  # FC
                                      [1, 0, 0, 0, 0, 0],  # Tair_K
                                      [0, 0, 0, 0, 0, 0],  # F1
                                      [0, 0, 0, 0, 0, 0],  # F2
                                      [0, 0, 0, 0, 0, 0],  # F3
                                      [0, 0, 0, 0, 0, 0],  # F4
                                      [0, 0, 1, -1, 0, 0],  # Wind
                                      [0, 0, 0, 0, 0, 0],  # RH
                                      [0, 0, 0, 0, 0, 0],  # CO2air
                                      [0, 0, 0, 0, 0, 0]],  # albedo
                             # Giving name of root variable (for display purposes)
                             y_name="Gs",
                             # Giving units of the root variable w/m2
                             y_units=[0, 0, 0, 0, 0, 1],
                             # y_units=[0, 0, 0, 0, 0, 1],
                             # Fixed constants： #rsmin = 72.（sm-1） ； gD = 0.1914 （Kpa）see Alfieri et al. 2008 abstract
                             fixed_consts=["72", "0.1914", "5000", "1."],
                             # Units of fixed constants
                             fixed_consts_units=[[0, 0, -1, 1, 0, 0], [0, 0, 0, 0, -1, 0], [0, -1, 1, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0]],

                             # Free constants names (for display purposes)
                             # free_consts_names=["F1_C1","F1_C2","F1_C3","F1_C4","F2_C1","F2_C2","F2_C3","F2_C4","F3_C1","F3_C2","F3_C3","F3_C4","F4_C1","F4_C2","F4_C3","F4_C4"],
                             free_consts_names=["F1_C1","F2_C1","F3_C1","F4_C1"],
                              # free_consts_names=["F1_C1","F2_C1",],
                             # Units offFree constants
                             # s/m
                             # free_consts_units=[[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0]],
                             free_consts_units=[[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 1]],
                             #free_consts_units=[[0, 1, -2, -1, 0, 0], [0, 0, 0, 0, 0, 0]],
                             # Run config
                             run_config=config.config0,

                             )

    # expression, logs = sr.SR(X, y,
    #                          # Giving names of variables (for display purposes)
    #                          X_names = [ "z"       , "v"        ],
    #                          # Giving units of input variables
    #                          X_units = [ [1, 0, 0] , [1, -1, 0] ],
    #                          # Giving name of root variable (for display purposes)
    #                          y_name  = "E",
    #                          # Giving units of the root variable
    #                          y_units = [2, -2, 1],
    #                          # Fixed constants
    #                          fixed_consts       = [ 1.      ],
    #                          # Units of fixed constants
    #                          fixed_consts_units = [ [0,0,0] ],
    #                          # Free constants names (for display purposes)
    #                          free_consts_names = [ "m"       , "g"        ],
    #                          # Units offFree constants
    #                          free_consts_units = [ [0, 0, 1] , [1, -2, 0] ],
    #                          # Run config
    #                          run_config = config.config0,
    #
    #                          )

    # Inspecting the best expression found
    # In ascii
    print("\nIn ascii:")
    print(expression.get_infix_pretty(do_simplify=True))
    # In latex
    print("\nIn latex")
    print(expression.get_infix_latex(do_simplify=True))
    # Free constants values
    print("\nFree constants values")
    print(expression.free_const_values.cpu().detach().numpy())

    # ### Inspecting pareto front expressions

    # In[ ]:

    # Inspecting pareto front expressions
    pareto_front_complexities, pareto_front_expressions, pareto_front_r, pareto_front_rmse = logs.get_pareto_front()
    for i, prog in enumerate(pareto_front_expressions):
        # Showing expression
        print(prog.get_infix_pretty(do_simplify=True))
        # Showing free constant
        free_consts = prog.free_const_values.detach().cpu().numpy()
        for j in range(len(free_consts)):
            print("%s = %f" % (prog.library.free_const_names[j], free_consts[j]))
        # Showing RMSE
        print("RMSE = {:e}".format(pareto_front_rmse[i]))
        print("-------------\n")

    # ### Loading pareto front expressions from log file

    # In[ ]:

    # Loading pareto front expressions from .csv log file as sympy expressions
    sympy_expressions = read_logs.read_pareto_csv("./SR_curves_pareto.csv")
    for expr in sympy_expressions:
        print(expr)

    # In[ ]:
