#!/usr/bin/env python
# coding: utf-8


# External packages
import numpy as np
from configs import config
import torch
from sym_r import sr
from utils import read_logs

if __name__ == '__main__':

    # Seed
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)



    # Dataset
    # z = np.random.uniform(-10, 10, 50)
    # v = np.random.uniform(-10, 10, 50)
    # X = np.stack((z, v), axis=0)
    # y = 1.234*9.807*z + 1.234*v**2
    data = np.load("./data/select_data.npy")
    X= data[:10000,:5]
    X =  X.transpose()
    #y = data[:500,-1]
    y = ((X[3]*(X[0]-X[1])+1.204*1004*X[2]*X[4])/((X[3]+0.662*(1+(X[4]/(1/70))))))
    #y_pre = ((X[3]*(X[0]-X[1])+1.204*1004*X[2]*X[4])/((X[3]+0.662*(1+(X[4]*70)))))
    #print('the bias of data is :',y_pre-y)
    # Where $X=(z,v)$, $z$ being a length of dimension $L^{1}, T^{0}, M^{0}$, v a velocity of dimension $L^{1}, T^{-1}, M^{0}$, $y=E$ if an energy of dimension $L^{2}, T^{-2}, M^{1}$.
    # It be noted that free constants search starts around 1. by default. Therefore when using default hyperparameters, normalizing the data around an order of magnitude of 1 is strongly recommended.
    # Running SR task
    expression, logs = sr.SR(X,y ,
                                # Giving names of variables (for display purposes)
                                X_names = [ "Rn","G","VPD","delta","ga"],
                                # Giving units of input variables
                                X_units = [[0,1,-2,-1,0,0],[0,1,-2,-1,0,0],[0,0,0,0,1,0],[-1,0,0,0,1,0],[0,0,1,-1,0,0]  ],
                                # Giving name of root variable (for display purposes)
                                y_name  = "ET",
                                # Giving units of the root variable
                                y_units = [0,1,-2,-1,0,0],
                                # Fixed constants
                                fixed_consts       = ["1","1004","1.204","0.662"],
                                # Units of fixed constan
                                fixed_consts_units = [ [0,0,0,0,0,0],[-1,1,0,0,0,-1],[0,0,-3,0,0,1],[-1,0,0,0,1,0]],
                                # Free constants names (for display purposes)
                                free_consts_names = ["Gsurface"],
                                # Units offFree constants
                                free_consts_units = [[0,0,1,-1,0,0]],
                                # Run config
                                run_config = config.config0,

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
        for j in range (len(free_consts)):
            print("%s = %f"%(prog.library.free_const_names[j], free_consts[j]))
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







