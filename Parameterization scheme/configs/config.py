import torch
import numpy as np
from reward import reward


"""
SR performances are highly dependent on hyperparameters.
For doing science, it is recommended to tune hyperparameters to the specific problem to get best performances.
"""

"""
Configuration config0 is a faster but less effective configuration adapted from config1 which was tuned on a few 
astrophysical cases for the paper: arXiv:2303.03192.
Better configurations will be added in the future.
"""
DEVICE = 'cuda' # cuda , cpu
positional_token_default_values = {
    'MAX_NAME_SIZE' : 10 ,
    'UNITS_VECTOR_SIZE' : 7,
    'DEFAULT_BEHAVIOR_ID' : 9999999,
    'INVALID_VAR_ID' : 9999999 ,
    'DEFAULT_COMPLEXITY' :1.,
    'DEFAULT_FREE_CONST_INIT_VAL' :1.,
    'MAX_NB_CHILDREN' : 2 ,
    'INVALID_TOKEN_NAME' : '-',
    'INVALID_POS' :9999999,
    'INVALID_DEPTH' :9999999,
    'DUMMY_TOKEN_NAME':'dummy'
}
data_gen = False  # False , True
# __________________________________________
# # PenmanMonteithPrior
# # Energy absorption secition
# X_ES_names=[ "Rn","G","delta"]
# # Atmosphere absorption secition
# X_AS_names=["VPD","ga","1.204", "1004"]
# #Resistance section
# X_RS_names=["delta","ga","0.662","Gsurface","1"]
#["Rn", "G", "delta", "rho", "Cp", "VPD", "Ga", "Psy", "LAI", "SWdown", "swc_root", "WP", "FC", "Tair_K","F1","F2","F3","F4","Wind","RH","CO2air","albedo"]
# __________________________________________
#SurfaceResistancePrior
# X_F1_names=[ "SWdown","LAI","F1_C1","albedo","F1", "1."]#,"Rn", "G", "delta", "rho", "Cp", "VPD", "Ga", "Psy"
# X_F2_names=[ "swc_root","WP","FC","F2_C1","F2", "1."]
# X_F3_names=[ "VPD","RH","F3","F3_C1", "1."]
# X_F4_names=[ "Tair_K","Wind","F4","F4_C1", "1."]

X_F1_names=[ "SWdown","LAI","F1_C1","Wind","RH","CO2air","albedo", "1."]#,"Rn", "G", "delta", "rho", "Cp", "VPD", "Ga", "Psy"
X_F2_names=[ "swc_root","WP","FC","F2_C1","Wind","RH","CO2air","albedo", "1."]
X_F3_names=[ "VPD","Wind","RH","CO2air","albedo","F3_C1", "1."]
X_F4_names=[ "Tair_K","Wind","RH","CO2air","albedo","F4_C1", "1."]
default_op_names = ["mul", "add", "sub", "div", "exp", "n2", "log"]#, "inv", "n2", "sqrt", "neg", "exp", "log", "sin", "cos"]
default_stop_after_n_epochs = 5
# Maximum length of expressions
MAX_LENGTH = 300

# ---------- REWARD CONFIG ----------
reward_config = {
                 "reward_function"     : reward.SquashedNRMSE,
                 "zero_out_unphysical" : True,
                 "zero_out_duplicates" : False,
                 "keep_lowest_complexity_duplicate" : False,
                 # "parallel_mode" : True,
                 # "n_cpus"        : None,
                }

# ---------- LEARNING CONFIG ----------
# Number of trial expressions to try at each epoch
BATCH_SIZE = int(1e4)
# Function returning the torch optimizer given a model
GET_OPTIMIZER = lambda model : torch.optim.Adam(
                                    model.parameters(),
                                    lr=0.0025,
                                                )
# Learning config
learning_config = {
    # Batch related
    'batch_size'       : BATCH_SIZE,
    'max_time_step'    : MAX_LENGTH,
    'n_epochs'         : int(500),
    # Loss related
    'gamma_decay'      : 0.7,
    'entropy_weight'   : 0.005,
    # Reward related
    'risk_factor'      : 0.005,
    # 'rewards_computer' : reward.make_RewardsComputer (**reward_config),
    # Optimizer
    'get_optimizer'    : GET_OPTIMIZER,
    'observe_units'    : True,
}

# ---------- FREE CONSTANT OPTIMIZATION CONFIG ----------
free_const_opti_args = {
            'loss'   : "MSE",
            'method' : 'LBFGS',
            'method_args': {
                        'n_steps' : 20,
                        'tol'     : 1e-8,
                        'lbfgs_func_args' : {
                            'max_iter'       : 5,
                            'line_search_fn' : "strong_wolfe",
                                             },
                            },
        }

# ---------- PRIORS CONFIG ----------
priors_config  = [
                #("UniformArityPrior", None),
                # LENGTH RELATED
                #("HardLengthPrior"  , {"min_length": 5, "max_length": MAX_LENGTH, }),
                # ("PenmanMonteithPrior",{"targets":['rho','Cp','Gamma','Gsurface','Rn','G','VPD','delta','ga'],"max":[1,1,2,1,1,1,1,2,2],"max_depth":6}),
                ("SurfaceResistancePrior",{"max_depth":4,"scale": 5}),
                #("PenmanMonteithPrior", {"targets" : ['1.204' ,'1004' ,'0.662' ,'Gsurface' , 'Rn' ,'G' ,'VPD', 'delta' ,'ga' ], "max" : [1,1,1,1,1,1,1,2,2,] ,"max_depth":6, "scale": 1}),
                # RELATIONSHIPS RELATED
                # ("NoUselessInversePrior"  , None),
                ("PhysicalUnitsPrior", {"prob_eps": np.finfo(np.float32).eps}), # PHYSICALITY
                # ("NestedFunctions", {"functions":["exp",], "max_nesting" : 1}),
                # ("NestedFunctions", {"functions":["log",], "max_nesting" : 1}),
                # ("NestedTrigonometryPrior", {"max_nesting" : 1}),
                #("OccurrencesPrior", {"targets" : ['Rn', 'G', 'VPD', 'delta', 'ga','C1' ,'C2', 'C3' ], "max" : [1,1,1,2,2,1,1,1] }),
                 ]

# ---------- RNN CELL CONFIG ----------
cell_config = {
    "hidden_size" : 128,
    "n_layers"    : 2,
    "is_lobotomized" : False,
}

# ---------- RUN CONFIG ----------
config0 = {
    "learning_config"      : learning_config,
    "reward_config"        : reward_config,
    "free_const_opti_args" : free_const_opti_args,
    "priors_config"        : priors_config,
    "cell_config"          : cell_config,
}

