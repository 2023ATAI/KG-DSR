from configs import config
import numpy as np

# --------------------- TOKEN DEFAULT VALUES ---------------------
# Max size for phytokens names
MAX_NAME_SIZE = config.positional_token_default_values['MAX_NAME_SIZE']
# Number of units in SI system
UNITS_VECTOR_SIZE = config.positional_token_default_values['UNITS_VECTOR_SIZE']
# Default behavior ID in dimensional analysis
DEFAULT_BEHAVIOR_ID = config.positional_token_default_values['DEFAULT_BEHAVIOR_ID']
# Element used in place of a NAN (which is a float) as var id in int arrays
INVALID_VAR_ID = config.positional_token_default_values['INVALID_VAR_ID']  # NAN only exists for floats
# Default complexity
DEFAULT_COMPLEXITY = config.positional_token_default_values['DEFAULT_COMPLEXITY']
# Default initial value for free const phytokens
DEFAULT_FREE_CONST_INIT_VAL = config.positional_token_default_values['DEFAULT_FREE_CONST_INIT_VAL']



class Tokens:
    def __init__(self, shape, invalid_token_idx):
        """
        Parameters
        ----------
        shape : (int, int) (batchsize,max_step)
            Shape of the matrix.
        invalid_token_idx : int
            Index of the invalid phytokens in the library of phytokens.

        """

        # -------------------------------------------------------------------------------------------------------
        # -------------------------------------- non_positional properties --------------------------------------
        # -------------------------------------------------------------------------------------------------------

        # ---- Shape ----
        assert len(shape)==2, "Shape of Tokens object must be 2D." # remove line when jit-ing class ?
        self.shape = shape                          # (int, int)(1,15)
        self.invalid_token_idx = invalid_token_idx  # int

        # ---- Index in library ----
        # Default value
        self.default_idx = self.invalid_token_idx
        # Property
        self.idx = np.full(shape=self.shape, fill_value=self.default_idx, dtype=int )

        # ---- Token main properties ----
        # Default values
        self.default_arity        = 0
        self.default_complexity   = DEFAULT_COMPLEXITY
        self.default_var_type     = 0
        self.default_var_id       = INVALID_VAR_ID
        # Properties
        self.arity        = np.full(shape=self.shape, fill_value=self.default_arity        , dtype=int)
        self.complexity   = np.full(shape=self.shape, fill_value=self.default_complexity   , dtype=float)
        self.var_type     = np.full(shape=self.shape, fill_value=self.default_var_type     , dtype=int)
        self.var_id       = np.full(shape=self.shape, fill_value=self.default_var_id       , dtype=int)

        # ---- Physical units : behavior id ----
        # Default value
        self.default_behavior_id = DEFAULT_BEHAVIOR_ID
        # Property
        self.behavior_id = np.full(shape=self.shape, fill_value=self.default_behavior_id, dtype=int)

        # ---- Physical units : power ----
        # Default values
        self.default_is_power = False
        self.default_power    = np.NAN
        # Properties
        self.is_power = np.full(shape=self.shape, fill_value=self.default_is_power ,  dtype=bool)
        self.power    = np.full(shape=self.shape, fill_value=self.default_power    ,  dtype=float)

        # -------------------------------------------------------------------------------------------------------
        # -------------------------------------- semi_positional properties --------------------------------------
        # -------------------------------------------------------------------------------------------------------

        # ---- Physical units : units ----
        # Default values
        self.default_is_constraining_phy_units = False
        self.default_phy_units                 = np.NAN
        # Properties
        self.is_constraining_phy_units = np.full(shape=self.shape,                        fill_value=self.default_is_constraining_phy_units  ,  dtype=bool)
        self.phy_units                 = np.full(shape=self.shape + (UNITS_VECTOR_SIZE,), fill_value=self.default_phy_units                  ,  dtype=float)

