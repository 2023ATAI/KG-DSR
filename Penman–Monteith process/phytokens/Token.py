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

class Token:
    """
        An object representing a unique mathematical symbol (non_positional & semi_positional), except idx (which
        represents the phytokens's idx in the library and is not encoded here).
        Attributes :
        ----------
        See phytokens.Token.__init__ for full description of parameters.

        name                      :  str (<MAX_NAME_SIZE)
        sympy_repr                :  str (<MAX_NAME_SIZE)
        arity                     :  int
        complexity                :  float
        var_type                  :  int
        function                  :  callable or None
        init_val                  :  float
        var_id                    :  int
        fixed_const               :  float-like
        behavior_id               :  int
        is_power                  :  bool
        power                     :  float

        is_constraining_phy_units :  bool
        phy_units                 :  numpy.array of shape (UNITS_VECTOR_SIZE,) of float

        Methods
        -------
        __call__(args)
            Calls the phytokens's function.
    """
    def __init__(self,
                 # ---- Token representation ----
                 name,
                 sympy_repr,
                 # ---- Token main properties ----
                 arity,
                 complexity  = DEFAULT_COMPLEXITY,
                 var_type    = 0,
                 # Function specific
                 function = None,
                 # Free constant specific
                 init_val = np.NAN,
                 # Input variable / free constant specific
                 var_id   = None,
                 # Fixed constant specific
                 fixed_const = np.NAN,
                 # ---- Physical units : behavior id ----
                 behavior_id               = None,
                 # ---- Physical units : power ----
                 is_power                  = False,
                 power                     = np.NAN,

                 # ---- Physical units : units (semi_positional) ----
                 is_constraining_phy_units = False,
                 phy_units                 = None,
                 ):
        """
        Note: __init___ accepts None for some parameters for ease of use which are then converted to the right value and
        type as attributes.
        Parameters
        ----------
        name : str
            A short name for the phytokens (eg. 'add' for addition).
        sympy_repr : str
            Sympy representation of mathematical operation.

        arity : int
            Number of argument of phytokens (eg. 2 for addition, 1 for sinus, 0 for input variables or constants).
            - This phytokens represents a function or a fixed const  (ie. var_type = 0 )      <=> arity >= 0
            - This phytokens represents input_var or free const      (ie. var_type = 1 or 2 ) <=> arity = 0
        complexity : float
            Complexity of phytokens.
        var_type : int
            - If this phytokens represents a function    : var_type = 0 (eg. add, mul, cos, exp).
            - If this phytokens represents an input_var  : var_type = 1 (input variable, eg. x0, x1).
            - If this phytokens represents a free const  : var_type = 2 (free constant,  eg. c0, c1).
            - If this phytokens represents a fixed const : var_type = 3 (eg. pi, 1)
        function : callable or None
            - This phytokens represents a function (ie. var_type = 0 ) <=> this represents the function associated with the
            phytokens. Function of arity = n must be callable using n arguments, each argument consisting in a numpy array
            of floats of shape (int,) or a single float number.
            - This phytokens represents an input_var, a free const or a fixed const (ie. var_type = 1, 2 or 3) <=>
            function = None
        init_val : float or np.NAN
            - This phytokens represents a function, a fixed const or an input variable (ie. var_type = 0, 1 or 3)
            <=> init_val = np.NAN
            - This phytokens represents a free const (ie. var_type = 2 )  <=>  init_val = non NaN float
        var_id : int or None
            - This phytokens represents an input_var or a free constant (ie. var_type = 1 or 2) <=> var_id is an integer
            representing the id of the input_var in the dataset or the id of the free const in the free const array.
            - This phytokens represents a function or a fixed constant (ie. var_type = 0 or 3) <=> var_id = None.
            (converted to INVALID_VAR_ID in __init__)
        fixed_const : float or np.NAN
            - This phytokens represents a fixed constant (ie. var_type = 3) <=> fixed_const = non NaN float
            - This phytokens represents a function, an input_var or a free const (ie. var_type = 0, 1 or 2 )
            <=>  fixed_const = non NaN float

        behavior_id : int
            Id encoding behavior of phytokens in the context of dimensional analysis (see functions for details).

        is_power : bool
            True if phytokens is a power phytokens (n2, sqrt, n3 etc.), False else.
        power : float or np.NAN
            - is_power = True <=> power is a float representing the power of a phytokens (0.5 for sqrt, 2 for n2 etc.)
            - is_power = False <=> power is np.NAN

        is_constraining_phy_units : bool
            - True if there are hard constraints regarding with this phytokens's physical units (eg. dimensionless op such
            as cos, sin exp, log etc. or input variable / constant representing physical quantity such as speed, etc.)
            - False if this phytokens's units are free ie there are no constraints associated with this phytokens's physical
            units (eg. add, mul phytokens).
        phy_units : numpy.array of size UNITS_VECTOR_SIZE of float or None
            - is_constraining_phy_units = False <=> phy_units = None (converted to vector of np.NAN in __init__)
            - is_constraining_phy_units = True  <=> phy_units = vector containing power of units.
            Ie. vector of zeros for dimensionless operations (eg. cos, sin, exp, log), vector containing power of units
            for constants or input variable (eg. [1, -1, 0, 0, 0, 0, 0] for a phytokens representing a velocity with the
            convention [m, s, kg, ...]).
        """

        # ---------------------------- Token representation ----------------------------
        # ---- Assertions ----
        assert isinstance(name,       str), "name       must be a string, %s is not a string" % (str(name))
        assert isinstance(sympy_repr, str), "sympy_repr must be a string, %s is not a string" % (str(sympy_repr))
        assert len(name)       < MAX_NAME_SIZE, "Token name       must be < than %i, MAX_NAME_SIZE can be changed." % (MAX_NAME_SIZE)
        assert len(sympy_repr) < MAX_NAME_SIZE, "Token sympy_repr must be < than %i, MAX_NAME_SIZE can be changed." % (MAX_NAME_SIZE)
        # ---- Attribution ----
        self.name       = name                                     # str (<MAX_NAME_SIZE)
        self.sympy_repr = sympy_repr                               # str (<MAX_NAME_SIZE)

        # ---------------------------- Token main properties ----------------------------
        # ---- Assertions ----
        assert isinstance(arity,      int),   "arity must be an int, %s is not an int" % (str(arity))
        assert isinstance(float(complexity), float), "complexity must be castable to float"
        assert isinstance(int(var_type), int) and int(var_type) <= 3, "var_type must be castable to a 0 <= int <= 3"
        assert isinstance(float(fixed_const), float), "fixed_const must be castable to a float"

        # Token representing input_var (eg. x0, x1 etc.)
        if var_type == 1:
            assert function is None,        'Token representing input_var (var_type = 1) must have function = None'
            assert arity == 0,              'Token representing input_var (var_type = 1) must have arity = 0'
            assert isinstance(var_id, int), 'Token representing input_var (var_type = 1) must have an int var_id'
            assert np.isnan(init_val),      'Token representing input_var (var_type = 1) must have init_val = NaN'
            assert np.isnan(float(fixed_const)), \
                                            'Token representing input_var (var_type = 1) must have a nan fixed_const'
        # Token representing function (eg. add, mul, exp, etc.)
        elif var_type == 0:
            assert callable(function), 'Token representing function (var_type = 0) must have callable function'
            assert arity >= 0,         'Token representing function (var_type = 0) must have arity >= 0'
            assert var_id is None,     'Token representing function (var_type = 0) must have var_id = None'
            assert np.isnan(init_val), 'Token representing function (var_type = 0) must have init_val = NaN'
            assert np.isnan(float(fixed_const)), \
                                       'Token representing function (var_type = 0) must have a nan fixed_const'
        # Token representing free constant (eg. c0, c1 etc.)
        elif var_type == 2:
            assert function is None,        'Token representing free const (var_type = 2) must have function = None'
            assert arity == 0,              'Token representing free const (var_type = 2) must have arity == 0'
            assert isinstance(var_id, int), 'Token representing free const (var_type = 2) must have an int var_id'
            assert isinstance(init_val, float) and not np.isnan(init_val), \
                                            'Token representing free const (var_type = 2) must have a non-nan float init_val'
            assert np.isnan(float(fixed_const)), \
                                            'Token representing free const (var_type = 2) must have a nan fixed_const'

        # Token representing a fixed constant (eg. 1, pi etc.)
        elif var_type == 3:
            assert function is None,   'Token representing fixed const (var_type = 3) must have function = None'
            assert arity == 0,         'Token representing fixed const (var_type = 3) must have arity == 0'
            assert var_id is None,     'Token representing fixed const (var_type = 3) must have var_id = None'
            assert np.isnan(init_val), 'Token representing fixed const (var_type = 3) must have init_val = NaN'
            assert not np.isnan(float(fixed_const)), \
                                       'Token representing fixed const (var_type = 3) must have a non-nan fixed_const'
            # not checking isinstance(fixed_const, float) as fixed_const can be a torch.tensor(float) or a float
            # ie. "float-like"

        # ---- Attribution ----
        self.arity       = arity                                   # int
        self.complexity  = float(complexity)                       # float
        self.var_type    = int(var_type)                           # int var_type指符号类型
        # Function specific
        self.function    = function                                # object (callable or None)
        # Free const specific
        self.init_val = init_val                                   # float
        # Input variable / free const specific
        if self.var_type == 1 or self.var_type == 2:
            self.var_id = var_id                                   # int  var_id 指符号对应的索引
        else:
            self.var_id = INVALID_VAR_ID                           # int
        # Fixed const spevific
        self.fixed_const = fixed_const                             # float-like

        # ---------------------------- Physical units : behavior id ----------------------------
        # ---- Assertions ----
        if behavior_id is not None:
            assert isinstance(behavior_id, int), 'Token behavior_id must be an int (%s is not an int)' % (str(behavior_id))
        # ---- Attribution ----
        if behavior_id is None:
            self.behavior_id = DEFAULT_BEHAVIOR_ID                 # int
        else:
            self.behavior_id = behavior_id                         # int

        # ---------------------------- Physical units : power ----------------------------
        assert isinstance(bool(is_power), bool), "is_power must be castable to bool"
        # ---- Assertions ----
        if is_power:
            assert isinstance(power, float) and not np.isnan(power), \
                        "Token with is_power=True must have a non nan float power (%s is not a float)" % (str(power))
        else:
            assert np.isnan(power), "Token with is_power=False must have a np.NAN power"
        # ---- Attribution ----
        self.is_power = is_power                               # bool
        self.power    = power                                  # float

        # ---------------------------- Physical units : phy_units (semi_positional) ----------------------------
        assert isinstance(bool(is_constraining_phy_units), bool), "is_constraining_phy_units must be castable to bool"
        # ---- Assertions ----
        if is_constraining_phy_units:
            assert phy_units is not None, 'Token having physical units constraint (is_constraining_phy_units = True) must contain physical units.'
            assert np.array(phy_units).shape == (UNITS_VECTOR_SIZE,), 'Physical units vectors must be of shape (%s,) not %s, pad with zeros you are not using all elements.' % (UNITS_VECTOR_SIZE, np.array(phy_units).shape)
            assert np.array(phy_units).dtype == float, 'Physical units vectors must contain float.'
            assert not np.isnan(np.array(phy_units)).any(), 'No NaN allowed in phy_units, to create a free constraint phytokens, use is_constraining_phy_units = False and phy_units = None (will result in phy_units = vect of np.NAN)'
        else:
            assert phy_units is None, 'Token not having physical units constraint (is_constraining_phy_units = False) can not contain physical units.'
        # ---- Attribution ----
        self.is_constraining_phy_units = bool(is_constraining_phy_units)  # bool
        if phy_units is None:
            # no list definition in default arg
            self.phy_units = np.full((UNITS_VECTOR_SIZE), np.NAN)  # (UNITS_VECTOR_SIZE,) of float
        else:
            # must be a numpy.array to support operations
            self.phy_units = np.array(phy_units)                   # (UNITS_VECTOR_SIZE,) of float

    def __call__(self, *args):
        # Assert number of args vs arity
        assert len(args) == self.arity, '%i arguments were passed to phytokens %s during call but phytokens has arity = %i' \
            % (len(args), self.name, self.arity,)

        if self.var_type == 0:
            return self.function(*args)

        elif self.var_type == 3:
            return self.fixed_const

        # Raise error for input_var and free const phytokens
        # x0(data_x0, data_x1) would trigger both errors -> use AssertionError for both for simplicity
        else:
            raise AssertionError("Token %s does not represent a function or a fixed constant (var_type=%s), it can not "
                                 "be called."% (self.name, str(self.var_type)))


    def __repr__(self):
        return self.name
