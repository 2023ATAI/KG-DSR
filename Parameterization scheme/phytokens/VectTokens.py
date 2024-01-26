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


# --------------------- POSITIONAL TOKENS DEFAULT VALUES IN PROGRAMS ---------------------
# VectPrograms.append, VectPrograms.update_relationships_pos only work with MAX_NB_CHILDREN = 2
MAX_NB_CHILDREN = config.positional_token_default_values['MAX_NB_CHILDREN']
# VectPrograms.append, VectPrograms.update_relationships_pos, VectPrograms.get_sibling_idx,
# VectPrograms.get_sibling_idx_of_step prior.RelationshipConstraintPrior get_property_of_relative,
# only work with MAX_NB_SIBLINGS = 1
MAX_NB_SIBLINGS = MAX_NB_CHILDREN - 1
# Max arity value
MAX_ARITY = MAX_NB_CHILDREN
# Out of range phytokens, pos >= (n_lengths + n_dangling)
INVALID_TOKEN_NAME = config.positional_token_default_values['INVALID_TOKEN_NAME']
INVALID_POS   = config.positional_token_default_values['INVALID_POS']
INVALID_DEPTH = config.positional_token_default_values['INVALID_DEPTH']
# Dummy phytokens, n_lengths <= pos < (n_lengths + n_dangling)
DUMMY_TOKEN_NAME = config.positional_token_default_values['DUMMY_TOKEN_NAME']
class VectTokens:
    """
    Object representing a matrix of positional phytokens (positional) ie:
     - non_positional properties: idx + phytokens properties attributes, see phytokens.Token.__init__ for full description.
     - semi_positional properties: See phytokens.Token.__init__ for full description of phytokens properties attributes.
     - positional properties which are contextual (family relationships, depth etc.).
    This only contains properties expressed as float, int, bool to be jit-able.

    Attributes : In their non-vectorized shapes (types are vectorized)
    ----------
    idx                       : int
        Encodes phytokens's nature, phytokens index in the library.

    ( name                    :  str (<MAX_NAME_SIZE) )
    ( sympy_repr              :  str (<MAX_NAME_SIZE) )
    arity                     :  int
    complexity                :  float
    var_type                  :  int
    ( function                :  callable or None  )
    ( init_val                  :  float           )
    var_id                    :  int
    ( fixed_const             : float              )
    behavior_id               :  int
    is_power                  :  bool
    power                     :  float

    is_constraining_phy_units :  bool
    phy_units                 :  numpy.array of shape (UNITS_VECTOR_SIZE,) of float

    pos                      : int
        Position in the program ie in time dim (eg. 0 for mul in program = [mul, x0, x1] )
    pos_batch                : int
        Position in the batch ie in batch dim.
    depth                    : int
        Depth in tree representation of program.
    has_parent_mask          : bool
        True if phytokens has parent, False else.
    has_siblings_mask         : bool
        True if phytokens has at least one sibling, False else.
    has_children_mask         : bool
        True if phytokens has at least one child, False else.
    has_ancestors_mask        : bool
        True if phytokens has at least one ancestor, False else. This is always true for valid phytokens as the phytokens itself
        counts as its own ancestor.
    parent_pos               : int
        Parent position in the program ie in time dim (eg. 0 for mul in program = [mul, x0, x1] )
    siblings_pos              : numpy.array of shape (MAX_NB_SIBLINGS,) of int
        Siblings position in the program ie in time dim (eg. 1 for x0 in program = [mul, x0, x1] )
    children_pos              : numpy.array of shape (MAX_NB_CHILDREN,) of int
        Children position in the program ie in time dim (eg. 2 for x1 in program = [mul, x0, x1] )
    ancestors_pos              : numpy.array of shape (shape[1],) of int`
        Ancestors positions in the program ie in time dim counting the phytokens itself as itw own ancestor.
        (eg. [0, 1, 4, 5, INVALID_POS, INVALID_POS] for x1 in program = [mul, add, sin, x0, log, x1]).
    n_siblings                : int
        Number of siblings.
    n_children                : int
        Number of children.
    n_ancestors               : int
        Number of ancestors. This is equal to depth+1 as the phytokens itself counts as its own ancestor.
    """

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

        # -------------------------------------------------------------------------------------------------------
        # ---------------------------------------- Positional properties ----------------------------------------
        # -------------------------------------------------------------------------------------------------------

        # ---- Position ----
        # Default values
        self.default_pos       = INVALID_POS
        self.default_pos_batch = INVALID_POS
        # Properties : position is the same in all elements of batch
        self.pos               = np.tile(np.arange(0, self.shape[1]), (self.shape[0], 1)).astype(int)
        self.pos_batch         = np.tile(np.arange(0, self.shape[0]), (self.shape[1], 1)).transpose().astype(int)

        # ---- Depth ----
        # Default value
        self.default_depth = INVALID_DEPTH
        # Property
        self.depth = np.full(shape=self.shape, fill_value=self.default_depth, dtype=int )

        # ---- Family relationships ----

        # Token family relationships: family mask
        # Default values
        self.default_has_parent_mask    = False
        self.default_has_siblings_mask  = False
        self.default_has_children_mask  = False
        self.default_has_ancestors_mask = False
        # Properties
        self.has_parent_mask    = np.full(shape=self.shape, fill_value=self.default_has_parent_mask    ,           dtype=bool)
        self.has_siblings_mask  = np.full(shape=self.shape, fill_value=self.default_has_siblings_mask  ,           dtype=bool)
        self.has_children_mask  = np.full(shape=self.shape, fill_value=self.default_has_children_mask  ,           dtype=bool)
        self.has_ancestors_mask = np.full(shape=self.shape, fill_value=self.default_has_ancestors_mask ,           dtype=bool)

        # Token family relationships: pos
        # Default values
        self.default_parent_pos    = INVALID_POS
        self.default_siblings_pos  = INVALID_POS
        self.default_children_pos  = INVALID_POS
        self.default_ancestors_pos = INVALID_POS
        # Properties
        self.parent_pos         = np.full(shape=self.shape,                      fill_value=self.default_parent_pos   , dtype=int)#父节点矩阵（batch_size,max_step）每一次设置时对此矩阵更新
        self.siblings_pos       = np.full(shape=self.shape + (MAX_NB_SIBLINGS,), fill_value=self.default_siblings_pos , dtype=int)
        self.children_pos       = np.full(shape=self.shape + (MAX_NB_CHILDREN,), fill_value=self.default_children_pos , dtype=int)
        self.ancestors_pos      = np.full(shape=self.shape + (self.shape[1], ),  fill_value=self.default_ancestors_pos, dtype=int)

        # Token family relationships: numbers
        # Default values
        self.default_n_siblings  = 0
        self.default_n_children  = 0
        self.default_n_ancestors = 0
        # Properties
        self.n_siblings         = np.full(shape=self.shape,  fill_value=self.default_n_siblings , dtype=int)
        self.n_children         = np.full(shape=self.shape,  fill_value=self.default_n_children , dtype=int)
        self.n_ancestors        = np.full(shape=self.shape,  fill_value=self.default_n_ancestors, dtype=int)
