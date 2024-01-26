import numpy as np

from prior.Prior import Prior


class PhysicalUnitsPrior(Prior):
    """
    Enforces that next token should be physically consistent units-wise with current program based on current units
    constraints computed live (during program generation). If there is no way get a constraint all tokens are allowed.
    """
    def __init__(self, library, programs, prob_eps = 0.):
        """
        Parameters
        ----------
        library : library.Library
        programs : program.VectPrograms
        prob_eps : float
            Value to return for the prior inplace of zeros (useful for avoiding sampling problems)
        """
        # ------- INITIALIZING -------
        Prior.__init__(self, library, programs)
        # Tolerance when comparing two units vectors (eg. 0.333333334 == 0.333333333)
        self.tol = 1e2*np.finfo(np.float32).eps
        # Value to return for the prior inplace of zeros.
        self.prob_eps = prob_eps

        # ------- LIB_IS_CONSTRAINING -------
        # mask : are tokens in the library constraining units-wise
        self.lib_is_constraining = self.lib.is_constraining_phy_units[:self.lib.n_choices]                              # (n_choices,)
        # mask : are tokens in the library constraining units-wise (expanding in a new batch_size axis)
        self.lib_is_constraining_padded = np.tile(self.lib_is_constraining, reps=(self.progs.batch_size, 1))            # (batch_size, n_choices,)

        # ------- LIB_UNITS -------
        # Units of choosable tokens in the library
        self.lib_units = self.lib.phy_units[:self.lib.n_choices]                                                        # (n_choices, UNITS_VECTOR_SIZE,)
        # Padded units of choosable tokens in the library (expanding in a new batch_size axis)
        self.lib_units_padded = np.tile(self.lib_units, reps=(self.progs.batch_size, 1, 1))                             # (batch_size, n_choices, UNITS_VECTOR_SIZE,)

    def __call__(self):

        # Current step
        curr_step = self.progs.curr_step

        # ------- COMPUTE REQUIRED UNITS -------
        # Updating programs with newest most constraining units constraints
        self.progs.assign_required_units(step=curr_step)

        # ------- IS_PHYSICAL -------
        # mask : is dummy at current step part of a physical program units-wise
        is_physical = self.progs.is_physical                                                                            # (batch_size,)
        # mask : is dummy at current step part of a physical program units-wise (expanding in a new n_choices axis)
        is_physical_padded = np.moveaxis( np.tile(is_physical, reps=(self.lib.n_choices, 1))                            # (batch_size, n_choices,)
                                              , source=0, destination=1)

        # ------- IS_CONSTRAINING -------
        # mask : does dummy at current step contain constraints units-wise
        is_constraining = self.progs.tokens.is_constraining_phy_units[:, curr_step]                                     # (batch_size,)
        # mask : does dummy at current step contain constraints units-wise (expanding in a new n_choices axis)
        is_constraining_padded = np.moveaxis( np.tile(is_constraining, reps=(self.lib.n_choices, 1))                    # (batch_size, n_choices,)
                                              , source=0, destination=1)
        # Number of programs in batch that constraining at this step
        n_constraining  = is_constraining.sum()
        # mask : for each token in batch, for each token in library are both tokens constraining
        mask_prob_is_constraining_info = self.lib_is_constraining_padded & is_constraining_padded                       # (batch_size, n_choices,)

        # Useful as to forbid a choice, the choosable token must be constraining and the current dummy must also be
        # constraining, otherwise the choice should be legal regardless of the units of any of these tokens
        # (non-constraining tokens should contain NaNs units).

        # ------- UNITS -------
        # Units requirements at current step dummies
        units_requirement       = self.progs.tokens.phy_units[:, curr_step, :]                                          # (batch_size, UNITS_VECTOR_SIZE)
        # Padded units requirements of dummies at current step (expanding in a new n_choices axis)
        units_requirement_padded = np.moveaxis(np.tile(units_requirement, reps=(self.lib.n_choices, 1, 1))              # (batch_size, n_choices, UNITS_VECTOR_SIZE)
                                               , source=0, destination=1)
        # mask : for each token in batch, is choosing token in library legal units-wise
        mask_prob_units_legality = (np.abs(units_requirement_padded - self.lib_units_padded) < self.tol).prod(axis=-1)  # (batch_size, n_choices)

        # ------- RESULT -------
        # Token in library should be allowed if there are no units constraints on any side (library, current dummies)
        # OR if the units are consistent OR if the program is unphysical.
        # Ie. all tokens in the library are allowed if there are no constraints on any sides or if the program is
        # unphysical anyway.
        mask_prob = np.logical_or.reduce((                                                                              # (batch_size, n_choices)
            (~ mask_prob_is_constraining_info),
            (~ is_physical_padded),
            mask_prob_units_legality,
                                          )).astype(float)
        mask_prob[mask_prob == 0] = self.prob_eps
        return mask_prob

    def __repr__(self):
        repr = "PhysicalUnitsPrior"
        return repr
