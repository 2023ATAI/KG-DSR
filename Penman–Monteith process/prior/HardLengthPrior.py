import numpy as np

from prior.Prior import Prior

class HardLengthPrior (Prior):
    """
    Forces programs to have lengths such that min_length <= lengths <= max_length finished.
    Enforces lengths <= max_length by forbidding non-terminal tokens when choosing non-terminal tokens would mean
    exceeding max length of program.
    Enforces min_length <= lengths by forbidding terminal tokens when choosing a terminal token would mean finishing a
    program before min_length.
    """

    def __init__(self, library, programs, min_length, max_length):
        """
        Parameters
        ----------
        library : library.Library
        programs : program.VectPrograms
        min_length : float
            Minimum length that programs are allowed to have.
        max_length : float
            Maximum length that programs are allowed to have.
        """
        # Assertions
        try: min_length = float(min_length)
        except ValueError: raise TypeError("max_length must be cast-able to a float")
        try: max_length = float(max_length)
        except ValueError: raise TypeError("max_length must be cast-able to a float")
        assert min_length <= programs.max_time_step, "min_length must be such as: min_length <= max_time_step"
        assert max_length <= programs.max_time_step, "max_length must be such as: max_length <= max_time_step"
        assert max_length >= 1,                      "max_length must be such as: max_length >= 1"
        assert min_length <= max_length,             "Must be: min_length <= max_length"

        Prior.__init__(self, library, programs)
        # Is token of the library a terminal token : mask
        terminal_arity = 0
        self.mask_lib_is_terminal = (self.lib.get_choosable_prop("arity") == terminal_arity)
        assert min_length < max_length, "Min length must be such that: min_length < max_length"
        self.min_length = min_length
        self.max_length = max_length
        self.reset_mask_prob()

    def __call__(self):
        # Reset probs
        self.reset_mask_prob()

        # --- MAX ---
        # Would library token exceed max length if chosen in next step : mask
        mask_would_exceed_max = np.add.outer(self.progs.n_completed, self.lib.get_choosable_prop("arity")) > self.max_length
        # Going to reach max length => next token must be terminal => prob for non-terminal must be = 0
        self.mask_prob[mask_would_exceed_max] *= 0 # = 0 for token exceeding max

        # --- MIN ---
        # Progs having only one dummy AND length (including dummies) < min : mask
        # These programs are going to finish at next step if we allow terminal tokens to be chosen.
        mask_going_to_finish_before_min = np.logical_and(self.progs.n_dangling == 1, self.progs.n_completed < self.min_length)
        # Going to be finished with length < min length => next token must be non-terminal => prob for terminal must be = 0
        mask_would_be_inferior_to_min = np.outer(mask_going_to_finish_before_min, self.mask_lib_is_terminal)
        self.mask_prob[mask_would_be_inferior_to_min] *= 0 # = 0 for terminal
        return self.mask_prob

    def __repr__(self):
        return "HardLengthPrior (min_length = %i, max_length = %i)"%(self.min_length, self.max_length)
