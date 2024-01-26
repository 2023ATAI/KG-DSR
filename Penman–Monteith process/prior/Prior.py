from abc import ABC
import numpy as np


class Prior (ABC):
    """
    Abstract prior.
    """
    def __init__(self, library, programs):
        """
        Parameters
        ----------
        library : library.Library
            Library of choosable phytokens.
        programs : program.VectPrograms
            Programs in the batch.
        """
        self.lib       = library
        self.progs     = programs
        self.get_default_mask_prob = lambda : np.ones((self.progs.batch_size, self.lib.n_choices), dtype = float)
        self.reset_mask_prob()

    def reset_mask_prob (self):
        """
        Resets mask of probabilities to one.
        """
        self.mask_prob = self.get_default_mask_prob()

    def __call__(self):
        """
        Returns probabilities of priors for each choosable phytokens in the library.
        Returns
        -------
        mask_probabilities : numpy.array of shape (self.progs.batch_size, self.lib.n_choices) of float
        """
        raise NotImplementedError
