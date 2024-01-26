import numpy as np

from prior.HardLengthPrior import HardLengthPrior
from prior.PenmanMonteithPrior import PenmanMonteithPrior
from prior.PhysicalUnitsPrior import PhysicalUnitsPrior

PRIORS_WO_ARGS = {
    # "UniformArityPrior"     : UniformArityPrior,
    # "NoUselessInversePrior" : NoUselessInversePrior,
}

# Priors that take additional arguments
PRIORS_W_ARGS = {
    #"HardLengthPrior"             : HardLengthPrior,
    "PenmanMonteithPrior":PenmanMonteithPrior,

# "SoftLengthPrior"             : SoftLengthPrior,
    # "RelationshipConstraintPrior" : RelationshipConstraintPrior,
    # "NestedFunctions"             : NestedFunctions,
    # "NestedTrigonometryPrior"     : NestedTrigonometryPrior,
    # "OccurrencesPrior"            : OccurrencesPrior,
    "PhysicalUnitsPrior"          : PhysicalUnitsPrior,
}

# All priors
PRIORS_DICT = {}
PRIORS_DICT.update(PRIORS_WO_ARGS)
PRIORS_DICT.update(PRIORS_W_ARGS)
def make_PriorCollection (library, programs, priors_config,):
    """
    Makes PriorCollection object from arguments.
    Parameters
    ----------
    library : library.Library
        Library of choosable phytokens.
    programs : program.VectPrograms
        Programs in the batch.
    priors_config : list of couples (str : dict)
        List of priors. List containing couples with prior name as first item in couple (see prior.PRIORS_DICT for list
        of available priors) and additional arguments (besides library and programs) to be passed to priors as second
        item of couple, leave None for priors that do not require arguments.
    Returns
    -------
    Prior.PriorCollection
    """
    type_err_msg = "priors_config should be a list containing couples with prior name string as first item in couple " \
                   "and additional arguments to be passed to priors dictionary as second item of couple, leave None " \
                   "for priors that do not require arguments."
    # Assertion
    assert isinstance(priors_config, list), type_err_msg
    # PriorCollection
    prior_collection = PriorCollection(library = library, programs = programs)
    # Individual priors
    priors = []
    # Iterating through individual priors
    for config in priors_config:
        # --- TYPE ASSERTIONS ---
        assert len(config) == 2, type_err_msg
        assert isinstance(config[0], str), type_err_msg
        assert isinstance(config[1], dict) or config[1] is None, type_err_msg
        # --- GETTING ITEMS ---
        name, args = config[0], config[1]
        # --- ASSERTIONS ---
        assert name in PRIORS_DICT, "Prior %s is not in the list of available priors :\n %s"%(name, PRIORS_DICT.keys())
        if name in PRIORS_W_ARGS:
            assert args is not None, "Arguments for making prior %s were not given." % (name)
        # --- MAKING PRIOR ---
        # If individual prior has additional args get them
        if name in PRIORS_W_ARGS:
            prior_args = args
        else:
            prior_args = {}
        # Appending individual prior
        prior = PRIORS_DICT[name](library = library, programs = programs, **prior_args)
        priors.append (prior)
    # Setting priors in PriorCollection
    prior_collection.set_priors(priors)
    return prior_collection


class PriorCollection:
    """
    Collection of prior.Prior, returns value of element-wise multiplication of constituent priors.
    """
    def __init__(self, library, programs,):
        """
        Parameters
        ----------
        library : library.Library
            Library of choosable phytokens.
        programs : program.VectPrograms
            Programs in the batch.
        """
        self.priors    = []
        self.lib       = library
        self.progs     = programs
        self.init_prob = np.ones( (self.progs.batch_size, self.lib.n_choices), dtype = float)

    def set_priors (self, priors):
        """
        Sets constituent priors.
        Parameters
        ----------
        priors : list of prior.Prior
        """
        for prior in priors:
            self.priors.append(prior)

    def __call__(self):
        """
        Returns probabilities of priors for each choosable phytokens in the library.
        Returns
        -------
        mask_probabilities : numpy.array of shape (self.progs.batch_size, self.lib.n_choices) of float
        """
        res = self.init_prob
        for prior in self.priors:
            res = np.multiply(res, prior())
        return res

    def __repr__(self):
        #repr = np.array([str(prior) for prior in self.priors])
        repr = "PriorCollection:"
        for prior in self.priors:
            repr += "\n- %s"%(prior)
        return str(repr)
