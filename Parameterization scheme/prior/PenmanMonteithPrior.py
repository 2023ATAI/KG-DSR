import numpy as np

import configs.config
from prior.Prior import Prior


class PenmanMonteithPrior (Prior):
    """
    """

    def __init__(self, library, programs, targets, max, max_depth, scale):
        # Initialize the Prior class with the provided library and programs
        Prior.__init__(self, library, programs)
        # Is token of the library a terminal token : mask
        terminal_arity = 0
        self.mask_lib_is_terminal = (self.lib.get_choosable_prop("arity") == terminal_arity)
        self.targets_str = targets  # (n_constraints,)
        self.n_constraints = len(self.targets_str)  # n_constraints <= n_choices
        self.targets = np.array([self.lib.lib_name_to_idx[tok_name] for tok_name in self.targets_str])  # (n_constraints,)
        # Max number of occurrences allowed for each target
        self.mask_prob[:, :] = 1
        self.max_depth = max_depth
        self.mask_depth_B = np.full(shape=(self.progs.batch_size), fill_value=False, dtype=bool)
        self.not_op_mask = np.ones(shape=self.lib.lib_choosable_name.shape)
        self.not_op_mask[np.where(self.lib.arity > 0)] = 0
        # to determine at which step the resistance section has progressed
        self.C_step = 0
        self.scale = float(scale)
        # _________________________________________________________________________________________________________
        # Define names for sections related to energy absorption, atmosphere absorption, and resistance
        X_ES_names = configs.config.X_ES_names
        X_AS_names = configs.config.X_AS_names
        X_RS_names = configs.config.X_RS_names
        # Create indices for tokens in each section
        # _________________________________________________________________________________________________________
        self.indices_ES = np.array(np.where(np.isin(self.lib.lib_choosable_name, X_ES_names)))
        self.indices_AS = np.array(np.where(np.isin(self.lib.lib_choosable_name, X_AS_names)))
        self.indices_RS = np.array((np.where(np.isin(self.lib.lib_choosable_name, X_RS_names))))
        self.ES_mask = np.zeros(shape=self.lib.lib_choosable_name.shape)
        self.ES_mask[self.indices_ES] = 1
        self.ES_mask[np.where(self.lib.arity > 0)] = 1
        self.AS_mask = np.zeros(shape=self.lib.lib_choosable_name.shape)
        self.AS_mask[self.indices_AS] = 1
        self.AS_mask[np.where(self.lib.arity > 0)] = 1
        self.RS_mask = np.zeros(shape=self.lib.lib_choosable_name.shape)
        self.RS_mask[self.indices_RS] = 1
        self.RS_mask[np.where(self.lib.arity > 0)] = 1
        # _________________________________________________________________________________________________________
        # Create a list of indices for each section
        indices_ES = np.array(np.where(np.isin(self.lib.lib_choosable_name, X_ES_names)))
        indices_AS = np.array(np.where(np.isin(self.lib.lib_choosable_name, X_AS_names)))
        indices_RS = np.array((np.where(np.isin(self.lib.lib_choosable_name, X_RS_names))))
        self.index_list = []
        self.index_list.append(indices_ES)
        self.index_list.append(indices_AS)
        self.index_list.append(indices_RS)
        # __________________________________
        # Initialize operator count arrays for each section
        self.opator_ES_count = np.zeros(self.mask_depth_B.shape[0])
        self.opator_AS_count = np.zeros(self.mask_depth_B.shape[0])
        self.opator_RS_count = np.zeros(self.mask_depth_B.shape[0])
    def __call__(self):

        # Reset probs
        self.reset_mask_prob()
        # mask_depth = np.outer(np.where(self.progs.tokens.depth[:, self.progs.curr_step] > self.max_depth), np.where(self.progs.tokens.depth[:, self.progs.curr_step] !=self.lib.behavior_id[-1]))
        mask_depth = np.where((self.progs.tokens.depth[:, self.progs.curr_step] > self.max_depth) & (
                    self.progs.tokens.depth[:, self.progs.curr_step] != self.progs.tokens.default_depth))
        self.mask_depth_B[mask_depth] = True

        # Define operators for each part of the PM formula
        if self.progs.curr_step == 0:
            # At the first step, set the mask to 0 and enable the operator at index 3
            self.mask_prob[:, :] = 0
            self.mask_prob[:, 3] = 1
        elif self.progs.curr_step == 1:
            # At the second step, set the mask to 0 and enable the operator at index 1
            self.mask_prob[:, :] = 0
            self.mask_prob[:, 1] = 1
        else:
            # Initialize mask_prob to 0
            self.mask_prob[:, :] = 0
            count_of_second = np.sum(self.progs.tokens.depth[:, :self.progs.curr_step + 1] == 2, axis=1)
            count_of_first = np.sum(self.progs.tokens.depth[:, :self.progs.curr_step + 1] == 1, axis=1)

            # Sample according to self.index_list[0]
            counts_0 = np.equal.outer(self.progs.tokens.idx, self.index_list[0]).sum(axis=1)
            # ——————————————————————
            ### Forcing a maximum count for each section
            max_A = np.ones(self.index_list[0].shape[1])
            # Check if the count of tokens in the first section is less than the maximum allowed
            is_target_allowed_0 = np.less(counts_0, max_A)
            # Set the mask for the PM formula to the energy absorption section
            self.mask_prob[np.where((count_of_first == 1) & (count_of_second == 1)), :] = self.ES_mask
            # Update the mask for the energy absorption section based on the maximum count allowed
            self.mask_prob[:, self.index_list[0]] = is_target_allowed_0.astype(float)
            # Mask out operators for energy absorption section based on count conditions
            self.mask_prob[np.where(self.opator_ES_count < self.index_list[0].size - 1)[0][:, np.newaxis],
            np.where(self.lib.arity[:self.mask_prob.shape[1]] != 2)[0]] = 0
            # Mask out operators for energy absorption section based on count conditions
            self.mask_prob[np.where(self.opator_ES_count >= self.index_list[0].size - 1)[0][:, np.newaxis],
            np.where(self.lib.arity[:self.mask_prob.shape[1]] == 2)[0]] = 0
            # Increment the count of operators for energy absorption section based on the arity condition
            self.opator_ES_count[self.progs.tokens.arity[:, self.progs.curr_step - 1] == 2] += 1
            # ———————————————————————————————————————————————————————————————————————————————————————————————————————————————
            # Sample according to self.index_list[1] only where self.index_list[0] is all False
            all_false_mask_0 = np.all(~is_target_allowed_0[:, 0, :], axis=1)
            # depth constrant for operator
            if np.any(all_false_mask_0):
                counts_1 = np.equal.outer(self.progs.tokens.idx, self.index_list[1]).sum(axis=1)
                ### Forcing a maximum count for each section
                max_B = np.ones(self.index_list[1].shape[1])
                # Check if the count of tokens in the second section is less than the maximum allowed
                is_target_allowed_1 = np.less(counts_1, max_B)
                # Set the mask for the PM formula to the atmosphere absorption section
                self.mask_prob[np.where((count_of_first == 1) & (count_of_second == 2)), :] = self.AS_mask
                # Update the mask for the atmosphere absorption section based on the maximum count allowed
                self.mask_prob[np.where(all_false_mask_0)[0][:, np.newaxis], self.index_list[1]] = is_target_allowed_1[
                                                                                                   all_false_mask_0, 0,
                                                                                                   :].astype(float)
                # Create a mask for the count of operators in the Energy Absorption section
                opator_EA_count = np.where((all_false_mask_0) & (self.opator_AS_count < self.index_list[1].size - 2))
                # Update the mask for the PM formula based on conditions related to the count of operators in the Energy Absorption section
                self.mask_prob[
                    opator_EA_count[0][:, np.newaxis], np.where(self.lib.arity[:self.mask_prob.shape[1]] != 2)[0]] = 0
                # Mask out operators for the Atmosphere Absorption section based on count conditions
                self.mask_prob[np.where(self.opator_AS_count >= self.index_list[1].size - 2)[0][:, np.newaxis],
                np.where(self.lib.arity[:self.mask_prob.shape[1]] == 2)[0]] = 0
                # Increment the count of operators for the Atmosphere Absorption section based on the arity condition
                self.opator_AS_count[self.progs.tokens.arity[:, self.progs.curr_step - 1] == 2] += 1
                # ———————————————————————————————————————————————————————————————————————————————————————————————————————————————
                # Sample according to self.index_list[2] only where both self.index_list[0] and self.index_list[1] are all False
                all_false_mask_1 = np.all(~is_target_allowed_1[:, 0, :], axis=1)

                if np.any(all_false_mask_1):
                    counts_2 = np.equal.outer(self.progs.tokens.idx, self.index_list[2]).sum(axis=1)
                    ### Forcing a maximum count for each section
                    max_C = np.ones(self.index_list[2].shape[1]) * 1
                    max_C[-2:] = 2
                    opator_EAR_max = 5
                    # Check if the count of tokens in the third section meets the maximum allowed counts
                    is_target_allowed_2 = np.less(counts_2, max_C)
                    # Set the mask for the PM formula to the 'Resistance' section for programs with count_of_first equal to 2
                    self.mask_prob[np.where((count_of_first == 2)), :] = self.RS_mask
                    # Update the mask for the 'Resistance' section based on the maximum count allowed
                    self.mask_prob[
                        np.where(all_false_mask_1)[0][:, np.newaxis], self.index_list[2]] = is_target_allowed_2[
                                                                                            all_false_mask_1, 0,
                                                                                            :].astype(float)
                    # # ———————————————————————————————————————————————————————————————————————————————————————————————————————————————
                    # Using a Gaussian function to assign probabilities. The length of the Gaussian function is determined based on the
                    # number of input variables. The purpose is to include all inputs in the search, and it also allows for selectively
                    # not choosing all variables. The code increments the step counter (self.C_step) and adjusts the probabilities based
                    # on the Gaussian values at different steps in the generation process.
                    # # ———————————————————————————————————————————————————————————————————————————————————————————————————————————————
                    # If we want length = 3, Gaussian value must be maximum at step = 2 (i.e., when generating token number 3)
                    length_loc = self.index_list[2].shape[1]
                    # => step_loc = length_loc - 1
                    self.step_loc = float(length_loc) - 1
                    # Value of gaussian at all steps
                    steps = np.arange(0, self.progs.max_time_step + 1)
                    # gaussian_vals[step_loc] = gaussian_vals[steps[step_loc]]
                    self.gaussian_vals = np.exp(-(steps - self.step_loc) ** 2 / (2 * self.scale))
                    # Programs having only one dummy (going to finish at the next step if choosing a terminal token): mask
                    mask_one_dummy_progs = (self.progs.n_dangling == 1)
                    # Before loc
                    if self.C_step < self.step_loc:
                        # Scale terminal token probs by gaussian where progs have only one dummy
                        mask_scale_terminal = np.outer(mask_one_dummy_progs, self.mask_lib_is_terminal)
                        self.mask_prob[mask_scale_terminal] *= self.gaussian_vals[self.C_step]
                    # At loc: gaussian value = 1.
                    # After loc
                    elif self.C_step > self.step_loc:
                        # Scale non-terminal tokens probs by gaussian
                        self.mask_prob[:, np.logical_not(self.mask_lib_is_terminal)] *= self.gaussian_vals[
                            self.C_step]
                    self.C_step += 1
                    # # ———————————————————————————————————————————————————————————————————————————————————————————————————————————————
                    self.mask_prob[np.where(self.opator_RS_count >= self.index_list[2].size - 2)[0][:, np.newaxis],
                    np.where(self.lib.arity[:self.mask_prob.shape[1]] == 2)[0]] = 0
                    self.opator_RS_count[self.progs.tokens.arity[:, self.progs.curr_step - 1] == 2] += 1
                    # # ———————————————————————————————————————————————————————————————————————————————————————————————————————————————
                    # print()
                    # Optional: Set additional conditions if needed
                    if np.sum(self.mask_depth_B) > 0:
                        self.mask_prob[self.mask_depth_B, :] = np.multiply(self.mask_prob[self.mask_depth_B, :],
                                                                           self.not_op_mask)
        return self.mask_prob

    def __repr__(self):
        return "PenmanMonteithPrior"
#